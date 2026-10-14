
# ── Standard libs ──────────────────────────────────────────────────────────────
import datetime
import gc
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import shutil
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Third-party ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from qwen_vl_utils import process_vision_info


# ══════════════════════════════════════════════════════════════════════════════
# 0. REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG  — chỉnh paths ở đây trước khi chạy
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    # ── Model ──────────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"

    # ── Paths (server) ─────────────────────────────────────────────────────────
    output_dir:  str = "/root/vifoodvqa_checkpoints"
    image_root:  str = "/root/ViFoodVQA_Data/data"
    train_jsonl: str = "/root/ViFoodVQA_Data/data/train.jsonl"
    val_jsonl:   str = "/root/ViFoodVQA_Data/data/val.jsonl"
    test_jsonl:  str = "/root/ViFoodVQA_Data/data/test.jsonl"

    # ── Data ───────────────────────────────────────────────────────────────────
    seed:              int          = 42
    train_sample_size: Optional[int] = None   # None = dùng hết
    val_sample_size:   Optional[int] = None
    test_sample_size:  Optional[int] = 50

    # ── Training ───────────────────────────────────────────────────────────────
    batch_size:                  int   = 4
    num_epochs:                  int   = 6
    learning_rate:               float = 1e-4
    weight_decay:                float = 0.01
    gradient_accumulation_steps: int   = 4
    max_grad_norm:               float = 1.0
    warmup_ratio:                float = 0.03
    log_interval:                int   = 20
    save_every_n_steps:          int   = 500

    # ── Generation ─────────────────────────────────────────────────────────────
    max_new_tokens: int   = 20
    temperature:    float = 0.1
    do_sample:      bool  = False

    # ── Quantization & LoRA ────────────────────────────────────────────────────
    use_4bit:             bool      = False
    gradient_checkpointing: bool    = True
    attn_implementation:  str       = "sdpa"
    use_lora:             bool      = True
    lora_r:               int       = 16
    lora_alpha:           int       = 32
    lora_dropout:         float     = 0.05
    lora_target_modules:  List[str] = None   # set in __init__

    def __init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = 'all-linear'

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ══════════════════════════════════════════════════════════════════════════════
class ViFoodVQADataset(Dataset):
    """Load ViFoodVQA samples từ DataFrame, resolve image paths on demand."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: str,
        max_samples: Optional[int] = None,
    ):
        self.image_root = Path(image_root)
        self.samples = []
        self.image_cache = {}

        for _, item in dataframe.iterrows():
            for qa in item.get("qa_pairs", []):
                self.samples.append(
                    {
                        "image_path": item["image"],
                        "label":      item.get("label", ""),
                        "question":   qa["question"],
                        "answer":     qa["answer"],
                    }
                )
                if max_samples and len(self.samples) >= max_samples:
                    break
            if max_samples and len(self.samples) >= max_samples:
                break
        print(f"[INFO] Dataset initialized with {len(self.samples)}QA pairs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample   = self.samples[idx]
        img_path = self.image_root / sample["image_path"]
        if str(img_path) in self.image_cache:
            image=self.image_cache[str(img_path)]
        else:
            try:
                image = Image.open(img_path).convert("RGB")
                if image.width > 1024 or image.height > 1024:
                    image.thumbnail((1024, 1024))
            except Exception as exc:
                print(f"[WARNING] Cannot load {img_path}: {exc}. Using placeholder.")
                image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        return {
            "image":      image,
            "question":   sample["question"],
            "answer":     sample["answer"],
            "label":      sample["label"],
            "image_path": str(img_path),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. COLLATOR
# ══════════════════════════════════════════════════════════════════════════════
class VQACollator:
    """Build model-ready tensors và mask prompt portion cho supervised tuning."""

    def __init__(self, processor):
        self.processor = processor
        self.assistant_token_ids=self.processor.tokenizer.encode(
            '<|im_start|>assistant\n',
            add_special_tokens=False
        )

    def __call__(self, batch: List[Dict]) -> Dict:
        full_messages = []
        for sample in batch:
            INSTRUCTION = "Trả lời ngắn gọn bằng tiếng Việt."
            full_messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image"]},
                        {"type": "text",  "text":f"{INSTRUCTION}\n{sample['question']}"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["answer"]}],
                },
            ])

        image_inputs, video_inputs = process_vision_info(full_messages)

        full_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in full_messages
        ]

        model_inputs = self.processor(
            text=full_texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
            min_pixels=256*28*28,
            max_pixels=768*28*28,
        )
        # Mask prompt tokens và padding tokens
        labels = model_inputs["input_ids"].clone()
        pad_token_id=self.processor.tokenizer.pad_token_id
        
        for idx in range(labels.size(0)):
            input_id_list=model_inputs['input_ids'][idx].tolist()
            
            target_idx=-1
            for j in range(len(input_id_list)-len(self.assistant_token_ids)+1):
                if input_id_list[j:j+len(self.assistant_token_ids)]==self.assistant_token_ids:
                    target_idx=j+len(self.assistant_token_ids)
                    break
            if target_idx!=-1:
                labels[idx,:target_idx]=-100
            else:
                print(f"Khong tim thay ranh gioi assistant tai sample idx {idx}")
        
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
            
        model_inputs["labels"]=labels
        return model_inputs


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINER
# ══════════════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VQAFinetune:
    def __init__(self, config: Config, device: torch.device = device):
        self.config         = config
        self.device         = device
        self.model_name     = config.model_name
        self.checkpoint_dir = Path(config.output_dir) / self.model_name.replace("/", "_")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler      = None
        self._setup_model()
       


    # ── Setup ──────────────────────────────────────────────────────────────────
    def _setup_model(self):
        print(f"[INFO] Loading model: {self.model_name}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name
        )

        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=self.config.attn_implementation,
            )
            self.model=prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            )
            print(f"[INFO] Model loaded with 4-bit quantization on {self.device}")
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="auto",
                attn_implementation=self.config.attn_implementation,
            )
            print(f"[INFO] Model loaded with fp16 on {self.device}")

        if self.config.gradient_checkpointing:
            if not self.config.use_4bit:
                self.model.gradient_checkpointing_enable()
            print("[INFO] Gradient checkpointing enabled")

        if self.config.use_lora:
            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            print("[INFO] LoRA enabled")
            self.model.print_trainable_parameters()
            
        self.model.train()
        
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total     = sum(p.numel() for p in self.model.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        print(f"[INFO] Total: {total:.1f}M | Trainable: {trainable:.1f}M")

    # ── Device helper ──────────────────────────────────────────────────────────
    def _batch_to_device(self, batch: Dict) -> Dict:
        device_batch = {}
        for key,value in batch.items():
            if torch.is_tensor(value):
                device_batch[key]=value.to(self.device)
            else:
                device_batch[key]=value
        return device_batch

    # ── Train one epoch ────────────────────────────────────────────────────────
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training {self.model_name}")
        

        for batch_idx, batch in enumerate(progress_bar):
            batch  = self._batch_to_device(batch)
            model_inputs = {k: v for k, v in batch.items()}

            outputs=self.model(**model_inputs)
            loss=outputs.loss
            
            if loss is None or torch.isnan(loss):
                continue    
            
            loss=loss/self.config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm
                )
                    
                self.optimizer.step()
                
                if hasattr(self,"lr_scheduler"):
                    self.lr_scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.config.gradient_accumulation_steps

            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / (batch_idx+1)
                progress_bar.set_postfix({"loss":f"{avg_loss:.4f}"})
                
            if (batch_idx+1)%self.config.save_every_n_steps == 0:
                self.save_checkpoint(f"step_{batch_idx+1}")

                
        #xu ly batch dư( nhưng len(dataset)%4==0 nên ổn
        if len(train_loader) % self.config.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.config.max_grad_norm)
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                
        return total_loss / len(train_loader)

    # ── Val loss (nhanh, không generate) ──────────────────────────────────────
    def validate(self,val_loader: DataLoader):
        self.model.eval()
        total_val_loss=0
        with torch.no_grad():
            for batch in val_loader:
                batch = self._batch_to_device(batch)
                model_inputs = {k: v for k, v in batch.items()}
                outputs = self.model(**model_inputs)
                loss = outputs.loss
                if loss is not None:
                    total_val_loss+=loss.item()
        return total_val_loss/len(val_loader) if len(val_loader)>0 else 0

    # ── Save ───────────────────────────────────────────────────────────────────
    def save_checkpoint(self, suffix: str = "final"):
        save_path = self.checkpoint_dir / f"{self.model_name.replace('/', '_')}_{suffix}"
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path,safe_serialization=True)
        processor_path=save_path/"processor"
        self.processor.save_pretrained(processor_path)
        # torch.save(self.optimizer.state_dict(), save_path / "optimizer.pt")

        metadata = {
            "checkpoint_type": suffix,
            "use_lora": self.config.use_lora,
            "use_4bit": self.config.use_4bit,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lr": self.config.learning_rate,
        }
        with open(save_path / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    # ── Main train loop ────────────────────────────────────────────────────────
    def train(self, train_loader: DataLoader,val_loader: DataLoader = None):
        final_train_loss = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 3
        history = {
            "train_loss": [],
            "val_loss": []
        }
        steps_per_epoch=len(train_loader)//self.config.gradient_accumulation_steps
        total_training_steps=steps_per_epoch*self.config.num_epochs

        num_warmup_steps=int(total_training_steps*self.config.warmup_ratio)

        self.lr_scheduler=get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps
        )
        print(f'Scheduler initialized.Total steps : {total_training_steps} | Warmup steps : {num_warmup_steps}')

        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            final_train_loss = self.train_epoch(train_loader)
            print(f"Train Loss: {final_train_loss:.4f}")
            
            history["train_loss"].append(float(final_train_loss))
            
            if val_loader is not None:
                val_loss=self.validate(val_loader)
                print(f"Val Loss: {val_loss:.4f}")

                history["val_loss"].append(float(val_loss))
                
                if val_loss < best_val_loss:
                    import shutil
                    if best_val_loss != float('inf'):
                        old_epoch = len(history["val_loss"]) - 1 
                        old_best_path = self.checkpoint_dir / f"{self.model_name.replace('/', '_')}_epoch_{old_epoch}_best_val"
                        if old_best_path.exists():
                            shutil.rmtree(old_best_path)
                            print(f"[INFO] Deleted old best checkpoint at epoch {old_epoch} to save disk space.")

                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(f"epoch_{epoch+1}_best_val")

                else:
                    patience_counter+=1
                    print(f"Val Loss increased from {best_val_loss:.4f} to {val_loss:.4f}")
                    if patience_counter>=patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                self.save_checkpoint(f"epoch_{epoch+1}")

        if val_loader is not None and best_val_loss != float('inf'):
            print(f"[INFO] Training finished. Best Val Loss achieved: {best_val_loss:.4f}")
            self.save_checkpoint("final_from_best")
        else:
            self.save_checkpoint("final")
        return history

                

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    config = Config()
    set_seed(config.seed)

    print("[INFO] Loading datasets...")
    train_df = pd.read_json(config.train_jsonl, lines=True)
    val_df   = pd.read_json(config.val_jsonl,   lines=True)

    train_dataset = ViFoodVQADataset(train_df, config.image_root, config.train_sample_size)
    val_dataset   = ViFoodVQADataset(val_df,   config.image_root, config.val_sample_size)

    print(f"[INFO] Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # Trainer (processor cần có trước khi tạo collator)
    trainer  = VQAFinetune(config)
    collator = VQACollator(trainer.processor)

    g = torch.Generator()
    g.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    print(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    history = trainer.train(train_loader, val_loader)

    # Save history to json để xem lại sau
    history_path = Path(config.output_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] History saved → {history_path}")


if __name__ == "__main__":
    main()
