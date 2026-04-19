import torch
import os
import json
import shutil
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ====================== CONFIG ======================
model_path = "/home/eric/base_models/Qwen3-14B"   # ← Change to your actual Qwen3 path

jsonl_file = "/home/eric/Master_training_data/step1v2_split.jsonl"

max_seq_length = 2048
r = 128
lora_alpha = 96                    # Better than 96 for Qwen3
use_rslora = True
# ===================================================

# === AGGRESSIVE SINGLE-PROCESS + OFFLINE MODE ===
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

print("=== Loading Qwen3-Coder with balanced sharding ===")

# Clear old cache
shutil.rmtree("/home/eric/scripts/unsloth_compiled_cache", ignore_errors=True)

print("Loading model on dual 5070 Ti...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    dtype=None,
    load_in_4bit=True,
    device_map="balanced",
    max_memory={0: "14.4GB", 1: "14GB"},
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.vocab_size = len(tokenizer)

model = FastLanguageModel.get_peft_model(
    model,
    r=r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.25,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=use_rslora,
)

print(f"LoRA configured - r={r}, alpha={lora_alpha}")

# ====================== ROBUST LOADING ======================
print("\nLoading JSONL...")

data_list = []
skipped = 0

with open(jsonl_file, "r", encoding="utf-8", errors="ignore") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            example = json.loads(line)
            data_list.append(example)
        except Exception:
            skipped += 1
            if skipped <= 10:
                print(f"Skipped line {line_num}")

print(f"Loaded {len(data_list)} examples, skipped {skipped} bad lines.")

dataset = Dataset.from_list(data_list)

# Basic validation
def is_valid(example):
    messages = example.get("messages")
    return (
        messages is not None and
        isinstance(messages, list) and
        len(messages) > 1 and
        all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages)
    )

dataset = dataset.filter(is_valid, num_proc=4)
print(f"After structural filter: {len(dataset)} examples remain.")

# ====================== FORMAT FOR QWEN3 ======================
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",   # Qwen3 uses the same template as Qwen2.5
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

def formatting_prompts_func(examples):
    texts = []
    for convos in examples["messages"]:
        try:
            text = tokenizer.apply_chat_template(
                convos,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        except Exception:
            texts.append("")
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=4)

# Remove empty texts
dataset = dataset.filter(lambda x: isinstance(x.get("text"), str) and len(x["text"].strip()) > 20, num_proc=4)

print(f"Final training dataset size: {len(dataset)} examples")

# ====================== TOKENIZE ======================
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset.column_names,
)

print(f"Tokenized dataset size: {len(tokenized_dataset)} examples")

# ====================== TRAINER ======================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=96,           # You had 96 before, 32 is safer
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=1e-5,                       # You had this low — keep if you want slow learning
        warmup_ratio=0.30,
        max_grad_norm=0.4,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        optim="paged_adamw_8bit",
        weight_decay=0.15,
        lr_scheduler_type="cosine",
        output_dir="/media/eric/Models/Merged_models/custom_qwen3_adapter",
        report_to="none",
        save_strategy="steps",
        save_steps=49,
        save_total_limit=5,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    ),
)

print("\nStarting training...")
trainer.train()

model.save_pretrained("/media/eric/Models/Merged_models/custom_qwen3_adapter")
tokenizer.save_pretrained("/media/eric/Models/Merged_models/custom_qwen3_adapter")
print("Training completed!")
