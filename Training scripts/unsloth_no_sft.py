import torch
import os
import json
import shutil
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ====================== CONFIG ======================
model_path = ""
jsonl_file = ""

max_seq_length = 2048
r = 256
lora_alpha = 512
use_rslora = True
# ===================================================

# === AGGRESSIVE SINGLE-PROCESS + OFFLINE MODE ===
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_HTTP_TIMEOUT"] = "5"
os.environ["UNSLOTH_SKIP_VERSION_CHECK"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["ACCELERATE_USE_FSDP"] = "0"
os.environ["ACCELERATE_DISABLE_DDP"] = "1"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "0"
os.environ["NO_DISTRIBUTED"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

print("=== SINGLE PROCESS + OFFLINE MODE - Balanced sharding only ===")

# FORCE CLEAR CACHE AGAIN
print("Clearing Unsloth compiled cache again...")
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
# This ensures the model's internal config matches the tokenizer's actual length
model.config.vocab_size = len(tokenizer)

model = FastLanguageModel.get_peft_model(
    model,
    r=r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.2,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=use_rslora,
)

# ====================== ROBUST LOADING WITH PANDAS ======================
print("\nLoading JSONL with pandas...")

try:
    df = pd.read_json(jsonl_file, lines=True, encoding="utf-8", encoding_errors="ignore")
    print(f"pandas successfully loaded {len(df)} rows.")
except Exception as e:
    print(f"pandas read_json failed: {e}")
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
                if skipped <= 20:
                    print(f"Skipped line {line_num}")
    df = pd.DataFrame(data_list)
    print(f"Manual fallback loaded {len(df)} rows, skipped {skipped}.")

if len(df) == 0:
    raise ValueError("No data could be loaded from the JSONL file!")

dataset = Dataset.from_pandas(df)

def is_valid(example):
    messages = example.get("messages")
    return (
        messages is not None and
        isinstance(messages, list) and
        len(messages) > 0 and
        all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in messages)
    )

dataset = dataset.filter(is_valid, num_proc=4)
print(f"After structural filter: {len(dataset)} examples remain.")

if len(dataset) == 0:
    raise ValueError("No valid examples after filtering!")

# Standardize and format
dataset = standardize_sharegpt(dataset)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
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

# Remove any remaining empty texts
dataset = dataset.filter(lambda x: isinstance(x.get("text"), str) and len(x["text"].strip()) > 20, num_proc=4)

print(f"Final training dataset size: {len(dataset)} examples")
if len(dataset) == 0:
    raise ValueError("Dataset empty after cleaning — check your data.")

# ====================== TOKENIZE FOR PLAIN TRAINER ======================
print("Tokenizing dataset for plain Trainer...")

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )

     # ADD THIS SAFETY CHECK:
    # Ensure no token ID is out of bounds for the embedding layer
    vocab_size = model.config.vocab_size
    for ids in tokenized["input_ids"]:
        if any(id >= vocab_size for id in ids):
            # This identifies the culprit if a custom token is the issue
            print(f"Warning: Found OOB token ID. Max allowed: {vocab_size-1}")

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset.column_names,  # remove original columns
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
        gradient_accumulation_steps=32,
        warmup_steps=100,
        num_train_epochs=1.34,
        learning_rate=1e-6,
        warmup_ratio=0.30,
        max_grad_norm=0.4,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        optim="paged_adamw_8bit",
        weight_decay=0.12,
        lr_scheduler_type="cosine",
        output_dir="",
        report_to="none",
        save_strategy="steps",
        save_steps=120,
        save_total_limit=3,
        dataloader_num_workers=0,
        no_cuda=False,
        remove_unused_columns=False,
    ),
)

print("\nStarting training with plain Trainer (should preserve balanced sharding)...")
trainer.train()

model.save_pretrained("")
tokenizer.save_pretrained("")
print("Training completed!")
