import os
import shutil
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ====================== CONFIG ======================
model_path = "/home/eric/base_models/Qwen2.5-Coder-14B-Instruct"
jsonl_file = "/home/eric/Master_training_data/step1v2_split.jsonl"

max_seq_length = 4096
r = 128
lora_alpha = 128
use_rslora = True
# ===================================================

print("Loading model...")
shutil.rmtree("/home/eric/scripts/unsloth_compiled_cache", ignore_errors=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    dtype=None,
    load_in_4bit=True,
    device_map="balanced",
    max_memory={0: "14.4GB", 1: "14GB"},
)

model = FastLanguageModel.get_peft_model(
    model,
    r=r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=use_rslora,
)

# ====================== DATA LOADING ======================
print("\nLoading data...")

df = pd.read_json(jsonl_file, lines=True, encoding="utf-8", encoding_errors="ignore")
dataset = Dataset.from_pandas(df)

def is_valid(example):
    messages = example.get("messages")
    return messages is not None and isinstance(messages, list) and len(messages) > 0

dataset = dataset.filter(is_valid, num_proc=4)
dataset = standardize_sharegpt(dataset)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5",
                               mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"})

def formatting_prompts_func(examples):
    texts = []
    for convos in examples["messages"]:
        try:
            text = tokenizer.apply_chat_template(convos, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        except:
            texts.append("")
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=4)
dataset = dataset.filter(lambda x: len(x["text"].strip()) > 20, num_proc=4)

print(f"Final dataset size: {len(dataset)} examples")

# ====================== TRAINER ======================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=4,
    packing=False,

    args=TrainingArguments(
        per_device_train_batch_size=3,
        gradient_accumulation_steps=32,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=9e-6,
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
        output_dir="/media/eric/Models/Merged_models/custom_echo2.1_adapter",
        report_to="none",
        save_strategy="steps",
        save_steps=48,
    ),
)

print("\nStarting training...")
trainer.train()

model.save_pretrained("/media/eric/Models/Merged_models/custom_echov2.1_adapter")
tokenizer.save_pretrained("/media/eric/Models/Merged_models/custom_echo2.1_adapter")
print("Training completed!")
