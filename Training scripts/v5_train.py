import os
import shutil
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback

# ====================== CONFIG ======================
model_path = "/home/eric/base_models/Qwen2.5-Coder-14B-Instruct"
jsonl_file = "/home/eric/Master_training_data/v5_base.jsonl"

max_seq_length = 7000
r = 128
lora_alpha = 128
use_rslora = True

val_split = 0.10          # 10% validation set
eval_steps = 15           # Evaluate every N steps
patience = 2             # Early stop after evals with no improvement
# ===================================================

print("Loading model and modified tokenizer...")
shutil.rmtree("/home/eric/scripts/unsloth_compiled_cache", ignore_errors=True)

# 1. Load the model and tokenizer (Unsloth automatically loads your disk configurations)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    dtype=None,
    load_in_4bit=True,
    device_map="balanced",
    max_memory={0: "15.4GB", 1: "15GB"},
)

print(f"Synchronized Vocabulary Size: {len(tokenizer)}")

# 2. Initialize PEFT targeting standard layers safely within 4-bit bounds
model = FastLanguageModel.get_peft_model(
    model,
    r=r,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=use_rslora,
)

model.config.use_cache = False

# ====================== DATA ======================
print("\nLoading and splitting data...")
df = pd.read_json(jsonl_file, lines=True, encoding="utf-8", encoding_errors="ignore")
dataset = Dataset.from_pandas(df)

def is_valid(example):
    messages = example.get("messages")
    return messages is not None and isinstance(messages, list) and len(messages) > 0

dataset = dataset.filter(is_valid, num_proc=10)
dataset = standardize_sharegpt(dataset)

custom_tuple = (tokenizer.chat_template, tokenizer.eos_token)

tokenizer = get_chat_template(
    tokenizer,
    chat_template=custom_tuple,  # Using your custom tuple layout configuration
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt", "tool": "tool"}
)

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

# Train / Val split
split_dataset = dataset.train_test_split(test_size=val_split, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

# ====================== TRAINER ======================
# Initialize SFTTrainer with packing=False so we can apply the response mask
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=10,
    packing=False,                    # Crucial: Disabled so token masking works perfectly

    args=TrainingArguments(
        per_device_train_batch_size=6,
        gradient_accumulation_steps=12,
        num_train_epochs=3,
        learning_rate=6e-5,
        warmup_ratio=0.30,
        max_grad_norm=0.4,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_steps=eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        weight_decay=0.12,
        lr_scheduler_type="cosine",
        output_dir="/media/eric/Models/Merged_models/custom_echo_instroder2.2/adapter",
        report_to="none",
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
)

# ====================== LOSS MASKING OVERRIDE ======================
# 3. Apply Unsloth's native responder mask targeting your custom template tags
print("\nApplying custom chat template response masking...")
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",      # Looks for the start of user instructions
    response_part="<|im_start|>assistant\n",   # Computes loss ONLY on content after this tag
)

print("\nStarting training...")
try:
    trainer.train(resume_from_checkpoint=True)
except:
    print("No checkpoint found, starting fresh.")
    trainer.train(resume_from_checkpoint=False)

model.save_pretrained("/media/eric/Models/Merged_models/custom_echo_instroder2.2/adapter")
tokenizer.save_pretrained("/media/eric/Models/Merged_models/custom_echo_instroder2.2/adapter")
print("Training completed successfully!")
