import torch
import logging
logging.basicConfig(level=logging.DEBUG)

from peft import AutoPeftModelForCausalLM
base_path = ""
adapter_path = ""# <<< Your LoRA folder
merged_path = ""

print("Loading PEFT model with automatic device mapping and offloading...")
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.float16,
    device_map="cpu",                  # Splits across both GPUs + CPU offload if needed
    offload_folder="offload_temp",      # Required for offloading
    offload_state_dict=True,
    low_cpu_mem_usage=True,             # Critical for not blowing RAM
)

print("Merging LoRA into base...")
merged = model.merge_and_unload(
    progressbar=True,
    safe_merge=False,                    # Memory-safer algorithm
)

print("Saving full merged model...")
merged.save_pretrained(
    merged_path,
    safe_serialization=True,
    max_shard_size="3GB",               # Keeps files manageable
)

print(f"Done! Merged model saved to {merged_path}")
