import torch
import logging
logging.basicConfig(level=logging.DEBUG)

from peft import PeftModel
from transformers import AutoModelForCausalLM

base_path = ""
adapter1 = ""
merged_path = ""

base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16, device_map="auto",)
model = PeftModel.from_pretrained(base, adapter1)
merged = model.merge_and_unload()
merged.save_pretrained(merged_path, safe_serialization=True, max_shard_size="5GB",)
print("Merged model saved to", merged_path)
