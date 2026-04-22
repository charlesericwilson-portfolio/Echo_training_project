# Lessons Learned – Echo Model Training & Agent Development
### Project Overview
Fine-tuning Qwen2.5-14B-Coder-Instruct for strong step-by-step reasoning, precise tool-calling syntax, and reliable "when/why" tool use decisions. 

This document captures the hard-earned lessons from building and training Echo across three long threads. It’s messy, frustrating, and sometimes funny — but that’s how real learning happens.

### 1. Environment & Dependency Hell

- Pure conda or uv pip venvs are extremely fragile when mixing Unsloth, Torch, CUDA, and unsloth_zoo.  
- Order of installation matters more than most people admit. Torch must be installed first, then transformers early, then Unsloth, then unsloth_zoo with `--no-deps`.
- Unsloth Zoo is often the main source of version conflicts and VRAM bloat, even if you don’t explicitly use it.
- Switching to Conda helped stabilize things when pip became unbearable.
- CUDA version mismatches (12.0 vs 12.8 vs 13.1) cause silent OOMs and weird runtime behavior. Always verify with `torch.version.cuda` and `nvcc --version`.

**Lesson:** Never assume “it should just work.” Pin versions aggressively and install in strict order.

### 2. Training Dynamics & Dataset Design

- **Randomized interleaved datasets beat clean structured ones.**
- Started with ~3767 high-quality examples, later expanded and split to fit context constraints then deduped and cleaned to get ~7338 examples.
 
- The chaotic version (tool calls → reasoning → personality → ethics → explanations, all mixed and in different order each pass) generalized much better than the deduplicated, logically ordered dataset — even with fewer effective epochs.

- Changing the order of the same data every epoch acts as powerful regularization. It prevents the model from memorizing sequences and forces it to learn more robust connections.

- Small models are extremely sensitive to data order and repetition. Clean, deduped datasets can cause early plateauing (loss stuck at ~1.2).

- High rank (r=192) + lower alpha (64) + higher LR works for 14B but can cause instability on smaller models.

**Lesson:** For small-to-medium models, embrace controlled chaos in the dataset. Interleaving and random order can be more effective than perfect structure.

### 3. VRAM Management & Sharding
We spent many hours trying to train a Qwen2.5-Coder-14B-Instruct model using QLoRA on two RTX 5070 Ti GPUs (16 GB each). The goal was stable training with reasonable sequence lengths.

### Final Working Configuration
- **Model**: Qwen2.5-Coder-14B-Instruct (4-bit)
- **Device Map**: `balanced` with `max_memory`
- **Batch Size**: 3
- **Gradient Accumulation**: 32
- **Max Seq Length**: 4098 (got unsloth sft trainer working properly)
- **LoRA Rank (r)**: 128
- **Trainer**: SFTTrainer
- **Learning Rate**: Very low (ended at ~9e-6 or lower)
- **Data**: Split long replies into shorter chunks with "continue" prompts

### Key Lessons Learned

1. **Unsloth's Fast Kernels Are Problematic on Blackwell**
   - The custom Triton kernels (`fast_rms_layernorm`, fused CE loss) frequently trigger `device-side assert` on RTX 50-series.
   - **Lesson**: On Blackwell + Qwen2.5, prefer standard HF + bitsandbytes + PEFT over Unsloth's accelerated paths when stability matters.

2. **SFTTrainer Was the Hidden Villain**
   - SFTTrainer silently injected Accelerate/DDP logic that fought against `device_map="balanced"`.
   - With limited VRAM (2x 5070 Ti), `device_map="auto"` and SFT trainer accelerate or torch run often fails to shard properly.
   - Unsloth’s built-in SFT trainer, accelerate under the hood, has issues with Qwen sharding and FSDP.  
   - Switching to Hugging Face Trainer + unsloth `device_map: "balanced"` was necessary to truly shard the model across both GPUs and get headroom.
   - Switching to plain `Trainer` + manual tokenization fixed the VRAM imbalance (from 10/7 GB → ~7-8 GB balanced).
   - Even with 5 GB free on one GPU, sudden spikes can still cause OOM due to fragmentation. Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps.
   **Lesson:** When VRAM is tight, don’t trust Unsloth’s SFT trainer for large models. Use HF Trainer + manual sharding settings for better control.
     
3. **Sequence Length Is Heavily Limited by RoPE Scaling**
   - Qwen2.5-Coder models have strict internal RoPE assumptions.
   - Qwen with RoPE scaling is very stubborn about context length in current Unsloth/HF setups. We could only reliably get to 2048.
   - Pushing beyond ~2048 tokens reliably triggered kernel crashes or shape mismatches in the fused loss.
   - **Workaround**: Split long conversations into multiple shorter examples with "Continue with the same scenario..." prompts.
   - Seq len is often the real bottleneck, not just VRAM.
   **Lesson:** Sometimes you have to work within the model’s current limitations (1024–2048) and make the dataset fit, rather than fighting for longer context.
   - Mistral 15B was surprisingly harder to set up to train than Qwen on this hardware.

4. **Batch Size vs Gradient Accumulation Trade-off**
   - `per_device_train_batch_size=2` often re-introduced device errors.
   - Staying at batch size 1 + high gradient accumulation (32) was more stable.

5. **Padding and Label Masking Matter A Lot**
   - `padding="max_length"` + proper `-100` masking for padding tokens was essential to avoid shape mismatches in the loss.

6. **Learning Rate Needs to Be Extremely Conservative**
   - Normal QLoRA rates (2e-4) caused loss to drop too fast or become unstable.
   - Ended up using very low rates (~2e-6) for smooth, controlled training.

7. **Save Strategy Is Critical for Long Runs**
   - Always use `save_strategy="steps"`, `save_steps=646`, `save_total_limit=3` for multi-hour trainings.

### What We Tried That Failed
- Mixing `device_map="balanced"` with `torchrun` / DDP
- Using SFTTrainer with Unsloth
- Aggressive sequence lengths (2048+) without proper RoPE scaling
- Relying on Unsloth's fused CE loss and RMS kernels on Blackwell

### Recommendations for Similar Setups
- Start with plain `Trainer` + standard PEFT when using Qwen2.5 on 50-series GPUs.
- Split long examples rather than fighting for high `max_seq_length`.
- Use very low learning rates if loss drops too fast.
- Always monitor VRAM split closely — imbalance usually means hidden DDP/Accelerate interference.

### What Worked Well
- Qwen2.5-Coder-Instruct is an exceptionally strong base model for reasoning and tool-use tasks — it significantly outperformed other 14B-class models I tried.
- Unsloth + 4-bit QLoRA enabled training on limited hardware.
- Using `lora_alpha = .5×r` (r=196, alpha=98) provided more stable training than previous alpha experiments.
- Deduplicating the dataset (reduced to 7338 examples) removed noise and improved training stability.
- Very low learning rates (down to 2e-6) successfully slowed down the loss drop, allowing more gradual learning.
- Saving checkpoints every 230 steps (end of each epoch) worked reliably with the regular `Trainer` + FSDP setup.

### Major Challenges & Pain Points
- **RoPE/YaRN scaling issues**: Only `max_seq_length = 1024-2048` worked reliably with Unsloth + regular Trainer + FSDP. Higher values caused errors or broken behavior.
- **Context length limitation**: Forced splitting of long reasoning traces into many short segments + adding "Continue from where you left off on part X" prompts.
- Splitting the data made loss drop faster than expected, even with very low LR, because the model was learning short fragments and continuation patterns.
- Gradient accumulation steps (16 vs 32) had almost no effect on wall-clock training time due to tiny per-device batch size + FSDP overhead.
- FSDP + regular `Trainer` was required for memory reasons, but made iteration slower and less flexible than SFTTrainer.

### Key Takeaways
- **Context length matters enormously** for deep reasoning and tool use. Hard context limits force compromises in data quality.
- **Data splitting has trade-offs**: It allows training but fragments long reasoning chains. The model learns continuation format well, but struggles more on clean, self-contained prompts.
- Low LR + moderate-to-high alpha + higher weight decay (0.1+) + higher dropout (0.15) is a good recipe when you want slower, more controlled learning.
- Always evaluate on **clean, unsplit** examples — training loss can be misleading.
- Deduplication is essential when working with repeated or split data.
- Document every hyperparameter experiment (r, alpha, LR, dropout, weight_decay, seq length). You will forget the details later.
- Qwen2.5-Coder is worth fighting the ecosystem quirks for this specific use case.

### What I Would Do Differently Next Time
- Spend more upfront time on data formatting and try to minimize aggressive splitting.
- Keep a larger set of clean, full-length examples for validation from the beginning.
- Test smaller models (7B or 32B) first for faster iteration before scaling to 14B.
- Consider models with more flexible RoPE scaling if long coherent reasoning is critical.
- Allocate more time for qualitative evaluation instead of only watching loss.

**I am not an expert in this this is just what worked for me there are many ways to do this, this is just the best I could get to work reliably.**

We’re still figuring it out — and that’s okay. The goal was never perfection on the first try. It was to build something useful while learning how all the pieces actually work together.

---
