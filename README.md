# Echo Model Training Repo

This repo contains all the experiments, datasets, scripts, and notes for building and fine-tuning the Echo models — the brain behind the Echo red-team agent.

**Goal**: Build a capable, practical local model optimized for red-team reasoning, tool use, session awareness, and ethical decision-making.

### Current Status (April 2026)

- **Best performing model**: Custom [Qwen2.5 Coder 14B Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) (Q4)
- Training is stable again after many [struggles](https://github.com/charlesericwilson-portfolio/Echo_training_project/blob/main/Lessons_learned.md).
- Seq length successfully increased to **2048**.
- The **randomized interleaved dataset** (tool calls, reasoning, personality, ethics, and explanations mixed in random order, in triplicate) produced the best results so far — reaching a loss of **0.7** after 1.33 effective epochs.

We had a lot of fun (and a lot of pain) figuring this out. The journey was messy, chaotic, and full of "why is this happening again?" moments, but we kept laughing through it and learned a ton.

### Training Journey

Started with a clean, structured, deduplicated dataset approach.  
Tried multiple configurations, different learning rates, and went up to 3 epochs on the clean data — but the loss plateaued at ~1.2 and the model stopped improving.

Then we tried something different: a **highly randomized interleaved dataset**.  
Same base data, but presented in completely different order each pass (never the same sequence twice). Effectively gave the model 3 epochs worth of exposure, but with maximum variety.

Result: The randomized version performed noticeably better and reached a loss of **0.7** — even though it only ran for one effective epochs.

This repo exists to document the messy, fun, and sometimes ridiculous process of trying to make a useful local model on consumer hardware.

### Tech Stack

- Base model: [Qwen2.5 Coder 14B Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) (also tested [Mistral 3 14B Reasoning](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) — Just as much if not more headache with rope scaling)
- Training: Unsloth + Hugging Face Trainer (SFTTrainer had too many issues with sharding always reverting to DDP)
- Hardware: 2x RTX 5070 Ti (32GB total VRAM)
- CUDA 13.0 + open driver 590 open Kubuntu 24.04

### Important [Lessons Learned](https://github.com/charlesericwilson-portfolio/Echo_training_project/blob/main/Lessons_learned.md)

- **Data order matters a lot.** Randomized interleaving (same data, different order every pass) helped the model generalize better and reduced overfitting compared to clean structured data.
- With limited VRAM, we had to switch to the Hugging Face Trainer instead of Unsloth’s built-in trainer to get proper model sharding across both GPUs.
- Qwen has been easier to train than Mistral on this hardware at this moment but plan on diving deeper.
- Seq length is now stable at 2048.

(For a deeper dive into the VRAM/sharding struggles and why we had to switch trainers, see the separate `LESSONS_LEARNED.md` file.)

### How to Reproduce

See the `training_scripts/` and `datasets/` folders for the exact scripts and data used in the best runs.

The randomized interleaved dataset approach is currently our best performer.

### Future Plans

- Keep refining the randomized dataset style
- Push for even better context handling at 2048+
- Possibly test Phi-4 or other bases if we hit walls again

This repo is the honest behind-the-scenes of the model training side. The actual agent wrapper lives in the main [Echo repo](https://github.com/charlesericwilson-portfolio/Echo_projectv0).

Built with a lot of help from Grok (xAI) and a ridiculous amount of stubbornness.

— Charles (Eric), April 2026

"Even when it was breaking, we were still having fun figuring it out."

---

