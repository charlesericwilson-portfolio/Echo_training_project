# Echo Model Training Repo

This repo contains all the experiments, datasets, scripts, and notes for building and fine-tuning the Echo models — the brain behind the Echo red-team agent.

**Goal**: Build a capable, practical local model optimized for red-team reasoning, tool use, session awareness, and ethical decision-making.

### Current Status (April 2026)

- **Best performing model**: Custom [Qwen2.5 Coder 14B Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) (Q4)
- Training is stable again after many [struggles](https://github.com/charlesericwilson-portfolio/Echo_training_project/blob/main/Lessons_learned.md).
- Seq length successfully increased to **4096** with new training [script](https://github.com/charlesericwilson-portfolio/Echo_training_project/blob/main/Training%20scripts/unsloth_train_gpt.py).
- The **randomized interleaved dataset** (tool calls, reasoning, personality, ethics, and explanations mixed in random order, in triplicate) produced the best results so far — reaching a loss of **0.7** after 3 actual epochs 1 training epoch.

We had a lot of fun (and a lot of pain) figuring this out. The journey was messy, chaotic, and full of "why is this happening again?" moments, but we kept laughing through it and learned a ton.

### Training Journey

Started with a clean, structured, deduplicated dataset approach.  
Tried multiple configurations, different learning rates, and went up to 3 epochs on the clean data — but the loss plateaued at ~1.2 and the model stopped improving.

Then we tried something different: a **highly randomized interleaved dataset**.  
Same base data, but presented in completely different order each pass (never the same sequence twice). Effectively gave the model 3 epochs worth of exposure, but with maximum variety.

Result: The randomized version performed noticeably better and reached a loss of **0.7** — even though it only ran for one effective epochs.

This repo exists to document the messy, fun, and sometimes ridiculous process of trying to make a useful local model on consumer hardware.

### Tech Stack

- Base model: [Qwen2.5 Coder 14B Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct), also testing [Mistral 3 14B Reasoning](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) and [Qwen 3 14B](https://huggingface.co/Qwen/Qwen3-14B)
- Training: initially Unsloth + Hugging Face Trainer but had difficulty increasing the batch size to fit the examples with Qwen models.
- Final Training: Unsloth + SFT trainer
- Hardware: 2x RTX 5070 Ti (32GB total VRAM)
- CUDA 13.0 + open driver 590 open Kubuntu 24.04

### Specific changes made to support the framework
I changed the tokenizer chat template to accept user, assistant, system, and tool message types.
The Problem with Standard Tool Result Handling
Most OpenAI-compatible chat templates only define three message roles: system, user, and assistant. When an agent framework needs to return tool output back to the model, the only available slot is user — so tool results get injected as if the human typed them.
This creates a fundamental semantic mismatch. The model was trained to treat user messages as new instructions requiring a response. So when it sees tool output injected as a user message, it reasons: a user gave me new information, I should act on it — and calls another tool. Which produces more output. Which gets injected as another user message. Which triggers another tool call. The loop never resolves because nothing in the token stream signals "this task is complete."
The Solution
By extending the tokenizer config to recognize a native tool role as a first-class message type, the model receives tool output in a semantically distinct slot it was trained to understand as feedback from its own actions, not as a new request from a user. It knows the wrapper executed the command on its behalf. It knows the output is the result of something it initiated. And it knows when the task is done because the feedback confirms completion rather than prompting further action.

### Important [Lessons Learned](https://github.com/charlesericwilson-portfolio/Echo_training_project/blob/main/Lessons_learned.md)

- **Data order matters a lot.** Randomized interleaving (same data, different order every pass) helped the model generalize better and reduced overfitting compared to clean structured data.
- With limited VRAM it was important to ensure model sharding across both GPUs was working properly.
- Qwen has been easier to train than Mistral on this hardware at this moment but plan on diving deeper.
- Seq length is now stable all the way up to 8192.

### How to Reproduce

See the `training_scripts/` and `datasets/` folders for the exact scripts and example data used in the best runs.

The randomized interleaved dataset approach is currently our best performer.

### Future Plans

- Keep refining the randomized dataset style
- Possibly test Phi-4 or other bases if we hit walls again

This repo is the honest behind-the-scenes of the model training side. The actual agent wrapper lives in the main [Echo repo](https://github.com/charlesericwilson-portfolio/Echo_projectv0).

Built with collaboration from Grok (xAI) and a ridiculous amount of stubbornness.

— Charles (Eric), May 2026

"Even when it was breaking, we were still having fun figuring it out."

---

