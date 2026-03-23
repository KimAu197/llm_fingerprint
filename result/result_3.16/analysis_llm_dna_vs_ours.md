# LLM DNA Paper Dataset vs. Our Dataset: Analysis

---

## Part 1 — LLM DNA 305-Model Dataset Analysis

The LLM DNA paper (Wu et al., ICLR 2026) evaluates on 305 text-generative LLMs from 153 organizations
on Hugging Face, covering decoder-only and encoder-decoder architectures from 100M to 72B parameters.

---

### 1.1 Model Family Distribution

| Family | Count |
|---|---|
| Qwen (Alibaba) | 33 |
| Llama – community derivatives | 20 |
| Meta Llama (official) | 16 |
| GPT legacy (GPT-1, GPT-Neo, GPT-J, GPT-NeoX, DialoGPT) | 12 |
| Mistral / Mixtral | 11 |
| EleutherAI (Pythia, GPT-Neo, Polyglot) | 11 |
| DeepSeek | 10 |
| Microsoft Phi | 10 |
| Vicuna | 8 |
| Yi (01-ai) | 7 |
| Falcon (TII) | 7 |
| BLOOM / BLOOMZ | 6 |
| T5 / Flan-T5 (Google) | 5 |
| Google Gemma | 5 |
| Databricks Dolly | 4 |
| Code Llama | 3 |
| DeepSeek-R1 Distill (Qwen-base) | 3 |
| MosaicML MPT | 3 |
| Other / niche community models | ~47 |

**Top contributing organizations:** Qwen (27 models), meta-llama (15), microsoft (14),
deepseek-ai (12), EleutherAI (11), 01-ai (7), google (7), tiiuae (7).

The paper deliberately spans a wide historical range: from early models (GPT-1, T5-small, Dolly)
through recent reasoning models (Qwen3, DeepSeek-R1, Phi-4), enabling a full evolutionary tree.

---

### 1.2 Parameter Size Distribution

| Size bucket | Count | % (of 291 with known size) |
|---|---|---|
| < 1B (tiny) | 21 | 7.2% |
| 1 – 3B (small) | 33 | 11.3% |
| 3 – 8B (mid-small) | 104 | 35.7% |
| 8 – 15B (mid) | 80 | 27.5% |
| 15 – 35B (large) | 42 | 14.4% |
| 35 – 75B (very large) | 11 | 3.8% |

14 models have unknown parameter counts (typically older community uploads).

**Takeaway:** The distribution is strongly centered at 3–15B (63% of models), which reflects
the practical accessibility of that size range on HuggingFace. Very large models (>35B) are rare (11 models).

---

### 1.3 Architecture Distribution

| Architecture | Count |
|---|---|
| Decoder Only | 267 (87.5%) |
| Unknown | 25 (8.2%) |
| Encoder-Decoder | 7 (2.3%) |
| Encoder Only | 6 (2.0%) |

The paper is almost entirely decoder-only. Encoder-decoder models (T5, BART, Flan-T5)
and encoder-only models (DialoGPT, DrugGPT, BioMedLM) are included as edge cases to test
whether LLM DNA generalizes across architecture types.

---

### 1.4 License Distribution (Top 10)

| License | Count |
|---|---|
| Apache-2.0 | 131 (42.9%) |
| MIT | 39 (12.8%) |
| Other | 37 (12.1%) |
| Unknown | 27 (8.9%) |
| Llama-2 | 23 (7.5%) |
| llama3 | 9 (2.9%) |
| CC-BY-NC-4.0 | 8 (2.6%) |
| BigScience RAIL 1.0 | 7 (2.3%) |
| Gemma | 5 (1.6%) |
| llama3.2 | 4 (1.3%) |

Apache-2.0 dominates, as most major open-weight model families (Qwen, DeepSeek, Falcon, EleutherAI)
use it. Proprietary/community licenses (Llama-2, Gemma, BigScience RAIL) collectively cover ~15%.

---

## Part 2 — Overlap with Our Dataset

Our dataset (`top_overlap_pairs.csv`) contains **88 models**, focused on a modern, fine-tuning-centric
slice of HuggingFace — primarily Qwen 2.x/3.x, Llama 3.x, Gemma 2, and community derivatives
from organizations like unsloth, nvidia, Gensyn, and arcee-ai.

---

### 2.1 Summary

| Metric | Value |
|---|---|
| Models in our set | 88 |
| Models in LLM DNA 305 | 305 |
| Exact matches | **19 (21.6% of our set)** |
| Our models not in DNA 305 | 69 (78.4%) |

---

### 2.2 The 19 Overlapping Models

| Model | Family | Size | Architecture |
|---|---|---|---|
| Qwen/Qwen2-0.5B-Instruct | Qwen2 | 0.5B | Decoder Only |
| Qwen/Qwen2-7B-Instruct | Qwen2 | 8B | Decoder Only |
| Qwen/Qwen2.5-0.5B-Instruct | Qwen2.5 | 0.5B | Decoder Only |
| Qwen/Qwen2.5-1.5B | Qwen2.5 | 2B | Decoder Only |
| Qwen/Qwen2.5-1.5B-Instruct | Qwen2.5 | 2B | Decoder Only |
| Qwen/Qwen2.5-3B-Instruct | Qwen2.5 | 3B | Decoder Only |
| Qwen/Qwen2.5-7B | Qwen2.5 | 8B | Decoder Only |
| Qwen/Qwen2.5-7B-Instruct | Qwen2.5 | 8B | Decoder Only |
| Qwen/Qwen2.5-Coder-7B | Qwen2.5-Coder | 8B | Decoder Only |
| Qwen/Qwen2.5-Math-1.5B-Instruct | Qwen2.5-Math | 2B | Decoder Only |
| Qwen/Qwen3-0.6B | Qwen3 | 0.8B | Decoder Only |
| Qwen/Qwen3-1.7B | Qwen3 | 2B | Decoder Only |
| Qwen/Qwen3-4B | Qwen3 | 4B | Decoder Only |
| meta-llama/Llama-3.1-8B-Instruct | Llama 3.1 | 8B | Decoder Only |
| meta-llama/Llama-3.2-1B-Instruct | Llama 3.2 | 1B | Decoder Only |
| meta-llama/Llama-3.2-3B-Instruct | Llama 3.2 | 3B | Decoder Only |
| mistralai/Mistral-7B-Instruct-v0.3 | Mistral 7B | 7B | Decoder Only |
| nvidia/OpenReasoning-Nemotron-1.5B | Nvidia Nemotron | 2B | Decoder Only |
| tiiuae/Falcon3-1B-Instruct | Falcon 3 | 2B | Decoder Only |

All 19 overlapping models are **official/base model releases** from major organizations.

---

### 2.3 Why the Overlap Is Low (78% of Our Set Is Exclusive)

Our 69 non-overlapping models are largely absent from the DNA 305 because they represent:

1. **Unsloth quantized/reformatted copies** (15 models: `unsloth/Llama-3.1-8B`,
   `unsloth/Qwen2.5-1.5B-Instruct`, etc.) — the DNA paper only includes the originals.

2. **Gensyn distributed-training replicas** (3 models: `Gensyn/Qwen2.5-*`) — exact
   functional clones of Qwen2.5 originals, not included in the DNA 305.

3. **Fine-tuned derivatives from smaller orgs** (e.g., `arcee-ai/Meraj-Mini`,
   `aisingapore/Llama-SEA-LION-v3-8B`, `allenai/Llama-3.1-Tulu-3-8B`,
   `NousResearch/Hermes-*`, `nvidia/OpenMath2-*`) — the DNA paper focuses on
   base models and a curated set of derivatives, not these.

4. **Newer/niche models** (`google/gemma-2-2b`, `google/gemma-2-2b-it`,
   `nvidia/DLER-*`, `PowerInfer/SmallThinker-3B-Preview`,
   `Writer/palmyra-mini`, `Menlo/*`) — released after or not selected for the DNA 305.

5. **Base variants vs. instruct** (`meta-llama/Llama-3.1-8B` base,
   `Qwen/Qwen2.5-3B` base, `Qwen/Qwen2.5-Coder-1.5B`, `Qwen/Qwen2.5-Math-1.5B`) —
   the DNA paper sometimes includes only the instruct version, or vice versa.

---

### 2.4 Complementary Coverage

| Dimension | LLM DNA 305 | Our 88 |
|---|---|---|
| Time range | 2019–2025 (broad, historical) | 2024–2025 (modern) |
| Model diversity | 153 orgs, all families | ~26 orgs, Qwen/Llama/Gemma heavy |
| Size range | 1M – 72B | 0.5B – 8B (small-mid) |
| Fine-tune depth | Mix of base + instruction-tuned | Mostly fine-tuned/instruction-tuned |
| Community derivatives | Limited | Heavy (unsloth, Gensyn, arcee-ai, nvidia) |
| Encoder/Enc-Dec models | Yes (13 models) | No (all decoder-only) |

The two datasets are **complementary rather than redundant**. The LLM DNA 305 covers breadth
and historical depth across the entire open-source LLM ecosystem. Our dataset covers
depth within the modern Qwen and Llama 3.x ecosystems, with a focus on fine-tuning lineage
and community repackaging patterns — exactly the territory that the DNA paper leaves sparse.
