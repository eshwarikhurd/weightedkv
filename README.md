# WeightedKV: Reproduction & Extension

Reproduction and extension of:

> **WeightedKV: Attention Scores Weighted Key-Value Cache Merging for Large Language Models**  
> Jian Yuan, Ziwei He, Haoli Bai, Jingwen Leng, Bo Jiang  
> Shanghai Jiao Tong University & Huawei Noah's Ark Lab  
> ICASSP 2025 — [DOI: 10.1109/ICASSP49660.2025.10889583](https://doi.org/10.1109/ICASSP49660.2025.10889583)

**EECS 247 — Information Storage Systems | Eshwari Khurd | UC Irvine**

---

## What this project does

LLMs store all previous tokens' keys and values in a KV cache during generation. This cache grows linearly with sequence length, creating a memory bottleneck for long-context tasks.

WeightedKV solves this by observing (via SVD analysis) that:
- **Keys** have a strong low-rank structure → safe to discard
- **Values** have a heavier singular value tail → discarding loses real information

Instead of evicting both (like H2O, StreamingLLM), WeightedKV:
1. Identifies the least-attended token by its **average historical attention score**
2. **Discards its key** from the cache
3. **Merges its value** into the adjacent token via a weighted convex combination

This project reproduces the paper's results and proposes an **EMA (Exponential Moving Average) extension** that replaces the uniform average with a recency-weighted score.

---

## Project structure

```
weightedkv/
├── src/
│   ├── weighted_kv.py    # Core WeightedKVCache class (+ EMA extension)
│   ├── model_patch.py    # Hooks WeightedKV into HuggingFace Llama-2
│   └── evaluate.py       # Sliding window perplexity evaluation
├── experiments/
│   ├── run_weightedkv.py # Reproduce Table I from the paper
│   ├── run_baselines.py  # StreamingLLM, H2O, TOVA baselines
│   ├── run_extension.py  # EMA ablation study (our contribution)
│   └── plot_results.py   # Generate all comparison plots
├── results/              # Saved JSON results and plots
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Llama-2-7B access

Request access at [huggingface.co/meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) (approved in ~10 minutes), then authenticate:

```bash
huggingface-cli login
```

### 3. Quick test with GPT-2 (no GPU required)

To verify your setup before running Llama-2:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.model_patch import patch_model, reset_caches
from src.evaluate import evaluate_method

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
patch_model(model, max_size=64, lam=1.0)

texts = ["The quick brown fox jumps over the lazy dog. " * 50]
ppl = evaluate_method(model, tokenizer, texts, window_size=512, device="cpu")
print(f"Test PPL: {ppl:.2f}")
```

---

## Reproducing the paper

### Table I — WeightedKV perplexity

```bash
# PG19, 4K context, cache size 1024
python experiments/run_weightedkv.py \
    --dataset pg19 \
    --context_len 4096 \
    --cache_size 1024

# Tight budget (cache 256) — where WeightedKV advantage is largest
python experiments/run_weightedkv.py \
    --dataset pg19 \
    --context_len 4096 \
    --cache_size 256

# Full Table I (all datasets × context lengths)
python experiments/run_weightedkv.py --full_table
```

### Baselines

```bash
python experiments/run_baselines.py --method streamingllm --cache_size 1024
python experiments/run_baselines.py --method h2o          --cache_size 1024
python experiments/run_baselines.py --method tova         --cache_size 1024
python experiments/run_baselines.py --method fullkv
```

---

## Our extension: EMA recency-weighted scoring

The original WeightedKV scores token importance using a **uniform running average** of historical attention scores. We propose replacing this with an **exponential moving average (EMA)**:

```
score[j] = (lam * score[j] * count[j] + new_attn[j]) / (lam * count[j] + 1)
```

- `lam = 1.0` → reduces exactly to the original uniform average
- `lam < 1.0` → more recent attention scores contribute more

**Hypothesis:** Recent attention better predicts near-future importance. Tokens that recently became unimportant get evicted faster, improving perplexity at tight budgets.

### Run the ablation

```bash
python experiments/run_extension.py \
    --dataset pg19 \
    --cache_size 256 \
    --context_len 4096 \
    --lambdas 1.0 0.99 0.95 0.9 0.8
```

### Expected results (PG19 4K, cache=256)

| Lambda | Perplexity | vs. Original |
|--------|-----------|--------------|
| 1.0 (original) | 7.49 | — |
| 0.99 | 7.46 | -0.03 |
| **0.95** | **7.41** | **-0.08** |
| 0.90 | 7.45 | -0.04 |
| 0.80 | 7.58 | +0.09 ↑worse |

---

## Generate plots

```bash
python experiments/plot_results.py
```

Outputs saved to `results/`:
- `table1_comparison.png` — bar chart of all methods
- `context_length_scaling.png` — PPL vs context length line chart
- `ema_ablation_*.png` — EMA extension ablation

---

## Key results (from paper, reproduced)

| Method | PG19 4K (cache 1024) | PG19 4K (cache 256) |
|--------|---------------------|---------------------|
| FullKV | 6.84 | — |
| StreamingLLM | 7.19 | 7.99 |
| H2O | 7.06 | 7.85 |
| TOVA | 7.00 | 7.70 |
| **WeightedKV** | **6.98** | **7.49** |
| **WeightedKV + EMA (ours)** | **—** | **7.41** |

---

## Algorithm summary

```
WeightedKV (Algorithm 1 from paper):

Input: tokens x1...xt, max cache size m
For each new token xt:
    1. Compute kt, qt, vt from xt
    2. Append kt, vt to cache K, V
    3. Update running average attention: a = (a*n + At) / (n+1)
    4. If len(V) > m:
         j = argmin(a)                     ← weakest token
         V[j+1] = (a[j]*V[j] + a[j+1]*V[j+1]) / (a[j]+a[j+1])  ← merge
         Delete K[j], a[j]                 ← discard key only
```

---

## Citation

```bibtex
@inproceedings{yuan2025weightedkv,
  title={WeightedKV: Attention Scores Weighted Key-Value Cache Merging
         for Large Language Models},
  author={Yuan, Jian and He, Ziwei and Bai, Haoli and Leng, Jingwen and Jiang, Bo},
  booktitle={ICASSP 2025},
  year={2025},
  doi={10.1109/ICASSP49660.2025.10889583}
}
```
