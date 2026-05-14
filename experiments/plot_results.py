"""
plot_results.py
---------------
Generate all comparison plots from saved results JSON files.

Reproduces:
  - Table I comparison bar chart (WeightedKV vs all baselines)
  - Figure 4 style: merge vs eviction ablation
  - EMA extension ablation chart

Usage:
    # After running all experiments:
    python experiments/plot_results.py
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

COLORS = {
    "FullKV":       "#1F507F",
    "StreamingLLM": "#888780",
    "H2O":          "#0F6E56",
    "TOVA":         "#BA7517",
    "CaM":          "#993C1D",
    "WeightedKV":   "#534AB7",
}

PAPER_RESULTS = {
    # (method, dataset, ctx, cache_size) -> perplexity
    # From Table I, WeightedKV paper ICASSP 2025
    ("FullKV",       "pg19", 4096,  None):  6.84,
    ("StreamingLLM", "pg19", 4096,  1024):  7.19,
    ("StreamingLLM", "pg19", 8192,  1024):  7.18,
    ("StreamingLLM", "pg19", 16384, 1024):  7.20,
    ("TOVA",         "pg19", 4096,  1024):  7.00,
    ("TOVA",         "pg19", 8192,  1024):  7.05,
    ("TOVA",         "pg19", 16384, 1024):  7.14,
    ("H2O",          "pg19", 4096,  1024):  7.06,
    ("H2O",          "pg19", 8192,  1024):  7.06,
    ("H2O",          "pg19", 16384, 1024):  7.24,
    ("WeightedKV",   "pg19", 4096,  1024):  6.98,
    ("WeightedKV",   "pg19", 8192,  1024):  6.99,
    ("WeightedKV",   "pg19", 16384, 1024):  7.10,
    ("StreamingLLM", "pg19", 4096,  256):   7.99,
    ("TOVA",         "pg19", 4096,  256):   7.70,
    ("H2O",          "pg19", 4096,  256):   7.85,
    ("WeightedKV",   "pg19", 4096,  256):   7.49,
}


def load_results():
    """Load all result JSON files from the results/ directory."""
    results = {}
    for f in glob.glob("results/*.json"):
        with open(f) as fp:
            data = json.load(fp)
        if "method" in data:
            key = (data["method"], data.get("dataset",""),
                   data.get("context_len",0), data.get("cache_size"))
            results[key] = data["perplexity"]
    return results


def plot_table1_comparison(results=None):
    """Bar chart comparing all methods on PG19 at 4K context, both cache sizes."""
    methods = ["StreamingLLM", "TOVA", "H2O", "WeightedKV"]
    cache_sizes = [1024, 256]
    x = np.arange(len(methods))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("PG19 Perplexity Comparison (4K Context)\nLlama-2-7B | WeightedKV Paper Table I",
                 fontsize=13)

    for ax_idx, cache_size in enumerate(cache_sizes):
        ax = axes[ax_idx]
        paper_ppls = [PAPER_RESULTS.get((m, "pg19", 4096, cache_size), None) for m in methods]
        repro_ppls = [results.get((m.lower().replace(" ",""), "pg19", 4096, cache_size), None)
                      for m in methods] if results else [None]*len(methods)

        bars1 = ax.bar(x - width/2, paper_ppls, width,
                       label="Paper (reported)", color=[COLORS.get(m, "#888") for m in methods],
                       alpha=0.85, edgecolor='white')
        valid_repro = [p for p in repro_ppls if p is not None]
        if valid_repro:
            bars2 = ax.bar(x + width/2, repro_ppls, width,
                           label="Ours (reproduced)",
                           color=[COLORS.get(m, "#888") for m in methods],
                           alpha=0.45, edgecolor='black', linewidth=0.8, linestyle='--')

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel("Perplexity (↓ better)", fontsize=11)
        ax.set_title(f"Cache Size = {cache_size}", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_ylim(min(p for p in paper_ppls if p) * 0.97,
                    max(p for p in paper_ppls if p) * 1.03)

        for bar, ppl in zip(bars1, paper_ppls):
            if ppl:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{ppl:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/table1_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved: results/table1_comparison.png")
    plt.close()


def plot_context_length_scaling():
    """Line plot showing perplexity vs context length for each method."""
    methods = ["StreamingLLM", "TOVA", "H2O", "WeightedKV"]
    ctx_lens = [4096, 8192, 16384]
    cache_size = 1024

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        ppls = [PAPER_RESULTS.get((method, "pg19", ctx, cache_size)) for ctx in ctx_lens]
        if all(p is not None for p in ppls):
            ax.plot([c//1024 for c in ctx_lens], ppls,
                    marker='o', label=method,
                    color=COLORS.get(method, "#888"),
                    linewidth=2, markersize=6)

    ax.set_xlabel("Context Length (K tokens)", fontsize=11)
    ax.set_ylabel("Perplexity (↓ better)", fontsize=11)
    ax.set_title("PG19 Perplexity vs Context Length\nCache Size=1024 | Llama-2-7B", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xticks([4, 8, 16])
    ax.set_xticklabels(["4K", "8K", "16K"])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/context_length_scaling.png", dpi=150, bbox_inches='tight')
    print("Saved: results/context_length_scaling.png")
    plt.close()


def plot_ema_ablation():
    """Load and plot EMA ablation results if available."""
    files = glob.glob("results/ema_ablation_*.json")
    if not files:
        print("No EMA ablation results found. Run run_extension.py first.")
        return

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        lambdas = sorted([float(k) for k in data.keys()], reverse=True)
        ppls = [data[str(l)] for l in lambdas]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2E74B5' if l == 1.0 else '#534AB7' for l in lambdas]
        bars = ax.bar([str(l) for l in lambdas], ppls, color=colors,
                      edgecolor='white', linewidth=0.5)

        for bar, ppl in zip(bars, ppls):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{ppl:.3f}', ha='center', va='bottom', fontsize=9)

        best_idx = ppls.index(min(ppls))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)

        ax.set_xlabel('Lambda (EMA decay factor)', fontsize=11)
        ax.set_ylabel('Perplexity (↓ better)', fontsize=11)
        ax.set_title('EMA Extension Ablation: WeightedKV on PG19\nCache=256 | lam=1.0 is original method',
                     fontsize=12)

        patch1 = mpatches.Patch(color='#2E74B5', label='Original (uniform avg, lam=1.0)')
        patch2 = mpatches.Patch(color='#534AB7', label='EMA extension (lam<1.0)')
        ax.legend(handles=[patch1, patch2], fontsize=9)

        plt.tight_layout()
        out = f.replace(".json", "_plot.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()


if __name__ == "__main__":
    print("Generating all plots...\n")
    results = load_results()
    plot_table1_comparison(results)
    plot_context_length_scaling()
    plot_ema_ablation()
    print("\nAll plots saved to results/")
