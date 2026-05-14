"""
run_extension.py
----------------
Ablation study for the EMA recency-weighted scoring extension.

Evaluates WeightedKV across lambda values {1.0, 0.99, 0.95, 0.9, 0.8}
on PG19 at cache size 256 and context lengths 4K, 8K.

lam=1.0 → original uniform average (paper baseline)
lam<1.0 → recency-weighted (our extension)

Usage:
    python experiments/run_extension.py \
        --dataset pg19 \
        --cache_size 256 \
        --context_len 4096 \
        --lambdas 1.0 0.99 0.95 0.9 0.8

    # Generate comparison plot
    python experiments/run_extension.py --plot_only
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_patch import patch_model, reset_caches
from src.evaluate import evaluate_method, load_pg19_samples

MODEL_ID = "meta-llama/Llama-2-7b-hf"


def run_ablation(args):
    print(f"\nEMA Ablation Study")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Cache size:  {args.cache_size}")
    print(f"  Context len: {args.context_len}")
    print(f"  Lambda vals: {args.lambdas}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    texts = load_pg19_samples(n=args.n_samples)
    results = {}

    for lam in args.lambdas:
        print(f"\n--- Lambda = {lam} ---")
        patch_model(model, max_size=args.cache_size, lam=lam)

        ppl = evaluate_method(
            model, tokenizer, texts,
            window_size=args.context_len,
            device="cuda" if torch.cuda.is_available() else "cpu",
            reset_fn=reset_caches,
            verbose=True,
        )
        results[str(lam)] = ppl
        print(f"Lambda={lam} | PPL={ppl:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    fname = f"results/ema_ablation_{args.dataset}_ctx{args.context_len}_cache{args.cache_size}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("EMA ABLATION RESULTS")
    print(f"{'='*50}")
    for lam, ppl in sorted(results.items(), key=lambda x: float(x[0]), reverse=True):
        marker = " <-- original" if float(lam) == 1.0 else ""
        print(f"  lambda={lam}: PPL={ppl:.4f}{marker}")

    best_lam = min(results, key=results.get)
    print(f"\nBest lambda: {best_lam} (PPL={results[best_lam]:.4f})")
    print(f"Improvement over uniform (lam=1.0): "
          f"{results.get('1.0', results.get('1', 0)) - results[best_lam]:.4f}")

    generate_plot(results, args)
    return results


def generate_plot(results, args):
    """Generate bar chart comparing PPL across lambda values."""
    try:
        import matplotlib.pyplot as plt

        lambdas = sorted([float(k) for k in results.keys()], reverse=True)
        ppls    = [results[str(l)] for l in lambdas]
        colors  = ['#2E74B5' if l == 1.0 else '#534AB7' for l in lambdas]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar([str(l) for l in lambdas], ppls, color=colors,
                      edgecolor='white', linewidth=0.5)

        # Annotate bars
        for bar, ppl in zip(bars, ppls):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{ppl:.2f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Lambda (EMA decay factor)', fontsize=11)
        ax.set_ylabel('Perplexity (↓ better)', fontsize=11)
        ax.set_title(
            f'EMA Ablation: WeightedKV on {args.dataset.upper()}\n'
            f'Context={args.context_len//1024}K | Cache={args.cache_size}',
            fontsize=12
        )
        ax.legend(
            handles=[
                plt.Rectangle((0,0),1,1, color='#2E74B5', label='Original (uniform avg)'),
                plt.Rectangle((0,0),1,1, color='#534AB7', label='EMA extension'),
            ],
            fontsize=9
        )

        # Highlight best
        best_idx = ppls.index(min(ppls))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)

        plt.tight_layout()
        plot_path = f"results/ema_ablation_{args.dataset}_ctx{args.context_len}_cache{args.cache_size}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {plot_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available — skipping plot generation.")


def plot_only():
    """Re-generate plot from saved results."""
    import glob
    files = glob.glob("results/ema_ablation_*.json")
    if not files:
        print("No saved ablation results found. Run the ablation first.")
        return
    for f in files:
        with open(f) as fp:
            results = json.load(fp)
        parts = os.path.basename(f).replace(".json","").split("_")
        class FakeArgs:
            dataset = parts[2]
            context_len = int(parts[3].replace("ctx",""))
            cache_size = int(parts[4].replace("cache",""))
        generate_plot(results, FakeArgs())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",     type=str,   default="pg19")
    parser.add_argument("--context_len", type=int,   default=4096)
    parser.add_argument("--cache_size",  type=int,   default=256)
    parser.add_argument("--lambdas",     type=float, nargs="+",
                        default=[1.0, 0.99, 0.95, 0.9, 0.8])
    parser.add_argument("--n_samples",   type=int,   default=5)
    parser.add_argument("--plot_only",   action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_only()
    else:
        run_ablation(args)
