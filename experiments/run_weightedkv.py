"""
run_weightedkv.py
-----------------
Reproduce Table I from the WeightedKV paper.

Evaluates sliding window perplexity on PG19, OpenWebText2,
ProofPile, and ArXiv at context lengths 4K/8K/16K.

Example usage:
    # PG19 at 4K context, cache size 1024 (paper default)
    python experiments/run_weightedkv.py \
        --dataset pg19 \
        --context_len 4096 \
        --cache_size 1024 \
        --lam 1.0

    # Cache size 256 (tight budget)
    python experiments/run_weightedkv.py \
        --dataset pg19 \
        --context_len 4096 \
        --cache_size 256

    # Run all datasets and context lengths (full Table I reproduction)
    python experiments/run_weightedkv.py --full_table
"""

import argparse
import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_patch import patch_model, unpatch_model, reset_caches
from src.evaluate import evaluate_method, load_pg19_samples

MODEL_ID = "meta-llama/Llama-2-7b-hf"

DATASETS = {
    "pg19":          {"n_samples": 100,  "loader": "load_pg19_samples"},
    "openwebtext2":  {"n_samples": 100,  "loader": "load_openwebtext2_samples"},
    "proofpile":     {"n_samples": 10,   "loader": "load_proofpile_samples"},
}

CONTEXT_LENGTHS = [4096, 8192, 16384]
CACHE_SIZES     = [1024, 256]


def run_single(args):
    print(f"\n{'='*60}")
    print(f"WeightedKV Evaluation")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Context len:  {args.context_len}")
    print(f"  Cache size:   {args.cache_size}")
    print(f"  Lambda (EMA): {args.lam}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Apply WeightedKV patch
    patch_model(model, max_size=args.cache_size, lam=args.lam)

    # Load data
    print(f"Loading {args.dataset} samples...")
    if args.dataset == "pg19":
        from src.evaluate import load_pg19_samples
        texts = load_pg19_samples(n=args.n_samples)
    elif args.dataset == "openwebtext2":
        from src.evaluate import load_openwebtext2_samples
        texts = load_openwebtext2_samples(n=args.n_samples)
    elif args.dataset == "proofpile":
        from src.evaluate import load_proofpile_samples
        texts = load_proofpile_samples(n=args.n_samples)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Evaluate
    print(f"\nEvaluating {len(texts)} samples...")
    avg_ppl = evaluate_method(
        model, tokenizer, texts,
        window_size=args.context_len,
        device="cuda" if torch.cuda.is_available() else "cpu",
        reset_fn=reset_caches,
        verbose=True,
    )

    # Save result
    result = {
        "method": "WeightedKV",
        "dataset": args.dataset,
        "context_len": args.context_len,
        "cache_size": args.cache_size,
        "lam": args.lam,
        "n_samples": len(texts),
        "perplexity": avg_ppl,
    }

    os.makedirs("results", exist_ok=True)
    fname = f"results/weightedkv_{args.dataset}_ctx{args.context_len}_cache{args.cache_size}_lam{args.lam}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResult saved to {fname}")
    print(f"Final PPL: {avg_ppl:.4f}")
    return avg_ppl


def run_full_table(args):
    """Reproduce the complete Table I across all datasets and context lengths."""
    results = {}
    for cache_size in CACHE_SIZES:
        for dataset in ["pg19", "openwebtext2", "proofpile"]:
            for ctx in CONTEXT_LENGTHS:
                args.dataset = dataset
                args.context_len = ctx
                args.cache_size = cache_size
                key = f"{dataset}_ctx{ctx}_cache{cache_size}"
                results[key] = run_single(args)

    print("\n" + "="*60)
    print("FULL TABLE I RESULTS")
    print("="*60)
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WeightedKV Evaluation")
    parser.add_argument("--dataset",     type=str,   default="pg19",
                        choices=["pg19", "openwebtext2", "proofpile"])
    parser.add_argument("--context_len", type=int,   default=4096,
                        choices=[4096, 8192, 16384])
    parser.add_argument("--cache_size",  type=int,   default=1024)
    parser.add_argument("--lam",         type=float, default=1.0,
                        help="EMA decay (1.0=uniform average, paper default)")
    parser.add_argument("--n_samples",   type=int,   default=5,
                        help="Number of samples to evaluate (use 100 for full repro)")
    parser.add_argument("--full_table",  action="store_true",
                        help="Run full Table I reproduction")
    args = parser.parse_args()

    if args.full_table:
        run_full_table(args)
    else:
        run_single(args)
