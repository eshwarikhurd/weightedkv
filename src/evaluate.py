"""
evaluate.py
-----------
Sliding window perplexity evaluation for KV cache methods.

Reproduces the evaluation protocol from WeightedKV (Table I):
  - Sliding window with stride = window_size // 2
  - Context lengths: 4096, 8192, 16384
  - Datasets: PG19, OpenWebText2, ProofPile, ArXiv
"""

import torch
import numpy as np
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def compute_sliding_window_perplexity(
    model,
    tokenizer,
    text: str,
    window_size: int = 4096,
    stride: Optional[int] = None,
    device: str = "cuda",
    reset_fn=None,
) -> float:
    """
    Compute sliding window perplexity over a long text.

    Args:
        model:       HuggingFace causal LM (patched or unpatched)
        tokenizer:   Corresponding tokenizer
        text:        Input text to evaluate
        window_size: Context window length in tokens
        stride:      Step size (default: window_size // 2)
        device:      Compute device
        reset_fn:    Optional callable to reset KV cache between windows
                     (pass reset_caches from model_patch.py)

    Returns:
        Perplexity score (lower is better)
    """
    if stride is None:
        stride = window_size // 2

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.shape[1]

    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + window_size, seq_len)
        target_len = end - prev_end

        window_ids = input_ids[:, begin:end]

        # Build labels: mask out the context (only predict target_len tokens)
        labels = window_ids.clone()
        labels[:, :-target_len] = -100

        if reset_fn is not None:
            reset_fn(model)

        with torch.no_grad():
            outputs = model(window_ids, labels=labels)
            nll = outputs.loss

        nlls.append(nll.item() * target_len)
        prev_end = end

        if end == seq_len:
            break

    avg_nll = sum(nlls) / sum(
        min(stride, seq_len - i * stride)
        for i in range(len(nlls))
    )
    return float(np.exp(avg_nll))


def load_pg19_samples(n: int = 5, split: str = "test"):
    """Load n samples from PG19 test set."""
    dataset = load_dataset("pg19", split=split, streaming=True, trust_remote_code=True)
    samples = []
    for _, item in zip(range(n), dataset):
        samples.append(item["text"])
    return samples


def load_openwebtext2_samples(n: int = 100, split: str = "test"):
    """Load n samples from OpenWebText2."""
    dataset = load_dataset(
        "the_pile", data_files="openwebtext2*",
        split=split, streaming=True, trust_remote_code=True
    )
    return [item["text"] for _, item in zip(range(n), dataset)]


def load_proofpile_samples(n: int = 10, split: str = "test"):
    """Load n samples from ProofPile."""
    dataset = load_dataset(
        "EleutherAI/proof-pile", split=split, streaming=True, trust_remote_code=True
    )
    return [item["text"] for _, item in zip(range(n), dataset)]


def evaluate_method(
    model,
    tokenizer,
    texts: list,
    window_size: int = 4096,
    device: str = "cuda",
    reset_fn=None,
    verbose: bool = True,
) -> float:
    """
    Evaluate perplexity over a list of texts and return the average.

    Args:
        model, tokenizer: HuggingFace model + tokenizer
        texts:            List of strings to evaluate
        window_size:      Sliding window context length
        device:           Compute device
        reset_fn:         Cache reset function (for patched models)
        verbose:          Print per-sample scores

    Returns:
        Average perplexity across all texts
    """
    scores = []
    for i, text in enumerate(texts):
        ppl = compute_sliding_window_perplexity(
            model, tokenizer, text,
            window_size=window_size,
            device=device,
            reset_fn=reset_fn,
        )
        scores.append(ppl)
        if verbose:
            print(f"  Sample {i+1}/{len(texts)} | PPL: {ppl:.2f}")

    avg = float(np.mean(scores))
    if verbose:
        print(f"  Average PPL: {avg:.2f}")
    return avg
