"""
run_baselines.py
----------------
Implements and evaluates all baseline KV cache methods from the paper:
  - FullKV     (no compression, upper bound)
  - StreamingLLM (keep initial + recent tokens)
  - H2O          (heavy-hitter oracle, cumulative attention scoring)
  - TOVA         (keep token with highest attention from previous step)

Usage:
    python experiments/run_baselines.py \
        --method streamingllm \
        --dataset pg19 \
        --context_len 4096 \
        --cache_size 1024
"""

import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
import types
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluate import evaluate_method, load_pg19_samples

MODEL_ID = "meta-llama/Llama-2-7b-hf"


# ─── StreamingLLM ─────────────────────────────────────────────────────────────

class StreamingLLMCache:
    """
    Keeps the first `n_sink` tokens (attention sinks) plus
    the most recent `max_size - n_sink` tokens.
    From: Xiao et al., ICLR 2024.
    """
    def __init__(self, max_size: int, n_sink: int = 4):
        self.max_size = max_size
        self.n_sink = n_sink
        self.K_sink = None   # [heads, n_sink, head_dim]
        self.V_sink = None
        self.K_recent = None # [heads, recent, head_dim]
        self.V_recent = None

    def update(self, new_k, new_v):
        if self.K_sink is None:
            if new_k.shape[1] <= self.n_sink:
                self.K_sink = new_k
                self.V_sink = new_v
            else:
                self.K_sink = new_k[:, :self.n_sink]
                self.V_sink = new_v[:, :self.n_sink]
                self.K_recent = new_k[:, self.n_sink:]
                self.V_recent = new_v[:, self.n_sink:]
        else:
            if self.K_recent is None:
                self.K_recent = new_k
                self.V_recent = new_v
            else:
                self.K_recent = torch.cat([self.K_recent, new_k], dim=1)
                self.V_recent = torch.cat([self.V_recent, new_v], dim=1)

            # Trim recent to budget
            recent_budget = self.max_size - self.n_sink
            if self.K_recent.shape[1] > recent_budget:
                self.K_recent = self.K_recent[:, -recent_budget:]
                self.V_recent = self.V_recent[:, -recent_budget:]

        K = torch.cat([self.K_sink, self.K_recent], dim=1) if self.K_recent is not None else self.K_sink
        V = torch.cat([self.V_sink, self.V_recent], dim=1) if self.V_recent is not None else self.V_sink
        return K, V

    def reset(self):
        self.K_sink = self.V_sink = self.K_recent = self.V_recent = None


# ─── H2O ─────────────────────────────────────────────────────────────────────

class H2OCache:
    """
    Heavy-Hitter Oracle: retains tokens with highest cumulative attention scores
    plus the most recent tokens.
    From: Zhang et al., NeurIPS 2023.
    """
    def __init__(self, max_size: int, n_recent: int = None):
        self.max_size = max_size
        self.n_recent = n_recent or max_size // 2
        self.K = None
        self.V = None
        self.cum_attn = None  # cumulative attention scores

    def update(self, new_k, new_v, attn_scores):
        # new_k: [heads, 1, head_dim], attn_scores: [heads, seq_len]
        if self.K is None:
            self.K = new_k
            self.V = new_v
            self.cum_attn = attn_scores
            return self.K, self.V

        self.K = torch.cat([self.K, new_k], dim=1)
        self.V = torch.cat([self.V, new_v], dim=1)
        self.cum_attn = self.cum_attn + attn_scores

        if self.K.shape[1] > self.max_size:
            self._evict()

        return self.K, self.V

    def _evict(self):
        seq_len = self.K.shape[1]
        n_heavy = self.max_size - self.n_recent

        # Mean cumulative attention across heads
        mean_cum = self.cum_attn.mean(dim=0)  # [seq_len]

        # Always keep the last n_recent tokens
        keep_recent = set(range(seq_len - self.n_recent, seq_len))

        # From the rest, keep top n_heavy by cumulative attention
        candidate_scores = [(i, mean_cum[i].item())
                            for i in range(seq_len - self.n_recent)]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        keep_heavy = {i for i, _ in candidate_scores[:n_heavy]}

        keep_idx = sorted(keep_heavy | keep_recent)
        keep_t = torch.tensor(keep_idx, device=self.K.device)

        self.K = self.K[:, keep_t]
        self.V = self.V[:, keep_t]
        self.cum_attn = self.cum_attn[:, keep_t]

    def reset(self):
        self.K = self.V = self.cum_attn = None


# ─── TOVA ─────────────────────────────────────────────────────────────────────

class TOVACache:
    """
    TOVA: evicts the token with the lowest attention score from the previous step.
    From: Oren et al., 2024.
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.K = None
        self.V = None

    def update(self, new_k, new_v, attn_scores):
        if self.K is None:
            self.K = new_k
            self.V = new_v
            return self.K, self.V

        self.K = torch.cat([self.K, new_k], dim=1)
        self.V = torch.cat([self.V, new_v], dim=1)

        if self.K.shape[1] > self.max_size:
            # Evict token with lowest attention in previous step
            mean_attn = attn_scores.mean(dim=0)
            j = mean_attn.argmin().item()
            j = min(j, self.K.shape[1] - 1)
            keep = list(range(j)) + list(range(j + 1, self.K.shape[1]))
            keep_t = torch.tensor(keep, device=self.K.device)
            self.K = self.K[:, keep_t]
            self.V = self.V[:, keep_t]

        return self.K, self.V

    def reset(self):
        self.K = self.V = None


# ─── Patching helpers ─────────────────────────────────────────────────────────

def patch_streamingllm(model, max_size=1024, n_sink=4):
    for layer in model.model.layers:
        cache = StreamingLLMCache(max_size=max_size, n_sink=n_sink)
        layer.self_attn._baseline_cache = cache
        layer.self_attn._baseline_type = "streamingllm"
    _apply_baseline_patch(model)


def patch_h2o(model, max_size=1024):
    for layer in model.model.layers:
        cache = H2OCache(max_size=max_size)
        layer.self_attn._baseline_cache = cache
        layer.self_attn._baseline_type = "h2o"
    _apply_baseline_patch(model)


def patch_tova(model, max_size=1024):
    for layer in model.model.layers:
        cache = TOVACache(max_size=max_size)
        layer.self_attn._baseline_cache = cache
        layer.self_attn._baseline_type = "tova"
    _apply_baseline_patch(model)


def _apply_baseline_patch(model):
    """Apply a generic baseline patch that routes to the per-layer cache."""
    for layer in model.model.layers:
        attn = layer.self_attn

        def make_forward(cache, btype):
            def forward(self, hidden_states, attention_mask=None,
                        position_ids=None, past_key_value=None,
                        output_attentions=False, use_cache=False, **kwargs):

                bsz, q_len, _ = hidden_states.size()
                query = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key   = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                if hasattr(self, 'rotary_emb'):
                    cos, sin = self.rotary_emb(value, seq_len=position_ids.max().item() + 1)
                    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                    query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

                if q_len == 1 and bsz == 1:
                    new_k = key[0]    # [heads, 1, head_dim]
                    new_v = value[0]

                    if btype == "streamingllm":
                        K_c, V_c = cache.update(new_k, new_v)
                        attn_w = None
                    else:
                        if cache.K is not None:
                            all_k = torch.cat([cache.K, new_k], dim=1)
                        else:
                            all_k = new_k
                        scale = self.head_dim ** -0.5
                        scores = torch.bmm(query[0], all_k.transpose(1, 2)) * scale
                        attn_w = F.softmax(scores, dim=-1)[:, 0, :]
                        K_c, V_c = cache.update(new_k, new_v, attn_w)

                    K_a = K_c.unsqueeze(0)
                    V_a = V_c.unsqueeze(0)
                    scale = self.head_dim ** -0.5
                    aw = torch.bmm(query[0], K_a[0].transpose(1, 2)) * scale
                    aw = F.softmax(aw, dim=-1)
                    out = torch.bmm(aw, V_a[0]).transpose(0, 1).contiguous().view(1, 1, -1)
                else:
                    scale = self.head_dim ** -0.5
                    aw = torch.matmul(query, key.transpose(2, 3)) * scale
                    if attention_mask is not None:
                        aw = aw + attention_mask
                    aw = F.softmax(aw, dim=-1)
                    out = torch.matmul(aw, value).transpose(1, 2).contiguous().view(bsz, q_len, -1)

                return self.o_proj(out), None, past_key_value
            return forward

        attn.forward = types.MethodType(
            make_forward(attn._baseline_cache, attn._baseline_type), attn
        )


def reset_baseline_caches(model):
    for layer in model.model.layers:
        if hasattr(layer.self_attn, '_baseline_cache'):
            layer.self_attn._baseline_cache.reset()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",      type=str, default="streamingllm",
                        choices=["fullkv", "streamingllm", "h2o", "tova"])
    parser.add_argument("--dataset",     type=str, default="pg19")
    parser.add_argument("--context_len", type=int, default=4096)
    parser.add_argument("--cache_size",  type=int, default=1024)
    parser.add_argument("--n_samples",   type=int, default=5)
    args = parser.parse_args()

    print(f"\nBaseline: {args.method.upper()} | "
          f"Dataset: {args.dataset} | "
          f"Context: {args.context_len} | "
          f"Cache: {args.cache_size}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    reset_fn = None
    if args.method == "streamingllm":
        patch_streamingllm(model, max_size=args.cache_size)
        reset_fn = reset_baseline_caches
    elif args.method == "h2o":
        patch_h2o(model, max_size=args.cache_size)
        reset_fn = reset_baseline_caches
    elif args.method == "tova":
        patch_tova(model, max_size=args.cache_size)
        reset_fn = reset_baseline_caches
    # fullkv: no patch needed

    texts = load_pg19_samples(n=args.n_samples)
    avg_ppl = evaluate_method(
        model, tokenizer, texts,
        window_size=args.context_len,
        device="cuda" if torch.cuda.is_available() else "cpu",
        reset_fn=reset_fn,
    )

    result = {
        "method": args.method,
        "dataset": args.dataset,
        "context_len": args.context_len,
        "cache_size": args.cache_size,
        "perplexity": avg_ppl,
    }
    os.makedirs("results", exist_ok=True)
    fname = f"results/{args.method}_{args.dataset}_ctx{args.context_len}_cache{args.cache_size}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {fname}")
    print(f"Final PPL: {avg_ppl:.4f}")
