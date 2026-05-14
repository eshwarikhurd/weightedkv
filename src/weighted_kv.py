"""
weighted_kv.py
--------------
Core WeightedKVCache implementation.

Reproduces Algorithm 1 from:
  "WeightedKV: Attention Scores Weighted Key-Value Cache Merging
   for Large Language Models" — Yuan et al., ICASSP 2025

Extension: EMA (exponential moving average) decay scoring via `lam` parameter.
  lam=1.0  → original uniform average (default, matches paper exactly)
  lam<1.0  → recency-weighted average (our extension)
"""

import torch


class WeightedKVCache:
    """
    Drop-in KV cache manager for autoregressive LLM generation.

    Usage:
        cache = WeightedKVCache(max_size=256, lam=1.0)
        for each generation step:
            K, V = cache.update(new_k, new_v, attn_scores)
            # use K, V for attention computation

    Args:
        max_size (int): Maximum number of KV pairs to retain (budget m).
        lam (float): EMA decay factor. 1.0 = uniform average (paper default).
                     Values in (0,1) give more weight to recent attention scores.
    """

    def __init__(self, max_size: int, lam: float = 1.0):
        assert 0 < lam <= 1.0, "lam must be in (0, 1]"
        self.max_size = max_size
        self.lam = lam

        self.K = None         # [num_heads, seq_len, head_dim]
        self.V = None         # [num_heads, seq_len, head_dim]
        self.avg_attn = None  # [num_heads, seq_len]
        self.counts = None    # [num_heads, seq_len]  — track number of updates

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        attn_scores: torch.Tensor,
    ):
        """
        Append new token KV, update scores, compress if over budget.

        Args:
            new_k:       [num_heads, 1, head_dim]
            new_v:       [num_heads, 1, head_dim]
            attn_scores: [num_heads, current_seq_len]  (after softmax)

        Returns:
            K: [num_heads, compressed_len, head_dim]
            V: [num_heads, compressed_len, head_dim]
        """
        if self.K is None:
            # First token — initialize cache
            self.K = new_k
            self.V = new_v
            self.avg_attn = attn_scores
            self.counts = torch.ones_like(attn_scores)
            return self.K, self.V

        # Append new token
        self.K = torch.cat([self.K, new_k], dim=1)
        self.V = torch.cat([self.V, new_v], dim=1)

        # Update running average attention scores
        # lam=1.0: standard cumulative average  a = (a*n + new) / (n+1)
        # lam<1.0: EMA-weighted             a = (lam*a*n + new) / (lam*n + 1)
        self.avg_attn = (
            self.lam * self.avg_attn * self.counts + attn_scores
        ) / (self.lam * self.counts + 1)
        self.counts += 1

        # Compress if over budget
        if self.K.shape[1] > self.max_size:
            self._merge_weakest()

        return self.K, self.V

    def _merge_weakest(self):
        """
        Find the token with lowest average attention, discard its key,
        and merge its value into the adjacent right neighbor via
        convex combination weighted by their relative attention scores.

        Implements lines 7-11 of Algorithm 1 in the paper.
        """
        # Average across heads for a single global decision
        mean_attn = self.avg_attn.mean(dim=0)  # [seq_len]

        # Find weakest token, ensure it has a right neighbor
        j = mean_attn.argmin().item()
        j = min(j, self.K.shape[1] - 2)

        # Weights for convex combination: shape [num_heads, 1, 1]
        w_j  = self.avg_attn[:, j    ].unsqueeze(-1).unsqueeze(-1)
        w_j1 = self.avg_attn[:, j + 1].unsqueeze(-1).unsqueeze(-1)
        denom = w_j + w_j1 + 1e-8  # epsilon prevents division by zero

        # Merge values (Eq. 3 from paper)
        merged_v = (w_j * self.V[:, j:j+1] + w_j1 * self.V[:, j+1:j+2]) / denom

        # Remove token j from all caches: value j replaced by merged, key j deleted
        self.V = torch.cat([self.V[:, :j], merged_v, self.V[:, j+2:]], dim=1)
        self.K = torch.cat([self.K[:, :j],            self.K[:, j+1:]], dim=1)
        self.avg_attn = torch.cat(
            [self.avg_attn[:, :j], self.avg_attn[:, j+1:]], dim=1
        )
        self.counts = torch.cat(
            [self.counts[:, :j], self.counts[:, j+1:]], dim=1
        )

    def reset(self):
        """Clear the cache (call between sequences)."""
        self.K = None
        self.V = None
        self.avg_attn = None
        self.counts = None

    @property
    def current_size(self) -> int:
        return 0 if self.K is None else self.K.shape[1]
