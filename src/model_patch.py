"""
model_patch.py
--------------
Hooks WeightedKVCache into a HuggingFace Llama-2 model by
monkey-patching the attention forward pass.

No model weights are modified. The patch is applied at runtime
and can be reversed by calling unpatch_model().

Usage:
    from model_patch import patch_model, unpatch_model

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    patch_model(model, max_size=256, lam=1.0)

    # ... run generation as normal ...

    unpatch_model(model)  # restore original attention
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from src.weighted_kv import WeightedKVCache


# Store original forward functions so we can restore them
_original_forwards = {}


def _make_patched_forward(layer_idx: int, cache: WeightedKVCache):
    """
    Returns a patched attention forward function that intercepts
    key/value computation and routes through WeightedKVCache.
    """

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        # Standard QKV projection
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, heads, seq, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings if present
        if hasattr(self, 'rotary_emb'):
            cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1)
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # During generation (q_len == 1), apply WeightedKV compression
        if q_len == 1 and bsz == 1:
            # Compute raw attention scores for the new token
            # key_states: [1, heads, 1, head_dim]; cache.K: [heads, cached_len, head_dim]
            new_k = key_states[0]    # [heads, 1, head_dim]
            new_v = value_states[0]  # [heads, 1, head_dim]

            if cache.K is not None:
                # Full key set for score computation: existing cache + new key
                all_keys = torch.cat([cache.K, new_k], dim=1)  # [heads, len+1, head_dim]
                scale = self.head_dim ** -0.5
                # query: [heads, 1, head_dim]; keys: [heads, len+1, head_dim]
                scores = torch.bmm(query_states[0], all_keys.transpose(1, 2)) * scale
                attn_weights = F.softmax(scores, dim=-1)  # [heads, 1, len+1]
                attn_scores_1d = attn_weights[:, 0, :]     # [heads, len+1]
            else:
                attn_scores_1d = torch.ones(
                    new_k.shape[0], 1, device=new_k.device
                )

            # Update cache and get compressed K, V
            K_compressed, V_compressed = cache.update(new_k, new_v, attn_scores_1d)

            # Reformat for attention computation
            K_for_attn = K_compressed.unsqueeze(0)  # [1, heads, len, head_dim]
            V_for_attn = V_compressed.unsqueeze(0)

            # Compute attention output with compressed cache
            scale = self.head_dim ** -0.5
            attn_w = torch.bmm(
                query_states[0],
                K_for_attn[0].transpose(1, 2)
            ) * scale
            attn_w = F.softmax(attn_w, dim=-1)
            attn_output = torch.bmm(attn_w, V_for_attn[0])  # [heads, 1, head_dim]
            attn_output = attn_output.transpose(0, 1).contiguous().view(1, 1, -1)

        else:
            # Prefill phase: standard full attention (no compression during prompt)
            scale = self.head_dim ** -0.5
            attn_w = torch.matmul(query_states, key_states.transpose(2, 3)) * scale
            if attention_mask is not None:
                attn_w = attn_w + attention_mask
            attn_w = F.softmax(attn_w, dim=-1)
            attn_output = torch.matmul(attn_w, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return patched_forward


def patch_model(model, max_size: int = 256, lam: float = 1.0):
    """
    Patch all attention layers in a Llama-2 model to use WeightedKVCache.

    Args:
        model:    HuggingFace Llama-2 model (AutoModelForCausalLM)
        max_size: KV cache budget per layer per head
        lam:      EMA decay factor (1.0 = paper default)
    """
    global _original_forwards

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        cache = WeightedKVCache(max_size=max_size, lam=lam)

        # Store original forward
        _original_forwards[layer_idx] = attn.forward

        # Bind patched forward to the attention module
        import types
        attn._wkv_cache = cache
        attn.forward = types.MethodType(
            _make_patched_forward(layer_idx, cache), attn
        )

    print(f"Patched {len(model.model.layers)} layers | "
          f"max_size={max_size} | lam={lam}")


def unpatch_model(model):
    """Restore original attention forward functions."""
    global _original_forwards
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in _original_forwards:
            layer.self_attn.forward = _original_forwards[layer_idx]
    _original_forwards = {}
    print("Model unpatched — original attention restored.")


def reset_caches(model):
    """Clear all WeightedKV caches between sequences."""
    for layer in model.model.layers:
        if hasattr(layer.self_attn, '_wkv_cache'):
            layer.self_attn._wkv_cache.reset()
