import json
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2DecoderLayer

from common import load_model, benchmark_decode, compute_perplexity
from rmsnorm_kernel import rmsnorm_triton
from fused_add_rmsnorm_kernel import fused_add_rmsnorm_triton


def patched_norm_forward(self, hidden_states):
    return rmsnorm_triton(hidden_states, self.weight, self.variance_epsilon)


def patched_layer_forward(self, hidden_states, attention_mask=None, position_ids=None,
                           past_key_values=None, use_cache=False, position_embeddings=None,
                           **kwargs):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    normed, residual = fused_add_rmsnorm_triton(
        hidden_states, residual, self.post_attention_layernorm.weight,
        self.post_attention_layernorm.variance_epsilon,
    )
    hidden_states = self.mlp(normed)
    hidden_states = residual + hidden_states
    return hidden_states


def patch_model(model):
    # swaps in the triton kernels by rebinding `forward` on the matching
    # modules, the rest of the model's control flow is untouched
    for module in model.modules():
        if isinstance(module, Qwen2DecoderLayer):
            module.forward = patched_layer_forward.__get__(module)
        elif isinstance(module, Qwen2RMSNorm):
            module.forward = patched_norm_forward.__get__(module)


if __name__ == "__main__":
    model, tokenizer = load_model()

    print("baseline, unmodified model")
    base_decode = benchmark_decode(model, tokenizer, n_runs=8)
    base_ppl = compute_perplexity(model, tokenizer)
    print(f"tokens per sec: {base_decode['mean_tokens_per_sec']:.2f} "
          f"(std {base_decode['std_tokens_per_sec']:.2f})")
    print(f"perplexity: {base_ppl:.4f}")

    patch_model(model)

    print("\npatched, fused triton kernels")
    patched_decode = benchmark_decode(model, tokenizer, n_runs=8)
    patched_ppl = compute_perplexity(model, tokenizer)
    print(f"tokens per sec: {patched_decode['mean_tokens_per_sec']:.2f} "
          f"(std {patched_decode['std_tokens_per_sec']:.2f})")
    print(f"perplexity: {patched_ppl:.4f}")

    speedup = patched_decode["mean_tokens_per_sec"] / base_decode["mean_tokens_per_sec"]
    ppl_delta = patched_ppl - base_ppl

    print("\nsummary")
    print(f"baseline tokens per sec: {base_decode['mean_tokens_per_sec']:.2f}")
    print(f"patched tokens per sec: {patched_decode['mean_tokens_per_sec']:.2f}")
    print(f"speedup: {speedup:.3f}x ({(speedup - 1) * 100:.1f} percent)")
    print(f"baseline perplexity: {base_ppl:.4f}")
    print(f"patched perplexity: {patched_ppl:.4f}")
    print(f"perplexity delta: {ppl_delta:.4f}")

    results = {
        "baseline": {
            "tokens_per_sec_mean": base_decode["mean_tokens_per_sec"],
            "tokens_per_sec_std": base_decode["std_tokens_per_sec"],
            "perplexity": base_ppl,
        },
        "patched": {
            "tokens_per_sec_mean": patched_decode["mean_tokens_per_sec"],
            "tokens_per_sec_std": patched_decode["std_tokens_per_sec"],
            "perplexity": patched_ppl,
        },
        "speedup": speedup,
        "perplexity_delta": ppl_delta,
    }
    with open("kernel_fusion_results.json", "w") as f:
        json.dump(results, f, indent=2)
