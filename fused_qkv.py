import torch
import torch.nn.functional as F


def fuse_qkv_weights(attn):
    # concatenates the three separate projection weights into one bigger
    # matrix, so one matmul call replaces three. this is exact, not an
    # approximation, splitting the fused output back apart gives identical
    # numbers to calling `q_proj`, `k_proj`, `v_proj` separately
    q_w, k_w, v_w = attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight
    q_b, k_b, v_b = attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias

    attn.qkv_weight = torch.cat([q_w, k_w, v_w], dim=0)
    attn.qkv_bias = torch.cat([q_b, k_b, v_b], dim=0)
    attn.qkv_split_sizes = (q_w.shape[0], k_w.shape[0], v_w.shape[0])


def fused_qkv_forward(hidden_states, qkv_weight, qkv_bias, split_sizes):
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    return torch.split(qkv, split_sizes, dim=-1)


def reference_qkv_forward(hidden_states, q_proj, k_proj, v_proj):
    return q_proj(hidden_states), k_proj(hidden_states), v_proj(hidden_states)


if __name__ == "__main__":
    import torch.nn as nn

    torch.manual_seed(0)
    hidden_size = 896
    q_out, k_out, v_out = 896, 128, 128

    class FakeAttn:
        pass

    attn = FakeAttn()
    attn.q_proj = nn.Linear(hidden_size, q_out, bias=True).to("cuda").half()
    attn.k_proj = nn.Linear(hidden_size, k_out, bias=True).to("cuda").half()
    attn.v_proj = nn.Linear(hidden_size, v_out, bias=True).to("cuda").half()

    fuse_qkv_weights(attn)

    x = torch.randn(4, 32, hidden_size, device="cuda", dtype=torch.float16)

    q_ref, k_ref, v_ref = reference_qkv_forward(x, attn.q_proj, attn.k_proj, attn.v_proj)
    q_fused, k_fused, v_fused = fused_qkv_forward(x, attn.qkv_weight, attn.qkv_bias, attn.qkv_split_sizes)

    print("q match:", torch.allclose(q_ref, q_fused, atol=1e-2, rtol=1e-2))
    print("k match:", torch.allclose(k_ref, k_fused, atol=1e-2, rtol=1e-2))
    print("v match:", torch.allclose(v_ref, v_fused, atol=1e-2, rtol=1e-2))

    import triton.testing as tt
    ms_ref = tt.do_bench(lambda: reference_qkv_forward(x, attn.q_proj, attn.k_proj, attn.v_proj))
    ms_fused = tt.do_bench(lambda: fused_qkv_forward(x, attn.qkv_weight, attn.qkv_bias, attn.qkv_split_sizes))
    print(f"reference: {ms_ref:.4f} ms, fused: {ms_fused:.4f} ms, speedup: {ms_ref / ms_fused:.2f}x")
