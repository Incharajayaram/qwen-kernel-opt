import torch
import triton
import triton.language as tl


@triton.jit
def fused_swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # this fuses `silu(gate) * up` into one kernel instead of three separate
    # elementwise ops, same block and mask pattern as the vector add example
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    # silu(x) is x times sigmoid(x), computed in float32 for stability
    silu = gate / (1.0 + tl.exp(-gate))
    silu = silu.to(tl.float16)

    out = silu * up
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_swiglu_triton(gate, up):
    orig_shape = gate.shape
    gate_flat = gate.reshape(-1)
    up_flat = up.reshape(-1)
    n_elements = gate_flat.numel()
    out = torch.empty_like(gate_flat)

    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    fused_swiglu_kernel[grid](gate_flat, up_flat, out, n_elements, BLOCK_SIZE=block_size)
    return out.reshape(orig_shape)


def reference_swiglu(gate, up):
    # matches `self.act_fn(self.gate_proj(x)) * self.up_proj(x)` from `Qwen2MLP`
    return torch.nn.functional.silu(gate) * up


if __name__ == "__main__":
    torch.manual_seed(0)
    intermediate_size = 4864

    gate = torch.randn(4, 32, intermediate_size, device="cuda", dtype=torch.float16)
    up = torch.randn(4, 32, intermediate_size, device="cuda", dtype=torch.float16)

    out_ref = reference_swiglu(gate, up)
    out_triton = fused_swiglu_triton(gate, up)

    max_diff = (out_ref - out_triton).abs().max().item()
    match = torch.allclose(out_ref, out_triton, atol=1e-2, rtol=1e-2)
    print(f"max difference: {max_diff}")
    print(f"allclose match: {match}")

    import triton.testing as tt
    ms_ref = tt.do_bench(lambda: reference_swiglu(gate, up))
    ms_triton = tt.do_bench(lambda: fused_swiglu_triton(gate, up))
    print(f"reference: {ms_ref:.4f} ms, triton: {ms_triton:.4f} ms, speedup: {ms_ref / ms_triton:.2f}x")
