import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    # one program handles one row of the input, this matches the row wise
    # nature of rms norm, every row is normalized independently of every other row
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    row_ptr = x_ptr + row * n_cols
    x = tl.load(row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # root mean square over this row, computed in float32 for stability
    # this mirrors what the reference `Qwen2RMSNorm.forward` does
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    rms = tl.sqrt(mean_sq + eps)
    x_norm = x / rms

    # the reference casts back to the input dtype before multiplying by the
    # weight, so we do the same here to stay numerically identical
    x_norm = x_norm.to(tl.float16)

    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    y = x_norm * weight

    out_row_ptr = out_ptr + row * n_cols
    tl.store(out_row_ptr + cols, y, mask=mask)


def rmsnorm_triton(x, weight, eps):
    # x is reshaped to two dimensions before calling this, one row per token
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    n_rows, n_cols = x2d.shape
    out = torch.empty_like(x2d)
    block_size = triton.next_power_of_2(n_cols)
    rmsnorm_kernel[(n_rows,)](x2d, weight, out, n_cols, eps, BLOCK_SIZE=block_size)
    return out.reshape(orig_shape)


def rmsnorm_reference(x, weight, eps):
    # this is a direct copy of `Qwen2RMSNorm.forward`, used to check correctness
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    hidden_size = 896
    eps = 1e-6

    x = torch.randn(4, 32, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)

    out_ref = rmsnorm_reference(x, weight, eps)
    out_triton = rmsnorm_triton(x, weight, eps)

    max_diff = (out_ref - out_triton).abs().max().item()
    match = torch.allclose(out_ref, out_triton, atol=1e-2, rtol=1e-2)
    print(f"max difference: {max_diff}")
    print(f"allclose match: {match}")

    import triton.testing as tt
    ms_ref = tt.do_bench(lambda: rmsnorm_reference(x, weight, eps))
    ms_triton = tt.do_bench(lambda: rmsnorm_triton(x, weight, eps))
    print(f"reference: {ms_ref:.4f} ms, triton: {ms_triton:.4f} ms, speedup: {ms_ref / ms_triton:.2f}x")
