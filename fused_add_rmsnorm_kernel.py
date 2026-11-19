import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_rmsnorm_kernel(x_ptr, residual_ptr, weight_ptr, out_ptr, residual_out_ptr,
                              n_cols, eps, BLOCK_SIZE: tl.constexpr):
    # this fuses the residual add that normally happens before a norm call
    # with the norm itself, so there is one kernel instead of two
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + row * n_cols + cols, mask=mask, other=0.0)

    # the sum becomes the new residual stream value, carried forward exactly
    # as `residual + hidden_states` does in the reference decoder layer
    summed = x + residual
    tl.store(residual_out_ptr + row * n_cols + cols, summed, mask=mask)

    summed_f32 = summed.to(tl.float32)
    mean_sq = tl.sum(summed_f32 * summed_f32, axis=0) / n_cols
    rms = tl.sqrt(mean_sq + eps)
    normed = summed_f32 / rms
    normed = normed.to(tl.float16)

    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    y = normed * weight

    tl.store(out_ptr + row * n_cols + cols, y, mask=mask)


def fused_add_rmsnorm_triton(x, residual, weight, eps):
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    residual2d = residual.reshape(-1, orig_shape[-1])
    n_rows, n_cols = x2d.shape

    out = torch.empty_like(x2d)
    residual_out = torch.empty_like(x2d)
    block_size = triton.next_power_of_2(n_cols)

    fused_add_rmsnorm_kernel[(n_rows,)](
        x2d, residual2d, weight, out, residual_out, n_cols, eps, BLOCK_SIZE=block_size
    )
    return out.reshape(orig_shape), residual_out.reshape(orig_shape)


def reference_add_rmsnorm(x, residual, weight, eps):
    # same two steps as the decoder layer does them separately, used to
    # check that fusing them did not change the result
    summed = residual + x
    input_dtype = summed.dtype
    summed_f32 = summed.to(torch.float32)
    variance = summed_f32.pow(2).mean(-1, keepdim=True)
    normed = summed_f32 * torch.rsqrt(variance + eps)
    out = weight * normed.to(input_dtype)
    return out, summed


if __name__ == "__main__":
    torch.manual_seed(0)
    hidden_size = 896
    eps = 1e-6

    x = torch.randn(4, 32, hidden_size, device="cuda", dtype=torch.float16)
    residual = torch.randn(4, 32, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float16)

    out_ref, residual_ref = reference_add_rmsnorm(x, residual, weight, eps)
    out_triton, residual_triton = fused_add_rmsnorm_triton(x, residual, weight, eps)

    out_match = torch.allclose(out_ref, out_triton, atol=1e-2, rtol=1e-2)
    residual_match = torch.allclose(residual_ref, residual_triton, atol=1e-3, rtol=1e-3)
    print(f"normed output match: {out_match}")
    print(f"residual output match: {residual_match}")

    import triton.testing as tt
    ms_ref = tt.do_bench(lambda: reference_add_rmsnorm(x, residual, weight, eps))
    ms_triton = tt.do_bench(lambda: fused_add_rmsnorm_triton(x, residual, weight, eps))
    print(f"reference: {ms_ref:.4f} ms, triton: {ms_triton:.4f} ms, speedup: {ms_ref / ms_triton:.2f}x")
