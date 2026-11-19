# qwen decode optimization

this project optimizes the decode path of `Qwen/Qwen2.5-0.5B-Instruct` on a single
nvidia gtx 1650 (4gb), without changing model quality.

## what was done

the model was profiled first with `torch.profiler` during decode. matrix
multiplication (`aten::mm`, cutlass gemm kernels) accounts for roughly 86
percent of gpu time and is compute bound on this hardware, individual gemms
take 260 to 500 microseconds each. the rest of the time goes to many small
elementwise ops (the pow, mean, add, rsqrt, mul inside every rms norm call),
each cheap on the gpu but each a separate call through pytorch's cpu side
dispatcher. for a model this small, decode throughput is sensitive to the
number of these small calls per step, not to raw flops.

two triton kernels target that:

- `rmsnorm_kernel.py`, a fused rms norm kernel, replacing the five or six
  separate ops `Qwen2RMSNorm.forward` normally runs with one kernel launch.
- `fused_add_rmsnorm_kernel.py`, which fuses the residual add that happens
  right before the second norm in every decoder layer with the norm itself,
  one kernel instead of two.

both are patched into the real model in `kernel_fusion_benchmark.py` and
measured together, giving a modest but real speedup. see `fused_qkv.py` and
`fused_swiglu_kernel.py` below for two further ideas that were tried,
verified correct, and did not help, kept because they are part of the story.

the much bigger win came from a different direction. `cudagraph_decode.py`
uses a static kv cache with `torch.compile(mode="reduce-overhead")`, which
captures the whole decode step as a cuda graph and removes python and kernel
launch overhead across the entire step, not just the two ops the triton
kernels target. this is the standard technique production inference engines
use for small models, and it produced by far the largest improvement here.

## correctness

every kernel is checked against the exact reference computation it replaces
with `torch.allclose` (`atol=1e-2, rtol=1e-2`) on fp16 tensors shaped like
the real model, inside each kernel file's own `if __name__ == "__main__"`
block. the two kernels used in the patched model are exact fusions of the
same math, same number of floating point operations in a different order,
not approximations.

quality is also checked end to end with perplexity on `wikitext 2`, since a
kernel could pass an isolated `allclose` check on random tensors and still
break something once wired into the real model. the compiled plus cuda
graph path generates text that differs token for token from eager decoding
after the first sentence or so, this was checked and is expected floating
point drift from a different reduction order under compilation, not a bug,
confirmed by comparing first step logits (same top token, max difference
under 0.06) and by perplexity on the compiled path barely moving from the
baseline.

## results

see `kernel_fusion_results.json` for the triton kernel numbers and
`cudagraph_benchmark.py`'s printed output for the cuda graph numbers.
methodology:

- model: `Qwen/Qwen2.5-0.5B-Instruct`, fp16
- gpu: nvidia geforce gtx 1650, 4gb
- triton kernel result: 4 prompts, 128 new tokens each, greedy decoding, 8
  timed runs after 2 warmup runs, cuda events, baseline and patched measured
  back to back in the same process
- cuda graph result: single prompt, 128 new tokens, greedy decoding, 5 timed
  runs after 2 warmup runs, cuda events. batch size 1 here because cuda
  graphs need a fixed shape across replays, and a growing attention mask
  across a padded batch would force a recompile on almost every step
- quality: perplexity on 20 sequences of 256 tokens from the `wikitext 2`
  test split

## how to reproduce

```
python3 rmsnorm_kernel.py              # standalone correctness and speed check
python3 fused_add_rmsnorm_kernel.py    # standalone correctness and speed check
python3 fused_swiglu_kernel.py         # standalone correctness and speed check, not used in the patched model
python3 fused_qkv.py                   # standalone correctness and speed check, not used in the patched model
python3 kernel_fusion_benchmark.py     # baseline vs patched, decode throughput and perplexity
python3 cudagraph_decode.py            # checks generation, compiled vs eager
python3 cudagraph_quality_check.py     # logit and perplexity comparison for the compiled path
python3 cudagraph_benchmark.py         # baseline vs compiled plus cuda graph, decode throughput
```

## files

- `common.py` model loading and the benchmark and perplexity functions shared by every script
- `rmsnorm_kernel.py` fused rms norm kernel plus its own correctness test
- `fused_add_rmsnorm_kernel.py` fused residual add plus rms norm kernel plus its own correctness test
- `fused_swiglu_kernel.py` fused swiglu activation kernel, correct but not used, kept because the writeup explains why
- `fused_qkv.py` fused qkv projection, correct but not used, kept because the writeup explains why
- `kernel_fusion_benchmark.py` patches the two triton kernels into the real model and measures baseline vs patched
- `cudagraph_decode.py` custom decode loop using a static cache and `torch.compile(mode="reduce-overhead")`
- `cudagraph_quality_check.py` logit and perplexity comparison between eager and the compiled plus cuda graph path
- `cudagraph_benchmark.py` baseline vs compiled plus cuda graph decode throughput
- `baseline_benchmark.py` unmodified model decode throughput and perplexity
