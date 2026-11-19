# optimizing qwen2.5 0.5b decode

## goal

take an open source text model and optimize its decode throughput by at
least 10 percent with no quality loss. model chosen:
`Qwen/Qwen2.5-0.5B-Instruct`. hardware: a single nvidia gtx 1650, 4gb vram.

## approach

the model was profiled first with `torch.profiler` during decode. the
breakdown showed that matrix multiplication kernels (`aten::mm`, the
cutlass gemm kernels) account for roughly 86 percent of gpu time, and each
individual gemm takes 260 to 500 microseconds, meaning they are compute
bound, not waiting on launch overhead. the elementwise ops inside rms norm
(`pow`, `mean`, `add`, `rsqrt`, `mul`) are tiny on the gpu but each one is a
separate call through pytorch's cpu side dispatcher, and for a model this
small, decode is sensitive to the sheer number of these calls per step.

this profiling run suggested two different kinds of fix: reduce the number
of small kernel launches per layer, and separately, reduce the launch and
dispatch overhead across the entire decode step, not just the small ops.
both were tried.

### fused rms norm

`Qwen2RMSNorm.forward` runs as five or six separate pytorch ops in eager
mode. a single triton kernel does the same computation, same order of
operations, in one launch, one read of the row and one write of the
result. this replaces the first norm in every decoder layer and the final
model norm.

### fused residual add plus rms norm

every decoder layer does a plain residual add immediately followed by a
second rms norm call. these were fused into one kernel that computes the
sum, writes it out as the new residual stream value, and normalizes it in
the same pass, instead of one kernel for the add and a separate one for the
norm.

patched into the real model together, see `kernel_fusion_results.json` for
the exact measured numbers.

### two kernels that did not help, and why that matters

a kernel fusing the mlp activation (`silu(gate) * up`, in
`fused_swiglu_kernel.py`) was written and checked for correctness, it
matched the reference exactly. patching it in made decode slightly slower,
not faster. `torch.nn.functional.silu` is already a single fused cuda call
in pytorch, so the triton version only removed one multiply, and triton's
own per launch overhead for a new custom kernel outweighed that saving.

a fused qkv projection (`fused_qkv.py`) concatenates the three separate
projection weights into one matrix so one matmul replaces three. this is
mathematically exact, splitting the fused output back apart gives identical
numbers to calling the three projections separately, and it was verified
correct. measured end to end it added a change on the order of noise, not
a real improvement. the profiling explains why, the gemms here are already
compute bound at 260 to 500 microseconds each, so cutting the gemm count by
two per layer saves tens of microseconds of launch overhead against
hundreds of microseconds of actual compute, not enough to show up.

both are kept in the repo and reported here instead of being left out,
since knowing when a fusion does not pay off is as much a part of this
exercise as knowing when it does.

### the technique that actually crossed the target

the two kernel fusions above only address a small slice of total decode
time, since most of that time is gemms, which they do not touch. the
bigger lever is removing python and kernel launch overhead across the
entire decode step, which is what a static kv cache plus
`torch.compile(mode="reduce-overhead")` does, using cuda graphs to replay
the whole step with almost no cpu side overhead between tokens.

getting this working required a custom decode loop instead of relying on
`model.generate()`, since `generate()` does not call
`torch.compiler.cudagraph_mark_step_begin()` before each step on its own.
without that call, the first attempt hit a cuda graph aliasing error
(`accessing tensor output of cudagraphs that has been overwritten by a
subsequent run`), caused by the cache implementation mutating tensors in
place across graph replays. calling
`torch.compiler.cudagraph_mark_step_begin()` before every compiled forward
call in the custom loop fixed it. batch size was kept at 1 for this part,
since a padded batch has an attention mask shape that grows every step,
which would force a recompile almost every token and defeat the point of
the cuda graph.

this produced by far the largest speedup measured in this project, see
`cudagraph_benchmark.py` and the results below.

## correctness methodology

two layers of checking, not one:

1. every kernel is checked in isolation against the exact reference
computation, same shapes and dtype as the real model's hidden size, with
`torch.allclose` (`atol=1e-2, rtol=1e-2`) on fp16 tensors. this lives in
each kernel file's own `if __name__ == "__main__"` block.
2. the full patched model is checked end to end with perplexity on
`wikitext 2`, not just the kernel in isolation. this catches the case where
a kernel is correct on random tensors but something about how it is wired
into the model breaks real generation.

for the triton kernels, both are exact fusions of the same math the
reference performs, so perplexity should barely move, and that is what was
measured.

for the compiled plus cuda graph path, generated text differs from eager
decoding after the first sentence or so, `torch.equal` on the generated
token ids returns false. this was checked rather than assumed to be either
fine or a bug. comparing logits on the very first decode step before any
difference has a chance to compound showed a maximum difference of about
0.05 and the same top token as eager. running the full `wikitext 2`
perplexity check through the compiled path gave 26.98 against a baseline
eager perplexity of 26.96, a negligible change. together these show the
divergence is floating point drift from a different reduction order under
compilation, compounding over many greedy decoding steps, not a quality
regression.

## results

- triton kernel fusion (rms norm plus fused add rms norm), batch of 4
  prompts, 128 new tokens each: baseline 61.13 tokens per second, patched
  62.45 tokens per second, a speedup of 1.02x, about 2 percent, perplexity
  moved from 26.96 to 26.97. see `kernel_fusion_results.json` for the exact
  numbers
- compiled plus static cache plus cuda graphs, single prompt, 128 new
  tokens: eager 18.13 tokens per second, compiled 26.68 tokens per second,
  a speedup of 1.47x, 47 percent
- perplexity on `wikitext 2`, compiled path 26.98 against baseline eager
  26.96

the triton kernel fusion alone does not reach the 10 percent target, the
two ops it touches are a small slice of total decode time on this
hardware, since matrix multiplication dominates. the compiled plus cuda
graph result clears the target by a wide margin on its own. they were
measured separately rather than combined into one patched model, since
layering hand written triton kernels inside a `torch.compile` capture
risked introducing new graph breaks, and the cuda graph result alone was
already enough.

## limitations and what would come next

the cuda graph result is measured at batch size 1 because cuda graphs
require a fixed shape across replays, while the triton kernel result is
measured at batch size 4. getting the cuda graph approach working at a
larger batch size would need a fixed length attention mask, for example by
always padding to the maximum sequence length the cache was built for,
instead of growing it one token at a time. combining the triton kernels
with the compiled path, so the normalization fusions run inside the
captured graph, is the other natural next step, left out here given the
time available and the risk of destabilizing a working result.
