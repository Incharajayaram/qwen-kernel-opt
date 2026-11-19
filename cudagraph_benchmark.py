import torch
from common import load_model
from cudagraph_decode import run_eager, build_compiled_runner, PROMPT, MAX_NEW_TOKENS


def time_eager(model, tokenizer, n_warmup=2, n_runs=5):
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    for _ in range(n_warmup):
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 1000.0)
    return times


def time_compiled(model, tokenizer, n_warmup=2, n_runs=5):
    run = build_compiled_runner(model, tokenizer)
    for _ in range(n_warmup):
        run()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        run()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / 1000.0)
    return times


if __name__ == "__main__":
    model, tokenizer = load_model()

    print("timing eager, batch 1")
    eager_times = time_eager(model, tokenizer)
    eager_tps = [MAX_NEW_TOKENS / t for t in eager_times]
    eager_mean = sum(eager_tps) / len(eager_tps)
    print(f"eager tokens per sec: {eager_mean:.2f}, runs: {[round(x, 2) for x in eager_tps]}")

    print("\ntiming compiled plus static cache, batch 1")
    compiled_times = time_compiled(model, tokenizer)
    compiled_tps = [MAX_NEW_TOKENS / t for t in compiled_times]
    compiled_mean = sum(compiled_tps) / len(compiled_tps)
    print(f"compiled tokens per sec: {compiled_mean:.2f}, runs: {[round(x, 2) for x in compiled_tps]}")

    speedup = compiled_mean / eager_mean
    print(f"\nspeedup: {speedup:.3f}x ({(speedup - 1) * 100:.1f} percent)")
