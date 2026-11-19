import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda"
DTYPE = torch.float16


def load_model():
    # loads the model and tokenizer once, used by every script in this project
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE)
    model.eval()
    return model, tokenizer


PROMPTS = [
    "Explain what a compiler does in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main differences between CPUs and GPUs?",
    "Describe how photosynthesis works.",
]


@torch.no_grad()
def benchmark_decode(model, tokenizer, max_new_tokens=128, n_warmup=2, n_runs=5):
    # measures decode throughput in tokens per second
    # decode is the repeated one token at a time generation step, this is where
    # kernel launch overhead dominates for small models like this one
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True).to(DEVICE)

    for _ in range(n_warmup):
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)

    n_prompts = inputs["input_ids"].shape[0]
    total_new_tokens = n_prompts * max_new_tokens
    tokens_per_sec = [total_new_tokens / t for t in times]

    mean_t = sum(times) / len(times)
    mean_tps = sum(tokens_per_sec) / len(tokens_per_sec)
    std_tps = (sum((x - mean_tps) ** 2 for x in tokens_per_sec) / len(tokens_per_sec)) ** 0.5

    return {
        "mean_seconds": mean_t,
        "mean_tokens_per_sec": mean_tps,
        "std_tokens_per_sec": std_tps,
        "raw_times": times,
        "sample_output": tokenizer.decode(out[0], skip_special_tokens=True),
    }


@torch.no_grad()
def compute_perplexity(model, tokenizer, n_sequences=20, seq_len=256):
    # perplexity on a slice of wikitext 2, this is the quality anchor
    # if a kernel change is numerically wrong this number will move, if it is
    # exact or near exact this number should barely change
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)

    total_nll = 0.0
    total_tokens = 0
    for i in range(n_sequences):
        start = i * seq_len
        end = start + seq_len
        if end >= input_ids.shape[1]:
            break
        chunk = input_ids[:, start:end]
        out = model(chunk, labels=chunk)
        n_tok = chunk.shape[1] - 1
        total_nll += out.loss.item() * n_tok
        total_tokens += n_tok

    avg_nll = total_nll / total_tokens
    ppl = torch.exp(torch.tensor(avg_nll)).item()
    return ppl
