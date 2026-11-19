import torch
from transformers import StaticCache
from common import load_model


def check_first_step_logit_diff(model, tokenizer):
    # compares the compiled plus static cache forward against plain eager
    # on the exact same input, before any decoding steps compound the
    # difference, to see how large the raw numerical drift from compilation is
    prompt = "Explain what a compiler does in simple terms."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        eager_out = model(input_ids=input_ids, use_cache=False)
    eager_logits = eager_out.logits[:, -1, :].float()

    past_key_values = StaticCache(config=model.config, max_cache_len=seq_len + 8,
                                   device="cuda", dtype=torch.float16)
    cache_position = torch.arange(seq_len, device="cuda")
    compiled_forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    with torch.no_grad():
        compiled_out = compiled_forward(input_ids=input_ids, past_key_values=past_key_values,
                                         use_cache=True, cache_position=cache_position)
    compiled_logits = compiled_out.logits[:, -1, :].float()

    max_diff = (eager_logits - compiled_logits).abs().max().item()
    same_argmax = torch.equal(eager_logits.argmax(-1), compiled_logits.argmax(-1))
    print(f"max logit difference at first step: {max_diff}")
    print(f"same top token at first step: {same_argmax}")


@torch.no_grad()
def compiled_perplexity(model, tokenizer, n_sequences=20, seq_len=256):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to("cuda")

    compiled_forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)

    total_nll = 0.0
    total_tokens = 0
    for i in range(n_sequences):
        start = i * seq_len
        end = start + seq_len
        if end >= input_ids.shape[1]:
            break
        chunk = input_ids[:, start:end]

        past_key_values = StaticCache(config=model.config, max_cache_len=seq_len,
                                       device="cuda", dtype=torch.float16)
        cache_position = torch.arange(chunk.shape[1], device="cuda")

        torch.compiler.cudagraph_mark_step_begin()
        out = compiled_forward(input_ids=chunk, past_key_values=past_key_values,
                                use_cache=True, cache_position=cache_position)
        logits = out.logits[:, :-1, :].float()
        labels = chunk[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        n_tok = chunk.shape[1] - 1
        total_nll += loss.item() * n_tok
        total_tokens += n_tok

    avg_nll = total_nll / total_tokens
    return torch.exp(torch.tensor(avg_nll)).item()


if __name__ == "__main__":
    model, tokenizer = load_model()

    check_first_step_logit_diff(model, tokenizer)

    print("computing perplexity through the compiled plus static cache path")
    ppl = compiled_perplexity(model, tokenizer)
    print(f"compiled perplexity: {ppl:.4f}")
    print("compare to baseline eager perplexity: 26.9643 (from final_results.json)")
