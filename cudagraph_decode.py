import torch
from transformers import StaticCache
from common import load_model

PROMPT = "Explain what a compiler does in simple terms."
MAX_NEW_TOKENS = 128


def run_eager(model, tokenizer):
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    return out


def build_compiled_runner(model, tokenizer):
    compiled_forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)

    def run():
        inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        max_cache_len = seq_len + MAX_NEW_TOKENS

        past_key_values = StaticCache(config=model.config, max_cache_len=max_cache_len,
                                       device="cuda", dtype=torch.float16)

        cache_position = torch.arange(seq_len, device="cuda")
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_key_values,
                        use_cache=True, cache_position=cache_position)
        logits = out.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = [input_ids, next_token]

        cache_position = torch.tensor([seq_len], device="cuda")
        for _ in range(MAX_NEW_TOKENS - 1):
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                out = compiled_forward(input_ids=next_token, past_key_values=past_key_values,
                                        use_cache=True, cache_position=cache_position)
            logits = out.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            cache_position = cache_position + 1

        return torch.cat(generated, dim=1)

    return run


if __name__ == "__main__":
    model, tokenizer = load_model()

    print("checking correctness, eager vs compiled plus static cache")
    eager_out = run_eager(model, tokenizer)
    compiled_run = build_compiled_runner(model, tokenizer)
    compiled_out = compiled_run()

    match = torch.equal(eager_out, compiled_out)
    print(f"generated tokens match exactly: {match}")
    if not match:
        print("eager:", tokenizer.decode(eager_out[0]))
        print("compiled:", tokenizer.decode(compiled_out[0]))
