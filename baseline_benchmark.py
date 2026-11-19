import json
from common import load_model, benchmark_decode, compute_perplexity

if __name__ == "__main__":
    model, tokenizer = load_model()

    print("running decode benchmark on unmodified model")
    decode_stats = benchmark_decode(model, tokenizer)
    print(f"mean tokens per sec: {decode_stats['mean_tokens_per_sec']:.2f} "
          f"(std {decode_stats['std_tokens_per_sec']:.2f})")

    print("computing perplexity on wikitext 2")
    ppl = compute_perplexity(model, tokenizer)
    print(f"perplexity: {ppl:.4f}")

    results = {
        "decode": {k: v for k, v in decode_stats.items() if k != "sample_output"},
        "perplexity": ppl,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved to baseline_results.json")
