import json
from pathlib import Path

root = Path("results/task2_ppo")
files = {
    "0.00": root / "kl_0_eval_generations.jsonl",
    "0.10": root / "kl_0p1_eval_generations.jsonl",
    "0.20": root / "kl_0p2_eval_generations.jsonl",
}

def load(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

data = {beta: load(path) for beta, path in files.items()}

print("Available records:", {b: len(v) for b, v in data.items()})
print("Example record keys:", list(next(iter(data.values()))[0].keys()))

def first(row, *keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None

def prompt_id(row, index):
    value = first(row, "prompt_id", "id", "example_id", "index")
    return str(value) if value is not None else f"row_{index}"

indexed = {
    beta: {prompt_id(row, i): row for i, row in enumerate(rows)}
    for beta, rows in data.items()
}

common = set.intersection(*(set(rows) for rows in indexed.values()))

def reward(row):
    value = first(row, "learned_reward", "reward_score", "reward", "rm_score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def response(row):
    value = first(
        row, "response", "generated_response", "completion",
        "generated_text", "output", "response_text"
    )
    return str(value) if value is not None else "[Response field not found]"

def length(row):
    value = first(row, "response_length", "generated_length", "length", "num_tokens")
    try:
        return int(value)
    except (TypeError, ValueError):
        return len(response(row).split())

def interest(pid):
    rows = [indexed[b][pid] for b in files]
    rewards = [reward(r) for r in rows]
    lengths = [length(r) for r in rows]
    rdiff = max(rewards) - min(rewards) if all(v is not None for v in rewards) else 0
    ldiff = max(lengths) - min(lengths)
    return rdiff, ldiff

ranked_reward = sorted(common, key=lambda p: interest(p)[0], reverse=True)[:5]
ranked_length = sorted(common, key=lambda p: interest(p)[1], reverse=True)[:5]

selected = list(dict.fromkeys(ranked_reward + ranked_length))

out = root / "kl_qualitative_comparison.txt"

with out.open("w", encoding="utf-8") as f:
    f.write("PPO KL-PRESSURE QUALITATIVE COMPARISON\n")
    f.write(f"Matched prompts available: {len(common)}\n")
    f.write("Selected: top 5 reward differences + top 5 length differences\n\n")

    for pid in selected:
        f.write("=" * 90 + "\n")
        f.write(f"PROMPT ID: {pid}\n")
        sample = indexed["0.00"][pid]
        prompt = first(sample, "prompt", "prompt_text", "question", "input")
        f.write(f"PROMPT:\n{prompt if prompt is not None else '[Prompt not found]'}\n")

        for beta in files:
            row = indexed[beta][pid]
            f.write("\n" + "-" * 70 + "\n")
            f.write(f"BETA = {beta}\n")
            f.write(f"REWARD = {reward(row)} | LENGTH = {length(row)}\n")
            f.write("RESPONSE:\n")
            f.write(response(row) + "\n")
        f.write("\n")

print(f"\nSaved {len(selected)} matched prompt comparisons to:")
print(out.resolve())
