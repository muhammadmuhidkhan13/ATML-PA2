import json
import random
from pathlib import Path

root = Path("results/task2_ppo")

paths = {
    "0.00": root / "kl_0_eval_generations.jsonl",
    "0.10": root / "kl_0p1_eval_generations.jsonl",
    "0.20": root / "kl_0p2_eval_generations.jsonl",
}

def load(path):
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["prompt_id"]: row for row in rows}

data = {beta: load(path) for beta, path in paths.items()}
common_ids = sorted(set.intersection(
    *(set(rows) for rows in data.values())
))

rng = random.Random(6304)
selected = rng.sample(common_ids, min(20, len(common_ids)))

output = root / "rq2_random_qualitative_comparison.txt"

with output.open("w", encoding="utf-8") as f:
    f.write("RQ2: RANDOM MATCHED QUALITATIVE COMPARISON\n")
    f.write("Selection seed: 6304\n")
    f.write(f"Total matched prompts: {len(common_ids)}\n")
    f.write(f"Sampled prompts: {len(selected)}\n")
    f.write("Selection independent of reward and response length.\n")
    f.write("These are final held-out responses after 8 PPO updates.\n\n")

    for i, pid in enumerate(selected, 1):
        f.write("=" * 85 + "\n")
        f.write(f"EXAMPLE {i} | PROMPT ID: {pid}\n\n")

        messages = data["0.00"][pid]["prompt_messages"]

        f.write("ORIGINAL PROMPT:\n")
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            f.write(f"[{role}]\n{content}\n\n")

        for beta in paths:
            row = data[beta][pid]
            f.write("-" * 65 + "\n")
            f.write(f"KL BETA = {beta}\n")
            f.write(f"RESPONSE LENGTH = {row['response_length']} tokens\n")
            f.write(f"TERMINATED WITH EOS = {row['terminated_with_eos']}\n")
            f.write("RESPONSE:\n")
            f.write(str(row["response"]) + "\n\n")

        f.write("\n")

print(f"Matched held-out prompts: {len(common_ids)}")
print(f"Random sample size: {len(selected)}")
print(f"Saved to: {output.resolve()}")
