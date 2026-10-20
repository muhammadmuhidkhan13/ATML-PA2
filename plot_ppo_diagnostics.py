import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

root = Path("results/task2_ppo")
log_path = root / "standard_train_log.jsonl"
summary_path = root / "standard_train_summary.json"

# Load existing training results
with log_path.open("r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

df = pd.DataFrame(rows)

with summary_path.open("r", encoding="utf-8") as f:
    summary = json.load(f)

# Exact column names from your PPO training logs
metrics = [
    ("Learned reward", "learned_reward"),
    ("Sampled KL", "sampled_kl"),
    ("Policy loss", "policy_loss"),
    ("Value loss", "value_loss"),
    ("Entropy", "entropy"),
    ("Clip fraction", "clip_fraction"),
    ("Gradient norm", "gradient_norm"),
    ("Response length", "response_length"),
]

# Verify all required metrics exist
required = ["update"] + [column for _, column in metrics]
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.sort_values("update")

# Read runtime and memory measurements
vram = summary.get("peak_vram_gib")
runtime = summary.get("wall_clock_seconds")

# Compact figure for academic report
fig, axes = plt.subplots(
    2, 4,
    figsize=(12, 4.8),
    sharex=True
)

axes = axes.flatten()

for ax, (title, column) in zip(axes, metrics):
    x = pd.to_numeric(df["update"])
    y = pd.to_numeric(df[column], errors="coerce")

    ax.plot(
        x, y,
        linewidth=1.5,
        marker="o",
        markersize=2.5
    )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=8)
    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-3, 4)
    )

    if len(x):
        ax.set_xlim(x.min(), x.max())

for ax in axes[4:]:
    ax.set_xlabel("PPO update", fontsize=8)

# Header with resource usage
details = []

if vram is not None:
    details.append(f"Peak VRAM: {float(vram):.2f} GiB")

if runtime is not None:
    details.append(f"Wall-clock: {float(runtime)/60:.1f} min")

fig.suptitle(
    "Standard PPO Continuation Diagnostics\n"
    + "  |  ".join(details),
    fontsize=12,
    fontweight="bold",
    y=1.03
)

fig.tight_layout()

# Save publication-quality outputs
output_dir = Path.cwd() / "ppo_figures"
output_dir.mkdir(parents=True, exist_ok=True)
png_path = output_dir / "standard_ppo_diagnostics.png"
pdf_path = output_dir / "standard_ppo_diagnostics.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

plt.close(fig)

print("\n=== STANDARD PPO DIAGNOSTICS ===")
print(f"Updates recorded: {len(df)}")
print(f"Peak VRAM: {vram} GiB")
print(f"Wall-clock time: {runtime} seconds")
print("All 8 required trajectories included.")
print(f"PNG saved: {png_path}")
print(f"PDF saved: {pdf_path}")
