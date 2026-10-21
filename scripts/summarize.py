"""Print the honest numbers from a run's metrics.jsonl as markdown: held-out eval history and train curve.

  .venv/bin/python scripts/summarize.py runs/v0
"""
import json
import sys

run = sys.argv[1] if len(sys.argv) > 1 else "runs/v0"
rows = [json.loads(l) for l in open(f"{run}/metrics.jsonl")]
evals = [r for r in rows if "eval" in r]
train = [r for r in rows if "train_ce" in r]
cfg = json.load(open(f"{run}/config.json"))
print(f"params {cfg['params']:,}; config {cfg['config']}; train base puzzles {cfg['train_base_puzzles']:,}; "
      f"batch {cfg['args']['batch']}")
print()
print("| iter | minutes | held-out puzzles | exact solve at halt | exact after 16 steps | blank-cell acc | mean steps |")
print("|---:|---:|---:|---:|---:|---:|---:|")
for e in evals:
    print(f"| {e['iter']:,} | {e['minutes']:.0f} | {e['n']} | {e['exact_at_halt']:.1%} | {e['exact_after_all_steps']:.1%} "
          f"| {e['blank_cell_acc_at_halt']:.1%} | {e['mean_steps']:.1f} |")
print()
step = max(1, len(train) // 12)
print("| iter | train CE | train blank-cell acc (per step) | train exact at halt |")
print("|---:|---:|---:|---:|")
for t in train[::step] + ([train[-1]] if train and (len(train) - 1) % step else []):
    print(f"| {t['iter']:,} | {t['train_ce']:.3f} | {t['train_blank_cell_acc']:.1%} | {t['train_exact_at_halt']:.1%} |")
