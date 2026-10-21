"""Independent check of a checkpoint on held-out test puzzles.

  .venv/bin/python scripts/evaluate.py --run runs/v0 [--n 1000]

Runs the test-time solve (up to n_sup steps, halt at q > 0) and counts a puzzle solved only if the
predicted grid is a valid Sudoku (every row, column and box is 1..9) and keeps every given clue. That
check does not use the stored solution, so it cannot be fooled by a labelling bug. It also reports the
match against the stored solution, a train-subset score, and scores by solver-difficulty tercile.
"""
import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trm import sudoku as S  # noqa: E402
from trm.model import TRM, Config, decode  # noqa: E402


def solve_batch(model, puzzles, batch=200):
    out, steps = [], []
    for i in range(0, len(puzzles), batch):
        p = mx.array(puzzles[i:i + batch].astype(np.int32))
        B = p.shape[0]
        x = model.embed_puzzle(p)
        y, z = model.init_state(B)
        final = np.zeros((B, 81), np.int32)
        used = np.zeros(B, np.int32)
        done = np.zeros(B, bool)
        for k in range(model.cfg.n_sup):
            y, z, logits, q = model.deep_recursion(x, y, z)
            g = np.array(decode(logits, p))
            new = ~done & ((np.array(q) > 0) | (k == model.cfg.n_sup - 1))
            final[new], used[new] = g[new], k + 1
            done |= new
        out.append(final)
        steps.append(used)
    return np.concatenate(out), np.concatenate(steps)


def score(model, puz, sol):
    pred, used = solve_batch(model, puz)
    valid = np.array([S.is_valid_solution(g) for g in pred])
    keeps = ((puz == 0) | (pred == puz)).all(1)
    rules_ok = valid & keeps
    match = (pred == sol).all(1)
    return rules_ok, match, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v0")
    ap.add_argument("--data", default="data/sudoku_hard.npz")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--train-n", type=int, default=1000)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.run, "config.json")))
    model = TRM(Config(**meta["config"]))
    model.load_weights(os.path.join(a.run, "model.safetensors"))
    d = np.load(a.data)
    tp, ts, td = d["test_puzzles"][:a.n], d["test_solutions"][:a.n], d["test_difficulty"][:a.n]
    ok, match, used = score(model, tp, ts)
    cuts = np.quantile(td, [1 / 3, 2 / 3])
    terc = np.digitize(td, cuts)
    by = {name: {"n": int((terc == k).sum()), "solved": round(float(ok[terc == k].mean()), 4),
                 "guess_range": [int(td[terc == k].min()), int(td[terc == k].max())]}
          for k, name in enumerate(["easier third", "middle third", "hardest third"])}
    trp, trs = d["train_puzzles"][:a.train_n], d["train_solutions"][:a.train_n]
    tok, _, _ = score(model, trp, trs)
    res = {"checkpoint": a.run, "test_puzzles": int(len(tp)), "test_solved_valid_sudoku": round(float(ok.mean()), 4),
           "test_matches_stored_solution": round(float(match.mean()), 4),
           "valid_but_not_stored": int((ok & ~match).sum()), "mean_steps": round(float(used.mean()), 2),
           "by_difficulty": by, "train_subset_puzzles": int(len(trp)), "train_subset_solved": round(float(tok.mean()), 4)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(a.run, "eval_full.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
