"""Export from a checkpoint: (1) real solve traces for the visualizer, (2) float32 weights for WebGPU.

  .venv/bin/python scripts/export.py --run runs/v0 --out viz/assets --traces 6

A trace records, for each deep-supervision step and each of its T latent recursions:
  after every z pass  the per-cell size of the change in z (how much the model "thought" about a cell),
  after the y update  the decoded grid, per-cell confidence (max softmax prob) and the halting logit q.
All numbers come from running the checkpoint; nothing is drawn by hand.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trm.model import TRM, Config, decode  # noqa: E402


def load(run):
    meta = json.load(open(os.path.join(run, "config.json")))
    model = TRM(Config(**meta["config"]))
    model.load_weights(os.path.join(run, "model.safetensors"))
    return model, meta


def solve_trace(model, puzzle, solution=None, max_steps=None):
    cfg = model.cfg
    p = mx.array(np.asarray(puzzle, np.int32)[None])
    x = model.embed_puzzle(p)
    y, z = model.init_state(1)
    steps = []
    halted_at = None
    for k in range(max_steps or cfg.n_sup):
        passes = []
        for t in range(cfg.T):
            think = []
            for i in range(cfg.n):
                z_new = model.net(x + y + z)
                think.append(np.array(mx.sqrt(((z_new - z) ** 2).mean(-1)))[0])
                z = z_new
            y = model.net(y + z)
            logits, q = model.readout(y)
            prob = mx.softmax(logits, -1)
            grid = np.array(decode(logits, p))[0]
            conf = np.array(prob.max(-1))[0]
            mx.eval(y, z)
            passes.append({"think": [np.round(v, 4).tolist() for v in think], "grid": grid.tolist(),
                           "conf": np.round(conf, 3).tolist(), "q": round(float(q[0]), 3),
                           "correct_cells": None if solution is None else int((grid == solution).sum())})
        steps.append({"step": k, "passes": passes})
        if halted_at is None and passes[-1]["q"] > 0:
            halted_at = k
    final = np.array(steps[halted_at if halted_at is not None else -1]["passes"][-1]["grid"])
    return {"puzzle": list(map(int, puzzle)), "solution": None if solution is None else list(map(int, solution)),
            "halted_at": halted_at, "solved_at_halt": None if solution is None else bool((final == solution).all()),
            "steps": steps}


def export_weights(model, out):
    names, offset, blobs = [], 0, []
    for k, v in tree_flatten(model.parameters()):
        a = np.array(v, np.float32)
        names.append({"name": k, "shape": list(a.shape), "offset": offset})
        offset += a.size
        blobs.append(a.ravel())
    np.concatenate(blobs).astype("<f4").tofile(os.path.join(out, "weights.bin"))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v0")
    ap.add_argument("--data", default="data/sudoku_hard.npz")
    ap.add_argument("--out", default="viz/assets")
    ap.add_argument("--traces", type=int, default=6)
    ap.add_argument("--pool", type=int, default=200, help="test puzzles scanned to pick traces")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, meta = load(a.run)
    d = np.load(a.data)
    tp, ts, td = d["test_puzzles"], d["test_solutions"], d["test_difficulty"]
    progress = json.load(open(os.path.join(a.run, "progress.json")))
    traces, solved_idx, other_idx = [], [], []
    for i in range(min(a.pool, len(tp))):
        tr = solve_trace(model, tp[i], ts[i])
        tr.update({"test_index": i, "solver_guesses": int(td[i])})
        (solved_idx if tr["solved_at_halt"] else other_idx).append((i, tr))
    pick = [tr for _, tr in solved_idx[: a.traces - 1]] + [tr for _, tr in other_idx[:1]]
    if len(pick) < a.traces:
        pick += [tr for _, tr in other_idx[1: 1 + a.traces - len(pick)]]
    manifest = export_weights(model, a.out)
    info = {"config": meta["config"], "params": meta["params"], "trained_iters": progress["iter"],
            "trained_minutes": progress["minutes"], "last_eval": progress["last_eval"],
            "pool_scanned": min(a.pool, len(tp)), "pool_solved_at_halt": len(solved_idx),
            "note": "traces are picked from held-out test puzzles: solved ones first, plus one it gets wrong"}
    full = os.path.join(a.run, "eval_full.json")
    if os.path.exists(full):
        info["eval_full"] = json.load(open(full))
    json.dump({"info": info, "traces": pick}, open(os.path.join(a.out, "traces.json"), "w"))
    json.dump({"config": meta["config"], "tensors": manifest}, open(os.path.join(a.out, "weights.json"), "w"), indent=1)
    print(json.dumps({k: info[k] for k in ("params", "trained_iters", "pool_scanned", "pool_solved_at_halt")}))


if __name__ == "__main__":
    main()
