"""Train a TRM from scratch on the generated hard Sudoku set.

  .venv/bin/python -m trm.train --out runs/v0 --minutes 300

Training follows the paper's deep supervision with halting, done the efficient way: the batch is a set
of slots. Each iteration runs one deep recursion on every slot and takes one optimizer step. A slot
keeps its (y, z) for the next iteration until it halts (q > 0) or reaches n_sup steps; then it gets a
fresh, randomly augmented puzzle and fresh (y, z).

Logged honestly: every eval line is on held-out test puzzles that were never trained on, with the EMA
weights, and says how many puzzles it used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from . import sudoku as S
from .model import TRM, Config, decode, num_params


def load_data(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def sample(rng, puzzles, solutions, n):
    idx = rng.integers(0, len(puzzles), n)
    ps, ss = [], []
    for i in idx:
        p, s = S.augment(puzzles[i:i + 1], solutions[i:i + 1], rng)
        ps.append(p[0])
        ss.append(s[0])
    return np.stack(ps).astype(np.int32), np.stack(ss).astype(np.int32)


def losses(model, puzzles, sols, y, z):
    x = model.embed_puzzle(puzzles)
    y2, z2, logits, q = model.deep_recursion(x, y, z)
    target = sols - 1
    ce = nn.losses.cross_entropy(logits, target).mean()
    correct = mx.all(mx.argmax(logits, axis=-1) == target, axis=-1)
    bce = nn.losses.binary_cross_entropy(q, correct.astype(mx.float32), with_logits=True).mean()
    return ce + 0.5 * bce, (y2, z2, logits, q, ce, bce)


def evaluate(model, puzzles, solutions, batch=250):
    """Run the full test-time solve: up to n_sup deep recursions, stop each puzzle when q > 0.

    Returns exact-puzzle accuracy at the halting step, accuracy if every puzzle runs all n_sup steps,
    accuracy after each step, blank-cell accuracy at halt, and the mean steps used."""
    n_sup = model.cfg.n_sup
    per_step = np.zeros(n_sup)
    halt_ok = cells_ok = cells_n = steps_used = 0
    for i in range(0, len(puzzles), batch):
        p = mx.array(puzzles[i:i + batch].astype(np.int32))
        s = solutions[i:i + batch].astype(np.int32)
        B = p.shape[0]
        x = model.embed_puzzle(p)
        y, z = model.init_state(B)
        halted = np.zeros(B, bool)
        at_halt = np.zeros((B, 81), np.int32)
        for k in range(n_sup):
            y, z, logits, q = model.deep_recursion(x, y, z)
            g = np.array(decode(logits, p))
            qn = np.array(q)
            ok = (g == s).all(1)
            per_step[k] += ok.sum()
            newly = (~halted) & ((qn > 0) | (k == n_sup - 1))
            at_halt[newly] = g[newly]
            steps_used += (k + 1) * newly.sum()
            halted |= newly
        blank = np.array(p) == 0
        halt_ok += (at_halt == s).all(1).sum()
        cells_ok += ((at_halt == s) & blank).sum()
        cells_n += blank.sum()
    n = len(puzzles)
    return {"n": n, "exact_at_halt": halt_ok / n, "exact_after_all_steps": per_step[-1] / n,
            "exact_by_step": [round(v / n, 4) for v in per_step], "blank_cell_acc_at_halt": cells_ok / cells_n,
            "mean_steps": steps_used / n}


def lr_at(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup)))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sudoku_hard.npz")
    ap.add_argument("--out", default="runs/v0")
    ap.add_argument("--train-n", type=int, default=20000, help="base puzzles used for training")
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--n-sup", type=int, default=16)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = load_data(a.data)
    tp, ts = data["train_puzzles"][:a.train_n], data["train_solutions"][:a.train_n]
    ep, es = data["test_puzzles"][:a.eval_n], data["test_solutions"][:a.eval_n]

    cfg = Config(d=a.d, layers=a.layers, n=a.n, T=a.T, n_sup=a.n_sup)
    model = TRM(cfg)
    model.freeze(keys=["y_init", "z_init"])
    ema = TRM(cfg)
    start_iter = 0
    ckpt = os.path.join(a.out, "model.safetensors")
    if a.resume and os.path.exists(ckpt):
        model.load_weights(os.path.join(a.out, "train_state.safetensors"))
        with open(os.path.join(a.out, "progress.json")) as f:
            start_iter = json.load(f)["iter"]
    mx.eval(model.parameters())
    ema.update(tree_map(lambda v: v, model.parameters()))
    if a.resume and os.path.exists(ckpt):
        ema.load_weights(ckpt)
    opt = optim.AdamW(learning_rate=a.lr, betas=[0.9, 0.95], weight_decay=a.wd)
    meta = {"config": cfg.to_dict(), "params": num_params(model), "args": vars(a),
            "train_base_puzzles": int(len(tp)), "eval_puzzles": int(len(ep))}
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("params %d (%.2fM), train base %d, eval %d" % (meta["params"], meta["params"] / 1e6, len(tp), len(ep)),
          flush=True)

    vg = nn.value_and_grad(model, losses)

    def step_fn(puzzles, sols, y, z):
        (loss, aux), grads = vg(model, puzzles, sols, y, z)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss, aux

    if not a.no_compile:
        state = [model.state, opt.state, mx.random.state]
        step_fn = partial(mx.compile, inputs=state, outputs=state)(step_fn)

    B = a.batch
    p_np, s_np = sample(rng, tp, ts, B)
    y, z = model.init_state(B)
    steps = np.zeros(B, np.int32)
    log = open(os.path.join(a.out, "metrics.jsonl"), "a")
    t0 = time.time()
    run = {"ce": 0.0, "bce": 0.0, "cell": 0.0, "k": 0, "halts": 0, "halt_ok": 0, "halt_steps": 0}
    it = start_iter
    while it < a.iters:
        opt.learning_rate = lr_at(it, a.lr, a.warmup, a.iters)
        pz, sz = mx.array(p_np), mx.array(s_np)
        loss, (y2, z2, logits, q, ce, bce) = step_fn(pz, sz, y, z)
        mx.eval(loss, y2, z2, logits, q, model.parameters(), opt.state)
        ema.update(tree_map(lambda e, p: a.ema * e + (1 - a.ema) * p, ema.parameters(), model.parameters()))
        steps += 1
        pred = np.array(mx.argmax(logits, axis=-1)) + 1
        blank = p_np == 0
        ok = (pred == s_np).all(1)
        run["ce"] += ce.item()
        run["bce"] += bce.item()
        run["cell"] += ((pred == s_np) & blank).sum() / blank.sum()
        run["k"] += 1
        halt = (np.array(q) > 0) | (steps >= a.n_sup)
        run["halts"] += int(halt.sum())
        run["halt_ok"] += int((ok & halt).sum())
        run["halt_steps"] += int(steps[halt].sum())
        y, z = mx.stop_gradient(y2), mx.stop_gradient(z2)
        if halt.any():
            nh = int(halt.sum())
            np_, ns_ = sample(rng, tp, ts, nh)
            p_np[halt], s_np[halt] = np_, ns_
            steps[halt] = 0
            m = mx.array(halt)[:, None, None]
            y0, z0 = model.init_state(B)
            y, z = mx.where(m, y0, y), mx.where(m, z0, z)
        it += 1
        elapsed = time.time() - t0
        done = it >= a.iters or elapsed > a.minutes * 60
        if it % a.log_every == 0 or done:
            k = max(1, run["k"])
            line = {"iter": it, "minutes": round(elapsed / 60, 2), "lr": opt.learning_rate.item(),
                    "train_ce": round(run["ce"] / k, 4), "train_bce": round(run["bce"] / k, 4),
                    "train_blank_cell_acc": round(run["cell"] / k, 4),
                    "train_exact_at_halt": round(run["halt_ok"] / max(1, run["halts"]), 4),
                    "train_mean_halt_steps": round(run["halt_steps"] / max(1, run["halts"]), 2)}
            print(json.dumps(line), flush=True)
            log.write(json.dumps(line) + "\n")
            log.flush()
            run = {kk: 0 for kk in run}
            run["ce"] = run["bce"] = run["cell"] = 0.0
        if it % a.eval_every == 0 or done:
            ev = evaluate(ema, ep, es)
            ev = {"iter": it, "minutes": round(elapsed / 60, 2), "eval": "held-out test, EMA weights",
                  **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()}}
            print(json.dumps(ev), flush=True)
            log.write(json.dumps(ev) + "\n")
            log.flush()
            ema.save_weights(ckpt)
            model.save_weights(os.path.join(a.out, "train_state.safetensors"))
            with open(os.path.join(a.out, "progress.json"), "w") as f:
                json.dump({"iter": it, "minutes": round(elapsed / 60, 2), "last_eval": ev}, f, indent=1)
        if done:
            break
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
