"""Generate the dataset: minimal Sudoku puzzles, keep the hardest by solver guesses, split train and test.

  .venv/bin/python scripts/make_data.py --n 50000 --keep 0.4 --test 1000

Writes data/sudoku_hard.npz with puzzles, solutions and difficulty for train and test. Test puzzles are
generated from a different seed range and are never augmented into training.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trm import sudoku as S  # noqa: E402


def gen(seeds, procs):
    with mp.Pool(procs) as pool:
        return pool.map(S.make_one, seeds, chunksize=64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--keep", type=float, default=0.4, help="fraction kept, hardest first")
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--out", default="data/sudoku_hard.npz")
    a = ap.parse_args()
    t = time.time()
    rows = gen(range(a.n), a.procs)
    test_rows = gen(range(10_000_000, 10_000_000 + int(a.test / a.keep) + 1), a.procs)
    def hardest(rs, k):
        rs = sorted(rs, key=lambda r: -r[2])[:k]
        p = np.array([r[0] for r in rs], np.int8)
        s = np.array([r[1] for r in rs], np.int8)
        d = np.array([r[2] for r in rs], np.int32)
        return p, s, d
    tp, ts, td = hardest(rows, int(a.n * a.keep))
    ep, es, ed = hardest(test_rows, a.test)
    seen = {bytes(x) for x in tp}
    overlap = sum(bytes(x) in seen for x in ep)
    assert all(S.is_valid_solution(x) for x in ts[:200]) and all(S.is_valid_solution(x) for x in es)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, train_puzzles=tp, train_solutions=ts, train_difficulty=td,
                        test_puzzles=ep, test_solutions=es, test_difficulty=ed)
    print("train %d (guesses min %d median %d max %d, clues median %d)" % (
        len(tp), td.min(), np.median(td), td.max(), np.median((tp > 0).sum(1))))
    print("test %d (guesses min %d median %d max %d); exact overlap with train: %d" % (
        len(ep), ed.min(), np.median(ed), ed.max(), overlap))
    print("%.0fs" % (time.time() - t))


if __name__ == "__main__":
    main()
