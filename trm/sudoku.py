"""Procedural hard Sudoku: a bitmask solver, a minimal-puzzle generator, and validity-preserving augmentation.

Grids are length-81 integer arrays, row-major. 0 means an empty cell in a puzzle; 1 to 9 are digits.

"Hard" here is measured, not claimed: each puzzle is scored by how many guesses a plain
most-constrained-cell backtracking solver needs, and the generator keeps the hardest ones.
That is the same idea behind Sudoku-Extreme (puzzles that are hard for solvers), but this is my own
generator, so the difficulty distribution is not the paper's.
"""
from __future__ import annotations

import random

import numpy as np

ALL = 0x3FE  # bits 1..9
ROW = [i // 9 for i in range(81)]
COL = [i % 9 for i in range(81)]
BOX = [(i // 27) * 3 + (i % 9) // 3 for i in range(81)]
PEERS = [sorted({j for j in range(81) if j != i and (ROW[j] == ROW[i] or COL[j] == COL[i] or BOX[j] == BOX[i])})
         for i in range(81)]
POP = [bin(m).count("1") for m in range(1024)]


def _setup(grid):
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(grid):
        if v:
            b = 1 << v
            if rows[ROW[i]] & b or cols[COL[i]] & b or boxes[BOX[i]] & b:
                return None
            rows[ROW[i]] |= b
            cols[COL[i]] |= b
            boxes[BOX[i]] |= b
    return rows, cols, boxes


def solve(grid, limit=2, rng=None):
    """Return (solutions found up to `limit`, guess count). Solutions are lists of 81 ints.

    With `rng`, candidate order is shuffled (used to make random full grids)."""
    g = list(grid)
    st = _setup(g)
    if st is None:
        return [], 0
    rows, cols, boxes = st
    empties = [i for i in range(81) if g[i] == 0]
    sols = []
    guesses = [0]

    def rec(k):
        if len(sols) >= limit:
            return
        best, bestmask, bestc = -1, 0, 10
        for idx in range(k, len(empties)):
            i = empties[idx]
            m = ALL & ~(rows[ROW[i]] | cols[COL[i]] | boxes[BOX[i]])
            c = POP[m]
            if c < bestc:
                best, bestmask, bestc = idx, m, c
                if c <= 1:
                    break
        if best == -1:
            sols.append(list(g))
            return
        if bestc == 0:
            return
        empties[k], empties[best] = empties[best], empties[k]
        i = empties[k]
        digits = [d for d in range(1, 10) if bestmask >> d & 1]
        if rng is not None:
            rng.shuffle(digits)
        if bestc > 1:
            guesses[0] += 1
        r, c_, b = ROW[i], COL[i], BOX[i]
        for d in digits:
            bit = 1 << d
            g[i] = d
            rows[r] |= bit
            cols[c_] |= bit
            boxes[b] |= bit
            rec(k + 1)
            rows[r] &= ~bit
            cols[c_] &= ~bit
            boxes[b] &= ~bit
            g[i] = 0
            if len(sols) >= limit:
                break
        empties[k], empties[best] = empties[best], empties[k]

    rec(0)
    return sols, guesses[0]


def random_full_grid(rng):
    sols, _ = solve([0] * 81, limit=1, rng=rng)
    return sols[0]


def minimal_puzzle(solution, rng):
    """Remove clues in random order while the solution stays unique. The result is minimal:
    removing any one remaining clue gives more than one solution."""
    p = list(solution)
    order = list(range(81))
    rng.shuffle(order)
    for i in order:
        keep = p[i]
        p[i] = 0
        sols, _ = solve(p, limit=2)
        if len(sols) != 1:
            p[i] = keep
    return p


def difficulty(puzzle):
    """Guesses the deterministic solver needs to prove the solution unique (a hardness score)."""
    _, g = solve(puzzle, limit=2)
    return g


def make_one(seed):
    rng = random.Random(seed)
    sol = random_full_grid(rng)
    puz = minimal_puzzle(sol, rng)
    return puz, sol, difficulty(puz)


def is_valid_solution(grid):
    a = np.asarray(grid).reshape(9, 9)
    want = set(range(1, 10))
    for k in range(9):
        if set(a[k]) != want or set(a[:, k]) != want:
            return False
    for br in range(3):
        for bc in range(3):
            if set(a[br * 3:br * 3 + 3, bc * 3:bc * 3 + 3].ravel()) != want:
                return False
    return True


def augment(puz, sol, rng):
    """One random symmetry of the Sudoku group: digit relabel, band/stack and row/col swaps, transpose.
    Every output is a valid puzzle with the matching unique solution. Works on (N, 81) arrays."""
    puz = np.asarray(puz).reshape(-1, 9, 9)
    sol = np.asarray(sol).reshape(-1, 9, 9)
    perm = np.concatenate([[0], rng.permutation(9) + 1])
    bands = rng.permutation(3)
    rows = np.concatenate([b * 3 + rng.permutation(3) for b in bands])
    stacks = rng.permutation(3)
    cols = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])
    out = []
    for a in (puz, sol):
        a = perm[a][:, rows][:, :, cols]
        out.append(a)
    if rng.random() < 0.5:
        out = [a.transpose(0, 2, 1) for a in out]
    return out[0].reshape(-1, 81), out[1].reshape(-1, 81)
