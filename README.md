# trm-mlx-watch

Watch a 4.3M-parameter recursive model think. It was trained from scratch on a MacBook.

This repo is a clean-room Tiny Recursive Model (TRM, [arXiv 2510.04871](https://arxiv.org/abs/2510.04871)) in MLX. It was written from the paper's description; no official code was read or copied. It is trained on hard Sudoku puzzles that this repo generates. A browser visualizer replays a real solve from the checkpoint, one recursion pass at a time, and can recompute that solve in WebGPU from the exported weights.

![20-second clip](clip/trm-solve.mp4)

## Results (checkpoint `runs/v0`)

All numbers come from this repo's own runs on its own puzzles, not from the paper.

| measure | value |
|---|---|
| parameters | 4,320,785 |
| training | 16,208 optimizer steps, 200 min, batch 32, M4 Pro (24 GB), MLX 0.32.4 |
| held-out test puzzles solved | **80.8% of 1,000** (`scripts/evaluate.py`) |
| how "solved" is checked | the output grid is a valid Sudoku and keeps every clue; it also matched the stored solution in all 808 cases |
| train-subset puzzles solved | 79.9% of 1,000 (so the test score is not memorization) |
| solved, by solver difficulty | easier third 80.7%, middle third 85.3%, hardest third 76.4% |
| mean supervision steps used (the model halts itself) | 6.2 of 16 |

Recursion is what does the work. On 500 held-out puzzles, the share solved after each supervision step was 0% after 1, 14.8% after 2, 62.0% after 4, 73.2% after 6 and 81.0% after 16.

Held-out eval during training (EMA weights, 500 test puzzles, from `runs/v0/metrics.jsonl`):

| iter | minutes | exact solve at halt | blank-cell accuracy |
|---:|---:|---:|---:|
| 2,000 | 11 | 0.0% | 24.5% |
| 4,000 | 32 | 0.0% | 51.0% |
| 6,000 | 54 | 9.0% | 64.6% |
| 8,000 | 78 | 29.2% | 73.9% |
| 10,000 | 100 | 43.4% | 79.7% |
| 12,000 | 124 | 78.8% | 91.7% |
| 14,000 | 164 | 83.6% | 93.4% |
| 16,208 | 200 | 80.4% | 92.6% |

The training run is only partly done. The schedule planned 40,000 steps and the 200-minute cap stopped it at 16,208, so the cosine learning-rate decay never finished. Training loss had one unstable stretch: near step 7,900, train blank-cell accuracy fell from about 70% to 25%, then recovered by step 10,500. The full log is `runs/v0.log`.

### What these numbers are not

- **Not Sudoku-Extreme.** The puzzles come from my own generator (`trm/sudoku.py`). It makes minimal puzzles (removing any one clue breaks uniqueness) from random grids, scores each one by how many guesses a backtracking solver needs, and keeps the hardest 40% of 50,000. Train has 20,000 puzzles (median 24 clues, median 92 solver guesses). Test has 1,000 puzzles from a separate seed range, with zero exact overlap with train. This set is probably easier than the paper's, so the 80.8% is not comparable to the paper's figures.
- **Not 7M parameters.** This is the paper's attention-free variant (an MLP across the 81 cells instead of self-attention), which the paper reports as its best Sudoku model. At d=512 with 2 layers it has 4.3M parameters.
- Isomorphic overlap between train and test (the same puzzle under a symmetry) was not checked. It is very unlikely for random minimal puzzles, but it is unmeasured.

## The model (`trm/model.py`)

There is one small network `net` (2 mixer blocks: an MLP across cells, then a SwiGLU across channels, with post-norm RMSNorm) and three states per cell: `x` (the embedded puzzle), `y` (the answer) and `z` (a latent).

- latent recursion: `z = net(x + y + z)` six times, then `y = net(y + z)`
- deep recursion: three latent recursions; only the last one carries gradients
- deep supervision: up to 16 deep recursions per puzzle, each its own optimizer step, with `y` and `z` carried over (detached)
- halting: `q = q_head(mean_cells(y))` is trained with BCE to predict "the whole grid is right"; a solve stops when `q > 0`

Training (`trm/train.py`) keeps a batch of slots. Each slot holds a puzzle and its carried state, and a slot gets a fresh, randomly augmented puzzle when it halts or reaches 16 steps. The augmentations are digit relabelling, band, stack, row and column permutations, and transpose. Training uses AdamW (lr 1e-4, weight decay 1.0, warmup 1,000 steps) and an EMA of the weights (0.999) for evaluation. `y_init` and `z_init` are fixed random vectors, not learned.

## The visualizer (`viz/`)

`scripts/export.py` runs the checkpoint on held-out test puzzles and writes `viz/assets/traces.json`. For every supervision step and recursion, it records the per-cell size of each `z` change (the amber glow), the decoded grid and confidence after each `y` update, and `q`. It also writes `weights.bin` and `weights.json` for the browser. The page replays the trace: green is right, red is wrong, brightness is confidence, and the bar is the halt signal.

**Run live (WebGPU)** loads the same weights and recomputes the solve with WGSL compute shaders (tiled matmul, SwiGLU, add+RMSNorm). It then reports how many answer grids match the MLX trace. In headless Chromium, on the SwiftShader software adapter, it reproduced 15 of 15 grids of trace #0 exactly (max |q| difference 0.0008). Speed on a real GPU adapter was not measured; SwiftShader took 128 s.

```sh
cd viz && python3 -m http.server 8000   # open http://localhost:8000
```

The clip `clip/trm-solve.mp4` is 20 s of the page replaying trace #0, a test puzzle that needs 1,727 solver guesses, which the model solves and halts on at supervision step 5. It was recorded with headless Playwright (`scripts/record_clip.mjs`), so it is a replay of the real trace, not a screen recording of live WebGPU.

## Reproduce

```sh
uv venv -p 3.13 .venv && uv pip install -p .venv mlx numpy pytest
.venv/bin/python scripts/make_data.py --n 50000 --keep 0.4 --test 1000      # ~6 min, 12 processes
.venv/bin/python -u -m trm.train --out runs/v0 --batch 32 --iters 40000 --minutes 200 --eval-every 2000 --eval-n 500 --warmup 1000
.venv/bin/python scripts/evaluate.py --run runs/v0
.venv/bin/python scripts/export.py --run runs/v0 --out viz/assets --traces 6 --pool 40
npm install && node tests/viz_smoke.mjs assets && node scripts/record_clip.mjs --speed 6
```

On a shared machine, wrap heavy commands in `python3 scripts/with_lock.py --wait -- ...`, which holds `/tmp/overnight-compute.lock`.

## Tests

- `.venv/bin/python -m pytest -q tests`: model size, state shapes, recursion trace order, "no gradient through the first T-1 recursions", clue-keeping decode, solver and augmentation validity, a training smoke run, and the export format.
- `node tests/viz_smoke.mjs assets`: serves `viz/`, checks that the board renders the puzzle, that the partial-training label is shown and that replay advances, then runs live WebGPU and checks it agrees with the MLX trace.
