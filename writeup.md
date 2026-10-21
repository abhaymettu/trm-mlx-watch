# Watch a 4.3M model think, trained on my laptop (draft)

## The 20-second clip

A hard Sudoku sits on the screen. My own solver needs 1,727 guesses to prove its solution unique. The model has never seen this puzzle. Over five supervision steps the board changes from mostly red guesses to all green. Each step runs three recursions, and each recursion runs six latent passes (the amber glow marks the cells whose latent changed most) and then one answer update. The halt bar sits deep in the negative while the model is still unsure. On step 5 it crosses zero and the model stops itself: 58 of 58 blanks right.

Every frame is a forward pass of the real checkpoint. A button on the page recomputes the same solve in the browser with WebGPU, from the exported weights, and checks it against the MLX run.

## What was built overnight

- A clean-room TRM in MLX, written from the paper alone. It uses one tiny network applied recursively, a latent `z` and an answer `y`, deep supervision with carried state, and a learned halting head. It is the attention-free variant, with 4.3M parameters.
- A procedural hard-Sudoku generator that makes minimal puzzles ranked by solver guesses. It produced 20,000 train puzzles and 1,000 held-out test puzzles.
- A training loop with slot-based adaptive halting, Sudoku-symmetry augmentation and EMA weights, run for 200 minutes on an M4 Pro MacBook. The machine was shared with other jobs and nearly out of swap, so it used batch 32.
- A browser visualizer with trace replay and a live WebGPU forward pass, a headless smoke test, and a scripted clip recorder.

## Honest numbers

- 80.8% of 1,000 held-out puzzles solved. "Solved" means a valid Sudoku that keeps every clue; all 808 also matched the stored solution.
- 79.9% of 1,000 training puzzles, so the test score is not memorization.
- 76.4% on the hardest third (120 to 1,727 solver guesses).
- Recursion matters: 0% solved after one supervision step, 62% after four, 81% after sixteen.
- The model halts on its own after 6.2 steps on average.

These numbers are not comparable to the paper. The puzzles are from my generator, not Sudoku-Extreme, and they are probably easier. The model is the 4.3M variant, not 7M. Training stopped at 16,208 of 40,000 planned steps, with one unstable stretch near step 7,900 that recovered.

## Not done yet

- WebGPU speed on a real GPU was not measured. Headless verification ran on SwiftShader, a software adapter.
- Any public clip should be re-recorded with Recordly. This one is a headless Playwright capture of the trace replay.
- Isomorphic train/test overlap was not checked.
- Next steps: finish the 40k-step schedule, evaluate on a published hard set, and try the 7M attention variant for comparison.
