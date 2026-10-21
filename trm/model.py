"""Tiny Recursive Model in MLX, written from the paper's description (arXiv 2510.04871), not from any code.

One small network `net` is used for every step. Three states, each (batch, 81 cells, d):
  x  the embedded puzzle (fixed),
  y  the current answer (decodes to a grid through `head`),
  z  a latent "scratchpad".

latent recursion, one pass:   z <- net(x + y + z)   repeated n times     ("think")
                              y <- net(y + z)                            ("update the answer")
deep recursion:               T-1 latent recursions without gradients, then one with gradients
deep supervision:             up to N_sup deep recursions per puzzle, each one its own optimizer step,
                              with y and z carried over (detached) between steps
halting:                      q = q_head(mean over cells of y); trained to predict "the decoded answer
                              is fully right"; at test time a solve stops when q > 0

For Sudoku the paper's best variant swaps self-attention for an MLP across the sequence (cells), which
is what `MixerBlock` does. Post-norm RMSNorm and SwiGLU, as in the paper.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

CELLS = 81
VOCAB = 10      # input tokens: 0 empty, 1..9 given digits
CLASSES = 9     # output: digits 1..9 as classes 0..8


@dataclass
class Config:
    d: int = 512
    layers: int = 2
    expansion: float = 4.0
    n: int = 6          # z updates per latent recursion
    T: int = 3          # latent recursions per deep recursion
    n_sup: int = 16     # max deep-supervision steps per puzzle

    def to_dict(self):
        return asdict(self)


def _hidden(dim, expansion):
    h = int(round(expansion * dim * 2 / 3))
    return max(8, (h + 7) // 8 * 8)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def __call__(self, h):
        g, u = mx.split(self.gate_up(h), 2, axis=-1)
        return self.down(nn.silu(g) * u)


class MixerBlock(nn.Module):
    """Cell mixing (an MLP across the 81 cells) then channel mixing, each with post-norm."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cell_mlp = SwiGLU(CELLS, _hidden(CELLS, cfg.expansion))
        self.chan_mlp = SwiGLU(cfg.d, _hidden(cfg.d, cfg.expansion))
        self.norm1 = nn.RMSNorm(cfg.d, eps=1e-5)
        self.norm2 = nn.RMSNorm(cfg.d, eps=1e-5)

    def __call__(self, h):
        t = self.cell_mlp(h.transpose(0, 2, 1)).transpose(0, 2, 1)
        h = self.norm1(h + t)
        return self.norm2(h + self.chan_mlp(h))


class TRM(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(VOCAB, cfg.d)
        self.blocks = [MixerBlock(cfg) for _ in range(cfg.layers)]
        self.head = nn.Linear(cfg.d, CLASSES, bias=False)
        self.q_head = nn.Linear(cfg.d, 1)
        self.y_init = mx.random.normal((cfg.d,)) * 0.02
        self.z_init = mx.random.normal((cfg.d,)) * 0.02

    # ---- the one tiny network
    def net(self, h):
        for b in self.blocks:
            h = b(h)
        return h

    def embed_puzzle(self, puzzles):
        return self.embed(puzzles) * (self.cfg.d ** 0.5)

    def init_state(self, batch):
        shape = (batch, CELLS, self.cfg.d)
        return mx.broadcast_to(self.y_init, shape), mx.broadcast_to(self.z_init, shape)

    def latent_recursion(self, x, y, z, trace=None):
        for i in range(self.cfg.n):
            z = self.net(x + y + z)
            if trace is not None:
                trace.append(("z", i, z))
        y = self.net(y + z)
        if trace is not None:
            trace.append(("y", self.cfg.n, y))
        return y, z

    def deep_recursion(self, x, y, z):
        """T-1 recursions with gradients stopped, then one with gradients (the paper's training rule)."""
        for _ in range(self.cfg.T - 1):
            y, z = self.latent_recursion(x, y, z)
        y, z = mx.stop_gradient(y), mx.stop_gradient(z)
        y, z = self.latent_recursion(x, y, z)
        return y, z, self.head(y), self.q_head(y.mean(axis=1)).squeeze(-1)

    def readout(self, y):
        return self.head(y), self.q_head(y.mean(axis=1)).squeeze(-1)


def num_params(model):
    from mlx.utils import tree_flatten
    return sum(v.size for k, v in tree_flatten(model.parameters()))


def decode(logits, puzzles):
    """Argmax digits 1..9; given clues are kept as given."""
    pred = mx.argmax(logits, axis=-1).astype(mx.int32) + 1
    return mx.where(puzzles > 0, puzzles.astype(mx.int32), pred)
