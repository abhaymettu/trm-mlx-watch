import mlx.core as mx
import numpy as np

from trm import sudoku as S
from trm.model import CELLS, CLASSES, TRM, Config, decode, num_params


def small():
    return TRM(Config(d=32, layers=2, n=2, T=2, n_sup=3))


def test_default_size_is_about_7m():
    p = num_params(TRM(Config()))
    assert 4e6 < p < 9e6, p


def test_shapes_and_trace_length():
    m = small()
    p = mx.zeros((4, CELLS), mx.int32)
    x = m.embed_puzzle(p)
    y, z = m.init_state(4)
    trace = []
    y2, z2 = m.latent_recursion(x, y, z, trace=trace)
    assert y2.shape == z2.shape == (4, CELLS, 32)
    assert [t[0] for t in trace] == ["z", "z", "y"]
    y3, z3, logits, q = m.deep_recursion(x, y, z)
    assert logits.shape == (4, CELLS, CLASSES) and q.shape == (4,)


def test_deep_recursion_gradient_only_through_last_pass():
    m = small()
    p = mx.array(np.random.default_rng(0).integers(0, 10, (2, CELLS)), mx.int32)
    x = m.embed_puzzle(p)
    y, z = m.init_state(2)

    def f(y0):
        return m.deep_recursion(x, y0, z)[2].sum()
    g = mx.grad(f)(y)
    assert float(mx.abs(g).sum()) == 0.0


def test_decode_keeps_clues():
    p = mx.array([[5] + [0] * 80], mx.int32)
    logits = mx.zeros((1, CELLS, CLASSES))
    g = np.array(decode(logits, p))
    assert g[0, 0] == 5 and (g[0, 1:] == 1).all()


def test_solver_and_augment_valid():
    puz, sol, d = S.make_one(7)
    sols, _ = S.solve(puz, limit=2)
    assert len(sols) == 1 and sols[0] == sol and S.is_valid_solution(sol)
    rng = np.random.default_rng(1)
    for _ in range(5):
        p, s = S.augment(np.array([puz]), np.array([sol]), rng)
        assert S.is_valid_solution(s[0])
        assert ((p[0] == 0) | (p[0] == s[0])).all()
        assert S.solve(list(p[0]), limit=2)[0] == [list(s[0])]


def test_train_smoke(tmp_path):
    from trm import train
    rc = train.main(["--out", str(tmp_path), "--d", "32", "--n", "2", "--T", "2", "--n-sup", "3",
                     "--batch", "8", "--iters", "12", "--eval-every", "12", "--eval-n", "8",
                     "--train-n", "64", "--warmup", "2"])
    assert rc == 0
    assert (tmp_path / "model.safetensors").exists()
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert any('"eval"' in l for l in lines)
