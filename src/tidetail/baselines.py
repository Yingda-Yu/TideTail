"""Forecasting baselines — train-only thresholds and shared evaluation."""
from __future__ import annotations

import numpy as np


def persistence(history, horizon):
    if history.size == 0:
        raise ValueError("history is empty")
    if horizon <= 0:
        return np.array([], dtype=float)
    return np.full(horizon, float(history[-1]), dtype=float)


def seasonal_baseline(history, horizon, period=144):
    if history.size < period:
        return persistence(history, horizon)
    out = np.empty(horizon, dtype=float)
    base = history[-period:]
    for k in range(horizon):
        out[k] = base[k % period]
    return out


def oversampled_persistence(history, horizon, n_bootstrap=5, seed=0):
    rng = np.random.default_rng(seed)
    sigma = max(0.01, 0.1 * np.nanstd(history))
    boot = np.stack([
        persistence(history, horizon) + rng.normal(0, sigma, horizon)
        for _ in range(n_bootstrap)
    ], axis=0)
    return boot.mean(axis=0)


class SimpleGRU:
    """Deterministic CPU smoke-test GRU in pure NumPy.

    Intentionally tiny. Trains via central-difference SGD so the CPU
    pipeline stays torch-free. Suitable only for smoke tests.
    """

    def __init__(self, input_size=1, hidden_size=4, horizon=1, seed=0):
        rng = np.random.default_rng(seed)
        scale = 0.5 / max(hidden_size, input_size) ** 0.5
        self.W = rng.normal(0, scale, (hidden_size, input_size))
        self.U = rng.normal(0, scale, (hidden_size, hidden_size))
        self.b = np.zeros(hidden_size)
        self.Wo = rng.normal(0, scale, (horizon, hidden_size))
        self.bo = np.zeros(horizon)
        self.hidden_size = hidden_size
        self.horizon = horizon

    def _step(self, x, h):
        z = 1.0 / (1.0 + np.exp(-(self.W @ x + self.U @ h + self.b)))
        h_tilde = np.tanh(self.W @ x + self.U @ h + self.b)
        return (1 - z) * h + z * h_tilde

    def forward(self, x_seq):
        """x_seq: (seq_len, input_size).  Returns (horizon,)."""
        h = np.zeros(self.hidden_size)
        for xt in x_seq:
            xt = np.asarray(xt, float).reshape(-1)
            h = self._step(xt, h)
        return self.Wo @ h + self.bo

    def train(self, X, Y, epochs=20, lr=0.01):
        """Central-difference SGD. X: (n, seq, in), Y: (n, horizon)."""
        eps = 1e-4
        params = [self.W, self.U, self.b, self.Wo, self.bo]

        def _loss():
            preds = np.stack([self.forward(x) for x in X])
            return float(np.mean((preds - Y) ** 2))

        for _ in range(epochs):
            for p in params:
                it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    orig = p[idx]
                    p[idx] = orig + eps
                    lp = _loss()
                    p[idx] = orig - eps
                    lm = _loss()
                    p[idx] = orig - lr * (lp - lm) / (2 * eps)
                    it.iternext()
        return _loss()

    def predict(self, X):
        return np.stack([self.forward(x) for x in X])
