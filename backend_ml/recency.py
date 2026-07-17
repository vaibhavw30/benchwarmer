"""Exponential recency weights for chronologically-sorted training rows.

Pure, no I/O — the TDD anchor of the recency-weighting feature (mirrors the
role of signal_research/calibration.py). See
docs/superpowers/specs/2026-07-17-rolling-window-training-design.md.
"""
import numpy as np


def recency_weights(n: int, half_life_games: float | None) -> np.ndarray:
    """Exponential recency weights for `n` rows sorted oldest->newest.

    The NEWEST row (index n-1) gets weight 1.0; older rows decay by games-ago:

        games_ago = (n - 1) - i          # newest row -> 0
        weight_i  = 0.5 ** (games_ago / half_life_games)

    half_life_games is measured in league-games (one row = one game).

    If half_life_games is None or not finite (np.inf), returns all-ones —
    uniform weighting, identical to pre-#5 behavior.

    Weights are intentionally NOT normalized: scikit-learn / XGBoost treat
    sample_weight as relative, and normalization would obscure the
    "newest == 1.0" invariant. Do not "fix" this.
    """
    if n == 0:
        return np.empty(0, dtype=float)
    if half_life_games is None or not np.isfinite(half_life_games):
        return np.ones(n, dtype=float)
    if half_life_games <= 0:
        raise ValueError(f"half_life_games must be > 0, got {half_life_games!r}")
    idx = np.arange(n, dtype=float)
    games_ago = (n - 1) - idx
    return 0.5 ** (games_ago / half_life_games)
