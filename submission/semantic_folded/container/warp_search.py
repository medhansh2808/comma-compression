"""Shortlist global integer shifts for token compression.

Token disagreement is a cheap search heuristic, not the final coding cost.
The encoder must compare actual token-stream and warp-signal bytes.
"""
from __future__ import annotations

import numpy as np

LEGACY_SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
CANDIDATE_SHIFTS = LEGACY_SHIFTS + tuple(
    (dx, dy)
    for dy in range(-2, 3)
    for dx in range(-2, 3)
    if (dx, dy) not in LEGACY_SHIFTS
)


def shift_tokens(tokens: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Same edge-replicated sampling convention as context_warp.apply_warp."""
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError("expected a 2D token map")
    height, width = tokens.shape
    ys = np.clip(np.arange(height) - dy, 0, height - 1)
    xs = np.clip(np.arange(width) - dx, 0, width - 1)
    return tokens[np.ix_(ys, xs)]


def shortlist(
    previous: np.ndarray,
    current: np.ndarray,
    extra_candidates: int = 4,
) -> list[tuple[int, int]]:
    """Return original shifts plus the best additional shifts by agreement."""
    previous = np.asarray(previous)
    current = np.asarray(current)
    if previous.ndim != 2 or previous.shape != current.shape:
        raise ValueError("expected token maps with matching 2D shapes")
    if extra_candidates < 0:
        raise ValueError("extra_candidates must be nonnegative")

    ranked = []
    for dx, dy in CANDIDATE_SHIFTS:
        if (dx, dy) in LEGACY_SHIFTS:
            continue
        mismatches = int(np.count_nonzero(
            shift_tokens(previous, dx, dy) != current
        ))
        ranked.append((mismatches, abs(dx) + abs(dy), dy, dx))

    ranked.sort()
    return list(LEGACY_SHIFTS) + [
        (dx, dy) for _, _, dy, dx in ranked[:extra_candidates]
    ]
