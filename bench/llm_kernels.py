"""
Reference LLM-oriented microkernels for the benchmark suite.

These are intentionally written in Python/NumPy so they can serve as baselines
before Zig implementations land.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def _as_f32_1d(arr: NDArray) -> NDArray[np.float32]:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 1:
        raise ValueError(f"logits must be 1D, got shape {out.shape}")
    return np.ascontiguousarray(out)


SPECIAL_TOKENS: dict[str, int] = {"<BOS>": 0, "<EOS>": 1, "<PAD>": 2}
_SPECIAL_PATTERN = "|".join(
    re.escape(tok) for tok in sorted(SPECIAL_TOKENS.keys(), key=len, reverse=True)
)
_PRETOKEN_PATTERN = re.compile(
    rf"{_SPECIAL_PATTERN}|[A-Za-z0-9_]+|[^\sA-Za-z0-9_]"
)


_DEFAULT_BPE_PAIRS = [
    (ord("t"), ord("h")),
    (ord("h"), ord("e")),
    (ord("i"), ord("n")),
    (ord("e"), ord("r")),
    (ord("a"), ord("n")),
    (ord("o"), ord("n")),
    (ord("r"), ord("e")),
    (ord("d"), ord("e")),
    (ord("o"), ord("r")),
    (ord("l"), ord("y")),
    (ord("s"), ord("t")),
    (ord("e"), ord("n")),
]


def build_demo_bpe_merges() -> tuple[
    dict[tuple[int, int], int], dict[tuple[int, int], int]
]:
    fixture = Path(__file__).parent / "fixtures" / "bpe_demo.json"
    pairs = _DEFAULT_BPE_PAIRS
    start_id = 256
    if fixture.exists():
        data = json.loads(fixture.read_text())
        pairs = [tuple(pair) for pair in data.get("merges", pairs)]
        start_id = int(data.get("start_id", start_id))
    merges: dict[tuple[int, int], int] = {}
    vocab: dict[tuple[int, int], int] = {}
    next_id = start_id
    for rank, pair in enumerate(pairs):
        merges[pair] = rank
        vocab[pair] = next_id
        next_id += 1
    return merges, vocab


_DEMO_BPE = build_demo_bpe_merges()


def pretokenize_text(text: str) -> list[str]:
    return [tok for tok in _PRETOKEN_PATTERN.findall(text) if tok.strip()]


def bpe_encode_bytes(
    token_bytes: bytes,
    merges: dict[tuple[int, int], int],
    vocab: dict[tuple[int, int], int],
) -> list[int]:
    tokens = list(token_bytes)
    while len(tokens) > 1:
        best_pair = None
        best_rank = None
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = merges.get(pair)
            if rank is None:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_pair = pair
        if best_pair is None:
            break
        merged = vocab[best_pair]
        new_tokens: list[int] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


def tokenize_bpe_demo(text: str) -> list[int]:
    merges, vocab = _DEMO_BPE
    out: list[int] = []
    for tok in pretokenize_text(text):
        if tok in SPECIAL_TOKENS:
            out.append(SPECIAL_TOKENS[tok])
            continue
        out.extend(bpe_encode_bytes(tok.encode("utf-8"), merges, vocab))
    return out


def apply_repetition_penalty_inplace(
    logits: NDArray[np.float32],
    past_tokens: Sequence[int] | None,
    penalty: float,
    window: int | None,
) -> None:
    if past_tokens is None or penalty == 1.0:
        return
    tokens = np.asarray(past_tokens, dtype=np.int64)
    if window is not None and window > 0 and tokens.size > window:
        tokens = tokens[-window:]
    if tokens.size == 0:
        return
    unique_tokens = np.unique(tokens)
    mask = (unique_tokens >= 0) & (unique_tokens < logits.size)
    if not np.any(mask):
        return
    logits[unique_tokens[mask]] /= np.float32(penalty)


def top_k_indices(logits: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    if k <= 0 or k >= logits.size:
        return np.arange(logits.size, dtype=np.int64)
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    return idx.astype(np.int64, copy=False)


def softmax_stable(logits: NDArray[np.float32]) -> NDArray[np.float64]:
    values = logits.astype(np.float64, copy=False)
    values = values - np.max(values)
    exp = np.exp(values)
    denom = exp.sum()
    if denom == 0.0:
        return np.full_like(exp, 1.0 / exp.size)
    return exp / denom


def sample_token_numpy(
    logits: NDArray,
    *,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    past_tokens: Sequence[int] | None = None,
    repetition_window: int | None = 128,
    rng: np.random.Generator | None = None,
    scratch: NDArray[np.float32] | None = None,
) -> int:
    """Sample a token from logits with top-k/top-p and repetition penalty.

    This function is optimized to benchmark the selection path, not Python list
    overhead. It assumes logits is a contiguous float32 buffer.
    """
    work = _as_f32_1d(logits)
    if scratch is not None:
        if scratch.shape != work.shape:
            raise ValueError("scratch buffer shape mismatch")
        np.copyto(scratch, work)
        work = scratch
    else:
        work = work.copy()

    apply_repetition_penalty_inplace(
        work, past_tokens, repetition_penalty, repetition_window
    )

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if temperature != 1.0:
        work /= np.float32(temperature)

    idx = top_k_indices(work, top_k)
    selected = work[idx]
    probs = softmax_stable(selected)

    if top_p < 1.0:
        if top_p <= 0.0:
            return int(idx[0])
        cumsum = np.cumsum(probs)
        cutoff = int(np.searchsorted(cumsum, top_p, side="left")) + 1
        idx = idx[:cutoff]
        probs = probs[:cutoff]
        probs = probs / probs.sum()

    if rng is None:
        rng = np.random.default_rng()
    return int(rng.choice(idx, p=probs))


def cosine_exact(
    queries: NDArray, corpus: NDArray, *, eps: float = 1e-8
) -> NDArray[np.float32]:
    q = np.asarray(queries, dtype=np.float32)
    c = np.asarray(corpus, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + eps)
    c_norm = c / (np.linalg.norm(c, axis=1, keepdims=True) + eps)
    return q_norm @ c_norm.T


def cosine_candidates(
    queries: NDArray,
    corpus: NDArray,
    candidates: NDArray,
    *,
    eps: float = 1e-8,
) -> NDArray[np.float32]:
    q = np.asarray(queries, dtype=np.float32)
    c = np.asarray(corpus, dtype=np.float32)
    cand = np.asarray(candidates, dtype=np.int64)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + eps)
    c_norm = c / (np.linalg.norm(c, axis=1, keepdims=True) + eps)

    out = np.empty((q_norm.shape[0], cand.shape[1]), dtype=np.float32)
    for i in range(q_norm.shape[0]):
        idx = cand[i]
        out[i] = q_norm[i] @ c_norm[idx].T
    return out


def kv_cache_ring_write(
    cache_k: NDArray[np.float32],
    cache_v: NDArray[np.float32],
    new_k: NDArray[np.float32],
    new_v: NDArray[np.float32],
    idx: int,
) -> None:
    cache_k[idx] = new_k
    cache_v[idx] = new_v
