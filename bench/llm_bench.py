#!/usr/bin/env python3
"""
LLM-oriented benchmark suite for snake.

Layer A microkernels (Python/NumPy baselines):
- sample_token (top-k/top-p, repetition penalty)
- cosine similarity (exact and candidate scoring)
- KV-cache ring buffer updates
- tokenizer (regex pretokenization + demo BPE merge loop, optional Rust ceiling)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from llm_kernels import (  # noqa: E402
    cosine_candidates,
    cosine_exact,
    kv_cache_ring_write,
    sample_token_numpy,
    tokenize_bpe_demo,
)

from cli_style import (  # noqa: E402
    Table,
    banner,
    c,
    Color,
    completion_banner,
    dim,
    format_number,
    format_time_us,
    info_line,
    section_header,
    success,
    warning,
)


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _summarize(samples_ns: np.ndarray) -> dict:
    return {
        "samples": int(samples_ns.size),
        "p50_ns": float(np.percentile(samples_ns, 50)),
        "p95_ns": float(np.percentile(samples_ns, 95)),
        "p99_ns": float(np.percentile(samples_ns, 99)),
        "mean_ns": float(samples_ns.mean()),
        "std_ns": float(samples_ns.std(ddof=0)),
    }


def _time_samples(fn, iterations: int, warmup: int) -> np.ndarray:
    for _ in range(warmup):
        fn()
    samples = np.empty(iterations, dtype=np.int64)
    for i in range(iterations):
        start = time.perf_counter_ns()
        fn()
        samples[i] = time.perf_counter_ns() - start
    return samples


def bench_sample_token(
    *,
    vocab_sizes: list[int],
    batch_sizes: list[int],
    iterations: int,
    warmup: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    repetition_window: int | None,
    seed: int,
) -> list[dict]:
    section_header("Token Sampling", "top-k/top-p with repetition penalty")

    info_line("Temperature", f"{temperature}")
    info_line("Top-K", f"{top_k}")
    info_line("Top-P", f"{top_p}")
    info_line("Repetition penalty", f"{repetition_penalty}")
    info_line("Repetition window", f"{repetition_window}")
    print()

    rng = np.random.default_rng(seed)
    results: list[dict] = []

    table = Table(
        [
            ("Vocab", 8, "right"),
            ("Batch", 6, "right"),
            ("p50", 12, "right"),
            ("p99", 12, "right"),
            ("per-token", 12, "right"),
        ]
    )
    table.print_header()

    for vocab in vocab_sizes:
        for batch in batch_sizes:
            logits = rng.standard_normal((batch, vocab), dtype=np.float32)
            scratch = np.empty_like(logits)

            past_tokens = None
            if repetition_penalty != 1.0:
                window = repetition_window or 0
                past_tokens = rng.integers(
                    0, vocab, size=(batch, max(window, 1)), dtype=np.int32
                )

            def step() -> int:
                token = 0
                for b in range(batch):
                    token = sample_token_numpy(
                        logits[b],
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        past_tokens=None if past_tokens is None else past_tokens[b],
                        repetition_window=repetition_window,
                        rng=rng,
                        scratch=scratch[b],
                    )
                return token

            samples = _time_samples(step, iterations, warmup)
            per_token = samples / batch

            step_stats = _summarize(samples)
            token_stats = _summarize(per_token)

            results.append(
                {
                    "kernel": "sample_token",
                    "config": {
                        "vocab_size": vocab,
                        "batch_size": batch,
                        "temperature": temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                        "repetition_window": repetition_window,
                    },
                    "step_ns": step_stats,
                    "token_ns": token_stats,
                }
            )

            table.print_row(
                [
                    c(Color.CYAN, format_number(vocab)),
                    str(batch),
                    format_time_us(step_stats["p50_ns"]),
                    format_time_us(step_stats["p99_ns"]),
                    format_time_us(token_stats["p50_ns"]),
                ]
            )

    return results


def bench_cosine_exact(
    *,
    dims: list[int],
    corpus_sizes: list[int],
    queries: int,
    iterations: int,
    warmup: int,
    seed: int,
) -> list[dict]:
    section_header("Cosine Similarity (Exact)", "Full corpus comparison")

    info_line("Queries per step", str(queries))
    print()

    rng = np.random.default_rng(seed)
    results: list[dict] = []

    table = Table(
        [
            ("Dim", 6, "right"),
            ("Corpus", 10, "right"),
            ("p50", 12, "right"),
            ("p99", 12, "right"),
            ("per-query", 12, "right"),
        ]
    )
    table.print_header()

    for dim in dims:
        for corpus in corpus_sizes:
            queries_arr = rng.standard_normal((queries, dim), dtype=np.float32)
            corpus_arr = rng.standard_normal((corpus, dim), dtype=np.float32)

            def step() -> None:
                cosine_exact(queries_arr, corpus_arr)

            samples = _time_samples(step, iterations, warmup)
            per_query = samples / queries
            step_stats = _summarize(samples)
            query_stats = _summarize(per_query)

            results.append(
                {
                    "kernel": "cosine_exact",
                    "config": {
                        "dim": dim,
                        "corpus_size": corpus,
                        "queries": queries,
                    },
                    "step_ns": step_stats,
                    "per_query_ns": query_stats,
                }
            )

            table.print_row(
                [
                    c(Color.CYAN, str(dim)),
                    format_number(corpus),
                    format_time_us(step_stats["p50_ns"]),
                    format_time_us(step_stats["p99_ns"]),
                    format_time_us(query_stats["p50_ns"]),
                ]
            )

    return results


def bench_cosine_candidates(
    *,
    dims: list[int],
    corpus_sizes: list[int],
    queries: int,
    candidate_k: int,
    iterations: int,
    warmup: int,
    seed: int,
) -> list[dict]:
    section_header("Cosine Similarity (Candidates)", "Subset scoring")

    info_line("Queries per step", str(queries))
    info_line("Candidate K", str(candidate_k))
    print()

    rng = np.random.default_rng(seed)
    results: list[dict] = []

    table = Table(
        [
            ("Dim", 6, "right"),
            ("Corpus", 10, "right"),
            ("K", 6, "right"),
            ("p50", 12, "right"),
            ("per-query", 12, "right"),
        ]
    )
    table.print_header()

    for dim in dims:
        for corpus in corpus_sizes:
            queries_arr = rng.standard_normal((queries, dim), dtype=np.float32)
            corpus_arr = rng.standard_normal((corpus, dim), dtype=np.float32)
            k = min(candidate_k, corpus)
            candidates = rng.integers(0, corpus, size=(queries, k), dtype=np.int64)

            def step() -> None:
                cosine_candidates(queries_arr, corpus_arr, candidates)

            samples = _time_samples(step, iterations, warmup)
            per_query = samples / queries
            step_stats = _summarize(samples)
            query_stats = _summarize(per_query)

            results.append(
                {
                    "kernel": "cosine_candidates",
                    "config": {
                        "dim": dim,
                        "corpus_size": corpus,
                        "queries": queries,
                        "candidate_k": k,
                    },
                    "step_ns": step_stats,
                    "per_query_ns": query_stats,
                }
            )

            table.print_row(
                [
                    c(Color.CYAN, str(dim)),
                    format_number(corpus),
                    str(k),
                    format_time_us(step_stats["p50_ns"]),
                    format_time_us(query_stats["p50_ns"]),
                ]
            )

    return results


def bench_kv_cache(
    *,
    max_len: int,
    dim: int,
    steps: int,
    iterations: int,
    warmup: int,
    seed: int,
) -> list[dict]:
    section_header("KV-Cache Ring Buffer", "Write operations")

    info_line("Max length", format_number(max_len))
    info_line("Dimension", str(dim))
    info_line("Steps per iteration", format_number(steps))
    print()

    rng = np.random.default_rng(seed)
    cache_k = rng.standard_normal((max_len, dim), dtype=np.float32)
    cache_v = rng.standard_normal((max_len, dim), dtype=np.float32)
    new_k = rng.standard_normal((steps, dim), dtype=np.float32)
    new_v = rng.standard_normal((steps, dim), dtype=np.float32)

    def step() -> int:
        idx = 0
        for i in range(steps):
            kv_cache_ring_write(cache_k, cache_v, new_k[i], new_v[i], idx)
            idx += 1
            if idx == max_len:
                idx = 0
        return idx

    samples = _time_samples(step, iterations, warmup)
    per_update = samples / steps
    step_stats = _summarize(samples)
    update_stats = _summarize(per_update)

    result = {
        "kernel": "kv_cache_ring",
        "config": {"max_len": max_len, "dim": dim, "steps": steps},
        "step_ns": step_stats,
        "per_update_ns": update_stats,
    }

    table = Table(
        [
            ("Metric", 14, "left"),
            ("p50", 12, "right"),
            ("p99", 12, "right"),
        ]
    )
    table.print_header()
    table.print_row(
        [
            c(Color.CYAN, "full step"),
            format_time_us(step_stats["p50_ns"]),
            format_time_us(step_stats["p99_ns"]),
        ]
    )
    table.print_row(
        [
            c(Color.CYAN, "per update"),
            format_time_us(update_stats["p50_ns"]),
            format_time_us(update_stats["p99_ns"]),
        ]
    )

    return [result]


def _generate_text(rng: np.random.Generator, length: int) -> str:
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "token",
        "stream",
        "vector",
        "cache",
        "model",
        "decode",
        "sample",
    ]
    punct = [".", ",", "!", "?", ";", ":"]
    parts: list[str] = []
    while len(" ".join(parts)) < length:
        if rng.random() < 0.1:
            parts.append("<BOS>")
        parts.append(rng.choice(words))
        if rng.random() < 0.15:
            parts[-1] = parts[-1] + rng.choice(punct)
        if rng.random() < 0.05:
            parts.append("<EOS>")
    return " ".join(parts)[:length]


def _load_corpus_text() -> str | None:
    path = Path(__file__).parent / "fixtures" / "tokenizer_corpus.txt"
    if not path.exists():
        return None
    return path.read_text().strip()


def _texts_from_corpus(base: str, length: int, batch: int) -> list[str]:
    if not base:
        return [""] * batch
    sep = " "
    base = " ".join(base.split())
    if len(base) < length + 1:
        repeat = (length // max(len(base), 1)) + 2
        base = (base + sep) * repeat
    doubled = base + sep + base
    texts: list[str] = []
    for i in range(batch):
        start = (i * 31) % len(base)
        texts.append(doubled[start : start + length])
    return texts


def _load_rust_tokenizer() -> tuple[object | None, str | None]:
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore
    except Exception:
        return None, "tokenizers not installed"

    vocab_path = Path(__file__).parent / "fixtures" / "tokenizer_vocab.json"
    merges_path = Path(__file__).parent / "fixtures" / "tokenizer_merges.txt"
    if not vocab_path.exists() or not merges_path.exists():
        return None, "tokenizer fixtures missing"

    try:
        model = models.BPE.from_file(
            str(vocab_path),
            str(merges_path),
            unk_token="[UNK]",
        )
    except Exception:
        try:
            vocab = json.loads(vocab_path.read_text())
            merges: list[tuple[str, str]] = []
            for line in merges_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                left, right = line.split()
                merges.append((left, right))
            model = models.BPE(vocab=vocab, merges=merges, unk_token="[UNK]")
        except Exception as exc:
            return None, f"tokenizer fixture load failed: {exc}"

    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer, None


def bench_tokenizer(
    *,
    text_lengths: list[int],
    batch: int,
    iterations: int,
    warmup: int,
    seed: int,
) -> list[dict]:
    section_header("Tokenizer", "BPE demo vs Rust ceiling")

    info_line("Batch size", str(batch))
    print()

    rng = np.random.default_rng(seed)
    results: list[dict] = []

    corpus_text = _load_corpus_text()
    rust_tokenizer, rust_error = _load_rust_tokenizer()

    table = Table(
        [
            ("Backend", 18, "left"),
            ("Length", 8, "right"),
            ("p50", 12, "right"),
            ("per-text", 12, "right"),
        ]
    )
    table.print_header()

    for length in text_lengths:
        if corpus_text:
            texts = _texts_from_corpus(corpus_text, length, batch)
        else:
            texts = [_generate_text(rng, length) for _ in range(batch)]

        def py_step() -> None:
            for text in texts:
                tokenize_bpe_demo(text)

        samples = _time_samples(py_step, iterations, warmup)
        step_stats = _summarize(samples)
        per_text = _summarize(samples / batch)
        results.append(
            {
                "kernel": "tokenizer_bpe_python",
                "config": {"text_length": length, "batch_size": batch},
                "step_ns": step_stats,
                "per_text_ns": per_text,
            }
        )

        table.print_row(
            [
                c(Color.CYAN, "bpe_python"),
                str(length),
                format_time_us(step_stats["p50_ns"]),
                format_time_us(per_text["p50_ns"]),
            ]
        )

        if rust_tokenizer is not None:

            def rust_step() -> None:
                for text in texts:
                    rust_tokenizer.encode(text)

            samples = _time_samples(rust_step, iterations, warmup)
            step_stats = _summarize(samples)
            per_text = _summarize(samples / batch)
            results.append(
                {
                    "kernel": "tokenizer_rust",
                    "config": {"text_length": length, "batch_size": batch},
                    "step_ns": step_stats,
                    "per_text_ns": per_text,
                }
            )

            table.print_row(
                [
                    c(Color.GREEN, "rust"),
                    str(length),
                    format_time_us(step_stats["p50_ns"]),
                    format_time_us(per_text["p50_ns"]),
                ]
            )
        elif rust_tokenizer is None and rust_error:
            results.append(
                {
                    "kernel": "tokenizer_rust",
                    "config": {"text_length": length, "batch_size": batch},
                    "status": "unavailable",
                    "reason": rust_error,
                }
            )

            table.print_row(
                [
                    dim("rust " + c(Color.YELLOW, "(skip)")),
                    str(length),
                    dim("—"),
                    dim(rust_error[:20]),
                ]
            )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-oriented benchmarks")
    parser.add_argument(
        "--kernel",
        default="all",
        choices=[
            "all",
            "sample_token",
            "cosine_exact",
            "cosine_candidates",
            "kv_cache",
            "tokenizer",
        ],
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=str, default=None)

    parser.add_argument("--vocab-sizes", type=str, default="32000,128000")
    parser.add_argument("--batch-sizes", type=str, default="1,8")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=128)

    parser.add_argument("--dims", type=str, default="384,768")
    parser.add_argument("--corpus-sizes", type=str, default="1000,10000")
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=1000)

    parser.add_argument("--kv-max-len", type=int, default=2048)
    parser.add_argument("--kv-dim", type=int, default=128)
    parser.add_argument("--kv-steps", type=int, default=4096)
    parser.add_argument("--text-lengths", type=str, default="128,1024,4096")
    parser.add_argument("--tokenizer-batch", type=int, default=8)

    args = parser.parse_args()

    results: list[dict] = []
    kernel = args.kernel

    banner(
        "🐍 snake LLM Benchmark Suite", f"Layer A baselines • NumPy {np.__version__}"
    )
    print()
    info_line("Iterations", str(args.iterations))
    info_line("Warmup", str(args.warmup))
    info_line("Seed", str(args.seed))

    if kernel in ("all", "sample_token"):
        results += bench_sample_token(
            vocab_sizes=_parse_int_list(args.vocab_sizes),
            batch_sizes=_parse_int_list(args.batch_sizes),
            iterations=args.iterations,
            warmup=args.warmup,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            repetition_window=args.repetition_window,
            seed=args.seed,
        )

    if kernel in ("all", "cosine_exact"):
        results += bench_cosine_exact(
            dims=_parse_int_list(args.dims),
            corpus_sizes=_parse_int_list(args.corpus_sizes),
            queries=args.queries,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
        )

    if kernel in ("all", "cosine_candidates"):
        results += bench_cosine_candidates(
            dims=_parse_int_list(args.dims),
            corpus_sizes=_parse_int_list(args.corpus_sizes),
            queries=args.queries,
            candidate_k=args.candidate_k,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
        )

    if kernel in ("all", "kv_cache"):
        results += bench_kv_cache(
            max_len=args.kv_max_len,
            dim=args.kv_dim,
            steps=args.kv_steps,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
        )

    if kernel in ("all", "tokenizer"):
        results += bench_tokenizer(
            text_lengths=_parse_int_list(args.text_lengths),
            batch=args.tokenizer_batch,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
        )

    if args.json:
        payload = {
            "meta": {
                "numpy_version": np.__version__,
                "iterations": args.iterations,
                "warmup": args.warmup,
            },
            "results": results,
        }
        if args.json == "-":
            json.dump(payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            path = Path(args.json)
            path.write_text(json.dumps(payload, indent=2))

    completion_banner("LLM benchmark complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
