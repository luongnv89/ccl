#!/usr/bin/env python3
"""Coding-agent shaped benchmark for a llama.cpp server.

For each prompt size, runs:
  cold  — cache_prompt=false (first turn, no reuse)
  warm  — cache_prompt=true, same prompt re-sent (simulates turn N+1 with
          identical project context — what coding agents actually do)

Outputs a table to stdout and (optionally) machine-readable JSON to --out.

Usage:
    python3 bench_agent.py [--base URL] [--runs N] [--out PATH]
                           [--max-prompt-tokens N]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_CASES = [
    # (label, approx prompt tokens, n_predict)
    ("256 / 128",   256,   128),
    ("2k / 256",    2048,  256),
    ("8k / 256",    8192,  256),
    ("32k / 256",   32768, 256),
    ("64k / 256",   65536, 256),
]

CODE_FILLER = (
    "def process_event(event: dict) -> Result:\n"
    "    if not event.get('id'):\n"
    "        raise InvalidEventError('missing id')\n"
    "    payload = event['payload']\n"
    "    return Result(status='ok', payload=payload)\n"
    "\n"
)


def post(base: str, path: str, payload: dict, timeout: float = 900.0) -> tuple[float, dict]:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    wall = time.perf_counter() - t0
    return wall, json.loads(body)


def healthcheck(base: str) -> None:
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as r:
            if json.loads(r.read()).get("status") != "ok":
                raise RuntimeError(f"server at {base} not healthy")
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError) as e:
        print(
            f"error: cannot reach llama-server at {base}: {e}\n"
            "Is the server running? Try: curl -s {base}/health",
            file=sys.stderr,
        )
        sys.exit(2)


def make_prompt(approx_tokens: int) -> str:
    target_chars = approx_tokens * 4  # ~1 token per 4 chars
    text = CODE_FILLER
    while len(text) < target_chars:
        text += CODE_FILLER
    text = text[:target_chars]
    return (
        "You are a senior engineer reviewing a Python codebase. "
        "Below is the project source. Read it carefully.\n\n"
        f"<project>\n{text}\n</project>\n\n"
        "Now write a detailed, multi-paragraph review of this code. "
        "Cover style, edge cases, error handling, performance, and "
        "suggest concrete refactors. Begin your review:\n\n"
    )


def one_call(base: str, prompt: str, n_predict: int, cache_prompt: bool) -> dict:
    wall, resp = post(
        base,
        "/completion",
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "cache_prompt": cache_prompt,
            "stream": False,
        },
    )
    t = resp.get("timings", {})
    return {
        "wall_s": wall,
        "prompt_n": t.get("prompt_n", 0),
        "predict_n": t.get("predicted_n", 0),
        "pp_tok_s": t.get("prompt_per_second", 0.0),
        "tg_tok_s": t.get("predicted_per_second", 0.0),
        "prompt_ms": t.get("prompt_ms", 0.0),
        "ttft_s": t.get("prompt_ms", 0.0) / 1000.0,
    }


def run_case(base: str, label: str, prompt_tok: int, n_predict: int, runs: int) -> dict:
    prompt = make_prompt(prompt_tok)

    cold = [one_call(base, prompt, n_predict, cache_prompt=False) for _ in range(runs)]
    # Prime the warm cache then measure.
    one_call(base, prompt, n_predict, cache_prompt=True)
    warm = [one_call(base, prompt, n_predict, cache_prompt=True) for _ in range(runs)]

    def agg(rows: list[dict], key: str) -> float:
        return statistics.mean(r[key] for r in rows)

    return {
        "label": label,
        "prompt_tokens_target": prompt_tok,
        "prompt_n": int(agg(cold, "prompt_n")),
        "cold": {
            "predict_n": int(agg(cold, "predict_n")),
            "pp_tok_s": agg(cold, "pp_tok_s"),
            "tg_tok_s": agg(cold, "tg_tok_s"),
            "ttft_s": agg(cold, "ttft_s"),
            "wall_s": agg(cold, "wall_s"),
        },
        "warm": {
            "predict_n": int(agg(warm, "predict_n")),
            "tg_tok_s": agg(warm, "tg_tok_s"),
            "ttft_s": agg(warm, "ttft_s"),
            "wall_s": agg(warm, "wall_s"),
        },
    }


def render(rows: list[dict]) -> None:
    header = (
        f"{'case':<11} {'prompt':>7} | "
        f"{'cold pp tk/s':>13} {'cold ttft s':>12} {'cold tg tk/s':>13} {'cold wall':>10} | "
        f"{'warm ttft s':>12} {'warm tg tk/s':>13} {'warm wall':>10}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        c, w = r["cold"], r["warm"]
        print(
            f"{r['label']:<11} {r['prompt_n']:>7} | "
            f"{c['pp_tok_s']:>13.1f} {c['ttft_s']:>12.2f} {c['tg_tok_s']:>13.1f} {c['wall_s']:>10.2f} | "
            f"{w['ttft_s']:>12.3f} {w['tg_tok_s']:>13.1f} {w['wall_s']:>10.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8001", help="llama-server base URL")
    ap.add_argument("--runs", type=int, default=3, help="runs per case (cold + warm)")
    ap.add_argument("--out", help="write JSON result here")
    ap.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=65536,
        help="skip cases larger than this (use to fit within server --ctx-size)",
    )
    args = ap.parse_args()

    healthcheck(args.base)

    cases = [(l, p, n) for (l, p, n) in DEFAULT_CASES if p <= args.max_prompt_tokens]
    if not cases:
        print(f"error: no cases fit under --max-prompt-tokens={args.max_prompt_tokens}", file=sys.stderr)
        sys.exit(2)

    print(f"Benchmarking {args.base} (coding-agent profile, runs={args.runs})")
    print("Warming up server...")
    one_call(args.base, "Hello.", 8, cache_prompt=False)

    rows = []
    for label, p, n in cases:
        print(f"  case: {label} (prompt~{p}, predict={n}) ...", flush=True)
        rows.append(run_case(args.base, label, p, n, args.runs))

    render(rows)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "base": args.base,
                    "runs": args.runs,
                    "cases": rows,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
