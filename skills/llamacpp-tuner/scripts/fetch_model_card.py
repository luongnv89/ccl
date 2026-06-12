#!/usr/bin/env python3
"""Fetch a Hugging Face model card for the llamacpp-tuner skill (Step 3).

The GGUF a user is running almost always comes from a *quantizer* repo
(Unsloth, bartowski, lmstudio-community), and that repo's card — the prose
``README.md``, not the API's frontmatter-only ``cardData`` — is where model
authors publish the facts and config advice that drive a good llama.cpp
launch: native context length, OOM/ctx-size guidance, recommended output
length, MTP/draft notes, and sometimes a literal ``llama-server`` command.

This script pulls that card so Step 4 can ground its config proposal in what
the model author actually recommends, instead of generic defaults.

It is **best-effort by design** (SKILL.md says "always *try*"): a missing
card, a 404, or no network must not stop the tuner. On any miss the script
prints a one-line human-readable note explaining what failed and exits ``0``
so the workflow continues and reports Step 3 as PARTIAL.

Two data sources are combined:

  1. ``https://huggingface.co/api/models/<repo>``  → structured facts:
     ``gguf`` (architecture, context_length, chat_template), ``cardData``
     (base_model, tags, license), and the ``.gguf`` ``siblings`` list.
  2. ``https://huggingface.co/<repo>/raw/main/README.md`` → the card *prose*,
     from which we extract only the perf-relevant sections and lines.

Sampling parameters (temperature / top_p / top_k / repetition_penalty) are
deliberately **excluded** from the recommendations: this skill tunes the
server, never the sampler. They are noted as "ignored (sampling)" only so a
human can see the script saw them.

Repo resolution, in priority order:
  * ``--repo <org/name>``                        (explicit; wins)
  * ``--model <path>`` containing ``models--org--repo`` (HF cache convention)
  * ``--model <path>`` whose parent dirs look like ``<org>/<repo>``
  * else: give up with a note telling the caller to pass ``--repo`` (the repo
    id is visible from ``/v1/models`` or ``/props`` on the running server).

Usage:
    python3 fetch_model_card.py --repo unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
    python3 fetch_model_card.py --model /path/to/models--unsloth--Foo-GGUF/snapshots/ab/Foo.gguf
    python3 fetch_model_card.py --repo org/name --json    # machine-readable

Exit codes:
    0  — card fetched, OR a clean best-effort miss (note printed). Both are
         non-error: Step 3 continues either way.
    2  — bad invocation (neither --repo nor --model given).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

API_URL = "https://huggingface.co/api/models/{repo}"
RAW_README_URL = "https://huggingface.co/{repo}/raw/main/README.md"
TIMEOUT = 15  # seconds — a hung fetch must not stall the tuner

# Lines/sections worth surfacing to Step 4. Perf- and config-relevant only.
KEEP_PATTERNS = re.compile(
    r"""
    context\s*length        | n_ctx        | ctx[-_ ]?size   | ctx_train       |
    \b256k\b | \b262,?144\b  | \b131,?072\b | \b1m\s+token     | yarn            |
    out[-\s]?of[-\s]?memory  | \boom\b      | reduc\w+\s+the\s+context           |
    output\s+length          | n_predict    | max[-_ ]?tokens                    |
    llama-server | llama-cli | \-\-ctx      | \-\-n-gpu-layers | \-\-flash-attn  |
    \-\-cache    | \-\-parallel | \-\-ubatch | \-\-batch      | \-\-threads      |
    mtp | multi[-\s]?token   | speculat\w+  | draft        | \-\-spec          |
    quant\w*     | q4_k | q5_k | q6_k | q8_0 | iq\d | ud-q | imatrix            |
    moe | expert | active\s+param | total\s+param | bandwidth                    |
    recommend\w* | best\s+practice | optimal | settings                         |
    chat\s+template | jinja
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Sampling knobs this skill never touches — flagged, not recommended.
SAMPLING_PATTERN = re.compile(
    r"temperature|top[-_ ]?p|top[-_ ]?k|min[-_ ]?p|"
    r"repetition[-_ ]?penalty|presence[-_ ]?penalty|frequency[-_ ]?penalty",
    re.IGNORECASE,
)


def resolve_repo(repo: str | None, model_path: str | None) -> str | None:
    """Resolve an ``org/name`` HF repo id from an explicit repo or a model path.

    The HF hub cache encodes the repo in the directory name as
    ``models--<org>--<name>`` (``--`` separates org from name, and any ``/``
    inside the name becomes ``--`` too — rare, but handled by re-joining).
    """
    if repo:
        return repo.strip().strip("/")
    if not model_path:
        return None

    # HF cache convention: .../models--org--name/snapshots/<sha>/file.gguf
    m = re.search(r"models--([^/]+?)--(.+?)/(?:snapshots|blobs|refs)/", model_path)
    if not m:
        # Path may end at the cache dir itself, no trailing /snapshots.
        m = re.search(r"models--([^/]+?)--([^/]+?)(?:/|$)", model_path)
    if m:
        org, name = m.group(1), m.group(2).replace("--", "/")
        return f"{org}/{name}"

    return None


def _get(url: str) -> tuple[str | None, str | None]:
    """GET a URL. Return ``(body, None)`` on success or ``(None, reason)``."""
    req = urllib.request.Request(url, headers={"User-Agent": "llamacpp-tuner/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read().decode("utf-8", "replace"), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return None, f"network error ({e.reason})"
    except Exception as e:  # pragma: no cover - defensive
        return None, f"{type(e).__name__}: {e}"


def _strip_frontmatter(md: str) -> str:
    """Drop a leading ``---`` YAML frontmatter block from card markdown."""
    if md.startswith("---"):
        end = md.find("\n---", 3)
        if end != -1:
            return md[end + 4 :]
    return md


def extract_recommendations(readme: str) -> tuple[list[str], list[str]]:
    """Pull perf-relevant lines from card prose; separate sampling mentions.

    Returns ``(kept, sampling)`` where ``kept`` is config/perf advice worth
    handing to Step 4 and ``sampling`` is the temperature/top-p style lines
    we explicitly do NOT act on (surfaced only for human awareness).
    """
    body = _strip_frontmatter(readme)
    kept: list[str] = []
    sampling: list[str] = []
    seen: set[str] = set()
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or len(line) > 400:
            continue
        # Skip obvious markdown chrome (images, badge rows, table separators).
        if line.startswith(("![", "<img", "|---", "| ---")) or set(line) <= {"|", "-", " ", ":"}:
            continue
        if SAMPLING_PATTERN.search(line):
            clean = re.sub(r"\s+", " ", line.lstrip("#-*> ").strip())
            if clean and clean not in seen:
                seen.add(clean)
                sampling.append(clean)
            continue
        if KEEP_PATTERNS.search(line):
            clean = re.sub(r"\s+", " ", line.lstrip("#-*> ").strip())
            if clean and clean not in seen:
                seen.add(clean)
                kept.append(clean)
    return kept, sampling


def fetch_card(repo: str, want_base: bool = True) -> dict:
    """Fetch and distill a model card. Always returns a dict (never raises)."""
    result: dict = {"repo": repo, "ok": False, "notes": []}

    api_body, api_err = _get(API_URL.format(repo=repo))
    if api_body:
        try:
            api = json.loads(api_body)
        except json.JSONDecodeError:
            api = {}
        gguf = api.get("gguf", {}) or {}
        card = api.get("cardData", {}) or {}
        result["facts"] = {
            "architecture": gguf.get("architecture"),
            "context_length": gguf.get("context_length"),
            "has_chat_template": bool(gguf.get("chat_template")),
            "base_model": card.get("base_model"),
            "license": card.get("license"),
            "tags": api.get("tags", [])[:12],
        }
        result["gguf_files"] = [
            s["rfilename"]
            for s in api.get("siblings", [])
            if s.get("rfilename", "").endswith(".gguf")
        ][:30]
        result["ok"] = True
    else:
        result["notes"].append(f"API metadata unavailable: {api_err}")

    readme, rd_err = _get(RAW_README_URL.format(repo=repo))
    if readme:
        kept, sampling = extract_recommendations(readme)
        result["recommendations"] = kept
        result["sampling_ignored"] = sampling
        result["ok"] = True
    else:
        result["notes"].append(f"card prose unavailable: {rd_err}")

    # Resolve the base model for architecture facts when the GGUF is a quant repo.
    base = (result.get("facts") or {}).get("base_model")
    if want_base and base:
        base_repo = base[0] if isinstance(base, list) else base
        if base_repo and base_repo != repo:
            result["base_repo"] = base_repo

    return result


def _print_human(result: dict) -> None:
    repo = result["repo"]
    facts = result.get("facts") or {}
    if not result.get("ok"):
        notes = "; ".join(result.get("notes", [])) or "no data"
        print(
            f"Model card for `{repo}` could not be fetched ({notes}). "
            "Continuing without author config hints — derive model facts from "
            "`/props` and the GGUF filename instead, and mark Step 3 PARTIAL."
        )
        return

    print(f"Model card: {repo}")
    if facts:
        line = []
        if facts.get("architecture"):
            line.append(f"arch={facts['architecture']}")
        if facts.get("context_length"):
            line.append(f"ctx_train={facts['context_length']}")
        if facts.get("base_model"):
            bm = facts["base_model"]
            line.append(f"base={bm[0] if isinstance(bm, list) else bm}")
        if facts.get("license"):
            line.append(f"license={facts['license']}")
        if line:
            print("  facts: " + ", ".join(line))
        if facts.get("has_chat_template"):
            print("  chat_template: present in GGUF metadata")

    recs = result.get("recommendations") or []
    if recs:
        print(f"  author config notes ({len(recs)} — feed into Step 4 as candidates):")
        for r in recs[:25]:
            print(f"    • {r}")
    else:
        print("  author config notes: none found in card prose")

    sampling = result.get("sampling_ignored") or []
    if sampling:
        print("  sampling params in card (IGNORED — this skill never tunes sampling):")
        for s in sampling[:6]:
            print(f"    · {s}")

    if result.get("base_repo"):
        print(
            f"  base model: {result['base_repo']} "
            "(re-run with --repo for architecture/ctx_train ground truth if the quant card is thin)"
        )

    for note in result.get("notes", []):
        print(f"  note: {note}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="fetch_model_card.py",
        description=(
            "Best-effort fetch of a Hugging Face model card (prose + structured "
            "facts) for the llamacpp-tuner Step 3. Never blocks the workflow: a "
            "missing card prints a note and exits 0."
        ),
    )
    ap.add_argument(
        "--repo", default=None, help="HF repo id, e.g. unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Path to the loaded .gguf (repo id is parsed from the HF cache path)",
    )
    ap.add_argument(
        "--no-base", action="store_true", help="Do not resolve the upstream base_model repo"
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human summary",
    )
    args = ap.parse_args(argv)

    repo = resolve_repo(args.repo, args.model)
    if not repo:
        # Bad invocation: we have nothing to fetch. Tell the caller how to fix it.
        print(
            "fetch_model_card: could not determine a Hugging Face repo id.\n"
            "  Pass --repo <org/name> (find it from the running server's "
            "/v1/models or /props), or --model <path-to-gguf> if it lives in "
            "the HF cache (.../models--org--name/...).",
            file=sys.stderr,
        )
        return 2

    result = fetch_card(repo, want_base=not args.no_base)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    # Best-effort contract: a miss is not a failure. Always exit 0 here so
    # Step 3 continues; the printed note tells the agent to mark it PARTIAL.
    return 0


if __name__ == "__main__":
    sys.exit(main())
