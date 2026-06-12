"""
Tests for the llamacpp-tuner skill's Step 3 model-card fetcher
(skills/llamacpp-tuner/scripts/fetch_model_card.py).

The script is best-effort by design: a missing card, a 404, a gated repo, or
no network must never stop the tuner — it prints a one-line note and exits 0.
A bad invocation (neither --repo nor --model) exits 2.

The pure logic (repo resolution from an HF cache path, frontmatter stripping,
and perf-vs-sampling line extraction) is exercised directly with no network.
``fetch_card`` is covered by monkeypatching the module's ``_get`` so the HTTP
layer is never touched. The CLI contract (exit 2 / exit 0) is exercised as a
subprocess, mirroring the real argparse path the SKILL.md workflow uses, the
same way test_tuner_preflight.py does.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "skills" / "llamacpp-tuner" / "scripts" / "fetch_model_card.py"


def _load_module():
    """Load fetch_model_card.py by file path (it isn't an importable package)."""
    spec = importlib.util.spec_from_file_location("fetch_model_card", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fmc = _load_module()


# --------------------------------------------------------------------------- #
# resolve_repo                                                                 #
# --------------------------------------------------------------------------- #
class TestResolveRepo:
    def test_explicit_repo_wins_over_model_path(self):
        # An explicit --repo must take priority over any path parsing.
        out = fmc.resolve_repo("Org/Name", "/x/models--a--b/snapshots/c/d.gguf")
        assert out == "Org/Name"

    def test_explicit_repo_is_trimmed(self):
        assert fmc.resolve_repo("  unsloth/Foo-GGUF/ ", None) == "unsloth/Foo-GGUF"

    def test_hf_cache_snapshot_path(self):
        path = (
            "/home/u/.cache/huggingface/hub/"
            "models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/"
            "snapshots/abc123/Qwen3.gguf"
        )
        assert fmc.resolve_repo(None, path) == "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"

    def test_bare_cache_dir_without_snapshots(self):
        assert fmc.resolve_repo(None, "/x/models--bartowski--Foo-GGUF") == "bartowski/Foo-GGUF"

    def test_blobs_path_resolves(self):
        path = "/cache/models--org--Some-Model/blobs/deadbeef"
        assert fmc.resolve_repo(None, path) == "org/Some-Model"

    def test_non_cache_path_returns_none(self):
        assert fmc.resolve_repo(None, "/some/random/model.gguf") is None

    def test_no_inputs_returns_none(self):
        assert fmc.resolve_repo(None, None) is None


# --------------------------------------------------------------------------- #
# _strip_frontmatter                                                           #
# --------------------------------------------------------------------------- #
class TestStripFrontmatter:
    def test_drops_closed_frontmatter_block(self):
        md = "---\nlicense: apache-2.0\ntags: [a, b]\n---\n# Title\nbody"
        out = fmc._strip_frontmatter(md)
        assert "license:" not in out
        assert "# Title" in out
        assert "body" in out

    def test_leaves_body_without_frontmatter_untouched(self):
        md = "# Title\nno frontmatter here"
        assert fmc._strip_frontmatter(md) == md

    def test_unclosed_frontmatter_is_returned_as_is(self):
        # No closing '---' -> nothing is stripped (avoids eating the whole card).
        md = "---\nlicense: apache-2.0\nstill going"
        assert fmc._strip_frontmatter(md) == md


# --------------------------------------------------------------------------- #
# extract_recommendations                                                      #
# --------------------------------------------------------------------------- #
class TestExtractRecommendations:
    SAMPLE = (
        "---\n"
        "license: apache-2.0\n"
        "---\n"
        "# Qwen3 Coder\n"
        "![banner](x.png)\n"
        "| --- | --- |\n"
        "Context Length: 262144 natively.\n"
        "If you hit out-of-memory, reduce the context to 32768.\n"
        "Recommended output length: 65536 tokens.\n"
        "Use temperature 0.7 and top_p 0.8 for best results.\n"
        "Run: llama-server --ctx-size 32768 --flash-attn\n"
        "Just some unrelated prose about the weather.\n"
    )

    def test_perf_lines_kept(self):
        kept, _ = fmc.extract_recommendations(self.SAMPLE)
        joined = "\n".join(kept)
        assert "Context Length: 262144 natively." in kept
        assert any("out-of-memory" in k for k in kept)
        assert any("llama-server --ctx-size" in k for k in kept)
        # Unrelated prose and markdown chrome are dropped.
        assert "weather" not in joined
        assert not any(k.startswith("![") for k in kept)

    def test_sampling_lines_separated_not_kept(self):
        kept, sampling = fmc.extract_recommendations(self.SAMPLE)
        assert any("temperature" in s for s in sampling)
        # A sampling line must never leak into the actionable "kept" bucket.
        assert not any("temperature" in k for k in kept)

    def test_frontmatter_is_stripped_before_extraction(self):
        kept, sampling = fmc.extract_recommendations(self.SAMPLE)
        assert not any("license" in line for line in kept + sampling)

    def test_dedup_identical_lines(self):
        md = "context length 262144\ncontext length 262144\ncontext length 262144\n"
        kept, _ = fmc.extract_recommendations(md)
        assert kept.count("context length 262144") == 1

    def test_overlong_lines_skipped(self):
        md = "context length " + ("x" * 500) + "\n"
        kept, _ = fmc.extract_recommendations(md)
        assert kept == []

    def test_empty_card_yields_empty_buckets(self):
        assert fmc.extract_recommendations("") == ([], [])


# --------------------------------------------------------------------------- #
# fetch_card (network layer monkeypatched)                                     #
# --------------------------------------------------------------------------- #
class TestFetchCard:
    def _patch_get(self, monkeypatch, responses):
        """Stub ``_get`` with a {url-substring: (body, err)} routing table."""

        def fake_get(url):
            for needle, value in responses.items():
                if needle in url:
                    return value
            return None, "HTTP 404"

        monkeypatch.setattr(fmc, "_get", fake_get)

    def test_happy_path_populates_facts_and_recs(self, monkeypatch):
        api = {
            "gguf": {
                "architecture": "qwen3moe",
                "context_length": 262144,
                "chat_template": "{{...}}",
            },
            "cardData": {
                "base_model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "license": "apache-2.0",
            },
            "tags": ["text-generation"],
            "siblings": [
                {"rfilename": "model-Q4_K_M.gguf"},
                {"rfilename": "README.md"},
            ],
        }
        readme = "# Card\nContext Length: 262144 natively.\nUse temperature 0.7.\n"
        self._patch_get(
            monkeypatch,
            {"api/models/": (json.dumps(api), None), "/raw/main/README.md": (readme, None)},
        )

        result = fmc.fetch_card("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF")

        assert result["ok"] is True
        assert result["facts"]["architecture"] == "qwen3moe"
        assert result["facts"]["context_length"] == 262144
        assert result["facts"]["has_chat_template"] is True
        assert result["gguf_files"] == ["model-Q4_K_M.gguf"]
        assert any("Context Length" in r for r in result["recommendations"])
        assert any("temperature" in s for s in result["sampling_ignored"])
        # Quant repo -> upstream base resolved for ground-truth re-fetch.
        assert result["base_repo"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        assert result["notes"] == []

    def test_total_miss_is_best_effort_not_ok(self, monkeypatch):
        # Both endpoints fail -> ok=False, notes populated, but no exception.
        self._patch_get(
            monkeypatch,
            {"api/models/": (None, "HTTP 401"), "/raw/main/README.md": (None, "HTTP 401")},
        )
        result = fmc.fetch_card("nobody/nope")
        assert result["ok"] is False
        assert len(result["notes"]) == 2
        assert "base_repo" not in result

    def test_api_only_still_ok(self, monkeypatch):
        api = {"gguf": {"architecture": "llama"}, "cardData": {}, "siblings": []}
        self._patch_get(
            monkeypatch,
            {"api/models/": (json.dumps(api), None), "/raw/main/README.md": (None, "HTTP 404")},
        )
        result = fmc.fetch_card("org/name")
        assert result["ok"] is True
        assert result["facts"]["architecture"] == "llama"
        assert any("card prose unavailable" in n for n in result["notes"])

    def test_malformed_api_json_does_not_crash(self, monkeypatch):
        self._patch_get(
            monkeypatch,
            {"api/models/": ("{not json", None), "/raw/main/README.md": (None, "HTTP 404")},
        )
        result = fmc.fetch_card("org/name")
        # A non-empty body marks ok=True before parsing; a JSONDecodeError then
        # degrades facts to all-None. The contract that matters: never crashes.
        assert result["facts"]["architecture"] is None
        assert result["facts"]["context_length"] is None
        assert result["gguf_files"] == []

    def test_base_model_as_list_is_unwrapped(self, monkeypatch):
        api = {"gguf": {}, "cardData": {"base_model": ["Qwen/Base", "Other/Thing"]}, "siblings": []}
        self._patch_get(
            monkeypatch,
            {"api/models/": (json.dumps(api), None), "/raw/main/README.md": (None, "HTTP 404")},
        )
        result = fmc.fetch_card("quant/repo")
        assert result["base_repo"] == "Qwen/Base"


# --------------------------------------------------------------------------- #
# CLI contract (subprocess)                                                    #
# --------------------------------------------------------------------------- #
def _run(*args: str, timeout: float = 45) -> subprocess.CompletedProcess[str]:
    # 45s clears the script's worst case (two 15s urllib timeouts) so a slow
    # or unreachable network can't turn the best-effort path into a test hang.
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


class TestCli:
    def test_bad_invocation_exits_two(self):
        proc = _run()  # neither --repo nor --model
        assert proc.returncode == 2
        assert "could not determine a Hugging Face repo id" in proc.stderr

    def test_help_flag_works(self):
        proc = _run("--help")
        assert proc.returncode == 0
        assert "Best-effort fetch of a Hugging Face model card" in proc.stdout

    @pytest.mark.integration
    def test_miss_exits_zero_with_note(self):
        # Network-dependent: a non-existent repo must still exit 0 (best-effort).
        proc = _run("--repo", "this-org/does-not-exist-xyz-99999")
        assert proc.returncode == 0
        assert "could not be fetched" in proc.stdout
