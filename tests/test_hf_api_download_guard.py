"""Issue #205 — constrain GGUF repo resolution and pin the download revision.

Covers the guard rails added to ``claude_codex_local._hf_api``:

* every download passes an explicit ``--revision`` (resolved HEAD sha,
  falling back to the default branch name),
* positional args sit behind a ``--`` separator and hostile components
  (leading ``-``, ``..`` segments, absolute paths) are rejected,
* the resolved destination must stay under ``local_dir``,
* the mirror search fallback only accepts allowlisted authors,
* the wizard records the pinned revision in wizard state.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import claude_codex_local
import claude_codex_local._hf_api as hf_api_mod
from claude_codex_local import wizard_steps
from claude_codex_local.wizard_state import WizardState

pb = claude_codex_local.core


@pytest.fixture(autouse=True)
def _clean_hf_caches():
    hf_api_mod._GGUF_MIRROR_CACHE.clear()
    hf_api_mod._MODEL_PAYLOAD_CACHE.clear()
    yield
    hf_api_mod._GGUF_MIRROR_CACHE.clear()
    hf_api_mod._MODEL_PAYLOAD_CACHE.clear()


@pytest.fixture
def cli_present(monkeypatch):
    monkeypatch.setattr(
        pb, "huggingface_cli_detect", lambda: {"present": True, "binary": "hf", "version": ""}
    )


def _popen_capture(monkeypatch, on_wait=None):
    calls: list[list[str]] = []

    class _FakeProc:
        def __init__(self, cmd, env=None):
            self.cmd = cmd

        def wait(self, timeout=None):
            if on_wait:
                on_wait()
            return 0

    def fake_popen(cmd, env=None):
        calls.append(list(cmd))
        return _FakeProc(cmd, env=env)

    monkeypatch.setattr(pb.subprocess, "Popen", fake_popen)
    return calls


# ---------------------------------------------------------------------------
# Command construction: --separator + revision pinning
# ---------------------------------------------------------------------------


class TestDownloadCommandGuards:
    def test_positional_args_follow_double_dash(
        self, isolated_state, cli_present, monkeypatch, tmp_path
    ):
        """repo_id/filename must come after `--`; flags before it (issue #205)."""
        pb_, _, _ = isolated_state
        monkeypatch.setattr(hf_api_mod, "huggingface_repo_revision", lambda r: None)
        local_dir = str(tmp_path / "dl")
        Path(local_dir).mkdir(parents=True)
        calls = _popen_capture(
            monkeypatch,
            on_wait=lambda: (Path(local_dir) / "m.gguf").write_bytes(b"x"),
        )
        result = pb_.huggingface_download_gguf(
            "org/repo", filename="m.gguf", local_dir=local_dir, stream=True
        )
        assert result["ok"] is True
        cmd = calls[0]
        sep = cmd.index("--")
        assert cmd[sep + 1] == "org/repo"
        assert cmd[sep + 2] == "m.gguf"
        assert len(cmd) == sep + 3
        # flags live before the separator
        assert "--revision" in cmd[:sep]

    def test_revision_resolved_to_head_sha(self, isolated_state, cli_present, monkeypatch):
        pb_, _, _ = isolated_state
        monkeypatch.setattr(hf_api_mod, "huggingface_repo_revision", lambda r: "abc1234")
        captured: list[list[str]] = []
        monkeypatch.setattr(
            pb_,
            "run",
            lambda cmd, **kw: (
                captured.append(list(cmd))
                or subprocess.CompletedProcess(cmd, 0, "/tmp/m.gguf\n", "")
            ),
        )
        result = pb_.huggingface_download_gguf("org/repo", stream=False)
        assert result["ok"] is True
        i = captured[0].index("--revision")
        assert captured[0][i + 1] == "abc1234"
        assert result["revision"] == "abc1234"

    def test_revision_falls_back_to_main_when_unresolvable(
        self, isolated_state, cli_present, monkeypatch
    ):
        pb_, _, _ = isolated_state
        monkeypatch.setattr(hf_api_mod, "huggingface_repo_revision", lambda r: None)
        captured: list[list[str]] = []
        monkeypatch.setattr(
            pb_,
            "run",
            lambda cmd, **kw: (
                captured.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0, "", "")
            ),
        )
        result = pb_.huggingface_download_gguf("org/repo", stream=False)
        assert result["ok"] is True
        i = captured[0].index("--revision")
        assert captured[0][i + 1] == "main"
        assert result["revision"] == "main"

    def test_explicit_revision_used_verbatim_without_resolution(
        self, isolated_state, cli_present, monkeypatch
    ):
        pb_, _, _ = isolated_state

        def _fail(*a, **kw):
            raise AssertionError("revision must not be resolved when caller pins one")

        monkeypatch.setattr(hf_api_mod, "huggingface_repo_revision", _fail)
        captured: list[list[str]] = []
        monkeypatch.setattr(
            pb_,
            "run",
            lambda cmd, **kw: (
                captured.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0, "", "")
            ),
        )
        result = pb_.huggingface_download_gguf("org/repo", stream=False, revision="v1.0")
        assert result["ok"] is True
        i = captured[0].index("--revision")
        assert captured[0][i + 1] == "v1.0"


# ---------------------------------------------------------------------------
# Component rejection: leading '-', '..' segments, absolute paths
# ---------------------------------------------------------------------------


class TestDownloadComponentRejection:
    def test_rejects_leading_dash_repo_id(self, isolated_state, monkeypatch):
        pb_, _, _ = isolated_state
        result = pb_.huggingface_download_gguf("-evil/repo", stream=False)
        assert result["ok"] is False
        assert "repo_id" in result["error"]

    def test_rejects_leading_dash_filename(self, isolated_state, monkeypatch):
        pb_, _, _ = isolated_state
        result = pb_.huggingface_download_gguf("org/repo", filename="--help", stream=False)
        assert result["ok"] is False
        assert "filename" in result["error"]

    def test_rejects_traversal_filename(self, isolated_state, monkeypatch):
        pb_, _, _ = isolated_state
        result = pb_.huggingface_download_gguf("org/repo", filename="../evil.gguf", stream=False)
        assert result["ok"] is False
        assert "filename" in result["error"]
        assert ".." in result["error"]

    def test_rejects_absolute_filename(self, isolated_state, monkeypatch):
        pb_, _, _ = isolated_state
        result = pb_.huggingface_download_gguf("org/repo", filename="/etc/shadow", stream=False)
        assert result["ok"] is False
        assert "absolute" in result["error"]

    def test_rejects_unsafe_include(self, isolated_state, monkeypatch):
        pb_, _, _ = isolated_state
        result = pb_.huggingface_download_gguf("org/repo", include="-o/etc/x", stream=False)
        assert result["ok"] is False
        assert "include" in result["error"]

    def test_rejects_unsafe_revision(self, isolated_state, cli_present, monkeypatch):
        pb_, _, _ = isolated_state
        result = pb_.huggingface_download_gguf("org/repo", revision="--rebase", stream=False)
        assert result["ok"] is False
        assert "revision" in result["error"]

    def test_no_subprocess_spawned_on_rejection(self, isolated_state, monkeypatch):
        pb_, _, _ = isolated_state
        monkeypatch.setattr(pb_, "run", lambda *a, **kw: pytest.fail("must not spawn"))
        result = pb_.huggingface_download_gguf("../escape", stream=False)
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Destination containment under local_dir
# ---------------------------------------------------------------------------


class TestDownloadPathContainment:
    def test_symlinked_destination_outside_local_dir_rejected(
        self, isolated_state, cli_present, monkeypatch, tmp_path
    ):
        """Pre-existing symlink pointing outside local_dir must abort pre-spawn."""
        pb_, _, _ = isolated_state
        monkeypatch.setattr(hf_api_mod, "huggingface_repo_revision", lambda r: None)
        local_dir = tmp_path / "dl"
        local_dir.mkdir()
        outside = tmp_path / "outside.gguf"
        outside.write_bytes(b"secret")
        link = local_dir / "link.gguf"
        link.symlink_to(outside)
        spawned = _popen_capture(monkeypatch)
        result = pb_.huggingface_download_gguf(
            "org/repo", filename="link.gguf", local_dir=str(local_dir), stream=True
        )
        assert result["ok"] is False
        assert "escapes local_dir" in result["error"]
        assert spawned == []

    def test_nested_destination_under_local_dir_allowed(
        self, isolated_state, cli_present, monkeypatch, tmp_path
    ):
        pb_, _, _ = isolated_state
        monkeypatch.setattr(hf_api_mod, "huggingface_repo_revision", lambda r: None)
        local_dir = tmp_path / "dl"
        local_dir.mkdir()

        def _write():
            sub = local_dir / "sub"
            sub.mkdir(parents=True, exist_ok=True)
            (sub / "m.gguf").write_bytes(b"gguf")

        _popen_capture(monkeypatch, on_wait=_write)
        result = pb_.huggingface_download_gguf(
            "org/repo", filename="sub/m.gguf", local_dir=str(local_dir), stream=True
        )
        assert result["ok"] is True
        assert result["path"] == str(local_dir / "sub" / "m.gguf")


# ---------------------------------------------------------------------------
# huggingface_repo_revision
# ---------------------------------------------------------------------------


class _FakeHttpResp:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestHuggingfaceRepoRevision:
    def test_returns_sha_from_model_payload(self, isolated_state, monkeypatch):
        import urllib.request

        pb_, _, _ = isolated_state
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeHttpResp({"sha": "cafebabe", "siblings": []}),
        )
        assert pb_.huggingface_repo_revision("org/repo") == "cafebabe"

    def test_none_when_network_fails(self, isolated_state, monkeypatch):
        import urllib.request

        pb_, _, _ = isolated_state

        def _raise(*a, **kw):
            raise OSError("no network")

        monkeypatch.setattr(urllib.request, "urlopen", _raise)
        assert pb_.huggingface_repo_revision("org/repo") is None

    def test_none_on_malformed_payload(self, isolated_state, monkeypatch):
        import urllib.request

        pb_, _, _ = isolated_state
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: _FakeHttpResp([1]))
        assert pb_.huggingface_repo_revision("org/repo") is None

    def test_list_repo_files_still_works_via_shared_payload(self, isolated_state, monkeypatch):
        import urllib.request

        pb_, _, _ = isolated_state
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **kw: _FakeHttpResp({"sha": "beef", "siblings": [{"rfilename": "m.gguf"}]}),
        )
        assert pb_.huggingface_list_repo_files("org/repo") == ["m.gguf"]


# ---------------------------------------------------------------------------
# Mirror search fallback constrained to the author allowlist
# ---------------------------------------------------------------------------


class TestMirrorSearchAllowlist:
    def test_non_allowlisted_search_hit_is_ignored(self, isolated_state, monkeypatch):
        pb_, _, _ = isolated_state
        probed: list[str] = []

        def _has_gguf(repo):
            probed.append(repo)
            return repo.startswith("bartowski/")

        monkeypatch.setattr(claude_codex_local._hf_api, "huggingface_repo_has_gguf", _has_gguf)
        monkeypatch.setattr(
            claude_codex_local._hf_api,
            "huggingface_search_models",
            lambda q, limit=10, **kw: ["attacker/Llama-3-GGUF", "bartowski/Llama-3-GGUF"],
        )
        resolved = pb_.resolve_gguf_mirror("Some/Llama-3-MLX-4bit")
        # The attacker repo is never even probed; the allowlisted hit wins.
        assert resolved == "bartowski/Llama-3-GGUF"
        assert all(not p.startswith("attacker/") for p in probed)

    def test_all_non_allowlisted_hits_yield_none(self, isolated_state, monkeypatch):
        pb_, _, _ = isolated_state
        # Only hostile authors host GGUFs here; the allowlist filter must skip
        # them without probing, so nothing resolves.
        monkeypatch.setattr(
            claude_codex_local._hf_api,
            "huggingface_repo_has_gguf",
            lambda repo: repo.startswith(("evil/", "also-evil/")),
        )
        monkeypatch.setattr(
            claude_codex_local._hf_api,
            "huggingface_search_models",
            lambda q, limit=10, **kw: ["evil/A-GGUF", "also-evil/B-GGUF"],
        )
        assert pb_.resolve_gguf_mirror("Foo/Bar-MLX-4bit") is None

    def test_allowlisted_author_still_resolves_via_search(self, isolated_state, monkeypatch):
        pb_, _, _ = isolated_state
        # No {author}/{base}-GGUF probe succeeds; only search finds unsloth's
        # unconventionally named repo.
        monkeypatch.setattr(
            claude_codex_local._hf_api,
            "huggingface_repo_has_gguf",
            lambda repo: repo == "unsloth/Llama-3-vLLM-GGUF",
        )
        monkeypatch.setattr(
            claude_codex_local._hf_api,
            "huggingface_search_models",
            lambda q, limit=10, **kw: ["unsloth/Llama-3-vLLM-GGUF"],
        )
        assert pb_.resolve_gguf_mirror("Llama-3") == "unsloth/Llama-3-vLLM-GGUF"


# ---------------------------------------------------------------------------
# Wizard persists the pinned revision in state (#205)
# ---------------------------------------------------------------------------


class TestWizardRecordsRevision:
    def test_download_model_impl_persists_revision(self, isolated_state, monkeypatch):
        pb_, wiz, _ = isolated_state
        monkeypatch.setattr(pb_, "huggingface_cli_detect", lambda: {"present": True})
        monkeypatch.setattr(pb_, "huggingface_list_repo_files", lambda repo: ["model-Q4_K_M.gguf"])

        def _fake_download(repo_id, filename=None, local_dir=None, *, include=None, stream=True):
            target = Path(local_dir) / "model-Q4_K_M.gguf"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x00" * 64)
            return {
                "ok": True,
                "path": str(target),
                "error": None,
                "bytes_downloaded": 64,
                "elapsed_seconds": 0.5,
                "not_found": False,
                "revision": "deadbeeef05e",
            }

        monkeypatch.setattr(pb_, "huggingface_download_gguf", _fake_download)
        monkeypatch.setattr(pb_, "machine_profile", lambda: {})

        state = WizardState(primary_engine="llamacpp", engine_model_tag="org/repo")
        assert wizard_steps._download_model_impl(state) is True
        assert state.profile["llamacpp_model_revision"] == "deadbeeef05e"

    def test_download_model_impl_omits_key_when_revision_missing(self, isolated_state, monkeypatch):
        pb_, wiz, _ = isolated_state
        monkeypatch.setattr(pb_, "huggingface_cli_detect", lambda: {"present": True})
        monkeypatch.setattr(pb_, "huggingface_list_repo_files", lambda repo: ["model-Q4_K_M.gguf"])

        def _fake_download(repo_id, filename=None, local_dir=None, *, include=None, stream=True):
            return {
                "ok": True,
                "path": "/tmp/model-Q4_K_M.gguf",
                "error": None,
                "bytes_downloaded": 64,
                "elapsed_seconds": 0.5,
                "not_found": False,
            }

        monkeypatch.setattr(pb_, "huggingface_download_gguf", _fake_download)
        monkeypatch.setattr(pb_, "machine_profile", lambda: {})

        state = WizardState(primary_engine="llamacpp", engine_model_tag="org/repo")
        assert wizard_steps._download_model_impl(state) is True
        assert "llamacpp_model_revision" not in state.profile
