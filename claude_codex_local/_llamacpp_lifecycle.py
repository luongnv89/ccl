from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_codex_local._config import (
    LLAMACPP_API_KEY,
    LLAMACPP_CTX_SIZE,
    LLAMACPP_DEFAULT_SPEC_DRAFT_N_MAX,
    LLAMACPP_LOG_DIR,
    LLAMACPP_MTP_ENABLED,
    LLAMACPP_N_GPU_LAYERS,
    LLAMACPP_PID_DIR,
    LLAMACPP_SERVER_HOST,
    LLAMACPP_SERVER_PORT,
    LLAMACPP_SPEC_DRAFT_N_MAX,
    LLAMACPP_THREADS,
    _is_local_base_url,
)
from claude_codex_local._shell import _auth_headers, command_version, ensure_path, llamacpp_base_url


def llamacpp_detect() -> dict[str, Any]:
    for candidate in ("llama-server", "llama-cpp-server", "server"):
        info = command_version(candidate, ["--version"])
        if info.get("present"):
            if candidate == "server" and "llama" not in info.get("version", "").lower():
                continue
            return {"present": True, "binary": candidate, "version": info.get("version", "")}
    return {"present": False, "version": ""}


def llamacpp_info() -> dict[str, Any]:
    import urllib.error
    import urllib.request

    base_url = llamacpp_base_url()
    is_remote = not _is_local_base_url(base_url)
    if is_remote:
        detect: dict[str, Any] = {"present": False, "binary": "", "version": ""}
    else:
        detect = llamacpp_detect()
    base: dict[str, Any] = {
        "present": detect.get("present", False),
        "binary": detect.get("binary", ""),
        "server_running": False,
        "server_port": LLAMACPP_SERVER_PORT,
        "base_url": base_url,
        "model": None,
        "remote": is_remote,
    }

    health_url = f"{base_url}/health"
    req = urllib.request.Request(health_url, headers=_auth_headers(LLAMACPP_API_KEY), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            base["server_running"] = resp.status in (200, 503)
    except (urllib.error.URLError, OSError):
        return base
    except Exception:
        return base

    if base["server_running"]:
        base["present"] = True

    base["models"] = []

    models_url = f"{base_url}/v1/models"
    req = urllib.request.Request(models_url, headers=_auth_headers(LLAMACPP_API_KEY), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            body = json.loads(resp.read())
            models_data = body.get("data", [])
            base["model"] = models_data[0]["id"] if models_data else None
            base["models"] = [
                {"name": m["id"], "format": "gguf", "local": not base.get("remote", False)}
                for m in models_data
                if isinstance(m, dict) and m.get("id")
            ]
    except (urllib.error.URLError, OSError):
        pass
    except Exception:
        pass
    return base


@dataclass
class LlamaServerHandle:
    pid: int
    port: int
    host: str
    model_path: str
    argv: list[str]
    log_path: str
    pid_file: str
    we_started_it: bool = True
    proc: subprocess.Popen[bytes] | None = field(default=None, repr=False, compare=False)


def safe_repo_slug(repo_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", repo_id.strip())
    return cleaned.strip("-") or "model"


def detect_llamacpp_gpu_offload(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    if LLAMACPP_N_GPU_LAYERS is not None:
        try:
            n = int(LLAMACPP_N_GPU_LAYERS)
            return {"n_gpu_layers": n, "kind": "env-override", "reason": "LLAMACPP_N_GPU_LAYERS"}
        except ValueError:
            pass

    if profile:
        sys_info = profile.get("llmfit_system") or {}
        sys_info = sys_info.get("system", sys_info) if isinstance(sys_info, dict) else {}
        if isinstance(sys_info, dict) and sys_info.get("has_gpu"):
            gpu_name = str(sys_info.get("gpu_name") or "").lower()
            if "apple" in gpu_name or "metal" in gpu_name or gpu_name.startswith("m"):
                return {
                    "n_gpu_layers": -1,
                    "kind": "metal",
                    "reason": f"llmfit detected {gpu_name}",
                }
            return {"n_gpu_layers": -1, "kind": "gpu", "reason": f"llmfit detected {gpu_name}"}

    machine = platform.machine().lower()
    if platform.system() == "Darwin" and machine in ("arm64", "aarch64"):
        return {"n_gpu_layers": -1, "kind": "metal", "reason": "Apple Silicon detected"}

    if shutil.which("nvidia-smi"):
        return {"n_gpu_layers": -1, "kind": "cuda", "reason": "nvidia-smi found on PATH"}

    return {"n_gpu_layers": 0, "kind": "cpu", "reason": "no GPU detected"}


def detect_llamacpp_threads(profile: dict[str, Any] | None = None) -> int:
    if LLAMACPP_THREADS is not None:
        try:
            n = int(LLAMACPP_THREADS)
            if n > 0:
                return n
        except ValueError:
            pass
    if profile:
        sys_info = profile.get("llmfit_system") or {}
        sys_info = sys_info.get("system", sys_info) if isinstance(sys_info, dict) else {}
        if isinstance(sys_info, dict):
            cores = sys_info.get("cpu_cores")
            if isinstance(cores, int) and cores > 0:
                return min(cores, 16)
    cpu = os.cpu_count() or 4
    return min(cpu, 16)


def probe_gguf_is_mtp(model_path: str | Path) -> dict[str, Any]:
    GGUF_MAGIC = b"GGUF"
    T_STRING = 8
    T_ARRAY = 9
    SCALAR_SIZE = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }
    MAX_KEY_LEN = 512
    MAX_STR_LEN = 4096
    MAX_KV_COUNT = 8192
    MAX_ARRAY_LEN = 65536

    try:
        path = Path(model_path)
    except (TypeError, OSError) as exc:
        return {"is_mtp": False, "reason": f"path-error: {exc}"}
    if not path.is_file():
        return {"is_mtp": False, "reason": "file-not-found"}

    try:
        with path.open("rb") as fh:
            magic = fh.read(4)
            if magic != GGUF_MAGIC:
                return {"is_mtp": False, "reason": "not-gguf"}
            head = fh.read(4 + 8 + 8)
            if len(head) < 20:
                return {"is_mtp": False, "reason": "truncated-header"}
            _version, _tensor_count, kv_count = struct.unpack("<IQQ", head)
            if _version < 2:
                return {"is_mtp": False, "reason": f"unsupported-gguf-version:{_version}"}
            raw_kv_count = int(kv_count)
            kv_count = min(raw_kv_count, MAX_KV_COUNT)
            truncated = raw_kv_count > MAX_KV_COUNT

            def read_string(limit: int) -> str:
                hdr = fh.read(8)
                if len(hdr) < 8:
                    raise ValueError("truncated string length")
                (slen,) = struct.unpack("<Q", hdr)
                if slen > limit:
                    fh.seek(int(slen), os.SEEK_CUR)
                    return ""
                data = fh.read(int(slen))
                if len(data) < slen:
                    raise ValueError("truncated string body")
                return data.decode("utf-8", errors="replace")

            def skip_value(vtype: int) -> None:
                size = SCALAR_SIZE.get(vtype)
                if size is not None:
                    fh.seek(size, os.SEEK_CUR)
                    return
                if vtype == T_STRING:
                    hdr = fh.read(8)
                    if len(hdr) < 8:
                        raise ValueError("truncated string skip")
                    (slen,) = struct.unpack("<Q", hdr)
                    fh.seek(int(slen), os.SEEK_CUR)
                    return
                if vtype == T_ARRAY:
                    arr_hdr = fh.read(12)
                    if len(arr_hdr) < 12:
                        raise ValueError("truncated array header")
                    elem_type, arr_len = struct.unpack("<IQ", arr_hdr)
                    elem_type = int(elem_type)
                    arr_len = int(arr_len)
                    if elem_type == T_ARRAY:
                        raise ValueError("nested-array forbidden by gguf spec")
                    elem_size = SCALAR_SIZE.get(elem_type)
                    if elem_size is not None:
                        fh.seek(elem_size * arr_len, os.SEEK_CUR)
                        return
                    if elem_type == T_STRING:
                        if arr_len > MAX_ARRAY_LEN:
                            raise ValueError(f"oversized-string-array: {arr_len} > {MAX_ARRAY_LEN}")
                        for _ in range(arr_len):
                            skip_value(T_STRING)
                        return
                    raise ValueError(f"unknown array element type: {elem_type}")
                raise ValueError(f"unknown gguf value type: {vtype}")

            for _ in range(kv_count):
                key = read_string(MAX_KEY_LEN)
                vtype_buf = fh.read(4)
                if len(vtype_buf) < 4:
                    return {"is_mtp": False, "reason": "truncated-value-type"}
                (vtype,) = struct.unpack("<I", vtype_buf)
                key_lc = key.lower()
                if ".mtp." in key_lc or key_lc.startswith("mtp."):
                    return {"is_mtp": True, "reason": f"metadata-key:{key}"}
                if key_lc.endswith(".nextn_predict_layers"):
                    return {"is_mtp": True, "reason": f"nextn-key:{key}"}
                if vtype == T_STRING and key_lc in {"general.architecture", "general.name"}:
                    val = read_string(MAX_STR_LEN)
                    if "mtp" in val.lower():
                        return {"is_mtp": True, "reason": f"{key}={val}"}
                else:
                    skip_value(int(vtype))
            return {
                "is_mtp": False,
                "reason": "scanned-no-mtp-truncated" if truncated else "scanned-no-mtp",
            }
    except (ValueError, OSError, struct.error) as exc:
        return {"is_mtp": False, "reason": f"probe-failed: {exc}"}


def detect_llamacpp_mtp(
    model_path: str | Path,
    *,
    extra_argv: list[str] | None = None,
) -> dict[str, Any]:
    raw = LLAMACPP_SPEC_DRAFT_N_MAX
    spec_n = LLAMACPP_DEFAULT_SPEC_DRAFT_N_MAX
    notes: list[str] = []
    if raw is not None:
        try:
            candidate = int(raw)
            if 1 <= candidate <= 16:
                spec_n = candidate
            else:
                notes.append(
                    f"LLAMACPP_SPEC_DRAFT_N_MAX={raw!r} out of range 1-16; "
                    f"using default {LLAMACPP_DEFAULT_SPEC_DRAFT_N_MAX}"
                )
        except (TypeError, ValueError):
            notes.append(
                f"LLAMACPP_SPEC_DRAFT_N_MAX={raw!r} is not an integer; "
                f"using default {LLAMACPP_DEFAULT_SPEC_DRAFT_N_MAX}"
            )

    decided_source: str | None = None
    decided_enabled: bool | None = None

    env_raw = LLAMACPP_MTP_ENABLED
    if env_raw is not None:
        token = env_raw.strip().lower()
        if token in {"0", "false", "off", "no"}:
            return {
                "enabled": False,
                "spec_draft_n_max": spec_n,
                "source": "env-override",
                "warning": None,
                "notes": notes,
            }
        if token in {"1", "true", "on", "yes"}:
            decided_enabled = True
            decided_source = "env-override"

    if decided_enabled is None:
        probe = probe_gguf_is_mtp(model_path)
        if probe.get("is_mtp"):
            decided_enabled = True
            decided_source = "gguf-metadata"
        else:
            name = Path(model_path).name
            if re.search(r"(?:^|[^a-z0-9])mtp(?:[^a-z0-9]|$)", name, re.IGNORECASE):
                decided_enabled = True
                decided_source = "filename"

    if not decided_enabled:
        return {
            "enabled": False,
            "spec_draft_n_max": spec_n,
            "source": "disabled",
            "warning": None,
            "notes": notes,
        }

    if extra_argv:
        argv = list(extra_argv)
        for i, tok in enumerate(argv):
            if tok == "--mmproj":
                return {
                    "enabled": False,
                    "spec_draft_n_max": None,
                    "source": "conflict",
                    "warning": "MTP disabled: --mmproj is not yet supported with --spec-type draft-mtp",
                    "notes": notes,
                }
            if tok in ("-np", "--parallel") and i + 1 < len(argv):
                try:
                    n_par = int(argv[i + 1])
                except ValueError:
                    continue
                if n_par > 1:
                    return {
                        "enabled": False,
                        "spec_draft_n_max": None,
                        "source": "conflict",
                        "warning": f"MTP disabled: -np {n_par} (>1) is not yet supported with --spec-type draft-mtp",
                        "notes": notes,
                    }

    assert decided_source is not None
    return {
        "enabled": True,
        "spec_draft_n_max": spec_n,
        "source": decided_source,
        "warning": None,
        "notes": notes,
    }


@dataclass
class LlamaServerConfig:
    binary: str = ""
    model_path: str = ""
    port: int = LLAMACPP_SERVER_PORT
    host: str = LLAMACPP_SERVER_HOST
    ctx_size: int = LLAMACPP_CTX_SIZE
    n_gpu_layers: int = 0
    threads: int = 4
    mtp: dict[str, Any] | None = None
    extra_argv: list[str] | None = None

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "binary": self.binary,
            "model_path": self.model_path,
            "port": self.port,
            "host": self.host,
            "ctx_size": self.ctx_size,
            "n_gpu_layers": self.n_gpu_layers,
            "threads": self.threads,
            "mtp": self.mtp,
            "extra_argv": self.extra_argv,
        }


def build_llamacpp_server_args(
    config: LlamaServerConfig | None = None,
    *,
    binary: str | None = None,
    model_path: str | None = None,
    port: int | None = None,
    host: str | None = None,
    ctx_size: int | None = None,
    n_gpu_layers: int | None = None,
    threads: int | None = None,
    mtp: dict[str, Any] | None = None,
    extra_argv: list[str] | None = None,
) -> list[str]:
    if config is not None:
        c_binary = binary if binary is not None else config.binary
        c_model_path = model_path if model_path is not None else config.model_path
        c_port = port if port is not None else config.port
        c_host = host if host is not None else config.host
        c_ctx_size = ctx_size if ctx_size is not None else config.ctx_size
        c_n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else config.n_gpu_layers
        c_threads = threads if threads is not None else config.threads
        c_mtp = mtp if mtp is not None else config.mtp
        c_extra_argv = extra_argv if extra_argv is not None else config.extra_argv
    else:
        c_binary = binary or ""
        c_model_path = model_path or ""
        c_port = port if port is not None else LLAMACPP_SERVER_PORT
        c_host = host if host is not None else LLAMACPP_SERVER_HOST
        c_ctx_size = ctx_size if ctx_size is not None else LLAMACPP_CTX_SIZE
        c_n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else 0
        c_threads = threads if threads is not None else 4
        c_mtp = mtp
        c_extra_argv = extra_argv

    extras = list(c_extra_argv) if c_extra_argv else []
    has_threads = "--threads" in extras
    has_spec_n = "--spec-draft-n-max" in extras
    argv = [
        c_binary,
        "--model",
        c_model_path,
        "--host",
        c_host,
        "--port",
        str(c_port),
        "--ctx-size",
        str(c_ctx_size),
        "--n-gpu-layers",
        str(c_n_gpu_layers),
    ]
    if not has_threads:
        argv += ["--threads", str(c_threads)]
    if c_mtp and c_mtp.get("enabled"):
        spec_n = int(c_mtp.get("spec_draft_n_max", LLAMACPP_DEFAULT_SPEC_DRAFT_N_MAX))
        argv += ["--spec-type", "draft-mtp"]
        if not has_spec_n:
            argv += ["--spec-draft-n-max", str(spec_n)]
    if extras:
        argv += extras
    return argv


def llamacpp_wait_until_ready(
    *,
    port: int = LLAMACPP_SERVER_PORT,
    host: str = LLAMACPP_SERVER_HOST,
    timeout: float = 120.0,
    poll_interval: float = 1.0,
    proc: subprocess.Popen[bytes] | None = None,
) -> bool:
    import time
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        except Exception:
            pass
        time.sleep(poll_interval)
    return False


def diagnose_llama_server_log(log_path: str | Path, *, tail_bytes: int = 16384) -> str | None:
    try:
        path = Path(log_path)
        if not path.is_file():
            return None
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > tail_bytes:
                fh.seek(-tail_bytes, os.SEEK_END)
            data = fh.read()
        text = data.decode("utf-8", errors="replace")
    except OSError:
        return None

    lower = text.lower()

    if "unknown model architecture" in lower:
        match = re.search(r"unknown model architecture[^\n]*?'([^']+)'", text)
        arch = match.group(1) if match else "this model"
        return (
            f"Your llama.cpp build doesn't recognise the model architecture "
            f"({arch}). Update llama.cpp from source (git pull && rebuild) and "
            f"retry, or pick a model GGUF for an architecture this build "
            f"already supports."
        )

    if "missing tensor" in lower:
        arch_match = re.search(r"arch\s*=\s*(\S+)", text)
        arch = arch_match.group(1) if arch_match else "this model's"
        tensor_match = re.search(r"missing tensor '([^']+)'", text)
        tensor = tensor_match.group(1) if tensor_match else None
        ssm_hint = ""
        if tensor and ("ssm" in tensor.lower() or "mamba" in tensor.lower()):
            ssm_hint = (
                " The missing tensor is a state-space (Mamba/SSM) weight, "
                "which means this build predates full support for hybrid "
                "Mamba/Attention models like Qwen3-Next."
            )
        elif tensor:
            ssm_hint = f" Missing tensor: {tensor}."
        return (
            f"Your llama.cpp build is too old for the {arch} architecture: "
            f"it parses the GGUF metadata but the tensor loader is "
            f"incomplete.{ssm_hint} Update llama.cpp from source "
            f"(git pull && rebuild) and retry, or use a GGUF for an "
            f"architecture this build already supports."
        )

    if "out of memory" in lower or "failed to allocate" in lower:
        return (
            "llama-server ran out of memory while loading the model. Try "
            "lowering --n-gpu-layers (set LLAMACPP_N_GPU_LAYERS=0 to force "
            "CPU), reducing --ctx-size, or pick a smaller quantisation."
        )

    return None


def llamacpp_start_server(
    *,
    model_path: str,
    profile: dict[str, Any] | None = None,
    port: int = LLAMACPP_SERVER_PORT,
    host: str = LLAMACPP_SERVER_HOST,
    ctx_size: int = LLAMACPP_CTX_SIZE,
    timeout: float = 120.0,
    extra_argv: list[str] | None = None,
) -> dict[str, Any]:
    base_url = llamacpp_base_url()
    if not _is_local_base_url(base_url):
        return {
            "ok": False,
            "handle": None,
            "argv": [],
            "error": (
                f"refusing to spawn local llama-server: LLAMACPP_BASE_URL "
                f"points to remote host {base_url!r}"
            ),
            "log_path": "",
            "mtp": None,
            "remote": True,
        }
    detect = llamacpp_detect()
    if not detect.get("present"):
        return {
            "ok": False,
            "handle": None,
            "argv": [],
            "error": "llama-server binary not found on PATH",
            "log_path": "",
            "mtp": None,
        }
    binary = shutil.which(detect["binary"]) or detect["binary"]

    if not Path(model_path).is_file():
        return {
            "ok": False,
            "handle": None,
            "argv": [],
            "error": f"model file not found: {model_path}",
            "log_path": "",
            "mtp": None,
        }

    gpu = detect_llamacpp_gpu_offload(profile)
    threads = detect_llamacpp_threads(profile)
    mtp = detect_llamacpp_mtp(model_path, extra_argv=extra_argv)
    server_config = LlamaServerConfig(
        binary=binary,
        model_path=model_path,
        port=port,
        host=host,
        ctx_size=ctx_size,
        n_gpu_layers=int(gpu["n_gpu_layers"]),
        threads=threads,
        mtp=mtp,
        extra_argv=extra_argv,
    )
    argv = build_llamacpp_server_args(config=server_config)

    LLAMACPP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    LLAMACPP_PID_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LLAMACPP_LOG_DIR / f"llama-server-{port}.log"
    pid_file = LLAMACPP_PID_DIR / f"llama-server-{port}.pid"

    popen_kwargs: dict[str, Any] = {
        "env": ensure_path(None),
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True

    try:
        log_handle = open(log_path, "ab", buffering=0)  # noqa: SIM115
    except OSError as exc:
        return {
            "ok": False,
            "handle": None,
            "argv": argv,
            "error": f"could not open log file {log_path}: {exc}",
            "log_path": str(log_path),
            "mtp": mtp,
        }

    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **popen_kwargs,
        )
    except FileNotFoundError as exc:
        with contextlib.suppress(OSError):
            log_handle.close()
        return {
            "ok": False,
            "handle": None,
            "argv": argv,
            "error": f"failed to spawn llama-server: {exc}",
            "log_path": str(log_path),
            "mtp": mtp,
        }
    except OSError as exc:
        log_handle.close()
        return {
            "ok": False,
            "handle": None,
            "argv": argv,
            "error": f"failed to spawn llama-server: {exc}",
            "log_path": str(log_path),
            "mtp": mtp,
        }
    finally:
        with contextlib.suppress(Exception):
            log_handle.close()

    with contextlib.suppress(OSError):
        pid_file.write_text(str(proc.pid))

    handle = LlamaServerHandle(
        pid=proc.pid,
        port=port,
        host=host,
        model_path=model_path,
        argv=argv,
        log_path=str(log_path),
        pid_file=str(pid_file),
        we_started_it=True,
        proc=proc,
    )

    ready = llamacpp_wait_until_ready(port=port, host=host, timeout=timeout, proc=proc)
    if not ready:
        llamacpp_stop_server(handle, grace_seconds=3.0)
        _cleanup_pid_file(str(pid_file))
        if proc.poll() is not None:
            err = (
                f"llama-server exited with status {proc.returncode} during startup; see {log_path}"
            )
        else:
            err = f"server did not become ready within {timeout:.0f}s"
        return {
            "ok": False,
            "handle": None,
            "argv": argv,
            "error": err,
            "hint": diagnose_llama_server_log(log_path),
            "log_path": str(log_path),
            "mtp": mtp,
        }

    if proc.poll() is not None:
        _cleanup_pid_file(str(pid_file))
        return {
            "ok": False,
            "handle": None,
            "argv": argv,
            "error": f"llama-server exited with status {proc.returncode} after readiness probe",
            "hint": diagnose_llama_server_log(log_path),
            "log_path": str(log_path),
            "mtp": mtp,
        }

    return {
        "ok": True,
        "handle": handle,
        "argv": argv,
        "error": None,
        "log_path": str(log_path),
        "mtp": mtp,
    }


def llamacpp_stop_server(handle: LlamaServerHandle, *, grace_seconds: float = 5.0) -> bool:
    if not handle.we_started_it:
        return False

    proc = handle.proc
    if proc is not None:
        if os.name == "posix":
            _signal_process(handle.pid, 15)
        else:
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.terminate()
        try:
            proc.wait(timeout=max(grace_seconds, 0.0))
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                _signal_process(handle.pid, 9)
            else:
                with contextlib.suppress(ProcessLookupError, OSError):
                    proc.kill()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                return False
        _cleanup_pid_file(handle.pid_file)
        return True

    pid = handle.pid
    _signal_process(pid, 15)

    deadline = time.monotonic() + max(grace_seconds, 0.0)
    while time.monotonic() < deadline:
        if _pid_gone(pid):
            _cleanup_pid_file(handle.pid_file)
            return True
        time.sleep(0.1)

    _signal_process(pid, 9)

    deadline2 = time.monotonic() + 2.0
    while time.monotonic() < deadline2:
        if _pid_gone(pid):
            _cleanup_pid_file(handle.pid_file)
            return True
        time.sleep(0.1)
    return _pid_gone(pid)


def llamacpp_stop_server_by_port(port: int, *, grace_seconds: float = 5.0) -> dict[str, Any]:
    pid_file = LLAMACPP_PID_DIR / f"llama-server-{port}.pid"
    if not pid_file.exists():
        return {
            "ok": False,
            "pid": None,
            "error": (
                f"no ccl-managed llama-server pid file for port {port}; "
                f"the server was started outside ccl"
            ),
        }
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        _cleanup_pid_file(str(pid_file))
        return {
            "ok": False,
            "pid": None,
            "error": f"unreadable pid file for port {port}",
        }

    if _pid_gone(pid):
        _cleanup_pid_file(str(pid_file))
        return {"ok": True, "pid": pid, "already_gone": True}

    _signal_process(pid, 15)
    deadline = time.monotonic() + max(grace_seconds, 0.0)
    while time.monotonic() < deadline:
        if _pid_gone(pid):
            _cleanup_pid_file(str(pid_file))
            return {"ok": True, "pid": pid}
        time.sleep(0.1)

    _signal_process(pid, 9)
    deadline2 = time.monotonic() + 2.0
    while time.monotonic() < deadline2:
        if _pid_gone(pid):
            _cleanup_pid_file(str(pid_file))
            return {"ok": True, "pid": pid}
        time.sleep(0.1)

    if _pid_gone(pid):
        _cleanup_pid_file(str(pid_file))
        return {"ok": True, "pid": pid}
    return {
        "ok": False,
        "pid": pid,
        "error": f"failed to stop llama-server pid {pid} on port {port}",
    }


def _pid_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return True
    except OSError:
        return True


def _signal_process(pid: int, sig: int) -> None:
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(pid), sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(pid, sig)


def _cleanup_pid_file(pid_file: str) -> None:
    with contextlib.suppress(OSError):
        Path(pid_file).unlink(missing_ok=True)


def smoke_test_llamacpp_model(
    model: str,
    prompt: str = "Reply with exactly READY",
    expected: str | None = "READY",
    max_tokens: int = 256,
) -> dict[str, Any]:
    import time
    import urllib.error
    import urllib.request

    url = f"{llamacpp_base_url()}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(url, data=payload, headers=_auth_headers(LLAMACPP_API_KEY))
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        duration_seconds = max(time.time() - start, 1e-6)
        choice = body["choices"][0]
        message = choice.get("message") or {}
        text = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or "").strip()
        finish_reason = choice.get("finish_reason")
        usage = body.get("usage") or {}
        raw_completion = usage.get("completion_tokens")
        completion_tokens = int(raw_completion) if isinstance(raw_completion, int) else None
        tokens_per_second: float | None = None
        if completion_tokens is not None and completion_tokens > 0:
            tokens_per_second = completion_tokens / duration_seconds
        ok_flag = (
            bool(text or reasoning)
            if expected is None
            else expected.upper() in text.upper() or expected.upper() in reasoning.upper()
        )
        result: dict[str, Any] = {
            "ok": ok_flag,
            "response": text,
            "tokens_per_second": tokens_per_second,
            "completion_tokens": completion_tokens,
            "duration_seconds": duration_seconds,
            "finish_reason": finish_reason,
        }
        if not ok_flag:
            snippet = (text or reasoning).replace("\n", " ").strip()
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            parts = [f"finish_reason={finish_reason}"]
            if reasoning and not text:
                parts.append("model produced reasoning but no final answer")
            if snippet:
                parts.append(f"saw: '{snippet}'")
            result["error"] = "; ".join(parts)
        return result
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
