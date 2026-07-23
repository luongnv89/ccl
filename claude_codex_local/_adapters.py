from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

from claude_codex_local._config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_KEY_FILE,
    ROUTER9_BASE_URL,
    _is_local_base_url,
    _normalize_base_url,
    _probe_openai_models_endpoint,
)
from claude_codex_local._shell import (
    _auth_headers,
    command_version,
    llamacpp_base_url,
    ollama_base_url,
)


class RuntimeAdapter(Protocol):
    name: str

    def detect(self) -> dict[str, Any]: ...

    def healthcheck(self) -> dict[str, Any]: ...

    def list_models(self) -> list[dict[str, Any]]: ...

    def run_test(self, model: str) -> dict[str, Any]: ...

    def recommend_params(self, mode: str) -> dict[str, Any]: ...


@dataclass
class OllamaAdapter:
    name: str = "ollama"

    def detect(self) -> dict[str, Any]:
        from claude_codex_local._ollama import ollama_info

        info = ollama_info()
        if info.get("present"):
            return info
        return command_version("ollama")

    def healthcheck(self) -> dict[str, Any]:
        from claude_codex_local._ollama import ollama_info, parse_ollama_list

        info = ollama_info()
        if info.get("server_reachable"):
            models = info.get("models", [])
            return {
                "ok": True,
                "detail": f"Ollama server up at {info['base_url']}, {len(models)} model(s) available",
            }
        cli = command_version("ollama")
        if not cli.get("present"):
            return {"ok": False, "detail": f"Ollama server not reachable at {ollama_base_url()}"}
        models = parse_ollama_list()
        return {"ok": True, "detail": f"{len(models)} model(s) installed"}

    def list_models(self) -> list[dict[str, Any]]:
        from claude_codex_local._ollama import parse_ollama_list

        return parse_ollama_list()

    def run_test(self, model: str) -> dict[str, Any]:
        from claude_codex_local._ollama import smoke_test_ollama_model

        return smoke_test_ollama_model(model)

    def recommend_params(self, mode: str) -> dict[str, Any]:
        return {"provider": "ollama", "extra_flags": []}


@dataclass
class LMStudioAdapter:
    name: str = "lmstudio"

    def detect(self) -> dict[str, Any]:
        from claude_codex_local._lmstudio import lms_binary

        lms = lms_binary()
        if not lms:
            return {"present": False, "version": ""}
        return command_version(lms, ["--version"])

    def healthcheck(self) -> dict[str, Any]:
        from claude_codex_local._lmstudio import lms_info

        info = lms_info()
        if not info.get("present"):
            return {"ok": False, "detail": "lms CLI not found"}
        if not info.get("server_running"):
            return {
                "ok": False,
                "detail": f"LM Studio server not running on port {info['server_port']}. Run: lms server start",
            }
        return {
            "ok": True,
            "detail": f"server up on port {info['server_port']}, {len(info['models'])} model(s) installed",
        }

    def list_models(self) -> list[dict[str, Any]]:
        from claude_codex_local._lmstudio import lms_info

        info = lms_info()
        return [
            {"name": m["path"], "format": m["format"], "local": True}
            for m in info.get("models", [])
        ]

    def run_test(self, model: str) -> dict[str, Any]:
        from claude_codex_local._lmstudio import smoke_test_lmstudio_model

        return smoke_test_lmstudio_model(model)

    def recommend_params(self, mode: str) -> dict[str, Any]:
        return {"provider": "lmstudio", "extra_flags": []}


@dataclass
class LlamaCppAdapter:
    name: str = "llamacpp"

    def detect(self) -> dict[str, Any]:
        from claude_codex_local._llamacpp_lifecycle import llamacpp_detect

        return llamacpp_detect()

    def healthcheck(self) -> dict[str, Any]:
        from claude_codex_local._llamacpp_lifecycle import llamacpp_info

        info = llamacpp_info()
        base_url = info.get("base_url", llamacpp_base_url())
        if info.get("server_running"):
            return {
                "ok": True,
                "detail": f"server up at {base_url}",
            }
        if info.get("remote"):
            return {
                "ok": False,
                "detail": f"remote llama.cpp server at {base_url} is not responding "
                f"(GET /health failed). Verify the remote host and LLAMACPP_BASE_URL.",
            }
        if not info.get("present"):
            return {"ok": False, "detail": "llama-server binary not found in PATH"}
        return {
            "ok": False,
            "detail": f"llama.cpp server not running at {base_url}. "
            f"The wizard's Step 5 will auto-start it after a model download; "
            f"run `ccl --resume` to continue.",
        }

    def list_models(self) -> list[dict[str, Any]]:
        from claude_codex_local._llamacpp_lifecycle import llamacpp_info

        info = llamacpp_info()
        if not info.get("server_running"):
            return []
        model = info.get("model")
        if model:
            return [{"name": model, "format": "gguf", "local": True}]
        return []

    def run_test(self, model: str) -> dict[str, Any]:
        from claude_codex_local._llamacpp_lifecycle import smoke_test_llamacpp_model

        return smoke_test_llamacpp_model(model)

    def recommend_params(self, mode: str) -> dict[str, Any]:
        return {"provider": "llamacpp", "extra_flags": []}


@dataclass
class VLLMAdapter:
    name: str = "vllm"
    _base_url: str | None = None
    _api_key: str | None = None
    _timeout: int = 60
    _max_tokens: int = 2048

    def __post_init__(self):
        base = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        self._base_url = _normalize_base_url(base) if isinstance(base, str) else base
        self._api_key = os.environ.get("VLLM_API_KEY", "")
        self._timeout = int(os.environ.get("VLLM_TIMEOUT", "60"))
        self._max_tokens = int(os.environ.get("VLLM_MAX_TOKENS", "2048"))

    def _full_url(self, endpoint: str) -> str:
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        return f"{self._base_url}{endpoint}"

    def _build_headers(self) -> dict[str, str]:
        return _auth_headers(self._api_key)

    def detect(self) -> dict[str, Any]:
        try:
            result = _probe_openai_models_endpoint(
                f"{self._base_url}/v1",
                service_name="vLLM",
                timeout=5,
                headers=self._build_headers(),
            )
        except Exception as exc:
            return {
                "present": False,
                "version": "",
                "error": f"vLLM probe failed unexpectedly: {exc}",
                "error_type": "unexpected_exception",
            }
        if result.get("ok"):
            headers = result.get("headers", {})
            return {
                "present": True,
                "version": headers.get("X-VLLM-Version", "unknown"),
                "base_url": self._base_url,
            }
        return {
            "present": False,
            "version": "",
            "error": result.get("error", "vLLM server not reachable"),
            "error_type": result.get("error_type", "unknown"),
        }

    def healthcheck(self) -> dict[str, Any]:
        detect_info = self.detect()
        if not detect_info.get("present"):
            return {
                "ok": False,
                "detail": f"vLLM server not reachable at {self._base_url}. "
                "Start vLLM server: vllm server --model <model_path>",
            }
        try:
            import urllib.error
            import urllib.request

            url = self._full_url("/v1/models")
            req = urllib.request.Request(url, headers=self._build_headers(), method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
                models = body.get("data", [])
                return {
                    "ok": True,
                    "detail": f"vLLM server up at {self._base_url}, {len(models)} model(s) available",
                }
        except urllib.error.URLError:
            return {
                "ok": False,
                "detail": f"vLLM server at {self._base_url} is not responding",
            }
        except Exception as exc:
            return {"ok": False, "detail": str(exc)}

    def list_models(self) -> list[dict[str, Any]]:
        try:
            result = _probe_openai_models_endpoint(
                f"{self._base_url}/v1",
                service_name="vLLM",
                timeout=5,
                headers=self._build_headers(),
            )
        except Exception:
            return []
        if not result.get("ok"):
            return []
        return [
            {
                "name": model_id,
                "format": "unknown",
                "local": _is_local_base_url(self._base_url or ""),
            }
            for model_id in result.get("models", [])
        ]

    def run_test(self, model: str) -> dict[str, Any]:
        from claude_codex_local._vllm import smoke_test_vllm_model

        return smoke_test_vllm_model(
            model,
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            max_tokens=self._max_tokens,
        )

    def recommend_params(self, mode: str) -> dict[str, Any]:
        return {"provider": "vllm", "extra_flags": []}


@dataclass
class Router9Adapter:
    name: str = "9router"

    def detect(self) -> dict[str, Any]:
        result = _probe_openai_models_endpoint(
            ROUTER9_BASE_URL,
            service_name="9router",
            timeout=5,
            headers={"Content-Type": "application/json"},
        )
        if result.get("ok"):
            return {"present": True, "version": "", "base_url": ROUTER9_BASE_URL}
        return {
            "present": False,
            "version": "",
            "error": result.get("error", "9router endpoint not reachable"),
            "error_type": result.get("error_type", "unknown"),
        }

    def healthcheck(self) -> dict[str, Any]:
        from claude_codex_local._router9 import smoke_test_router9_models

        result = smoke_test_router9_models()
        if result.get("ok"):
            count = len(result.get("models", []))
            return {
                "ok": True,
                "detail": f"server up at {ROUTER9_BASE_URL}, {count} model(s) available",
            }
        return {"ok": False, "detail": result.get("error", "unreachable")}

    def list_models(self) -> list[dict[str, Any]]:
        from claude_codex_local._router9 import smoke_test_router9_models

        result = smoke_test_router9_models()
        if not result.get("ok"):
            return []
        return [
            {"name": m, "format": "cloud-routed", "local": False} for m in result.get("models", [])
        ]

    def run_test(self, model: str) -> dict[str, Any]:
        from claude_codex_local._router9 import smoke_test_router9_models

        return smoke_test_router9_models()

    def recommend_params(self, mode: str) -> dict[str, Any]:
        return {"provider": "9router", "extra_flags": []}


@dataclass
class OpenRouterAdapter:
    name: str = "openrouter"

    def detect(self) -> dict[str, Any]:
        result = _probe_openai_models_endpoint(
            OPENROUTER_BASE_URL,
            service_name="OpenRouter",
            timeout=5,
            headers={"Content-Type": "application/json"},
        )
        if result.get("ok"):
            return {"present": True, "version": "", "base_url": OPENROUTER_BASE_URL}
        return {
            "present": False,
            "version": "",
            "error": result.get("error", "OpenRouter endpoint not reachable"),
            "error_type": result.get("error_type", "unknown"),
        }

    def healthcheck(self) -> dict[str, Any]:
        from claude_codex_local._openrouter import smoke_test_openrouter_models

        result = smoke_test_openrouter_models()
        if result.get("ok"):
            count = len(result.get("models", []))
            return {
                "ok": True,
                "detail": f"endpoint up at {OPENROUTER_BASE_URL}, {count} model(s) available",
            }
        return {"ok": False, "detail": result.get("error", "unreachable")}

    def list_models(self) -> list[dict[str, Any]]:
        from claude_codex_local._openrouter import smoke_test_openrouter_models

        result = smoke_test_openrouter_models()
        if not result.get("ok"):
            return []
        return [
            {"name": m, "format": "cloud-routed", "local": False} for m in result.get("models", [])
        ]

    def run_test(self, model: str) -> dict[str, Any]:
        from claude_codex_local._openrouter import smoke_test_openrouter_model

        api_key = os.environ.get("CCL_OPENROUTER_API_KEY", "").strip()
        if not api_key and OPENROUTER_KEY_FILE.exists():
            api_key = OPENROUTER_KEY_FILE.read_text().strip()
        return smoke_test_openrouter_model(model, api_key=api_key)

    def recommend_params(self, mode: str) -> dict[str, Any]:
        return {"provider": "openrouter", "extra_flags": []}


_ENGINE_ADAPTER_MAP: dict[str, type] = {
    "llamacpp": LlamaCppAdapter,
    "lmstudio": LMStudioAdapter,
    "ollama": OllamaAdapter,
    "openrouter": OpenRouterAdapter,
    "9router": Router9Adapter,
    "vllm": VLLMAdapter,
}

_ADAPTER_PREFERENCE_ORDER: tuple[str, ...] = (
    "lmstudio",
    "ollama",
    "llamacpp",
    "vllm",
    "9router",
    "openrouter",
)


def _build_adapters() -> list:
    from claude_codex_local.engines import ALL_ENGINES

    registered = set(ALL_ENGINES)

    adapters: list = []
    for engine_name in _ADAPTER_PREFERENCE_ORDER:
        if engine_name in registered:
            adapter_cls = _ENGINE_ADAPTER_MAP.get(engine_name)
            if adapter_cls is not None:
                adapters.append(adapter_cls())
    seen = set(_ADAPTER_PREFERENCE_ORDER)
    for engine_name in ALL_ENGINES:
        if engine_name not in seen:
            adapter_cls = _ENGINE_ADAPTER_MAP.get(engine_name)
            if adapter_cls is not None:
                adapters.append(adapter_cls())
    return adapters


ALL_ADAPTERS: list = _build_adapters()
