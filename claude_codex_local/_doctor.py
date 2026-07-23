from __future__ import annotations

import argparse
import json
import re
import subprocess
from typing import Any

from claude_codex_local._model_selection import MODE_CHOICES

# Re-export run so that _patch_run (which patches _doctor.run) can target it.
from claude_codex_local._shell import run as _run_from_shell

# Alias for _patch_run compatibility — tests patch _doctor.run directly.
run = _run_from_shell


def smoke_test_codex(model: str, runtime: str = "ollama") -> dict[str, Any]:
    import claude_codex_local.core as _core

    env = _core.state_env()
    provider = "lmstudio" if runtime == "lmstudio" else "ollama"
    try:
        cp = _core.run(
            [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--oss",
                "--local-provider",
                provider,
                "-m",
                model,
                "Reply with exactly READY",
            ],
            env=env,
            timeout=240,
        )
        merged = (cp.stdout + "\n" + cp.stderr).strip()
        normalized = re.sub(r"[^a-z]", "", merged.lower())
        ok = "ready" in normalized
        auth_noise = (
            "failed to refresh available models" in merged.lower()
            or "401 unauthorized" in merged.lower()
        )
        return {
            "ok": ok,
            "output": cp.stdout.strip(),
            "stderr": cp.stderr.strip(),
            "auth_noise": auth_noise,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout after 240s"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def doctor(run_codex_smoke: bool, mode: str = "balanced") -> dict[str, Any]:
    # Import core at call time so that test monkeypatches on core.machine_profile
    # (or core._machine_profile_in_process_cache) take effect instead of bypassing
    # the patch via a direct _machine_profile import.
    import claude_codex_local.core as _core
    from claude_codex_local._model_selection import select_best_model

    profile = _core.machine_profile()
    recommendation = select_best_model(profile, mode)
    issues: list[str] = []
    fixes: list[str] = []

    for tool_name, tool_info in profile["tools"].items():
        if not tool_info.get("present"):
            issues.append(f"Missing tool: {tool_name}")

    if not profile["ollama"]["models"] and not profile["lmstudio"].get("models"):
        issues.append("No models found in Ollama or LM Studio.")

    if recommendation["status"] == "download-required":
        issues.append("No suitable local coding model is installed.")
        fixes.extend(recommendation["next_steps"])

    codex_smoke = (
        smoke_test_codex(recommendation["selected_model"], recommendation["runtime"])
        if run_codex_smoke
        else None
    )
    if codex_smoke and not codex_smoke.get("ok"):
        issues.append("Codex local smoke test failed.")
    elif codex_smoke and codex_smoke.get("auth_noise"):
        fixes.append("Codex emits a harmless 401 model-refresh warning in local-only mode.")

    return {
        "profile": profile,
        "recommendation": recommendation,
        "issues": issues,
        "fixes": fixes,
        "codex_smoke": codex_smoke,
    }


def print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m claude_codex_local.core",
        description=(
            "claude-codex-local debug CLI — machine profile, model recommendation, "
            "doctor, and adapter introspection. These commands dump JSON for "
            "scripting and debugging; the user-facing binary is `ccl`."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("profile")

    rec_cmd = sub.add_parser("recommend")
    rec_cmd.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="balanced",
        help="Preset: balanced (default), fast (smallest/fastest), quality (highest score)",
    )

    doctor_cmd = sub.add_parser("doctor")
    doctor_cmd.add_argument("--run-codex-smoke", action="store_true")
    doctor_cmd.add_argument("--mode", choices=MODE_CHOICES, default="balanced")

    sub.add_parser("adapters")

    engine_cmd = sub.add_parser(
        "engine",
        help="Run a per-engine lifecycle script through the uniform engine contract",
    )
    engine_cmd.add_argument("engine", help="Engine name, e.g. ollama, llamacpp, vllm, 9router")
    engine_cmd.add_argument(
        "action",
        help="Lifecycle action: install, config, optimize, test, or benchmark",
    )
    engine_cmd.add_argument(
        "--model", default="", help="Model/tag/path for test and benchmark actions"
    )
    engine_cmd.add_argument(
        "--execute",
        action="store_true",
        help="Run actions that touch a live engine. Without this, test/benchmark actions dry-run.",
    )

    args = parser.parse_args()

    if args.command == "profile":
        import claude_codex_local.core as _core

        print_payload(_core.machine_profile())
    elif args.command == "recommend":
        import claude_codex_local.core as _core

        print_payload(_core.select_best_model(_core.machine_profile(), args.mode))
    elif args.command == "doctor":
        print_payload(doctor(args.run_codex_smoke, args.mode))
    elif args.command == "adapters":
        from claude_codex_local._adapters import ALL_ADAPTERS

        result = []
        for adapter in ALL_ADAPTERS:
            result.append(
                {
                    "name": adapter.name,
                    "detect": adapter.detect(),
                    "healthcheck": adapter.healthcheck(),
                    "models": adapter.list_models(),
                    "recommend_params": {m: adapter.recommend_params(m) for m in MODE_CHOICES},
                }
            )
        print_payload({"adapters": result})
    elif args.command == "engine":
        from claude_codex_local.engines import run_engine_action
        from claude_codex_local.engines.registry import EngineLifecycleError

        kwargs: dict[str, Any] = {
            "model": args.model,
            "dry_run": not args.execute,
        }
        if args.action == "optimize":
            import claude_codex_local.core as _core

            kwargs["profile"] = _core.machine_profile(run_llmfit=False)
        try:
            engine_result = run_engine_action(
                args.engine,
                args.action,
                **kwargs,
            )
        except EngineLifecycleError as exc:
            parser.error(str(exc))
        print_payload(engine_result)


if __name__ == "__main__":
    main()
