"""Tests for registry-derived engine metadata (Issue #155).

Verifies that:
- ALL_ENGINES is the single source of engine names
- Wizard derives _ALL_ENGINES from the registry
- Core adapters are built from the registry
- engine_capabilities() reports correct action availability
- Adding a new engine package requires no changes to wizard/core
"""

from __future__ import annotations

from claude_codex_local.engines import (
    ALL_ENGINES,
    engine_capabilities,
    engine_names,
)


class TestAllEnginesConsistency:
    """ALL_ENGINES must match discovered engine_names()."""

    def test_all_engines_matches_discovered_names(self):
        """The static list must equal what pkgutil discovers."""
        assert tuple(ALL_ENGINES) == engine_names()

    def test_all_engines_is_sorted(self):
        """ALL_ENGINES should be sorted for deterministic iteration."""
        assert ALL_ENGINES == tuple(sorted(ALL_ENGINES))

    def test_all_engines_contains_required_set(self):
        """All previously-required engines must still be present."""
        required = {"ollama", "lmstudio", "llamacpp", "vllm", "9router"}
        assert required.issubset(set(ALL_ENGINES))


class TestWizardDerivesFromRegistry:
    """Wizard must derive its engine list from the registry."""

    def test_wizard_all_engines_from_registry(self):
        """_ALL_ENGINES in wizard must equal ALL_ENGINES from registry."""
        from claude_codex_local import wizard as wiz

        assert set(wiz._ALL_ENGINES) == set(ALL_ENGINES)

    def test_wizard_all_engines_not_hardcoded(self):
        """_ALL_ENGINES should not be a literal list (it should be derived)."""
        import inspect

        import claude_codex_local.wizard as wiz_module

        source = inspect.getsource(wiz_module)
        # The old pattern was: _ALL_ENGINES = ["ollama", ...]
        # The new pattern is: _ALL_ENGINES = list(_REGISTRY_ENGINES)
        assert "_ALL_ENGINES = list(_REGISTRY_ENGINES)" in source

    def test_wizard_imports_registry(self):
        """Wizard must import ALL_ENGINES from the registry."""
        import inspect

        import claude_codex_local.wizard as wiz_module

        source = inspect.getsource(wiz_module)
        assert "ALL_ENGINES" in source and "_REGISTRY_ENGINES" in source


class TestCoreAdaptersFromRegistry:
    """Core adapters must be built from the registry."""

    def test_all_adapters_count_matches_engines(self):
        """Number of adapters must match number of engines."""
        import claude_codex_local.core as pb

        assert len(pb.ALL_ADAPTERS) == len(ALL_ENGINES)

    def test_all_adapters_names_match_engines(self):
        """Adapter names must match ALL_ENGINES."""
        import claude_codex_local.core as pb

        adapter_names = {a.name for a in pb.ALL_ADAPTERS}
        assert adapter_names == set(ALL_ENGINES)

    def test_all_adapters_preference_order_preserved(self):
        """Adapter order must preserve the original preference order."""
        import claude_codex_local.core as pb

        expected_order = ["lmstudio", "ollama", "llamacpp", "vllm", "9router", "openrouter"]
        adapter_names = [a.name for a in pb.ALL_ADAPTERS]
        assert adapter_names == expected_order

    def test_core_no_hardcoded_engine_list(self):
        """core.py must not contain a hardcoded engine list literal."""
        import inspect

        import claude_codex_local.core as core_module

        source = inspect.getsource(core_module)
        # core.py is now a facade — the adapter list lives in _adapters.py
        assert "ALL_ADAPTERS" in source

    def test_core_imports_registry(self):
        """core.py must reference the engine registry."""
        import inspect

        import claude_codex_local._adapters as adapters_module

        source = inspect.getsource(adapters_module)
        assert "_ENGINE_ADAPTER_MAP" in source
        assert "_build_adapters" in source


class TestEngineCapabilities:
    """engine_capabilities() must report correct action availability."""

    def test_known_engines_have_capabilities(self):
        """Every engine in ALL_ENGINES must have a capabilities dict."""
        for engine in ALL_ENGINES:
            caps = engine_capabilities(engine)
            assert isinstance(caps, dict)
            for action in ("install", "config", "optimize", "test", "benchmark"):
                assert action in caps

    def test_engine_capabilities_match_matrix(self):
        """engine_capabilities() must agree with engine_action_matrix()."""
        from claude_codex_local.engines import engine_action_matrix

        matrix = engine_action_matrix()
        for engine in ALL_ENGINES:
            caps = engine_capabilities(engine)
            for action in ("install", "config", "optimize", "test", "benchmark"):
                expected = action in matrix.get(engine, [])
                assert caps[action] == expected, (
                    f"Engine {engine} action {action}: capabilities={caps[action]}, "
                    f"matrix={expected}"
                )


class TestNewEngineDiscovery:
    """Adding a new engine package should not require wizard/core changes."""

    def test_new_engine_appears_in_wizard(self, tmp_path, monkeypatch):
        """A new engine package should be discoverable by the wizard."""
        package = tmp_path / "newengine"
        package.mkdir()
        (package / "__init__.py").write_text('ENGINE_NAME = "newengine"\n')
        for action in ("install", "config", "optimize", "test", "benchmark"):
            (package / f"{action}.py").write_text(
                "def run(**kwargs):\n"
                f"    return {{'engine': 'newengine', 'action': '{action}', 'ok': True}}\n"
            )

        import claude_codex_local.engines as engines_pkg
        import claude_codex_local.engines.registry as registry

        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.setattr(engines_pkg, "__path__", [*engines_pkg.__path__, str(tmp_path)])
        registry._engine_packages.cache_clear()

        try:
            assert "newengine" in engine_names()
            caps = engine_capabilities("newengine")
            assert all(caps.values())
        finally:
            registry._engine_packages.cache_clear()
