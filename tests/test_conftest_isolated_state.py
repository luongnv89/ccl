"""
Order-independence guards for the ``isolated_state`` fixture (issue #185).

The fixture historically used ``return`` instead of ``yield``: it reloaded
the config/core/wizard modules to rebind state-derived constants and never
restored them, so after any isolated test the suite kept pointing at a tmp
directory pytest had already deleted — outcomes depended on execution order.

``test_isolated_state_redirects_state_constants`` consumes the fixture and
``test_constants_restored_after_isolated_state`` asserts the teardown put
every state-derived constant back. The pair is order-sensitive *by design*
under forward (definition-order) execution; in a reversed run both tests
still pass, they simply stop proving the property.
"""

from __future__ import annotations

import sys
import types

import conftest

import claude_codex_local._config as _config
import claude_codex_local.core as core
import claude_codex_local.wizard as wizard
import claude_codex_local.wizard_cli as wizard_cli
import claude_codex_local.wizard_state as wizard_state


def _state_constants():
    """State-derived constants that must survive an isolated_state round-trip."""
    return {
        "_config.STATE_DIR": _config.STATE_DIR,
        "core.STATE_DIR": core.STATE_DIR,
        "wizard_state.STATE_DIR": wizard_state.STATE_DIR,
        "wizard_state.STATE_FILE": wizard_state.STATE_FILE,
        "wizard.STATE_DIR": wizard.STATE_DIR,
        "wizard.STATE_FILE": wizard.STATE_FILE,
        "wizard_cli.STATE_FILE": wizard_cli.STATE_FILE,
    }


# Captured at collection time, before any test (and therefore any fixture
# reload) has run — the pristine module state every test is entitled to see.
_PRISTINE_CONSTANTS = _state_constants()


def test_isolated_state_redirects_state_constants(isolated_state):
    pb, wiz, state_dir = isolated_state
    assert state_dir == pb.STATE_DIR
    assert _config.STATE_DIR == state_dir
    assert wizard_state.STATE_DIR == state_dir
    assert wizard_cli.STATE_FILE == state_dir / "wizard-state.json"
    assert _state_constants() != _PRISTINE_CONSTANTS


def test_constants_restored_after_isolated_state():
    # Runs right after the test above (definition order): the previous
    # test's isolated_state teardown must have restored every constant.
    assert _state_constants() == _PRISTINE_CONSTANTS


def test_restore_modules_roundtrip():
    """_restore_modules puts a mutated module's __dict__ back verbatim."""
    mod_name = "_ccl_test_restore_roundtrip_dummy"
    dummy = types.ModuleType(mod_name)
    dummy.STATE_DIR = "/pristine"
    dummy.marker = object()
    sys.modules[mod_name] = dummy
    try:
        saved, fresh = conftest._snapshot_modules([mod_name])
        assert fresh == []
        dummy.STATE_DIR = "/sandbox"
        del dummy.marker
        dummy.extra = 42
        conftest._restore_modules(saved, fresh)
        assert dummy.STATE_DIR == "/pristine"
        assert not hasattr(dummy, "extra")
        assert hasattr(dummy, "marker")
    finally:
        sys.modules.pop(mod_name, None)


def test_snapshot_modules_tracks_unimported_names():
    saved, fresh = conftest._snapshot_modules(["_ccl_test_never_imported_dummy"])
    assert saved == {}
    assert fresh == ["_ccl_test_never_imported_dummy"]
