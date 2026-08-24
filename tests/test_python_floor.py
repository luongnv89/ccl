"""
Guard tests for the supported Python floor (F-DEP-017).

The floor is declared in five places that must stay in sync:
  * pyproject.toml  -> requires-python
  * pyproject.toml  -> Programming Language classifiers
  * pyproject.toml  -> ruff target-version
  * pyproject.toml  -> mypy python_version
  * ci.yml          -> test-job version matrix

If the floor ever moves again, update all five sites and this module.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PYTHON_FLOOR = "3.11"


def _pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _ci_matrix() -> list[str]:
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    match = re.search(r"python-version:\s*\[([^\]]+)\]", text)
    assert match, "python-version matrix not found in ci.yml"
    return [v.strip().strip('"') for v in match.group(1).split(",")]


def test_requires_python_declares_floor() -> None:
    pyproject = _pyproject()
    assert pyproject["project"]["requires-python"] == f">={PYTHON_FLOOR}"


def test_classifiers_match_supported_versions() -> None:
    classifiers = [c for c in _pyproject()["project"]["classifiers"] if "Python ::" in c]
    assert f"Programming Language :: Python :: {PYTHON_FLOOR}" in classifiers
    assert "Programming Language :: Python :: 3.10" not in classifiers


def test_ruff_target_version_matches_floor() -> None:
    assert _pyproject()["tool"]["ruff"]["target-version"] == f"py{PYTHON_FLOOR.replace('.', '')}"


def test_mypy_python_version_matches_floor() -> None:
    assert _pyproject()["tool"]["mypy"]["python_version"] == PYTHON_FLOOR


def test_ci_matrix_has_no_leg_below_floor() -> None:
    matrix = _ci_matrix()

    def minor(version: str) -> int:
        return int(version.split(".")[1])

    floor_minor = minor(PYTHON_FLOOR)
    assert matrix
    assert all(minor(v) >= floor_minor for v in matrix), matrix
    assert PYTHON_FLOOR in matrix
