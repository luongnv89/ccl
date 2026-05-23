"""Per-engine lifecycle scripts.

Each engine package owns the same narrow action surface:
install, config, optimize, test, and benchmark.  Core callers use the
registry helpers here instead of branching on concrete engines.
"""

from .registry import ACTIONS, engine_action_matrix, engine_names, run_engine_action

__all__ = ["ACTIONS", "engine_action_matrix", "engine_names", "run_engine_action"]
