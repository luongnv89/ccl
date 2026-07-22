"""Model picker strategy pattern.

Each engine or capability group gets its own ``ModelPickerStrategy`` that
encapsulates the engine-specific model-picking behavior.  The wizard's
``step_2_4_pick_model`` delegates to the registered strategy rather than
branching on ``engine`` directly.

Adding a new engine with a custom picker:
  1. Write a strategy class implementing ``ModelPickerStrategy``.
  2. Call ``register_picker("new-engine", NewEnginePicker())`` at module
     level (e.g. inside your engine package).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_codex_local.wizard import WizardState


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class ModelPickerStrategy(ABC):
    """Encapsulates the model-picking behaviour for one engine or
    capability group.

    ``pick_model`` is called by the wizard during Step 4.
    """

    @abstractmethod
    def pick_model(
        self,
        state: WizardState,
        non_interactive: bool = False,
    ) -> bool: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PICKERS: dict[str, ModelPickerStrategy] = {}


def get_picker(engine: str) -> ModelPickerStrategy | None:
    """Return the registered picker for *engine*, or ``None``."""
    return _PICKERS.get(engine)


def register_picker(engine: str, strategy: ModelPickerStrategy) -> None:
    """Register *strategy* as the picker for *engine*."""
    _PICKERS[engine] = strategy


# ---------------------------------------------------------------------------
# Engine-specific strategies
# ---------------------------------------------------------------------------


class Router9Picker(ModelPickerStrategy):
    """Model picker for the 9router cloud-routing engine."""

    def pick_model(self, state: WizardState, non_interactive: bool = False) -> bool:
        from claude_codex_local.wizard import _step_4_pick_model_9router_impl

        return _step_4_pick_model_9router_impl(state, non_interactive)


class OpenRouterPicker(ModelPickerStrategy):
    """Model picker for the OpenRouter hosted-SaaS engine."""

    def pick_model(self, state: WizardState, non_interactive: bool = False) -> bool:
        from claude_codex_local.wizard import _step_4_pick_model_openrouter_impl

        return _step_4_pick_model_openrouter_impl(state, non_interactive)


class VLLMPicker(ModelPickerStrategy):
    """Model picker for the vLLM engine (reads model from /v1/models)."""

    def pick_model(self, state: WizardState, non_interactive: bool = False) -> bool:
        from claude_codex_local.wizard import _step_4_pick_model_vllm_impl

        return _step_4_pick_model_vllm_impl(state, non_interactive)


class LocalPicker(ModelPickerStrategy):
    """Model picker for local engines (ollama, LM Studio, llama.cpp).

    Handles llmfit profiles, merged model lists, running-server detection,
    and interactive / non-interactive model selection.
    """

    def pick_model(self, state: WizardState, non_interactive: bool = False) -> bool:
        from claude_codex_local.wizard import _step_4_pick_model_local_impl

        return _step_4_pick_model_local_impl(state, non_interactive)


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------

register_picker("9router", Router9Picker())
register_picker("openrouter", OpenRouterPicker())
register_picker("vllm", VLLMPicker())
register_picker("ollama", LocalPicker())
register_picker("lmstudio", LocalPicker())
register_picker("llamacpp", LocalPicker())
