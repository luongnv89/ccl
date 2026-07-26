"""
UI helpers for the wizard — banner, output formatting, panels.

Exports:
    console        — shared Rich Console instance
    print_welcome_banner  — ASCII CCL banner + tagline
    header, ok, warn, fail, info — step output helpers
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel as Panel

from claude_codex_local import __version__

console = Console()

_CCL_BANNER = r"""
  ██████╗ ██████╗██╗
 ██╔════╝██╔════╝██║
 ██║     ██║     ██║
 ██║     ██║     ██║
 ╚██████╗╚██████╗███████╗
  ╚═════╝ ╚═════╝╚══════╝
"""

_CCL_TAGLINE = "Hit your limit? Need privacy? Just swap the model."
_CCL_REPO_URL = "https://github.com/luongnv89/claude-codex-local"


def print_welcome_banner() -> None:
    """Print the ASCII 3D CCL banner, tagline, version, and repo URL."""
    console.print(_CCL_BANNER, style="bold cyan", highlight=False)
    console.print(f"  [bold white]{_CCL_TAGLINE}[/bold white]")
    console.print(f"  [dim]v{__version__}  ·  [link={_CCL_REPO_URL}]{_CCL_REPO_URL}[/link][/dim]")
    console.print()


def header(title: str) -> None:
    console.print()
    console.print(Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="cyan"))


def ok(msg: str) -> None:
    console.print(f"[green]\u2713[/green] {msg}")


def warn(msg: str) -> None:
    console.print(f"[yellow]![/yellow] {msg}")


def fail(msg: str) -> None:
    console.print(f"[red]\u2717[/red] {msg}")


def info(msg: str) -> None:
    console.print(f"[dim]\u00b7[/dim] {msg}")
