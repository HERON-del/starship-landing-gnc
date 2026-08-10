"""
Problem registry.

Adding a new problem for a future week is exactly three steps:

    1. drop a module in `src/gnc/problems/`
    2. subclass `Problem`, declare `params()`, implement `solve()`
    3. decorate it with `@register`

The server exposes it automatically and the UI builds its controls from the
declared parameters. No JavaScript changes, ever.
"""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from typing import Any

from .types import Param, Trajectory

_REGISTRY: dict[str, "Problem"] = {}


class Problem(ABC):
    """Base class for anything the viewer can solve and render."""

    #: url-safe identifier, unique
    slug: str = ""
    #: shown in the problem dropdown
    title: str = ""
    #: one-line description shown under the title
    summary: str = ""
    #: roadmap phase, e.g. "Day 1", "Week 2"
    phase: str = ""
    #: rough scale of the scene in metres, used to frame the camera
    scene_scale: float = 100.0

    @abstractmethod
    def params(self) -> list[Param]:
        """Declare every tunable knob."""

    @abstractmethod
    def solve(self, values: dict[str, Any]) -> Trajectory:
        """Solve for the given parameter values and return a trajectory."""

    # -- convenience -------------------------------------------------
    def defaults(self) -> dict[str, Any]:
        return {p.key: p.default for p in self.params()}

    def merge(self, values: dict[str, Any] | None) -> dict[str, Any]:
        """Fill any missing key with its declared default, and coerce types."""
        merged = self.defaults()
        if not values:
            return merged

        by_key = {p.key: p for p in self.params()}
        for k, v in values.items():
            if k not in by_key:
                continue
            spec = by_key[k]
            try:
                if spec.kind == "int":
                    merged[k] = int(v)
                elif spec.kind == "float":
                    merged[k] = float(v)
                elif spec.kind == "bool":
                    merged[k] = bool(v)
                else:
                    merged[k] = v
            except (TypeError, ValueError):
                merged[k] = spec.default
        return merged

    def describe(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "title": self.title,
            "summary": self.summary,
            "phase": self.phase,
            "scene_scale": self.scene_scale,
            "params": [p.to_dict() for p in self.params()],
        }


def register(cls: type[Problem]) -> type[Problem]:
    """Class decorator that adds a problem to the registry."""
    inst = cls()
    if not inst.slug:
        raise ValueError(f"{cls.__name__} must define a slug")
    if inst.slug in _REGISTRY:
        raise ValueError(f"duplicate problem slug: {inst.slug}")
    _REGISTRY[inst.slug] = inst
    return cls


def load_all() -> None:
    """Import every module in gnc.problems so decorators run."""
    from . import problems

    for mod in pkgutil.iter_modules(problems.__path__):
        importlib.import_module(f"{problems.__name__}.{mod.name}")


def all_problems() -> list[Problem]:
    return sorted(_REGISTRY.values(), key=lambda p: (p.phase, p.title))


def get(slug: str) -> Problem:
    if slug not in _REGISTRY:
        raise KeyError(slug)
    return _REGISTRY[slug]
