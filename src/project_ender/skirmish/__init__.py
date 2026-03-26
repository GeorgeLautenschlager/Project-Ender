"""Skirmish — bundled demo game for Ender.

A minimal 1-D tactical game used to validate the full Ender pipeline.
See DELIVERY.md for rules and rationale.
"""

from project_ender.skirmish.adapter import SkirmishAdapter
from project_ender.skirmish.game import Commander, SkirmishGame, SkirmishState

__all__ = ["SkirmishAdapter", "SkirmishGame", "SkirmishState", "Commander"]
