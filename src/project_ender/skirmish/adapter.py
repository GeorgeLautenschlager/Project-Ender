"""SkirmishAdapter — DomainAdapter implementation for the Skirmish demo game."""

from __future__ import annotations

from typing import Any

from project_ender.adapter import Action, DomainAdapter
from project_ender.skirmish.game import (
    MAX_TURNS,
    NUM_ZONES,
    WIN_HOLD_TURNS,
    Commander,
    SkirmishState,
)

# Action space: 18 actions as defined in DELIVERY.md
_ACTION_SPACE: list[Action] = [
    *[
        Action(
            id=z,
            name=f"Reinforce zone {z}",
            description=f"Move 1 unit from an adjacent friendly zone into zone {z}",
        )
        for z in range(NUM_ZONES)
    ],
    *[
        Action(
            id=7 + z,
            name=f"Attack zone {z}",
            description=(
                f"Commit 2 units from an adjacent friendly zone into zone {z}"
            ),
        )
        for z in range(NUM_ZONES)
    ],
    Action(id=14, name="Hold", description="No movement; consolidate in place"),
    Action(
        id=15,
        name="Feint left",
        description="Move 1 unit from the rightmost occupied zone one step left",
    ),
    Action(
        id=16,
        name="Feint right",
        description="Move 1 unit from the leftmost occupied zone one step right",
    ),
    Action(
        id=17,
        name="Concentrate centre",
        description="Move 1 unit from the flank zone furthest from zone 3 to centre",
    ),
]


class SkirmishAdapter(DomainAdapter):
    """
    Translates between SkirmishState and the Ender protocol.

    State vector (~20 floats, from the active commander's perspective):
        friendly_units_z0..z6   — 7 floats, normalised by STARTING_UNITS*2
        enemy_units_z0..z6      — 7 floats
        zones_held_friendly      — float 0–1
        zones_held_enemy         — float 0–1
        turns_elapsed            — float 0–1
        consecutive_hold_count   — float 0–1
    """

    _MAX_UNITS_PER_ZONE = 12.0  # normalisation ceiling

    @property
    def action_space(self) -> list[Action]:
        return _ACTION_SPACE

    def encode_state(self, world: Any) -> list[float]:
        state: SkirmishState = world
        active = state.active_commander()
        opp = Commander.BLUE if active == Commander.RED else Commander.RED

        friendly = [
            state.units[active][z] / self._MAX_UNITS_PER_ZONE for z in range(NUM_ZONES)
        ]
        enemy = [
            state.units[opp][z] / self._MAX_UNITS_PER_ZONE for z in range(NUM_ZONES)
        ]
        zones_held_f = state.zones_held(active) / NUM_ZONES
        zones_held_e = state.zones_held(opp) / NUM_ZONES
        turns = state.turn / MAX_TURNS
        hold_streak = state.consecutive_hold[active] / WIN_HOLD_TURNS

        return [
            *friendly,
            *enemy,
            zones_held_f,
            zones_held_e,
            turns,
            hold_streak,
        ]

    def encode_state_summary(self, world: Any) -> str:
        state: SkirmishState = world
        active = state.active_commander()
        opp = Commander.BLUE if active == Commander.RED else Commander.RED
        u = state.units

        lines = [
            f"Turn {state.turn}. You are {active.value}.",
            "",
            "Zone layout (0=left, 6=right):",
        ]
        for z in range(NUM_ZONES):
            f_units = u[active][z]
            e_units = u[opp][z]
            if f_units > e_units:
                ctrl = "YOUR zone"
            elif e_units > f_units:
                ctrl = "ENEMY zone"
            else:
                ctrl = "contested" if f_units > 0 else "empty"
            lines.append(f"  Zone {z}: you={f_units}, enemy={e_units} — {ctrl}")

        lines += [
            "",
            f"You hold {state.zones_held(active)} zones "
            f"(streak: {state.consecutive_hold[active]} turns). "
            f"Enemy holds {state.zones_held(opp)} zones "
            f"(streak: {state.consecutive_hold[opp]} turns).",
            f"Your units: {state.total_units(active)}. "
            f"Enemy units: {state.total_units(opp)}.",
            "Win condition: hold ≥4 zones for 3 consecutive turns, "
            "or eliminate all enemy units.",
        ]
        return "\n".join(lines)

    def decode_action(self, action_id: int) -> str:
        for a in _ACTION_SPACE:
            if a.id == action_id:
                return a.name
        raise ValueError(f"Unknown action_id: {action_id}")

    def compute_reward(self, event: Any) -> float:
        """
        Event dict expected keys: reward (float).
        The game loop computes rewards and passes them in; this method
        simply extracts the float for the protocol layer.
        """
        if isinstance(event, dict):
            return float(event.get("reward", 0.0))
        return float(event)

    def valid_action_ids(self, world: Any) -> list[int]:
        """Filter actions that have no legal source zone."""
        state: SkirmishState = world
        active = state.active_commander()
        u = state.units[active]
        valid: list[int] = []

        # Reinforce zone N (0–6): need adjacent zone with ≥1 unit
        for z in range(NUM_ZONES):
            adj = [zz for zz in (z - 1, z + 1) if 0 <= zz < NUM_ZONES and u[zz] >= 1]
            if adj:
                valid.append(z)

        # Attack zone N (7–13): need adjacent zone with ≥2 units
        for z in range(NUM_ZONES):
            adj = [zz for zz in (z - 1, z + 1) if 0 <= zz < NUM_ZONES and u[zz] >= 2]
            if adj:
                valid.append(7 + z)

        # Hold (14): always valid
        valid.append(14)

        # Feint left (15): need any zone in 1–6 with ≥1 unit
        if any(u[z] >= 1 for z in range(1, NUM_ZONES)):
            valid.append(15)

        # Feint right (16): need any zone in 0–5 with ≥1 unit
        if any(u[z] >= 1 for z in range(NUM_ZONES - 1)):
            valid.append(16)

        # Concentrate (17): need any zone ≠3 with ≥1 unit
        if any(u[z] >= 1 for z in range(NUM_ZONES) if z != 3):
            valid.append(17)

        return sorted(set(valid))
