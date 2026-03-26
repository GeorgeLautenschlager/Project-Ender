"""Skirmish game state and logic.

Rules:
- 1-D map: 7 zones in a line (zones 0-6)
- Two commanders: Red (left, home zones 0-2) and Blue (right, home zones 4-6)
- Each commander starts with 6 units spread across their home zones
- Zone held = more friendly units than enemy units in that zone
- Win: hold >= 4 of 7 zones for 3 consecutive turns, OR eliminate all enemy units
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, NamedTuple

if TYPE_CHECKING:
    pass

NUM_ZONES = 7
STARTING_UNITS = 6
WIN_ZONES = 4
WIN_HOLD_TURNS = 3
MAX_TURNS = 200


class Commander(Enum):
    RED = "Red"
    BLUE = "Blue"


class GameResult(NamedTuple):
    winner: Commander | None  # None = draw (max turns reached)
    turn: int
    reason: str


# PlayerFn: callable (state, valid_action_ids) -> chosen_action_id
PlayerFn = Callable[["SkirmishState", list[int]], int]


@dataclass
class SkirmishState:
    """
    Full game state.

    `units[RED][zone]` and `units[BLUE][zone]` track unit counts per zone.
    Perspective is absolute (not relative to the current player).
    """

    units: dict[Commander, list[int]] = field(default_factory=dict)
    turn: int = 0
    # How many consecutive turns each commander has held >= WIN_ZONES zones
    consecutive_hold: dict[Commander, int] = field(default_factory=dict)
    result: GameResult | None = None

    @classmethod
    def initial(cls) -> SkirmishState:
        """Set up starting positions: 6 units split across home zones."""
        red_units = [2, 2, 2, 0, 0, 0, 0]
        blue_units = [0, 0, 0, 0, 2, 2, 2]
        return cls(
            units={Commander.RED: red_units, Commander.BLUE: blue_units},
            consecutive_hold={Commander.RED: 0, Commander.BLUE: 0},
        )

    def is_terminal(self) -> bool:
        return self.result is not None

    def total_units(self, commander: Commander) -> int:
        return sum(self.units[commander])

    def zones_held(self, commander: Commander) -> int:
        """Count zones where commander has strictly more units than opponent."""
        opp = Commander.BLUE if commander == Commander.RED else Commander.RED
        return sum(
            1
            for z in range(NUM_ZONES)
            if self.units[commander][z] > self.units[opp][z]
        )

    def active_commander(self) -> Commander:
        """Whose turn it is (alternating, Red goes first)."""
        return Commander.RED if self.turn % 2 == 0 else Commander.BLUE

    def copy(self) -> SkirmishState:
        return SkirmishState(
            units={
                Commander.RED: list(self.units[Commander.RED]),
                Commander.BLUE: list(self.units[Commander.BLUE]),
            },
            turn=self.turn,
            consecutive_hold=dict(self.consecutive_hold),
            result=self.result,
        )


def _apply_action(state: SkirmishState, commander: Commander, action_id: int) -> None:
    """
    Mutate *state* by executing *action_id* for *commander*.

    Actions:
        0-6:  Reinforce zone N — move 1 unit from best adjacent zone
        7-13: Attack zone N   — move 2 units from best adjacent zone
        14:   Hold
        15:   Feint left      — move 1 unit from rightmost occupied zone leftward
        16:   Feint right     — move 1 unit from leftmost occupied zone rightward
        17:   Concentrate     — move 1 unit toward zone 3 from nearest flank zone
    """
    u = state.units[commander]

    if 0 <= action_id <= 6:
        target = action_id
        src = _best_adjacent(u, target, needed=1)
        if src is not None:
            u[src] -= 1
            u[target] += 1

    elif 7 <= action_id <= 13:
        target = action_id - 7
        src = _best_adjacent(u, target, needed=2)
        if src is not None:
            u[src] -= 2
            u[target] += 2

    elif action_id == 14:
        pass  # Hold

    elif action_id == 15:  # Feint left
        src = max((z for z in range(1, NUM_ZONES) if u[z] > 0), default=None)
        if src is not None:
            u[src] -= 1
            u[src - 1] += 1

    elif action_id == 16:  # Feint right
        src = min((z for z in range(NUM_ZONES - 1) if u[z] > 0), default=None)
        if src is not None:
            u[src] -= 1
            u[src + 1] += 1

    elif action_id == 17:  # Concentrate centre
        candidates = [z for z in range(NUM_ZONES) if z != 3 and u[z] > 0]
        if candidates:
            src = max(candidates, key=lambda z: abs(z - 3))
            u[src] -= 1
            u[src + 1 if src < 3 else src - 1] += 1


def _best_adjacent(units: list[int], target: int, needed: int) -> int | None:
    """Return the adjacent zone index with the most units (>= needed), or None."""
    candidates = [
        z
        for z in (target - 1, target + 1)
        if 0 <= z < NUM_ZONES and units[z] >= needed
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda z: units[z])


def _resolve_combat(state: SkirmishState) -> None:
    """
    Attrition: in any zone occupied by both sides, the smaller force is
    eliminated and the larger loses an equal number of units.
    """
    for z in range(NUM_ZONES):
        r = state.units[Commander.RED][z]
        b = state.units[Commander.BLUE][z]
        if r > 0 and b > 0:
            fought = min(r, b)
            state.units[Commander.RED][z] -= fought
            state.units[Commander.BLUE][z] -= fought


def step(
    state: SkirmishState,
    action_id: int,
) -> tuple[SkirmishState, float, float]:
    """
    Advance the game by one action.

    Returns:
        (new_state, reward_active, reward_opponent)
        where *active* is the commander whose turn it was.
    """
    if state.is_terminal():
        raise ValueError("Game is already over")

    new = state.copy()
    active = new.active_commander()
    opp = Commander.BLUE if active == Commander.RED else Commander.RED

    # Snapshot zone control before the action
    held_before = new.zones_held(active)
    opp_held_before = new.zones_held(opp)
    opp_units_before = new.total_units(opp)
    active_units_before = new.total_units(active)

    _apply_action(new, active, action_id)
    _resolve_combat(new)

    # Reward computation
    reward_active = 0.0
    reward_opp = 0.0

    held_after = new.zones_held(active)
    opp_held_after = new.zones_held(opp)

    zones_captured = max(0, held_after - held_before)
    zones_lost = max(0, opp_held_after - opp_held_before)

    reward_active += zones_captured * 0.3
    reward_active -= zones_lost * 0.3
    reward_opp += zones_lost * 0.3
    reward_opp -= zones_captured * 0.3

    enemy_eliminated = max(0, opp_units_before - new.total_units(opp))
    friendly_lost = max(0, active_units_before - new.total_units(active))
    reward_active += enemy_eliminated * 0.1
    reward_active -= friendly_lost * 0.1
    reward_opp += friendly_lost * 0.1
    reward_opp -= enemy_eliminated * 0.1

    reward_active -= 0.01  # time pressure

    # Update consecutive hold tracking
    for cmd in Commander:
        if new.zones_held(cmd) >= WIN_ZONES:
            new.consecutive_hold[cmd] += 1
        else:
            new.consecutive_hold[cmd] = 0

    new.turn += 1

    # Check terminal conditions
    if new.total_units(opp) == 0:
        new.result = GameResult(winner=active, turn=new.turn, reason="elimination")
        reward_active += 1.0
        reward_opp -= 1.0
    elif new.total_units(active) == 0:
        new.result = GameResult(winner=opp, turn=new.turn, reason="elimination")
        reward_active -= 1.0
        reward_opp += 1.0
    elif new.consecutive_hold[active] >= WIN_HOLD_TURNS:
        new.result = GameResult(winner=active, turn=new.turn, reason="zone control")
        reward_active += 1.0
        reward_opp -= 1.0
    elif new.consecutive_hold[opp] >= WIN_HOLD_TURNS:
        new.result = GameResult(winner=opp, turn=new.turn, reason="zone control")
        reward_active -= 1.0
        reward_opp += 1.0
    elif new.turn >= MAX_TURNS:
        new.result = GameResult(winner=None, turn=new.turn, reason="turn limit")

    return new, reward_active, reward_opp


class SkirmishGame:
    """
    Convenience wrapper that manages the game loop.

    Players are callables: (state, valid_actions) -> action_id
    """

    def __init__(
        self,
        red_player: PlayerFn,
        blue_player: PlayerFn,
    ) -> None:
        self._players = {Commander.RED: red_player, Commander.BLUE: blue_player}

    def run(self, verbose: bool = True) -> GameResult:
        from project_ender.skirmish.adapter import SkirmishAdapter

        adapter = SkirmishAdapter()
        state = SkirmishState.initial()
        if verbose:
            _print_state(state)

        while not state.is_terminal():
            active = state.active_commander()
            valid = adapter.valid_action_ids(state)
            action_id = self._players[active](state, valid)
            state, _, _ = step(state, action_id)
            if verbose:
                _print_state(state)

        assert state.result is not None
        if verbose:
            _print_result(state.result)
        return state.result


# ── Terminal UI ───────────────────────────────────────────────────────────────

_ZONE_LABELS = [str(i) for i in range(NUM_ZONES)]


def _print_state(state: SkirmishState) -> None:
    r = state.units[Commander.RED]
    b = state.units[Commander.BLUE]
    active_name = state.active_commander().value

    print(f"\n-- Turn {state.turn} -- {active_name} to move --")
    print(f"  Zone  : {'  '.join(_ZONE_LABELS)}")
    print(f"  Red   : {'  '.join(str(x).rjust(1) for x in r)}")
    print(f"  Blue  : {'  '.join(str(x).rjust(1) for x in b)}")

    control = []
    for z in range(NUM_ZONES):
        if r[z] > b[z]:
            control.append("R")
        elif b[z] > r[z]:
            control.append("B")
        else:
            control.append("-")
    print(f"  Ctrl  : {'  '.join(control)}")

    rh = state.zones_held(Commander.RED)
    bh = state.zones_held(Commander.BLUE)
    rc = state.consecutive_hold[Commander.RED]
    bc = state.consecutive_hold[Commander.BLUE]
    print(f"  Held  : Red={rh} (streak {rc})  Blue={bh} (streak {bc})")
    ru = state.total_units(Commander.RED)
    bu = state.total_units(Commander.BLUE)
    print(f"  Units : Red={ru}  Blue={bu}")


def _print_result(result: GameResult) -> None:
    print("\n" + "=" * 48)
    if result.winner:
        winner = result.winner.value
        print(f"  {winner} wins by {result.reason} on turn {result.turn}!")
    else:
        print(f"  Draw -- turn limit reached ({result.turn} turns).")
    print("=" * 48)
