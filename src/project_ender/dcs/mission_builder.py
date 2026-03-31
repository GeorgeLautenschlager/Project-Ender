"""
mission_builder.py
Project Ender — PyDCS mission generator

Generates a simple Caucasus test mission with:
  - Batumi airfield (Blue coalition)
  - 2× FA-18C_hornet parked cold on the ramp
  - 3× T-80UD vehicles near Kobuleti (Red coalition, something to query)
  - Two mission-start triggers: load MOOSE.lua, then load dcs_mcp_server.lua

Usage:
    python -m project_ender.dcs.build_mission [--output test.miz] [--moose path/to/MOOSE.lua]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Lua files relative to the project root
_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/project_ender/dcs -> repo root
_DEFAULT_SERVER_LUA = _REPO_ROOT / "lua" / "dcs_mcp_server.lua"
_DEFAULT_MOOSE_LUA = _REPO_ROOT / "lua" / "vendor" / "MOOSE.lua"


def build_mission(
    output: Path,
    moose_lua: Path | None = None,
    server_lua: Path = _DEFAULT_SERVER_LUA,
) -> None:
    """
    Generate a .miz file and write it to *output*.

    Args:
        output: Destination path for the generated mission file.
        moose_lua: Path to MOOSE.lua. If None, skips the MOOSE trigger.
        server_lua: Path to dcs_mcp_server.lua.
    """
    try:
        import dcs
        from dcs import action, triggers
        from dcs.planes import FA_18C_hornet
        from dcs.vehicles import Armor
    except ImportError as e:
        raise RuntimeError(
            "pydcs is required for mission generation. "
            "Install it with: pip install pydcs"
        ) from e

    m = dcs.Mission()  # Caucasus map by default
    m.set_description_text(
        "Project Ender test mission — Batumi training area. "
        "Socket server on port 7374."
    )


    # -------------------------------------------------------------------------
    # Coalitions / Countries
    # -------------------------------------------------------------------------
    usa = m.coalition["blue"].country("USA")
    russia = m.coalition["red"].country("Russia")

    # -------------------------------------------------------------------------
    # Batumi airport — Blue FA-18C flight, cold on ramp
    # -------------------------------------------------------------------------
    batumi = m.terrain.airports["Batumi"]

    blue_fg = m.flight_group_from_airport(
        country=usa,
        name="Enfield",
        aircraft_type=FA_18C_hornet,
        airport=batumi,
        start_type=dcs.mission.StartType.Cold,
        group_size=2,
    )
    blue_fg.uncontrolled = True  # parked/cold — won't taxi unless activated

    log.info("Added Blue flight: Enfield (2× FA-18C) at Batumi")

    # -------------------------------------------------------------------------
    # Red vehicle group near Kobuleti
    # -------------------------------------------------------------------------
    # Kobuleti is ~20 km NE of Batumi; approximate DCS coords:
    kobuleti_x = -319_500
    kobuleti_y = 625_500

    red_vg = m.vehicle_group(
        country=russia,
        name="Red Armour",
        _type=Armor.T_80UD,
        position=dcs.mapping.Point(kobuleti_x, kobuleti_y, m.terrain),
        group_size=3,
        heading=270,
    )
    log.info("Added Red vehicle group: 3× T-80UD near Kobuleti")

    # -------------------------------------------------------------------------
    # Embed Lua scripts as mission resources, trigger on start
    # -------------------------------------------------------------------------
    def _add_script_trigger(lua_path: Path, comment: str) -> None:
        if not lua_path.exists():
            log.warning("Lua script not found, skipping trigger: %s", lua_path)
            return
        res_key = m.map_resource.add_resource_file(str(lua_path))
        trigger = triggers.TriggerStart(comment=comment)
        trigger.add_action(action.DoScriptFile(res_key))
        m.triggerrules.triggers.append(trigger)
        log.info("Added trigger '%s' → %s", comment, lua_path.name)

    # MOOSE first (must initialise before server script uses it)
    if moose_lua is not None:
        _add_script_trigger(moose_lua, "Load MOOSE")
    else:
        log.info("No MOOSE path provided — skipping MOOSE trigger")

    _add_script_trigger(server_lua, "Load Ender MCP server")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    log.info("Mission saved: %s", output)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate a Project Ender DCS test mission")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("test.miz"),
        help="Output .miz file path (default: test.miz)",
    )
    parser.add_argument(
        "--moose",
        type=Path,
        default=None,
        help=(
            "Path to MOOSE.lua. Defaults to lua/vendor/MOOSE.lua if it exists. "
            "Omit to skip MOOSE loading."
        ),
    )
    args = parser.parse_args()

    # Auto-detect MOOSE if not specified
    moose_path: Path | None = args.moose
    if moose_path is None and _DEFAULT_MOOSE_LUA.exists():
        moose_path = _DEFAULT_MOOSE_LUA
        log.info("Auto-detected MOOSE at %s", moose_path)

    build_mission(output=args.output, moose_lua=moose_path)


if __name__ == "__main__":
    main()
