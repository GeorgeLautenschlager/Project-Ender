"""
mcp_server.py
Project Ender — DCS MCP server

Exposes two tools that let Claude (or any MCP client) query and command a
running DCS mission via the Lua socket server embedded in the .miz file.

Configuration (environment variables):
    DCS_HOST  — IP address of the Windows machine running DCS (default: 127.0.0.1)
    DCS_PORT  — TCP port of the Lua socket server (default: 7374)

Usage:
    DCS_HOST=192.168.1.x python -m project_ender.dcs.mcp_server

Then add this server to your MCP client config (e.g. Claude Desktop):
    {
      "mcpServers": {
        "dcs-ender": {
          "command": "python",
          "args": ["-m", "project_ender.dcs.mcp_server"],
          "env": {"DCS_HOST": "192.168.1.x"}
        }
      }
    }
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DCS_HOST = os.environ.get("DCS_HOST", "127.0.0.1")
DCS_PORT = int(os.environ.get("DCS_PORT", "7374"))
TIMEOUT = 10  # seconds

# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _send_command(cmd: dict[str, Any]) -> dict[str, Any]:
    """
    Open a short-lived TCP connection to the DCS Lua socket server,
    send *cmd* as a JSON line, and return the parsed response.

    Raises:
        ConnectionRefusedError: DCS not running or server not loaded.
        TimeoutError: Server did not respond within TIMEOUT seconds.
        ValueError: Response is not valid JSON.
    """
    payload = json.dumps(cmd).encode() + b"\n"
    with socket.create_connection((DCS_HOST, DCS_PORT), timeout=TIMEOUT) as sock:
        sock.sendall(payload)
        # Read until newline
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
    line = buf.split(b"\n")[0]
    return json.loads(line)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="dcs-ender",
    instructions=(
        "Tools for querying and commanding a live DCS World mission. "
        f"Connects to {DCS_HOST}:{DCS_PORT}."
    ),
)


@mcp.tool()
def get_mission_state() -> dict[str, Any]:
    """
    Return the current DCS mission state.

    Response includes:
      - time: mission clock in seconds
      - units: list of all live aircraft (name, type, coalition, lat, lon, alt, heading, alive)
      - airbases: list of all airbases with their current coalition

    Call this first to orient yourself before issuing commands.
    """
    result = _send_command({"cmd": "get_state"})
    if not result.get("ok"):
        raise RuntimeError(f"DCS error: {result.get('error', 'unknown')}")
    return result.get("data", result)  # type: ignore[return-value]


@mcp.tool()
def spawn_flight(
    airport: str,
    aircraft_type: str = "FA-18C_hornet",
    count: int = 2,
    callsign: str = "Enfield",
    coalition: str = "blue",
) -> str:
    """
    Spawn a flight of aircraft at the given airbase.

    Args:
        airport: Exact DCS airbase name, e.g. "Batumi", "Kutaisi", "Tbilisi-Lochini".
                 Use get_mission_state() to see available airbases.
        aircraft_type: DCS aircraft type ID, e.g. "FA-18C_hornet", "F-16C_50", "Su-27".
        count: Number of aircraft to spawn (1–4).
        callsign: Group callsign prefix used for naming (e.g. "Enfield", "Dodge").
        coalition: "blue" or "red".

    Returns a confirmation string with the spawned group name.
    Spawned aircraft start cold and uncontrolled on the ramp.
    """
    count = max(1, min(4, count))
    result = _send_command({
        "cmd": "spawn_flight",
        "airport": airport,
        "type": aircraft_type,
        "count": count,
        "callsign": callsign,
        "coalition": coalition,
    })
    if not result.get("ok"):
        raise RuntimeError(f"DCS error: {result.get('error', 'unknown')}")
    data = result.get("data", {})
    group_name = data.get("group_name", "unknown")
    unit_count = data.get("unit_count", count)
    return f"Spawned group '{group_name}' ({unit_count}× {aircraft_type}) at {airport}."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger(__name__).info(
        "DCS MCP server starting — will connect to %s:%d", DCS_HOST, DCS_PORT
    )
    mcp.run()


if __name__ == "__main__":
    main()
