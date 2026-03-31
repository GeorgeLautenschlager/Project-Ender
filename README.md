# Ender

Oracle-distilled game AI. An LLM acts as a tactical commander during live play,
logging decisions that train a small neural network. Reinforcement learning then
refines the network beyond the oracle's capabilities.

See [ARCHITECTURE.md](ARCHITECTURE.md) for design details and [DELIVERY.md](DELIVERY.md)
for the milestone roadmap.

---

## Status

**M1 complete** — Claude plays Skirmish; decisions logged to SQLite; corpus pipeline runs end-to-end.

| Milestone | Goal | Status |
|-----------|------|--------|
| M0 | Repo scaffold + Skirmish game loop | Done |
| M1 | Oracle integration (Claude plays Skirmish) | Done |
| M2 | Behavioural cloning (policy network) | Planned |
| M3 | RL fine-tuning | Planned |
| M4 | Feudal Carriers integration | Planned |
| M5 | DCS integration | In progress |

---

## Quick Start

```bash
pip install poetry
poetry install
```

### Play Skirmish (two humans, no server needed)

```bash
python -m project_ender.skirmish
```

### Validate the full pipeline (random bots via Ender socket)

Terminal 1 — start the Ender server:
```bash
python -m project_ender.server
```

Terminal 2 — run a game through the socket:
```bash
python -m project_ender.skirmish --red ender --blue ender
```

### Watch the oracle play (no server needed)

The oracle (Claude CLI) can drive either or both sides directly, without the
Ender socket server:

```bash
# Oracle (Red) vs random bot (Blue)
python -m project_ender.skirmish --red oracle --blue random

# Oracle vs oracle — watch two commanders reason against each other
python -m project_ender.skirmish --red oracle --blue oracle

# Human (Red) vs oracle (Blue)
python -m project_ender.skirmish --blue oracle
```

Each oracle turn prints the chosen action and the model's reasoning.

**Confidence threshold** — the Commander logs a warning when oracle confidence
falls below the threshold (M2 will use this to switch to the policy network):

```bash
python -m project_ender.skirmish --red oracle --blue oracle --confidence-threshold 0.7
```

### Oracle via Ender server

The server can also run in oracle mode, answering every `StateRequest` with a
real model decision:

```bash
# Terminal 1 — server with oracle backend
python -m project_ender.server --oracle

# Terminal 2 — Skirmish clients talk to the oracle server
python -m project_ender.skirmish --red ender --blue ender
```

### Other modes

```bash
# Human (Red) vs random bot (Blue) — no server needed
python -m project_ender.skirmish --blue random

# Human (Red) vs Ender server (Blue)
python -m project_ender.skirmish --blue ender
```

### Corpus pipeline (label states → build consensus)

```bash
# 1. Generate a corpus of game states via scripted self-play
python -m project_ender.corpus --games 50 --db corpus.db

# 2. Label unlabelled states with the oracle
python -m project_ender.labeller --db corpus.db --backend claude

# 3. Build soft-target consensus from all labels
python -m project_ender.consensus --db corpus.db
```

---

## DCS Integration (M5)

Ender can connect to a running DCS World mission via a Lua socket server
embedded in the `.miz` file and an MCP server that bridges Claude to DCS over
the network.

### Prerequisites

```bash
pip install pydcs mcp
# or, if using poetry:
poetry install -E dcs
```

Optional: download [MOOSE.lua](https://github.com/FlightControl-Master/MOOSE/releases)
and place it at `lua/vendor/MOOSE.lua`. The mission builder will auto-detect and
embed it. If absent, the mission still works — MOOSE features just won't be
available in-mission.

### 1. Generate a test mission

```bash
python -m project_ender.dcs.build_mission --output test.miz
```

This produces a Caucasus mission with:
- 2× FA-18C at Batumi (Blue, cold on ramp)
- 3× T-80UD near Kobuleti (Red)
- Lua TCP socket server loaded at mission start (port 7374)

### 2. Run the mission in DCS

Copy `test.miz` to your Windows DCS machine and open it. When the mission
starts, check `DCS.log` or the DCS scripting console for:

```
[Ender] DCS MCP socket server listening on :7374
```

Make sure the Windows firewall allows inbound TCP on port 7374.

### 3. Start the MCP server

On your dev machine (Linux/Mac), point the MCP server at the DCS host:

```bash
DCS_HOST=192.168.1.x python -m project_ender.dcs.mcp_server
```

This starts a stdio MCP server exposing two tools:

| Tool | Description |
|------|-------------|
| `get_mission_state()` | Returns all live units (name, type, coalition, lat/lon, heading) and airbases |
| `spawn_flight(airport, aircraft_type, count, callsign, coalition)` | Spawns a flight cold on the ramp at the given airbase |

### 4. Connect from Claude

Add the server to your MCP client config (e.g. Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "dcs-ender": {
      "command": "python",
      "args": ["-m", "project_ender.dcs.mcp_server"],
      "env": { "DCS_HOST": "192.168.1.x" }
    }
  }
}
```

Then ask Claude to `get_mission_state()` to see what's on the map, or
`spawn_flight(airport="Batumi")` to put more aircraft on the ramp.

### Architecture

```
┌──────────────┐         TCP :7374         ┌──────────────────┐
│  MCP Server  │◄────────────────────────► │  DCS World       │
│  (Python,    │    JSON lines over LAN    │  (Windows PC)    │
│   Linux)     │                           │                  │
└──────┬───────┘                           │  dcs_mcp_server  │
       │ stdio                             │  .lua (embedded) │
┌──────┴───────┐                           └──────────────────┘
│  Claude      │
│  (MCP client)│
└──────────────┘
```

---

## Project Layout

```
src/project_ender/
├── adapter.py          # DomainAdapter ABC — implement this per game
├── protocol.py         # Wire protocol: StateRequest, ActionResponse, RewardEvent
├── server.py           # Ender socket server (random or oracle mode)
├── client.py           # Thin Python client for the Ender socket
├── commander.py        # Commander: oracle-only now; policy blend in M2
├── logger.py           # DecisionLogger — SQLite writer, multi-teacher schema
├── corpus.py           # Corpus generation: scripted self-play → states table
├── labeller.py         # Labelling runner: queries oracle, writes labels
├── consensus.py        # Consensus builder: merges labels into soft targets
├── skirmish/
│   ├── adapter.py      # SkirmishAdapter (encode_state, decode_action, compute_reward)
│   ├── game.py         # Game state, rules, step(), terminal UI
│   └── __main__.py     # CLI entry point
└── dcs/
    ├── mission_builder.py  # PyDCS: generates .miz with embedded Lua server
    └── mcp_server.py       # MCP server: get_state + spawn_flight tools
lua/
├── dcs_mcp_server.lua      # Lua TCP server embedded in missions
└── vendor/
    └── MOOSE.lua            # (download separately — not committed)
```

---

## Development

```bash
make test       # pytest + coverage
make lint       # ruff
make typecheck  # mypy
make format     # black
make check      # all of the above
```
