# Ender

Oracle-distilled game AI. An LLM acts as a tactical commander during live play,
logging decisions that train a small neural network. Reinforcement learning then
refines the network beyond the oracle's capabilities.

See [ARCHITECTURE.md](ARCHITECTURE.md) for design details and [DELIVERY.md](DELIVERY.md)
for the milestone roadmap.

---

## Status

**M0 complete** — runnable skeleton with Skirmish demo game.

| Milestone | Goal | Status |
|-----------|------|--------|
| M0 | Repo scaffold + Skirmish game loop | Done |
| M1 | Oracle integration (Claude plays Skirmish) | Planned |
| M2 | Behavioural cloning (policy network) | Planned |
| M3 | RL fine-tuning | Planned |
| M4 | Feudal Carriers integration | Planned |
| M5 | DCS integration | Planned |

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

### Other modes

```bash
# Human (Red) vs random bot (Blue) — no server needed
python -m project_ender.skirmish --blue random

# Human (Red) vs Ender server (Blue)
python -m project_ender.skirmish --blue ender
```

---

## Project Layout

```
src/project_ender/
├── adapter.py          # DomainAdapter ABC — implement this per game
├── protocol.py         # Wire protocol: StateRequest, ActionResponse, RewardEvent
├── server.py           # Ender socket server (M0 stub: random valid actions)
├── client.py           # Thin Python client for the Ender socket
└── skirmish/
    ├── adapter.py      # SkirmishAdapter (encode_state, decode_action, compute_reward)
    ├── game.py         # Game state, rules, step(), terminal UI
    └── __main__.py     # CLI entry point
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
