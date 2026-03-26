# Ender — Delivery Plan

## Philosophy

Build the thinnest possible slice that demonstrates the full pipeline working end-to-end.
The bundled demo game exists to make that slice self-contained — no external game required
to try Ender out of the box.

---

## Bundled Demo: *Skirmish*

A minimal tactical game bundled with Ender to validate and demonstrate the full pipeline.
Small enough to implement in an afternoon. Complex enough to require real strategic decisions.

### Rules

- 1D map: 7 zones in a line (zones 0–6)
- Two commanders: Red (left) and Blue (right)
- Each commander starts with 6 units distributed across their home zones
- Each turn, a commander issues one order
- Zones with more friendly units than enemy units are "contested" or "held"
- Win condition: hold 4 of 7 zones for 3 consecutive turns, or eliminate all enemy units

### State Vector (per commander, ~20 floats)

```
[
  friendly_units_z0..z6,   # 7 floats: unit count per zone, normalised
  enemy_units_z0..z6,      # 7 floats
  zones_held_friendly,     # float: 0–1
  zones_held_enemy,        # float: 0–1
  turns_elapsed,           # float: normalised
  consecutive_hold_count,  # float: normalised
]
```

### Action Space (18 actions)

```
0–6:   Reinforce zone N (move 1 unit from adjacent friendly zone)
7–13:  Attack zone N (commit 2 units from adjacent zone)
14:    Hold (no movement, consolidate)
15:    Feint left (redistribute units toward left flank)
16:    Feint right (redistribute units toward right flank)
17:    Concentrate centre (pull units toward zone 3)
```

### Reward Signals

```
+0.3   capture a zone
-0.3   lose a zone
+0.1   per enemy unit eliminated
-0.1   per friendly unit lost
+1.0   win the game
-1.0   lose the game
-0.01  per turn elapsed (time pressure)
```

### Why This Works as a Demo

- State vector is tiny and human-readable
- Action space is small enough to train quickly
- Win/loss is unambiguous
- The 1D map is easy to render in a terminal or simple UI
- A human can watch the oracle's reasoning and see whether it makes sense
- Self-play is immediately available (two Skirmish instances)

---

## Milestones

### M0 — Repo Scaffold
*Goal: runnable skeleton, nothing trained*

- [x] Python package structure (`project_ender/`)
- [x] `DomainAdapter` abstract base class
- [x] Ender protocol (JSON schema, local socket server stub)
- [x] Thin Python client (for Skirmish)
- [x] `SkirmishAdapter` implementation
- [x] Skirmish game loop (terminal UI, two human players or scripted)
- [x] README pointing at this doc set

**Exit criterion:** Two humans can play Skirmish through the Ender socket.

---

### M1 — Oracle Integration
*Goal: Claude makes decisions in Skirmish*

**Decide before starting:**
- **Oracle cost controls.** During collection the oracle runs on every decision. Decide on
  a configurable episode cap or token budget to bound API spend before the first run.
  A reasonable default: cap collection at 500 episodes (~5,000 states) for the initial
  training corpus. This fits overnight on local models and keeps Claude API costs bounded
  during validation runs.
- **Single-teacher or multi-teacher from the start?** Starting with Claude-only is simpler
  for M1 validation. The corpus schema should support multiple teachers from day one so
  local model labelling passes can be added in M2 without a migration.

- [ ] Oracle layer (Anthropic API call, prompt template, structured output parser)
- [ ] Decision logger (SQLite, schema supports multiple teachers: `state_id`, `teacher`,
      `action_id`, `confidence`, `reasoning`, `timestamp`)
- [ ] Corpus generation script (scripted self-play → unlabelled states table)
- [ ] Labelling runner (loads teacher, labels corpus, writes to labels table, unloads)
- [ ] Consensus builder (merges label sets into soft_targets + agreement_score)
- [ ] Commander blend (oracle-only mode, confidence threshold stub)
- [ ] Skirmish human vs. oracle mode

**Exit criterion:** Claude plays Skirmish. Decisions are logged to SQLite with reasoning.
Corpus pipeline runs end-to-end: generate states → label → compute consensus.
Watch it play a few games and verify the reasoning field makes sense.

---

### M2 — Behavioural Cloning
*Goal: trained network plays like the oracle(s)*

- [ ] Policy network (PyTorch MLP, configurable depth/width)
- [ ] Behavioural cloning trainer (soft-target cross-entropy, train/val split, loss curves)
- [ ] Local model labelling runs (Qwen3-14B, GPT-OSS-20B via Ollama — overnight sequential)
- [ ] Consensus rebuild after each new teacher pass
- [ ] Policy inference endpoint (forward pass, softmax, confidence score)
- [ ] Commander blend (policy mode, threshold switching)
- [ ] Skirmish oracle vs. policy mode (watch them play each other)

**Exit criterion:** Policy trained on multi-teacher consensus corpus. Plays recognisable
Skirmish strategy. Oracle vs. policy win rate is somewhere interesting (not 0% or 100%).
Agreement score distribution logged — verify high-agreement states produce cleaner training
signal than low-agreement ones.

---

### M3 — RL Fine-tuning
*Goal: policy improves beyond oracle baseline*

- [ ] Gymnasium environment wrapper for Skirmish
- [ ] PPO training loop (Stable-Baselines3)
- [ ] Reward event integration
- [ ] Training metrics (win rate over time, reward curve, vs-oracle benchmark)
- [ ] Self-play mode (two policy instances, shared replay buffer)

**Exit criterion:** RL policy win rate vs. oracle improves measurably over training.
Ideally produces at least one emergent behaviour worth documenting.

---

### M4 — Feudal Carriers Integration
*Goal: prove the protocol works across a language boundary*

- [ ] Java client library (`EnderClient.java`, thin socket wrapper)
- [ ] `FeudalCarriersAdapter` (encode_state, decode_action, compute_reward)
- [ ] Integration test: FC game loop talking to Ender server
- [ ] FC oracle commander running in development build

**Exit criterion:** Claude commands a fleet in Feudal Carriers. Decisions logged.

---

### M5 — DCS Integration
*Goal: prove the protocol works in Lua*

- [ ] Lua client (MOOSE-compatible, socket or HTTP)
- [ ] DCS mission adapter (state from DCS scripting API, action space defined)
- [ ] Oracle commander running in a test mission

**Exit criterion:** Claude commands a DCS mission. Logged decisions match observable
in-game behaviour.

---

## Open Questions

- **Ender server as subprocess vs. daemon?** For game integration, does it make more sense
  to launch Ender as a sidecar subprocess (simpler) or a persistent local daemon with a
  well-known port (more flexible for multiplayer)?

- **Training cadence.** Should M3 train online (update weights between episodes) or offline
  (batch training after N episodes)? Online is more interesting but more unstable.

- **Multi-domain models.** Long-term: can a single policy generalise across Skirmish and FC
  if the state vector and action space are normalised to a common schema? Probably not useful,
  but worth noting as a research question.

- **MLP vs. LSTM for time-sensitive domains.** The MLP policy is stateless — each decision
  sees only the current state vector. For DCS, where the last 30 seconds of events carries
  strategic information, an LSTM (which maintains hidden state across timesteps) may
  eventually be worth exploring. Not a now problem, but worth knowing the limitation exists.
