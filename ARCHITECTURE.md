# Ender — Architecture

## Overview

Ender is a Python library and local server for oracle-distilled game AI. An LLM acts as an expert
commander during live play, logging decisions that train a small neural network. Reinforcement
learning then refines the network beyond the oracle's capabilities. Once trained, the network
runs inference for free.

---

## Components

### Domain Adapter (per-game, any language)

The only piece a game author implements. Three functions and one data structure:

```
encode_state(world) → float[]
    Compress current world state into a fixed-length feature vector.
    This is where domain expertise lives. Design it carefully.

decode_action(action_id: int) → GameCommand
    Translate a network output integer into a concrete game command.

compute_reward(event) → float
    Map game events (objective captured, unit destroyed, mission end) to reward signals.
    Sparse rewards (win/loss only) will stall training. Design intermediate signals.

action_space: Action[]
    The fixed menu of decisions available to the commander.
    Aim for a few dozen. Too granular = slow training. Too coarse = no strategy.
```

The adapter also owns a **human-readable state summary** — a separate representation of world
state formatted as prose or structured text for the oracle prompt. The float vector is for the
network; the summary is for the LLM.

---

### Ender Protocol

Games communicate with Ender over a local socket (default) or HTTP. All messages are JSON.

**State Request** — game asks Ender for a decision:
```json
{
  "state_vector": [0.8, 0.2, 1.0, ...],
  "state_summary": "North airfield held by enemy. Two friendly squads available...",
  "valid_actions": [0, 3, 7, 12]
}
```

**Action Response** — Ender returns a decision:
```json
{
  "action_id": 7,
  "confidence": 0.84,
  "source": "policy"
}
```
`source` is either `"policy"` or `"oracle"` — useful for logging and UI.

**Reward Event** — game reports outcome:
```json
{
  "action_id": 7,
  "reward": 0.4,
  "terminal": false
}
```

---

### Oracle Layer (reusable)

Wraps one or more LLMs. Receives a state request, constructs a prompt, parses structured
output. Can operate in single-teacher mode (one LLM) or multi-teacher pool mode.

**Prompt contract:**
```
System:
  You are a tactical commander. Given the battlefield state and a numbered list of
  available actions, choose exactly one action.
  Respond ONLY with JSON: { "action_id": <int>, "reasoning": "<string>" }
  Do not include any other text.

User:
  [state_summary]

  Available actions:
  [numbered action_space filtered to valid_actions]
```

The `reasoning` field is logged alongside every decision. This produces a human-readable
audit trail of oracle thinking, useful for debugging reward shaping and reviewing emergent
network behaviour later.

---

#### Multi-Teacher Distillation (recommended)

Rather than collecting labels from a single oracle, Ender supports a **teacher pool** where
multiple models label the same states independently. This is a recognised technique called
**Multi-Teacher Knowledge Distillation**.

Instead of a hard label (`action_id: 7`), each state accumulates a **soft target** —
a probability distribution over the action space derived from teacher consensus:

```
State S — teacher votes:
  Claude:       action 7 — confidence 0.91
  Qwen3-14B:    action 7 — confidence 0.84
  GPT-OSS-20B:  action 3 — confidence 0.61

Soft target vector (18 actions):
  action 7 → 0.63  (two confident teachers agree)
  action 3 → 0.27  (one dissenting opinion)
  others   → 0.10  (residual mass)
```

Training on soft targets rather than hard labels carries more information — the network
learns not just *what* to do but *how certain* experts are, and which alternatives are
plausible. This produces more robust policies than single-teacher hard-label training.

**Agreement score** (0.0–1.0) measures how strongly teachers converged on a single action.
High-agreement states are clean training examples. Low-agreement states represent genuinely
ambiguous situations — the most interesting cases, where calibrated uncertainty in the
trained policy matters most.

**Two-phase corpus pipeline:**

State generation and labelling are deliberately decoupled:

```
Phase A — Corpus Generation (fast, no LLM needed)
  Play N games using scripted/random opponents.
  Log every (state_vector, state_summary, valid_action_ids) encountered.
  Store in SQLite as unlabelled states. Run once.

Phase B — Labelling Runs (sequential, one model at a time)
  Load teacher → label entire corpus → save labels → unload → next teacher.
  Each teacher reads the same state rows and writes its opinions alongside them.
  Adding a new teacher later costs only one more labelling run.

Phase C — Consensus (fast, CPU only)
  Merge label sets into soft targets. Score by agreement.
  Produce final training dataset.
```

This is particularly suited to hardware-constrained setups where multiple large models
cannot be loaded simultaneously. Models run sequentially overnight, each labelling the
same fixed corpus. The corpus is generated once; teachers can be added incrementally.

**Corpus schema** (follows the HuggingFace offline RL dataset convention):

```sql
CREATE TABLE states (
    id               INTEGER PRIMARY KEY,
    game_id          TEXT,
    turn             INTEGER,
    state_vector     BLOB,         -- float[] serialised
    state_summary    TEXT,
    valid_action_ids TEXT,         -- JSON array
    created_at       TIMESTAMP
);

CREATE TABLE labels (
    state_id    INTEGER REFERENCES states(id),
    teacher     TEXT,              -- e.g. "claude", "qwen3_14b", "gpt_oss_20b"
    action_id   INTEGER,
    confidence  FLOAT,
    reasoning   TEXT,
    labelled_at TIMESTAMP,
    PRIMARY KEY (state_id, teacher)
);

CREATE TABLE consensus (
    state_id        INTEGER REFERENCES states(id),
    soft_targets    BLOB,          -- float[action_space_size]
    agreement_score FLOAT,         -- 0.0–1.0
    teacher_count   INTEGER,
    PRIMARY KEY (state_id)
);
```

**Cost routing:** API-based oracles (Claude) are expensive at volume. Local models
(Qwen3-14B, GPT-OSS-20B via Ollama) cost only electricity. A practical split: use local
models for bulk labelling of unambiguous states, route low-agreement states to the API
oracle where quality judgement matters most.

**Decision Logger** writes every `(state_id, teacher, action_id, confidence, reasoning)`
tuple to the labels table. In single-teacher mode this collapses to one label per state.

---

### Training Pipeline (reusable)

Three phases, each building on the previous.

**Phase 1 — Collection**
Generate a corpus of states via scripted self-play, then run labelling passes with each
teacher model sequentially (see Oracle Layer — two-phase corpus pipeline). No training yet.

State **diversity** matters more than volume. 5,000 states sampled across varied game
situations (early game, late game, winning position, losing position, contested centre)
will produce a better policy than 50,000 states from mid-game stalemates. The generation
phase should deliberately explore the state space rather than playing out representative
games to completion.

**Phase 2 — Behavioural Cloning**
Supervised training on logged (state_vector, action_id) pairs. Standard cross-entropy loss.
The network learns to imitate the oracle. Output: a policy that "plays like Claude on a budget."

**Phase 3 — RL Fine-tuning**
Swap the oracle out and run the policy network in the live environment (or a simulation).
Reward signals from `compute_reward()` shape the network beyond oracle behaviour.
Recommended algorithm: PPO (stable, well-understood, good for discrete action spaces).

**Phase 4 — Self-play (optional)**
Two instances of the policy network command opposing forces. Generates infinite training data
at zero cost. Tends to produce emergent strategies neither instance was explicitly trained for.
This is where interesting things happen.

---

### Policy Network (reusable)

A small MLP. The architecture is intentionally unremarkable — the domain adapter and reward
shaping matter far more than network depth.

```
Input:  state_vector (float[N], game-defined length)
Hidden: 2–3 layers, 128–256 units, ReLU
Output: logits over action_space (float[A])
```

Inference is a single forward pass. Runs on CPU. Suitable for hundreds of concurrent instances.

---

### Commander Blend

During the transition from oracle to policy, a confidence threshold blends the two:

```python
if policy.confidence(state) >= threshold:
    return policy.act(state)      # source: "policy"
else:
    return oracle.act(state)      # source: "oracle", log for training
```

The threshold is configurable. Starting low (0.5) and raising it as training matures lets
you ship early while the network is still learning, and phases the oracle cost out automatically.

---

## Deployment Topology

```
┌─────────────────────────────┐
│  Game Process               │
│  (Java / Lua / Python / ...) │
│                             │
│  Domain Adapter             │
│    └─► Ender Client (thin)  │──── local socket / HTTP ────┐
└─────────────────────────────┘                             │
                                                            ▼
                                               ┌────────────────────────┐
                                               │  Ender Server          │
                                               │  (Python process)      │
                                               │                        │
                                               │  Commander Blend       │
                                               │    ├─ Policy Network   │
                                               │    └─ Oracle Layer     │
                                               │                        │
                                               │  Decision Logger       │
                                               │  Training Pipeline     │
                                               └────────────────────────┘
```

For single-player scenarios, Ender runs as a local sidecar. For multiplayer servers, a single
Ender instance can serve multiple concurrent game sessions — the policy network is stateless
at inference time.

---

## Key Design Decisions

**Single shared model per domain, not per-session.** All sessions contribute to the same
training corpus and run the same policy. Multiplayer servers become a distributed training
signal rather than a cost scaling problem.

**Multi-teacher distillation over single-teacher cloning.** Soft targets derived from
teacher consensus carry more information than hard labels from any single oracle. States
where teachers disagree are the most valuable — they represent genuine strategic ambiguity
and produce better-calibrated uncertainty in the trained policy.

**Corpus schema follows the HuggingFace offline RL convention.** Storing `(state, action,
reward, terminal)` tuples in a standard format means existing tooling (Decision Transformer,
SB3 replay buffers) can consume the corpus directly without conversion.

**SQLite for the decision log.** Zero infrastructure, portable, adequate for the volumes
involved. Swap out if you need distributed training across multiple servers.

**Protocol-first integration.** Games never link against Ender directly. This keeps the
language boundary clean and means Ender can be updated, retrained, or replaced without
touching game code.

**Reward shaping is the product.** The framework is generic. The quality of `compute_reward()`
determines whether the trained commander is interesting or degenerate. Document it carefully.

**Reward outcomes, not casualties.** Penalising unit losses produces timid, passive commanders
that optimise for avoiding punishment rather than winning. Reward objective capture, mission
success, and threat suppression. Let the RL phase discover which trades are worth making.

**The corpus serves two pipelines simultaneously.** Every `(state_summary → action + reasoning)`
pair collected for MLP training is also valid training data for LoRA fine-tuning a local oracle.
This means corpus collection is never wasted — it feeds both the distillation pipeline and the
oracle improvement pipeline with the same data.

---

## Training Flywheel

The architecture supports a self-improving loop across training iterations. Each iteration
produces better teachers, which produce better labels, which produce better students.

```
Iteration 1:
  Generic local models (Qwen3-14B, GPT-OSS-20B)
    → label corpus
    → train MLP v1 + LoRA-fine-tune oracle v1

Iteration 2:
  Fine-tuned oracle v1 (domain-aware, better labels)
    → label new/expanded corpus
    → train MLP v2 + LoRA-fine-tune oracle v2

Iteration N:
  Increasingly capable oracle
    → labels increasingly subtle strategic situations
    → policy network improves beyond what earlier oracles could teach
```

The key insight is that fine-tuning a local oracle on domain-specific data costs roughly
**5,000–10,000 high-quality examples** — exactly the corpus size you accumulate in the first
overnight labelling run. You don't need a separate data collection effort; the MLP training
corpus *is* the LoRA fine-tuning corpus.

**Practical LoRA requirements on your hardware:**

- 14B model at Q4 quantisation: fits in ~10GB VRAM, fine-tunable on a 4080 with QLoRA
- Fine-tuning run on 5,000–10,000 examples: hours, not days
- Output: a small adapter file (~10–50MB) that merges back into the base model weights
- The fine-tuned model becomes the primary local oracle for the next collection round

**What improves with each iteration:**

The generic base model understands strategy abstractly. After fine-tuning on domain-specific
(state_summary → action + reasoning) pairs, it understands *your* action space, *your* reward
structure, and the specific tactical vocabulary of the game. Its decisions become more
consistent, its structured output more reliable, and its reasoning more grounded in the
domain. This reduces the disagreement rate between teachers, which tightens the consensus
distribution, which produces cleaner soft targets for MLP training.

**The ceiling:**

The flywheel eventually plateaus — once the oracle's domain knowledge is saturated, more
fine-tuning on the same distribution won't help. At that point the interesting path is
generating *harder* states: edge cases, asymmetric scenarios, situations the current policy
handles poorly. Feeding those back through the oracle produces labels for the exact situations
where improvement matters most. This is a later-stage concern, but worth designing toward
from the start by ensuring the corpus generation phase can be parameterised to target
specific regions of the state space.
