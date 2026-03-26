# Ender — Rationale

## The Problem

AI-driven game commanders are compelling but expensive. Large language models (LLMs) like Claude
produce high-quality, context-aware strategic decisions, but their cost scales linearly with usage.
A single-player scenario is affordable. A multiplayer server handling dozens of concurrent events
is not — and once you have multiple servers, a patronage model becomes the only viable path.

The deeper problem is that LLM latency is also poorly suited to real-time environments. A 500ms
round-trip to an API is acceptable for turn-based play but introduces friction in anything faster.

## The Insight

LLM decisions are expensive to generate but cheap to imitate.

If you log enough (state, decision) pairs from an LLM acting as a commander, you can train a small
neural network to approximate its behaviour. Inference on that network costs microseconds and runs
on a CPU. At that point the LLM has served its purpose: it was a teacher, not a runtime dependency.

This is a well-understood pattern in ML research — **knowledge distillation** or **imitation learning**
— but it's rarely applied to game AI in a principled, reusable way.

## Why Reinforcement Learning on Top

Behavioural cloning produces a network that plays like the oracle on average. It doesn't exceed it.
The oracle may have blind spots, habits, or limitations the network inherits uncritically.

Reinforcement learning (RL) fine-tuning gives the network a feedback signal the oracle never had:
actual win/loss outcomes from real play. Over time the network can diverge from the oracle in
productive ways — finding strategies the LLM wouldn't, because it isn't reasoning, it's optimising.

The thematic resonance with the name is intentional: a student trained by an expert, then surpassing
them through accumulated experience.

## Why This Becomes a General Framework

The core insight — oracle distillation into a policy network, refined by RL — is domain-agnostic.
The only domain-specific work is:

- How do you encode world state as a float vector?
- What is the discrete action space?
- What events constitute a reward signal?

Everything else — the oracle prompt contract, the training pipeline, the blending logic — is
identical regardless of whether the domain is a WWII air combat sim, a space fleet tactics game,
or something else entirely.

The framework is worth building as a standalone library precisely because the marginal cost of
adding a new domain is just one adapter.

## Why "Ender"

The name comes from Orson Scott Card's *Ender's Game*. The titular character is trained in
simulation by an intelligence that knows more than him, then surpasses his teachers through
accumulated experience and pattern recognition they didn't anticipate.

This maps almost literally onto the Ender pipeline:
- The oracle is the Mazer Rackham figure: wise, expensive, eventually unnecessary
- The policy network is Ender: learning fast, eventually operating independently
- Self-play is Battle School: endless simulation that produces emergent mastery

Short, memorable, meaningful to the target audience, no geopolitical baggage.

## Language Choice

The training and inference pipeline is implemented in Python. The rationale is simple: the
entire RL and ML ecosystem (PyTorch, Stable-Baselines3, Gymnasium) is Python-native. Implementing
this in Java would mean swimming upstream constantly.

Game integration uses a thin adapter protocol (JSON over a local socket or REST). Games implement
a lightweight client in whatever language they prefer. Feudal Carriers speaks Java. DCS speaks Lua.
Neither needs to know Ender is Python.
