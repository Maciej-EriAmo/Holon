# EriAmo: Holon — The Conscious Browser Layer

**A Temporal Perception Architecture for Large Language Models**

Maciej Mazur¹  
¹ Independent AI Researcher, Warsaw, Poland  
GitHub: github.com/Maciej-EriAmo/Holomem  

---

## Abstract

We present **Holon**, a session memory architecture that operates as a stateful perception layer between the user and any large language model. Unlike retrieval-augmented generation (RAG) or sliding window approaches, Holon maintains a compact matrix Φ ∈ ℝ^(k×d) — a *geometry of perception* — that evolves across sessions through time-based exponential decay. The system produces measurable reductions in context drift over long sessions while compressing the effective context window. Crucially, Φ is never injected into the LLM prompt; it acts exclusively as a selection mechanism, incurring zero additional token cost. We demonstrate that Holon is model-agnostic, runs on ARM hardware (Android/Termux), and preserves semantic coherence across interruptions of arbitrary duration through a biologically-inspired temporal decay model.

---

## 1. Introduction

Contemporary LLMs process each session as a stateless sequence. A model with a 32K token context window has no memory of what was discussed yesterday, no awareness that three hours have passed since the last message, and no mechanism for distinguishing which fragments of a long conversation are semantically central versus peripheral noise.

The standard engineering response — sliding window context management — is a brute-force solution. It sends the last N message pairs to the model, discarding everything beyond that horizon. This approach is computationally wasteful at the top of the window and informationally blind beyond it.

Holon takes a different approach rooted in three observations:

1. **The context problem is geometric, not volumetric.** What matters is not *how much* history is present, but *which parts* are structurally relevant to the current query.

2. **Sessions have temporal structure.** A conversation interrupted for eight hours is not equivalent to one that flowed continuously. The model should behave differently.

3. **Perception can be separated from content.** A system can maintain a compact representation of *how to interpret* a session without storing the session verbatim.

These observations motivate the Holon architecture.

---

## 2. Architecture

### 2.1 Two Memory Classes

Holon divides session state into two classes:

- **T-class (Temporal)**: Items currently in the context window — recent exchanges, active code, the current query. Subject to normal LLM attention.
- **P-class (Permanent)**: The perception matrix Φ ∈ ℝ^(k×d), stored outside the context window. Persists between sessions. Never sent to the LLM.

This separation is fundamental. Φ does not carry information *for* the model — it carries information *about how to filter* what goes to the model.

### 2.2 The Perception Matrix Φ

Φ consists of k row vectors, each of dimension d (the embedding dimension). Each row represents a learned attractor in semantic space:

- Row 0: technical/code patterns (slow decay, half-life 48h)
- Row 1: project/architecture patterns (medium decay, 24h)
- Row 2: conversational patterns (fast decay, 8h)
- Row 3: current working context (fastest decay, 4h)

The rows are kept mutually orthogonal through a repulsion term applied after each update, ensuring that each attractor specializes rather than collapsing toward a common center.

### 2.3 Turn Cycle

Each full turn proceeds in strict order:

1. **Recall** (Φ → T): Query embedding activates the nearest Φ row; top-N store items near that attractor are flagged for inclusion.
2. **Add**: New item (user message) added to store with age=0.
3. **Vacuum**: Items scored by `relevance × exp(-age/τ)`. Low-scoring items removed. Hard cap enforced.
4. **Build window**: Top-n items by weighted proximity to dynamic Φ-center selected.
5. **Update Φ**: Weakest row moves toward weighted pattern of active items (recalled items weight 2×, new items 1.5×, old 1×).
6. **age++** (in `after_turn` only — one increment per full exchange).

### 2.4 Time Decay

When the system is closed and reopened after δ hours, Φ evolves:

```
Φ[k] ← Φ[k] × exp(−ln2 × δ / half_life[k])
```

Each row decays at its own rate. After 12 hours, conversational patterns have largely faded while technical patterns remain strong. The system generates a wake message describing what happened during the absence:

> *"3.2 hours since last session. 15 turns, 4 patterns in memory. Conversational patterns slightly faded."*

This is not cosmetic. The decay directly affects which items survive vacuum and which Φ rows dominate recall — the system genuinely "remembers differently" after time has passed.

### 2.5 Context Budget

All messages assembled by Holon are subject to a token budget:

- Model limit: 4096 tokens (Mistral 7B baseline)
- Reserved for response: 512 tokens
- Reserved for system prompt: 256 tokens
- Reserved for query: 512 tokens
- **Available for Holon memory: 2816 tokens (~9856 characters at 3.5 chars/token for Polish)**

Each memory item is capped at 300 characters. Items are included in decreasing priority order until the budget is exhausted.

---

## 3. Key Properties

### 3.1 Zero Prompt Overhead

Φ is used only for selection. It never appears in the prompt sent to the LLM. This is architecturally essential: a raw embedding vector (32 floats ≈ 200 tokens) added to every prompt would consume roughly 7% of the total context budget for zero semantic benefit to the model.

### 3.2 Model Agnosticism

Holon requires only:
- An embedding function (Gemini API or hash fallback)
- An OpenAI-compatible chat endpoint

It has been tested with Gemini 2.5 Flash and Mistral 7B via Featherless. The same Python implementation runs on Android (Termux ARM) without modification.

### 3.3 Persistent Identity

The perception matrix Φ is saved to disk after every turn. On reload, it is evolved forward by the elapsed time before any new turn begins. A session interrupted for a week and resumed produces different behavior than one resumed after five minutes — not because different content is present, but because the geometry of perception has changed.

---

## 4. Experimental Results

### 4.1 Session Stability (24-turn interactive session, Gemini 2.5 Flash)

| Metric | Value |
|---|---|
| Store size (stable) | 4 items throughout |
| Average relevance (turn 1) | 0.036 |
| Average relevance (turn 24) | 0.198 |
| Phi norms | [1.0, 1.0, 1.0, 1.0] |
| Persistence | Confirmed across restarts |
| Time awareness | Active (delta reported correctly) |

Φ converges from random initialization toward session-specific geometry over approximately 20-30 turns with a real embedding model.

### 4.2 Context Drift (30-turn benchmark, hash embedder, Mistral baseline)

| Metric | Holon | Baseline (window=5) | Delta |
|---|---|---|---|
| Avg context drift | 0.5921 | 0.6165 | **−4.0%** |
| Last-10 drift | 0.5419 | 0.6156 | **−12.0%** |
| Avg store size | 3.9 | 5.0 | −22% |

The effect grows with session length. At turn 25, baseline drift reached 1.02 (response nearly orthogonal to query) while Holon maintained 0.28. This demonstrates the core property: Holon degrades gracefully over long sessions while sliding window approaches exhibit sudden failure when facts scroll out of the window.

---

## 5. The Conscious Browser Layer

The name "Holon" reflects the architectural position of the system. A holon (Koestler, 1967) is simultaneously a whole and a part — self-contained yet embedded in a larger structure. Holon the system is:

- **Self-contained**: It runs independently of the LLM, requires no model modification, no retraining, no access to weights.
- **Embedded**: It shapes every interaction with the LLM without the LLM's awareness.

The "conscious browser layer" framing positions Holon as the system that sits between human intent and model response — not processing language, but processing *context*. It answers the question: *given everything that has happened in this relationship, what should the model attend to now?*

This is structurally similar to what a skilled human assistant does when preparing a briefing: they do not hand over every email ever written; they distill the geometry of the situation.

---

## 6. Relation to HoloMem

HoloMem is the name reserved for the next architectural step: direct injection of Φ into the model's attention mechanism as an additive pre-softmax bias:

```
attention = softmax(QK^T / √d + α · (Q · Φ_center^T))
```

This requires access to model weights and is planned as a separate publication. Holon (the system described here) operates at the prompt layer and requires no such access. The two are composable: Holon can be deployed today on any model; HoloMem requires either fine-tuning or inference-level access.

---

## 7. Comparison with Related Work

| System | Context management | Time awareness | Model-agnostic | Prompt overhead |
|---|---|---|---|---|
| Sliding window | Recency truncation | None | Yes | None |
| RAG | Similarity retrieval | None | Yes | High (chunks) |
| MemGPT | Hierarchical paging | None | Partial | High |
| **Holon** | **Geometric selection via Φ** | **Exponential decay** | **Yes** | **Zero** |

The key differentiator is the combination of learned geometric selection with temporal decay and zero prompt overhead.

---

## 8. Implementation

Holon is implemented in a single Python file (`holon.py`, ~900 lines) with two dependencies: `numpy` and `google-genai`. It runs on Python 3.10+ including Android (Termux ARM). An Android (Kotlin) implementation with persistent storage is also available.

Persistence uses atomic file writes (write to `.tmp`, then rename) ensuring that sudden process termination cannot corrupt the memory file.

Source: github.com/Maciej-EriAmo/Holomem

---

## 9. Conclusion

Holon demonstrates that effective long-session memory does not require model modification, large vector databases, or significant prompt overhead. A compact k×d matrix, evolved through time-aware exponential decay and updated through interference-weighted gradient steps, provides meaningful context coherence at negligible computational cost.

The temporal dimension — the system's awareness of how long it has been dormant — is not an aesthetic feature. It is what distinguishes a memory system from a cache.

---

## References

Koestler, A. (1967). *The Ghost in the Machine*. Hutchinson.

Brown, T. et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.

Packer, C. et al. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.

Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.

Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*.

