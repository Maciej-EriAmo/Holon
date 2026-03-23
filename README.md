# EriAmo: Holon — The Conscious Browser Layer

**A Temporal Perception Architecture for Large Language Models**

Maciej Mazur · Independent AI Researcher, Poland
[github.com/Maciej-EriAmo/Holon](https://github.com/Maciej-EriAmo/Holon)

---

## Quick Start

```bash
pip install numpy
python holon.py
```

No GPU. No API key required. Runs on Android (Termux ARM), Raspberry Pi, NUC.

Optional — better semantics with Gemini embeddings:
```bash
export GEMINI_API_KEY=your_key
python holon.py
```

---

## What is Holon?

Holon is a **temporal perception layer** that sits between the user and any LLM.

Instead of sending the last N messages to the model (sliding window), Holon maintains a compact matrix **Φ ∈ ℝ^(k×d)** — a *geometry of perception* — that learns what matters across sessions and evolves through time-based exponential decay.

```
User
 ↓
KuRz (offline self-learning embedder)   ← "reptile brain" — fast, no API
 ↓
Φ  (perception matrix, persists across sessions)
 ↓
Context window  (geometric selection, not recency)
 ↓
LLM  (any OpenAI-compatible endpoint)
 ↓
Response
```

**Three layers:**

| Layer | Role | Cost |
|---|---|---|
| KuRz | Self-learning embedder, co-occurrence clustering | Zero — no API, no GPU |
| Φ | Geometry of perception, time decay, session identity | O(k×d) |
| LLM | Language generation | Normal inference |

---

## Key Properties

**Zero prompt overhead** — Φ is used only for selection. Never injected into the prompt.

**Time-aware** — When reopened after 8 hours, conversational patterns have faded but technical patterns remain. The system genuinely remembers differently after time passes.

**Model-agnostic** — Works with Gemini, Mistral, Bielik, or any OpenAI-compatible endpoint.

**Runs anywhere** — Tested on Android (Termux ARM), NUC 7i5, Colab T4.

---

## Repository Structure

```
holon.py          — core system (Φ + vacuum + recall + time decay)
kurz.py           — KuRz offline embedder (co-occurrence learning)
benchmark.py      — rolling benchmark: KuRz-warm vs KuRz-cold vs Hash vs Baseline
browser/          — Firefox extension (conscious browser layer)
docs/             — papers EN + PL
```

---

## What needs GPU testing

The prompt-layer system works and is benchmarked (see Section 4).

**The next step — HoloMem — requires model access:**

```
attention = softmax(QK^T / √d + α · (Q · Φ_center^T))
```

This is an additive pre-softmax bias — composable with LoRA, zero architectural changes.

**What we need:**
- Run perplexity test on Bielik-11B or similar Polish LLM
- Measure: does Φ bias reduce perplexity vs baseline?
- Expected: ~5-12% reduction (consistent with Harmonic Attention results)

See [TESTING.md](TESTING.md) for exact test protocol.

---

## Experimental Results

### Session stability (24 turns, Gemini 2.5 Flash)

| Metric | Value |
|---|---|
| Store size (stable) | 4 items throughout |
| Avg relevance turn 1 | 0.036 |
| Avg relevance turn 24 | 0.198 |
| Phi norms | [1.0, 1.0, 1.0, 1.0] |
| Persistence | Confirmed across restarts |
| Time awareness | Active |

### Context drift (30-turn benchmark)

| Metric | Holon | Baseline w=5 | Delta |
|---|---|---|---|
| Avg context drift | 0.5921 | 0.6165 | **−4.0%** |
| Last-10 drift | 0.5419 | 0.6156 | **−12.0%** |
| Avg store size | 3.9 | 5.0 | −22% |

---

## Relation to HoloMem

HoloMem is the attention-layer version — Φ injected directly as bias before softmax. Requires model weights. Planned as separate publication.

Holon (this repo) operates at the prompt layer. Deployable today on any model.

Both are composable:

```
QK^T + λ·cos(φᵢ - φⱼ)   ← Harmonic Attention (additive bias)
     + α·(Q·Φ^T)          ← HoloMem (additive bias)
```

---

## License

Apache 2.0

---

## References

Koestler, A. (1967). *The Ghost in the Machine*.
Damasio, A. (1994). *Descartes' Error*.
Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*.
Packer, C. et al. (2023). MemGPT. *arXiv:2310.08560*.
