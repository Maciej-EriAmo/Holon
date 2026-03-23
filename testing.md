# TESTING.md — HoloMem GPU Test Protocol

**For: SpeakLeash / Bielik team**
**Contact: Maciej Mazur — github.com/Maciej-EriAmo/Holon**

---

## What we want to test

HoloMem adds an additive bias to the attention mechanism:

```python
# Standard attention
scores = Q @ K.T / sqrt(d)

# HoloMem attention
phi_center = weighted_sum(phi_rows, query)   # k×d matrix → d vector
scores = Q @ K.T / sqrt(d) + alpha * (Q @ phi_center.T)
```

**Hypothesis:** injecting Φ (learned perception geometry) as attention bias
reduces perplexity by 5-12% — consistent with Harmonic Attention results.

---

## Minimum viable test (1-2 hours)

**Model:** Bielik-11B-v2.x or any Polish LLM with accessible attention weights

**Setup:**
```bash
git clone https://github.com/Maciej-EriAmo/Holon
pip install numpy torch transformers
```

**Test script** (pseudocode — adapt to your inference setup):

```python
import numpy as np
from holon import HoloMem, Embedder, Config

# 1. Build Phi from a sample session (50-100 turns of Polish text)
cfg = Config(dim=256, k=4)
emb = Embedder(dim=256)
mem = HoloMem(emb, cfg, "phi_test.json")
mem.start_session()

for text in polish_sample_texts:          # any Polish corpus sample
    mem.turn(text, "")
    mem.after_turn(text, "ok")

phi = mem.phi                              # shape: (4, 256)

# 2. Compute phi_center for a test query
query_emb = emb.encode("test query")
sims = phi @ query_emb
weights = softmax(sims)
phi_center = weights @ phi                 # shape: (256,)

# 3. In model forward pass — add bias before softmax
alpha = 0.05                              # start small, tune if needed
# attention_scores += alpha * (Q @ phi_center.T)
```

**Metrics to collect:**
- Perplexity on held-out Polish text: baseline vs HoloMem
- Memory usage delta (should be negligible — one d-vector per layer)
- Inference speed delta (should be <1%)

---

## Expected results

Based on Harmonic Attention experiments (same additive bias family):
- ~5-12% perplexity reduction
- Effect strongest on long documents / multi-turn conversations
- No effect on single-sentence inference (Φ needs session context to be meaningful)

---

## What Holon provides without GPU

Already working and tested (20/20 unit tests):

| Feature | Status |
|---|---|
| KuRz offline embedder | Working, no API |
| Φ perception matrix | Working, time decay confirmed |
| Session persistence | Working, atomic save |
| Context drift reduction | −12% last-10 turns |
| Firefox browser plugin | Working |
| Android (Termux ARM) | Working |

---

## Questions / contact

GitHub Issues: github.com/Maciej-EriAmo/Holon/issues
