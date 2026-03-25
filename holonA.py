"""
Holon v5.6 — Full Predictive Coding + Asystent z przypomnieniami + Trwałe fakty
===========================================================================
Autor koncepcji: Maciej Mazur
===========================================================================
POPRAWKI v5.4 (relative to Grok's version):
[R1] recall_at — przywrócono +hours_ago (cofnięta regresja FIX-5)
[R2] evolve_phi — przywrócono level: int=0 + obsługa 2D hl_list (cofnięta regresja G-3)
     Config: phi_half_life_hours jako 2D lista (phi_levels × k)
[R3] _vacuum — pola Config zamiast getattr fallback (FIX-4 przywrócony)
[R4] FACT_PATTERNS — wyciągnięte jako stała klasy HoloMem, używana w turn() i after_turn()
[R5] _parse_reminder — typ zwracany kompatybilny z Python 3.9+ (Tuple zamiast tuple[])
[NOTE] phase_shift (fftfreq) — implementacja Groka LEPSZA matematycznie, zostawiona.
       shift=0.9 w ruminate sensowne przy fftfreq (≈90% cyklu na najwyższej freq).
       phase_shifts %= 1.0 poprawne przy fftfreq (jednostki: cykle, zakres [0,1)).
[NOTE] W_gen global norm clamp — akceptowalne z W_gen*=0.999 per k-loop.
[NOTE] merge threshold 0.95 — dobra zmiana, zostawiona.
[NOTE] serwer API (OpenAIClient + Groq/DeepSeek) — bez zmian per życzenie.
"""

import os
import json
import math
import time
import uuid
import hashlib
import re
import datetime
import threading
import requests
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass, field

try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    print("[Info] Brak biblioteki python-dateutil – użyję prostszego parsowania czasu.")

try:
    from kurz import KuRz as _KuRz
except ImportError:
    class _KuRz:
        def __init__(self, dim=256, dict_path=None):
            self.dim = dim
            self.dict_path = dict_path
            self.vocab_size = 10000
            self.calls = 0
        def encode(self, text):
            self.calls += 1
            return np.random.randn(self.dim).astype(np.float32)
        def save_dict(self):
            pass


# ============================================================
# KONFIGURACJA
# ============================================================

@dataclass
class Config:
    k:             int   = 4
    n:             int   = 7
    threshold:     float = 0.20
    lr:            float = 0.01
    alpha:         float = 0.05
    top_n_recall:  int   = 2
    dim:           int   = 256
    time_dim:      int   = 8

    @property
    def total_dim(self) -> int:
        return self.dim + self.time_dim

    # [R2] 2D decay (phi_levels × k) — każdy poziom Φ ma własne tempo zaniku.
    # Poziom 0 (krótki): szybki; poziom 2 (rdzeń): najwolniejszy.
    # evolve_phi wybiera wiersz odpowiadający level=.
    phi_half_life_hours: list = field(default_factory=lambda: [
        [ 8.0,  6.0,  4.0,  2.0],   # poziom 0 — epizody / robocza
        [24.0, 16.0, 12.0,  8.0],   # poziom 1 — projekty / tematy
        [72.0, 48.0, 36.0, 24.0],   # poziom 2 — rdzeń / tożsamość
    ])
    store_decay_hours:   float = 72.0
    phi_min_norm:        float = 0.1
    phi_ortho_beta:      float = 0.05
    vacuum_age_tau:      float = 50.0
    recall_age_penalty:  float = 0.02
    aii_adapt_range:     float = 0.15
    vacuum_warmup_turns: int   = 8
    phi_stability_decay: float = 0.95
    phi_stability_max:   float = 5.0
    coherence_threshold: float = 0.4

    phi_levels:           int   = 3
    # phase_shifts w cyklach [0, 1) — fftfreq-based (1 cykl = pełen obrót)
    phase_shifts:         list  = field(default_factory=lambda: [0.0, 0.33, 0.66])
    rumination_interval:  int   = 12
    rumination_threshold: float = 0.35
    # rumination_shifts w cyklach [0, 1)
    rumination_shifts:    list  = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75])

    surprise_adapt_rate: float = 0.02
    surprise_trigger:    float = 0.4
    lr_min:              float = 0.001
    lr_max:              float = 0.05
    precision_mode:      str   = 'error'

    # [R3] Pola używane w _vacuum — zamiast getattr fallback
    soft_vacuum_interval:  int   = 4
    soft_decay_factor:     float = 0.92
    hard_prune_interval:   int   = 20
    hard_prune_store_max:  int   = 120
    phase_shifts_learnable: bool = False
    insight_prompt_template: str = (
        "Wygeneruj insight (max 2 zdania) dla niespójności {max_inc:.3f}:"
    )


# ============================================================
# ITEM
# ============================================================

@dataclass
class Item:
    id:            str
    content:       str
    embedding:     list
    age:           int   = 0
    recalled:      bool  = False
    relevance:     float = 1.0
    created_at:    float = field(default_factory=time.time)
    is_insight:    bool  = False
    insight_level: int   = -1
    cluster_size:  int   = 1
    is_reminder:   bool  = False
    is_fact:       bool  = False
    is_work:       bool  = False   # praca nad projektem — chronione jak fakt
    _norm:         float = field(default=-1.0, repr=False)

    def emb_np(self) -> np.ndarray:
        return np.array(self.embedding, dtype=np.float32)

    def emb_content(self, content_dim: int = 256) -> np.ndarray:
        return np.array(self.embedding[:content_dim], dtype=np.float32)

    def emb_time(self, content_dim: int = 256) -> np.ndarray:
        return np.array(self.embedding[content_dim:], dtype=np.float32)

    def norm(self) -> float:
        if self._norm < 0:
            self._norm = float(np.linalg.norm(self.embedding))
        return self._norm


# ============================================================
# HOLOGRAFIA + PHASE SHIFT
# ============================================================

class HolographicInterference:
    _unitary_cache: dict = {}
    _ANCHOR_SEED: str = os.environ.get("HOLON_ANCHOR_SEED", "holon-eriamo-4242")

    @staticmethod
    def _to_unitary(v: np.ndarray) -> np.ndarray:
        # [B3] round(4) przed tobytes — float32 epsilony nie powinny dać osobnych kluczy
        key = np.round(v, 4).tobytes()
        if key in HolographicInterference._unitary_cache:
            return HolographicInterference._unitary_cache[key]
        v_fft = np.fft.fft(v)
        result = v_fft / (np.abs(v_fft) + 1e-8)
        if len(HolographicInterference._unitary_cache) >= 512:
            HolographicInterference._unitary_cache.clear()
        HolographicInterference._unitary_cache[key] = result
        return result

    @staticmethod
    def _salt_key(key: np.ndarray, item_id: str) -> np.ndarray:
        combined = (item_id + HolographicInterference._ANCHOR_SEED).encode()
        h = int(hashlib.sha256(combined).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(h)
        salt = rng.standard_normal(len(key)).astype(np.float32) * 0.1
        salted = key + salt
        return salted / (np.linalg.norm(salted) + 1e-8)

    @staticmethod
    def bind(v1: np.ndarray, v2: np.ndarray, item_id: str = "") -> list:
        assert len(v1) == len(v2), f"bind: dim mismatch {len(v1)} != {len(v2)}"
        key = HolographicInterference._salt_key(v2, item_id) if item_id else v2
        v2_u = HolographicInterference._to_unitary(key)
        bound = np.fft.ifft(np.fft.fft(v1) * v2_u).real.astype(np.float32)
        return bound.tolist()

    @staticmethod
    def unbind(bound_data: list, key: np.ndarray, item_id: str = "") -> np.ndarray:
        key = HolographicInterference._salt_key(key, item_id) if item_id else key
        bound = np.array(bound_data, dtype=np.float32)
        key_u = HolographicInterference._to_unitary(key)
        unbound = np.fft.ifft(np.fft.fft(bound) * np.conj(key_u)).real.astype(np.float32)
        return unbound / (np.linalg.norm(unbound) + 1e-8)

    @staticmethod
    def phase_shift(v: np.ndarray, shift: float) -> np.ndarray:
        """Rotacja fazowa FFT przez twierdzenie o przesunięciu.
        shift w cyklach [0, 1) — fftfreq-based.
        shift=0.5 → przesunięcie o pół cyklu na każdej częstotliwości.
        Zmiana: fftfreq zamiast arange/dim — poprawna matematycznie (Grok).
        """
        if abs(shift) < 1e-6:
            return np.asarray(v, dtype=np.float32).copy()
        v_c   = np.asarray(v, dtype=np.complex128)
        fft_v = np.fft.fft(v_c)
        dim   = len(v)
        freqs  = np.fft.fftfreq(dim)            # w cyklach/próbkę
        angles = 2.0 * np.pi * freqs * shift    # shift w cyklach
        rotated = np.fft.ifft(fft_v * np.exp(1j * angles)).real.astype(np.float32)
        n = np.linalg.norm(rotated)
        return rotated / (n + 1e-8)


# ============================================================
# CZAS JAKO WYMIAR PRZESTRZENI
# ============================================================

_HOLON_EPOCH: float = float(os.environ.get("HOLON_EPOCH", str(time.time())))


def time_embed(timestamp: float, time_dim: int = 8) -> np.ndarray:
    if time_dim <= 0:
        return np.zeros(0, dtype=np.float32)
    delta_days = (timestamp - _HOLON_EPOCH) / 86400.0
    vec = np.zeros(time_dim, dtype=np.float32)
    n_sincos = (time_dim - 1) // 2
    scales = [1.0 / 24.0, 1.0, 7.0, 30.0, 365.0]
    scales = scales[:n_sincos]
    for i, scale in enumerate(scales):
        angle = 2.0 * math.pi * delta_days / (scale + 1e-8)
        vec[i * 2]     = math.sin(angle)
        vec[i * 2 + 1] = math.cos(angle)
    vec[-1] = float(np.clip(delta_days / 365.0, -10.0, 10.0))
    return vec


# ============================================================
# EMBEDDER
# ============================================================

class Embedder:
    def __init__(self, dim: int = 256, dict_path: Optional[str] = None,
                 cache_size: int = 256, time_dim: int = 8):
        self.dim       = dim
        self.time_dim  = time_dim
        self._kurz     = _KuRz(dim=dim, dict_path=dict_path)
        self._cache: dict = {}
        self._cache_size  = cache_size
        self._cache_hits  = 0

    def encode(self, text: str, timestamp: float = None) -> np.ndarray:
        key = (text or "")[:200]
        if timestamp is None:
            if key in self._cache:
                self._cache_hits += 1
                return self._cache[key]
            vec = self._kurz.encode(text or "")
            self._cache[key] = vec
            if len(self._cache) > self._cache_size:
                del self._cache[next(iter(self._cache))]
            return vec
        content = self._kurz.encode(text or "")
        t_vec   = time_embed(timestamp, self.time_dim)
        full = np.concatenate([content * 0.7, t_vec * 0.3])
        n = np.linalg.norm(full)
        return full / (n + 1e-8)

    def encode_timed(self, text: str) -> np.ndarray:
        return self.encode(text, timestamp=time.time())

    def save(self) -> None:
        if self._kurz.dict_path:
            self._kurz.save_dict()

    @property
    def vocab_size(self) -> int: return self._kurz.vocab_size

    @property
    def calls(self) -> int: return self._kurz.calls


# ============================================================
# AII STATE
# ============================================================

class AIIState:
    WEIGHTS = {"radosc": 1.3, "zaskoczenie": 1.3, "strach": 1.2,
               "zlosc": 1.2,  "smutek": 0.8,      "neutral": 1.0}
    VACUUM_SIGNALS = {"radosc": +1.0, "zaskoczenie": +0.5, "strach": -1.0,
                      "zlosc":  -1.0, "smutek": -0.5,      "neutral": 0.0}
    KEYWORDS = {
        "radosc":      ["super", "swietnie", "doskonale", "rewelacja", "great"],
        "zaskoczenie": ["wow", "niesamowite", "naprawde", "really"],
        "strach":      ["blad", "error", "crash", "problem", "awaria", "fail", "bug"],
        "zlosc":       ["nie dziala", "znowu", "broken", "wrong"],
        "smutek":      ["niestety", "szkoda", "nie pomaga"],
        "focus":       ["implementacja", "debug", "refaktor", "kod",
                        "architektura", "softmax", "eriamo", "holon"],
    }
    T = 0.3

    def __init__(self, embedder=None):
        self.embedder      = embedder
        self.emotion       = "neutral"
        self.vacuum_signal = 0.0
        self.focus_active  = False
        self.ref_emotions  = {}
        if embedder is not None:
            self._build_refs(embedder)

    def _build_refs(self, embedder):
        ref_texts = {
            "radosc":      "sukces świetnie doskonale rewelacja",
            "zaskoczenie": "wow niesamowite zaskoczenie naprawdę",
            "strach":      "błąd error problem awaria krytyczne",
            "zlosc":       "nie działa zepsute błąd znowu",
            "smutek":      "niestety szkoda smutno żal",
            "focus":       "implementacja kod architektura debug refactor",
        }
        for emo, text in ref_texts.items():
            v = embedder.encode(text)
            self.ref_emotions[emo] = v / (np.linalg.norm(v) + 1e-8)

    def update(self, text: str, text_emb: np.ndarray = None):
        t = text.lower()
        if self.ref_emotions and text_emb is not None:
            dim = min(len(text_emb), len(next(iter(self.ref_emotions.values()))))
            t_c = text_emb[:dim] / (np.linalg.norm(text_emb[:dim]) + 1e-8)
            sims = {emo: float(np.dot(t_c, ref[:dim])) / self.T
                    for emo, ref in self.ref_emotions.items()}
            self.focus_active = sims.get("focus", 0) > 0.45
            best_e, best_s = "neutral", 0.4
            for emo, sim in sims.items():
                if emo == "focus": continue
                if sim > best_s:
                    best_s, best_e = sim, emo
            self.emotion = best_e
            sig = self.VACUUM_SIGNALS.get(best_e, 0.0)
        else:
            self.focus_active = any(kw in t for kw in self.KEYWORDS["focus"])
            best_e, best_hits, sig = "neutral", 0, 0.0
            for emo, kws in self.KEYWORDS.items():
                if emo == "focus": continue
                hits = sum(1 for kw in kws if kw in t)
                if hits > best_hits:
                    best_hits, best_e = hits, emo
                    sig = self.VACUUM_SIGNALS.get(emo, 0.0)
            self.emotion = best_e
        self.vacuum_signal = 0.7 * self.vacuum_signal + 0.3 * sig

    def get_emotion_weight(self) -> float:
        w = self.WEIGHTS.get(self.emotion, 1.0)
        return w * 1.5 if self.focus_active else w

    def get_threshold_multiplier(self, adapt_range: float) -> float:
        return 1.0 + adapt_range * self.vacuum_signal

    def to_dict(self) -> dict:
        return {"emotion": self.emotion,
                "vacuum_signal": round(self.vacuum_signal, 3),
                "focus": self.focus_active}

    def from_dict(self, data: dict) -> None:
        if not data: return
        self.emotion       = data.get("emotion", "neutral")
        self.vacuum_signal = float(data.get("vacuum_signal", 0.0))
        self.focus_active  = data.get("focus", False)


# ============================================================
# TIME DECAY
# ============================================================

class TimeDecay:
    @staticmethod
    def decay_factor(delta_hours: float, half_life_hours: float) -> float:
        return math.exp(-0.693 * delta_hours / (half_life_hours + 1e-8))

    @staticmethod
    def evolve_phi(phi: np.ndarray, delta_hours: float,
                   hl_list: list, min_norm: float,
                   level: int = 0) -> np.ndarray:
        """[R2] hl_list może być 1D (stary format) lub 2D (phi_levels × k).
        Przy 2D wybieramy wiersz odpowiadający poziomowi Φ.
        """
        if abs(delta_hours) < 0.1:
            return phi
        evolved = phi.copy()
        # Wybierz właściwy wiersz hl_list dla tego poziomu
        if hl_list and isinstance(hl_list[0], list):
            row = hl_list[level] if level < len(hl_list) else hl_list[-1]
        else:
            row = hl_list  # fallback: stary format 1D
        for k in range(len(phi)):
            hl = row[k] if k < len(row) else 24.0
            df = TimeDecay.decay_factor(abs(delta_hours), hl)
            evolved[k] = phi[k] * df
            n = np.linalg.norm(evolved[k])
            if n < min_norm:
                evolved[k] = evolved[k] / (n + 1e-8) * min_norm
        return evolved

    @staticmethod
    def wake_message(delta_hours: float, turns: int, store_size: int,
                     coherence: float) -> str:
        if delta_hours < 0.1:
            return ""
        h = int(delta_hours)
        if delta_hours < 1:
            period = f"{int(delta_hours * 60)} minut"
        elif delta_hours < 24:
            period = f"{h} {'godzinę' if h == 1 else 'godziny' if h < 5 else 'godzin'}"
        elif delta_hours < 168:
            d = int(delta_hours / 24)
            period = f"{d} {'dzień' if d == 1 else 'dni'}"
        else:
            w = int(delta_hours / 168)
            period = f"{w} {'tydzień' if w == 1 else 'tygodnie' if w < 5 else 'tygodni'}"
        note = "Wzorce konwersacyjne lekko przybladły." if delta_hours < 8 else \
               "Część kontekstu wyparowała." if delta_hours < 48 else \
               "Zostały głównie wzorce długoterminowe."
        return f"[Minęło {period}. Było {turns} tur, {store_size} wzorców w pamięci. {note}]"


# ============================================================
# PERSISTENT MEMORY
# ============================================================

class PersistentMemory:
    def __init__(self, path: str = "holon_memory.json", dim: int = 264):
        self.path = Path(path)
        seed_str  = os.environ.get("HOLON_ANCHOR_SEED", "4242")
        seed_int  = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
        rng       = np.random.RandomState(seed_int)
        anchor    = rng.randn(dim).astype(np.float32)
        self.eriamo_anchor = anchor / (np.linalg.norm(anchor) + 1e-8)
        self.dim  = dim

    @staticmethod
    def _init_phi(cfg: Config) -> np.ndarray:
        total = cfg.total_dim
        phi = np.random.randn(cfg.phi_levels, cfg.k, total).astype(np.float32) * 0.01
        norms = np.linalg.norm(phi, axis=2, keepdims=True)
        return phi / (norms + 1e-8)

    @staticmethod
    def _phi_center_static(phi: np.ndarray, level: int = 2) -> np.ndarray:
        layer = phi[level] if phi.ndim == 3 else phi
        norms = np.linalg.norm(layer, axis=1)
        exp_n = np.exp(norms - norms.max())
        weights = exp_n / (exp_n.sum() + 1e-8)
        center = sum(weights[k] * layer[k] for k in range(len(layer)))
        n = np.linalg.norm(center)
        return center / (n + 1e-8)

    def _safe_bind(self, emb: np.ndarray, state: np.ndarray) -> list:
        m = min(len(emb), len(state))
        return HolographicInterference.bind(emb[:m], state[:m])

    def save(self, phi: np.ndarray, store: list, turns: int,
             aii: dict = None, stability=None):
        state_now   = PersistentMemory._phi_center_static(phi, level=2)
        anchor_trim = self.eriamo_anchor[:len(state_now)]
        h_coherence = HolographicInterference.bind(state_now, anchor_trim)

        data = {
            "timestamp":     time.time(),
            "turns":         turns,
            "phi":           phi.tolist(),
            "phi_stability": stability if stability is not None else [],
            "h_coherence":   h_coherence,
            "aii":           aii or {},
            "store": [
                {
                    "id":          i.id,
                    "content":     i.content,
                    "embedding":   self._safe_bind(i.emb_np(), state_now),
                    "age":         i.age,
                    "relevance":   i.relevance,
                    "created_at":  i.created_at,
                    "is_reminder": i.is_reminder,
                    "is_fact":     i.is_fact,
                    "is_work":     i.is_work,
                }
                for i in store if i.age >= 1
            ],
        }
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            tmp.replace(self.path)
        except Exception:
            try: self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            finally:
                if tmp.exists(): tmp.unlink()

    def load(self, cfg: Config) -> dict:
        tmp = self.path.with_suffix(".tmp")
        if tmp.exists(): tmp.unlink()
        if not self.path.exists():
            return {"phi": self._init_phi(cfg), "store": [], "turns": 0,
                    "delta_hours": 0.0, "aii": {}, "phi_stability": None,
                    "loaded": False, "coherence": 1.0, "wake": ""}

        try:
            data        = json.loads(self.path.read_text())
            saved_at    = data["timestamp"]
            delta_hours = (time.time() - saved_at) / 3600.0
            turns       = data["turns"]
            total_dim   = cfg.total_dim

            phi_raw = np.array(data["phi"], dtype=np.float32)
            if phi_raw.ndim == 2:
                phi_raw = np.stack([phi_raw.copy() * (1.0 - 0.05 * l)
                                    for l in range(cfg.phi_levels)])
            if phi_raw.shape[2] < total_dim:
                pad = np.zeros((*phi_raw.shape[:2], total_dim - phi_raw.shape[2]),
                               dtype=np.float32)
                phi_raw = np.concatenate([phi_raw, pad], axis=2)
                norms = np.linalg.norm(phi_raw, axis=2, keepdims=True)
                phi_raw = phi_raw / (norms + 1e-8)

            state_at_save = PersistentMemory._phi_center_static(phi_raw, level=2)
            h_coherence = data.get("h_coherence")
            if h_coherence is None:
                print("[Memory] Info: Brak h_coherence. Ładowanie bez weryfikacji.")
                coherence = 1.0
                recovered_state = state_at_save
            else:
                h_arr    = np.array(h_coherence, dtype=np.float32)
                use_dim  = min(len(h_arr), len(self.eriamo_anchor))
                recovered_state = HolographicInterference.unbind(
                    h_arr[:use_dim].tolist(), self.eriamo_anchor[:use_dim])
                s_dim = len(state_at_save)
                if len(recovered_state) < s_dim:
                    pad = np.zeros(s_dim - len(recovered_state), dtype=np.float32)
                    recovered_state = np.concatenate([recovered_state, pad])
                    recovered_state /= (np.linalg.norm(recovered_state) + 1e-8)
                coherence = float(np.dot(recovered_state[:s_dim], state_at_save))

            # [R2] evolve_phi z level=lv — 2D hl_list per poziom
            phi_today = np.zeros_like(phi_raw)
            for lv in range(cfg.phi_levels):
                phi_today[lv] = TimeDecay.evolve_phi(
                    phi_raw[lv], delta_hours,
                    cfg.phi_half_life_hours, cfg.phi_min_norm, level=lv)

            store = []
            if coherence >= cfg.coherence_threshold:
                max_age = cfg.store_decay_hours * 4
                for obj in data.get("store", []):
                    age_now = obj["age"] + int(delta_hours * 4)
                    if age_now > max_age: continue
                    emb_arr  = np.array(obj["embedding"], dtype=np.float32)
                    use_dim  = min(len(emb_arr), len(recovered_state))
                    rec_emb  = HolographicInterference.unbind(
                        emb_arr[:use_dim].tolist(), recovered_state[:use_dim])
                    raw_emb = rec_emb.tolist()
                    if len(raw_emb) < total_dim:
                        created = obj.get("created_at", time.time())
                        t_vec   = time_embed(created, total_dim - len(raw_emb)).tolist()
                        raw_emb = raw_emb + t_vec
                        v = np.array(raw_emb, dtype=np.float32)
                        raw_emb = (v / (np.linalg.norm(v) + 1e-8)).tolist()
                    store.append(Item(
                        id=obj["id"], content=obj["content"], embedding=raw_emb,
                        age=age_now, recalled=False, relevance=obj["relevance"],
                        created_at=obj.get("created_at", time.time()),
                        is_reminder=obj.get("is_reminder", False),
                        is_fact=obj.get("is_fact", False),
                        is_work=obj.get("is_work", False)))
            else:
                print(f"[Memory] Utrata koherencji ({coherence:.2f}). Depolaryzacja.")

            return {
                "phi": phi_today, "store": store, "turns": turns,
                "delta_hours": delta_hours, "aii": data.get("aii", {}),
                "phi_stability": data.get("phi_stability"),
                "wake": TimeDecay.wake_message(delta_hours, turns, len(store), coherence),
                "loaded": True, "coherence": coherence,
            }
        except Exception as e:
            print(f"[Memory] Błąd wczytania: {e}")
            return {"phi": self._init_phi(cfg), "store": [], "turns": 0,
                    "delta_hours": 0.0, "aii": {}, "phi_stability": None,
                    "loaded": False, "coherence": 0.0, "wake": ""}

    def delete(self):
        if self.path.exists(): self.path.unlink()

    def exists(self) -> bool:
        return self.path.exists()


# ============================================================
# HOLOMEM
# ============================================================

class HoloMem:
    # [R4] Deklaracje osobiste — traktuję jako is_fact
    FACT_PATTERNS: Tuple[str, ...] = (
        "mój ulubiony", "jestem", "mam na imię", "nazywam się",
        "lubię", "pracuję nad",
    )

    # Słowa kluczowe aktywnej pracy nad projektami — traktuję jako is_work.
    # Rozszerzone o konkretne projekty Macieja + generyczne terminy programistyczne.
    # Focus wykrywany też przez AIIState.focus_active (neural cosine similarity).
    FOCUS_PATTERNS: Tuple[str, ...] = (
        # Projekty
        "holon", "holomem", "eriamo", "kurz", "harmonic attention",
        "adml", "archmind", "fehm", "qrm", "bielik", "speakleash",
        # Generyczne terminy pracy nad kodem
        "implementuję", "implementacja", "debuguję", "refaktoruję",
        "klasa ", "metoda ", "funkcja ", "def ", "class ",
        "algorytm", "architektura", "moduł", "integracja",
        "trenuję", "fine-tuning", "embedding", "transformer",
        "naprawiam", "poprawka", "błąd w", "fix:",
    )

    def __init__(self, embedder: Embedder, cfg: Config = None,
                 memory_path: str = "holon_memory.json"):
        self.embedder = embedder
        self.cfg      = cfg or Config(dim=embedder.dim)
        self.memory   = PersistentMemory(memory_path, dim=self.cfg.total_dim)

        self.phi: np.ndarray = None
        self.store: list     = []
        self.turns: int      = 0
        self.phi_stability   = np.zeros((self.cfg.phi_levels, self.cfg.k),
                                        dtype=np.float32)
        self.aii             = AIIState(embedder)
        self._session_start_turn = 0
        self._delta_hours    = 0.0
        self.insight_llm_callback = None
        self.last_error: Optional[np.ndarray] = None
        self.prev_phi_center: Optional[np.ndarray] = None
        self._last_surprise: float = 0.0
        self.W_time = np.random.randn(self.cfg.total_dim, self.cfg.total_dim) * 0.01
        self.W_gen  = np.random.randn(self.cfg.total_dim, self.cfg.total_dim) * 0.01
        self.temporal_error: Optional[np.ndarray] = None

    def start_session(self) -> dict:
        res              = self.memory.load(self.cfg)
        self.phi         = res["phi"]
        self.store       = res["store"]
        self.turns       = res["turns"]
        self._delta_hours= res["delta_hours"]
        self.aii.from_dict(res.get("aii", {}))
        saved_stab = res.get("phi_stability")
        if saved_stab is not None:  # [B2] if saved_stab: crashuje na numpy array
            try:
                arr = np.array(saved_stab, dtype=np.float32)
                if arr.shape == (self.cfg.phi_levels, self.cfg.k):
                    self.phi_stability = arr
                elif arr.ndim == 1 and len(arr) == self.cfg.k:
                    self.phi_stability = np.stack([arr * (0.5 ** lv)
                                                   for lv in range(self.cfg.phi_levels)])
            except Exception:
                pass
        self._session_start_turn = self.turns
        return res

    # ── Helpers ───────────────────────────────────────────────

    def _align(self, a: np.ndarray, b: np.ndarray):
        la, lb = len(a), len(b)
        if la == lb: return a, b
        m = min(la, lb)
        return a[:m], b[:m]

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8: return 0.0
        if abs(na - 1.0) < 0.01 and abs(nb - 1.0) < 0.01:
            return float(np.dot(a, b))
        return float(np.dot(a, b) / (na * nb))

    def _csim(self, a: np.ndarray, b: np.ndarray) -> float:
        a_, b_ = self._align(a, b)
        return self._cosine_sim(a_, b_)

    def _phi_center(self, query_emb: np.ndarray = None, level: int = 2) -> np.ndarray:
        layer = self.phi[level]
        if query_emb is not None:
            q_dim = len(query_emb)
            sims = np.array([
                self._cosine_sim(query_emb, layer[k][:q_dim])
                for k in range(self.cfg.k)
            ], dtype=np.float32)
            exp_s = np.exp(sims - sims.max())
            weights = exp_s / (exp_s.sum() + 1e-8)
        else:
            norms = np.linalg.norm(layer, axis=1)
            exp_n = np.exp(norms - norms.max())
            weights = exp_n / (exp_n.sum() + 1e-8)
        center = sum(weights[k] * layer[k] for k in range(self.cfg.k))
        n = np.linalg.norm(center)
        return center / (n + 1e-8)

    # ── Recall ────────────────────────────────────────────────

    def _recall(self, query_emb: np.ndarray):
        if not self.store: return
        scores = {}
        for k in range(self.cfg.k):
            # Mixed-level attractor — core dominuje, short/mid wzmacniają kontekst
            # _cosine_sim normalizuje wewnętrznie, brak explicit norm jest OK
            attractor = (0.6 * self.phi[2][k] +
                         0.3 * self.phi[1][k] +
                         0.1 * self.phi[0][k])
            for item in self.store:
                emb = item.emb_np()
                s_att = self._csim(emb, attractor)
                s_qry = self._csim(emb, query_emb)
                score = (0.6 * s_att + 0.4 * s_qry) / \
                        (1.0 + item.age * self.cfg.recall_age_penalty)
                if item.is_fact:
                    # [B7] Wiek-zależna premia — nowe fakty ważniejsze, stare nie dominują zawsze
                    score *= (1.0 + 0.2 / (1.0 + item.age * 0.1))
                if item.is_work:
                    # Aktywna praca nad projektem: wyższy boost, wolniej maleje z wiekiem
                    score *= (1.0 + 0.4 / (1.0 + item.age * 0.05))
                if id(item) not in scores or score > scores[id(item)][0]:
                    scores[id(item)] = (score, item, k)
        ranked = sorted(scores.values(), key=lambda x: -x[0])
        for _, item, k in ranked[:self.cfg.top_n_recall]:
            item.recalled = True
            self.phi_stability[2][k] += 1.0

    # ── Vacuum ────────────────────────────────────────────────

    def _vacuum(self, query_emb: np.ndarray):
        center = self._phi_center(query_emb, level=2)
        tau = self.cfg.vacuum_age_tau
        threshold = self.cfg.threshold * self.aii.get_threshold_multiplier(
            self.cfg.aii_adapt_range)
        session_age = self.turns - self._session_start_turn
        if session_age < self.cfg.vacuum_warmup_turns:
            threshold *= (0.5 + 0.5 * session_age / self.cfg.vacuum_warmup_turns)

        cdim = self.cfg.dim
        center_c = center[:cdim] / (np.linalg.norm(center[:cdim]) + 1e-8)

        # [R3] Używamy pól Config zamiast getattr fallback
        svi = self.cfg.soft_vacuum_interval
        sdf = self.cfg.soft_decay_factor
        if self.turns > 0 and self.turns % svi == 0:
            for item in self.store:
                if not item.is_insight:
                    item.relevance *= sdf

        for item in self.store:
            sem = self._cosine_sim(item.emb_content(cdim), center_c)
            item.relevance = 0.6 * sem + 0.4 * item.relevance
            item.relevance = max(0.05, item.relevance)

        def _score(item):
            sim = self._cosine_sim(item.emb_content(cdim), center_c)
            uncertainty = 1.0 - abs(sim)
            energy  = -sim
            entropy = 0.1 * uncertainty + 0.05 * (item.age / (tau + 1e-8))
            fe = energy + entropy
            return -(fe - 0.2 * item.relevance)

        # Fakty i praca nad projektami chronione przed usunięciem
        # [B6] Dodano i.relevance > 0.3 — stabilizuje pamięć przy niestabilnym threshold (AII)
        self.store = [i for i in self.store
                      if (i.age <= 1 and i.relevance > 0.2) or i.recalled
                      or i.is_fact or i.is_work or i.relevance > 0.3 or _score(i) >= threshold]

        # [R3] Używamy pól Config zamiast getattr fallback
        hpi = self.cfg.hard_prune_interval
        hpm = self.cfg.hard_prune_store_max
        MAX_STORE = min(self.cfg.n * 6, hpm)
        if (self.turns > 0 and self.turns % hpi == 0) or len(self.store) > MAX_STORE:
            self.store.sort(key=_score, reverse=True)
            self.store = self.store[:MAX_STORE]

    # ── Update Φ ──────────────────────────────────────────────

    def _update_phi(self, window: list):
        if not window: return
        window_ids = {id(i) for i in window}
        active = [i for i in self.store
                  if id(i) in window_ids or i.age <= 1 or i.recalled]
        if not active: return

        emotion_w = self.aii.get_emotion_weight()
        tdim = self.cfg.total_dim

        pattern = np.zeros(tdim, dtype=np.float32)
        for item in active:
            phase  = math.exp(-item.age / self.cfg.vacuum_age_tau)
            weight = 2.0 if item.recalled else (1.5 if item.age <= 1 else 1.0)
            sign   = 1.0 if (item.recalled or item.age <= 1) else -0.3
            emb = item.emb_np()
            if len(emb) < tdim:
                emb = np.concatenate([emb, np.zeros(tdim - len(emb), dtype=np.float32)])
            pattern += sign * phase * weight * emotion_w * emb

        n = np.linalg.norm(pattern)
        if n < 1e-8: return
        pattern /= n

        recalled_count = sum(1 for i in window if i.recalled)
        importance     = emotion_w * (1.0 + 0.3 * recalled_count)
        if importance < 1.4:
            level = 0
        elif importance < 3.0:
            level = 1
        else:
            level = 2

        shift           = self.cfg.phase_shifts[level]
        shifted_pattern = HolographicInterference.phase_shift(pattern, shift)

        layer = self.phi[level]
        sims  = np.array([float(np.dot(shifted_pattern, layer[k]) /
                               (np.linalg.norm(layer[k]) + 1e-8))
                          for k in range(self.cfg.k)], dtype=np.float32)
        exp_s   = np.exp(sims - sims.max())
        weights = exp_s / (exp_s.sum() + 1e-8)
        weights[0] *= 0.1
        weights /= (weights.sum() + 1e-8)

        self.W_gen *= 0.999

        for k in range(self.cfg.k):
            layer_old = layer[k].copy()

            mu_k = np.tanh(self.W_gen @ layer[k])
            mu_k /= (np.linalg.norm(mu_k) + 1e-8)
            eps_local = shifted_pattern - mu_k

            if self.last_error is not None:
                le = self.last_error[:len(eps_local)]
                te = (self.temporal_error[:len(eps_local)]
                      if self.temporal_error is not None
                      else np.zeros_like(eps_local))
                def _norm(v):
                    nv = np.linalg.norm(v)
                    return v / (nv + 1e-8)
                eps_total = (0.6 * _norm(eps_local) +
                             0.25 * _norm(le) +
                             0.15 * _norm(te))
            else:
                eps_total = eps_local

            eps_total = np.clip(eps_total, -0.3, 0.3)

            if self.cfg.precision_mode == 'error':
                sigma_k = np.linalg.norm(eps_local)
                precision_k = 1.0 / (sigma_k + 1e-4)
            else:
                conf_k = max(0.0, float(np.dot(mu_k, shifted_pattern)))
                sigma_k = 1.0 / (1.0 + conf_k + 1e-8)
                precision_k = 1.0 / (sigma_k**2 + 1e-8)
            precision_k = min(5.0, precision_k)

            lr_k = self.cfg.lr * weights[k] * precision_k

            layer[k] += lr_k * eps_total
            layer[k] *= 0.9995
            layer[k] /= (np.linalg.norm(layer[k]) + 1e-8)

            self.W_gen += lr_k * np.outer(eps_local, layer_old)

        # Global norm clamp na W_gen (alternatywa dla row-norm — akceptowalne z *=0.999)
        w_norm = np.linalg.norm(self.W_gen)
        if w_norm > 5.0:
            self.W_gen *= 5.0 / w_norm

        self.phi_stability[level] += weights
        self.phi_stability[level] = np.clip(self.phi_stability[level], 0, self.cfg.phi_stability_max)
        self.phi_stability[level] *= self.cfg.phi_stability_decay

        self._last_surprise = float(np.mean([
            np.linalg.norm(shifted_pattern - np.tanh(self.W_gen @ self.phi[level][k]))
            for k in range(self.cfg.k)
        ]))

        if self._last_surprise > self.cfg.surprise_trigger:
            self.cfg.lr *= (1.0 + self.cfg.surprise_adapt_rate)
        else:
            self.cfg.lr *= (1.0 - self.cfg.surprise_adapt_rate * 0.5)
        self.cfg.lr = float(np.clip(self.cfg.lr, self.cfg.lr_min, self.cfg.lr_max))

        self.phi *= 0.999
        self.phi = np.clip(self.phi, -1.0, 1.0)

        beta = self.cfg.phi_ortho_beta
        if beta > 0.0:
            for lv in range(self.cfg.phi_levels):
                phi_new = self.phi[lv].copy()
                for i in range(self.cfg.k):
                    row = self.phi[lv][i].copy()
                    for j in range(self.cfg.k):
                        if i != j:
                            row -= beta * float(np.dot(row, self.phi[lv][j])) * self.phi[lv][j]
                    phi_new[i] = row / (np.linalg.norm(row) + 1e-8)
                self.phi[lv] = phi_new

        lr_cross = self.cfg.lr * 0.3
        for lv in range(self.cfg.phi_levels - 1):
            low  = self._phi_center(level=lv)
            high = self._phi_center(level=lv + 1)
            p    = min(len(low), len(high))
            e    = (high[:p] - low[:p])
            n    = np.linalg.norm(e)
            if n > 1e-8:
                e = np.clip(e / n, -0.3, 0.3)
                for k in range(self.cfg.k):
                    self.phi[lv][k][:p]   += lr_cross * e
                    self.phi[lv][k]       /= (np.linalg.norm(self.phi[lv][k]) + 1e-8)
                    self.phi[lv+1][k][:p] -= lr_cross * 0.5 * e
                    self.phi[lv+1][k]     /= (np.linalg.norm(self.phi[lv+1][k]) + 1e-8)

        for lv in range(self.cfg.phi_levels):
            if np.std(self.phi_stability[lv]) > 2.0:
                wi = int(np.argmin(self.phi_stability[lv]))
                noise = np.random.randn(tdim).astype(np.float32) * 0.005
                self.phi[lv][wi] += noise
                self.phi[lv][wi] /= (np.linalg.norm(self.phi[lv][wi]) + 1e-8)

    # ── Semantic merge ────────────────────────────────────────

    def _semantic_merge(self, item: "Item", new_emb: np.ndarray) -> None:
        cdim = self.cfg.dim
        c1   = item.emb_content(cdim)
        c2   = new_emb[:cdim]
        t1   = item.emb_time(cdim)
        t2   = new_emb[cdim:]

        c_merged = (item.cluster_size * c1 + c2) / (item.cluster_size + 1.0)
        # [B5] Czas cykliczny (sinusy) — średnia psuje fazę gdy stary/nowy timestamp
        # leży blisko po przeciwnych stronach cyklu (amplitude collapse).
        # Zachowaj najnowszy sygnał czasu (t2) — brak korupcji fazy.
        t_merged = t2

        merged = np.concatenate([c_merged, t_merged])
        merged /= (np.linalg.norm(merged) + 1e-8)

        old_size = item.cluster_size
        item.cluster_size += 1
        item.created_at = (old_size * item.created_at + time.time()) / item.cluster_size

        item.embedding    = merged.tolist()
        item.relevance    = min(5.0, item.relevance + 0.2)
        item.age          = 0
        item._norm        = -1.0

    # ── Ruminacja ─────────────────────────────────────────────

    def ruminate(self, force: bool = False) -> Optional[str]:
        if not force and (self.turns % self.cfg.rumination_interval != 0):
            return None

        core  = self.phi[2].mean(axis=0)
        short = self.phi[0].mean(axis=0)
        mid   = self.phi[1].mean(axis=0)

        projections   = [HolographicInterference.phase_shift(core, s)
                         for s in self.cfg.rumination_shifts]
        inconsistencies = [abs(float(np.dot(p, short)) - float(np.dot(p, mid)))
                           for p in projections]
        max_inc = max(inconsistencies)

        if self.cfg.phase_shifts_learnable:
            target = self.cfg.rumination_threshold / 2.0
            lr_ps  = 0.05
            for lv in range(self.cfg.phi_levels):
                lv_lr = lr_ps * (0.5 ** (self.cfg.phi_levels - 1 - lv))
                self.cfg.phase_shifts[lv] += lv_lr * (target - max_inc)
                # fftfreq: shift w cyklach — modulo 1.0 jest poprawne (1 cykl = pełen obrót)
                self.cfg.phase_shifts[lv] %= 1.0

        if max_inc > self.cfg.rumination_threshold or force:
            if self.insight_llm_callback is not None:
                try:
                    prompt = self.cfg.insight_prompt_template.format(max_inc=max_inc)
                    reflection = self.insight_llm_callback(prompt)[:400]
                except Exception:
                    reflection = ""
            else:
                reflection = ""

            if not reflection:
                reflection = (f"[Ruminacja t={self.turns}] "
                              f"Niespójność temporalna: {max_inc:.3f}. "
                              f"Krótki vs średni termin rozchodzą się — "
                              f"konsoliduję rdzeń.")
            refl_emb = self.embedder.encode_timed(reflection)
            # shift=0.9 ≈ 90% cyklu przy fftfreq — sensowna separacja w przestrzeni
            shifted = HolographicInterference.phase_shift(refl_emb, 0.9)
            if len(shifted) < self.cfg.total_dim:
                pad = np.zeros(self.cfg.total_dim - len(shifted), dtype=np.float32)
                shifted = np.concatenate([shifted, pad])
            shifted = shifted[:self.cfg.total_dim]
            shifted /= (np.linalg.norm(shifted) + 1e-8)
            for k in range(self.cfg.k):
                self.phi[2][k] = 0.85 * self.phi[2][k] + 0.15 * shifted
                self.phi[2][k] /= (np.linalg.norm(self.phi[2][k]) + 1e-8)
            print(f"\n[Ruminacja] Niespójność: {max_inc:.3f} → rdzeń zaktualizowany")
            return reflection
        return None

    # ── Build window + messages ───────────────────────────────

    def _build_window(self, query_emb: np.ndarray) -> list:
        center   = self._phi_center(query_emb, level=2)
        cdim     = self.cfg.dim
        center_c = center[:cdim] / (np.linalg.norm(center[:cdim]) + 1e-8)
        # Fakty i praca nad projektami zawsze w oknie
        protected     = [i for i in self.store if i.age <= 1 or i.recalled or i.is_fact or i.is_work]
        protected_ids = {id(i) for i in protected}
        candidates    = sorted(
            [i for i in self.store if id(i) not in protected_ids],
            key=lambda x: -self._cosine_sim(x.emb_content(cdim), center_c))
        return (protected + candidates)[:self.cfg.n]

    def _build_messages(self, window: list, user_message: str,
                        system_prompt: str) -> list:
        msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
        if window:
            ctx = [i for i in window if i.content != user_message]
            # Trzy sekcje w kolejności priorytetu: projekty → fakty → kontekst
            work_items = [i for i in ctx if i.is_work]
            fact_items = [i for i in ctx if i.is_fact and not i.is_work]
            regular    = [i for i in ctx if not i.is_fact and not i.is_work]
            mem_parts  = []
            if work_items:
                work_lines = [f"• {i.content[:400]}" for i in work_items]
                mem_parts.append(
                    "AKTYWNE PROJEKTY (najwyższy priorytet — to nad czym pracujemy):\n"
                    + "\n".join(work_lines))
            if fact_items:
                fact_lines = [f"• {i.content[:300]}" for i in fact_items]
                mem_parts.append(
                    "TRWAŁE FAKTY (zawsze prawdziwe — nie mów że nie wiesz):\n"
                    + "\n".join(fact_lines))
            if regular:
                max_chars = max(200, 9856 // max(1, len(regular)))
                reg_lines = [f"[t-{i.age}{'★' if i.recalled else ''}] {i.content[:max_chars]}"
                             for i in regular]
                mem_parts.append("Pamięć sesji:\n" + "\n---\n".join(reg_lines))
            if mem_parts:
                msgs.append({"role": "system", "content": "\n\n".join(mem_parts)})
        msgs.append({"role": "user", "content": user_message})
        return msgs

    # ── Turn ──────────────────────────────────────────────────

    def turn(self, user_message: str, system_prompt: str = "") -> list:
        q_timed = self.embedder.encode(user_message, timestamp=time.time())
        current_center = self._phi_center(level=2)

        if self.prev_phi_center is not None:
            pred_center = self.W_time @ self.prev_phi_center
            pred_center /= (np.linalg.norm(pred_center) + 1e-8)

            temporal_error = current_center - pred_center
            temporal_error /= (np.linalg.norm(temporal_error) + 1e-8)
            self.temporal_error = temporal_error.copy()

            raw_spatial = q_timed[:len(current_center)] - current_center
            raw_spatial = np.clip(raw_spatial, -0.5, 0.5)

            if self.last_error is None:
                self.last_error = 0.7 * raw_spatial + 0.3 * temporal_error[:len(raw_spatial)]
            else:
                combined = 0.7 * raw_spatial + 0.3 * temporal_error[:len(raw_spatial)]
                self.last_error = 0.7 * self.last_error + 0.3 * combined

            grad = np.outer(current_center - self.prev_phi_center, self.prev_phi_center)
            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-6:
                grad /= g_norm
            self.W_time += self.cfg.lr * 0.1 * grad
            decay = 0.999 - 0.2 * min(1.0, self._last_surprise)
            self.W_time = decay * self.W_time + (1 - decay) * np.eye(self.cfg.total_dim)
            w_norm = np.linalg.norm(self.W_time)
            if w_norm > 5.0:
                self.W_time *= 5.0 / w_norm
        else:
            raw_error = q_timed[:len(current_center)] - current_center
            self.last_error = np.clip(raw_error, -0.5, 0.5)
            self.temporal_error = None

        self._recall(q_timed)

        skip = False
        if self.store:
            best_sim, best_item = -1.0, None
            for i in self.store:
                sim = self._csim(q_timed, i.emb_np())
                if sim > best_sim:
                    best_sim, best_item = sim, i
            # Nie scalaj jeśli to fakt lub praca nad projektem
            is_new_fact = (any(p in user_message.lower() for p in self.FACT_PATTERNS)
                           and "?" not in user_message)
            is_new_work = (self.aii.focus_active or
                           any(p in user_message.lower() for p in self.FOCUS_PATTERNS))
            if best_sim > 0.95 and not best_item.is_fact and not best_item.is_work \
                    and not is_new_fact and not is_new_work:
                self._semantic_merge(best_item, q_timed)
                skip = True

        if not skip:
            self.store.append(Item(
                id=str(uuid.uuid4()),
                content=user_message[:500],
                embedding=q_timed.tolist(),
                age=0))

        self._vacuum(q_timed)
        window = self._build_window(q_timed)
        self._update_phi(window)
        for item in self.store: item.recalled = False
        self.turns += 1
        self.prev_phi_center = self._phi_center(level=2).copy()
        return self._build_messages(window, user_message, system_prompt)

    # ── After turn ────────────────────────────────────────────

    def after_turn(self, user_message: str, response: str) -> None:
        response = (response or "[brak odpowiedzi]")
        MAX_C    = 500
        combined = f"User: {user_message[:MAX_C]}\nAssistant: {response[:MAX_C]}"
        t_now    = time.time()
        comb_emb = self.embedder.encode(combined, timestamp=t_now)
        self.aii.update(user_message + " " + response, comb_emb)

        skip = False
        if self.store:
            best_sim, best_item = -1.0, None
            for i in self.store:
                sim = self._csim(comb_emb, i.emb_np())
                if sim > best_sim:
                    best_sim, best_item = sim, i
            # Nie scalaj jeśli to fakt lub praca nad projektem
            is_new_fact = (any(p in user_message.lower() for p in self.FACT_PATTERNS)
                           and "?" not in user_message)
            is_new_work = (self.aii.focus_active or
                           any(p in user_message.lower() for p in self.FOCUS_PATTERNS))
            if best_sim > 0.95 and not best_item.is_fact and not best_item.is_work \
                    and not is_new_fact and not is_new_work:
                self._semantic_merge(best_item, comb_emb)
                skip = True

        if not skip:
            # Detekcja: fakt osobisty lub aktywna praca nad projektem
            is_fact = (any(p in user_message.lower() for p in self.FACT_PATTERNS)
                       and "?" not in user_message)
            is_work = (self.aii.focus_active or
                       any(p in user_message.lower() for p in self.FOCUS_PATTERNS))
            self.store.append(Item(
                id=str(uuid.uuid4()),
                content=combined[:800],
                embedding=comb_emb.tolist(),
                relevance=self.aii.get_emotion_weight(),
                is_fact=is_fact,
                is_work=is_work))

        self._vacuum(comb_emb)
        self._update_phi(self._build_window(comb_emb))
        for it in self.store: it.age += 1

        self.ruminate()

        self.memory.save(self.phi, self.store, self.turns,
                         self.aii.to_dict(), self.phi_stability.tolist())
        if hasattr(self.embedder, 'save'):
            self.embedder.save()

    # ── Przypomnienia ─────────────────────────────────────────

    def add_reminder(self, text: str, timestamp: float) -> None:
        emb = self.embedder.encode(text, timestamp=timestamp)
        reminder = Item(
            id=str(uuid.uuid4()),
            content=text,
            embedding=emb.tolist(),
            created_at=timestamp,
            is_reminder=True,
            relevance=2.0
        )
        self.store.append(reminder)
        print(f"[Przypomnienie] Dodano: '{text}' na "
              f"{datetime.datetime.fromtimestamp(timestamp)}")

    def get_upcoming_reminders(self, within_seconds: int = 3600) -> list:
        now = time.time()
        upcoming = []
        for item in self.store:
            if item.is_reminder and now <= item.created_at <= now + within_seconds:
                upcoming.append(item)
        upcoming.sort(key=lambda x: x.created_at)
        return upcoming

    # ── Temporalne API ────────────────────────────────────────

    def recall_at(self, query: str, target_time: float, top_k: int = 5) -> list:
        """Cofanie w czasie — rekonstrukcja Φ z danego momentu."""
        hours_ago = (time.time() - target_time) / 3600.0
        # [R1] +hours_ago — przywrócono poprawkę FIX-5
        # evolve_phi używa abs() wewnętrznie; sign nie ma znaczenia dla decay,
        # ale +hours_ago jest semantycznie poprawne.
        phi_then = np.zeros_like(self.phi)
        for lv in range(self.cfg.phi_levels):
            phi_then[lv] = TimeDecay.evolve_phi(
                self.phi[lv], hours_ago,
                self.cfg.phi_half_life_hours, self.cfg.phi_min_norm, level=lv)

        q_full  = self.embedder.encode(query, timestamp=target_time)
        cdim    = self.cfg.dim
        q_c     = q_full[:cdim]

        layer  = phi_then[2]
        norms  = np.linalg.norm(layer, axis=1)
        exp_n  = np.exp(norms - norms.max())
        wts    = exp_n / (exp_n.sum() + 1e-8)
        center = sum(wts[k] * layer[k] for k in range(self.cfg.k))
        center_c = center[:cdim] / (np.linalg.norm(center[:cdim]) + 1e-8)

        scored = []
        for item in self.store:
            e_c  = item.emb_content(cdim)
            e_t  = item.emb_np()[cdim:]
            q_t  = q_full[cdim:]
            s1   = self._cosine_sim(e_c, q_c)
            s2   = self._cosine_sim(e_t, q_t) if len(e_t) == len(q_t) else 0.0
            s3   = self._cosine_sim(e_c, center_c)
            scored.append((item, 0.5 * s1 + 0.2 * s2 + 0.3 * s3))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def trajectory(self, topic: str, top_k: int = 10) -> list:
        q_full = self.embedder.encode(topic, timestamp=time.time())
        cdim   = self.cfg.dim
        q_c    = q_full[:cdim]
        scored = [(item.created_at, item.content)
                  for item in self.store
                  if self._cosine_sim(item.emb_content(cdim), q_c) > 0.3]
        scored.sort(key=lambda x: x[0])
        return scored[:top_k]

    # ── Stats & reset ─────────────────────────────────────────

    def set_insight_callback(self, cb) -> None:
        self.insight_llm_callback = cb

    def stats(self) -> dict:
        if self.phi is None:
            return {"turns": 0, "store": 0, "phi_norms": [], "phi_stability": [],
                    "aii": self.aii.to_dict(), "delta_hours": 0.0,
                    "warning": "start_session() not called"}
        phi_norms = [np.linalg.norm(self.phi[lv], axis=1).tolist()
                     for lv in range(self.cfg.phi_levels)]
        return {
            "turns":         self.turns,
            "store":         len(self.store),
            "phi_norms":     phi_norms,
            "phi_stability": self.phi_stability.tolist(),
            "aii":           self.aii.to_dict(),
            "delta_hours":   round(self._delta_hours, 2),
            "phase_shifts":  list(self.cfg.phase_shifts),
            "last_error_norm": round(float(np.linalg.norm(self.last_error)), 4)
                               if self.last_error is not None else 0.0,
            "temporal_drift": round(float(np.linalg.norm(
                self._phi_center(level=2) - self.prev_phi_center))
                if self.prev_phi_center is not None else 0.0, 4),
            "surprise":       round(self._last_surprise, 4),
            "lr_current":     round(self.cfg.lr, 5),
        }

    def reset(self):
        self.memory.delete()
        self.phi             = PersistentMemory._init_phi(self.cfg)
        self.phi_stability   = np.zeros((self.cfg.phi_levels, self.cfg.k),
                                        dtype=np.float32)
        self.store           = []
        self.turns           = 0
        self._delta_hours    = 0.0
        self.aii             = AIIState(self.embedder)
        self._session_start_turn = 0
        self.last_error      = None
        self.prev_phi_center = None
        self._last_surprise  = 0.0
        self.W_time = np.random.randn(self.cfg.total_dim, self.cfg.total_dim) * 0.01
        self.W_gen  = np.random.randn(self.cfg.total_dim, self.cfg.total_dim) * 0.01
        self.temporal_error = None


# ============================================================
# UNIWERSALNY KLIENT API (OpenAI-compatible — bez zmian)
# ============================================================

class OpenAIClient:
    def __init__(self, api_key: str,
                 base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-chat"):
        self.api_key  = api_key
        self.base_url = base_url.rstrip('/')
        self.model    = model

    def chat_completion(self, messages: List[Dict[str, str]],
                        temperature: float = 0.7,
                        max_tokens: int = 1024) -> str:
        filtered = []
        for msg in messages:
            content = msg.get("content", "")
            if content and content.strip():
                filtered.append(msg)
            else:
                print(f"[API] Ostrzeżenie: Pominięto wiadomość z pustą treścią: {msg}")
        if not filtered:
            return "[Błąd: brak wiadomości do wysłania]"

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": filtered,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            print(f"[API] Błąd HTTP {e.response.status_code}")
            try:
                print(f"[API] Treść błędu: {e.response.json()}")
            except Exception:
                print(f"[API] Treść błędu (tekst): {e.response.text}")
            return f"[Błąd komunikacji z API: {e}]"
        except Exception as e:
            print(f"[API] Błąd: {e}")
            return f"[Błąd komunikacji z API: {e}]"



# ============================================================
# REMINDER WATCHER — wątek tła, aktywnie przerywa konwersację
# ============================================================

class ReminderWatcher:
    """Wątek tła sprawdzający co CHECK_INTERVAL sekund czy jakieś
    przypomnienie minęło. Jeśli tak — drukuje powiadomienie do konsoli
    przerywając aktualny prompt (\n przed komunikatem), i oznacza
    item jako fired (is_reminder=False) żeby nie powtarzać.

    Działanie w CLI:
      - Użytkownik pisze coś → wątek przerywa go sygnałem dźwiękowym + tekstem
      - Przy następnym input() użytkownik zobaczy komunikat nad swoim cursorem

    Opcjonalny callback on_fire(item) pozwala podłączyć inne UI (np. Telegram).
    """
    CHECK_INTERVAL: int = 15  # sekundy między sprawdzeniami

    def __init__(self, holomem: "HoloMem",
                 on_fire=None,
                 check_interval: int = CHECK_INTERVAL):
        self.holomem        = holomem
        self.on_fire        = on_fire
        self._check_interval = check_interval
        self._stop_event    = threading.Event()
        self._thread        = threading.Thread(
            target=self._run, daemon=True, name="ReminderWatcher")

    def start(self) -> None:
        self._stop_event.clear()
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.wait(self._check_interval):
            self._check()

    def _check(self) -> None:
        now = time.time()
        fired = []
        for item in self.holomem.store:
            if item.is_reminder and item.created_at <= now:
                fired.append(item)

        for item in fired:
            # Oznacz jako wykonane — zmień na zwykły item żeby nie wracało
            item.is_reminder = False

            msg = (f"\n\a"  # \a = bell (dźwięk terminalowy)
                   f"╔══════════════════════════════════════╗\n"
                   f"║  🔔 PRZYPOMNIENIE: {item.content[:35]:<35} ║\n"
                   f"╚══════════════════════════════════════╝")
            print(msg, flush=True)

            if self.on_fire:
                try:
                    self.on_fire(item)
                except Exception as e:
                    print(f"[ReminderWatcher] callback error: {e}")


# ============================================================
# SESSION
# ============================================================

class Session:
    DEFAULT_SYSTEM = (
        "Jesteś asystentem z pamięcią sesji. "
        "Sekcja 'Pamięć sesji' zawiera fakty z tej rozmowy — traktuj je jako PEWNĄ WIEDZĘ.\n"
        "ZASADY:\n"
        "1. Jeśli fakt jest w pamięci sesji — podaj go wprost.\n"
        "2. Data, godzina, imię, nazwa projektu — jeśli podano, PAMIĘTASZ to.\n"
        "3. Twoja wiedza z treningu jest drugorzędna wobec faktów z pamięci sesji.\n"
        "4. Nie mów 'nie pamiętam' jeśli fakt jest w kontekście.\n"
        "Odpowiadaj w języku rozmówcy. Domyślnie po polsku."
    )

    def __init__(self, memory_path="holon_memory.json", cfg=None,
                 system=None, model=None, api_key: Optional[str] = None):
        self.system = system or self.DEFAULT_SYSTEM

        self._api_key = (api_key
                         or os.environ.get("GROQ_API_KEY")
                         or os.environ.get("DEEPSEEK_API_KEY", ""))
        if not self._api_key:
            print("[Session] Ostrzeżenie: Brak klucza API. "
                  "Ustaw GROQ_API_KEY lub DEEPSEEK_API_KEY.")

        if self._api_key:
            if self._api_key.startswith("gsk_"):
                base_url      = "https://api.groq.com/openai/v1"
                default_model = model or "llama-3.3-70b-versatile"
            else:
                base_url      = "https://api.deepseek.com/v1"
                default_model = model or "deepseek-chat"
            self._client = OpenAIClient(
                api_key=self._api_key, base_url=base_url, model=default_model)
        else:
            self._client = None

        cfg_ = cfg or Config()
        emb  = Embedder(dim=cfg_.dim,
                        dict_path=memory_path.replace(".json", "_kurz.json"),
                        time_dim=cfg_.time_dim)
        self.holomem = HoloMem(emb, cfg_, memory_path)
        # Watcher startuje po start() — wtedy holomem.store jest załadowany
        self._watcher: Optional[ReminderWatcher] = None

    def start(self) -> str:
        res = self.holomem.start_session()
        s   = self.holomem.stats()
        aii = s["aii"]
        print(f"\n[Holon v5.4] tur={s['turns']} store={s['store']} "
              f"delta={s['delta_hours']}h "
              f"aii={aii['emotion']}(focus:{aii['focus']})")
        # Uruchom wątek przypomnień
        self._watcher = ReminderWatcher(self.holomem)
        self._watcher.start()
        print("[ReminderWatcher] Uruchomiony (sprawdzanie co "
              f"{ReminderWatcher.CHECK_INTERVAL}s)")
        return res.get("wake", "")

    def _parse_reminder(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """[R5] Typ zwracany kompatybilny z Python 3.9+.
        [B1] Guard: parsuj tylko jeśli tekst zawiera "przypomnij"/"remind".
        [B4] Content = tekst PO słowie kluczowym (nie sub z całości).
             Poprzednia wersja: reminder_pattern.sub('', text) usuwała słowo kluczowe
             ale zostawiała wszystko PRZED nim — "super przypomnij..." dawało content "super...".
             Nowa wersja: bierzemy text[match.end():] → tylko to co po keyword.
        """
        reminder_pattern = re.compile(r'(?:przypomnij|remind)(?:\s+m(?:i|e))?',
                                      re.IGNORECASE)
        # [B1] Nie ma słowa kluczowego → to nie jest polecenie przypomnienia
        kw_match = reminder_pattern.search(text)
        if not kw_match:
            return None, None

        # [B4] Bierz TYLKO tekst PO słowie kluczowym
        after_kw = text[kw_match.end():].strip()
        if not after_kw:
            return None, None

        def _clean_content(s: str) -> str:
            """Usuń wyrażenia czasowe i normalizuj whitespace."""
            s = re.sub(r'jutro\s+o\s+\d{1,2}:\d{2}\b', '', s, flags=re.IGNORECASE)
            s = re.sub(r'o\s+\d{1,2}:\d{2}\b', '', s, flags=re.IGNORECASE)
            s = re.sub(r'za\s+\d+\s+(?:godzin|godziny|godzinę)\b', '', s, flags=re.IGNORECASE)
            s = re.sub(r'za\s+\d+\s+(?:minut|minuty|minutę)\b', '', s, flags=re.IGNORECASE)
            return re.sub(r'\s{2,}', ' ', s).strip()

        match_jutro = re.search(r'jutro\s+o\s+(\d{1,2}):(\d{2})\b', after_kw, re.IGNORECASE)
        if match_jutro:
            hour, minute = map(int, match_jutro.groups())
            dt = datetime.datetime.now() + datetime.timedelta(days=1)
            dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return _clean_content(after_kw), dt.timestamp()

        match_time = re.search(r'o\s+(\d{1,2}):(\d{2})\b', after_kw, re.IGNORECASE)
        if match_time:
            hour, minute = map(int, match_time.groups())
            now = datetime.datetime.now()
            dt  = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if dt < now:
                dt += datetime.timedelta(days=1)
            return _clean_content(after_kw), dt.timestamp()

        match_za_h = re.search(r'za\s+(\d+)\s+(?:godzin|godziny|godzinę)\b',
                                after_kw, re.IGNORECASE)
        if match_za_h:
            hours = int(match_za_h.group(1))
            dt = datetime.datetime.now() + datetime.timedelta(hours=hours)
            return _clean_content(after_kw), dt.timestamp()

        match_za_m = re.search(r'za\s+(\d+)\s+(?:minut|minuty|minutę)\b',
                                after_kw, re.IGNORECASE)
        if match_za_m:
            minutes = int(match_za_m.group(1))
            dt = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
            return _clean_content(after_kw), dt.timestamp()

        if HAS_DATEUTIL:
            if any(kw in after_kw.lower()
                   for kw in ['o ', 'jutro', 'za ', 'godzin', 'minut']):
                return None, None
            try:
                dt = date_parser.parse(after_kw, fuzzy=True)
                if dt.year == 1900:
                    dt = dt.replace(year=datetime.datetime.now().year)
                timestamp = dt.timestamp()
                if timestamp < time.time():
                    dt += datetime.timedelta(days=1)
                    timestamp = dt.timestamp()
                return _clean_content(after_kw), timestamp
            except Exception:
                pass
        return None, None

    def chat(self, user_input: str) -> str:
        current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_context = f"Aktualna data i godzina: {current_time_str}"

        if len(user_input) > 5:
            reminder_text, reminder_time = self._parse_reminder(user_input)
            if reminder_text and reminder_time:
                self.holomem.add_reminder(reminder_text, reminder_time)
                user_input = (f"{user_input}\n[SYSTEM: Ustawiono przypomnienie na "
                              f"{datetime.datetime.fromtimestamp(reminder_time)}]")

        upcoming = self.holomem.get_upcoming_reminders(within_seconds=3600)
        reminder_msg = ""
        if upcoming:
            reminder_lines = []
            for r in upcoming:
                delta   = r.created_at - time.time()
                minutes = int(delta // 60)
                reminder_lines.append(f"- {r.content} (za {minutes} minut)")
            reminder_msg = ("[PRZYPOMNIENIA] Nadchodzące wydarzenia:\n"
                            + "\n".join(reminder_lines) + "\n")

        messages = self.holomem.turn(user_input, self.system)

        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + time_context
        else:
            messages.insert(0, {"role": "system", "content": time_context})

        if reminder_msg:
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += "\n\n" + reminder_msg
            else:
                messages.insert(0, {"role": "system", "content": reminder_msg})

        response = self._call_llm(messages)
        self.holomem.after_turn(user_input, response)

        s   = self.holomem.stats()
        aii = s["aii"]
        print(f"  [store={s['store']} "
              f"aii={aii['emotion']}(focus:{aii['focus']}) "
              f"vac={aii['vacuum_signal']:+.2f} "
              f"lr={s['lr_current']:.5f}]", flush=True)
        return response

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        if not self._client:
            return "[Mock] Brak klucza API. Ustaw GROQ_API_KEY lub DEEPSEEK_API_KEY."
        try:
            return self._client.chat_completion(messages, temperature=0.7, max_tokens=1024)
        except Exception as e:
            print(f"[Session] Błąd podczas wywołania API: {e}")
            return f"[Błąd LLM: {e}]"

    def set_insight_callback(self, cb) -> None:
        self.holomem.set_insight_callback(cb)

    def stats(self) -> dict:
        return self.holomem.stats()

    def reset(self):
        self.holomem.reset()
        print("[Holon] Pamięć wyczyszczona.")

    def stop_watcher(self) -> None:
        """Zatrzymaj wątek przypomnień (wywołaj przed wyjściem z programu)."""
        if self._watcher:
            self._watcher.stop()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Holon v5.4 — Predictive Coding + Przypomnienia + Trwałe fakty")
    print("=" * 60)

    session  = Session(memory_path="holon_memory.json")
    wake_msg = session.start()
    if wake_msg:
        print(f"\n{wake_msg}\n")

    print("Komendy: quit, stats, reset, ruminate")
    print("Przykłady przypomnień:")
    print("  przypomnij mi o spotkaniu o 15:00")
    print("  przypomnij jutro o 14:30 o przerwie")
    print("  przypomnij za 2 godziny o przerwie")
    print("  przypomnij za 5 minut o przerwie")
    print("Fakty (zapamiętywane trwale):")
    print("  mój ulubiony kwiat to ...")
    print("  jestem ...")
    print("-" * 60)

    try:
        while True:
            try:
                user = input("\nTy: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nDo widzenia.")
                break
            if not user: continue
            if user.lower() == "quit": break
            if user.lower() == "stats":
                print(f"\n[Stats] {session.stats()}"); continue
            if user.lower() == "reset":
                session.reset(); continue
            if user.lower() == "ruminate":
                session.holomem.ruminate(force=True); continue

            print("\nAsystent: ", end="", flush=True)
            print(session.chat(user))
    finally:
        session.stop_watcher()
