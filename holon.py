"""
Holon v4.0 — Holograficzna Pamięć w Czasoprzestrzeni
===========================================================
Autor koncepcji: Maciej Mazur
Implementacja: Partner AI

ZMIANY v4.0 — Oś czasu jako wymiar przestrzeni:
[TIME-1] Czas jako wymiar przestrzeni embeddingów.
  Każdy item kodowany z timestamp systemu → punkt w R^dim × R_time.
  Sinusoidalne positional encoding (8 osi) dołączone do KuRz embeddings.

[TIME-2] recall_at(query, target_time) — cofanie się w czasie.
  Phi z przeszłości = inna geometria percepcji = inne wspomnienia.
  TimeDecay.evolve_phi() używany do rekonstrukcji stanu Phi w danym momencie.

[TIME-3] Synergiczny zanik: TimeDecay (Phi) + time_embed (Item) spójne.
  Oba mechanizmy czasu mówią to samo — jeden model temporalny.

[TIME-4] trajectory(topic) — jak myślenie o temacie ewoluowało w czasie.
  Zwraca listę (timestamp, item) posortowaną chronologicznie.

[TIME-5] Migracja plików v3.x → v4.0 (auto-detect dim w load()).
  Stare pliki bez time_embed ładowane w trybie kompatybilności.

[BREAK] dim_content + TIME_DIM = dim_total. Config.dim = 256 (content).
  Całkowity wymiar: 264 (256 + 8 osi czasu). Stare pliki: patrz TIME-5.
"""

import os
import json
import math
import time
import uuid
import hashlib
import numpy as np
from typing import Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from kurz import KuRz as _KuRz


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
    dim:           int   = 256       # content dim (KuRz)
    time_dim:      int   = 8         # osi czasu (sinusoidy)

    @property
    def total_dim(self) -> int:
        """Całkowity wymiar embeddingu: content + czas."""
        return self.dim + self.time_dim

    phi_half_life_hours: list = field(default_factory=lambda: [48.0, 24.0, 8.0, 4.0])
    store_decay_hours: float = 72.0
    phi_min_norm: float = 0.1
    phi_ortho_beta: float = 0.05

    vacuum_age_tau: float = 50.0
    recall_age_penalty: float = 0.02
    aii_adapt_range: float = 0.15
    vacuum_warmup_turns: int = 8

    phi_stability_decay: float = 0.95
    phi_stability_max:   float = 5.0

    # Minimalny próg koherencji przy odczycie hologramu (poniżej = szum)
    coherence_threshold: float = 0.4


# ============================================================
# ITEM — jednostka pamięci
# ============================================================

@dataclass
class Item:
    id:         str
    content:    str
    embedding:  list
    age:        int    = 0
    recalled:   bool   = False
    relevance:  float  = 1.0
    created_at: float  = field(default_factory=time.time)
    _norm:      float  = field(default=-1.0, repr=False)  # cached ||v||

    def emb_np(self) -> np.ndarray:
        return np.array(self.embedding, dtype=np.float32)

    def emb_content(self, content_dim: int = 256) -> np.ndarray:
        """Content-część embeddingu bez osi czasu."""
        return np.array(self.embedding[:content_dim], dtype=np.float32)

    def norm(self) -> float:
        """Zwraca ||embedding|| — liczy raz, potem z cache."""
        if self._norm < 0:
            self._norm = float(np.linalg.norm(self.embedding))
        return self._norm


# ============================================================
# HOLOGRAFIA (Splot i Koherencja)
# ============================================================

class HolographicInterference:
    """
    Tworzenie par hologramów. Splot kołowy w domenie częstotliwości (FFT).
    P4: cache unitary keys — unikamy wielokrotnego FFT tych samych kluczy.
    """
    _unitary_cache: dict = {}  # klucz: id(array) lub hash → unitary FFT

    @staticmethod
    def _to_unitary(v: np.ndarray) -> np.ndarray:
        # Cache unitary po hash zawartości (klucze Φ się nie zmieniają często)
        key = v.tobytes()
        if key in HolographicInterference._unitary_cache:
            return HolographicInterference._unitary_cache[key]
        v_fft = np.fft.fft(v)
        result = v_fft / (np.abs(v_fft) + 1e-8)
        # Ogranicz cache do 512 wpisów
        if len(HolographicInterference._unitary_cache) >= 512:
            HolographicInterference._unitary_cache.clear()
        HolographicInterference._unitary_cache[key] = result
        return result

    # Secret seed — można nadpisać przez env var HOLON_ANCHOR_SEED
    _ANCHOR_SEED: str = os.environ.get("HOLON_ANCHOR_SEED", "holon-eriamo-4242")

    @staticmethod
    def _salt_key(key: np.ndarray, item_id: str) -> np.ndarray:
        """FIX-P1: Salt = hash(item_id + secret_seed) — nieodwracalne bez seeda."""
        combined = (item_id + HolographicInterference._ANCHOR_SEED).encode()
        h = int(hashlib.sha256(combined).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(h)
        salt = rng.standard_normal(len(key)).astype(np.float32) * 0.1
        salted = key + salt
        return salted / (np.linalg.norm(salted) + 1e-8)

    @staticmethod
    def bind(v1: np.ndarray, v2: np.ndarray, item_id: str = "") -> list:
        key = HolographicInterference._salt_key(v2, item_id) if item_id else v2
        v2_unitary = HolographicInterference._to_unitary(key)
        v1_fft = np.fft.fft(v1)
        bound_fft = v1_fft * v2_unitary
        bound = np.fft.ifft(bound_fft).real.astype(np.float32)
        return bound.tolist()

    @staticmethod
    def unbind(bound_data: list, key: np.ndarray, item_id: str = "") -> np.ndarray:
        key = HolographicInterference._salt_key(key, item_id) if item_id else key
        bound_arr = np.array(bound_data, dtype=np.float32)
        key_unitary = HolographicInterference._to_unitary(key)
        bound_fft = np.fft.fft(bound_arr)
        unbound_fft = bound_fft * np.conj(key_unitary)
        unbound = np.fft.ifft(unbound_fft).real.astype(np.float32)
        return unbound / (np.linalg.norm(unbound) + 1e-8)


# ============================================================
# CZAS JAKO WYMIAR PRZESTRZENI (v4.0)
# ============================================================

# Epoch bazowy — czas pierwszej sesji na tej instalacji
_HOLON_EPOCH: float = float(os.environ.get("HOLON_EPOCH", str(time.time())))


def time_embed(timestamp: float, time_dim: int = 8) -> np.ndarray:
    """
    Sinusoidalne kodowanie czasu z liniowym komponentem anty-aliasing.
    Pary (sin, cos) dla: godzin, dni, tygodni, miesięcy.
    Ostatnia oś: liniowy czas — zapobiega kolizjom okresowym (co tydzień).
    Fix-4: zdarzenia co tydzień mają teraz różne embeddingi dzięki vec[-1].
    """
    delta_days = (timestamp - _HOLON_EPOCH) / 86400.0
    vec = np.zeros(time_dim, dtype=np.float32)
    n_sincos = (time_dim - 1) // 2  # zostawiamy 1 os na komponent liniowy
    scales = [1.0 / 24, 1.0, 7.0, 30.0]
    for i, scale in enumerate(scales[:n_sincos]):
        angle = 2.0 * math.pi * delta_days / (scale + 1e-8)
        vec[i * 2]     = math.sin(angle)
        vec[i * 2 + 1] = math.cos(angle)
    # Liniowy komponent — unikalny dla każdego momentu, normalizowany do roku
    vec[-1] = float(np.clip(delta_days / 365.0, -10.0, 10.0))
    return vec


# ============================================================
# EMBEDDER
# ============================================================

class Embedder:
    def __init__(self, dim: int = 256, dict_path: Optional[str] = None,
                 cache_size: int = 256, time_dim: int = 8):
        self.dim      = dim
        self.time_dim = time_dim
        self._kurz    = _KuRz(dim=dim, dict_path=dict_path)
        self._cache: dict = {}
        self._cache_size  = cache_size
        self._cache_hits  = 0

    def encode(self, text: str, timestamp: float = None) -> np.ndarray:
        """
        Encode tekstu z opcjonalną komponentą czasową.
        timestamp=None  → wektor content-only (dim)        — dla queries
        timestamp=float → wektor [content | time] (total)  — dla zapisu do store
        """
        key = text[:200]
        if timestamp is None:
            if key in self._cache:
                self._cache_hits += 1
                return self._cache[key]
            vec = self._kurz.encode(text)
            self._cache[key] = vec
            if len(self._cache) > self._cache_size:
                del self._cache[next(iter(self._cache))]
            return vec
        # Z timestampem — nie cachujemy (każdy moment inny)
        # Fix-5: content i time mają różne wagi (0.9 / 0.1)
        # Czas nie powinien dominować nad semantyką
        CONTENT_W = 0.9
        TIME_W    = 0.1
        content = self._kurz.encode(text)
        t_vec   = time_embed(timestamp, self.time_dim)
        full    = np.concatenate([content * CONTENT_W, t_vec * TIME_W])
        norm    = np.linalg.norm(full)
        return full / (norm + 1e-8)

    def encode_timed(self, text: str) -> np.ndarray:
        """Skrót: encode z aktualnym timestamp systemu."""
        return self.encode(text, timestamp=time.time())

    def save(self) -> None:
        if self._kurz.dict_path:
            self._kurz.save_dict()

    @property
    def vocab_size(self) -> int:
        return self._kurz.vocab_size

    @property
    def calls(self) -> int:
        return self._kurz.calls


# ============================================================
# AII STATE
# ============================================================

class AIIState:
    SIGNALS = {
        "radosc":      (["super", "swietnie", "doskonale", "rewelacja", "great"], 1.3, +1.0),
        "zaskoczenie": (["wow", "niesamowite", "naprawde", "really"], 1.3, +0.5),
        "strach":      (["blad", "error", "crash", "problem", "awaria", "fail", "bug"], 1.2, -1.0),
        "zlosc":       (["nie dziala", "znowu", "broken", "wrong"], 1.2, -1.0),
        "smutek":      (["niestety", "szkoda", "nie pomaga"], 0.8, -0.5),
        "focus":       (["implementacja", "debug", "refaktor", "logika", "kod", "architektura", "softmax", "eriamo"], 1.5, 0.0),
    }

    def __init__(self):
        self.emotion = "neutral"
        self.vacuum_signal = 0.0
        self.focus_active = False

    def update(self, text: str) -> None:
        t = text.lower()
        self.focus_active = any(kw in t for kw in self.SIGNALS["focus"][0])

        best_e, best_s, best_sig = "neutral", 0, 0.0
        for emo, (kws, _, sig) in self.SIGNALS.items():
            if emo == "focus": continue
            hits = sum(1 for kw in kws if kw in t)
            if hits > best_s:
                best_s, best_e, best_sig = hits, emo, sig

        self.emotion = best_e
        self.vacuum_signal = 0.7 * self.vacuum_signal + 0.3 * best_sig

    def get_emotion_weight(self) -> float:
        w = {"radosc": 1.3, "zaskoczenie": 1.3, "strach": 1.2, "zlosc": 1.2, "smutek": 0.8, "neutral": 1.0}.get(self.emotion, 1.0)
        return w * 1.5 if self.focus_active else w

    def get_threshold_multiplier(self, adapt_range: float) -> float:
        return 1.0 + adapt_range * self.vacuum_signal

    def to_dict(self) -> dict:
        return {"emotion": self.emotion, "vacuum_signal": round(self.vacuum_signal, 3), "focus": self.focus_active}

    def from_dict(self, data: dict) -> None:
        if not data: return
        self.emotion = data.get("emotion", "neutral")
        self.vacuum_signal = float(data.get("vacuum_signal", 0.0))
        self.focus_active = data.get("focus", False)


# ============================================================
# TIME DECAY
# ============================================================

class TimeDecay:
    @staticmethod
    def decay_factor(delta_hours: float, half_life_hours: float) -> float:
        return math.exp(-0.693 * delta_hours / (half_life_hours + 1e-8))

    @staticmethod
    def evolve_phi(phi: np.ndarray, delta_hours: float, hl_list: list, min_norm: float) -> np.ndarray:
        if delta_hours < 0.1: return phi
        evolved = phi.copy()
        for k in range(len(phi)):
            hl = hl_list[k] if k < len(hl_list) else 24.0
            row = phi[k] * TimeDecay.decay_factor(delta_hours, hl)
            if np.linalg.norm(row) < min_norm:
                noise = np.random.randn(len(phi[k])).astype(np.float32) * 0.01
                row = phi[k] * min_norm + noise
            evolved[k] = row / (np.linalg.norm(row) + 1e-8)
        return evolved

    @staticmethod
    def wake_message(delta_hours: float, turns: int, store_size: int, coherence: float = 1.0) -> str:
        if delta_hours < 0.1: return ""
        desc = f"{int(delta_hours*60)} min" if delta_hours < 1 else f"{delta_hours:.1f} h"
        state = "Pamięć świeża." if delta_hours < 2 else "Wzorce konwersacyjne lekko przybladły."
        coh_str = f" | Koherencja stanu: {coherence:.2f}" if coherence < 0.99 else ""
        return f"[Minęło {desc}. Było {turns} tur, {store_size} wzorców w pamięci. {state}{coh_str}]"


# ============================================================
# PERSISTENT MEMORY
# ============================================================

class PersistentMemory:
    def __init__(self, path: str = "holon_memory.json", dim: int = 256):
        self.path = Path(path)

        # FIX-8: Anchor seed konfigurowalny przez env var dla bezpieczeństwa.
        # W pełnym wdrożeniu użyj silnego seeda z hasła lub fingerprinta HW.
        seed_str = os.environ.get("HOLON_ANCHOR_SEED", "4242")
        seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed_int)
        anchor = rng.randn(dim).astype(np.float32)
        self.eriamo_anchor = anchor / (np.linalg.norm(anchor) + 1e-8)

    @staticmethod
    def _phi_center_static(phi: np.ndarray) -> np.ndarray:
        """Weighted center Φ przez softmax norm — spójny z HoloMem._phi_center()."""
        norms = np.linalg.norm(phi, axis=1)
        exp_n = np.exp(norms - norms.max())
        weights = exp_n / (exp_n.sum() + 1e-8)
        center = sum(weights[k] * phi[k] for k in range(len(phi)))
        n = np.linalg.norm(center)
        return center / (n + 1e-8)

    def save(self, phi: np.ndarray, store: list, turns: int, aii: dict = None, stability: list = None):
        # FIX #1: state_now = weighted phi_center (nie mean) — spójny z vacuum/recall
        state_now = PersistentMemory._phi_center_static(phi)

        # Hologram Koherencji: Wiąże stan teraźniejszy z wektorem środowiskowym
        h_coherence = HolographicInterference.bind(state_now, self.eriamo_anchor)

        data = {
            "timestamp": time.time(),
            "turns": turns,
            "phi": phi.tolist(),
            "phi_stability": stability or [],
            "h_coherence": h_coherence,
            "aii": aii or {},
            "store": [
                {
                    "id": i.id,
                    "content": i.content,
                    # Zapisujemy Hologram Danych (Splot z Φ)
                    "embedding": HolographicInterference.bind(i.emb_np()[:len(state_now)], state_now),
                    "age": i.age,
                    "relevance": i.relevance,
                    "created_at": i.created_at
                }
                for i in store if i.age >= 1
            ]
        }
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            tmp.replace(self.path)
        except Exception:
            try:
                self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            finally:
                if tmp.exists():
                    tmp.unlink()

    def load(self, cfg: Config) -> dict:
        tmp = self.path.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()

        if not self.path.exists():
            return {"phi": self._init_phi(cfg), "store": [], "turns": 0, "delta_hours": 0.0,
                    "aii": {}, "phi_stability": None, "loaded": False, "coherence": 1.0}

        try:
            data = json.loads(self.path.read_text())
            saved_at = data["timestamp"]
            delta_hours = (time.time() - saved_at) / 3600.0
            turns = data["turns"]

            # Wczytanie surowego Φ (stan w chwili zapisu)
            phi_raw = np.array(data["phi"], dtype=np.float32)

            # FIX-1 + v4.0: weighted center zamiast mean
            # TIME-5: migracja phi v3→v4 — rozszerz do total_dim
            total_dim = getattr(cfg, "total_dim", cfg.dim)
            if phi_raw.shape[1] < total_dim:
                pad = np.zeros((phi_raw.shape[0], total_dim - phi_raw.shape[1]),
                               dtype=np.float32)
                phi_raw = np.concatenate([phi_raw, pad], axis=1)
                phi_raw /= (np.linalg.norm(phi_raw, axis=1, keepdims=True) + 1e-8)
            state_at_save = PersistentMemory._phi_center_static(phi_raw)

            # FIX-2: Guard dla plików pre-v3.3 bez klucza h_coherence.
            # Zamiast crashować unbind(None, ...), traktujemy brak hologramu
            # jako plik zaufany (kompatybilność wsteczna), ale logujemy info.
            h_coherence = data.get("h_coherence")
            if h_coherence is None:
                print("[Memory] Info: Brak h_coherence (plik pre-v3.3). Ładowanie bez weryfikacji.")
                coherence = 1.0
                recovered_state_now = state_at_save
            else:
                # TIME-5: migracja — anchor i h_coherence moga byc krotsze niz total_dim
                h_arr   = np.array(h_coherence, dtype=np.float32)
                anchor  = self.eriamo_anchor
                h_dim   = len(h_arr)
                a_dim   = len(anchor)
                # Dopasuj wymiary: uzyj min(h_dim, a_dim) dla unbind
                use_dim = min(h_dim, a_dim)
                recovered_state_now = HolographicInterference.unbind(
                    h_arr[:use_dim].tolist(), anchor[:use_dim]
                )
                # Wyrownaj do dlugosci state_at_save przed dot product
                s_dim = len(state_at_save)
                if len(recovered_state_now) < s_dim:
                    pad = np.zeros(s_dim - len(recovered_state_now), dtype=np.float32)
                    recovered_state_now = np.concatenate([recovered_state_now, pad])
                    n = np.linalg.norm(recovered_state_now)
                    recovered_state_now = recovered_state_now / (n + 1e-8)
                coherence = float(np.dot(recovered_state_now[:s_dim], state_at_save))

            # Ewolucja Φ w czasie — dopiero po weryfikacji koherencji
            phi_today = TimeDecay.evolve_phi(phi_raw, delta_hours, cfg.phi_half_life_hours, cfg.phi_min_norm)

            store = []
            if coherence >= cfg.coherence_threshold:
                max_age = cfg.store_decay_hours * 4
                for obj in data.get("store", []):
                    age_now = obj["age"] + int(delta_hours * 4)
                    if age_now <= max_age:
                        # Rozpakowanie hologramu — dopasuj wymiary klucza do embeddingu
                        emb_arr = np.array(obj["embedding"], dtype=np.float32)
                        key_for_unbind = recovered_state_now[:len(emb_arr)]
                        recreated_emb = HolographicInterference.unbind(emb_arr.tolist(), key_for_unbind)
                        created = obj.get("created_at", time.time())
                        raw_emb = recreated_emb.tolist()
                        # TIME-5: migracja — jeśli embedding nie ma komponentu czasu
                        if len(raw_emb) < total_dim:
                            t_vec = time_embed(created, total_dim - len(raw_emb)).tolist()
                            raw_emb = raw_emb + t_vec
                            v = np.array(raw_emb, dtype=np.float32)
                            raw_emb = (v / (np.linalg.norm(v) + 1e-8)).tolist()
                        store.append(Item(
                            id=obj["id"], content=obj["content"], embedding=raw_emb,
                            age=age_now, recalled=False, relevance=obj["relevance"],
                            created_at=created
                        ))
            else:
                print(f"[Memory] Ostrzeżenie: Utrata koherencji stanu ({coherence:.2f}). Pamięć uległa depolaryzacji.")

            return {
                "phi": phi_today, "store": store, "turns": turns, "delta_hours": delta_hours,
                "aii": data.get("aii", {}), "phi_stability": data.get("phi_stability"),
                "wake": TimeDecay.wake_message(delta_hours, turns, len(store), coherence),
                "loaded": True, "coherence": coherence
            }
        except Exception as e:
            print(f"[Memory] Błąd wczytania/depolaryzacji: {e}")
            return {"phi": self._init_phi(cfg), "store": [], "turns": 0, "delta_hours": 0.0,
                    "aii": {}, "phi_stability": None, "loaded": False, "coherence": 0.0}

    def delete(self):
        if self.path.exists():
            self.path.unlink()

    def exists(self) -> bool:
        return self.path.exists()

    @staticmethod
    def _init_phi(cfg: Config) -> np.ndarray:
        total = getattr(cfg, "total_dim", cfg.dim)
        phi = np.random.randn(cfg.k, total).astype(np.float32) * 0.01
        return phi / (np.linalg.norm(phi, axis=1, keepdims=True) + 1e-8)


# ============================================================
# HOLOMEM CORE
# ============================================================

class HoloMem:
    def __init__(self, embedder: Embedder, cfg: Config = None, memory_path: str = "holon_memory.json"):
        self.embedder = embedder
        self.cfg = cfg or Config(dim=embedder.dim)
        self.memory = PersistentMemory(memory_path, dim=self.cfg.total_dim)

        self.phi: np.ndarray = None
        self.store: list[Item] = []
        self.turns: int = 0
        self.phi_stability = np.zeros(self.cfg.k, dtype=np.float32)

        self.aii = AIIState()
        self._session_start_turn = 0
        self._delta_hours = 0.0

    def start_session(self) -> dict:
        res = self.memory.load(self.cfg)
        self.phi = res["phi"]
        self.store = res["store"]
        self.turns = res["turns"]
        self._delta_hours = res["delta_hours"]
        self.aii.from_dict(res.get("aii", {}))

        saved_stab = res.get("phi_stability")
        if saved_stab and len(saved_stab) == self.cfg.k:
            self.phi_stability = np.array(saved_stab, dtype=np.float32)
            if self._delta_hours > 0.1:
                for k in range(self.cfg.k):
                    hl = self.cfg.phi_half_life_hours[k] if k < len(self.cfg.phi_half_life_hours) else 24.0
                    self.phi_stability[k] *= TimeDecay.decay_factor(self._delta_hours, hl)
        else:
            self.phi_stability = np.zeros(self.cfg.k, dtype=np.float32)

        self._session_start_turn = self.turns
        return res

    def _align(self, a: np.ndarray, b: np.ndarray):
        """Wyrownaj dwa wektory do wspolnego wymiaru (min). Bezpieczne porownanie."""
        la, lb = len(a), len(b)
        if la == lb: return a, b
        m = min(la, lb)
        return a[:m], b[:m]

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray,
                    na: float = -1.0, nb: float = -1.0) -> float:
        """Cosine similarity z opcjonalnym cache norm."""
        if na < 0: na = float(np.linalg.norm(a))
        if nb < 0: nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        if abs(na - 1.0) < 0.01 and abs(nb - 1.0) < 0.01:
            return float(np.dot(a, b))
        return float(np.dot(a, b) / (na * nb))

    def _cosine_sim_item(self, item: "Item", b: np.ndarray) -> float:
        """Cosine similarity Item vs vector — używa Item.norm() cache."""
        a_, b_ = self._align(item.emb_np(), b); return self._cosine_sim(a_, b_)

    def turn(self, user_message: str, system_prompt: str = "") -> list[dict]:
        # v4.0: query bez czasu (szukamy semantycznie, nie temporalnie)
        query_emb = self.embedder.encode(user_message)  # query bez czasu — przestrzeń content-only
        self._recall(query_emb)

        skip = False
        if self.store:
            best_sim = -1.0
            best_item = None
            for i in self.store:
                a_, b_ = self._align(query_emb, i.emb_np()); sim = self._cosine_sim(a_, b_)
                if sim > best_sim:
                    best_sim = sim
                    best_item = i

            if best_sim > 0.92:
                best_item.age = max(0, best_item.age - 2)
                skip = True

        if not skip:
            self.store.append(Item(id=str(uuid.uuid4()), content=user_message[:500], embedding=query_emb.tolist(), age=0))

        self._vacuum(query_emb)
        window = self._build_window(query_emb)
        self._update_phi(window)

        for item in self.store: item.recalled = False
        self.turns += 1
        return self._build_messages(window, user_message, system_prompt)

    def after_turn(self, user_message: str, response: str) -> None:
        # FIX-3: response moze byc None (Gemini filtruje tresc)
        response = response or "[brak odpowiedzi]"
        self.aii.update(user_message + " " + response)
        # FIX-RT1: Limit długości — zapobiegamy memory inflation
        MAX_CONTENT = 500
        combined = f"User: {user_message[:MAX_CONTENT]}\nAssistant: {response[:MAX_CONTENT]}"
        event_time = time.time()
        comb_emb = self.embedder.encode(combined, timestamp=event_time)

        skip = False
        if self.store:
            best_sim = -1.0
            best_item = None
            for i in self.store:
                a_, b_ = self._align(comb_emb, i.emb_np()); sim = self._cosine_sim(a_, b_)
                if sim > best_sim:
                    best_sim = sim
                    best_item = i

            if best_sim > 0.92:
                best_item.age = max(0, best_item.age - 2)
                skip = True

        if not skip:
            weight = self.aii.get_emotion_weight()
            self.store.append(Item(id=str(uuid.uuid4()), content=combined[:1000], embedding=comb_emb.tolist(), relevance=weight))

        self._vacuum(comb_emb)
        self._update_phi(self._build_window(comb_emb))
        for it in self.store: it.age += 1

        self.memory.save(self.phi, self.store, self.turns, self.aii.to_dict(), self.phi_stability.tolist())
        if hasattr(self.embedder, 'save'):
            self.embedder.save()

    def _recall(self, query_emb: np.ndarray):
        if not self.store: return
        scores = {}
        for k in range(self.cfg.k):
            attractor = self.phi[k]
            for item in self.store:
                emb_i = item.emb_np()
                ei_a, att_a = self._align(emb_i, attractor)
                ei_q, qe_a  = self._align(emb_i, query_emb)
                sim = 0.6 * self._cosine_sim(ei_a, att_a) + 0.4 * self._cosine_sim(ei_q, qe_a)
                score = sim / (1.0 + item.age * self.cfg.recall_age_penalty)
                if id(item) not in scores or score > scores[id(item)][0]:
                    scores[id(item)] = (score, item, k)

        ranked = sorted(scores.values(), key=lambda x: -x[0])
        for _, item, k in ranked[:self.cfg.top_n_recall]:
            item.recalled = True
            self.phi_stability[k] = min(self.phi_stability[k] + 1.0, self.cfg.phi_stability_max)

    def _phi_center(self, query_emb: np.ndarray = None) -> np.ndarray:
        if query_emb is not None:
            q_dim = len(query_emb)
            sims = np.array([
                float(np.dot(query_emb, self.phi[k][:q_dim]) /
                      (np.linalg.norm(query_emb) * np.linalg.norm(self.phi[k][:q_dim]) + 1e-8))
                for k in range(len(self.phi))
            ], dtype=np.float32)
            exp_s = np.exp(sims - sims.max())
            weights = exp_s / (exp_s.sum() + 1e-8)
            center = sum(weights[k] * self.phi[k] for k in range(len(self.phi)))
        else:
            center = self.phi.mean(axis=0)
        return center / (np.linalg.norm(center) + 1e-8)

    def _vacuum(self, query_emb: np.ndarray):
        center = self._phi_center(query_emb)
        tau = self.cfg.vacuum_age_tau
        threshold = self.cfg.threshold * self.aii.get_threshold_multiplier(self.cfg.aii_adapt_range)

        session_age = self.turns - self._session_start_turn
        if session_age < self.cfg.vacuum_warmup_turns:
            threshold *= (0.5 + 0.5 * (session_age / self.cfg.vacuum_warmup_turns))

        cdim = self.cfg.dim  # content dim — porownujemy bez osi czasu
        center_c = center[:cdim] / (np.linalg.norm(center[:cdim]) + 1e-8)
        for item in self.store:
            semantic = self._cosine_sim(item.emb_content(cdim), center_c)
            item.relevance = 0.6 * semantic + 0.4 * item.relevance

        self.store = [i for i in self.store if i.age <= 1 or i.recalled or (i.relevance * math.exp(-i.age / tau)) >= threshold]

        MAX_STORE = self.cfg.n * 6
        if len(self.store) > MAX_STORE:
            self.store.sort(key=lambda i: (i.age <= 1, i.relevance * math.exp(-i.age / tau)), reverse=True)
            self.store = self.store[:MAX_STORE]

    def _build_window(self, query_emb: np.ndarray) -> list:
        center = self._phi_center(query_emb)
        protected = [i for i in self.store if i.age <= 1 or i.recalled]

        # FIX-5: Zastąpiono `i not in protected` (O(n²)) set-em id-ów → O(n).
        protected_ids = {id(i) for i in protected}
        cdim_b = self.cfg.dim
        center_b = center[:cdim_b] / (np.linalg.norm(center[:cdim_b]) + 1e-8)
        candidates = sorted(
            [i for i in self.store if id(i) not in protected_ids],
            key=lambda x: -self._cosine_sim(x.emb_content(cdim_b), center_b)
        )

        # FIX-7: Usunięto martwy kod fallback `rest`.
        # `protected + candidates` pokrywa cały store, więc `rest` zawsze był [].
        ordered = protected + candidates
        return ordered[:self.cfg.n]

    def _update_phi(self, window: list):
        if not window: return

        window_ids = {id(item) for item in window}
        active_items = [i for i in self.store if id(i) in window_ids or i.age <= 1 or i.recalled]
        if not active_items: return

        # FIX-C: emotion_weight wzmacnia ślady emocjonalne w Φ
        emotion_w = self.aii.get_emotion_weight()

        tdim = self.cfg.total_dim
        cdim = self.cfg.dim
        pattern = np.zeros(tdim, dtype=np.float32)
        for item in active_items:
            phase = math.exp(-item.age / self.cfg.vacuum_age_tau)
            weight = 2.0 if item.recalled else (1.5 if item.age <= 1 else 1.0)
            sign = 1.0 if (item.recalled or item.age <= 1) else -0.3
            emb = item.emb_np()
            # Wyrownaj do total_dim — stare itemy (bez osi czasu) padujemy zerami
            if len(emb) < tdim:
                emb = np.concatenate([emb, np.zeros(tdim - len(emb), dtype=np.float32)])
            pattern += sign * phase * weight * emotion_w * emb

        pattern /= (np.linalg.norm(pattern) + 1e-8)

        # FIX-P4: Soft update — wszystkie komponenty Φ uczą się
        # weights = softmax(sim(pattern, phi[k])) — bliższe atraktory uczą się mocniej
        sims = np.array([
            float(np.dot(pattern, self.phi[k]) /
                  (np.linalg.norm(self.phi[k]) + 1e-8))
            for k in range(self.cfg.k)
        ], dtype=np.float32)
        exp_s   = np.exp(sims - sims.max())
        weights = exp_s / (exp_s.sum() + 1e-8)

        # phi[0] = rdzeń osobowości — tarcza 10x
        weights[0] *= 0.1

        for k in range(self.cfg.k):
            lr_eff = self.cfg.lr * weights[k]
            self.phi[k] = (1 - lr_eff) * self.phi[k] + lr_eff * pattern
            self.phi[k] /= (np.linalg.norm(self.phi[k]) + 1e-8)

        self.phi_stability *= self.cfg.phi_stability_decay

        beta = self.cfg.phi_ortho_beta
        if beta > 0.0:
            phi_new = self.phi.copy()
            for i in range(len(self.phi)):
                row = self.phi[i].copy()
                for j in range(len(self.phi)):
                    if i != j:
                        row -= beta * float(np.dot(row, self.phi[j])) * self.phi[j]
                phi_new[i] = row / (np.linalg.norm(row) + 1e-8)
            self.phi = phi_new

        # FIX-A: Online decay Φ — zapobiega monotonicznym wzrostom
        # Phi nie powinno rosnąć bez końca między sesjami
        self.phi *= 0.999

        # FIX-A: Anti-drift — reiniekcja szumu gdy jeden atraktor dominuje
        # Sprawdzamy różnorodność phi_stability (std > próg = dryf)
        if np.std(self.phi_stability) > 2.0:
            weakest_idx = int(np.argmin(self.phi_stability))
            noise = np.random.randn(self.cfg.total_dim).astype(np.float32) * 0.005
            self.phi[weakest_idx] += noise
            self.phi[weakest_idx] /= (np.linalg.norm(self.phi[weakest_idx]) + 1e-8)

    def _build_messages(self, window: list, user_message: str, system_prompt: str) -> list:
        msgs = [{"role": "system", "content": system_prompt}] if system_prompt else []
        if window:
            ctx_items = [i for i in window if i.content != user_message]
            if ctx_items:
                max_chars = max(200, 9856 // max(1, len(ctx_items)))
                parts = [f"[t-{i.age}{'★' if i.recalled else ''}] {i.content[:max_chars]}" for i in ctx_items]
                msgs.append({"role": "system", "content": "Pamięć sesji:\n" + "\n---\n".join(parts)})
        msgs.append({"role": "user", "content": user_message})
        return msgs

    def stats(self) -> dict:
        # FIX-6: Guard przed wywołaniem stats() przed start_session() (phi == None).
        if self.phi is None:
            return {"turns": 0, "store": 0, "phi_norms": [], "phi_stability": [],
                    "aii": self.aii.to_dict(), "delta_hours": 0.0, "warning": "start_session() not called"}
        phi_norms = np.linalg.norm(self.phi, axis=1).tolist()
        return {
            "turns": self.turns,
            "store": len(self.store),
            "phi_norms": [round(n, 4) for n in phi_norms],
            "phi_stability": [round(s, 2) for s in self.phi_stability.tolist()],
            "aii": self.aii.to_dict(),
            "delta_hours": round(self._delta_hours, 2)
        }

    # ── v4.0: Temporalne API ─────────────────────────────────

    def recall_at(self, query: str, target_time: float, top_k: int = 5) -> list:
        """Cofanie w czasie: recall przez Phi z danego momentu."""
        hours_ago = (time.time() - target_time) / 3600.0
        # Fix-3: decay NIE jest odwracalny — phi_then to aproksymacja.
        # Cofamy przez odwrócenie znaku delty (matematyczna aproksymacja stanu).
        # Docelowo: checkpointy Φ co N tur + interpolacja (v4.1).
        phi_then  = TimeDecay.evolve_phi(
            self.phi, -hours_ago, self.cfg.phi_half_life_hours, self.cfg.phi_min_norm
        )
        q_full  = self.embedder.encode(query, timestamp=target_time)
        cdim    = self.cfg.dim
        q_c     = q_full[:cdim]  # tylko content — do porownania z emb_content

        # Centrum Phi z tamtego momentu
        norms   = np.linalg.norm(phi_then, axis=1)
        exp_n   = np.exp(norms - norms.max())
        weights = exp_n / (exp_n.sum() + 1e-8)
        center  = sum(weights[i] * phi_then[i] for i in range(len(phi_then)))
        center_c = center[:cdim] / (np.linalg.norm(center[:cdim]) + 1e-8)

        # Fix-2: rozdziel content i time similarity — nie mieszaj przestrzeni
        scored = []
        for item in self.store:
            e_c = item.emb_content(cdim)         # content bez czasu
            e_t = item.emb_np()[cdim:]            # tylko czas
            q_t = q_full[cdim:]                  # czas z query
            s_content = self._cosine_sim(e_c, q_c)
            s_time    = self._cosine_sim(e_t, q_t) if len(e_t) == len(q_t) else 0.0
            s_center  = self._cosine_sim(e_c, center_c)
            # Wagi: content dominuje, czas jest wskazówką, center = geometria Φ
            score = 0.5 * s_content + 0.2 * s_time + 0.3 * s_center
            scored.append((item, score))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def trajectory(self, topic: str, top_k: int = 10) -> list:
        """Trajektoria tematu: (timestamp, content, sim) posortowana chronologicznie."""
        q_full = self.embedder.encode(topic, timestamp=time.time())
        cdim   = self.cfg.dim
        q_c    = q_full[:cdim]  # tylko content do porownania
        scored = [(item.created_at, item.content,
                   self._cosine_sim(item.emb_content(cdim), q_c))
                  for item in self.store
                  if self._cosine_sim(item.emb_content(cdim), q_c) > 0.3]
        # Zwroc (timestamp, content_str) posortowane chronologicznie
        scored.sort(key=lambda x: x[0])
        return [(ts, content) for ts, content, sim in scored[:top_k]]

    def reset(self):
        self.memory.delete()
        self.phi = PersistentMemory._init_phi(self.cfg)
        self.phi_stability = np.zeros(self.cfg.k, dtype=np.float32)
        self.store = []
        self.turns = 0
        self._delta_hours = 0.0
        # FIX-4: Reset stanu emocjonalnego i licznika sesji — wcześniej przenosiły
        # się między sesjami, powodując błędny warmup i skażony vacuum_signal.
        self.aii = AIIState()
        self._session_start_turn = 0


# ============================================================
# SESJA INTERAKTYWNA
# ============================================================

class Session:
    DEFAULT_SYSTEM = (
        "Jesteś asystentem z pamięcią sesji. "
        "Sekcja 'Pamięć sesji' zawiera fakty z tej rozmowy — traktuj je jako PEWNĄ WIEDZĘ. "
        "ZASADY BEZWZGLĘDNE:\n"
        "1. Jeśli fakt jest w pamięci sesji — podaj go wprost. Nie mów 'nie wiem' ani 'nie mam dostępu'.\n"
        "2. Data, godzina, imię, nazwa projektu — jeśli użytkownik podał, PAMIĘTASZ to.\n"
        "3. Twoja wiedza z treningu jest DRUGORZĘDNA wobec faktów z pamięci sesji.\n"
        "4. Jeśli użytkownik koryguje informację — zapamiętaj korektę jako nowy fakt.\n"
        "Odpowiadaj w języku rozmówcy. Domyślnie po polsku."
    )

    def __init__(self, memory_path="holon_memory.json", cfg=None, system=None, model="gemini-2.5-flash"):
        self.model = model
        self.system = system or self.DEFAULT_SYSTEM
        self._gemini_key = os.environ.get("GEMINI_API_KEY", "")

        cfg_ = cfg or Config()
        emb = Embedder(dim=cfg_.dim, dict_path=memory_path.replace(".json", "_kurz.json"))
        self.holomem = HoloMem(emb, cfg_, memory_path)
        self._gemini_client = None

        if self._gemini_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self._gemini_key)
            except Exception as e:
                print(f"[Session] Błąd ładowania Google GenAI: {e}")

    def start(self) -> str:
        res = self.holomem.start_session()
        s = self.holomem.stats()
        print(f"\n[Holon] tur={s['turns']} store={s['store']} delta={s['delta_hours']}h aii={s['aii']}")
        return res.get("wake", "")

    def chat(self, user_input: str) -> str:
        messages = self.holomem.turn(user_input, self.system)
        response = self._call_llm(messages)
        self.holomem.after_turn(user_input, response)

        s = self.holomem.stats()
        print(f"  [store={s['store']} stab={s['phi_stability']} aii={s['aii']['emotion']}(focus:{s['aii']['focus']})]", flush=True)
        return response

    def _call_llm(self, messages: list) -> str:
        if not self._gemini_client:
            return "[Tryb Mock] Wypowiedź odebrana. Ustaw klucz API, by rozmawiać z modelem."

        try:
            from google.genai import types
            system_parts, contents = [], []

            for msg in messages:
                if msg["role"] == "system":
                    system_parts.append(msg["content"])
                elif msg["role"] == "user":
                    contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
                elif msg["role"] == "assistant":
                    contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))

            sys_inst = "\n".join(system_parts) if system_parts else None
            response = self._gemini_client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=sys_inst,
                    max_output_tokens=1024,
                    temperature=0.7,
                )
            )
            # FIX-3: response.text może być None gdy Gemini filtruje treść.
            # None przekazane do after_turn → embedder.encode(None) → crash.
            return response.text or "[Brak odpowiedzi modelu — treść zfiltrowana]"
        except Exception as e:
            return f"[Błąd LLM: {e}]"

    def stats(self) -> dict:
        return self.holomem.stats()

    def reset(self):
        self.holomem.reset()
        print("[HoloMem] Pamięć wyczyszczona.")


# ============================================================
# URUCHOMIENIE PROGRAMU
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Holon v4.0 — Holograficzna Pamięć w Czasoprzestrzeni")
    print("=" * 60)

    session = Session(memory_path="holon_memory.json", model="gemini-2.5-flash")
    wake_msg = session.start()

    if wake_msg:
        print(f"\n{wake_msg}\n")

    print("Komendy: 'quit' (wyjście), 'stats' (statystyki), 'reset' (czyszczenie pamięci).")
    print("-" * 60)

    while True:
        try:
            user = input("\nTy: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDo widzenia.")
            break

        if not user: continue
        if user.lower() == "quit": break
        if user.lower() == "stats":
            print(f"\n[Stats] {session.stats()}")
            continue
        if user.lower() == "reset":
            session.reset()
            continue

        print("\nAsystent: ", end="", flush=True)
        resp = session.chat(user)
        print(resp)
