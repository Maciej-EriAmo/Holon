"""
Holon v2.5 — Holograficzna Pamiec Kontekstu z Poczuciem Czasu
===========================================================
Autor koncepcji: Maciej Mazur
Implementacja: Claude (Anthropic)

Architektura:
  T-class (okno)  — aktualny kontekst, kod, ostatnie wymiany
  P-class (Φ)     — geometria percepcji poza oknem, trwa między sesjami
  Vacuum          — selekcja relevancji, usuwa szum
  Recall          — reaktywacja historycznych wzorców
  TimeDecay       — Φ ewoluuje gdy system jest zamknięty

Wymagania:
  pip install numpy google-genai
"""

import os
import json
import math
import time
import uuid
import numpy as np
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict


# ============================================================
# KONFIGURACJA
# ============================================================

@dataclass
class Config:
    k:            int   = 4      # liczba wektorów percepcji w Φ
    n:            int   = 5      # rozmiar okna kontekstu
    threshold:    float = 0.35   # próg vacuum
    lr:           float = 0.01   # learning rate Φ
    alpha:        float = 0.05   # siła hologramu w attention
    top_n_recall: int   = 2      # ile itemów przywołuje recall
    dim:          int   = 768    # wymiar embeddingów (gemini-embedding-001)

    # Tempo zaniku wierszy Φ [godziny do utraty 50% siły]
    # wiersz 0: wzorzec techniczny/kod    → wolny zanik
    # wiersz 1: wzorzec projektu          → średni zanik
    # wiersz 2: wzorzec konwersacyjny     → szybki zanik
    # wiersz 3: wzorzec bieżący           → najszybszy zanik
    phi_half_life_hours: list = field(
        default_factory=lambda: [48.0, 24.0, 8.0, 4.0]
    )

    # Store items znikają po N godzinach
    store_decay_hours: float = 72.0

    # Minimalna norma wiersza Φ po zaniku
    phi_min_norm: float = 0.1

    # Ortogonalizacja Φ — wymusza specjalizację wierszy
    # beta=0.0 wyłącza, beta=0.05 zalecane
    phi_ortho_beta: float = 0.05

    # Vacuum: tau zaniku przez wiek [w turnach]
    # effective_score = relevance * exp(-age / tau)
    vacuum_age_tau: float = 50.0

    # Recall: penalizacja wieku
    # score = sim * (1 / (1 + age * penalty))
    recall_age_penalty: float = 0.02


# ============================================================
# ITEM — jednostka pamięci
# ============================================================

@dataclass
class Item:
    id:        str
    content:   str
    embedding: list   # FloatArray jako lista do serializacji JSON
    age:       int    = 0
    recalled:  bool   = False
    relevance: float  = 1.0
    created_at: float = field(default_factory=time.time)

    def emb_np(self) -> np.ndarray:
        return np.array(self.embedding, dtype=np.float32)


# ============================================================
# EMBEDDER
# ============================================================

class Embedder:
    """Prawdziwy embedder przez Gemini API lub hash fallback."""

    def __init__(self, api_key: Optional[str] = None, dim: int = 768):
        self.dim = dim
        self._client = None
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if key:
            from google import genai
            self._client = genai.Client(api_key=key)
            print(f"Embedder: gemini-embedding-001 (dim={dim})")
        else:
            print(f"Embedder: hash fallback (dim={dim}) — brak GEMINI_API_KEY")

    def encode(self, text: str) -> np.ndarray:
        if self._client:
            try:
                from google.genai import types
                result = self._client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="SEMANTIC_SIMILARITY",
                        output_dimensionality=self.dim,
                    ),
                )
                vec = np.array(result.embeddings[0].values, dtype=np.float32)
                return self._normalize(vec)
            except Exception as e:
                print(f"[Embedder] Błąd API: {e} — fallback")

        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            h = abs(hash(word)) % self.dim
            vec[h] += 1.0
        return self._normalize(vec)

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)


# ============================================================
# TIME DECAY — poczucie czasu
# ============================================================

class TimeDecay:
    """
    Φ ewoluuje gdy system jest zamknięty.

    Wzór: siła = siła_oryginalna * exp(-ln2 * dt / half_life)
    Po czasie half_life siła spada o 50%.

    Każdy wiersz Φ ma inny half_life — wzorce techniczne trwają dłużej.
    """

    @staticmethod
    def decay_factor(delta_hours: float, half_life_hours: float) -> float:
        return math.exp(-0.693 * delta_hours / (half_life_hours + 1e-8))

    @staticmethod
    def evolve_phi(phi: np.ndarray, delta_hours: float,
                   half_lives: list, min_norm: float) -> np.ndarray:
        if delta_hours < 0.1:
            return phi

        evolved = phi.copy()
        for k in range(len(phi)):
            half_life   = half_lives[k] if k < len(half_lives) else 24.0
            decay       = TimeDecay.decay_factor(delta_hours, half_life)
            row         = phi[k] * decay
            norm        = np.linalg.norm(row)

            if norm < min_norm:
                # Reinicjalizuj z śladową pamięcią
                noise      = np.random.randn(len(phi[k])).astype(np.float32) * 0.01
                row        = phi[k] * min_norm + noise

            norm = np.linalg.norm(row)
            evolved[k] = row / (norm + 1e-8)

        return evolved

    @staticmethod
    def wake_message(delta_hours: float, turns: int, store_size: int) -> str:
        """Komunikat co się stało gdy system był zamknięty."""
        if delta_hours < 0.1:
            return ""

        if delta_hours < 1:
            time_desc = f"{int(delta_hours * 60)} minut"
        elif delta_hours < 24:
            time_desc = f"{delta_hours:.1f} godzin"
        elif delta_hours < 168:
            time_desc = f"{delta_hours / 24:.1f} dni"
        else:
            time_desc = f"{delta_hours / 168:.1f} tygodni"

        if delta_hours < 2:
            mem_state = "Pamięć świeża."
        elif delta_hours < 12:
            mem_state = "Wzorce konwersacyjne lekko przybladły."
        elif delta_hours < 48:
            mem_state = "Wzorce bieżące zanikły. Kontekst techniczny zachowany."
        elif delta_hours < 168:
            mem_state = "Większość wzorców zanikła. Pamiętam styl i projekt."
        else:
            mem_state = "Długa nieobecność. Zachowałem tylko rdzeń wzorców."

        return (
            f"[Minęło {time_desc} od ostatniej sesji. "
            f"Było {turns} turnów, {store_size} wzorców w pamięci. "
            f"{mem_state}]"
        )


# ============================================================
# PAMIĘĆ DŁUGOTRWAŁA — zapis/odczyt
# ============================================================

class PersistentMemory:
    """Zapisuje i wczytuje Φ + store + timestamp."""

    def __init__(self, path: str = "holon_memory.json"):
        self.path = Path(path)

    def save(self, phi: np.ndarray, store: list, turns: int):
        data = {
            "timestamp": time.time(),
            "turns":     turns,
            "phi":       phi.tolist(),
            "store": [
                {
                    "id":        item.id,
                    "content":   item.content,
                    "embedding": item.embedding,
                    "age":       item.age,
                    "relevance": item.relevance,
                    "created_at": item.created_at,
                }
                for item in store
                if item.age >= 1
            ]
        }
        # Atomowy zapis: .tmp -> rename
        # Chroni przed uszkodzeniem przy crash/brak baterii
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
        """
        Zwraca słownik:
          phi, store, turns, delta_hours, loaded
        """
        # Usun sieroty .tmp z poprzednich crashy
        tmp = self.path.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()

        if not self.path.exists():
            return {
                "phi":         self._init_phi(cfg),
                "store":       [],
                "turns":       0,
                "delta_hours": 0.0,
                "loaded":      False,
            }

        try:
            data        = json.loads(self.path.read_text())
            saved_at    = data["timestamp"]
            delta_hours = (time.time() - saved_at) / 3600.0
            turns       = data["turns"]

            # Wczytaj Φ i przeewoluuj w czasie
            phi_raw = np.array(data["phi"], dtype=np.float32)
            phi     = TimeDecay.evolve_phi(
                phi_raw, delta_hours,
                cfg.phi_half_life_hours, cfg.phi_min_norm
            )

            # Wczytaj store z filtrem wiekowym
            max_age = cfg.store_decay_hours * 4  # heurystyka
            store   = []
            for obj in data.get("store", []):
                # Każde pełne turn = 1 age increment (po FIX 3)
                # delta_hours * 24 / 6 ≈ ~4 turns/godzinę (heurystyka na podstawie avg session)
                turns_per_hour = 4
                age_now = obj["age"] + int(delta_hours * turns_per_hour)
                if age_now > max_age:
                    continue
                store.append(Item(
                    id         = obj["id"],
                    content    = obj["content"],
                    embedding  = obj["embedding"],
                    age        = age_now,
                    recalled   = False,
                    relevance  = obj["relevance"],
                    created_at = obj.get("created_at", time.time()),
                ))

            return {
                "phi":         phi,
                "store":       store,
                "turns":       turns,
                "delta_hours": delta_hours,
                "loaded":      True,
            }

        except Exception as e:
            print(f"[Memory] Błąd wczytania: {e}")
            return {
                "phi":         self._init_phi(cfg),
                "store":       [],
                "turns":       0,
                "delta_hours": 0.0,
                "loaded":      False,
            }

    def delete(self):
        if self.path.exists():
            self.path.unlink()

    def exists(self) -> bool:
        return self.path.exists()

    @staticmethod
    def _init_phi(cfg: Config) -> np.ndarray:
        phi = np.random.randn(cfg.k, cfg.dim).astype(np.float32) * 0.01
        norms = np.linalg.norm(phi, axis=1, keepdims=True)
        return phi / (norms + 1e-8)


# ============================================================
# HOLOMEM — rdzeń systemu
# ============================================================

class HoloMem:
    """
    Główna klasa. Łączy:
      - Φ (geometria percepcji)
      - vacuum (selekcja relevancji)
      - recall (reaktywacja wzorców)
      - time decay (poczucie czasu)
      - persistent memory (zapis między sesjami)
    """

    def __init__(
        self,
        embedder:    Embedder,
        cfg:         Config          = None,
        memory_path: str             = "holon_memory.json",
    ):
        self.embedder = embedder
        self.cfg      = cfg or Config(dim=embedder.dim)
        self.memory   = PersistentMemory(memory_path)

        self.phi:   np.ndarray = None
        self.store: list       = []
        self.turns: int        = 0

        self._delta_hours = 0.0
        self._loaded      = False

    # -----------------------------------------------------------------------
    # START SESJI
    # -----------------------------------------------------------------------

    def start_session(self) -> str:
        """Wczytaj pamięć długotrwałą. Zwraca komunikat powitalny."""
        result = self.memory.load(self.cfg)

        self.phi          = result["phi"]
        self.store        = result["store"]
        self.turns        = result["turns"]
        self._delta_hours = result["delta_hours"]
        self._loaded      = result["loaded"]

        msg = TimeDecay.wake_message(
            self._delta_hours, self.turns, len(self.store)
        )
        return msg

    # -----------------------------------------------------------------------
    # GŁÓWNY CYKL TURNU
    # -----------------------------------------------------------------------

    def turn(self, user_message: str, system_prompt: str = "") -> list:
        """
        Jeden turn. Zwraca listę messages gotową do wysłania do LLM.

        Kolejność:
          1. Recall (Φ → T)
          2. Dodaj nowy item
          3. Vacuum (T → T[t+1])
          4. Zbuduj okno z Φ[t]
          5. Aktualizuj Φ[t+1]
          6. Wiek++
        """
        query_emb = self.embedder.encode(user_message)

        # 1. Recall
        self._recall(query_emb)

        # 2. Nowy item
        # FIX: użyj query_emb zamiast liczyć drugi raz (było podwójne embedowanie)
        new_emb = query_emb
        # Deduplikacja — nie dodawaj jeśli bardzo podobny item już istnieje
        skip = False
        if self.store:
            max_sim = max(
                self._cosine_sim(new_emb, np.array(i.embedding, dtype=np.float32))
                for i in self.store
            )
            if max_sim > 0.95:
                for i in self.store:
                    if self._cosine_sim(new_emb, np.array(i.embedding, dtype=np.float32)) > 0.95:
                        i.age = max(0, i.age - 2)
                        break
                skip = True

        if not skip:
            new_item = Item(
                id        = str(uuid.uuid4()),
                content   = user_message,
                embedding = new_emb.tolist(),
                age       = 0,
            )
            self.store.append(new_item)

        # 3. Vacuum
        self._vacuum(query_emb)

        # 4. Okno z Φ[t]
        window = self._build_window(query_emb)

        # 5. Aktualizuj Φ
        self._update_phi(window)

        # 6. Reset recalled (age++ tylko w after_turn — jeden przyrost na pełną wymianę)
        for item in self.store:
            item.recalled = False

        self.turns += 1

        # Zbuduj messages
        return self._build_messages(window, user_message, system_prompt, query_emb)

    def after_turn(self, user_message: str, response: str):
        """
        Wywołaj po otrzymaniu odpowiedzi od LLM.
        Zapisuje odpowiedź do store, ocenia jakość, aktualizuje pamięć.
        """
        # Feedback signal — ocena jakości odpowiedzi
        # Wysoka spójność emb(response) z emb(query) → dobra odpowiedź
        query_emb    = self.embedder.encode(user_message)
        response_emb = self.embedder.encode(response)
        coherence    = self._cosine_sim(query_emb, response_emb)

        # Sygnały niskiej jakości tłumią wagę tej wymiany
        # Wyklucz własne komunikaty błędu systemu (zaczynają się od "[Błąd")
        is_system_error = response.startswith("[Błąd")
        low_quality = not is_system_error and any(
            phrase in response.lower() for phrase in [
                "nie wiem", "przepraszam", "nie jestem pewien",
                "i don't not know", "i'm not sure"
            ]
        )
        quality_weight = 0.5 if low_quality else max(0.3, coherence)

        combined = f"User: {user_message}\nAssistant: {response}"
        combined_emb = self.embedder.encode(combined)

        # Deduplikacja przed dodaniem
        skip = False
        if self.store:
            max_sim = max(
                self._cosine_sim(combined_emb, np.array(i.embedding, dtype=np.float32))
                for i in self.store
            )
            if max_sim > 0.95:
                for i in self.store:
                    if self._cosine_sim(combined_emb, np.array(i.embedding, dtype=np.float32)) > 0.95:
                        i.age = max(0, i.age - 2)
                        break
                skip = True

        if not skip:
            item = Item(
                id        = str(uuid.uuid4()),
                content   = combined,
                embedding = combined_emb.tolist(),
                age       = 0,
                relevance = quality_weight,
            )
            self.store.append(item)

        query_emb_for_window = self.embedder.encode(user_message)
        self._vacuum(query_emb_for_window)
        window = self._build_window(query_emb_for_window)
        self._update_phi(window)
        for it in self.store:
            it.age += 1

        # Zapisz pamięć
        self.memory.save(self.phi, self.store, self.turns)

    # -----------------------------------------------------------------------
    # HOLOMEM LOGIC
    # -----------------------------------------------------------------------

    def _phi_center(self, query_emb: np.ndarray = None) -> np.ndarray:
        if query_emb is not None:
            # Dynamiczne centrum — ważona suma wierszy Φ względem zapytania
            # weights = softmax(sim(query, phi[k]))
            sims    = np.array([float(np.dot(query_emb, self.phi[k]) /
                       (np.linalg.norm(query_emb) * np.linalg.norm(self.phi[k]) + 1e-8))
                       for k in range(len(self.phi))], dtype=np.float32)
            exp_s   = np.exp(sims - sims.max())
            weights = exp_s / (exp_s.sum() + 1e-8)
            center  = sum(weights[k] * self.phi[k] for k in range(len(self.phi)))
        else:
            center = self.phi.mean(axis=0)
        norm = np.linalg.norm(center)
        return center / (norm + 1e-8)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _recall(self, query_emb: np.ndarray):
        if not self.store:
            return

        # Multi-attractor recall — każdy wiersz Φ to osobny atraktor
        # Zbieramy kandydatów ze wszystkich K attraktorów
        penalty  = self.cfg.recall_age_penalty
        scores   = {}  # item_id → max score

        for k in range(self.cfg.k):
            attractor = self.phi[k]
            for item in self.store:
                # FIX: świeże itemy też kandydują — ale z bonusem
                # (age=0/1 dostaną wyższy score przez penalty=0)
                emb_i      = item.emb_np()
                sim_attr   = self._cosine_sim(emb_i, attractor)
                sim_query  = self._cosine_sim(emb_i, query_emb)
                # Połączenie: atraktor + query — nie gubisz różnorodności Φ
                combined   = 0.6 * sim_attr + 0.4 * sim_query
                score      = combined / (1.0 + item.age * penalty)
                if id(item) not in scores or score > scores[id(item)][0]:
                    scores[id(item)] = (score, item)

        # Weź top_n_recall z wszystkich atraktorów
        ranked = sorted(scores.values(), key=lambda x: -x[0])
        for score, item in ranked[:self.cfg.top_n_recall]:
            item.recalled = True

    def _vacuum(self, query_emb: np.ndarray = None):
        center = self._phi_center(query_emb)
        tau    = self.cfg.vacuum_age_tau

        for item in self.store:
            semantic = s
