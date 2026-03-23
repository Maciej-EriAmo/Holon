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
  pip install numpy
  (opcjonalnie google-genai do chatu przez Gemini)
"""

import os
import json
import math
import time
import uuid
import numpy as np
from typing import Optional, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from kurz import KuRz as _KuRz


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
    dim:          int   = 256    # wymiar embeddingów (KuRz offline)

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
    """
    KuRz — cichy offline embedder z co-occurrence learning.
    Uczy sie semantyki z kazdego encode() bez zadnego API.
    Phi tworzy geometrie percepcji, KuRz jest sorterem slow.

    Parametry:
        dim         — przestrzen embeddingu (default 256, mniejsze = szybsze)
        dict_path   — persystencja slownika miedzy sesjami
    """

    def __init__(self, api_key: Optional[str] = None, dim: int = 256,
                 dict_path: Optional[str] = None):
        # api_key ignorowany — KuRz nie uzywa zewnetrznych API
        self.dim = dim
        self._kurz = _KuRz(dim=dim, dict_path=dict_path)
        print(f"Embedder: KuRz offline (dim={dim}, vocab={self._kurz.vocab_size})")

    def encode(self, text: str) -> np.ndarray:
        return self._kurz.encode(text)

    def save(self, path: Optional[str] = None) -> None:
        """Zapisz slownik KuRz (co-occurrence + word→axis)."""
        if path:
            self._kurz.save_dict(path)
        elif self._kurz.dict_path:
            self._kurz.save_dict()

    @property
    def vocab_size(self) -> int:
        return self._kurz.vocab_size

    @property
    def calls(self) -> int:
        return self._kurz.calls

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)



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

    def start_session(self) -> dict:
        """
        Wczytaj pamiec dlugoterwala.
        Zwraca dict: loaded, delta_hours, wake (komunikat powitalny).
        """
        result = self.memory.load(self.cfg)

        self.phi          = result["phi"]
        self.store        = result["store"]
        self.turns        = result["turns"]
        self._delta_hours = result["delta_hours"]
        self._loaded      = result["loaded"]

        wake = TimeDecay.wake_message(
            self._delta_hours, self.turns, len(self.store)
        )
        return {
            "loaded":      self._loaded,
            "delta_hours": self._delta_hours,
            "turns":       self.turns,
            "store_size":  len(self.store),
            "wake":        wake,
        }

    # -----------------------------------------------------------------------
    # GŁÓWNY CYKL TURNU
    # -----------------------------------------------------------------------

    def turn(self, user_message: str, system_prompt: str = "") -> list[dict]:
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

    def after_turn(self, user_message: str, response: str) -> None:
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

        # Zapisz pamięć Phi + KuRz
        self.memory.save(self.phi, self.store, self.turns)
        if hasattr(self.embedder, 'save'):
            self.embedder.save()

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
            semantic = self._cosine_sim(item.emb_np(), center)
            # Łączymy semantykę z quality_weight — nie nadpisujemy feedbacku
            # Blend: semantyka + feedback jakości z after_turn
            # Dla nowych itemów (age<=1) relevance=1.0 → blend zachowuje jakość
            item.relevance = 0.6 * semantic + 0.4 * item.relevance

        def _score(item):
            # Degradacja relevance przez wiek — stare losowo podobne znikają
            time_weight = math.exp(-item.age / (tau + 1e-8))
            return item.relevance * time_weight

        self.store = [
            item for item in self.store
            if item.age <= 1
            or item.recalled
            or _score(item) >= self.cfg.threshold
        ]
        # Hard cap — store nie może rosnąć bez kontroli
        MAX_STORE = self.cfg.n * 6  # max 6x rozmiar okna
        if len(self.store) > MAX_STORE:
            # Zachowaj najnowsze i najważniejsze
            center = self._phi_center()
            self.store.sort(
                key=lambda i: (i.age <= 1, _score(i)),
                reverse=True
            )
            self.store = self.store[:MAX_STORE]

    def _build_window(self, query_emb: np.ndarray = None) -> list:
        center = self._phi_center(query_emb)
        protected  = [i for i in self.store if i.age <= 1 or i.recalled]
        candidates = sorted(
            [i for i in self.store if i.age > 1 and not i.recalled],
            key=lambda x: -self._cosine_sim(x.emb_np(), center)
        )
        ordered  = protected + candidates
        selected = ordered[:self.cfg.n]

        # Edge case: uzupełnij do n (id() zamiast __eq__ — efektywne)
        if len(selected) < self.cfg.n:
            sel_ids = {id(i) for i in selected}
            rest    = [i for i in self.store if id(i) not in sel_ids]
            selected += rest[:self.cfg.n - len(selected)]

        return selected

    def _update_phi(self, window: list):
        if not window:
            return

        window_ids = {id(item) for item in window}
        active_embs = np.array([
            item.emb_np() for item in self.store
            if item.age <= 1 or item.recalled or id(item) in window_ids
        ], dtype=np.float32)

        if len(active_embs) == 0:
            return

        # Interferencja: konstruktywna (recalled/nowe) + destruktywna (stare tło)
        # phase = exp(-age/tau) — siła fazy maleje z wiekiem
        tau     = self.cfg.vacuum_age_tau
        pattern = np.zeros(active_embs.shape[1], dtype=np.float32)
        active_items = [
            item for item in self.store
            if id(item) in window_ids or item.age <= 1 or item.recalled
        ]
        for item, emb in zip(active_items, active_embs):
            phase = math.exp(-item.age / (tau + 1e-8))
            if item.recalled:
                sign   = 1.0    # konstruktywna — reaktywowane wzorce wzmacniają
                weight = 2.0
            elif item.age <= 1:
                sign   = 1.0    # konstruktywne — nowe informacje
                weight = 1.5
            else:
                sign   = -0.3   # lekko destruktywne — stare tło tłumione
                weight = 1.0
            pattern += sign * phase * weight * emb

        # Soft normalization — zachowuje kierunek bez brutalnej saturacji tanh
        # pattern / (1 + ||pattern||) zamiast tanh — nie ścina informacji
        p_norm  = np.linalg.norm(pattern)
        pattern = pattern / (1.0 + p_norm)
        norm    = np.linalg.norm(pattern)
        pattern = pattern / (norm + 1e-8)

        norms   = np.linalg.norm(self.phi, axis=1)
        # phi[0] = rdzeń osobowości — aktualizuje się 10x wolniej
        norms_adj       = norms.copy()
        norms_adj[0]   *= 10.0  # sztuczne zawyżenie normy → rzadziej wybierany
        weakest = int(np.argmin(norms_adj))
        lr_eff  = self.cfg.lr * (0.1 if weakest == 0 else 1.0)
        updated = (1 - lr_eff) * self.phi[weakest] + lr_eff * pattern
        norm    = np.linalg.norm(updated)
        self.phi[weakest] = updated / (norm + 1e-8)

        # Ortogonalizacja Φ — wymusza specjalizację wierszy
        # Każdy wiersz odpycha się od pozostałych
        beta = self.cfg.phi_ortho_beta
        if beta > 0.0:
            phi_new = self.phi.copy()
            for i in range(len(self.phi)):
                row = self.phi[i].copy()
                for j in range(len(self.phi)):
                    if i != j:
                        # Odejmij składową równoległą do phi[j]
                        row -= beta * float(np.dot(row, self.phi[j])) * self.phi[j]
                norm = np.linalg.norm(row)
                phi_new[i] = row / (norm + 1e-8)
            self.phi = phi_new

    def _build_messages(self, window: list, user_message: str,
                        system_prompt: str,
                        query_emb: np.ndarray = None) -> list:
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Φ używane TYLKO do selekcji — nie wstrzykujemy nic do promptu
        # (phi_hint usunięty: 32 floaty = ~200 tokenów kosztu bez wartości dla LLM)

        if window:
            mem_parts = [
                f"[t-{item.age}{'*' if item.recalled else ''}] {item.content[:100]}"
                for item in window
                if item.content != user_message
            ]
            if mem_parts:
                # Token budget: model_limit=4096, rezerwujemy 512 na odpowiedź,
                # 256 na system prompt, 512 na query → max 2816 tokenów na pamięć
                # Heurystyka polska: 1 token ≈ 3.5 znaku (polskie słowa dłuższe)
                TOKEN_BUDGET  = 2816
                CHARS_BUDGET  = int(TOKEN_BUDGET * 3.5)  # ~9856 znaków
                # Dodatkowo hard cap per item — nie pozwól jednemu itemowi zająć wszystko
                MAX_ITEM_CHARS = 300

                trimmed = []
                used = 0
                for part in mem_parts:
                    part = part[:MAX_ITEM_CHARS]
                    if used + len(part) > CHARS_BUDGET:
                        break
                    trimmed.append(part)
                    used += len(part)

                if trimmed:
                    messages.append({
                        "role":    "system",
                        "content": "Kontekst sesji:\n" + "\n".join(trimmed),
                    })

        messages.append({"role": "user", "content": user_message})
        return messages

    # -----------------------------------------------------------------------
    # STATYSTYKI
    # -----------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        center = self._phi_center()
        avg_rel = float(np.mean([
            self._cosine_sim(i.emb_np(), center) for i in self.store
        ])) if self.store else 0.0

        phi_norms = np.linalg.norm(self.phi, axis=1).tolist()

        return {
            "turns":       self.turns,
            "store_size":  len(self.store),
            "avg_rel":     round(avg_rel, 4),
            "phi_norms":   [round(n, 4) for n in phi_norms],
            "delta_hours": round(self._delta_hours, 2),
            "memory_file": str(self.memory.path),
            "memory_saved": self.memory.exists(),
        }

    def reset(self):
        self.memory.delete()
        self.phi   = PersistentMemory._init_phi(self.cfg)
        self.store = []
        self.turns = 0
        self._delta_hours = 0.0


# ============================================================
# SESJA INTERAKTYWNA
# ============================================================

class Session:
    """
    Wrapper łączący HoloMem z wywołaniem LLM.
    Działa z dowolnym dostawcą OpenAI-compatible API.
    """

    DEFAULT_SYSTEM = (
        "Jesteś pomocnym asystentem. "
        "Odpowiadaj w języku rozmówcy. Domyślnie po polsku."
    )

    def __init__(
        self,
        memory_path:  str    = "holon_memory.json",
        cfg:          Config = None,
        system:       str    = None,
        model:        str    = "gemini-2.5-flash",
    ):
        self.model  = model
        self.system = system or self.DEFAULT_SYSTEM
        self._gemini_key = os.environ.get("GEMINI_API_KEY", "")

        emb = Embedder(
            api_key=self._gemini_key,
            dim=(cfg or Config()).dim
        )
        self.holomem = HoloMem(emb, cfg or Config(), memory_path)
        self._gemini_client = None
        if self._gemini_key:
            from google import genai
            self._gemini_client = genai.Client(api_key=self._gemini_key)

    def start(self) -> str:
        result = self.holomem.start_session()
        s      = self.holomem.stats()
        print(f"\n[Holon] turns={s['turns']} store={s['store_size']} "
              f"rel={s['avg_rel']} delta={s['delta_hours']}h")
        return result.get("wake", "")

    def chat(self, user_input: str) -> str:
        messages = self.holomem.turn(user_input, self.system)
        response = self._call_llm(messages)
        self.holomem.after_turn(user_input, response)
        self._embedder.save()
        s = self.holomem.stats()
        print(f"  [store={s['store_size']} rel={s['avg_rel']}]", flush=True)
        return response

    def _call_llm(self, messages: list) -> str:
        if not self._gemini_client:
            return "[Błąd: brak GEMINI_API_KEY]"
        try:
            from google.genai import types

            # Konwertuj messages na format Gemini
            # system → system_instruction, reszta → contents
            system_parts = []
            contents     = []

            for msg in messages:
                role    = msg["role"]
                content = msg["content"]
                if role == "system":
                    system_parts.append(content)
                elif role == "user":
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part(text=content)]
                    ))
                elif role == "assistant":
                    contents.append(types.Content(
                        role="model",
                        parts=[types.Part(text=content)]
                    ))

            system_instruction = "\n".join(system_parts) if system_parts else None

            response = self._gemini_client.models.generate_content(
                model   = self.model,
                contents = contents,
                config  = types.GenerateContentConfig(
                    system_instruction = system_instruction,
                    max_output_tokens  = 1024,
                    temperature        = 0.7,
                )
            )
            return response.text
        except Exception as e:
            return f"[Błąd LLM: {e}]"

    def stats(self) -> dict[str, Any]:
        return self.holomem.stats()

    def reset(self):
        self.holomem.reset()
        print("[HoloMem] Pamięć wyczyszczona.")


# ============================================================
# URUCHOMIENIE
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Holon v2.5 — Holograficzna Pamiec Kontekstu z Poczuciem Czasu")
    print("=" * 60)

    session = Session(
        memory_path = "holon_memory.json",
        model       = "gemini-2.5-flash",
    )

    wake = session.start()
    if wake:
        print(f"\n{wake}\n")

    print("Wpisz 'quit' aby wyjść, 'stats' dla statystyk, 'reset' aby wyczyścić.")
    print("-" * 60)

    while True:
        try:
            user = input("\nTy: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDo widzenia.")
            break

        if not user:
            continue
        if user.lower() == "quit":
            break
        if user.lower() == "stats":
            s = session.stats()
            print(f"\n[Stats] turns={s['turns']} store={s['store_size']} "
                  f"rel={s['avg_rel']} phi_norms={s['phi_norms']} "
                  f"delta={s['delta_hours']}h saved={s['memory_saved']}")
            continue
        if user.lower() == "reset":
            session.reset()
            continue

        print("\nAsystent: ", end="", flush=True)
        response = session.chat(user)
        print(response)
