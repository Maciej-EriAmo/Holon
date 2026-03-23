"""
KuRz — Dynamic Offline Embedder with Co-occurrence Learning
Port z Holona (JavaScript) + co-occurrence axis merging.

Jak działa co-occurrence:
  Jeśli słowa A i B często pojawiają się razem w jednym tekście,
  ich osie są przyciągane do siebie przez weighted average.
  Po n_merge_threshold wspólnych wystąpień → A i B dostają tę samą oś.
  Efekt: "attention" + "uwaga" + "percepcja" zlewają się w jeden klaster.

Zero zewnętrznych zależności (tylko numpy).
"""

import re
import json
import numpy as np
from pathlib import Path
from typing  import Optional
from collections import defaultdict


STOP_WORDS = frozenset([
    # Polish
    'w', 'i', 'z', 'na', 'do', 'że', 'o', 'to', 'jak', 'jest', 'nie', 'za',
    'po', 'od', 'co', 'dla', 'ten', 'tak', 'ale', 'czy', 'jako', 'przez',
    'tylko', 'aby', 'tej', 'tego', 'temu', 'które', 'który', 'która',
    # English
    'the', 'of', 'and', 'that', 'this', 'with', 'you', 'not', 'or', 'be',
    'are', 'from', 'at', 'as', 'your', 'all', 'have', 'for', 'its', 'was',
    'but', 'they', 'their', 'been', 'has',
])


class KuRz:
    """
    Self-learning offline embedder z co-occurrence axis merging.

    Parametry:
        dim              — liczba osi (przestrzeń embeddingów)
        dict_path        — ścieżka do pliku JSON z persystencją
        merge_threshold  — ile wspólnych wystąpień żeby scalić osie (default 3)
        merge_alpha      — siła przyciągania przy scalaniu (0..1, default 0.5)
    """

    def __init__(
        self,
        dim: int = 128,
        dict_path: Optional[str] = None,
        merge_threshold: int = 3,
        merge_alpha: float = 0.5,
    ):
        self.dim             = dim
        self.dict_path       = Path(dict_path) if dict_path else None
        self.merge_threshold = merge_threshold
        self.merge_alpha     = merge_alpha

        self._word_axis: dict[str, int]        = {}
        self._next_free: int                   = 0
        self._cooc: dict[tuple, int]           = defaultdict(int)  # (w1,w2) → count
        self.calls: int                        = 0
        self._merges: int                      = 0  # statystyki

        if self.dict_path and self.dict_path.exists():
            self.load_dict()

    # ── Tokenizacja ────────────────────────────────────────────

    def tokenize(self, text: str) -> list[str]:
        words = re.findall(r'\b[^\W\d_]{3,}\b', text.lower())
        return [w for w in words if w not in STOP_WORDS]

    # ── Przypisanie osi ────────────────────────────────────────

    def _get_axis(self, word: str) -> int:
        if word in self._word_axis:
            return self._word_axis[word]

        if self._next_free < self.dim:
            axis = self._next_free
            self._next_free += 1
        else:
            # Deterministyczny hash — identyczny między sesjami
            h = 5381
            for ch in word.encode('utf-8'):
                h = ((h << 5) + h + ch) & 0xFFFFFFFF
            axis = h % self.dim

        self._word_axis[word] = axis
        return axis

    # ── Co-occurrence update ───────────────────────────────────

    def _update_cooc(self, tokens: list[str]) -> None:
        """
        Zlicza współwystąpienia par słów w jednym tekście.
        Para (A, B) zawsze w kolejności leksykograficznej → jeden klucz.
        Gdy count >= merge_threshold → scala osie A i B.
        """
        unique = list(set(tokens))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair = (min(unique[i], unique[j]), max(unique[i], unique[j]))
                self._cooc[pair] += 1

                if self._cooc[pair] == self.merge_threshold:
                    self._merge_axes(unique[i], unique[j])

    def _merge_axes(self, word_a: str, word_b: str) -> None:
        """
        Scala osie dwóch słów przez weighted average.
        Słowo z niższym indeksem osi "przyciąga" drugie.
        Po scaleniu oba słowa wskazują na tę samą oś.
        """
        axis_a = self._word_axis.get(word_a)
        axis_b = self._word_axis.get(word_b)

        if axis_a is None or axis_b is None:
            return
        if axis_a == axis_b:
            return  # już na tej samej osi

        # Oś dominująca = niższy indeks (bardziej "zakorzeniona")
        dominant   = min(axis_a, axis_b)
        subordinate= max(axis_a, axis_b)

        # Przepisz wszystkie słowa z subordinate → dominant
        for word, ax in self._word_axis.items():
            if ax == subordinate:
                self._word_axis[word] = dominant

        self._merges += 1

    # ── Encode ─────────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        self.calls += 1
        tokens = self.tokenize(text)
        vec    = np.zeros(self.dim, dtype=np.float32)

        if not tokens:
            return vec

        freqs: dict[str, int] = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1

        for word, count in freqs.items():
            vec[self._get_axis(word)] += count

        # Aktualizuj co-occurrence po encode (uczy się z każdego tekstu)
        self._update_cooc(list(freqs.keys()))

        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 1e-8 else vec

    # ── Cosine similarity ──────────────────────────────────────

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ── Persistencja ───────────────────────────────────────────

    def export_dict(self) -> dict:
        return {
            'dim':             self.dim,
            'next_free':       self._next_free,
            'merge_threshold': self.merge_threshold,
            'merge_alpha':     self.merge_alpha,
            'merges':          self._merges,
            'word_axis':       self._word_axis,
            'cooc':            {f'{a}|{b}': c for (a, b), c in self._cooc.items()},
        }

    def import_dict(self, data: dict) -> None:
        if not data:
            return
        if data.get('dim') and data['dim'] != self.dim:
            raise ValueError(
                f"KuRz dim mismatch: saved={data['dim']} current={self.dim}"
            )
        self._word_axis      = {k: int(v) for k, v in data.get('word_axis', {}).items()}
        self._next_free      = min(self.dim, data.get('next_free', 0))
        self.merge_threshold = data.get('merge_threshold', self.merge_threshold)
        self.merge_alpha     = data.get('merge_alpha',     self.merge_alpha)
        self._merges         = data.get('merges', 0)
        # Przywróć co-occurrence counts
        raw_cooc = data.get('cooc', {})
        self._cooc = defaultdict(int)
        for key, count in raw_cooc.items():
            parts = key.split('|', 1)
            if len(parts) == 2:
                self._cooc[(parts[0], parts[1])] = int(count)

    def save_dict(self, path: Optional[str] = None) -> None:
        p = Path(path) if path else self.dict_path
        if p is None:
            raise ValueError("Brak dict_path — podaj path= lub ustaw dict_path w konstruktorze")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(self.export_dict(), f, ensure_ascii=False, indent=2)

    def load_dict(self, path: Optional[str] = None) -> None:
        p = Path(path) if path else self.dict_path
        if p is None or not p.exists():
            return
        with open(p, encoding='utf-8') as f:
            self.import_dict(json.load(f))

    # ── Statystyki ─────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._word_axis)

    @property
    def axes_used(self) -> int:
        # Ile unikalnych osi faktycznie zajętych (po merge może być mniej niż vocab)
        return len(set(self._word_axis.values()))

    def clusters(self) -> dict[int, list[str]]:
        """Zwraca słowa pogrupowane po osi — widać efekty co-occurrence."""
        result: dict[int, list[str]] = defaultdict(list)
        for word, axis in self._word_axis.items():
            result[axis].append(word)
        return dict(sorted(result.items()))

    def __repr__(self) -> str:
        return (
            f"KuRz(dim={self.dim}, vocab={self.vocab_size}, "
            f"axes={self.axes_used}/{self.dim}, "
            f"merges={self._merges}, calls={self.calls})"
        )
