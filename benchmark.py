import os, json, time
import numpy as np
from holon import HoloMem, Embedder, Config
from kurz import KuRz

# Konfiguracja środowiska
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL      = "gemini-2.0-flash-lite"

SYSTEM = "Asystent. Odpowiadaj zwięźle. Pamiętaj fakty."

SCRIPT = [
    ("fact",  "Mój projekt ma nazwę kodową SIGMA.",                    None),
    ("noise", "Czym różni się lista od krotki w Pythonie?",            None),
    ("noise", "Co to jest dekorator w Pythonie?",                      None),
    ("probe", "Jak nazywa się mój projekt? Podaj nazwę kodową.",       "SIGMA"),
    ("fact",  "Używam biblioteki loguru do logowania.",                None),
    ("noise", "Wyjaśnij różnicę między asyncio a threading.",         None),
    ("noise", "Jak działa GIL w Pythonie?",                           None),
    ("noise", "Co to jest context manager?",                          None),
    ("probe", "Jakiej biblioteki używam do logowania?",               "loguru"),
    ("fact",  "Klasa główna nazywa się DataPipeline i ma metodę process(batch_size=64).", None),
    ("noise", "Jak działa garbage collector w Pythonie?",             None),
    ("noise", "Co to jest metaclass?",                                None),
    ("probe", "Jak nazywa się klasa główna i jaki ma parametr?",      "DataPipeline"),
    ("fact",  "Baza danych to PostgreSQL, hasło: sigma2024.",          None),
    ("noise", "Co to jest ACID w bazach danych?",                     None),
    ("probe", "Jak nazywa się mój projekt? Przypomnij nazwę kodową.", "SIGMA"),
    ("probe", "Jakiej biblioteki używam do logowania w projekcie?",   "loguru"),
    ("probe", "Jakie jest hasło do bazy danych?",                     "sigma2024"),
    ("probe", "Wymień wszystkie fakty: nazwa projektu, logi, baza.",   "SIGMA"),
]

# Globalny client — tworzony raz
_GEMINI_CLIENT = None
def _get_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai
        _GEMINI_CLIENT = genai.Client(api_key=GEMINI_KEY)
    return _GEMINI_CLIENT

def call_llm(messages):
    """Gemini — bezpośrednie wywołanie przez google-genai."""
    try:
        from google.genai import types
        client = _get_client()

        sys_parts = [m["content"] for m in messages if m["role"] == "system"]
        contents  = []
        for m in messages:
            if m["role"] == "user":
                contents.append(types.Content(role="user",  parts=[types.Part(text=m["content"])]))
            elif m["role"] == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=m["content"])]))

        for attempt in range(4):
            try:
                resp = client.models.generate_content(
                    model    = MODEL,
                    contents = contents,
                    config   = types.GenerateContentConfig(
                        system_instruction = "\n".join(sys_parts) or None,
                        max_output_tokens  = 128,
                        temperature        = 0.1,
                    )
                )
                return resp.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    wait = 20 * (attempt + 1)
                    print(f"\n  [Rate limit] czekam {wait}s...")
                    time.sleep(wait)
                else:
                    return f"[Błąd: {err[:60]}]"
        return "[Błąd: wyczerpano retry]"
    except Exception as e:
        return f"[Błąd: {e}]"

# Reszta klas (Baseline, CachedEmbedder) pozostaje bez zmian jak w poprzednim kodzie...
class Baseline:
    def __init__(self, window):
        self.window = window
        self.history = []
    def messages(self, query):
        msgs = [{"role": "system", "content": SYSTEM}]
        # Trimming — skróć każdą wiadomość do 200 znaków
        for u, a in self.history[-self.window:]:
            msgs += [
                {"role": "user",      "content": u[:200]},
                {"role": "assistant", "content": a[:200]},
            ]
        msgs.append({"role": "user", "content": query})
        return msgs
    def update(self, q, a): self.history.append((q, a))

class CachedEmbedder(Embedder):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._cache = {}
    def encode(self, text):
        k = text[:200]
        if k in self._cache: return self._cache[k]
        for attempt in range(4):
            try:
                vec = super().encode(text)
                self._cache[k] = vec
                return vec
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    w = 15 * (attempt + 1)
                    print(f"\n  [Embed limit] czekam {w}s...")
                    time.sleep(w)
                else:
                    break
        vec = self._hash_embed(text)
        self._cache[k] = vec
        return vec

def check_fact(response, expected):
    return expected.lower() in response.lower()

def run():
    print("=" * 70)
    print(f"Holon v2.5 — Benchmark Kroczacy [{MODEL}]")
    print("KuRz-warm vs KuRz-cold vs Hash vs Baseline w=5")
    print("=" * 70)
    if not GEMINI_KEY:
        print("BLAD: ustaw GEMINI_API_KEY")
        return

    # KuRz-warm: pre-train na 3x scenariusz (symulacja poprzednich sesji)
    from kurz import KuRz
    kurz_warm = KuRz(dim=256, merge_threshold=5)
    for _ in range(3):
        for _, query, _ in SCRIPT:
            kurz_warm.encode(query)
    print(f"KuRz-warm: vocab={kurz_warm.vocab_size} merges={kurz_warm._merges}")

    # Konfiguracje embedderow
    def make_embedder(name):
        emb = Embedder(dim=256)
        if name == "KuRz-warm":
            # Wstrzyknij nauczony slownik
            from collections import defaultdict as _dd
            emb._kurz._word_axis = dict(kurz_warm._word_axis)
            cooc = _dd(int); cooc.update(kurz_warm._cooc)
            emb._kurz._cooc      = cooc
            emb._kurz._merges    = kurz_warm._merges
            emb._kurz._next_free = kurz_warm._next_free
        elif name == "KuRz-cold":
            pass  # swiezy KuRz bez historii
        elif name == "Hash":
            # Wyczysc axis mapping - pure hash
            emb._kurz._word_axis = {}
            emb._kurz._next_free = emb._kurz.dim  # wymus hash path
        return emb

    EMBEDDER_NAMES = ["KuRz-warm", "KuRz-cold", "Hash"]
    all_results = {}
    bl5_results = []

    for emb_name in EMBEDDER_NAMES:
        print(f"\n{'='*30} {emb_name} {'='*30}")
        embedder = make_embedder(emb_name)
        cfg = Config(dim=256, lr=0.03, threshold=0.42, vacuum_age_tau=40.0, n=3)
        hm  = HoloMem(embedder, cfg, f"hm_{emb_name}_bench.json")
        bl5 = Baseline(5)
        hm.start_session()

        res_hm = []
        res_bl = []

        print(f"\n{'#':>3} | {'typ':>5} | {'HM':>3} | {'B5':>3} | Oczekiwane")
        print("-" * 55)

        for i, (turn_type, query, expected) in enumerate(SCRIPT):
            r_hm = call_llm(hm.turn(query, SYSTEM))
            hm.after_turn(query, r_hm)
            time.sleep(3)
            r_b5 = call_llm(bl5.messages(query))
            bl5.update(query, r_b5)
            time.sleep(3)

            if turn_type == "probe":
                h  = check_fact(r_hm, expected)
                b  = check_fact(r_b5, expected)
                res_hm.append(h)
                res_bl.append(b)
                mark = ["OK" if x else "X" for x in [h, b]]
                print(f"{i+1:>3} | probe | {mark[0]:>3} | {mark[1]:>3} | {expected}")
                print(f"       HM: {r_hm[:65]}")
            else:
                s = hm.stats()
                print(f"{i+1:>3} | {turn_type:>5} |  -  |  -  | store={s['store_size']} rel={s['avg_rel']:.3f}")

        all_results[emb_name] = res_hm
        if not bl5_results:
            bl5_results = res_bl

        import os as _os
        if _os.path.exists(f"hm_{emb_name}_bench.json"):
            _os.unlink(f"hm_{emb_name}_bench.json")

    # Tabela porowncza
    n = len(bl5_results)
    best_score = max(sum(v) for v in all_results.values())

    print(f"\n\n{'='*70}")
    print(f"PODSUMOWANIE — retencja faktow ({n} prob)")
    print(f"{'='*70}")
    print(f"{'System':<20} {'Trafien':>8} {'%':>8}")
    print("-" * 42)
    for name, res in all_results.items():
        pct    = sum(res)/n*100 if n else 0
        marker = " <-- BEST" if sum(res) == best_score else ""
        print(f"{'Holon+'+name:<20} {sum(res):>5}/{n} {pct:>7.1f}%{marker}")
    print(f"{'Baseline w=5':<20} {sum(bl5_results):>5}/{n} {sum(bl5_results)/n*100:>7.1f}%")

    # Per-probe
    probes = [(i, exp) for i, (t,_,exp) in enumerate(SCRIPT) if t == "probe"]
    print(f"\nPer-probe detail:")
    hdr = f"  {'#':>3} | {'Expected':<15} | " + " | ".join(f"{n:>11}" for n in list(all_results.keys()) + ["Baseline-5"])
    print(hdr)
    print("  " + "-" * len(hdr))
    for pi, (si, exp) in enumerate(probes):
        row  = f"  {si+1:>3} | {exp:<15} | "
        row += " | ".join(f"{'OK':>11}" if all_results[nm][pi] else f"{'X':>11}" for nm in all_results)
        row += f" | {'OK':>10}" if bl5_results[pi] else f" | {'X':>10}"
        print(row)

    # Zapisz JSON
    out = {
        "model":   MODEL,
        "probes":  n,
        "results": {k: {"hits": sum(v), "pct": round(sum(v)/n*100,1)} for k,v in all_results.items()},
        "baseline_w5": {"hits": sum(bl5_results), "pct": round(sum(bl5_results)/n*100,1)},
        "per_probe": {k: v for k,v in all_results.items()},
    }
    import json as _json
    with open("benchmark_results.json", "w") as f:
        _json.dump(out, f, indent=2)
    print(f"\nWyniki -> benchmark_results.json")


if __name__ == "__main__":
    run()
