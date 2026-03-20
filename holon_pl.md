# EriAmo: Holon — Warstwa Swiadomej Percepcji dla Modeli Jezykowych

**Architektura Temporalnej Percepcji Kontekstu**

Maciej Mazur¹  
¹ Niezalezny badacz AI, Polska  
GitHub: github.com/Maciej-EriAmo/Holon

---

## Streszczenie

Prezentujemy **Holon** — architekture pamieci sesji dzialajaca jako stanowa warstwa percepcji miedzy uzytkownikiem a dowolnym duzym modelem jezykowym. W odroznieniu od podejsc RAG lub okna przesuwanego, Holon utrzymuje zwarta macierz Φ ∈ R^(k×d) — *geometrie percepcji* — ktora ewoluuje miedzy sesjami poprzez eksponencjalny zanik zalezny od czasu. System produkuje mierzalne redukcje dryfu kontekstu w dlugich sesjach przy jednoczesnej kompresji efektywnego okna kontekstu. Co istotne, Φ nigdy nie jest wstrzykiwane do promptu LLM — dziala wylacznie jako mechanizm selekcji, nie powodujac zadnych dodatkowych kosztow tokenowych. Demonstrujemy, ze Holon jest niezalezny od modelu, dziala na sprz.ecie ARM (Android/Termux) i zachowuje spojnosc semantyczna przez przerwy o dowolnym czasie trwania dzieki biologicznie inspirowanemu modelowi zaniku temporalnego.

---

## 1. Wprowadzenie

Wspolczesne modele jezykowe przetwarzaja kazda sesje jako bezstanowa sekwencje. Model z oknem kontekstu 32K tokenow nie ma pamieci o tym, co bylo omawiane wczoraj, nie ma swiadomosci ze minely trzy godziny od ostatniej wiadomosci i nie ma mechanizmu odrozniajacego, ktore fragmenty dlugiej rozmowy sa semantycznie centralne od tych bedacych szumem peryferyjnym.

Standardowa inzynierska odpowiedz — zarzadzanie kontekstem przez okno przesuwne — jest rozwiazaniem brute-force. Wysyla do modelu ostatnie N par wiadomosci, odrzucajac wszystko poza tym horyzontem. To podejscie jest obliczeniowo marnotrawne na gorze okna i informacyjnie slepe poza nim.

Holon przyjmuje inne podejscie oparte na trzech obserwacjach:

1. **Problem kontekstu jest geometryczny, nie objetos.ciowy.** Liczy sie nie *ile* historii jest obecne, ale *ktore jej czesci* sa strukturalnie istotne dla biezacego zapytania.

2. **Sesje maja strukture temporalna.** Rozmowa przerwana na osiem godzin nie jest rownowazna tej, ktora plynela nieprzerwanie. Model powinien zachowywac sie inaczej.

3. **Percepcja moze byc oddzielona od tresci.** System moze utrzymywac zwarta reprezentacje *jak interpretowac* sesje bez przechowywania sesji dosłownie.

Te obserwacje motywuja architekture Holon.

---

## 2. Architektura

### 2.1 Dwie Klasy Pamieci

Holon dzieli stan sesji na dwie klasy:

- **Klasa T (Temporalna)**: Itemy aktualnie w oknie kontekstu — ostatnie wymiany, aktywny kod, biezace zapytanie. Podlegaja normalnej uwadze LLM.
- **Klasa P (Permanentna)**: Macierz percepcji Φ ∈ R^(k×d), przechowywana poza oknem kontekstu. Trwa miedzy sesjami. Nigdy nie wysylana do LLM.

To rozdzielenie jest fundamentalne. Φ nie niesie informacji *dla* modelu — niesie informacje *o tym, jak filtrowac* to, co trafia do modelu.

### 2.2 Macierz Percepcji Φ

Φ sklada sie z k wektorow wierszowych, kazdy wymiaru d (wymiar embeddingu). Kazdy wiersz reprezentuje wyuczony atraktor w przestrzeni semantycznej:

- Wiersz 0: wzorce techniczne/kod (wolny zanik, polokres 48h)
- Wiersz 1: wzorce projektu/architektury (sredni zanik, 24h)
- Wiersz 2: wzorce konwersacyjne (szybki zanik, 8h)
- Wiersz 3: biezacy kontekst roboczy (najszybszy zanik, 4h)

Wiersze sa utrzymywane wzajemnie ortogonalne przez czlon odpychania stosowany po kazdej aktualizacji, zapewniajac ze kazdy atraktor sie specjalizuje zamiast zlewac sie w wspolne centrum.

### 2.3 Cykl Turnu

Kazdy pelny turn przebiega w scislej kolejnosci:

1. **Recall** (Φ → T): Embedding zapytania aktywuje najblizszy wiersz Φ; top-N itemow ze store bliskich temu atraktorowi zostaje oznaczonych do wlaczenia.
2. **Dodanie**: Nowy item (wiadomosc uzytkownika) dodany do store z wiekiem=0.
3. **Vacuum**: Itemy oceniane przez `relevancja × exp(-wiek/τ)`. Itemy o niskim wyniku usuniete. Enforced hard cap.
4. **Budowanie okna**: Top-n itemow wedlug wazonej bliskosci do dynamicznego centrum Φ.
5. **Aktualizacja Φ**: Najslabszy wiersz przesuwa sie w kierunku wazonego wzorca aktywnych itemow (przypomniane itemy waga 2×, nowe 1.5×, stare 1×).
6. **wiek++** (tylko w `after_turn` — jeden przyrost na pelna wymiane).

### 2.4 Zanik Temporalny

Gdy system jest zamkniety i ponownie otwarty po δ godzinach, Φ ewoluuje:

```
Φ[k] ← Φ[k] × exp(−ln2 × δ / polokres[k])
```

Kazdy wiersz zaniku z wlasna szybkoscia. Po 12 godzinach wzorce konwersacyjne w znacznej mierze zanikly, podczas gdy wzorce techniczne pozostaja silne. System generuje komunikat przebudzenia opisujacy co sie stalo podczas nieobecnosci:

> *"[Minely 3.2 godziny od ostatniej sesji. Bylo 15 turnow, 4 wzorce w pamieci. Wzorce konwersacyjne lekko przybladly.]"*

To nie jest kosmetyczne. Zanik bezposrednio wplywa na to, ktore itemy przezywaja vacuum i ktore wiersze Φ dominuja recall — system naprawde "pamięta inaczej" po uplywie czasu.

### 2.5 Budzet Tokenow

Wszystkie wiadomosci skladane przez Holon podlegaja budzetowi tokenow:

- Limit modelu: 4096 tokenow (Mistral 7B baseline)
- Zarezerwowane na odpowiedz: 512 tokenow
- Zarezerwowane na system prompt: 256 tokenow
- Zarezerwowane na zapytanie: 512 tokenow
- **Dostepne dla pamieci Holon: 2816 tokenow (~9856 znakow przy 3.5 znakow/token dla polskiego)**

Kazdy item pamieci jest ograniczony do 300 znakow. Itemy sa wlaczane malejacym priorytetem az do wyczerpania budzetu.

---

## 3. Kluczowe Wlasciwosci

### 3.1 Zerowy Naklad Promptu

Φ jest uzywane wylacznie do selekcji. Nigdy nie pojawia sie w prompcie wyslalnym do LLM. Jest to architektonicznie istotne: surowy wektor embeddingu (32 floaty ≈ 200 tokenow) dodany do kazdego promptu pochonalby okolo 7% calkowitego budzetu kontekstu dla zerowej korzysci semantycznej dla modelu.

### 3.2 Niezaleznosc od Modelu

Holon wymaga jedynie:
- Funkcji embeddingu (Gemini API lub hash fallback)
- Endpointu czatu kompatybilnego z OpenAI

Byl testowany z Gemini 2.5 Flash i Mistral 7B przez Featherless. Ta sama implementacja Python dziala na Androidzie (Termux ARM) bez modyfikacji.

### 3.3 Trwala Tozsamosc

Macierz percepcji Φ jest zapisywana na dysk po kazdym turnie. Przy wczytaniu jest ewoluowana do przodu o uplyniely czas przed rozpoczeciem nowego turnu. Sesja przerwana na tydzien i wznowiona produkuje inne zachowanie niz wznowiona po pieciu minutach — nie dlatego ze rozna tresc jest obecna, ale dlatego ze geometria percepcji sie zmienila.

---

## 4. Wyniki Eksperymentalne

### 4.1 Stabilnosc Sesji (24 turny, Gemini 2.5 Flash)

| Metryka | Wartosc |
|---|---|
| Rozmiar store (stabilny) | 4 itemy przez cala sesje |
| Srednia relevancja (turn 1) | 0.036 |
| Srednia relevancja (turn 24) | 0.198 |
| Normy Phi | [1.0, 1.0, 1.0, 1.0] |
| Persistencja | Potwierdzona przez restarty |
| Swiadomosc czasu | Aktywna (delta raportowana poprawnie) |

Φ zbiega sie z losowej inicjalizacji w kierunku geometrii specyficznej dla sesji w okolo 20-30 turnach z prawdziwym modelem embeddingu.

### 4.2 Dryf Kontekstu (30 turnow, hash embedder, baseline Mistral)

| Metryka | Holon | Baseline (okno=5) | Delta |
|---|---|---|---|
| Sredni dryf kontekstu | 0.5921 | 0.6165 | **−4.0%** |
| Dryf ostatnich 10 | 0.5419 | 0.6156 | **−12.0%** |
| Sredni rozmiar store | 3.9 | 5.0 | −22% |

Efekt rosnie z dlugoscia sesji. Na turnie 25 dryf bazowy osiagnal 1.02 (odpowiedz prawie ortogonalna do zapytania) podczas gdy Holon utrzymal 0.28. Demonstruje to podstawowa wlasciwosc: Holon degraduje sie lagodnie w dlugich sesjach, podczas gdy podejscia z oknem przesuwanym wykazuja nagle awarie gdy fakty wychodza poza horyzont okna.

---

## 5. Warstwa Swiadomej Percepcji

Nazwa "Holon" odzwierciedla architektoniczna pozycje systemu. Holon (Koestler, 1967) jest jednoczesnie caloscia i czescia — samowystarczalny, a jednoczesnie osadzony w wiekszej strukturze. Holon jako system jest:

- **Samowystarczalny**: Dziala niezaleznie od LLM, nie wymaga modyfikacji modelu, retreningu ani dostepu do wag.
- **Osadzony**: Ksztaltuje kazda interakcje z LLM bez wiedzy LLM.

Framing "swiadomej warstwy przegladarki" pozycjonuje Holon jako system siedzacy miedzy ludzka intencja a odpowiedzia modelu — nie przetwarzajacy jezyka, lecz przetwarzajacy *kontekst*. Odpowiada na pytanie: *biorac pod uwage wszystko co sie wydarzylo w tej relacji, na co model powinien zwrocic uwage teraz?*

Jest to strukturalnie podobne do tego, co robi doswiadczony asystent przygotowujac briefing: nie przekazuje wszystkich kiedykolwiek napisanych emaili; destyluje geometrie sytuacji.

---

## 6. Relacja z HoloMem

HoloMem jest nazwa zarezerwowana dla nastepnego kroku architektonicznego: bezposredniego wstrzykiwania Φ w mechanizm uwagi modelu jako addytywny bias przed softmaxem:

```
uwaga = softmax(QK^T / √d + α · (Q · Φ_center^T))
```

Wymaga to dostepu do wag modelu i jest planowane jako osobna publikacja. Holon (system opisany tutaj) dziala na warstwie promptu i nie wymaga takiego dostepu. Oba sa skladalne: Holon moze byc wdrozony dzis na dowolnym modelu; HoloMem wymaga fine-tuningu lub dostepu na poziomie inferencji.

---

## 7. Porownanie z Pokrewnymi Pracami

| System | Zarzadzanie kontekstem | Swiadomosc czasu | Niezaleznosc od modelu | Naklad promptu |
|---|---|---|---|---|
| Okno przesuwne | Obcinanie przez recencje | Brak | Tak | Brak |
| RAG | Wyszukiwanie podobienstwa | Brak | Tak | Wysoki (chunki) |
| MemGPT | Hierarchiczne stronicowanie | Brak | Czesciowe | Wysoki |
| **Holon** | **Geometryczna selekcja przez Φ** | **Zanik eksponencjalny** | **Tak** | **Zero** |

Kluczowym wyroznikiem jest kombinacja wyuczonej geometrycznej selekcji z zanikaniem temporalnym i zerowym nakladem promptu.

---

## 8. Implementacja

Holon jest zaimplementowany w jednym pliku Python (`holon.py`, ~900 linii) z dwiema zaleznosci: `numpy` i `google-genai`. Dziala na Pythonie 3.10+ wlacznie z Androidem (Termux ARM). Dostepna jest rowniez implementacja Android (Kotlin) z persistentnym przechowywaniem danych.

Persistencja uzywa atomowych zapisow plikow (zapis do `.tmp`, nastepnie rename) zapewniajac, ze nagle zakonczenie procesu nie moze uszkodzic pliku pamieci.

Zrodlo: github.com/Maciej-EriAmo/Holomem

---

## 9. Wnioski

Holon demonstruje, ze efektywna pamiec dlugich sesji nie wymaga modyfikacji modelu, duzych baz wektorowych ani znaczacego nakladu promptu. Zwarta macierz k×d, ewoluujaca przez swiadomy czasu zanik eksponencjalny i aktualizowana przez wazony gradientowo krok interferencji, zapewnia znaczaca spojnosc kontekstu przy zaniedbywalnym koszcie obliczeniowym.

Wymiar temporalny — swiadomosc systemu o tym, jak dlugo byl uslipiony — nie jest cecha estetyczna. To jest to, co roznia system pamieci od pamieci podrecznej (cache).

---

## Literatura

Koestler, A. (1967). *The Ghost in the Machine*. Hutchinson.

Brown, T. i in. (2020). Language Models are Few-Shot Learners. *NeurIPS*.

Packer, C. i in. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv:2310.08560*.

Lewis, P. i in. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.

Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*.
