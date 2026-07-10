# Detekcija Zloupotrebe Kreditnih Kartica

Projekat mašinskog učenja — binarna klasifikacija transakcija kreditnim karticama kao legitimnih ili prevarantskih, korišćenjem klasičnih ML modela (Logistička regresija, Stablo odlučivanja, Random Forest, XGBoost).

> **Grana `18_atributa`:** varijanta glavnog projekta koja trenira modele na **18 izabranih atributa** (umesto svih 30), radi poređenja sa `main` granom. Detalji u sekciji „Izabrani atributi". Rezultati u `results/` se regenerišu pokretanjem pipeline-a i odražavaju 18 atributa.

---

## Dataset

Korišćen je **Credit Card Fraud Detection** dataset (Université Libre de Bruxelles), dostupan na Kaggle platformi. Sastoji se od **284.807 transakcija** od kojih je samo **0.17% prevarantskih** — izrazito neuravnotežen skup podataka.

- **Originalni dataset**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Svaka transakcija sadrži:
- `V1`–`V28` — PCA transformisane originalne karakteristike (anonimizovane zbog poverljivosti)
- `Amount` — iznos transakcije
- `Time` — vreme u sekundama od prve transakcije u skupu
- `Class` — oznaka (0 = legitimna, 1 = prevara)

---

## Pristup

### Analiza duplikata (`src/duplicate_analysis.py`)
- Dijagnostika sirovih podataka **pre uklanjanja**: koliko ima tačnih duplikata (identičnih po svim 30 atributa i klasi) i koliko su obogaćeni prevarama u odnosu na baznu stopu.
- Ništa ne menja niti briše — samo izveštaj (`duplicate_analysis.txt`) koji opravdava odluku o uklanjanju duplikata.

### Priprema podataka (`src/data_preparation.py`)
- **Uklanjanje tačnih duplikata** (1.081 red, artefakti sa identičnim vrednostima) na sirovim podacima, pre transformacije — sprečava da isti primer završi i u trening i u test skupu.
- Transformacija `Time` kolone u `Hour` (sat u toku dana) — deterministička transformacija bez učenja parametara.
- Skaliranje `Amount` kolone vrši se **nakon** podele na skupove kako bi se izbeglo curenje podataka (data leakage).

### Podela podataka (`src/train.py` — `split_data`)
- Podela po transakcijama: **70% trening / 15% validacija / 15% test**, stratifikovana po klasi.
- `RobustScaler` se fita isključivo na trening skupu, a primenjuje na sva tri skupa.

### Balansiranje klasa (`src/train.py` — `apply_smote`)
- Primenjena **SMOTE** tehnika sintetičkog presamplovanja isključivo na trening skupu.
- Unutar unakrsne validacije SMOTE se primenjuje unutar svakog folda kako bi se sprečilo curenje podataka.

### Podešavanje hiperparametara (`src/train.py` — `tune_hyperparameters`)
- **RandomizedSearchCV** sa 5-fold stratifikovanom unakrsnom validacijom.
- Optimizovano po **AUPRC** — najpouzdanija metrika za neuravnotežene skupove.
- Hiperparametri se biraju unakrsnom validacijom (5-fold) na trening skupu; test skup se koristi samo jednom, na kraju.

### Evaluacija (`src/evaluate.py`)
- Metrike (po modelu): Preciznost, Odziv, F1, **F2**, ROC AUC, AUPRC — svaka sačuvana za svaki model, uz zbirnu tabelu na kraju `model_comparison.txt`.
- Matrice konfuzije po modelu, zajednička ROC i Precision-Recall kriva, stubičasti grafik poređenja.
- **Baseline** (nasumičan klasifikator) — dokaz da modeli uče prave obrasce.
- **Analiza grešaka** (`error_analysis.txt`) — karakteristike propuštenih prevara vs uhvaćenih.
- **Podešavanje praga po F2** (`threshold_optimization.txt`) — optimalan prag odluke na validaciji (F2 naglašava odziv). Pragovi se čuvaju i mašinski čitljivo u `models/best_thresholds.json` odakle ih Streamlit aplikacija učitava kao podrazumevani prag po modelu.
- **Potvrda praga na test skupu** (`threshold_payoff_test.txt`) — po modelu, poređenje metrika na pragu 0.5 vs na F2-optimalnom pragu (izabranom na validaciji), primenjeno na test skup — pokazuje da izabrani prag i na nevidljivim podacima podiže odziv (uz očekivani pad preciznosti).

### Izabrani atributi (grana `18_atributa`)
- Ova grana trenira modele **isključivo na 18 atributa** izabranih na `main` grani (selekcija po **pravilu 1-SE**: najmanji broj atributa čiji je CV AUPRC statistički nerazlučiv od svih 30).
- Lista je fiksna u `train.SELECTED_FEATURES`: `V14, V4, V12, V17, V10, V11, V16, V3, V2, V7, V9, V21, V18, V8, Hour, V5, V19, V28`.
- **`Amount` nije među njima** (nije prošao selekciju), pa se ne koristi kao atribut. I dalje se čuva u val/test skupu (`Scaled_Amount`) samo radi prikaza iznosa u aplikaciji i analize grešaka po iznosu.
- Zato ova grana **nema korak selekcije atributa** u pipeline-u (taj posao je obavljen na `main`) — cilj je da se uporedi sa `main` (30 atributa) i utvrdi da li manji, pametno izabran skup daje jednako dobre rezultate.

---

## Struktura projekta

```
.
├── data/
│   ├── raw/                       # Sirovi dataset (creditcard.csv)
│   └── processed/                 # Procesiran dataset i train/val/test skupovi
├── models/                        # Istrenirani modeli (.pkl) i scaler.pkl
├── results/
│   ├── figures/                   # Matrice konfuzije, ROC kriva, EDA grafici
│   └── metrics/                   # Poređenje na testu, tuning, izbor na validaciji, rang atributa
├── src/
│   ├── duplicate_analysis.py      # Analiza duplikata u sirovim podacima
│   ├── data_preparation.py        # Priprema i čišćenje podataka
│   ├── eda_analysis.py            # Eksplorativna analiza (EDA)
│   ├── train.py                   # Podela, skaliranje, SMOTE, tuning, treniranje
│   ├── evaluate.py                # Evaluacija modela i generisanje grafika
│                                  # (nema feature_selection.py — grana koristi fiksnih 18 atributa)
├── app/
│   └── app.py                     # Streamlit web aplikacija
├── pipeline.py                    # Glavni ulaz — pokreće ceo pipeline
├── pyproject.toml                 # Zavisnosti projekta (koristi uv sync)
├── uv.lock                        # Zaključane verzije zavisnosti
└── requirements.txt               # Zavisnosti za pip (alternativa)
```

---

## Pokretanje

Projekat koristi [`uv`](https://github.com/astral-sh/uv) za upravljanje zavisnostima.

```bash
# Instalacija zavisnosti
uv sync
```

### Brzo pokretanje (istrenirani modeli su uključeni u repozitorijum)

Istrenirani modeli i test skup su već uključeni — moguće je odmah pokrenuti aplikaciju bez treniranja:

```bash
uv run python -m streamlit run app/app.py
```

### Pokretanje od nule

Za potpuno treniranje od nule potrebno je preuzeti `creditcard.csv` sa [Kaggle-a](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) i smestiti ga u `data/raw/`, zatim pokrenuti:

```bash
# Pokretanje celog pipeline-a (analiza duplikata → priprema → EDA → treniranje na 18 atributa → izbor modela/prag → evaluacija)
uv run python pipeline.py
```

---

## Rezultati (test skup, ~42.700 transakcija)

| Model | Preciznost | Odziv | F1 | ROC AUC | AUPRC |
|---|---|---|---|---|---|
| Logistička regresija | 0.0512 | 0.8592 | 0.0967 | 0.9630 | 0.6231 |
| Stablo odlučivanja | 0.0605 | 0.7887 | 0.1124 | 0.8912 | 0.6054 |
| Random Forest (100) | 0.8710 | 0.7606 | 0.8120 | 0.9635 | 0.8115 |
| **Random Forest (150)** | **0.8710** | **0.7606** | **0.8120** | **0.9684** | **0.8115** |
| XGBoost | 0.8169 | 0.8169 | 0.8169 | 0.9791 | 0.8141 |
| Baseline (nasumičan) | 0.0000 | 0.0000 | — | — | 0.0008 |

**Izbor modela (na validaciji): RandomForest_150** (AUPRC 0.8564). Tri ansambla su i ovde praktično izjednačena — na test skupu je XGBoost blago najbolji (AUPRC 0.8141), dok su RF-100 i RF-150 na 0.8115. Svi modeli su **drastično bolji od baseline-a** (0.0008), što dokazuje da uče prave obrasce.

> **Poređenje sa `main` (30 atributa):** na jakim ansamblima 18 atributa daje praktično isti AUPRC (RF ~0.8115 vs ~0.8143 na main, XGBoost čak bolji: 0.8141 vs 0.8109) — smanjenje broja atributa za 40% bez značajnog gubitka. Detaljno poređenje: `poredjenje_rezultata/uporedi.py`.

> **Napomena o izboru modela:** Model se bira **isključivo** na validacionom skupu (`results/metrics/validation_selection.txt`); test skup se koristi samo za finalni izveštaj i ne utiče ni na treniranje ni na izbor. Pošto su tri modela gotovo izjednačena, koji je „najbolji" zavisi od skupa.

> **Napomena o duplikatima:** Uklanja se 1.081 tačan duplikat pre podele kako isti primer ne bi završio i u trening i u test skupu (curenje). Rezultati su time pouzdaniji, iako blago niži nego sa zadržanim duplikatima.

> **Napomena o baseline-u:** 0.0008 je izmerena vrednost `DummyClassifier(strategy='stratified')`; teorijski AUPRC nasumičnog klasifikatora jednak je stopi prevara u test skupu (≈ 0.0017 — linija na PR krivoj). Isti red veličine — modeli su i dalje ~500–1000× iznad.

---

## Optimalni hiperparametri (RandomizedSearchCV)

| Model | Parametri | AUPRC (CV) |
|---|---|---|
| Logistička regresija | `C=1` | 0.7089 |
| Stablo odlučivanja | `max_depth=10, min_samples_leaf=4, criterion=entropy` | 0.6329 |
| Random Forest (100) | `max_features=sqrt, max_depth=None` | 0.8268 |
| Random Forest (150) | `max_features=sqrt, max_depth=None` | 0.8278 |
| XGBoost | `learning_rate=0.3, max_depth=7, subsample=0.7, colsample_bytree=0.8` | 0.8380 |

---

## Web aplikacija

Streamlit aplikacija omogućava interaktivnu demonstraciju sistema:
- Odabir modela za analizu
- Podešavanje praga osetljivosti (threshold) putem klizača
- Simulacija legitimne transakcije ili prevare iz test skupa
- Prikaz verovatnoće prevare i odluke sistema

---

## Tehnički detalji

- **Metrike**: AUPRC kao primarna metrika; ROC AUC, F1, Preciznost i Odziv kao dopunske
- **Skaliranje**: `RobustScaler` na `Amount` koloni — otporan na ekstremne vrednosti karakteristične za fraud podatke
- **Balansiranje**: SMOTE isključivo na trening skupu, unutar CV foldova
- **Tuning**: RandomizedSearchCV, 5-fold stratifikovana unakrsna validacija, optimizovano po AUPRC
- **Izbor modela**: najbolji model se bira isključivo na validacionom skupu; finalni rezultat se prijavljuje na test skupu
- **Anti-data leakage**: podela pre skaliranja; scaler se fita samo na trening skupu; validacija služi samo za izbor; test skup nedirnut do finalne evaluacije
