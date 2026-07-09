# Detekcija Zloupotrebe Kreditnih Kartica

Projekat mašinskog učenja — binarna klasifikacija transakcija kreditnim karticama kao legitimnih ili prevarantskih, korišćenjem klasičnih ML modela (Logistička regresija, Stablo odlučivanja, Random Forest, XGBoost).

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

### Odabir atributa (`src/feature_selection.py`)
- Atributi se rangiraju po važnosti iz već istreniranog Random Forest modela (treniran isključivo na trening skupu).
- Za svaki top-k podskup meri se **AUPRC** kroz **5-fold stratifikovanu unakrsnu validaciju na trening skupu** (SMOTE unutar svakog folda) — test skup se **ne dodiruje**.
- Broj atributa se bira **pravilom 1 standardne greške (1-SE)**: najmanji k čiji je CV AUPRC unutar jedne standardne greške od najboljeg — najparsimoničniji model koji je statistički nerazlučiv od najboljeg.
- Rezultat: rang lista + grafik `feature_selection_auprc_vs_k.png` (AUPRC vs broj atributa sa pragom 1-SE).
- Deo je pipeline-a kao **KORAK 6** (posle treniranja, pre finalne evaluacije na testu). Na ovoj grani (svih 30 atributa) služi kao **informativna analiza** — finalni modeli i dalje koriste sve atribute. Predlog broja atributa je osnova za posebnu granu koja modele trenira na izabranom podskupu i tek onda jednom ocenjuje na test skupu.

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
│   └── feature_selection.py       # Analiza važnosti atributa
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
# Pokretanje celog pipeline-a (analiza duplikata → priprema → EDA → treniranje → izbor modela/prag → selekcija atributa → evaluacija)
uv run python pipeline.py
```

---

## Rezultati (test skup, ~42.700 transakcija)

| Model | Preciznost | Odziv | F1 | ROC AUC | AUPRC |
|---|---|---|---|---|---|
| Logistička regresija | 0.0516 | 0.8592 | 0.0974 | 0.9630 | 0.6796 |
| Stablo odlučivanja | 0.0375 | 0.7746 | 0.0716 | 0.8673 | 0.6401 |
| **Random Forest (100)** | **0.9032** | **0.7887** | **0.8421** | **0.9745** | **0.8143** |
| Random Forest (150) | 0.8889 | 0.7887 | 0.8358 | 0.9538 | 0.8144 |
| XGBoost | 0.7671 | 0.7887 | 0.7778 | 0.9594 | 0.8109 |
| Baseline (nasumičan) | 0.0000 | 0.0000 | — | — | 0.0008 |

**Izbor modela (na validaciji): RandomForest_100** (AUPRC 0.8560). Tri ansambla su praktično izjednačena — na test skupu RF-150 (0.8144) i RF-100 (0.8143) su gotovo identični, XGBoost tik iza (0.8109). Svi modeli su **drastično bolji od baseline-a** (0.0008), što dokazuje da uče prave obrasce.

> **Napomena o izboru modela:** Model se bira **isključivo** na validacionom skupu (`results/metrics/validation_selection.txt`); test skup se koristi samo za finalni izveštaj i ne utiče ni na treniranje ni na izbor. Pošto su tri modela gotovo izjednačena, koji je „najbolji" zavisi od skupa.

> **Napomena o duplikatima:** Uklanjanjem 1.081 tačnog duplikata metrike su blago pale (npr. XGBoost AUPRC 0.842 → 0.811) — potvrda da su duplikati blago naduvavali rezultate (isti primer u trening i test skupu). Novi brojevi su pouzdaniji.

> **Napomena o baseline-u:** 0.0008 je izmerena vrednost `DummyClassifier(strategy='stratified')`; teorijski AUPRC nasumičnog klasifikatora jednak je stopi prevara u test skupu (≈ 0.0017 — linija na PR krivoj). Isti red veličine — modeli su i dalje ~500–1000× iznad.

---

## Optimalni hiperparametri (RandomizedSearchCV)

| Model | Parametri | AUPRC (CV) |
|---|---|---|
| Logistička regresija | `C=100` | 0.7368 |
| Stablo odlučivanja | `max_depth=10, min_samples_leaf=8, criterion=entropy` | 0.5870 |
| Random Forest (100) | `max_features=log2, max_depth=None` | 0.8379 |
| Random Forest (150) | `max_features=sqrt, max_depth=None` | 0.8383 |
| XGBoost | `learning_rate=0.3, max_depth=7, subsample=0.8, colsample_bytree=1.0` | 0.8433 |

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
