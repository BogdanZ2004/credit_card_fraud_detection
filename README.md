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

### Priprema podataka (`src/data_preparation.py`)
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
- Metrike: Preciznost, Odziv, F1, ROC AUC, AUPRC.
- Matrice konfuzije po modelu i zajednička ROC kriva svih modela.

### Odabir atributa (`src/feature_selection.py`)
- Exhaustivna analiza: isprobava top-1 do top-30 atributa po važnosti iz Random Forest modela.
- Evaluacija na validacionom skupu radi ispravnog poređenja.
- Rezultat je rang lista koja pokazuje koliko atributa je dovoljno za maksimalan odziv — koristi se da se proveri da li su svi atributi neophodni ili se neki mogu ukloniti bez gubitka performansi.
- Nije deo glavnog pipeline-a — pokreće se zasebno: `uv run python src/feature_selection.py`

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
# Pokretanje celog pipeline-a (priprema → EDA → treniranje → evaluacija)
uv run python pipeline.py
```

---

## Rezultati (test skup, ~42.700 transakcija)

| Model | Preciznost | Odziv | F1 | ROC AUC | AUPRC |
|---|---|---|---|---|---|
| Logistička regresija | 0.0553 | 0.8784 | 0.1040 | 0.9627 | 0.7389 |
| Stablo odlučivanja | 0.0664 | 0.7703 | 0.1223 | 0.8977 | 0.6931 |
| Random Forest (100) | 0.8429 | 0.7973 | 0.8194 | 0.9710 | 0.8339 |
| Random Forest (150) | 0.8551 | 0.7973 | 0.8252 | 0.9773 | 0.8361 |
| **XGBoost** | **0.8493** | **0.8378** | **0.8435** | **0.9702** | **0.8428** |

**Najbolji model: XGBoost** — najviši AUPRC i na validaciji (0.8690) i na test skupu (0.8428), uz najviši Odziv (0.8378).

> **Napomena o izboru modela:** Model se bira na **validacionom skupu** (`results/metrics/validation_selection.txt`), gde XGBoost ima najviši AUPRC (0.8690), tesno ispred Random Forest-a (0.8682). Isti model je najbolji i na test skupu. XGBoost koristi early stopping sa strpljivošću 50 (staje oko 121. iteracije).

---

## Optimalni hiperparametri (RandomizedSearchCV)

| Model | Parametri | AUPRC (CV) |
|---|---|---|
| Logistička regresija | `C=0.1` | 0.7675 |
| Stablo odlučivanja | `max_depth=10, min_samples_leaf=8, criterion=entropy` | 0.6409 |
| Random Forest (100) | `max_features=sqrt, max_depth=None` | 0.8448 |
| Random Forest (150) | `max_features=sqrt, max_depth=None` | 0.8477 |
| XGBoost | `learning_rate=0.3, max_depth=7, subsample=0.7, colsample_bytree=0.8` | 0.8449 |

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
- **Early stopping**: XGBoost prati AUPRC na validacionom skupu i staje kad prestane napredovanje (sprečava preprilagođavanje)
- **Izbor modela**: najbolji model se bira na validacionom skupu, finalni rezultat se prijavljuje na test skupu
- **Anti-data leakage**: podela pre skaliranja; scaler se fita samo na trening skupu; test skup nedirnut do finalne evaluacije
