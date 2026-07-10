# Detekcija zloupotrebe kreditnih kartica — grana `18_atributa`

Varijanta glavnog projekta koja trenira modele na **18 najznačajnijih atributa** (umesto svih 30), radi poređenja sa `main` granom. Atributi su izabrani na `main` grani pravilom 1-SE (AUPRC + 5-fold CV).

Skup podataka: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle, ULB) — 284.807 transakcija, a posle uklanjanja 1.081 tačnog duplikata ostaje 283.726.

## Izabranih 18 atributa

V14, V4, V12, V17, V10, V11, V16, V3, V2, V7, V9, V21, V18, V8, Hour, V5, V19, V28.

`Amount` nije prošao selekciju, pa **nije atribut modela** — ali se i dalje čuva (skaliran) u skupovima radi prikaza iznosa u aplikaciji i analize grešaka po iznosu.

## Rezultati (test skup, prag 0.5)

| Model | Preciznost | Odziv | F1 | F2 | ROC-AUC | AUPRC |
|---|---|---|---|---|---|---|
| Logistička regresija | 0.0512 | 0.8592 | 0.0967 | 0.2068 | 0.9630 | 0.6231 |
| Stablo odlučivanja | 0.0605 | 0.7887 | 0.1124 | 0.2316 | 0.8912 | 0.6054 |
| Random Forest (100) | 0.8710 | 0.7606 | 0.8120 | 0.7803 | 0.9635 | 0.8115 |
| **Random Forest (150)** | 0.8710 | 0.7606 | 0.8120 | 0.7803 | 0.9684 | 0.8115 |
| XGBoost | 0.8169 | 0.8169 | 0.8169 | 0.8169 | 0.9791 | 0.8141 |

- **Izabrani model (na validaciji): Random Forest (150)** (AUPRC 0.8564); test se koristi samo za finalni izveštaj.
- XGBoost je na 18 atributa najbalansiraniji i ima najviši AUPRC na testu (0.8141).
- **Poređenje sa `main` granom:** kod ansambala je razlika po AUPRC zanemarljiva (XGBoost je čak i malo bolji sa 18 atributa), pa smanjenje sa 30 na 18 atributa praktično ne šteti kvalitetu, uz jednostavniji model.

## Pokretanje aplikacije (Streamlit)

Istrenirani modeli (na 18 atributa), skaler i test skup su već u repozitorijumu, pa aplikacija radi odmah, bez treniranja:

```bash
uv sync
uv run python -m streamlit run app/app.py
```

Aplikacija omogućava izbor modela, podešavanje praga klizačem (podrazumevano F2-optimalni prag) i simulaciju legitimne transakcije ili prevare iz test skupa.

## Pokretanje treniranja od nule

Preuzmi `creditcard.csv` sa [Kaggle-a](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) i smesti ga u `data/raw/`, pa pokreni pipeline:

```bash
uv sync
uv run python pipeline.py
```

Pipeline ima 7 koraka (bez selekcije atributa — lista 18 atributa je fiksna, izabrana na `main` grani):
analiza duplikata → priprema podataka → EDA → treniranje (na 18 atributa) → izbor modela → podešavanje praga (F2) → evaluacija na test skupu.

## Struktura

```
data/raw/          # Sirovi dataset (creditcard.csv — preuzeti sa Kaggle-a)
data/processed/    # Procesirani podaci i test skup
models/            # Istrenirani modeli (na 18 atributa), scaler.pkl i best_thresholds.json
results/           # Metrike (.txt) i grafici (.png)
src/               # duplicate_analysis, data_preparation, eda_analysis, train, evaluate
app/app.py         # Streamlit aplikacija
pipeline.py        # Pokreće ceo tok od početka do kraja
```

Detaljna dokumentacija projekta (`Dokumentacija_RA143_2023.pdf`) nalazi se na `main` grani.
