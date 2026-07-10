# Detekcija zloupotrebe kreditnih kartica

Projekat mašinskog učenja za binarnu klasifikaciju transakcija kao legitimnih ili prevarantskih, na izrazito neuravnoteženom skupu (0,167% prevara). Poredi se pet modela: logistička regresija, stablo odlučivanja, Random Forest (100 i 150 stabala) i XGBoost.

Skup podataka: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle, ULB) — 284.807 transakcija, a posle uklanjanja 1.081 tačnog duplikata ostaje 283.726.

## Rezultati (test skup, prag 0.5)

| Model | Preciznost | Odziv | F1 | F2 | ROC-AUC | AUPRC |
|---|---|---|---|---|---|---|
| Logistička regresija | 0.0516 | 0.8592 | 0.0974 | 0.2080 | 0.9630 | 0.6796 |
| Stablo odlučivanja | 0.0375 | 0.7746 | 0.0716 | 0.1571 | 0.8673 | 0.6401 |
| **Random Forest (100)** | **0.9032** | **0.7887** | **0.8421** | **0.8092** | **0.9745** | **0.8143** |
| Random Forest (150) | 0.9016 | 0.7746 | 0.8333 | 0.7971 | 0.9723 | 0.8153 |
| XGBoost | 0.7671 | 0.7887 | 0.7778 | 0.7843 | 0.9594 | 0.8109 |

- **Primarna metrika je AUPRC** (najrealnija na neuravnoteženim podacima); tačnost se ne koristi jer bi model koji sve proglasi legitimnim imao ~99,8% tačnosti bez ikakve koristi.
- **Izabrani model: Random Forest (100)** — bira se isključivo na validacionom skupu (AUPRC 0.8560); test se koristi samo za finalni izveštaj.
- Tri ansambla su praktično izjednačena (~0.81 AUPRC) i drastično bolji od nasumičnog baseline-a (0.0008).
- **Prag odluke** se bira po F2 meri na validaciji (F2 naglašava odziv jer je propuštena prevara skuplja od lažne uzbune) i čuva se u `models/best_thresholds.json`.

Detaljne metrike i grafici su u `results/`, a kompletan opis u `Dokumentacija_RA143_2023.pdf`.

## Pokretanje aplikacije (Streamlit)

Istrenirani modeli, skaler i test skup su već u repozitorijumu, pa aplikacija radi odmah, bez treniranja:

```bash
uv sync
uv run python -m streamlit run app/app.py
```

Aplikacija omogućava izbor modela, podešavanje praga klizačem (podrazumevano se postavlja F2-optimalni prag za izabrani model) i simulaciju legitimne transakcije ili prevare iz test skupa.

## Pokretanje treniranja od nule

Preuzmi `creditcard.csv` sa [Kaggle-a](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) i smesti ga u `data/raw/`, pa pokreni ceo pipeline:

```bash
uv sync
uv run python pipeline.py
```

Pipeline ima 8 koraka (redosled je namerno takav da se izbegne curenje podataka — test se dira tek na kraju):
analiza duplikata → priprema podataka → EDA → treniranje → izbor modela na validaciji → podešavanje praga (F2) → selekcija atributa → evaluacija na test skupu.

## Struktura

```
data/raw/          # Sirovi dataset (creditcard.csv — preuzeti sa Kaggle-a)
data/processed/    # Procesirani podaci i test skup
models/            # Istrenirani modeli, scaler.pkl i best_thresholds.json
results/           # Metrike (.txt) i grafici (.png)
src/               # duplicate_analysis, data_preparation, eda_analysis, train, evaluate, feature_selection
app/app.py         # Streamlit aplikacija
pipeline.py        # Pokreće ceo tok od početka do kraja
```

## Grane

- **`main`** — koristi svih 30 atributa.
- **`18_atributa`** — koristi 18 najznačajnijih atributa (izabranih pravilom 1-SE). Poređenje pokazuje da kod ansambala smanjenje sa 30 na 18 atributa praktično ne šteti kvalitetu, uz jednostavniji model.
