import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')  # crtanje u fajl bez ekrana (bezbedno na serveru)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier


def exhaustive_feature_selection(processed_data_path, original_model_path, metrics_dir,
                                 figures_dir, n_splits=5, n_estimators=100):
    """Selekcija broja atributa: AUPRC kroz 5-fold CV + pravilo 1-SE, samo na trening skupu."""
    print("1. Učitavanje podataka i originalnog modela...")
    df = pd.read_csv(processed_data_path)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Ista podela kao u train.py; selekcija se radi samo na trening skupu (test se ne dira)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )

    # X_train ostaje neskaliran; skaliranje ide unutar svakog CV folda (bez curenja)
    X_train = X_train.copy()

    try:
        original_rf = joblib.load(original_model_path)
    except FileNotFoundError:
        print(f"Greška: Nije pronađen model na putanji {original_model_path}")
        return

    # Rang atributa po važnosti iz istreniranog RF-a; imena iz modela (feature_names_in_).
    # 'Scaled_Amount' iz modela odgovara sirovoj koloni 'Amount' u X_train-u.
    importances = original_rf.feature_importances_
    feature_df = pd.DataFrame({
        'Atribut': list(original_rf.feature_names_in_),
        'Značaj': importances
    }).sort_values(by='Značaj', ascending=False).reset_index(drop=True)

    ranked_features = feature_df['Atribut'].tolist()
    n_features = len(ranked_features)

    # SMOTE unutar svakog folda (bez curenja); metrika AUPRC (average_precision)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"\n2. Sweep top-1 .. top-{n_features} atributa "
          f"({n_splits}-fold CV, AUPRC, skaler + SMOTE unutar folda).")
    print(f"   UPOZORENJE: trenira se {n_splits} × {n_features} = {n_splits * n_features} "
          f"modela — potrajaće!\n")

    rezultati = []
    for k in range(1, n_features + 1):
        top_k = ranked_features[:k]
        # 'Scaled_Amount' iz ranga predstavlja sirovi 'Amount' u X_train-u
        raw_cols = ['Amount' if c == 'Scaled_Amount' else c for c in top_k]
        X_k = X_train[raw_cols]

        # Skaliranje 'Amount' unutar folda ako je u podskupu; RF n_jobs=1 (foldovi se paralelizuju)
        steps = []
        if 'Amount' in raw_cols:
            steps.append(('scaler', ColumnTransformer(
                [('amount', RobustScaler(), ['Amount'])], remainder='passthrough')))
        steps.append(('smote', SMOTE(random_state=42)))
        steps.append(('model', RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=1)))
        pipe = ImbPipeline(steps)
        scores = cross_val_score(pipe, X_k, y_train, cv=cv, scoring='average_precision', n_jobs=-1)

        mean = float(scores.mean())
        # Standardna greška srednje vrednosti = std(foldova) / sqrt(broj foldova)
        se = float(scores.std(ddof=1) / np.sqrt(len(scores)))
        rezultati.append({'k': k, 'mean_auprc': mean, 'se': se, 'kolone': top_k})
        print(f"   -> top {k:2d}: AUPRC = {mean:.4f} ± {se:.4f}")

    means = np.array([r['mean_auprc'] for r in rezultati])
    ses = np.array([r['se'] for r in rezultati])

    # (a) k sa maksimalnim prosečnim AUPRC
    best_idx = int(np.argmax(means))
    best_k = rezultati[best_idx]['k']
    best_mean = float(means[best_idx])
    best_se = float(ses[best_idx])

    # (b) Pravilo 1-SE: najmanji k čiji AUPRC nije lošiji od (najbolji − 1 SE) -> parsimonija
    prag_1se = best_mean - best_se
    onese_idx = int(np.argmax(means >= prag_1se))  # prvi (najmanji) k iznad praga
    onese_k = rezultati[onese_idx]['k']
    onese_cols = rezultati[onese_idx]['kolone']

    # --- Grafik: AUPRC vs broj atributa, sa pragom 1-SE i oba izbora ---
    os.makedirs(figures_dir, exist_ok=True)
    ks = [r['k'] for r in rezultati]
    plt.figure(figsize=(10, 6))
    plt.errorbar(ks, means, yerr=ses, marker='o', capsize=3, label='AUPRC (CV, ±1 SE)')
    plt.axhline(prag_1se, color='gray', linestyle='--', label=f'Prag 1-SE ({prag_1se:.4f})')
    plt.axvline(best_k, color='green', linestyle=':', label=f'Maks. AUPRC (k={best_k})')
    plt.axvline(onese_k, color='red', linestyle=':', label=f'Izbor 1-SE (k={onese_k})')
    plt.xlabel('Broj atributa (top-k po važnosti)')
    plt.ylabel('AUPRC (5-fold CV na trening skupu)')
    plt.title('AUPRC vs broj atributa — izbor po pravilu 1 standardne greške')
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = os.path.join(figures_dir, 'feature_selection_auprc_vs_k.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # --- Izveštaj ---
    os.makedirs(metrics_dir, exist_ok=True)
    fajl_putanja = os.path.join(metrics_dir, 'rang_lista_atributa.txt')
    with open(fajl_putanja, 'w', encoding='utf-8') as f:
        f.write("=== SELEKCIJA ATRIBUTA (AUPRC, 5-fold CV na treningu, pravilo 1-SE) ===\n")
        f.write("Rang je iz istreniranog RandomForest-a; ocena po AUPRC uz SMOTE unutar folda.\n")
        f.write("Test skup se NE koristi — izbor je isključivo na trening skupu (CV).\n\n")

        f.write("--- Rang atributa po važnosti ---\n")
        for i, row in feature_df.iterrows():
            f.write(f"{i + 1:2d}. {row['Atribut']:15s}  značaj = {row['Značaj']:.4f}\n")

        f.write("\n--- AUPRC po broju atributa (mean ± SE) ---\n")
        for r in rezultati:
            oznaka = ""
            if r['k'] == best_k:
                oznaka += "  <-- maks. AUPRC"
            if r['k'] == onese_k:
                oznaka += "  <-- izbor 1-SE"
            f.write(f"k={r['k']:2d}: {r['mean_auprc']:.4f} ± {r['se']:.4f}{oznaka}\n")

        f.write(f"\nNajbolji prosečni AUPRC: k={best_k} ({best_mean:.4f} ± {best_se:.4f})\n")
        f.write(f"Prag 1-SE: {prag_1se:.4f}  (= najbolji AUPRC − 1 SE)\n")
        f.write(f"Izbor po pravilu 1-SE: k={onese_k} atributa\n")
        f.write(f"Izabrani atributi: {', '.join(onese_cols)}\n\n")
        f.write("Tumačenje: k(1-SE) je najmanji broj atributa čiji je CV AUPRC statistički\n")
        f.write("neizrazliv (unutar 1 standardne greške) od najboljeg. To je predlog za\n")
        f.write("sekundarnu granu; finalni modeli se tamo treniraju na ovih k atributa i\n")
        f.write("tek onda jednom ocenjuju na test skupu.\n")

    print(f"\n3. Najbolji AUPRC: k={best_k} ({best_mean:.4f} ± {best_se:.4f})")
    print(f"   Izbor po pravilu 1-SE: k={onese_k} atributa")
    print(f"   Izabrani atributi: {', '.join(onese_cols)}")
    print(f"   Rang lista: {fajl_putanja}")
    print(f"   Grafik:     {plot_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH      = os.path.join(BASE_DIR, "data", "processed", "creditcard_processed.csv")
    ORIGINAL_MODEL = os.path.join(BASE_DIR, "models", "RandomForest_100.pkl")
    METRICS_DIR    = os.path.join(BASE_DIR, "results", "metrics")
    FIGURES_DIR    = os.path.join(BASE_DIR, "results", "figures")

    exhaustive_feature_selection(DATA_PATH, ORIGINAL_MODEL, METRICS_DIR, FIGURES_DIR)
