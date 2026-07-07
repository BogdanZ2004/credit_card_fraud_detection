import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')  # crtanje u fajl bez ekrana (bezbedno na serveru)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix
)

def select_best_model_on_validation(val_data_path, models_dir, metrics_dir):
    # Poredi sve istrenirane modele na validacionom skupu i bira najbolji (po AUPRC)
    print("Izbor najboljeg modela na validacionom skupu...")
    df = pd.read_csv(val_data_path)
    X_val = df.drop('Class', axis=1)
    y_val = df['Class']

    os.makedirs(metrics_dir, exist_ok=True)
    results_file = os.path.join(metrics_dir, 'validation_selection.txt')

    rezultati = []
    for model_file in sorted(os.listdir(models_dir)):
        if not model_file.endswith('.pkl') or model_file == 'scaler.pkl':
            continue
        model_name = model_file.replace('.pkl', '')
        model = joblib.load(os.path.join(models_dir, model_file))
        y_proba = model.predict_proba(X_val)[:, 1]
        pr_precision, pr_recall, _ = precision_recall_curve(y_val, y_proba)
        auprc = auc(pr_recall, pr_precision)
        rezultati.append((model_name, auprc))

    # Najbolji je onaj sa najvišim AUPRC na validaciji
    rezultati.sort(key=lambda x: x[1], reverse=True)
    best_name, best_auprc = rezultati[0]

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=== IZBOR MODELA NA VALIDACIONOM SKUPU (po AUPRC) ===\n")
        f.write("Model se bira ovde; finalni rezultat se prijavljuje na test skupu.\n\n")
        for name, auprc in rezultati:
            oznaka = "  <-- NAJBOLJI" if name == best_name else ""
            f.write(f"{name}: AUPRC = {auprc:.4f}{oznaka}\n")
        f.write(f"\nIzabrani model: {best_name} (AUPRC = {best_auprc:.4f})\n")

    print(f"   Najbolji model na validaciji: {best_name} (AUPRC = {best_auprc:.4f})")
    print(f"   Sačuvano u: {results_file}")
    return best_name


def evaluate_models(test_data_path, models_dir, figures_dir, metrics_dir):
    print("1. Učitavanje Test seta...")
    df = pd.read_csv(test_data_path)
    X_test = df.drop('Class', axis=1)
    y_test = df['Class']

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_file_path = os.path.join(metrics_dir, 'model_comparison.txt')

    with open(metrics_file_path, 'w', encoding='utf-8') as f:
        f.write("=== POREĐENJE MODELA ZA DETEKCIJU PREVARA ===\n\n")

    print("\n2. Evaluacija modela...")
    # Liste za zajedničke grafike (ROC kriva, PR kriva, stubičasto poređenje)
    roc_curves = []
    pr_curves = []
    summary = []

    for model_file in sorted(os.listdir(models_dir)):
        # Preskačemo scaler.pkl jer nije model za klasifikaciju
        if not model_file.endswith('.pkl') or model_file == 'scaler.pkl':
            continue

        model_name = model_file.replace('.pkl', '')
        print(f"\n   -> Testiram model: {model_name}")

        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Računanje svih metrika na test skupu
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)
        auprc = auc(pr_recall, pr_precision)
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves.append((model_name, fpr, tpr, roc_auc))
        pr_curves.append((model_name, pr_recall, pr_precision, auprc))
        summary.append((model_name, precision, recall, f1, auprc))

        result_text = (
            f"--- {model_name} ---\n"
            f"Preciznost (Precision): {precision:.4f}\n"
            f"Odziv (Recall):         {recall:.4f}  <-- NAJBITNIJE\n"
            f"F1-skor:                {f1:.4f}\n"
            f"ROC AUC:                {roc_auc:.4f}\n"
            f"AUPRC:                  {auprc:.4f}\n\n"
        )
        print(result_text)

        with open(metrics_file_path, 'a', encoding='utf-8') as f:
            f.write(result_text)

        # Matrica konfuzije prikazuje tačne i netačne predikcije po klasi
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Legitimne (0)', 'Prevare (1)'],
                    yticklabels=['Legitimne (0)', 'Prevare (1)'])
        plt.title(f'Matrica konfuzije - {model_name}')
        plt.ylabel('Stvarna klasa')
        plt.xlabel('Predviđena klasa')

        cm_fig_path = os.path.join(figures_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(cm_fig_path, bbox_inches='tight')
        plt.close()
        print(f"      [Matrica konfuzije sačuvana]")

    # Zajednička ROC kriva za sve modele radi lakšeg poređenja
    plt.figure(figsize=(8, 6))
    for model_name, fpr, tpr, roc_auc in roc_curves:
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC kriva - Poređenje modela')
    plt.legend(loc='lower right')
    roc_fig_path = os.path.join(figures_dir, 'roc_curve.png')
    plt.savefig(roc_fig_path, bbox_inches='tight')
    plt.close()
    print(f"\nROC kriva sačuvana u: {roc_fig_path}")

    # Zajednička Precision-Recall kriva — prikladnija od ROC-a za neuravnotežene podatke (vizualizuje AUPRC)
    plt.figure(figsize=(8, 6))
    for model_name, rec, prec, auprc in pr_curves:
        plt.plot(rec, prec, label=f"{model_name} (AUPRC = {auprc:.4f})")
    baseline = y_test.mean()  # nasumičan klasifikator = udeo prevara
    plt.axhline(baseline, color='k', linestyle='--', label=f'Random ({baseline:.4f})')
    plt.xlabel('Odziv (Recall)')
    plt.ylabel('Preciznost (Precision)')
    plt.title('Precision-Recall kriva - Poređenje modela')
    plt.legend(loc='upper right')
    pr_fig_path = os.path.join(figures_dir, 'pr_curve.png')
    plt.savefig(pr_fig_path, bbox_inches='tight')
    plt.close()
    print(f"PR kriva sačuvana u: {pr_fig_path}")

    # Stubičasti grafik: poređenje modela po ključnim metrikama (za dokumentaciju)
    names = [s[0] for s in summary]
    x = np.arange(len(names))
    w = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - w, [s[4] for s in summary], w, label='AUPRC')
    plt.bar(x,     [s[3] for s in summary], w, label='F1')
    plt.bar(x + w, [s[2] for s in summary], w, label='Odziv')
    plt.xticks(x, names, rotation=20, ha='right')
    plt.ylabel('Vrednost')
    plt.ylim(0, 1)
    plt.title('Poređenje modela po ključnim metrikama (test skup)')
    plt.legend()
    bar_fig_path = os.path.join(figures_dir, 'model_comparison_chart.png')
    plt.savefig(bar_fig_path, bbox_inches='tight')
    plt.close()
    print(f"Grafik poređenja sačuvan u: {bar_fig_path}")
    print(f"Sve metrike su sačuvane u: {metrics_file_path}")

if __name__ == "__main__":
    VAL_FILE = "../data/processed/val_set.csv"
    TEST_FILE = "../data/processed/test_set.csv"
    MODELS_DIR = "../models"
    FIGURES_DIR = "../results/figures"
    METRICS_DIR = "../results/metrics"

    select_best_model_on_validation(VAL_FILE, MODELS_DIR, METRICS_DIR)
    evaluate_models(TEST_FILE, MODELS_DIR, FIGURES_DIR, METRICS_DIR)
