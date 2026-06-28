import pandas as pd
import os
import joblib
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
    roc_curves = []

    for model_file in sorted(os.listdir(models_dir)):
        if not model_file.endswith('.pkl') or model_file == 'scaler.pkl':
            continue
            
        model_name = model_file.replace('.pkl', '')
        print(f"\n   -> Testiram model: {model_name}")
        
        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Računanje metrika
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)
        auprc = auc(pr_recall, pr_precision)
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves.append((model_name, fpr, tpr, roc_auc))

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

        # Matrica konfuzije
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
    print(f"Sve metrike su sačuvane u: {metrics_file_path}")

if __name__ == "__main__":
    TEST_FILE = "../data/processed/test_set.csv"
    MODELS_DIR = "../models"
    FIGURES_DIR = "../results/figures"
    METRICS_DIR = "../results/metrics"
    
    evaluate_models(TEST_FILE, MODELS_DIR, FIGURES_DIR, METRICS_DIR)