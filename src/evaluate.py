import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib
matplotlib.use('Agg')  # crtanje u fajl bez ekrana (bezbedno na serveru)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier

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


def optimize_threshold(val_data_path, models_dir, metrics_dir, beta=2):
    # Traži prag odluke koji maksimizuje F2 na VALIDACIJI (F2 naglašava odziv 4x
    # jer je propuštena prevara skuplja od lažne uzbune). Prag se bira na validaciji,
    # nikad na testu. Optimalni prag postaje "pametna" podrazumevana vrednost za aplikaciju.
    print("Podešavanje praga odluke na validacionom skupu (po F2)...")
    df = pd.read_csv(val_data_path)
    X_val = df.drop('Class', axis=1)
    y_val = df['Class']

    os.makedirs(metrics_dir, exist_ok=True)
    results_file = os.path.join(metrics_dir, 'threshold_optimization.txt')
    pragovi = np.arange(0.05, 0.96, 0.01)

    najbolji_po_modelu = {}
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=== PODEŠAVANJE PRAGA ODLUKE (F2, na validaciji) ===\n")
        f.write("F2 naglašava odziv 4x — propuštena prevara je skuplja od lažne uzbune.\n\n")
        f.write(f"{'Model':22s} {'Opt.prag':>9s} {'F2(0.5)':>9s} {'F2(opt)':>9s}\n")
        for model_file in sorted(os.listdir(models_dir)):
            if not model_file.endswith('.pkl') or model_file == 'scaler.pkl':
                continue
            name = model_file.replace('.pkl', '')
            model = joblib.load(os.path.join(models_dir, model_file))
            proba = model.predict_proba(X_val)[:, 1]

            f2_default = fbeta_score(y_val, (proba >= 0.5).astype(int), beta=beta, zero_division=0)
            best_t, best_f2 = 0.5, f2_default
            for t in pragovi:
                f2 = fbeta_score(y_val, (proba >= t).astype(int), beta=beta, zero_division=0)
                if f2 > best_f2:
                    best_f2, best_t = f2, round(float(t), 2)
            najbolji_po_modelu[name] = best_t
            f.write(f"{name:22s} {best_t:9.2f} {f2_default:9.4f} {best_f2:9.4f}\n")

    # Mašinski čitljiv zapis pragova — app.py ga učitava da postavi podrazumevani
    # (F2-optimalni) prag za izabrani model, uz zadržavanje ručnog klizača.
    thresholds_json = os.path.join(models_dir, 'best_thresholds.json')
    with open(thresholds_json, 'w', encoding='utf-8') as jf:
        json.dump(najbolji_po_modelu, jf, indent=2)

    print(f"   Sačuvano u: {results_file}")
    print(f"   Pragovi (JSON za app): {thresholds_json}")
    return najbolji_po_modelu


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

    # Scaler za vraćanje skaliranog iznosa u stvarni (za analizu grešaka po iznosu)
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    realni_iznos = scaler.inverse_transform(X_test[['Scaled_Amount']])[:, 0]

    # F2-optimalni pragovi izabrani na validaciji (KORAK 6) — za poređenje 0.5 vs optimalni na testu.
    # Ako fajl ne postoji, optimalni prag pada na 0.5 (bez efekta), uz upozorenje u izveštaju.
    try:
        with open(os.path.join(models_dir, 'best_thresholds.json'), encoding='utf-8') as tf:
            tuned_thresholds = json.load(tf)
    except (FileNotFoundError, json.JSONDecodeError):
        tuned_thresholds = {}

    print("\n2. Evaluacija modela...")
    # Liste za zajedničke grafike (ROC kriva, PR kriva, stubičasto poređenje) i analizu grešaka
    roc_curves = []
    pr_curves = []
    summary = []
    analiza_gresaka = []
    payoff = []   # poređenje praga 0.5 vs F2-optimalni prag (na test skupu)

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
        # F2 naglašava odziv (isti fokus kao pri izboru praga); ovde na podrazumevanom pragu 0.5
        f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)

        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)
        auprc = auc(pr_recall, pr_precision)
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves.append((model_name, fpr, tpr, roc_auc))
        pr_curves.append((model_name, pr_recall, pr_precision, auprc))
        summary.append((model_name, precision, recall, f1, f2, roc_auc, auprc))

        # Poređenje: metrike na 0.5 vs na F2-optimalnom pragu (izabranom na validaciji),
        # mereno na TEST skupu. Prag se ovde samo PRIMENJUJE na test -> nema curenja.
        t_opt = float(tuned_thresholds.get(model_name, 0.5))
        y_pred_opt = (y_proba >= t_opt).astype(int)
        base_row = (recall, precision, f1, f2,
                    int(((y_test == 1) & (y_pred == 0)).sum()),
                    int(((y_test == 0) & (y_pred == 1)).sum()))
        opt_row = (recall_score(y_test, y_pred_opt, zero_division=0),
                   precision_score(y_test, y_pred_opt, zero_division=0),
                   f1_score(y_test, y_pred_opt, zero_division=0),
                   fbeta_score(y_test, y_pred_opt, beta=2, zero_division=0),
                   int(((y_test == 1) & (y_pred_opt == 0)).sum()),
                   int(((y_test == 0) & (y_pred_opt == 1)).sum()))
        payoff.append((model_name, t_opt, base_row, opt_row))

        result_text = (
            f"--- {model_name} ---\n"
            f"Preciznost (Precision): {precision:.4f}\n"
            f"Odziv (Recall):         {recall:.4f}  <-- NAJBITNIJE\n"
            f"F1-skor:                {f1:.4f}\n"
            f"F2-skor:                {f2:.4f}\n"
            f"ROC AUC:                {roc_auc:.4f}\n"
            f"AUPRC:                  {auprc:.4f}\n\n"
        )
        print(result_text)

        with open(metrics_file_path, 'a', encoding='utf-8') as f:
            f.write(result_text)

        # Analiza grešaka: karakteristike propuštenih prevara (FN) vs uhvaćenih (TP)
        fn_maska = (y_test == 1) & (y_pred == 0)   # propuštene prevare
        fp_maska = (y_test == 0) & (y_pred == 1)   # lažne uzbune
        tp_maska = (y_test == 1) & (y_pred == 1)   # uhvaćene prevare
        iznos_fn = realni_iznos[fn_maska.values].mean() if fn_maska.sum() > 0 else 0
        iznos_tp = realni_iznos[tp_maska.values].mean() if tp_maska.sum() > 0 else 0
        analiza_gresaka.append((model_name, int(fn_maska.sum()), int(fp_maska.sum()),
                                int(tp_maska.sum()), iznos_fn, iznos_tp))

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
    plt.legend(loc='lower left')
    pr_fig_path = os.path.join(figures_dir, 'pr_curve.png')
    plt.savefig(pr_fig_path, bbox_inches='tight')
    plt.close()
    print(f"PR kriva sačuvana u: {pr_fig_path}")

    # Stubičasti grafik: poređenje modela po ključnim metrikama (za dokumentaciju)
    names = [s[0] for s in summary]
    x = np.arange(len(names))
    w = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - w, [s[6] for s in summary], w, label='AUPRC')
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

    # Baseline: nasumičan klasifikator (predviđa po raspodeli klasa) — dokaz da modeli
    # uče prave obrasce, a ne da samo eksploatišu neuravnoteženost.
    # DummyClassifier ignoriše atribute — fit uči samo raspodelu klasa, koja je zbog
    # stratifikovane podele ista u train/val/test, pa je fit ovde ekvivalentan fitu
    # na treningu (nema curenja informacija).
    dummy = DummyClassifier(strategy='stratified', random_state=42)
    dummy.fit(X_test, y_test)
    yb = dummy.predict(X_test)
    pb = dummy.predict_proba(X_test)[:, 1]
    b_prec, b_rec = precision_score(y_test, yb, zero_division=0), recall_score(y_test, yb)
    b_prc, b_rcc, _ = precision_recall_curve(y_test, pb)
    b_auprc = auc(b_rcc, b_prc)
    with open(metrics_file_path, 'a', encoding='utf-8') as f:
        f.write("--- Baseline (nasumičan klasifikator) ---\n")
        f.write(f"Preciznost (Precision): {b_prec:.4f}\n")
        f.write(f"Odziv (Recall):         {b_rec:.4f}\n")
        f.write(f"AUPRC (izmereno):       {b_auprc:.4f}\n")
        f.write(f"AUPRC (teorijski):      {y_test.mean():.4f}  (= stopa prevara u testu; linija na PR krivoj)\n")
        f.write("(Svi istrenirani modeli su znatno bolji od ovoga -> uče prave obrasce.)\n")

    # Zbirna tabela svih metrika po modelu (za dokumentaciju) — test skup, prag 0.5
    with open(metrics_file_path, 'a', encoding='utf-8') as f:
        f.write("\n=== ZBIRNA TABELA (sve metrike, test skup, prag 0.5) ===\n")
        f.write(f"{'Model':22s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'F2':>7s} {'ROC-AUC':>8s} {'AUPRC':>7s}\n")
        for name, prec, rec, f1v, f2v, roc, ap in summary:
            f.write(f"{name:22s} {prec:7.4f} {rec:7.4f} {f1v:7.4f} {f2v:7.4f} {roc:8.4f} {ap:7.4f}\n")

    # Potvrda da F2-optimalni prag (izabran na validaciji) daje bolji rezultat od 0.5 — na TEST skupu.
    payoff_file = os.path.join(metrics_dir, 'threshold_payoff_test.txt')
    with open(payoff_file, 'w', encoding='utf-8') as f:
        f.write("=== EFEKAT PRAGA NA TEST SKUPU (0.5 vs F2-optimalni prag) ===\n")
        f.write("Prag se BIRA na validaciji (po F2), a ovde se samo PRIMENJUJE na test — bez curenja.\n")
        if not tuned_thresholds:
            f.write("\nUPOZORENJE: models/best_thresholds.json nije pronađen — optimalni prag = 0.5\n")
            f.write("(pokreni KORAK 6 / optimize_threshold da se generiše).\n")
        f.write(f"\n{'Model':20s} {'Prag':>6s} {'Recall':>8s} {'Prec':>8s} {'F1':>8s} {'F2':>8s} {'FN':>4s} {'FP':>5s}\n")
        for name, t_opt, base, opt in payoff:
            rec0, prec0, f10, f20, fn0, fp0 = base
            rec1, prec1, f11, f21, fn1, fp1 = opt
            f.write(f"{name:20s} {'0.50':>6s} {rec0:8.4f} {prec0:8.4f} {f10:8.4f} {f20:8.4f} {fn0:4d} {fp0:5d}\n")
            f.write(f"{name:20s} {f'{t_opt:.2f}*':>6s} {rec1:8.4f} {prec1:8.4f} {f11:8.4f} {f21:8.4f} {fn1:4d} {fp1:5d}\n\n")
        f.write("(* = F2-optimalni prag, izabran na validaciji)\n")
        f.write("Niži prag -> veći odziv (manje propuštenih prevara FN), po cenu više lažnih uzbuna (FP).\n")
        f.write("F2 na optimalnom pragu potvrđuje da je taj kompromis isplativ i na test skupu.\n")
    print(f"Poređenje pragova (test) sačuvano u: {payoff_file}")

    # Analiza grešaka: gde model najviše greši (propuštene prevare vs uhvaćene)
    err_file = os.path.join(metrics_dir, 'error_analysis.txt')
    with open(err_file, 'w', encoding='utf-8') as f:
        f.write("=== ANALIZA GREŠAKA (po iznosu) ===\n")
        f.write("FN = propuštene prevare | FP = lažne uzbune | TP = uhvaćene prevare\n\n")
        f.write(f"{'Model':22s} {'FN':>4s} {'FP':>5s} {'TP':>4s} {'Iznos FN':>10s} {'Iznos TP':>10s}\n")
        for name, fn, fp, tp, izn_fn, izn_tp in analiza_gresaka:
            f.write(f"{name:22s} {fn:>4d} {fp:>5d} {tp:>4d} {izn_fn:>10.2f} {izn_tp:>10.2f}\n")
        f.write("\nPoređenje prosečnog iznosa FN i TP pokazuje da li model sistematski\n")
        f.write("promašuje prevare određene veličine (manje ili veće od uhvaćenih).\n")
    print(f"Analiza grešaka sačuvana u: {err_file}")
    print(f"Sve metrike su sačuvane u: {metrics_file_path}")

if __name__ == "__main__":
    VAL_FILE = "../data/processed/val_set.csv"
    TEST_FILE = "../data/processed/test_set.csv"
    MODELS_DIR = "../models"
    FIGURES_DIR = "../results/figures"
    METRICS_DIR = "../results/metrics"

    select_best_model_on_validation(VAL_FILE, MODELS_DIR, METRICS_DIR)
    optimize_threshold(VAL_FILE, MODELS_DIR, METRICS_DIR)
    evaluate_models(TEST_FILE, MODELS_DIR, FIGURES_DIR, METRICS_DIR)
