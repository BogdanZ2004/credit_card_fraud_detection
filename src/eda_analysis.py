import pandas as pd
import matplotlib
matplotlib.use('Agg')  # crtanje u fajl bez ekrana (bezbedno na serveru)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_path, figures_dir, metrics_dir):
    print("Učitavanje podataka za analizu...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Greška: Fajl {input_path} nije pronađen.")
        return

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Provera kvaliteta podataka: nedostajuće vrednosti i duplikati
    print("0. Provera kvaliteta podataka (nedostajuće vrednosti, duplikati)...")
    ukupno = len(df)
    broj_prevara = int(df['Class'].sum())
    nedostajuce = int(df.isnull().sum().sum())
    duplikati = int(df.duplicated().sum())

    summary_path = os.path.join(metrics_dir, 'eda_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== EDA SAŽETAK ===\n\n")
        f.write(f"Ukupno transakcija:       {ukupno}\n")
        f.write(f"Legitimne (0):            {ukupno - broj_prevara}\n")
        f.write(f"Prevare (1):              {broj_prevara} ({broj_prevara / ukupno * 100:.3f}%)\n")
        f.write(f"Nedostajuće vrednosti:    {nedostajuce}\n")
        f.write(f"Duplikati (redovi):       {duplikati}\n")
    print(f"   Nedostajuće vrednosti: {nedostajuce} | Duplikati: {duplikati}")
    print(f"   Sažetak sačuvan u: {summary_path}")

    # Grafik koji prikazuje koliko ima legitimnih a koliko prevarantskih transakcija
    print("1. Generisanje grafika raspodele klasa...")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='Class', hue='Class', data=df, palette='Set2', legend=False)
    plt.title('Raspodela klasa (0: Legitimne, 1: Prevare)', fontsize=14)
    # Logaritamska skala zbog ogromne razlike između klasa
    plt.yscale('log')
    plt.ylabel('Broj transakcija (Log skala)')

    # Tačan broj transakcija iznad svake kolone
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + 0.35, p.get_height() + 50))

    class_fig_path = os.path.join(figures_dir, 'class_distribution.png')
    plt.savefig(class_fig_path, bbox_inches='tight')
    plt.close()

    fraud_pct = (df['Class'].value_counts()[1] / len(df)) * 100
    print(f"   -> Procenat prevara: {fraud_pct:.3f}%")
    print(f"   -> Slika sačuvana u: {class_fig_path}")

    # Matrica korelacije koja pokazuje koje varijable su međusobno povezane
    print("\n2. Generisanje matrice korelacije (ovo može potrajati par sekundi)...")
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap='coolwarm_r', annot=False, fmt='.2f')
    plt.title('Matrica korelacije svih atributa', fontsize=16)

    corr_fig_path = os.path.join(figures_dir, 'correlation_matrix.png')
    plt.savefig(corr_fig_path, bbox_inches='tight')
    plt.close()
    print(f"   -> Slika sačuvana u: {corr_fig_path}")

    # Top 10 atributa po apsolutnoj korelaciji sa ciljnom varijablom Class
    print("\n==================================================")
    print("Atributi koji imaju najveći uticaj na detekciju prevare:")
    print("==================================================")
    korelacija_sa_klasom = corr['Class'].drop('Class').abs().sort_values(ascending=False)
    print(korelacija_sa_klasom.head(10))

if __name__ == "__main__":
    INPUT_FILE = "../data/processed/creditcard_processed.csv"
    FIGURES_DIR = "../results/figures"
    METRICS_DIR = "../results/metrics"

    run_eda(INPUT_FILE, FIGURES_DIR, METRICS_DIR)
