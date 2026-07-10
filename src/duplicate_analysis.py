import pandas as pd
import os


def analyze_duplicates(raw_path, metrics_dir):
    """Analiza tačnih duplikata u sirovim podacima: da li su češće prevare (dijagnostika, ništa se ne menja)."""
    print("Učitavanje sirovih podataka...")
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"Greška: Fajl {raw_path} nije pronađen.")
        return

    os.makedirs(metrics_dir, exist_ok=True)
    report_path = os.path.join(metrics_dir, 'duplicate_analysis.txt')

    ukupno = len(df)
    ukupno_prevara = int(df['Class'].sum())
    stopa_prevara = ukupno_prevara / ukupno  # udeo prevara u CELOM skupu (bazna stopa)

    # SVI redovi koji učestvuju u nekoj duplikat-grupi (keep=False označava i original i kopije)
    maska_svi = df.duplicated(keep=False)
    dup_svi = df[maska_svi]

    # Grupišemo po svim kolonama da dobijemo veličinu svake duplikat-grupe
    velicine = dup_svi.groupby(list(df.columns), sort=False).size().reset_index(name='n')
    fraud_grupe = velicine[velicine['Class'] == 1]
    legit_grupe = velicine[velicine['Class'] == 0]

    broj_grupa = len(velicine)
    broj_fraud_grupa = len(fraud_grupe)
    broj_legit_grupa = len(legit_grupe)

    # "Suvišne kopije" = ono što drop_duplicates() obriše (n-1 po grupi)
    visak_ukupno = int((velicine['n'] - 1).sum())
    visak_prevara = int((fraud_grupe['n'] - 1).sum())
    visak_legit = int((legit_grupe['n'] - 1).sum())

    # Prevare među SVIM redovima u duplikat-grupama (original + kopije)
    svi_redova = int(velicine['n'].sum())
    svi_prevara = int((fraud_grupe['n']).sum())
    stopa_dup = (svi_prevara / svi_redova) if svi_redova else 0.0
    # Koliko su duplikati "obogaćeni" prevarama u odnosu na baznu stopu
    obogacenje = (stopa_dup / stopa_prevara) if stopa_prevara else 0.0

    najveca_fraud = int(fraud_grupe['n'].max()) if broj_fraud_grupa else 0
    najveca_legit = int(legit_grupe['n'].max()) if broj_legit_grupa else 0

    # Verdikt u ljudskom obliku
    if obogacenje >= 2:
        verdikt = (f"Duplikati SU obogaćeni prevarama ({obogacenje:.1f}x iznad bazne stope) — "
                   f"ide u prilog hipotezi da ponovljene transakcije nose signal prevare.")
    elif obogacenje >= 1.2:
        verdikt = (f"Duplikati su blago obogaćeni prevarama ({obogacenje:.1f}x) — slab signal.")
    else:
        verdikt = (f"Duplikati NISU obogaćeni prevarama ({obogacenje:.1f}x) — verovatnije su "
                   f"artefakti/legitimne ponovljene naplate nego signal prevare.")

    lines = []
    lines.append("=== ANALIZA DUPLIKATA (sirovi podaci, PRE uklanjanja) ===")
    lines.append("Duplikat = red identičan po SVIM kolonama (Time, V1-V28, Amount, Class).\n")
    lines.append(f"Ukupno redova:                          {ukupno}")
    lines.append(f"Ukupno prevara (Class=1):               {ukupno_prevara}  ({stopa_prevara * 100:.4f}%)\n")

    lines.append("--- Obim duplikata ---")
    lines.append(f"Redova u duplikat-grupama (orig+kopije): {svi_redova}")
    lines.append(f"Suvišnih kopija (obrisao bi drop_duplicates): {visak_ukupno}")
    lines.append(f"Broj jedinstvenih duplikat-grupa:       {broj_grupa}\n")

    lines.append("--- Šta se KONKRETNO briše (suvišne kopije) ---")
    lines.append(f"Ukupno obrisanih redova:                {visak_ukupno}")
    lines.append(f"   -> prevare (Class=1):                {visak_prevara}")
    lines.append(f"   -> legitimne (Class=0):              {visak_legit}\n")

    lines.append("--- Da li su duplikati prevare? ---")
    lines.append(f"Stopa prevara MEĐU duplikatima:         {stopa_dup * 100:.4f}%")
    lines.append(f"Stopa prevara u celom skupu (bazna):    {stopa_prevara * 100:.4f}%")
    lines.append(f"Obogaćenje (enrichment):                {obogacenje:.2f}x\n")

    lines.append("--- Duplikat-grupe po klasi ---")
    lines.append(f"Grupe koje su prevara:                  {broj_fraud_grupa}  (najveća ima {najveca_fraud} kopija)")
    lines.append(f"Grupe koje su legitimne:                {broj_legit_grupa}  (najveća ima {najveca_legit} kopija)\n")

    lines.append("--- Zaključak ---")
    lines.append(verdikt)
    lines.append("")
    lines.append("NAPOMENA (curenje podataka): nezavisno od gornjeg, zadržavanje TAČNIH duplikata")
    lines.append("uz nasumičnu podelu train/test znači da IDENTIČAN red može završiti i u treningu")
    lines.append("i u testu — model ga 'zapamti', pa su test metrike naduvane. Ako se duplikati")
    lines.append("zadrže kao signal, podelu treba raditi tako da identični redovi ne prelaze granicu")
    lines.append("train/test (a to je teško bez ID-a kartice/transakcije u ovom skupu).")

    report = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report + "\n")

    print("\n" + report)
    print(f"\nIzveštaj sačuvan u: {report_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
    METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")

    analyze_duplicates(RAW_DATA, METRICS_DIR)
