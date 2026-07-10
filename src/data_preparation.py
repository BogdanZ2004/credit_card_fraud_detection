import pandas as pd
import os

def prepare_data(input_path, output_path):
    print("Učitavanje sirovih podataka...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Greška: Fajl {input_path} nije pronađen. Proveri putanju!")
        return

    # Tačni duplikati se brišu na sirovim podacima da isti primer ne završi i u treningu i u testu
    pre = len(df)
    df = df.drop_duplicates()
    print(f"1. Uklonjeno {pre - len(df)} tačnih duplikata (ostalo {len(df)} redova)")

    print("2. Transformacija 'Time' kolone u 'Hour'...")
    # Sekunde pretvaramo u sat u toku dana jer je doba dana bitno za detekciju prevare
    df['Hour'] = (df['Time'] // 3600) % 24
    df = df.drop(['Time'], axis=1)

    # Skaliranje Amount-a se radi u train.py nakon podele kako bi se izbeglo curenje podataka

    print("3. Reorganizacija kolona...")
    # Hour i Amount idu na početak, Class na kraj
    cols = ['Hour', 'Amount'] + [col for col in df.columns if col not in ['Hour', 'Amount', 'Class']] + ['Class']
    df = df[cols]

    print("4. Čuvanje procesuiranih podataka...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Gotovo! Podaci su uspešno sačuvani na: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "../data/raw/creditcard.csv"
    OUTPUT_FILE = "../data/processed/creditcard_processed.csv"

    prepare_data(INPUT_FILE, OUTPUT_FILE)
