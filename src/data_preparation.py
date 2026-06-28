import pandas as pd
import os

def prepare_data(input_path, output_path):
    print("Učitavanje sirovih podataka...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Greška: Fajl {input_path} nije pronađen. Proveri putanju!")
        return

    print("1. Transformacija 'Time' kolone u 'Hour'...")
    # Pretvaramo sekunde u sate (ostatak pri deljenju sa 24)
    # Ovo pomaže algoritmu da prepozna doba dana kada se prevare dešavaju
    df['Hour'] = (df['Time'] // 3600) % 24
    df = df.drop(['Time'], axis=1)

    # Skaliranje Amount se NE radi ovde — mora se raditi NAKON podele na
    # trening/test skup kako bi se izbeglo curenje podataka (data leakage).
    # RobustScaler se fita samo na trening skupu u train.py.

    print("2. Reorganizacija kolona...")
    # Stavljamo 'Hour' i 'Amount' na početak, a 'Class' na sam kraj
    cols = ['Hour', 'Amount'] + [col for col in df.columns if col not in ['Hour', 'Amount', 'Class']] + ['Class']
    df = df[cols]

    print("3. Čuvanje procesuiranih podataka...")
    # Osiguravamo da processed folder postoji
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Čuvamo obradjene podatke bez index kolone
    df.to_csv(output_path, index=False)
    print(f"Gotovo! Podaci su uspešno sačuvani na: {output_path}")

if __name__ == "__main__":
    # Relativne putanje u odnosu na lokaciju src/ foldera
    INPUT_FILE = "../data/raw/creditcard.csv"
    OUTPUT_FILE = "../data/processed/creditcard_processed.csv"
    
    prepare_data(INPUT_FILE, OUTPUT_FILE)