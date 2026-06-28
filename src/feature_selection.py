import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score

def exhaustive_feature_selection(processed_data_path, original_model_path, metrics_dir):
    print("1. Učitavanje podataka i originalnog modela...")
    df = pd.read_csv(processed_data_path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

    scaler = RobustScaler()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    X_train['Scaled_Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_val['Scaled_Amount'] = scaler.transform(X_val[['Amount']])
    X_test['Scaled_Amount'] = scaler.transform(X_test[['Amount']])
    X_train = X_train.drop('Amount', axis=1)
    X_val = X_val.drop('Amount', axis=1)
    X_test = X_test.drop('Amount', axis=1)

    # Učitavanje originalnog modela da bismo izvukli koje su kolone najbitnije
    try:
        original_rf = joblib.load(original_model_path)
    except FileNotFoundError:
        print(f"Greška: Nije pronađen model na putanji {original_model_path}")
        return

    importances = original_rf.feature_importances_

    # Rangiranje svih 30 atributa
    feature_df = pd.DataFrame({
        'Atribut': X_train.columns,
        'Značaj': importances
    }).sort_values(by='Značaj', ascending=False)

    sve_kolone = len(feature_df)
    svi_rezultati = []

    print(f"\n2. Započinjem eksperiment (Isprobavanje od 1 do {sve_kolone} kolona).")
    print("   ⏳ UPOZORENJE: Ovo će potrajati! Trenira se 30 modela zaredom...\n")

    # Petlja koja ide od 1 do 30
    for k in range(1, sve_kolone + 1):
        top_k_features = feature_df['Atribut'].head(k).tolist()
        print(f"   -> [Model {k}/{sve_kolone}] Treniram sa top {k} atributa...")
        
        # Smanjujemo podatke samo na trenutni broj najboljih kolona
        X_train_reduced = X_train[top_k_features]
        X_val_reduced   = X_val[top_k_features]

        # Sprovodimo SMOTE na smanjenom skupu
        smote = SMOTE(random_state=42)
        X_train_smote_red, y_train_smote_red = smote.fit_resample(X_train_reduced, y_train)

        # Treniramo model
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_smote_red, y_train_smote_red)

        # Ocenjujemo model na validacionom skupu
        y_pred = rf.predict(X_val_reduced)
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)

        # Zapisujemo rezultate za trenutni broj kolona
        svi_rezultati.append({
            'Broj_Atributa': k,
            'Odziv_Recall': recall,
            'Preciznost_Precision': precision,
            'Kolone': ", ".join(top_k_features)
        })

    print("\n3. Završeno treniranje! Rangiranje rezultata...")
    # Pakovanje u tabelu radi lakšeg sortiranja
    rezultati_df = pd.DataFrame(svi_rezultati)
    
    # Sortiramo tako da najbolji Odziv (Recall) bude prvi. 
    # Ako imaju isti odziv, gledamo ko ima bolju Preciznost.
    rezultati_df = rezultati_df.sort_values(by=['Odziv_Recall', 'Preciznost_Precision'], ascending=[False, False])
    
    # Resetujemo indeks za lepši prikaz ranga
    rezultati_df = rezultati_df.reset_index(drop=True)

    print("4. Čuvanje u fajl...")
    os.makedirs(metrics_dir, exist_ok=True)
    fajl_putanja = os.path.join(metrics_dir, 'rang_lista_atributa.txt')
    
    with open(fajl_putanja, 'w', encoding='utf-8') as f:
        f.write("=== SVEOTBUHVATNA RANG LISTA BROJA ATRIBUTA ===\n")
        f.write("Sortirano primarno po najboljem Odzivu, a sekundarno po Preciznosti.\n\n")
        
        for index, row in rezultati_df.iterrows():
            f.write(f"🏆 MESTO #{index + 1} | Korišćeno {row['Broj_Atributa']} atributa\n")
            f.write(f"   Odziv (Recall):  {row['Odziv_Recall']:.4f}\n")
            f.write(f"   Preciznost:      {row['Preciznost_Precision']:.4f}\n")
            f.write(f"   Korišćene kolone: {row['Kolone']}\n")
            f.write("-" * 65 + "\n")

    print(f"\n✅ Apsolutno sve je završeno!")
    print(f"Rang lista je sačuvana u: {fajl_putanja}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH      = os.path.join(BASE_DIR, "data", "processed", "creditcard_processed.csv")
    ORIGINAL_MODEL = os.path.join(BASE_DIR, "models", "RandomForest_100.pkl")
    METRICS_DIR    = os.path.join(BASE_DIR, "results", "metrics")

    exhaustive_feature_selection(DATA_PATH, ORIGINAL_MODEL, METRICS_DIR)