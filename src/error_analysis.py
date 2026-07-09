import pandas as pd
import os
import joblib

def analyze_errors(models_dir, test_set_path, original_data_path, output_dir):
    # 1. Učitavanje podataka
    # Provera da li fajlovi postoje
    if not os.path.exists(test_set_path) or not os.path.exists(original_data_path):
        print("Greška: Neki od ulaznih fajlova (.csv) nisu pronađeni.")
        return

    df_test = pd.read_csv(test_set_path)
    df_orig = pd.read_csv(original_data_path)
    
    # Preuzimanje originalnog iznosa nazad u test set
    # Napomena: Ako indexi nisu isti, ovo može biti problem, ali ovde pretpostavljamo da jesu
    df_test['Amount'] = df_orig.loc[df_test.index, 'Amount']
    
    # Izdvajanje X i y (pazi da 'Amount' ne uđe u X_test ako ga model ne očekuje)
    # Ako tvoji modeli očekuju 30 ili 24 kolone, ne smeš proslediti 'Amount' nazad u predict
    X_test = df_test.drop(['Class', 'Amount'], axis=1) 
    y_test = df_test['Class']
    
    os.makedirs(output_dir, exist_ok=True)
    fajl_izvestaja = os.path.join(output_dir, 'error_analysis.txt')
    
    print(f"Započinjem analizu grešaka...")
    
    with open(fajl_izvestaja, 'w', encoding='utf-8') as f:
        f.write("=== ANALIZA GREŠAKA (po iznosu) ===\n")
        f.write("Model".ljust(20) + "\tFN\tFP\tTP\tAvg_FN_Amt\tAvg_TP_Amt\n\n")
        
        for file in os.listdir(models_dir):
            # Sigurnosna provera: samo .pkl fajlovi koji NISU scaler
            if file.endswith(".pkl") and "scaler" not in file.lower():
                model_path = os.path.join(models_dir, file)
                model = joblib.load(model_path)
                
                # Provera da li objekat ima .predict metodu
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                    
                    fn = df_test[(y_test == 1) & (y_pred == 0)]
                    fp = df_test[(y_test == 0) & (y_pred == 1)]
                    tp = df_test[(y_test == 1) & (y_pred == 1)]
                    
                    avg_fn = fn['Amount'].mean() if not fn.empty else 0
                    avg_tp = tp['Amount'].mean() if not tp.empty else 0
                    
                    line = (f"{file[:18]:<20}\t{len(fn)}\t{len(fp)}\t{len(tp)}\t"
                            f"{avg_fn:.2f}\t\t{avg_tp:.2f}\n")
                    f.write(line)
                    print(f"Analiziran: {file}")
                else:
                    print(f"Preskačem {file} (nije klasifikator)")
    
    print(f"\nAnaliza uspešno sačuvana u: {fajl_izvestaja}")

if __name__ == "__main__":
    # Definišemo putanje od roota projekta
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(SCRIPT_DIR) 
    
    analyze_errors(
        models_dir=os.path.join(BASE_DIR, "models"),
        test_set_path=os.path.join(BASE_DIR, "data", "processed", "test_set.csv"),
        original_data_path=os.path.join(BASE_DIR, "data", "raw", "creditcard.csv"),
        output_dir=os.path.join(BASE_DIR, "results", "metrics")
    )