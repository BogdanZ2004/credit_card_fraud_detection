import os
import sys

# Dodajemo src/ u path kako bi import modula radio bez obzira odakle se pokreće
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from duplicate_analysis import analyze_duplicates
from data_preparation import prepare_data
from eda_analysis import run_eda
from train import train_pipeline
from evaluate import select_best_model_on_validation, optimize_threshold, evaluate_models

# Apsolutne putanje ka svim folderima i fajlovima projekta
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DATA    = os.path.join(BASE_DIR, 'data', 'raw', 'creditcard.csv')
PROCESSED   = os.path.join(BASE_DIR, 'data', 'processed', 'creditcard_processed.csv')
VAL_SET     = os.path.join(BASE_DIR, 'data', 'processed', 'val_set.csv')
TEST_SET    = os.path.join(BASE_DIR, 'data', 'processed', 'test_set.csv')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
FIGURES_DIR = os.path.join(BASE_DIR, 'results', 'figures')
METRICS_DIR = os.path.join(BASE_DIR, 'results', 'metrics')

# Ova grana trenira modele na 18 atributa izabranih na main grani (pravilo 1-SE),
# pa nema koraka selekcije. Redosled koraka sprečava curenje — test se dira tek na kraju.

if __name__ == "__main__":
    print("=" * 60)
    print("KORAK 1: Analiza duplikata (sirovi podaci, pre uklanjanja)")
    print("=" * 60)
    analyze_duplicates(RAW_DATA, METRICS_DIR)

    print("\n" + "=" * 60)
    print("KORAK 2: Priprema podataka")
    print("=" * 60)
    prepare_data(RAW_DATA, PROCESSED)

    print("\n" + "=" * 60)
    print("KORAK 3: Eksplorativna analiza podataka (EDA)")
    print("=" * 60)
    run_eda(PROCESSED, FIGURES_DIR, METRICS_DIR)

    print("\n" + "=" * 60)
    print("KORAK 4: Treniranje modela (na 18 izabranih atributa)")
    print("=" * 60)
    train_pipeline(PROCESSED, MODELS_DIR, VAL_SET, TEST_SET, METRICS_DIR)

    print("\n" + "=" * 60)
    print("KORAK 5: Izbor najboljeg modela na validacionom skupu")
    print("=" * 60)
    select_best_model_on_validation(VAL_SET, MODELS_DIR, METRICS_DIR)

    print("\n" + "=" * 60)
    print("KORAK 6: Podešavanje praga odluke na validaciji (F2)")
    print("=" * 60)
    optimize_threshold(VAL_SET, MODELS_DIR, METRICS_DIR)

    print("\n" + "=" * 60)
    print("KORAK 7: Evaluacija modela na test skupu")
    print("=" * 60)
    evaluate_models(TEST_SET, MODELS_DIR, FIGURES_DIR, METRICS_DIR)

    print("\n" + "=" * 60)
    print("Pipeline uspešno završen!")
    print(f"  Metrike:  {METRICS_DIR}")
    print(f"  Grafici:  {FIGURES_DIR}")
    print(f"  Modeli:   {MODELS_DIR}")
    print("=" * 60)
