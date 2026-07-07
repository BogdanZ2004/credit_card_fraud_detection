import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Prostori pretrage hiperparametara za svaki model koji se koriste u RandomizedSearchCV
SEARCH_SPACES = {
    "LogisticRegression": {
        'params': {
            'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        },
        'n_iter': 6,
    },
    "DecisionTree": {
        'params': {
            'model__max_depth':        [5, 10, 15, 20, None],
            'model__min_samples_leaf': [1, 2, 4, 8],
            'model__criterion':        ['gini', 'entropy'],
        },
        'n_iter': 20,
    },
    "RandomForest_100": {
        'params': {
            'model__max_depth':    [10, 20, None],
            'model__max_features': ['sqrt', 'log2'],
        },
        'n_iter': 6,
    },
    "RandomForest_150": {
        'params': {
            'model__max_depth':    [10, 20, None],
            'model__max_features': ['sqrt', 'log2'],
        },
        'n_iter': 6,
    },
    "XGBoost": {
        'params': {
            'model__learning_rate':    [0.01, 0.05, 0.1, 0.3],
            'model__max_depth':        [3, 5, 7],
            'model__subsample':        [0.7, 0.8, 1.0],
            'model__colsample_bytree': [0.7, 0.8, 1.0],
        },
        'n_iter': 30,
    },
}

def get_base_models():
    # n_jobs=1 jer RandomizedSearchCV već paralelizuje foldove — dvostruka paralelizacija pravi konflikte
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree":       DecisionTreeClassifier(random_state=42),
        "RandomForest_100":   RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
        "RandomForest_150":   RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1),
        "XGBoost":            XGBClassifier(n_estimators=50, random_state=42, n_jobs=1),
    }


def split_data(df, random_state=42):
    # Podela na 70% trening, 15% validacija, 15% test — stratifikovano po klasi
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=random_state, stratify=y_temp
    )

    print(f"   Trening: {len(X_train)} | Validacija: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, models_dir):
    # Scaler se fita SAMO na trening skupu kako bi se izbeglo curenje podataka
    scaler = RobustScaler()
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    X_train['Scaled_Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_val['Scaled_Amount']   = scaler.transform(X_val[['Amount']])
    X_test['Scaled_Amount']  = scaler.transform(X_test[['Amount']])

    X_train = X_train.drop('Amount', axis=1)
    X_val   = X_val.drop('Amount', axis=1)
    X_test  = X_test.drop('Amount', axis=1)

    # Čuvamo scaler kako bi ga app.py mogao koristiti pri predikciji
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    return X_train, X_val, X_test


def apply_smote(X_train, y_train, random_state=42):
    # Sintetičko presamplovanje manjinske klase samo na trening skupu
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


def tune_hyperparameters(X_train, y_train, metrics_dir):
    # SMOTE je unutar pipeline-a svakog folda kako ne bi curio u validacioni deo folda
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_models = get_base_models()
    best_params_all = {}

    os.makedirs(metrics_dir, exist_ok=True)
    results_file = os.path.join(metrics_dir, 'tuning_results.txt')

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=== REZULTATI PODEŠAVANJA HIPERPARAMETARA (RandomizedSearchCV) ===\n")
        f.write("Optimizovano po AUPRC | SMOTE primenjen unutar svakog folda\n\n")

    for name, model in base_models.items():
        print(f"   -> Podešavam {name}...")
        space = SEARCH_SPACES[name]

        # Pipeline kombinuje SMOTE i model kako bi se SMOTE primenjivao unutar svakog folda
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=space['params'],
            n_iter=space['n_iter'],
            scoring='average_precision',
            cv=cv,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_params = {k.replace('model__', ''): v for k, v in search.best_params_.items()}
        best_params_all[name] = best_params

        result = (
            f"--- {name} ---\n"
            f"Najbolji parametri: {best_params}\n"
            f"Najbolji AUPRC (CV): {search.best_score_:.4f}\n\n"
        )
        print("      " + result.replace("\n", "\n      ").strip())

        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(result)

    print(f"\n   Rezultati sačuvani u: {results_file}")
    return best_params_all


def train_models(X_train_smote, y_train_smote, X_val, y_val, best_params, models_dir):
    os.makedirs(models_dir, exist_ok=True)

    # Finalni modeli se treniraju sa najboljim hiperparametrima i n_jobs=-1 za brzinu
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42,
            **best_params.get('LogisticRegression', {})
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=42,
            **best_params.get('DecisionTree', {})
        ),
        "RandomForest_100": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
            **best_params.get('RandomForest_100', {})
        ),
        "RandomForest_150": RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1,
            **best_params.get('RandomForest_150', {})
        ),
        # XGBoost koristi early stopping: prati AUPRC na validaciji i staje kad prestane da napreduje
        # strpljivost 50 (~10% od 500) — dovoljno da preskoči privremene padove validacije
        "XGBoost": XGBClassifier(
            n_estimators=500, random_state=42, n_jobs=-1,
            early_stopping_rounds=50, eval_metric='aucpr',
            **best_params.get('XGBoost', {})
        ),
    }

    for name, model in models.items():
        print(f"   -> Treniram {name}...")
        if name == "XGBoost":
            # eval_set je validacioni skup — na osnovu njega early stopping bira najbolju iteraciju
            model.fit(X_train_smote, y_train_smote, eval_set=[(X_val, y_val)], verbose=False)
            print(f"      [Early stopping: najbolja iteracija = {model.best_iteration}]")
        else:
            model.fit(X_train_smote, y_train_smote)
        joblib.dump(model, os.path.join(models_dir, f"{name}.pkl"))
        print(f"      [Sačuvano: {name}.pkl]")


def train_pipeline(processed_data_path, models_dir, val_data_path, test_data_path, metrics_dir):
    print("1. Učitavanje procesuiranih podataka...")
    df = pd.read_csv(processed_data_path)

    print("\n2. Podela na Trening / Validacioni / Test skup (70% / 15% / 15%)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    print("\n3. Skaliranje 'Amount' kolone (fit samo na trening skupu)...")
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test, models_dir)

    print("\n4. Podešavanje hiperparametara (RandomizedSearchCV, SMOTE unutar folda)...")
    best_params = tune_hyperparameters(X_train, y_train, metrics_dir)

    print("\n5. Primena SMOTE tehnike SAMO na Trening setu...")
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    print("\n6. Treniranje finalnih modela (XGBoost sa early stopping-om na validaciji)...")
    train_models(X_train_smote, y_train_smote, X_val, y_val, best_params, models_dir)

    print("\n7. Čuvanje Validacionog i Test seta za evaluaciju...")
    val_df = X_val.copy()
    val_df['Class'] = y_val
    val_df.to_csv(val_data_path, index=False)

    test_df = X_test.copy()
    test_df['Class'] = y_test
    test_df.to_csv(test_data_path, index=False)

    print("\nSve je uspešno završeno!")


if __name__ == "__main__":
    INPUT_FILE  = "../data/processed/creditcard_processed.csv"
    MODELS_DIR  = "../models"
    VAL_OUTPUT  = "../data/processed/val_set.csv"
    TEST_OUTPUT = "../data/processed/test_set.csv"
    METRICS_DIR = "../results/metrics"

    train_pipeline(INPUT_FILE, MODELS_DIR, VAL_OUTPUT, TEST_OUTPUT, METRICS_DIR)
