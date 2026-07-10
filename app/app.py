import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(page_title="Fraud Detection", layout="centered")

# Apsolutne putanje kako bi app radio bez obzira odakle se pokreće
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test_set.csv")
THRESHOLDS_PATH = os.path.join(MODELS_DIR, "best_thresholds.json")

@st.cache_data
def load_data():
    return pd.read_csv(TEST_DATA_PATH)

@st.cache_data
def load_thresholds():
    # F2-optimalni pragovi po modelu (generiše evaluate.optimize_threshold).
    # Ako fajl ne postoji (pipeline još nije pokrenut), pada nazad na prazno -> prag 0.50.
    try:
        with open(THRESHOLDS_PATH, encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

@st.cache_resource
def load_scaler():
    # Scaler služi da skalirani iznos vratimo u čitljiv (originalni) iznos za prikaz
    return joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

# Učitavamo sve modele osim scalera koji nije klasifikator
available_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl') and f != 'scaler.pkl']
model_names = [f.replace('.pkl', '') for f in available_models]

st.title("Detekcija Zloupotrebe Kreditnih Kartica")
st.write("Interaktivni sistem sa podešavanjem osetljivosti u realnom vremenu.")
st.markdown("---")

selected_model_name = st.selectbox("Odaberi model za analizu transakcije:", sorted(model_names))

@st.cache_resource
def load_selected_model(model_name):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    return joblib.load(model_path)

try:
    model = load_selected_model(selected_model_name)
    df = load_data()
except Exception as e:
    st.error(f"Greška pri učitavanju modela ili podataka.\nDetalji: {e}")
    st.stop()

# Klizač kojim korisnik podešava granicu između legitimne i prevarantske transakcije
st.markdown("### Podešavanje osetljivosti sistema (Threshold)")
st.write("Ako smanjiš prag, model će biti 'paranoičniji' i lakše će blokirati kartice.")

# Svaki model ima svoj F2-optimalni prag (izabran na validaciji). Kada se promeni model,
# klizač se automatski postavlja na taj prag; korisnik i dalje može ručno da ga menja.
thresholds = load_thresholds()
default_threshold = float(thresholds.get(selected_model_name, 0.50))
if st.session_state.get('threshold_model') != selected_model_name:
    st.session_state['threshold_slider'] = default_threshold
    st.session_state['threshold_model'] = selected_model_name

threshold = st.slider(
    "Prag za proglašavanje prevare:",
    min_value=0.01, max_value=0.99, step=0.01, key='threshold_slider',
)
st.caption(f"Preporučeni (F2-optimalni) prag za model **{selected_model_name}**: {default_threshold:.2f}")
st.markdown("---")

st.subheader("1. Pristigla je nova transakcija")
col1, col2 = st.columns(2)

# Čuvamo izabranu transakciju u session_state kako se ne bi resetovala pri svakom kliku
if 'selected_tx' not in st.session_state:
    st.session_state.selected_tx = None
    st.session_state.actual_class = None

with col1:
    if st.button("Simuliraj LEGITIMNU transakciju"):
        sample = df[df['Class'] == 0].sample(1)
        st.session_state.selected_tx = sample.drop('Class', axis=1)
        st.session_state.actual_class = 0

with col2:
    if st.button("Simuliraj PREVARU"):
        sample = df[df['Class'] == 1].sample(1)
        st.session_state.selected_tx = sample.drop('Class', axis=1)
        st.session_state.actual_class = 1

# Predikcija i prikaz rezultata kada je transakcija odabrana
if st.session_state.selected_tx is not None:
    tx_data = st.session_state.selected_tx.iloc[0]

    # Vraćamo skalirani iznos u originalni (čitljiv) iznos pomoću scaler-a
    scaler = load_scaler()
    stvarni_iznos = scaler.inverse_transform([[tx_data['Scaled_Amount']]])[0][0]

    st.write("### Detalji presretnute transakcije:")
    st.json({
        "Sat transakcije (Hour)": f"{int(tx_data['Hour'])}:00",
        "Iznos transakcije": f"{stvarni_iznos:.2f}",
        "V14 parametar": round(tx_data['V14'], 4),
        "V4 parametar": round(tx_data['V4'], 4)
    })

    st.subheader(f"2. Odluka Modela: {selected_model_name}")

    if st.button("Pitaj Sistem", type="primary"):
        with st.spinner("Analiza u toku..."):
            # Verovatnoća prevare i odluka na osnovu praga.
            # Scaled_Amount se izbacuje jer nije atribut modela (čuva se samo za prikaz iznosa).
            proba = model.predict_proba(st.session_state.selected_tx.drop('Scaled_Amount', axis=1))[0]
            verovatnoca_prevare = proba[1]
            is_fraud = verovatnoca_prevare >= threshold

            if is_fraud:
                st.error("**ODBIJENO! SISTEM JE BLOKIRAO KARTICU.**")
                st.write(f"Model je {verovatnoca_prevare*100:.1f}% siguran da je ovo prevara. *(Prag je na {threshold*100:.1f}%)*")
            else:
                st.success("**ODOBRENO! TRANSAKCIJA JE ČISTA.**")
                st.write(f"Sumnja na prevaru je samo {verovatnoca_prevare*100:.1f}%, što je ispod tvog praga od {threshold*100:.1f}%.")

            stvarni_status = "Prevara" if st.session_state.actual_class == 1 else "Legitimna"
            st.info(f"*(Stvarni status transakcije iz baze: **{stvarni_status}**)*")
