"""
Application Streamlit - Prediction du Risque d'AVC (Stroke Prediction)

Interface web interactive permettant de :
- Saisir les caracteristiques d'un patient
- Obtenir une prediction du risque d'AVC
- Visualiser les informations du projet
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import sys

# --- Configuration de la page ---
st.set_page_config(
    page_title="Prediction du Risque d'AVC",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Chemins des fichiers ---
# Methode robuste pour trouver le dossier de l'app sur Streamlit Cloud
def find_base_dir():
    """Trouve le dossier contenant app.py de maniere robuste."""
    # Methode 1 : __file__
    if '__file__' in globals():
        d = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(d):
            return d
    # Methode 2 : cwd
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, 'app.py')):
        return cwd
    # Methode 3 : sys.argv[0]
    if len(sys.argv) > 0:
        d = os.path.dirname(os.path.abspath(sys.argv[0]))
        if os.path.exists(d):
            return d
    return os.getcwd()

BASE_DIR = find_base_dir()
MODEL_PATH = os.path.join(BASE_DIR, "stroke_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "stroke_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")

# --- Chargement du modele et du scaler ---
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_features():
    with open(FEATURES_PATH, 'rb') as f:
        return pickle.load(f)

# --- Verification des fichiers ---
missing_files = []
for path, name in [(MODEL_PATH, "stroke_model.pkl"),
                    (SCALER_PATH, "stroke_scaler.pkl"),
                    (FEATURES_PATH, "feature_columns.pkl")]:
    if not os.path.exists(path):
        missing_files.append(name)

if missing_files:
    st.error(f"Fichiers manquants : {', '.join(missing_files)}")
    st.markdown(f"**Dossier recherche :** `{BASE_DIR}`")
    st.markdown("**Fichiers presents dans ce dossier :**")
    if os.path.exists(BASE_DIR):
        files = os.listdir(BASE_DIR)
        for f in sorted(files):
            st.code(f"  {f}")
    else:
        st.code(f"  Le dossier {BASE_DIR} n'existe pas !")
    st.markdown("---")
    st.info("**Solution :** Assurez-vous que les 3 fichiers `.pkl` sont dans le **meme dossier** que `app.py` dans votre depot GitHub, puis poussez-les avec `git add` et `git commit`.")
    st.stop()

model = load_model()
scaler = load_scaler()
feature_columns = load_features()

# --- CSS personnalise ---
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1a237e;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #455a64;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .result-high {
        background: linear-gradient(135deg, #ff5252, #d32f2f);
        color: white;
    }
    .result-low {
        background: linear-gradient(135deg, #66bb6a, #2e7d32);
        color: white;
    }
    .info-card {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ========================
# SIDEBAR - Informations
# ========================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("A propos du projet")
    st.markdown("""
    **Projet ML - Prediction du Risque d'AVC**

    Cette application utilise un modele de
    Machine Learning (Regression Logistique)
    entraine sur le dataset **Stroke Prediction**
    de Kaggle (5 110 patients, 12 variables).

    ---
    **Modele utilise :**
    - Regression Logistique
    - SMOTE pour equilibrer les classes
    - StandardScaler pour la normalisation
    - KPI prioritaire : Recall (minimiser les faux negatifs)

    ---
    **Avertissement :**
    Cette application est un outil d'aide a la
    decision et ne remplace en aucun cas un
    diagnostic medical professionnel.
    """)

    st.markdown("---")
    st.caption("Projet Machine Learning | Stroke Prediction")

    # Navigation
    st.markdown("---")
    page = st.radio("Navigation", ["Prediction", "Informations sur les donnees"], label_visibility="collapsed")


# ========================
# PAGE PRINCIPALE
# ========================

st.markdown('<h1 class="main-title">🧠 Prediction du Risque d\'AVC</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Saisissez les informations du patient pour evaluer le risque d\'accident vasculaire cerebral</p>', unsafe_allow_html=True)

if page == "Prediction":
    # ========================
    # FORMULAIRE DE SAISIE
    # ========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Informations demographiques")

        gender = st.selectbox(
            "Genre",
            options=["Male", "Female"],
            format_func=lambda x: "Homme" if x == "Male" else "Femme"
        )

        age = st.slider(
            "Age (annees)",
            min_value=1,
            max_value=100,
            value=45,
            step=1
        )

        ever_married = st.selectbox(
            "Deja marie(e)",
            options=["Yes", "No"],
            format_func=lambda x: "Oui" if x == "Yes" else "Non"
        )

        work_type = st.selectbox(
            "Type de travail",
            options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            format_func=lambda x: {
                "Private": "Prive",
                "Self-employed": "Travailleur independant",
                "Govt_job": "Fonctionnaire",
                "children": "Enfant",
                "Never_worked": "Jamais travaille"
            }[x]
        )

        residence_type = st.selectbox(
            "Type de residence",
            options=["Urban", "Rural"],
            format_func=lambda x: "Urbain" if x == "Urban" else "Rural"
        )

    with col2:
        st.subheader("🏥 Informations medicales")

        hypertension = st.selectbox(
            "Hypertension",
            options=[0, 1],
            format_func=lambda x: "Oui" if x == 1 else "Non",
            index=1
        )

        heart_disease = st.selectbox(
            "Maladie cardiaque",
            options=[0, 1],
            format_func=lambda x: "Oui" if x == 1 else "Non",
            index=1
        )

        avg_glucose_level = st.slider(
            "Taux moyen de glucose (mg/dL)",
            min_value=50.0,
            max_value=300.0,
            value=106.0,
            step=0.1
        )

        bmi = st.slider(
            "Indice de masse corporelle (BMI)",
            min_value=10.0,
            max_value=100.0,
            value=28.1,
            step=0.1
        )

        smoking_status = st.selectbox(
            "Statut tabagique",
            options=["never smoked", "formerly smoked", "smokes", "Unknown"],
            format_func=lambda x: {
                "never smoked": "Jamais fume",
                "formerly smoked": "Ancien fumeur",
                "smokes": "Fumeur",
                "Unknown": "Inconnu"
            }[x]
        )

    st.markdown("---")

    # ========================
    # BOUTON DE PREDICTION
    # ========================
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_button = st.button("🔍 Evaluer le risque d'AVC", type="primary", use_container_width=True)

    if predict_button:
        # --- Construction du dataframe d'entree ---
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'Residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'work_type': [work_type],
            'smoking_status': [smoking_status]
        })

        # --- Encodage (identique au notebook) ---
        input_data['gender'] = input_data['gender'].map({'Male': 0, 'Female': 1})
        input_data['ever_married'] = input_data['ever_married'].map({'No': 0, 'Yes': 1})
        input_data['Residence_type'] = input_data['Residence_type'].map({'Rural': 0, 'Urban': 1})

        # One-Hot Encoding pour work_type et smoking_status
        input_encoded = pd.get_dummies(input_data, columns=['work_type', 'smoking_status'], drop_first=False)

        # Assurer que toutes les colonnes sont presentes (ajouter les manquantes a 0)
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reordonner les colonnes selon l'ordre d'entrainement
        input_encoded = input_encoded[feature_columns]

        # --- Normalisation ---
        input_scaled = scaler.transform(input_encoded)

        # --- Prediction ---
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        # --- Affichage des resultats ---
        st.markdown("---")

        # Resume du patient
        with st.expander("📋 Resume du patient", expanded=True):
            resume_col1, resume_col2 = st.columns(2)
            with resume_col1:
                st.write(f"**Genre** : {'Homme' if gender == 'Male' else 'Femme'}")
                st.write(f"**Age** : {age} ans")
                st.write(f"**Marie(e)** : {'Oui' if ever_married == 'Yes' else 'Non'}")
                st.write(f"**Residence** : {'Urbain' if residence_type == 'Urban' else 'Rural'}")
            with resume_col2:
                st.write(f"**Hypertension** : {'Oui' if hypertension == 1 else 'Non'}")
                st.write(f"**Maladie cardiaque** : {'Oui' if heart_disease == 1 else 'Non'}")
                st.write(f"**Glucose moyen** : {avg_glucose_level:.1f} mg/dL")
                st.write(f"**BMI** : {bmi:.1f}")

        # Resultat principal
        st.markdown("<br>", unsafe_allow_html=True)
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

        with result_col2:
            if prediction[0] == 1:
                st.markdown(f"""
                <div class="result-box result-high">
                    <h2>⚠️ RISQUE D'AVC DETECTE</h2>
                    <p style="font-size: 1.2rem; margin: 10px 0;">
                        Le modele predit un risque d'AVC pour ce patient.
                    </p>
                    <p style="font-size: 2rem; font-weight: bold;">
                        Probabilite : {probability*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box result-low">
                    <h2>✅ FAIBLE RISQUE D'AVC</h2>
                    <p style="font-size: 1.2rem; margin: 10px 0;">
                        Le modele predit un faible risque d'AVC pour ce patient.
                    </p>
                    <p style="font-size: 2rem; font-weight: bold;">
                        Probabilite : {probability*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # Metriques detaillees
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.markdown("""
            <div class="metric-card">
                <p style="font-size: 0.9rem; color: #455a64;">Probabilite d'AVC</p>
                <p style="font-size: 1.8rem; font-weight: bold; color: #1565c0;">
            """ + f"{probability*100:.2f}%" + """
                </p>
            </div>
            """, unsafe_allow_html=True)

        with metric_col2:
            st.markdown("""
            <div class="metric-card">
                <p style="font-size: 0.9rem; color: #455a64;">Niveau de confiance</p>
                <p style="font-size: 1.8rem; font-weight: bold; color: #1565c0;">
            """ + f"{max(probability, 1-probability)*100:.2f}%" + """
                </p>
            </div>
            """, unsafe_allow_html=True)

        with metric_col3:
            if probability > 0.5:
                severity = "Eleve"
                color = "#d32f2f"
            elif probability > 0.3:
                severity = "Modere"
                color = "#f57c00"
            elif probability > 0.1:
                severity = "Faible"
                color = "#fbc02d"
            else:
                severity = "Tres faible"
                color = "#388e3c"

            st.markdown(f"""
            <div class="metric-card">
                <p style="font-size: 0.9rem; color: #455a64;">Niveau de risque</p>
                <p style="font-size: 1.8rem; font-weight: bold; color: {color};">
                    {severity}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Facteurs de risque
        st.markdown("---")
        st.subheader("📈 Analyse des facteurs de risque")

        risk_factors = []
        if age >= 55:
            risk_factors.append(("Age >= 55 ans", "Eleve", "#d32f2f"))
        elif age >= 45:
            risk_factors.append(("Age entre 45 et 54 ans", "Modere", "#f57c00"))
        else:
            risk_factors.append(("Age < 45 ans", "Faible", "#388e3c"))

        if avg_glucose_level > 140:
            risk_factors.append(("Glucose > 140 mg/dL", "Eleve", "#d32f2f"))
        elif avg_glucose_level > 100:
            risk_factors.append(("Glucose 100-140 mg/dL", "Modere", "#f57c00"))
        else:
            risk_factors.append(("Glucose < 100 mg/dL", "Normal", "#388e3c"))

        if bmi >= 30:
            risk_factors.append(("BMI >= 30 (Obesite)", "Eleve", "#d32f2f"))
        elif bmi >= 25:
            risk_factors.append(("BMI 25-29.9 (Surpoids)", "Modere", "#f57c00"))
        else:
            risk_factors.append(("BMI < 25 (Normal)", "Faible", "#388e3c"))

        if hypertension == 1:
            risk_factors.append(("Hypertension", "Present", "#d32f2f"))
        else:
            risk_factors.append(("Hypertension", "Absent", "#388e3c"))

        if heart_disease == 1:
            risk_factors.append(("Maladie cardiaque", "Present", "#d32f2f"))
        else:
            risk_factors.append(("Maladie cardiaque", "Absent", "#388e3c"))

        if smoking_status == "smokes":
            risk_factors.append(("Tabagisme", "Fumeur actif", "#d32f2f"))
        elif smoking_status == "formerly smoked":
            risk_factors.append(("Tabagisme", "Ancien fumeur", "#f57c00"))
        else:
            risk_factors.append(("Tabagisme", "Non fumeur", "#388e3c"))

        for factor, level, color in risk_factors:
            col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
            with col_f1:
                st.write(f"**{factor}**")
            with col_f2:
                st.markdown(f"<span style='color: {color}; font-weight: bold;'>{level}</span>", unsafe_allow_html=True)
            with col_f3:
                st.write("")

        st.markdown("---")
        st.warning("⚠️ **Important** : Cette prediction est realisee par un modele de Machine Learning a titre indicatif. Elle ne remplace en aucun cas un avis medical professionnel. Consultez toujours un medecin pour un diagnostic.")


elif page == "Informations sur les donnees":
    st.subheader("📋 Informations sur le jeu de donnees")

    tab1, tab2, tab3 = st.tabs(["Description des variables", "Pipeline ML", "Methodologie"])

    with tab1:
        st.markdown("""
        Le dataset **Stroke Prediction Dataset** (Kaggle - fedesoriano) contient **5 110 patients** avec 12 variables :

        | Variable | Type | Description |
        |---|---|---|
        | `id` | int | Identifiant unique (supprime) |
        | `gender` | categorielle | Male / Female / Other |
        | `age` | numerique | Age du patient |
        | `hypertension` | binaire | 0 = non, 1 = oui |
        | `heart_disease` | binaire | 0 = non, 1 = oui |
        | `ever_married` | categorielle | Yes / No |
        | `work_type` | categorielle | Private, Self-employed, Govt_job, children, Never_worked |
        | `Residence_type` | categorielle | Urban / Rural |
        | `avg_glucose_level` | numerique | Taux moyen de glucose sanguin |
        | `bmi` | numerique | Indice de masse corporelle (201 valeurs manquantes) |
        | `smoking_status` | categorielle | formerly smoked / never smoked / smokes / Unknown |
        | `stroke` | binaire (cible) | 0 = pas d'AVC, 1 = AVC |

        **Desequilibre de classes** : ~4.9% de cas positifs (AVC) -> fort desequilibre necessitant SMOTE.
        """)

    with tab2:
        st.markdown("""
        ### Pipeline de preprocessing

        1. **Suppression de `id`** : identifiant sans valeur predictive
        2. **Suppression de `gender = Other`** : bruit marginal (1 observation)
        3. **Imputation du BMI** : par la mediane (robuste aux outliers)
        4. **Encodage** :
           - `gender` : Male=0, Female=1
           - `ever_married` : No=0, Yes=1
           - `Residence_type` : Rural=0, Urban=1
           - `work_type` et `smoking_status` : One-Hot Encoding (drop_first=True)
        5. **Train/Test Split** : stratifie (test_size=0.2, random_state=42)
        6. **StandardScaler** : centrage-reduction
        7. **SMOTE** : uniquement sur le train (eviter la fuite d'information)

        ### Modele retenu

        **Regression Logistique** (meilleur equilibre performance / generalisation)
        - `max_iter=1000`, `random_state=42`
        - Complementee par XGBoost pour les analyses d'importance des variables

        ### Modeles testes
        1. Logistic Regression
        2. Decision Tree
        3. Random Forest
        4. KNN
        5. XGBoost
        """)

    with tab3:
        st.markdown("""
        ### Formulation du probleme

        **Probleme de classification binaire** :
        - `0` -> Pas d'AVC
        - `1` -> AVC

        ### KPIs (indicateurs de performance)

        - **Recall** (prioritaire) : minimiser les faux negatifs
          - Ne pas manquer un vrai cas d'AVC est critique
        - **AUC-ROC** : capacite de discrimination globale
        - **F1-score** : compromis precision/recall
        - **Accuracy** : indicateur general

        ### Pourquoi le Recall est prioritaire ?

        Dans un contexte medical, un **faux negatif** (AVC non detecte)
        est bien plus grave qu'un **faux positif** (alerte inutile).
        Le cout d'un diagnostic manque est potentiellement mortel,
        tandis qu'un faux positif entraine simplement des examens
        complementaires.
        """)
