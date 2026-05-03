"""
===========================================
  Stroke Prediction - Interface Streamlit
  Projet ML : Prediction du Risque d'AVC
  GradientBoostingClassifier (sans pickle)
===========================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import io

# ============================================================
# Configuration de la page
# ============================================================
st.set_page_config(
    page_title="Prediction AVC",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS personnalise
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .result-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        color: #155724;
    }
    .risk-high {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .info-box {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border: 1px solid #0c5460;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Entrainement du modele au demarrage (cache - 1 seul appel)
# ============================================================
@st.cache_resource
def train_model():
    """
    Telecharge les donnees et entraine le GradientBoostingClassifier.
    Identique au notebook : random_state=42, meme preprocessing.
    Cache avec @st.cache_resource => entraine seulement au 1er lancement.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    # --- Chargement des donnees ---
    url = (
        "https://raw.githubusercontent.com/Saikrishna-89/"
        "STROKE-PREDICTION-USING-HEALTHCARE-DATA/main/"
        "healthcare-dataset-stroke-data.csv"
    )
    response = urllib.request.urlopen(url, timeout=30)
    df = pd.read_csv(io.BytesIO(response.read()))

    # --- Preprocessing (exactement comme le notebook) ---
    df = df.drop(columns=["id"])
    df = df[df["gender"] != "Other"].reset_index(drop=True)

    bmi_median = df["bmi"].median()
    df["bmi"] = df["bmi"].fillna(bmi_median)

    # Encodage binaire
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    df["ever_married"] = df["ever_married"].map({"No": 0, "Yes": 1})
    df["Residence_type"] = df["Residence_type"].map({"Rural": 0, "Urban": 1})

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=["work_type", "smoking_status"], drop_first=True)

    # --- Separation features / cible ---
    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Entrainement du meilleur modele ---
    model = GradientBoostingClassifier(
        random_state=42,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=200,
    )
    model.fit(X_train, y_train)

    return model, bmi_median, X.columns.tolist()


# --- Chargement du modele (entraine au 1er lancement uniquement) ---
with st.spinner("Chargement du modele..."):
    model, bmi_median, feature_columns = train_model()


# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1 style="color: #2c3e50;">🧠 Prediction du Risque d'AVC</h1>
    <p style="font-size: 18px; color: #7f8c8d;">
        Application de <strong>Machine Learning</strong> pour predire le
        risque d'accident vasculaire cerebral
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# Sidebar - Informations
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("A propos")
    st.markdown("""
    Cette application utilise un modele de **Gradient Boosting** entraine
    sur le dataset **Stroke Prediction Dataset** (Kaggle - fedesoriano)
    contenant **5 110 patients**.

    **Variables utilisees :**
    - Genre
    - Age
    - Hypertension
    - Maladie cardiaque
    - Etat marital
    - Type de travail
    - Type de residence
    - Taux de glucose moyen
    - IMC (BMI)
    - Statut tabagique

    ---
    ⚠️ **Avertissement** : Cette application est un outil d'aide
    a la decision et ne remplace en aucun cas un diagnostic medical
    professionnel.
    """)

    st.markdown("---")
    st.markdown("""
    **Modele :** Gradient Boosting
    **Accuracy :** ~94.3%
    **Metrique prioritaire :** Recall
    **Dataset :** 5 110 patients
    """)

# ============================================================
# Formulaire de saisie
# ============================================================
st.subheader("📋 Informations du Patient")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Donnees demographiques")
    gender = st.selectbox(
        "Genre",
        options=["Male", "Female"],
        format_func=lambda x: "👨 Homme" if x == "Male" else "👩 Femme",
        index=0,
    )
    age = st.slider("Age", min_value=0, max_value=100, value=45, step=1)
    ever_married = st.selectbox(
        "Marie(e) / A deja ete marie(e)",
        options=["Yes", "No"],
        format_func=lambda x: "✅ Oui" if x == "Yes" else "❌ Non",
        index=0,
    )
    work_type = st.selectbox(
        "Type de travail",
        options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        format_func=lambda x: {
            "Private": "🏢 Prive",
            "Self-employed": "🧑‍💼 Independant",
            "Govt_job": "🏛️ Fonction publique",
            "children": "👶 Enfant",
            "Never_worked": "🚫 N'a jamais travaille",
        }[x],
        index=0,
    )
    residence_type = st.selectbox(
        "Type de residence",
        options=["Urban", "Rural"],
        format_func=lambda x: "🏙️ Urbain" if x == "Urban" else "🌾 Rural",
        index=0,
    )

with col2:
    st.markdown("#### Donnees medicales")
    hypertension = st.selectbox(
        "Hypertension",
        options=[0, 1],
        format_func=lambda x: "✅ Oui" if x == 1 else "❌ Non",
        index=0,
    )
    heart_disease = st.selectbox(
        "Maladie cardiaque",
        options=[0, 1],
        format_func=lambda x: "✅ Oui" if x == 1 else "❌ Non",
        index=0,
    )
    avg_glucose_level = st.slider(
        "Taux moyen de glucose (mg/dL)",
        min_value=50.0,
        max_value=300.0,
        value=100.0,
        step=0.1,
    )
    bmi = st.slider(
        "IMC (BMI)",
        min_value=10.0,
        max_value=100.0,
        value=28.0,
        step=0.1,
    )
    smoking_status = st.selectbox(
        "Statut tabagique",
        options=["never smoked", "formerly smoked", "smokes", "Unknown"],
        format_func=lambda x: {
            "never smoked": "🚭 Jamais fume",
            "formerly smoked": "🚬 Ancien fumeur",
            "smokes": "🔥 Fumeur",
            "Unknown": "❓ Inconnu",
        }[x],
        index=0,
    )

# ============================================================
# Bouton de prediction
# ============================================================
st.markdown("---")
predict_button = st.button(
    "🔍 Predire le risque d'AVC", type="primary", use_container_width=True
)

if predict_button:
    # --------------------------------------------------------
    # Preparation des donnees (meme pipeline que le notebook)
    # --------------------------------------------------------
    input_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "work_type": work_type,
        "smoking_status": smoking_status,
    }

    df_input = pd.DataFrame([input_data])

    # Encodage binaire
    df_input["gender"] = df_input["gender"].map({"Male": 0, "Female": 1})
    df_input["ever_married"] = df_input["ever_married"].map({"No": 0, "Yes": 1})
    df_input["Residence_type"] = df_input["Residence_type"].map({"Rural": 0, "Urban": 1})

    # One-Hot Encoding
    df_input = pd.get_dummies(df_input, columns=["work_type", "smoking_status"], drop_first=True)

    # S'assurer que toutes les colonnes du modele sont presentes
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    # Reordonner les colonnes
    df_input = df_input[feature_columns]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0]

    prob_no_stroke = probability[0] * 100
    prob_stroke = probability[1] * 100

    # --------------------------------------------------------
    # Affichage des resultats
    # --------------------------------------------------------
    st.markdown("## 📊 Resultat de la prediction")

    if prediction == 0:
        st.markdown(
            f"""
        <div class="result-box risk-low">
            <h2 style="margin: 0;">✅ FAIBLE RISQUE D'AVC</h2>
            <p style="font-size: 16px; margin: 10px 0 0 0;">
                D'apres les informations fournies, le modele estime un
                faible risque d'AVC.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="result-box risk-high">
            <h2 style="margin: 0;">⚠️ RISQUE ELEVE D'AVC</h2>
            <p style="font-size: 16px; margin: 10px 0 0 0;">
                D'apres les informations fournies, le modele estime un
                risque eleve d'AVC.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Probabilites
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3 style="color: #28a745; margin-bottom: 5px;">Pas d'AVC</h3>
            <h1 style="color: #28a745; margin: 0;">{prob_no_stroke:.1f}%</h1>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_prob2:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3 style="color: #dc3545; margin-bottom: 5px;">Risque d'AVC</h3>
            <h1 style="color: #dc3545; margin: 0;">{prob_stroke:.1f}%</h1>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Facteurs de risque identifies
    st.markdown("---")
    st.subheader("🩺 Analyse des facteurs de risque")

    risk_factors = []

    if age >= 55:
        risk_factors.append(
            ("⏳ Age avance (≥55 ans)", "L'age est un facteur de risque majeur d'AVC.")
        )
    if hypertension == 1:
        risk_factors.append(
            ("🩸 Hypertension", "L'hypertension arterielle augmente significativement le risque.")
        )
    if heart_disease == 1:
        risk_factors.append(
            ("❤️ Maladie cardiaque", "Les maladies cardiaques sont associees a un risque accru.")
        )
    if avg_glucose_level > 140:
        risk_factors.append(
            ("🍬 Taux de glucose eleve (>140 mg/dL)", "L'hyperglycemie est un facteur de risque important.")
        )
    if bmi >= 30:
        risk_factors.append(
            ("⚖️ Obesite (IMC ≥ 30)", "Le surpoids et l'obesite augmentent le risque cardiovascularire.")
        )
    if smoking_status in ["smokes", "formerly smoked"]:
        risk_factors.append(
            ("🚬 Tabagisme", "Le tabac endommage les vaisseaux sanguins et augmente le risque.")
        )
    if gender == "Male":
        risk_factors.append(
            ("👨 Genre masculin", "Les hommes ont un risque legerement plus eleve.")
        )

    if risk_factors:
        for factor, desc in risk_factors:
            st.markdown(
                f"""
            <div class="info-box">
                <strong>{factor}</strong><br>
                {desc}
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.success(
            "✅ Aucun facteur de risque majeur identifie. "
            "Continuez a maintenir un mode de vie sain !"
        )

    # Resume du patient
    st.markdown("---")
    with st.expander("📝 Resume des donnees saisies"):
        summary_data = {
            "Caracteristique": [
                "Genre",
                "Age",
                "Hypertension",
                "Maladie cardiaque",
                "Marie(e)",
                "Type de travail",
                "Residence",
                "Glucose moyen (mg/dL)",
                "IMC (BMI)",
                "Tabagisme",
            ],
            "Valeur": [
                gender,
                f"{age} ans",
                "Oui" if hypertension == 1 else "Non",
                "Oui" if heart_disease == 1 else "Non",
                "Oui" if ever_married == "Yes" else "Non",
                work_type,
                residence_type,
                f"{avg_glucose_level:.1f}",
                f"{bmi:.1f}",
                smoking_status,
            ],
        }
        st.dataframe(
            pd.DataFrame(summary_data), hide_index=True, use_container_width=True
        )

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #95a5a6; font-size: 13px;">
    <p>🧠 Projet ML - Prediction du Risque d'AVC | Gradient Boosting Classifier</p>
    <p>Dataset : Kaggle Stroke Prediction Dataset (fedesoriano) | 5 110 patients</p>
    <p>⚠️ Cette application ne remplace pas un avis medical professionnel.</p>
</div>
""",
    unsafe_allow_html=True,
)