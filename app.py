"""
===========================================
  Stroke Prediction - Interface Streamlit
  Projet ML : Prédiction du Risque d'AVC
===========================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import importlib
import sys

# ============================================================
# Configuration de la page
# ============================================================
st.set_page_config(
    page_title="Prédiction AVC",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS personnalisé
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
# Chargement du modèle (robuste - compatible tous Python/sklearn)
# ============================================================
def _apply_sklearn_pickle_compat():
    """
    Patch de compatibilité pour les pickle sklearn générés sur une
    version différente de Python ou scikit-learn.
    Résout : AttributeError: module 'sklearn._loss._loss' has no
    attribute '__pyx_unpickle_CyHalfBinomialLoss'
    """
    # Modules potentiels où le unpickler Cython peut manquer
    _module_paths = [
        'sklearn._loss._loss',
        'sklearn._loss.loss',
    ]
    for _mod_path in _module_paths:
        try:
            _mod = importlib.import_module(_mod_path)
        except (ImportError, ModuleNotFoundError):
            continue

        # Liste des noms de unpickler Cython à patcher
        _unpickler_names = [
            '__pyx_unpickle_CyHalfBinomialLoss',
            '__pyx_unpickle_HalfBinomialLoss',
        ]
        for _uname in _unpickler_names:
            if hasattr(_mod, _uname):
                continue  # Déjà présent, pas besoin de patcher

            def _make_cython_unpickler_fallback():
                """Crée un fallback unpickler compatible Cython."""
                def _fallback(__pyx_type, __pyx_state):
                    obj = __pyx_type.__new__(__pyx_type)
                    if __pyx_state is not None:
                        if hasattr(obj, '__setstate_cython__'):
                            obj.__setstate_cython__(__pyx_state)
                        elif hasattr(obj, '__setstate__'):
                            obj.__setstate__(__pyx_state)
                    return obj
                return _fallback

            setattr(_mod, _uname, _make_cython_unpickler_fallback())


def _train_model_fallback():
    """
    Entraîne le modèle à l'exécution si le pickle échoue.
    Résultat identique grâce à random_state=42.
    """
    import urllib.request
    import io
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    url = "https://raw.githubusercontent.com/Saikrishna-89/STROKE-PREDICTION-USING-HEALTHCARE-DATA/main/healthcare-dataset-stroke-data.csv"
    response = urllib.request.urlopen(url)
    df = pd.read_csv(io.BytesIO(response.read()))

    # Même preprocessing que le notebook
    df = df.drop(columns=['id'])
    df = df[df['gender'] != 'Other'].reset_index(drop=True)
    bmi_med = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(bmi_med)

    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1})
    df['Residence_type'] = df['Residence_type'].map({'Rural': 0, 'Urban': 1})
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)

    X = df.drop('stroke', axis=1)
    y = df['stroke']
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        random_state=42, learning_rate=0.1, max_depth=5, n_estimators=200
    )
    model.fit(X_train, y_train)

    return {
        'model': model,
        'bmi_median': bmi_med,
        'feature_columns': X.columns.tolist(),
    }


@st.cache_resource
def load_model():
    """
    Charge le modèle avec 3 niveaux de fallback :
    1) pickle direct
    2) pickle avec patch de compatibilité
    3) réentraînement à l'exécution
    """
    # --- Tentative 1 : pickle direct ---
    try:
        with open("stroke_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        return model_data
    except Exception:
        pass

    # --- Tentative 2 : patch + pickle ---
    try:
        _apply_sklearn_pickle_compat()
        with open("stroke_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        return model_data
    except Exception:
        pass

    # --- Tentative 3 : réentraînement ---
    return _train_model_fallback()


model_data = load_model()
model = model_data['model']
bmi_median = model_data['bmi_median']
feature_columns = model_data['feature_columns']

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1 style="color: #2c3e50;">🧠 Prédiction du Risque d'AVC</h1>
    <p style="font-size: 18px; color: #7f8c8d;">
        Application de <strong>Machine Learning</strong> pour prédire le risque d'accident vasculaire cérébral
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# Sidebar - Informations
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
    st.title("À propos")
    st.markdown("""
    Cette application utilise un modèle de **Gradient Boosting** entraîné 
    sur le dataset **Stroke Prediction Dataset** (Kaggle - fedesoriano) 
    contenant **5 110 patients**.
    
    **Variables utilisées :**
    - Genre
    - Âge
    - Hypertension
    - Maladie cardiaque
    - État marital
    - Type de travail
    - Type de résidence
    - Taux de glucose moyen
    - IMC (BMI)
    - Statut tabagique
    
    ---
    ⚠️ **Avertissement** : Cette application est un outil d'aide 
    à la décision et ne remplace en aucun cas un diagnostic médical 
    professionnel.
    """)
    
    st.markdown("---")
    st.markdown("""
    **Modèle :** Gradient Boosting  
    **Accuracy :** ~94.3%  
    **Métrique prioritaire :** Recall  
    **Dataset :** 5 110 patients
    """)

# ============================================================
# Formulaire de saisie
# ============================================================
st.subheader("📋 Informations du Patient")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Données démographiques")
    gender = st.selectbox(
        "Genre",
        options=["Male", "Female"],
        format_func=lambda x: "👨 Homme" if x == "Male" else "👩 Femme",
        index=0
    )
    age = st.slider("Âge", min_value=0, max_value=100, value=45, step=1)
    ever_married = st.selectbox(
        "Marié(e) / A déjà été marié(e)",
        options=["Yes", "No"],
        format_func=lambda x: "✅ Oui" if x == "Yes" else "❌ Non",
        index=0
    )
    work_type = st.selectbox(
        "Type de travail",
        options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        format_func=lambda x: {
            "Private": "🏢 Privé",
            "Self-employed": "🧑‍💼 Indépendant",
            "Govt_job": "🏛️ Fonction publique",
            "children": "👶 Enfant",
            "Never_worked": "🚫 N'a jamais travaillé"
        }[x],
        index=0
    )
    residence_type = st.selectbox(
        "Type de résidence",
        options=["Urban", "Rural"],
        format_func=lambda x: "🏙️ Urbain" if x == "Urban" else "🌾 Rural",
        index=0
    )

with col2:
    st.markdown("#### Données médicales")
    hypertension = st.selectbox(
        "Hypertension",
        options=[0, 1],
        format_func=lambda x: "✅ Oui" if x == 1 else "❌ Non",
        index=0
    )
    heart_disease = st.selectbox(
        "Maladie cardiaque",
        options=[0, 1],
        format_func=lambda x: "✅ Oui" if x == 1 else "❌ Non",
        index=0
    )
    avg_glucose_level = st.slider(
        "Taux moyen de glucose (mg/dL)",
        min_value=50.0,
        max_value=300.0,
        value=100.0,
        step=0.1
    )
    bmi = st.slider(
        "IMC (BMI)",
        min_value=10.0,
        max_value=100.0,
        value=28.0,
        step=0.1
    )
    smoking_status = st.selectbox(
        "Statut tabagique",
        options=["never smoked", "formerly smoked", "smokes", "Unknown"],
        format_func=lambda x: {
            "never smoked": "🚭 Jamais fumé",
            "formerly smoked": "🚬 Ancien fumeur",
            "smokes": "🔥 Fumeur",
            "Unknown": "❓ Inconnu"
        }[x],
        index=0
    )

# ============================================================
# Bouton de prédiction
# ============================================================
st.markdown("---")
predict_button = st.button("🔍 Prédire le risque d'AVC", type="primary", use_container_width=True)

if predict_button:
    # --------------------------------------------------------
    # Préparation des données (même pipeline que le notebook)
    # --------------------------------------------------------
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'work_type': work_type,
        'smoking_status': smoking_status
    }

    # Création du DataFrame
    df_input = pd.DataFrame([input_data])

    # Encodage (même logique que le notebook)
    df_input['gender'] = df_input['gender'].map({'Male': 0, 'Female': 1})
    df_input['ever_married'] = df_input['ever_married'].map({'No': 0, 'Yes': 1})
    df_input['Residence_type'] = df_input['Residence_type'].map({'Rural': 0, 'Urban': 1})

    # One-Hot Encoding
    df_input = pd.get_dummies(df_input, columns=['work_type', 'smoking_status'], drop_first=True)

    # S'assurer que toutes les colonnes du modèle sont présentes
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    # Réordonner les colonnes
    df_input = df_input[feature_columns]

    # --------------------------------------------------------
    # Prédiction
    # --------------------------------------------------------
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0]

    prob_no_stroke = probability[0] * 100
    prob_stroke = probability[1] * 100

    # --------------------------------------------------------
    # Affichage des résultats
    # --------------------------------------------------------
    st.markdown("## 📊 Résultat de la prédiction")

    if prediction == 0:
        st.markdown(f"""
        <div class="result-box risk-low">
            <h2 style="margin: 0;">✅ FAIBLE RISQUE D'AVC</h2>
            <p style="font-size: 16px; margin: 10px 0 0 0;">
                D'après les informations fournies, le modèle estime un faible risque d'AVC.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box risk-high">
            <h2 style="margin: 0;">⚠️ RISQUE ÉLEVÉ D'AVC</h2>
            <p style="font-size: 16px; margin: 10px 0 0 0;">
                D'après les informations fournies, le modèle estime un risque élevé d'AVC.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Probabilités
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #28a745; margin-bottom: 5px;">Pas d'AVC</h3>
            <h1 style="color: #28a745; margin: 0;">{prob_no_stroke:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    with col_prob2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #dc3545; margin-bottom: 5px;">Risque d'AVC</h3>
            <h1 style="color: #dc3545; margin: 0;">{prob_stroke:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    # Facteurs de risque identifiés
    st.markdown("---")
    st.subheader("🩺 Analyse des facteurs de risque")

    risk_factors = []

    if age >= 55:
        risk_factors.append(("⏳ Âge avancé (≥55 ans)", "L'âge est un facteur de risque majeur d'AVC."))
    if hypertension == 1:
        risk_factors.append(("🩸 Hypertension", "L'hypertension artérielle augmente significativement le risque."))
    if heart_disease == 1:
        risk_factors.append(("❤️ Maladie cardiaque", "Les maladies cardiaques sont associées à un risque accru."))
    if avg_glucose_level > 140:
        risk_factors.append(("🍬 Taux de glucose élevé (>{140} mg/dL)", "L'hyperglycémie est un facteur de risque important."))
    if bmi >= 30:
        risk_factors.append(("⚖️ Obésité (IMC ≥ 30)", "Le surpoids et l'obésité augmentent le risque cardiovasculaire."))
    if smoking_status in ["smokes", "formerly smoked"]:
        risk_factors.append(("🚬 Tabagisme", "Le tabac endommage les vaisseaux sanguins et augmente le risque."))
    if gender == "Male":
        risk_factors.append(("👨 Genre masculin", "Les hommes ont un risque légèrement plus élevé."))

    if risk_factors:
        for factor, desc in risk_factors:
            st.markdown(f"""
            <div class="info-box">
                <strong>{factor}</strong><br>
                {desc}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Aucun facteur de risque majeur identifié. Continuez à maintenir un mode de vie sain !")

    # Résumé du patient
    st.markdown("---")
    with st.expander("📝 Résumé des données saisies"):
        summary_data = {
            "Caractéristique": [
                "Genre", "Âge", "Hypertension", "Maladie cardiaque",
                "Marié(e)", "Type de travail", "Résidence",
                "Glucose moyen (mg/dL)", "IMC (BMI)", "Tabagisme"
            ],
            "Valeur": [
                gender, f"{age} ans", "Oui" if hypertension == 1 else "Non",
                "Oui" if heart_disease == 1 else "Non",
                "Oui" if ever_married == "Yes" else "Non",
                work_type, residence_type,
                f"{avg_glucose_level:.1f}", f"{bmi:.1f}", smoking_status
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; font-size: 13px;">
    <p>🧠 Projet ML - Prédiction du Risque d'AVC | Gradient Boosting Classifier</p>
    <p>Dataset : Kaggle Stroke Prediction Dataset (fedesoriano) | 5 110 patients</p>
    <p>⚠️ Cette application ne remplace pas un avis médical professionnel.</p>
</div>
""", unsafe_allow_html=True)
