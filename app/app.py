import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

with open(MODEL_DIR / "model_metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

FEATURES = metadata["input_features"]
MLP_THRESHOLD = float(metadata["mlp"]["threshold"])
RF_THRESHOLD = float(metadata["random_forest"]["threshold"])
DEVICE = torch.device("cpu")


# --------------------------------------------------
# Model architecture
# --------------------------------------------------
class HealthMLP(nn.Module):
    def __init__(self, input_dim=21):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    rf_model = joblib.load(MODEL_DIR / "random_forest.joblib")

    mlp = HealthMLP(input_dim=len(FEATURES))
    checkpoint = torch.load(
        MODEL_DIR / "health_mlp_best.pt",
        map_location=DEVICE,
        weights_only=True,
    )
    mlp.load_state_dict(checkpoint)
    mlp.eval()

    return preprocessor, rf_model, mlp


# --------------------------------------------------
# Page Configuration & Advanced Enterprise Theme
# --------------------------------------------------
st.set_page_config(
    page_title="Health Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    /* Hero Banner */
    .hero-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 2.25rem 2.5rem;
        margin-top: 2.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }

    .hero-category {
        color: #2563eb;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-container h1 {
        color: #0f172a;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0 0 0.6rem 0;
    }

    .hero-container p {
        color: #475569;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.6;
    }

    /* Compact Disclaimer Callout */
    .disclaimer-callout {
        background: #fffbe2;
        border: 1px solid #fef08a;
        border-left: 4px solid #eab308;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin-bottom: 2rem;
    }

    .disclaimer-text {
        color: #713f12;
        font-size: 0.825rem;
        line-height: 1.4;
    }

    .disclaimer-tag {
        font-weight: 700;
        color: #854d0e;
        letter-spacing: 0.3px;
    }

    /* Compact Info/Warning Alerts */
    div.stInfo, div.stWarning {
        padding: 0.4rem 0.75rem !important;
        border-radius: 6px !important;
        font-size: 0.875rem !important;
    }

    /* Section Typography */
    .section-title {
        color: #0f172a;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        margin: 2rem 0 0.25rem 0;
    }

    .section-desc {
        color: #64748b;
        font-size: 0.88rem;
        margin-bottom: 1.25rem;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #1e40af !important;
        font-weight: 700 !important;
        font-size: 1.85rem !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre;
        border-radius: 6px;
        color: #64748b;
        font-weight: 500;
        font-size: 0.9rem;
        padding: 0 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1e40af !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }

    /* Submit Button */
    .stButton > button, .stFormSubmitButton > button {
        background-color: #1e40af !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 1.5rem !important;
        box-shadow: 0 4px 6px -1px rgba(30, 64, 175, 0.2);
        transition: all 0.2s ease;
    }

    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 6px 8px -1px rgba(30, 64, 175, 0.3);
    }

    /* Result Cards */
    .enterprise-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }

    .card-header {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.75rem;
    }

    .card-score {
        color: #0f172a;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.03em;
    }

    .card-threshold {
        color: #64748b;
        font-size: 0.825rem;
        margin-bottom: 1.25rem;
    }

    .status-tag-positive {
        display: inline-block;
        background-color: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.825rem;
        font-weight: 600;
    }

    .status-tag-negative {
        display: inline-block;
        background-color: #f0fdf4;
        color: #16a34a;
        border: 1px solid #bbf7d0;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.825rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-category">Research Infrastructure</div>
        <h1>Diagnostic Assessment Platform</h1>
        <p>
            Comparative evaluation workspace processing 21 multi-modal clinical and lifestyle 
            parameters across neural network and ensemble classification models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Compact Disclaimer Callout (No Icon)
st.markdown(
    """
    <div class="disclaimer-callout">
        <div class="disclaimer-text">
            <span class="disclaimer-tag">RESEARCH USE ONLY:</span> 
            This workspace is configured for empirical model comparison and does not constitute verified clinical medical decisions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Key Overview Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Input Features", len(FEATURES))
with col2:
    st.metric("Active Models", 2)
with col3:
    st.metric("MLP Test ROC-AUC", "0.837")

st.markdown('<div class="section-title">Patient Profile Input Configuration</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Select indicators across clinical history, daily habits, and demographics.</div>', unsafe_allow_html=True)


# --------------------------------------------------
# Input labels and options
# --------------------------------------------------
YES_NO_LABELS = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol check within 5 years",
    "Smoker": "Smoked at least 100 cigarettes in life",
    "Stroke": "History of stroke",
    "HeartDiseaseorAttack": "Coronary heart disease or heart attack",
    "PhysActivity": "Physical activity outside work",
    "Fruits": "Consume fruit at least once daily",
    "Veggies": "Consume vegetables at least once daily",
    "HvyAlcoholConsump": "Heavy alcohol consumption status",
    "AnyHealthcare": "Active healthcare coverage",
    "NoDocbcCost": "Skipped physician visit due to cost",
    "DiffWalk": "Difficulty walking or climbing stairs",
}

AGE_OPTIONS = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39",
    5: "40–44", 6: "45–49", 7: "50–54", 8: "55–59",
    9: "60–64", 10: "65–69", 11: "70–74",
    12: "75–79", 13: "80 or older",
}

EDUCATION_OPTIONS = {
    1: "Never attended / Kindergarten only",
    2: "Grades 1–8",
    3: "Grades 9–11",
    4: "High school graduate",
    5: "College / technical school (1–3 years)",
    6: "College graduate",
}

INCOME_OPTIONS = {
    1: "Less than $10,000",
    2: "$10,000 – $14,999",
    3: "$15,000 – $19,999",
    4: "$20,000 – $24,999",
    5: "$25,000 – $34,999",
    6: "$35,000 – $49,999",
    7: "$50,000 – $74,999",
    8: "$75,000 or more",
}


def yes_no_input(feature):
    return int(
        st.selectbox(
            YES_NO_LABELS[feature],
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            key=f"input_{feature}",
        )
    )


# --------------------------------------------------
# Form with Structured Tabs Layout
# --------------------------------------------------
with st.form("health_input_form"):
    values = {}
    
    tab1, tab2, tab3 = st.tabs([
        "01. Clinical & Medical History",
        "02. Lifestyle & Physical Vitals",
        "03. Demographic Parameters"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            for feature in ["HighBP", "HighChol", "CholCheck", "Stroke"]:
                values[feature] = yes_no_input(feature)
        with c2:
            for feature in ["HeartDiseaseorAttack", "DiffWalk"]:
                values[feature] = yes_no_input(feature)

            values["GenHlth"] = int(
                st.selectbox(
                    "General health self-assessment",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: {
                        1: "1 - Excellent",
                        2: "2 - Very good",
                        3: "3 - Good",
                        4: "4 - Fair",
                        5: "5 - Poor",
                    }[x],
                )
            )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            for feature in ["Smoker", "PhysActivity", "Fruits", "Veggies"]:
                values[feature] = yes_no_input(feature)

            values["BMI"] = float(
                st.number_input(
                    "Body Mass Index (BMI)",
                    min_value=12.0,
                    max_value=98.0,
                    value=25.0,
                    step=0.5,
                )
            )

        with c2:
            for feature in ["HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost"]:
                values[feature] = yes_no_input(feature)

            values["MentHlth"] = int(
                st.number_input("Poor mental health days (past 30 days)", min_value=0, max_value=30, value=0)
            )
            values["PhysHlth"] = int(
                st.number_input("Poor physical health days (past 30 days)", min_value=0, max_value=30, value=0)
            )

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            values["Sex"] = int(
                st.selectbox(
                    "Biological sex",
                    options=[0, 1],
                    format_func=lambda x: {0: "Female", 1: "Male"}[x],
                )
            )
            values["Age"] = int(
                st.selectbox(
                    "Age category",
                    options=list(AGE_OPTIONS),
                    format_func=lambda x: AGE_OPTIONS[x],
                )
            )
        with c2:
            values["Education"] = int(
                st.selectbox(
                    "Education level",
                    options=list(EDUCATION_OPTIONS),
                    format_func=lambda x: EDUCATION_OPTIONS[x],
                )
            )
            values["Income"] = int(
                st.selectbox(
                    "Household income bracket",
                    options=list(INCOME_OPTIONS),
                    format_func=lambda x: INCOME_OPTIONS[x],
                )
            )

    st.write("")
    submitted = st.form_submit_button("Run Diagnostic Evaluation", use_container_width=True)


# --------------------------------------------------
# Prediction Results
# --------------------------------------------------
if submitted:
    try:
        input_df = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)

        with st.spinner("Processing tensor pipelines..."):
            preprocessor, rf_model, mlp = load_artifacts()

            processed = np.asarray(preprocessor.transform(input_df), dtype=np.float32)

            with torch.no_grad():
                input_tensor = torch.from_numpy(processed)
                mlp_score = float(torch.sigmoid(mlp(input_tensor)).item())

            rf_probabilities = rf_model.predict_proba(input_df)[0]
            positive_index = list(rf_model.classes_).index(1)
            rf_score = float(rf_probabilities[positive_index])

        mlp_prediction = int(mlp_score >= MLP_THRESHOLD)
        rf_prediction = int(rf_score >= RF_THRESHOLD)

        st.markdown('<div class="section-title">Comparative Model Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Independent predictions produced by neural and ensemble architectures.</div>', unsafe_allow_html=True)

        left, right = st.columns(2)

        with left:
            tag = '<span class="status-tag-positive">Risk Indicator Positive</span>' if mlp_prediction else '<span class="status-tag-negative">Normal Risk Profile</span>'
            st.markdown(
                f"""
                <div class="enterprise-card">
                    <div class="card-header">Neural Network (PyTorch MLP)</div>
                    <div class="card-score">{mlp_score:.1%}</div>
                    <div class="card-threshold">Decision Threshold: {MLP_THRESHOLD:.2f}</div>
                    {tag}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(mlp_score)

        with right:
            tag = '<span class="status-tag-positive">Risk Indicator Positive</span>' if rf_prediction else '<span class="status-tag-negative">Normal Risk Profile</span>'
            st.markdown(
                f"""
                <div class="enterprise-card">
                    <div class="card-header">Ensemble (Random Forest)</div>
                    <div class="card-score">{rf_score:.1%}</div>
                    <div class="card-threshold">Decision Threshold: {RF_THRESHOLD:.2f}</div>
                    {tag}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(rf_score)

        st.write("")
        if mlp_prediction == rf_prediction:
            st.info("Model Consensus: Both classification models returned matching evaluation outputs.")
        else:
            st.warning("Model Divergence: Output predictions vary between neural network and random forest models.")

        # Benchmark Metrics
        st.markdown('<div class="section-title">Baseline Evaluation Metrics</div>', unsafe_allow_html=True)
        metrics_path = REPORT_DIR / "test_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            st.dataframe(metrics, use_container_width=True, hide_index=True)
        else:
            st.info("Test metrics dataset unavailable.")

    except Exception as e:
        st.error("Model execution failed.")
        st.exception(e)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; padding-top:3rem; border-top: 1px solid #e2e8f0; margin-top:3.5rem;">
        <p style="color:#64748b; font-size:0.85rem; margin-bottom:0.25rem;">
            Health Analytics Intelligence Platform • Enterprise Environment
        </p>
        <p style="color:#94a3b8; font-size:0.75rem;">
            Dataset: UCI CDC Diabetes Health Indicators
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)