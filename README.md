# MLP Tabular Health Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-f7931e?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![License](https://img.shields.io/badge/License-Not%20Specified-lightgrey)

An interactive health prediction dashboard built with Python and Streamlit. The application integrates two pre-trained machine learning models, Multi-Layer Perceptron (MLP) and Random Forest, to compare predictions based on health-related indicators.

## Overview

This project explores how machine learning models can be integrated into an interactive application for tabular health data.

Users can enter health-related information, view predictions from both models, and explore model performance metrics through a simple dashboard.

The models were trained separately, and their saved artifacts are integrated into the local application.

**Important:** This project is for educational purposes only. It is not a clinically validated diagnostic tool.

## Features

- Interactive Streamlit dashboard
- Input form for 21 health-related features
- Pre-trained MLP neural network integration
- Pre-trained Random Forest classifier integration
- Side-by-side model prediction comparison
- Configured classification thresholds
- Model performance metrics and visualizations
- Reusable preprocessing pipeline
- Organized application, model, and report directories

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | MLP neural network inference |
| Scikit-learn | Preprocessing and Random Forest |
| Pandas | Data handling and metrics |
| NumPy | Numerical operations |
| Joblib | Loading saved model artifacts |
| Streamlit | Interactive web application |
| Matplotlib | Visualization support |

## Model Performance

The following results were obtained from the existing held-out test evaluation.

| Metric | MLP | Random Forest |
|---|---:|---:|
| Accuracy | 81.47% | 80.88% |
| Precision | 39.16% | 37.92% |
| Recall | 61.62% | 60.30% |
| F1-score | 47.89% | 46.56% |
| ROC-AUC | 83.70% | 82.53% |

The MLP achieved slightly better results across the reported metrics. However, the relatively low precision of both models means that false-positive predictions remain an important limitation.

These results reflect the available evaluation dataset and do not demonstrate clinical reliability.

## Project Structure

```text
mlp-tabular-health-prediction/
├── app/
│   └── app.py
├── models/
│   ├── health_mlp_best.pt
│   ├── model_metadata.json
│   ├── preprocessor.joblib
│   └── random_forest.joblib
├── reports/
│   └── test_metrics.csv
├── mlp_tabular_health_prediction.ipynb
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/mlp-tabular-health-prediction.git
cd mlp-tabular-health-prediction
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The local environment was tested with Scikit-learn 1.6.1 because the saved preprocessing artifact was created using that version.

If your `requirements.txt` does not pin the compatible version, install it with:

```bash
python -m pip install scikit-learn==1.6.1
```

### 4. Run the application

```bash
python -m streamlit run app/app.py
```

Streamlit will provide a local URL to open the dashboard in your browser.

## Dataset

The project uses the **CDC Diabetes Health Indicators** dataset available through the UCI Machine Learning Repository.

- Dataset: CDC Diabetes Health Indicators
- UCI Dataset ID: 891
- Records: 253,680
- Input features: 21
- Target: `Diabetes_binary`

Target categories:

- `0`: No diabetes
- `1`: Prediabetes or diabetes

Dataset source: [UCI Machine Learning Repository — CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

## Key Learning Outcomes

- Integrating saved machine learning models into a web application
- Managing Python dependencies and model compatibility
- Debugging preprocessing and artifact-loading errors
- Comparing classification metrics
- Developing interactive interfaces with Streamlit
- Organizing a machine learning project for maintainability

## Limitations

- Model performance depends on the dataset and training process.
- Precision is relatively low for both models.
- Model scores should not be interpreted as calibrated clinical probabilities.
- The application has not been clinically validated.
- Predictions must not be used as a substitute for professional medical advice.

## Future Improvements

- Add automated unit and integration tests.
- Improve input validation and error handling.
- Add model explainability using feature importance or SHAP.
- Evaluate calibration and performance across different demographic groups.
- Deploy the dashboard after testing its complete functionality.
