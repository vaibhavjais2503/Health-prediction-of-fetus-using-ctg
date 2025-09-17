import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import pandas as pd
import base64
from datetime import datetime
from pathlib import Path

# ---------- Paths (safe for Streamlit Cloud & local) ----------
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
...

DATA_DIR   = BASE_DIR / "data"
LOGS_DIR   = BASE_DIR / "logs"

for d in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- quick visibility so you can confirm the right file is running ---
with st.sidebar:
    st.caption(f"Running: {Path(__file__).as_posix()}")
    st.caption(f"Models dir: {MODELS_DIR.as_posix()}")
    st.caption(f"Models present: {[p.name for p in MODELS_DIR.glob('*')]}")

# CTG feature list (order must match your training)
features = [
    "baseline_value", "accelerations", "fetal_movement",
    "uterine_contractions", "light_decelerations",
    "severe_decelerations", "prolongued_decelerations",
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width", "histogram_min", "histogram_max",
    "histogram_number_of_peaks", "histogram_number_of_zeroes",
    "histogram_mode", "histogram_mean", "histogram_median",
    "histogram_variance", "histogram_tendency"
]

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

def load_models():
    try:
        rf_path  = MODELS_DIR / "random_forest_model.pkl"
        gb_path  = MODELS_DIR / "gradient_boosting_model.pkl"
        svm_path = MODELS_DIR / "svm_model.pkl"
        nn_path  = MODELS_DIR / "neural_network_model.keras"
        sc_path  = MODELS_DIR / "data_scaler.pkl"

        models = {
            'Random Forest': joblib.load(rf_path),
            'Gradient Boosting': joblib.load(gb_path),
            'SVM': joblib.load(svm_path),
            'Neural Network': tf.keras.models.load_model(nn_path)
        }
        scaler = joblib.load(sc_path)
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def validate_input(input_data):
    if len(input_data) != len(features):
        st.error(f"Please provide all {len(features)} features")
        return False
    if any(v is None for v in input_data):
        st.error("All input fields must be filled")
        return False
    return True

def predict_fetal_health(input_data, model, scaler, model_type):
    X = scaler.transform([input_data])
    labels = {0: "Normal", 1: "Suspicious", 2: "Pathological"}

    if model_type in ['Random Forest', 'Gradient Boosting', 'SVM']:
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
    else:
        proba = model.predict(X)[0]
        pred = int(np.argmax(proba))

    st.session_state.prediction_history.append({
        'timestamp': datetime.now(),
        'prediction': labels[pred],
        'probabilities': proba.tolist(),
        'features': input_data
    })
    return labels[pred], proba

def plot_probability_bar_chart(probabilities):
    fig = go.Figure([go.Bar(x=["Normal","Suspicious","Pathological"], y=probabilities)])
    fig.update_layout(title="Prediction Probabilities", yaxis_range=[0,1])
    return fig

def plot_radar_chart(input_data):
    fig = go.Figure(go.Scatterpolar(r=input_data, theta=features, fill='toself'))
    fig.update_layout(showlegend=False, title="Feature Values Radar Chart")
    return fig

def plot_prediction_history():
    if not st.session_state.prediction_history: return None
    df = pd.DataFrame(st.session_state.prediction_history)
    fig = go.Figure()
    for i, name in enumerate(["Normal","Suspicious","Pathological"]):
        y = [p[i] for p in df['probabilities']]
        fig.add_trace(go.Scatter(x=df['timestamp'], y=y, mode='lines+markers', name=name))
    fig.update_layout(title="Prediction History", yaxis_range=[0,1])
    return fig

def plot_feature_correlation_heatmap():
    if len(st.session_state.prediction_history) < 2: return None
    feats = pd.DataFrame([p['features'] for p in st.session_state.prediction_history[-50:]], columns=features)
    corr = feats.corr()
    return go.Figure(data=go.Heatmap(z=corr, x=features, y=features))

def plot_feature_boxplots():
    if len(st.session_state.prediction_history) < 5: return None
    feats = pd.DataFrame([p['features'] for p in st.session_state.prediction_history[-50:]], columns=features)
    fig = go.Figure()
    for f in features:
        fig.add_trace(go.Box(y=feats[f], name=f, boxpoints='outliers'))
    fig.update_layout(showlegend=False, title="Feature Distributions")
    return fig

def render_batch_prediction():
    st.header("Batch Prediction")
    file = st.file_uploader("Upload CSV file", type="csv")
    if file is not None and st.button("Run Batch Prediction"):
        df = pd.read_csv(file)
        models, scaler = load_models()
        if models and scaler:
            rows = []
            for _, r in df.iterrows():
                x = [r[f] for f in features]
                pred, prob = predict_fetal_health(x, models['Random Forest'], scaler, 'Random Forest')
                rows.append({'prediction': pred, 'confidence': float(np.max(prob))})
            out = pd.DataFrame(rows)
            st.dataframe(out)
            st.markdown(get_download_link(out, 'predictions.csv', 'Download Predictions CSV'),
                        unsafe_allow_html=True)

def render_prediction_page():
    models, scaler = load_models()
    if not models or not scaler:
        st.error("Failed to load models")
        return

    col1, col2 = st.columns([1,3])
    with col1:
        st.header("Input Parameters")
        x = [st.number_input(f.replace('_',' ').title(), step=0.1, format="%.2f")
             for f in features]

    with col2:
        st.header("Model Selection")
        model_choice = st.selectbox("Select Model", list(models.keys()))
        if st.button("Predict Fetal Health"):
            if validate_input(x):
                pred, prob = predict_fetal_health(x, models[model_choice], scaler, model_choice)
                st.success(f"Predicted Fetal Health: {pred}")
                tabs = st.tabs(["Probability Chart","Feature Radar","Prediction History","Feature Correlations","Feature Distributions"])
                with tabs[0]: st.plotly_chart(plot_probability_bar_chart(prob), use_container_width=True)
                with tabs[1]: st.plotly_chart(plot_radar_chart(x), use_container_width=True)
                with tabs[2]:
                    h = plot_prediction_history()
                    st.plotly_chart(h, use_container_width=True) if h else st.info("Make more predictions to see history")
                with tabs[3]:
                    c = plot_feature_correlation_heatmap()
                    st.plotly_chart(c, use_container_width=True) if c else st.info("More predictions needed for correlations")
                with tabs[4]:
                    b = plot_feature_boxplots()
                    st.plotly_chart(b, use_container_width=True) if b else st.info("More predictions needed for distributions")

def main():
    st.title("Fetal Health Prediction")
    page = st.sidebar.selectbox("Navigation", ["Prediction", "Batch Processing"])
    if page == "Prediction":
        render_prediction_page()
    else:
        render_batch_prediction()

if __name__ == "__main__":
    main()
