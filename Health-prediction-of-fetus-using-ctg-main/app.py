import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from security_utils import initialize_security, SecurityManager
import os
import io
import base64
import json
from datetime import datetime

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Global features list
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

# Store historical predictions in session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

def get_download_link(df, filename, text):
    """Generate a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def load_models():
    try:
        models = {
            'Random Forest': joblib.load('models/random_forest_model.pkl'),
            'Gradient Boosting': joblib.load('models/gradient_boosting_model.pkl'),
            'SVM': joblib.load('models/svm_model.pkl'),
            'Neural Network': tf.keras.models.load_model('models/neural_network_model.keras')
        }
        scaler = joblib.load('models/data_scaler.pkl')
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def validate_input(input_data):
    if len(input_data) != len(features):
        st.error(f"Please provide all {len(features)} features")
        return False
    if any(value is None for value in input_data):
        st.error("All input fields must be filled")
        return False
    return True

def predict_fetal_health(input_data, model, scaler, model_type):
    input_scaled = scaler.transform([input_data])
    health_categories = {0: "Normal", 1: "Suspicious", 2: "Pathological"}
    
    if model_type in ['Random Forest', 'Gradient Boosting', 'SVM']:
        prediction = int(model.predict(input_scaled)[0])
        proba = model.predict_proba(input_scaled)[0]
    else:  # Neural Network
        proba = model.predict(input_scaled)[0]
        prediction = np.argmax(proba)
    
    # Store prediction in history
    st.session_state.prediction_history.append({
        'timestamp': datetime.now(),
        'prediction': health_categories[prediction],
        'probabilities': proba.tolist(),
        'features': input_data
    })
    
    return health_categories[prediction], proba

def plot_probability_bar_chart(probabilities):
    categories = ["Normal", "Suspicious", "Pathological"]
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=probabilities,
            marker_color=['green', 'yellow', 'red']
        )
    ])
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Health Category",
        yaxis_title="Probability",
        yaxis_range=[0, 1]
    )
    return fig

def plot_radar_chart(input_data):
    fig = go.Figure(data=go.Scatterpolar(
        r=input_data,
        theta=features,
        fill='toself'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min(input_data), max(input_data)]
            )),
        showlegend=False,
        title="Feature Values Radar Chart"
    )
    return fig

def plot_prediction_history():
    if not st.session_state.prediction_history:
        return None
        
    history_df = pd.DataFrame(st.session_state.prediction_history)
    
    fig = go.Figure()
    categories = ["Normal", "Suspicious", "Pathological"]
    
    for i, category in enumerate(categories):
        probabilities = [p[i] for p in history_df['probabilities']]
        fig.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=probabilities,
            name=category,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Prediction History",
        xaxis_title="Time",
        yaxis_title="Probability",
        yaxis_range=[0, 1]
    )
    return fig

def plot_feature_correlation_heatmap(input_data):
    # Create a correlation matrix using the latest 50 predictions
    if len(st.session_state.prediction_history) < 2:
        return None
        
    recent_features = pd.DataFrame([p['features'] for p in st.session_state.prediction_history[-50:]], 
                                 columns=features)
    correlation_matrix = recent_features.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=features,
        y=features,
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        xaxis_title="Features",
        yaxis_title="Features"
    )
    return fig

def plot_feature_boxplots(input_data):
    if len(st.session_state.prediction_history) < 5:
        return None
        
    recent_features = pd.DataFrame([p['features'] for p in st.session_state.prediction_history[-50:]], 
                                 columns=features)
    
    fig = go.Figure()
    for feature in features:
        fig.add_trace(go.Box(
            y=recent_features[feature],
            name=feature,
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title="Feature Distribution Box Plots",
        xaxis_title="Features",
        yaxis_title="Values",
        showlegend=False,
        xaxis={'tickangle': 45}
    )
    return fig

def render_profile_page():
    st.header("User Profile")
    user = st.session_state.user
    
    # Display user information
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Account Information")
        st.write(f"Role: {user['role']}")
        st.write(f"User ID: {user['user_id']}")
        
        # Change password
        st.subheader("Change Password")
        old_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        if st.button("Update Password"):
            if new_password != confirm_password:
                st.error("New passwords don't match!")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters long!")
            else:
                success = st.session_state.security_manager.update_password(
                    user['user_id'],
                    old_password,
                    new_password
                )
                if success:
                    st.success("Password updated successfully!")
                else:
                    st.error("Failed to update password!")
    
    with col2:
        st.subheader("Activity Log")
        activity_log = st.session_state.security_manager.get_user_activity(user['user_id'])
        if activity_log:
            df = pd.DataFrame(activity_log)
            st.dataframe(df)

def render_batch_prediction():
    st.header("Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if st.button("Run Batch Prediction"):
            models, scaler = load_models()
            if models and scaler:
                results = []
                for _, row in df.iterrows():
                    input_data = [row[feature] for feature in features]
                    prediction, probabilities = predict_fetal_health(
                        input_data,
                        models['Random Forest'],
                        scaler,
                        'Random Forest'
                    )
                    results.append({
                        'prediction': prediction,
                        'confidence': max(probabilities)
                    })
                
                results_df = pd.DataFrame(results)
                st.write("Prediction Results:")
                st.dataframe(results_df)
                
                st.markdown(get_download_link(
                    results_df,
                    'predictions.csv',
                    'Download Predictions CSV'
                ), unsafe_allow_html=True)

def render_main_app():
    # Add logout and navigation
    menu_col, title_col, logout_col = st.columns([1, 4, 1])
    
    with menu_col:
        page = st.selectbox(
            "Navigation",
            ["Prediction", "Batch Processing", "Profile"]
        )
    
    with title_col:
        st.title("Fetal Health Prediction")
    
    with logout_col:
        if st.button("Logout"):
            st.session_state.security_manager.log_event(
                st.session_state.user['user_id'],
                'logout',
                'User logged out'
            )
            st.session_state.clear()
            st.rerun()
    
    # Render selected page
    if page == "Prediction":
        render_prediction_page()
    elif page == "Batch Processing":
        render_batch_prediction()
    elif page == "Profile":
        render_profile_page()

def render_prediction_page():
    # Load models
    models, scaler = load_models()
    if not models or not scaler:
        st.error("Failed to load models")
        return
    
    # Create layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.header("Input Parameters")
        input_data = []
        for feature in features:
            value = st.number_input(
                feature.replace('_', ' ').title(),
                step=0.1,
                format="%.2f"
            )
            input_data.append(value)
    
    with col2:
        st.header("Model Selection")
        model_choice = st.selectbox(
            "Select Model",
            list(models.keys())
        )
        
        if st.button("Predict Fetal Health"):
            if validate_input(input_data):
                st.session_state.security_manager.log_event(
                    st.session_state.user['user_id'],
                    'prediction',
                    f'Model used: {model_choice}'
                )
                
                prediction, probabilities = predict_fetal_health(
                    input_data,
                    models[model_choice],
                    scaler,
                    model_choice
                )
                
                st.success(f"Predicted Fetal Health: {prediction}")
                
                tabs = st.tabs([
                    "Probability Chart",
                    "Feature Radar",
                    "Prediction History",
                    "Feature Correlations",
                    "Feature Distributions"
                ])
                
                with tabs[0]:
                    st.plotly_chart(plot_probability_bar_chart(probabilities))
                
                with tabs[1]:
                    st.plotly_chart(plot_radar_chart(input_data))
                
                with tabs[2]:
                    history_chart = plot_prediction_history()
                    if history_chart:
                        st.plotly_chart(history_chart)
                    else:
                        st.info("Make more predictions to see the history chart")
                
                with tabs[3]:
                    correlation_chart = plot_feature_correlation_heatmap(input_data)
                    if correlation_chart:
                        st.plotly_chart(correlation_chart)
                    else:
                        st.info("Make more predictions to see the correlation heatmap")
                
                with tabs[4]:
                    boxplot_chart = plot_feature_boxplots(input_data)
                    if boxplot_chart:
                        st.plotly_chart(boxplot_chart)
                    else:
                        st.info("Make more predictions to see the feature distributions")

def render_login():
    st.title("Login")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            success, user_data = st.session_state.security_manager.verify_user(username, password)
            if success:
                st.session_state.user = user_data
                st.session_state.authenticated = True
                st.session_state.security_manager.log_event(
                    user_data['user_id'],
                    'login',
                    'Successful login'
                )
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with col2:
        st.header("Reset Password")
        reset_username = st.text_input("Enter Username", key="reset_username")
        if st.button("Reset Password"):
            if reset_username:
                success = st.session_state.security_manager.initiate_password_reset(
                    reset_username
                )
                if success:
                    st.success("Password reset instructions sent!")
                else:
                    st.error("Username not found!")

def main():
    # Initialize security
    initialize_security()
    
    # Check authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        render_login()
    else:
        render_main_app()

if __name__ == "__main__":
    main()