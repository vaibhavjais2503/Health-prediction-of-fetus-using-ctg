import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import tensorflow as tf
import joblib

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("\nAvailable devices:")
print(tf.config.list_physical_devices())
print("\nIgnore 303 errors")

def check_model_creation():
    models_folder = 'models'

    # Check if models folder is empty
    model_files = os.listdir(models_folder)
    if not model_files:
        print("\nModels folder is empty.")
        return

    # Get the current time
    current_time = time.time()

    # Check creation times of files in the folder
    creation_times = []
    for model_file in model_files:
        file_path = os.path.join(models_folder, model_file)
        if os.path.isfile(file_path):
            creation_time = os.path.getctime(file_path)
            creation_times.append(creation_time)

    # Compare creation times with the current time
    if all(abs(current_time - ct) <= 60 for ct in creation_times):
        print("\nModels have been successfully trained.")
    else:
        print("\nModels were not created at the same time or within 60 seconds of the program run.")

def train_models(data_path):
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # Use LabelEncoder to transform labels
    le = LabelEncoder()
    y = le.fit_transform(df['fetal_health'])

    # Separate features
    X = df.drop('fetal_health', axis=1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train)

    # SVM
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Neural Network
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    nn_model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    nn_model.fit(
        X_train_scaled, y_train, 
        epochs=50, 
        batch_size=32, 
        validation_split=0.2,
        verbose=0
    )

    # Save models and scaler
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(gb_model, 'models/gradient_boosting_model.pkl')
    joblib.dump(svm_model, 'models/svm_model.pkl')
    joblib.dump(scaler, 'models/data_scaler.pkl')
    nn_model.save('models/neural_network_model.keras')
    check_model_creation()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, '..', 'data')
    file_path = os.path.join(data_folder, 'fetal_health.csv')
    
    train_models(file_path)
