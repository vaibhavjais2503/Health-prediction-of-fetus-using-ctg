import os
import yaml

def create_config_file():
    config = {
        'database_path': 'data/fetal_health.db',  # optional now
        'model_paths': {
            'Random Forest': 'models/random_forest_model.pkl',
            'Gradient Boosting': 'models/gradient_boosting_model.pkl',
            'SVM': 'models/svm_model.pkl',
            'Neural Network': 'models/neural_network_model.keras'
        },
        'scaler_path': 'models/data_scaler.pkl'
    }
    os.makedirs('config', exist_ok=True)
    with open('config/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return config

def setup_directories():
    directories = ['data', 'models', 'logs', 'reports', 'uploads', 'backups']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    print("Creating directories...")
    setup_directories()

    print("Generating config.yaml...")
    create_config_file()

    print("\nSetup completed successfully!")
    print("Now place your trained models inside the models/ folder.")

if __name__ == "__main__":
    main()
