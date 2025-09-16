# setup_database.py
import sqlite3
import os
import secrets
import base64
from cryptography.fernet import Fernet
import yaml
import hashlib
from datetime import datetime

def generate_secret_key():
    """Generate a secure secret key for encryption"""
    return base64.urlsafe_b64encode(secrets.token_bytes(32))

def create_config_file():
    """Create the initial configuration file with security settings"""
    config = {
        'secret_key': generate_secret_key().decode('utf-8'),
        'encryption_key': Fernet.generate_key().decode('utf-8'),
        'database_path': 'data/fetal_health.db',
        'model_paths': {
            'Random Forest': 'models/random_forest_model.pkl',
            'Gradient Boosting': 'models/gradient_boosting_model.pkl',
            'SVM': 'models/svm_model.pkl',
            'Neural Network': 'models/neural_network_model.keras'
        },
        'scaler_path': 'models/data_scaler.pkl',
        'auth': {
            'session_duration': 3600,
            'max_login_attempts': 3,
            'password_min_length': 8
        }
    }
    
    # Create config directory if it doesn't exist
    os.makedirs('config', exist_ok=True)
    
    # Write configuration to file
    with open('config/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config

def setup_database(config):
    """Initialize the database with required tables"""
    db_path = config['database_path']
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database and create tables
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            failed_login_attempts INTEGER DEFAULT 0,
            account_locked BOOLEAN DEFAULT 0,
            UNIQUE(username)
        )
        """)
        
        # Create predictions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            patient_id TEXT,
            input_data BLOB NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_version TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """)
        
        # Create audit log table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            action TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """)
        
        # Create session table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_token TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """)
        
        # Create role permissions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS role_permissions (
            role TEXT NOT NULL,
            permission TEXT NOT NULL,
            PRIMARY KEY (role, permission)
        )
        """)
        
        # Insert default roles and permissions
        default_roles = [
            ('admin', 'all'),
            ('doctor', 'predict'),
            ('doctor', 'view_history'),
            ('doctor', 'export_reports'),
            ('nurse', 'predict'),
            ('nurse', 'view_history')
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO role_permissions (role, permission) VALUES (?, ?)",
            default_roles
        )
        
        conn.commit()

def create_admin_user(config):
    """Create the initial admin user"""
    import getpass
    
    print("Creating admin user...")
    username = input("Enter admin username: ")
    password = getpass.getpass("Enter admin password: ")
    
    # Generate salt
    salt = secrets.token_hex(16)
    
    # Hash password with salt
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    
    # Connect to database and insert admin user
    with sqlite3.connect(config['database_path']) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO users (id, username, password_hash, salt, role)
        VALUES (?, ?, ?, ?, ?)
        """, (
            secrets.token_hex(16),
            username,
            password_hash,
            salt,
            'admin'
        ))
        conn.commit()

def setup_directories():
    """Create necessary directories for the application"""
    directories = [
        'data',
        'models',
        'logs',
        'reports',
        'uploads',
        'backups'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    """Main setup function"""
    print("Starting setup...")
    
    # Create directories
    print("Creating directories...")
    setup_directories()
    
    # Create configuration
    print("Generating configuration...")
    config = create_config_file()
    
    # Setup database
    print("Setting up database...")
    setup_database(config)
    
    # Create admin user
    create_admin_user(config)
    
    print("\nSetup completed successfully!")
    print("\nImportant: Make sure to secure your config.yaml file and backup your encryption keys!")
    print("\nNext steps:")
    print("1. Review and customize config/config.yaml")
    print("2. Set up your SSL certificate if deploying to production")
    print("3. Configure your backup strategy")
    print("4. Place your trained models in the 'models' directory")

if __name__ == "__main__":
    main()