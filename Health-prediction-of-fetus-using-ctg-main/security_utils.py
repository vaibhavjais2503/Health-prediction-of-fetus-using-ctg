import streamlit as st
from cryptography.fernet import Fernet
import base64
import sqlite3
from datetime import datetime, timedelta
import secrets
import hashlib
import os
import json
import pandas as pd

class SecurityManager:
    def __init__(self, secret_key=None):
        """Initialize the security manager with a Fernet key"""
        try:
            if secret_key:
                if isinstance(secret_key, str):
                    key_bytes = base64.urlsafe_b64decode(secret_key.encode())
                    if len(key_bytes) != 32:
                        raise ValueError("Invalid key length")
                    self.key = secret_key.encode()
                else:
                    self.key = secret_key
            else:
                self.key = Fernet.generate_key()
            
            self.fernet = Fernet(self.key)
            
        except Exception as e:
            print(f"Error with provided key: {e}. Generating new key.")
            self.key = Fernet.generate_key()
            self.fernet = Fernet(self.key)

        # Initialize database
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect('data/fetal_health.db') as conn:
            cursor = conn.cursor()
            
            # Users table
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
                reset_token TEXT,
                reset_token_expiry TIMESTAMP
            )
            """)
            
            # Audit log table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                action TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """)
            
            # Predictions history table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                input_data TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """)
            
            # User settings table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                preferences TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """)
            
            conn.commit()
    
    def get_key(self):
        """Return the current key in string format"""
        return self.key.decode() if isinstance(self.key, bytes) else self.key

    def encrypt_data(self, data):
        """Encrypt data"""
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
        return self.fernet.encrypt(str(data).encode()).decode()

    def decrypt_data(self, encrypted_data):
        """Decrypt data"""
        try:
            decrypted = self.fernet.decrypt(encrypted_data.encode()).decode()
            try:
                return json.loads(decrypted)
            except json.JSONDecodeError:
                return decrypted
        except Exception as e:
            st.error(f"Decryption failed: {str(e)}")
            return None

    def hash_password(self, password):
        """Hash a password with a salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        ).hex()
        return password_hash, salt

    def verify_password(self, password, stored_hash, salt):
        """Verify a password against its hash"""
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000
        ).hex()
        return secrets.compare_digest(password_hash, stored_hash)

    def create_user(self, username, password, role='user'):
        """Create a new user"""
        try:
            password_hash, salt = self.hash_password(password)
            with sqlite3.connect('data/fetal_health.db') as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO users (id, username, password_hash, salt, role) 
                    VALUES (?, ?, ?, ?, ?)""",
                    (secrets.token_hex(16), username, password_hash, salt, role)
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def verify_user(self, username, password):
        """Verify user credentials"""
        with sqlite3.connect('data/fetal_health.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, password_hash, salt, role, failed_login_attempts 
                FROM users WHERE username = ?""",
                (username,)
            )
            result = cursor.fetchone()
            
            if not result:
                return False, None
            
            user_id, stored_hash, salt, role, failed_attempts = result
            
            if failed_attempts >= 3:
                cursor.execute(
                    "UPDATE users SET failed_login_attempts = failed_login_attempts + 1 WHERE id = ?",
                    (user_id,)
                )
                conn.commit()
                return False, "Account locked due to too many failed attempts"
            
            if self.verify_password(password, stored_hash, salt):
                # Reset failed attempts and update last login
                cursor.execute(
                    """UPDATE users SET 
                    failed_login_attempts = 0, 
                    last_login = CURRENT_TIMESTAMP 
                    WHERE id = ?""",
                    (user_id,)
                )
                conn.commit()
                return True, {'user_id': user_id, 'role': role}
            
            # Increment failed attempts
            cursor.execute(
                "UPDATE users SET failed_login_attempts = failed_login_attempts + 1 WHERE id = ?",
                (user_id,)
            )
            conn.commit()
            return False, None

    def update_password(self, user_id, old_password, new_password):
        """Update user password"""
        with sqlite3.connect('data/fetal_health.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT password_hash, salt FROM users WHERE id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return False
            
            stored_hash, salt = result
            
            if self.verify_password(old_password, stored_hash, salt):
                new_hash, new_salt = self.hash_password(new_password)
                cursor.execute(
                    """UPDATE users SET 
                    password_hash = ?, 
                    salt = ? 
                    WHERE id = ?""",
                    (new_hash, new_salt, user_id)
                )
                conn.commit()
                return True
            return False

    def initiate_password_reset(self, username):
        """Initiate password reset process"""
        with sqlite3.connect('data/fetal_health.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM users WHERE username = ?",
                (username,)
            )
            result = cursor.fetchone()
            
            if result:
                reset_token = secrets.token_urlsafe(32)
                expiry = datetime.now() + timedelta(hours=24)
                
                cursor.execute(
                    """UPDATE users SET 
                    reset_token = ?,
                    reset_token_expiry = ? 
                    WHERE username = ?""",
                    (reset_token, expiry, username)
                )
                conn.commit()
                return True
            return False

    def save_prediction(self, user_id, model_name, prediction, confidence, input_data):
        """Save prediction results"""
        with sqlite3.connect('data/fetal_health.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO predictions 
                (id, user_id, model_name, prediction, confidence, input_data) 
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    secrets.token_hex(16),
                    user_id,
                    model_name,
                    prediction,
                    confidence,
                    self.encrypt_data(input_data)
                )
            )
            conn.commit()

    def get_predictions_history(self, user_id=None, limit=100):
        """Get prediction history"""
        with sqlite3.connect('data/fetal_health.db') as conn:
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    """SELECT model_name, prediction, confidence, timestamp 
                    FROM predictions 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?""",
                    (user_id, limit)
                )
            else:
                cursor.execute(
                    """SELECT model_name, prediction, confidence, timestamp 
                    FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT ?""",
                    (limit,)
                )
            
            columns = ['model', 'prediction', 'confidence', 'timestamp']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_user_activity(self, user_id, limit=50):
        """Get user activity history"""
        with sqlite3.connect('data/fetal_health.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT action, details, timestamp 
                FROM audit_log 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?""",
                (user_id, limit)
            )
            
            columns = ['action', 'details', 'timestamp']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def log_event(self, user_id, action, details=None):
        """Log an event to the audit log"""
        with sqlite3.connect('data/fetal_health.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO audit_log (id, user_id, action, details) 
                VALUES (?, ?, ?, ?)""",
                (secrets.token_hex(16), user_id, action, details)
            )
            conn.commit()

    def get_user_settings(self, user_id):
        """Get user settings"""
        with sqlite3.connect('data/fetal_health.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT preferences FROM user_settings WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            
            if result:
                return json.loads(result[0])
            return {}

    def update_user_settings(self, user_id, settings):
        """Update user settings"""
        with sqlite3.connect('data/fetal_health.db') as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO user_settings (user_id, preferences) 
                VALUES (?, ?)""",
                (user_id, json.dumps(settings))
            )
            conn.commit()

def initialize_security():
    """Initialize security manager in session state"""
    if 'security_manager' not in st.session_state:
        st.session_state.security_manager = SecurityManager()