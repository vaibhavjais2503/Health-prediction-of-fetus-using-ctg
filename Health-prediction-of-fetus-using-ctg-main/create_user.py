from security_utils import SecurityManager

def create_initial_user():
    security_manager = SecurityManager()
    success = security_manager.create_user(
        username="admin",
        password="admin123",  # Change this to a secure password
        role="admin"
    )
    if success:
        print("Initial user created successfully")
    else:
        print("Failed to create initial user")

if __name__ == "__main__":
    create_initial_user()