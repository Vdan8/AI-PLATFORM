# test_import.py
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(project_root, "backend")

# Ensure the 'backend' directory is in sys.path
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

print("sys.path:")
print(sys.path)
print("\nEnvironment Variables (os.environ):")
for key, value in os.environ.items():
    if "OPENAI" in key.upper():
        print(f"{key}={value}")
    elif "DATABASE" in key.upper():
        print(f"{key}={value}")
    elif "PYTHON" in key.upper():
        print(f"{key}={value}")

try:
    from backend.app.api import auth
    print("Import of backend.app.api.auth successful")
except ImportError as e:
    print(f"Import error (auth): {e}")

try:
    from backend.config.config import Settings
    settings = Settings()
    print("Settings initialized successfully")
    print(f"OPENAI_API_KEY from Settings: {settings.OPENAI_API_KEY}")
except ImportError as e:
    print(f"Import error (config): {e}")
except Exception as e:
    print(f"Error initializing Settings: {e}")

try:
    from backend.app.main import app
    print("Import of backend.app.main successful")
except ImportError as e:
    print(f"Import error (main): {e}")