"""
Configuration module for loading environment variables.

This module loads environment variables from TWO locations for security:
1. Root .env - Shared safe variables (SUPABASE_URL, SUPABASE_ANON_KEY)
2. backend_ml/.env.local - Backend-only secrets (SUPABASE_SERVICE_KEY, API keys)

All other modules should import config values from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory (parent of backend_ml)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

# Load environment variables from TWO files:
# 1. Root .env (shared safe variables)
root_env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=root_env_path)

# 2. Backend .env.local (backend-only secrets) - overrides root if same key exists
backend_env_path = BACKEND_DIR / '.env.local'
load_dotenv(dotenv_path=backend_env_path, override=True)

# Supabase Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')  # From root .env (safe, public)
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')  # From backend_ml/.env.local (secret!)

# NBA API Configuration
NBA_API_KEY = os.getenv('NBA_API_KEY')

# Model Configuration
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v1.0.0')
MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.6'))

# Validate required environment variables
def validate_config():
    """Validate that required environment variables are set."""
    required_vars = {
        'SUPABASE_URL': SUPABASE_URL,
        'SUPABASE_SERVICE_KEY': SUPABASE_SERVICE_KEY,
    }

    missing_vars = [var for var, value in required_vars.items() if not value]

    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please create a .env file in the project root and set these variables.\n"
            f"See .env.example for reference."
        )

# Example usage in other modules:
# from config import SUPABASE_URL, SUPABASE_SERVICE_KEY
