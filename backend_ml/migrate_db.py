#!/usr/bin/env python3
"""
Database Migration Script - Add Ensemble Model Columns

This script adds the necessary columns for the ensemble model to the game_predictions table.
Run this once after upgrading to the ensemble model.
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from supabase import create_client

# Load environment variables
load_dotenv()
load_dotenv(Path(__file__).parent / '.env.local', override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

def run_migration():
    """Add ensemble model columns to game_predictions table"""

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env.local")
        return False

    print("🔄 Connecting to Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    # Migration SQL
    migration_sql = """
    -- Add ensemble model columns
    ALTER TABLE game_predictions
    ADD COLUMN IF NOT EXISTS xgb_home_prob DECIMAL(5,4),
    ADD COLUMN IF NOT EXISTS ridge_home_prob DECIMAL(5,4),
    ADD COLUMN IF NOT EXISTS models_agree BOOLEAN;

    -- Add index for model disagreements
    CREATE INDEX IF NOT EXISTS idx_models_disagree
    ON game_predictions(models_agree)
    WHERE models_agree = false;
    """

    try:
        print("🔄 Running migration SQL...")

        # Supabase Python client doesn't support raw SQL execution
        # We need to use the REST API or instruct user to run in SQL editor
        print("\n" + "="*80)
        print("⚠️  MANUAL MIGRATION REQUIRED")
        print("="*80)
        print("\nThe Supabase Python client doesn't support ALTER TABLE commands.")
        print("Please run the following SQL in your Supabase SQL Editor:\n")
        print("1. Go to: https://supabase.com/dashboard/project/YOUR_PROJECT/sql/new")
        print("2. Copy and paste this SQL:\n")
        print("-" * 80)
        print(migration_sql)
        print("-" * 80)
        print("\n3. Click 'RUN' to execute the migration")
        print("\nAfter running the migration, predictions will upload successfully!")
        print("="*80)

        return False

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    run_migration()
