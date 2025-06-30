# utils/db.py
import sqlite3

def init_db():
    conn = sqlite3.connect("feature_planning.db")
    c = conn.cursor()

    # Project metadata table
    c.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            metadata TEXT,
            created_at TIMESTAMP
        )
    ''')

    # User journey maps
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_journeys (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            journey_map TEXT,
            created_at TIMESTAMP
        )
    ''')

    # Feature plans
    c.execute('''
        CREATE TABLE IF NOT EXISTS feature_plans (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            feature_plan TEXT,
            created_at TIMESTAMP
        )
    ''')

    # User actions
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_actions (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            action_list TEXT,
            created_at TIMESTAMP
        )
    ''')

    # Refined plans
    c.execute('''
        CREATE TABLE IF NOT EXISTS refined_plans (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            feedback TEXT,
            refined_plan TEXT,
            created_at TIMESTAMP
        )
    ''')

    # VectorDB entries
    c.execute('''
        CREATE TABLE IF NOT EXISTS vectordb_entries (
            id TEXT PRIMARY KEY,
            project_id TEXT,
            text TEXT,
            created_at TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
