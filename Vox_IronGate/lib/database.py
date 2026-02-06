"""
Database Management
"""
import os
import json
from .config import DATABASE_FILE, log_event

client_db = {
    "clients": {},
    "banned_ips": {},
    "sessions": {}
}

def load_database():
    global client_db
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, "r") as f:
                data = json.load(f)
                client_db.update(data)
        except Exception as e:
            log_event("DB_ERROR", details={"error": str(e)})

def save_database():
    try:
        with open(DATABASE_FILE, "w") as f:
            json.dump(client_db, f, indent=4)
    except Exception as e:
        log_event("DB_SAVE_ERROR", details={"error": str(e)})

def get_client(client_id):
    return client_db["clients"].get(client_id)

def add_client(client_id, data):
    client_db["clients"][client_id] = data
    save_database()
