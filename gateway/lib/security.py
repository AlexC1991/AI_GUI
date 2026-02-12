"""
Security & Session Management
"""
import time
import secrets
import random
from colorama import Fore
from .config import MAX_LOGIN_ATTEMPTS, BAN_DURATION, SESSION_TIMEOUT, RATE_LIMIT_Window, RATE_LIMIT_Max, log_event
from .database import client_db, save_database

# Decoy URLs for unauthorized access
DECOY_URLS = [
    "https://www.google.com",
    "https://www.bing.com",
    "https://www.yahoo.com",
    "https://www.wikipedia.org",
    "https://weather.com",
    "https://www.cnn.com",
    "https://www.bbc.com",
    "https://www.amazon.com"
]

def get_decoy_url():
    """Return a random decoy URL"""
    return random.choice(DECOY_URLS)

# In-memory session tracking
active_sessions = {}
failed_attempts = {}

def is_banned(ip):
    """Check if IP is banned"""
    # [RELAXED] Localhost bypass
    if ip in ["127.0.0.1", "::1"]:
        return False
        
    ban_data = client_db.get("banned_ips", {}).get(ip)
    if ban_data:
        if time.time() < ban_data["lift_time"]:
            return True
        else:
            del client_db["banned_ips"][ip]
            save_database()
    return False

def record_failed_attempt(ip):
    """Log failed attempt and ban if necessary"""
    if ip in ["127.0.0.1", "::1"]: return
    
    attempts = failed_attempts.get(ip, 0) + 1
    failed_attempts[ip] = attempts
    
    if attempts >= MAX_LOGIN_ATTEMPTS:
        ban_time = time.time() + BAN_DURATION
        client_db.setdefault("banned_ips", {})[ip] = {
            "lift_time": ban_time,
            "reason": "Too many failed login attempts"
        }
        save_database()
        log_event("IP_BANNED", ip, {"duration": BAN_DURATION})
        print(f"{Fore.RED}[SECURITY] Banned {ip} for {BAN_DURATION}s")

def get_session(token):
    session = active_sessions.get(token)
    if session:
        if time.time() > session["expiry"]:
            del active_sessions[token]
            return None
        # Refresh session
        session["expiry"] = time.time() + SESSION_TIMEOUT
    return session

def create_session(client_id, ip_address):
    token = secrets.token_hex(32)
    active_sessions[token] = {
        "client_id": client_id,
        "ip": ip_address,
        "expiry": time.time() + SESSION_TIMEOUT,
        "created_at": time.time()
    }
    return token

def check_rate_limit(ip):
    if ip in ["127.0.0.1", "::1"]: return True
    
    # Simple token bucket or window
    return True # Placeholder for now to avoid blocking legitimate users during debug

def reset_ip(ip):
    """Clear bans and failed attempts for an IP"""
    if ip in client_db.get("banned_ips", {}):
        del client_db["banned_ips"][ip]
        save_database()
        return True
    
    if ip in failed_attempts:
        del failed_attempts[ip]
        return True
        
    return False

def ban_ip(ip, duration=3600):
    """Manually ban an IP"""
    client_db.setdefault("banned_ips", {})[ip] = time.time() + duration
    save_database()
    # Also clear auth
    if ip in authenticated_ips:
        del authenticated_ips[ip]
    return True

def kick_client(client_id):
    """Kick a client by ID (remove sessions)"""
    # 1. Remove from authenticated IPs
    ips_to_remove = []
    for ip, data in authenticated_ips.items():
        if data["client_id"] == client_id:
            ips_to_remove.append(ip)
    
    for ip in ips_to_remove:
        del authenticated_ips[ip]
        
    # 2. Remove active web sessions
    sessions_to_remove = []
    for token, session in active_sessions.items():
        if session.get("client_id") == client_id:
            sessions_to_remove.append(token)
            
    for token in sessions_to_remove:
        del active_sessions[token]
        
    return len(ips_to_remove) + len(sessions_to_remove) > 0

def trust_ip(ip):
    """Manually trust an IP as HOST"""
    register_ip(ip, "HOST")
    return True

# IP Authentication (Beacon / Activation)
authenticated_ips = {}

def register_ip(ip, client_id):
    """Register an IP as authenticated via Client Exe"""
    authenticated_ips[ip] = {
        "client_id": client_id,
        "expiry": time.time() + SESSION_TIMEOUT
    }
    # Clear any previous bans
    reset_ip(ip)

def is_ip_authenticated(ip):
    """Check if IP is authenticated via Client Exe"""
    # Debug/Localhost
    if ip in ["127.0.0.1", "::1"]:
        return {"client_id": "HOST_DEBUG"}
        
    data = authenticated_ips.get(ip)
    if data:
        if time.time() > data["expiry"]:
            del authenticated_ips[ip]
            return None
        # Refresh expiry on active use?
        data["expiry"] = time.time() + SESSION_TIMEOUT
        return data
    return None

