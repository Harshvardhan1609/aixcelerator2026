import sqlite3
import hashlib
import os
import json
import time
import uuid
from typing import Optional, Dict, List, Any, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_portal.db")

def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with dict-like row factory."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hashes a password with a unique salt using SHA-256."""
    if salt is None:
        salt = uuid.uuid4().hex[:16]
    combined = (password + salt).encode("utf-8")
    pwd_hash = hashlib.sha256(combined).hexdigest()
    return pwd_hash, salt

def init_database():
    """Initializes SQLite database schema for users, threads, messages, and settings."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            full_name TEXT NOT NULL,
            user_bio TEXT DEFAULT '',
            response_pref TEXT DEFAULT '',
            created_at REAL NOT NULL
        )
    """)
    
    # Chat Threads table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        )
    """)
    
    # Messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            msg_id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources_json TEXT DEFAULT '[]',
            token_info_json TEXT DEFAULT '{}',
            timestamp REAL NOT NULL,
            FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()

def purge_demo_accounts():
    """Removes demo accounts for production deployment."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username IN ('student', 'test_student_2026')")
    conn.commit()
    conn.close()

# ----------------------------------------------------------------------
# USER AUTHENTICATION & PROFILE FUNCTIONS
# ----------------------------------------------------------------------

def register_user(username: str, password: str, full_name: str) -> Tuple[bool, str]:
    """Registers a new user ID with credentials."""
    username = username.strip().lower()
    full_name = full_name.strip()
    
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not password or len(password) < 4:
        return False, "Password must be at least 4 characters."
    if not full_name:
        full_name = username.capitalize()
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return False, f"Username '{username}' already exists. Please choose another ID."
        
    pwd_hash, salt = hash_password(password)
    try:
        cursor.execute("""
            INSERT INTO users (username, password_hash, salt, full_name, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (username, pwd_hash, salt, full_name, time.time()))
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except Exception as e:
        conn.close()
        return False, f"Registration failed: {str(e)}"

def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Authenticates a user against credentials."""
    username = username.strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return False, None, "Invalid username or user ID does not exist."
        
    pwd_hash, _ = hash_password(password, salt=row["salt"])
    if pwd_hash != row["password_hash"]:
        return False, None, "Incorrect password. Please try again."
        
    user_dict = {
        "username": row["username"],
        "full_name": row["full_name"],
        "user_bio": row["user_bio"] or "",
        "response_pref": row["response_pref"] or ""
    }
    return True, user_dict, "Login successful!"

def save_custom_instructions(username: str, user_bio: str, response_pref: str) -> bool:
    """Saves ChatGPT-style custom instructions for the user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE users
        SET user_bio = ?, response_pref = ?
        WHERE username = ?
    """, (user_bio.strip(), response_pref.strip(), username))
    conn.commit()
    conn.close()
    return True

def get_custom_instructions(username: str) -> Dict[str, str]:
    """Retrieves custom instructions for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_bio, response_pref FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "user_bio": row["user_bio"] or "",
            "response_pref": row["response_pref"] or ""
        }
    return {"user_bio": "", "response_pref": ""}

# ----------------------------------------------------------------------
# CHAT THREAD MANAGEMENT (Like ChatGPT)
# ----------------------------------------------------------------------

def get_user_threads(username: str) -> List[Dict[str, Any]]:
    """Fetches all chat threads belonging to a user, sorted by recency."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM threads
        WHERE username = ?
        ORDER BY updated_at DESC
    """, (username,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_new_thread(username: str, title: str = "New Chat") -> str:
    """Creates a new chat thread for a user."""
    thread_id = str(uuid.uuid4())
    now = time.time()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO threads (thread_id, username, title, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """, (thread_id, username, title, now, now))
    conn.commit()
    conn.close()
    return thread_id

def delete_thread(thread_id: str) -> bool:
    """Deletes a thread and its messages."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
    cursor.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
    conn.commit()
    conn.close()
    return True

def rename_thread(thread_id: str, new_title: str) -> bool:
    """Renames an existing chat thread."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE threads
        SET title = ?, updated_at = ?
        WHERE thread_id = ?
    """, (new_title.strip()[:60], time.time(), thread_id))
    conn.commit()
    conn.close()
    return True

def save_message(
    thread_id: str,
    role: str,
    content: str,
    sources: Optional[List[Any]] = None,
    token_info: Optional[Dict[str, Any]] = None
) -> str:
    """Appends a message to a thread and updates the thread's updated_at timestamp."""
    msg_id = str(uuid.uuid4())
    now = time.time()
    sources_json = json.dumps(sources or [])
    token_info_json = json.dumps(token_info or {})
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (msg_id, thread_id, role, content, sources_json, token_info_json, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (msg_id, thread_id, role, content, sources_json, token_info_json, now))
    
    # Auto-update thread title if this is the first user query and title is default
    if role == "user":
        cursor.execute("SELECT title FROM threads WHERE thread_id = ?", (thread_id,))
        row = cursor.fetchone()
        if row and row["title"] == "New Chat":
            auto_title = content[:38] + ("..." if len(content) > 38 else "")
            cursor.execute("UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?", (auto_title, now, thread_id))
        else:
            cursor.execute("UPDATE threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id))
    else:
        cursor.execute("UPDATE threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id))
        
    conn.commit()
    conn.close()
    return msg_id

def get_thread_messages(thread_id: str) -> List[Dict[str, Any]]:
    """Loads all messages for a specific chat thread in chronological order."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM messages
        WHERE thread_id = ?
        ORDER BY timestamp ASC
    """, (thread_id,))
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for r in rows:
        try:
            sources = json.loads(r["sources_json"]) if r["sources_json"] else []
        except Exception:
            sources = []
        try:
            token_info = json.loads(r["token_info_json"]) if r["token_info_json"] else None
        except Exception:
            token_info = None
            
        messages.append({
            "msg_id": r["msg_id"],
            "role": r["role"],
            "content": r["content"],
            "sources": sources,
            "token_info": token_info,
            "timestamp": r["timestamp"]
        })
    return messages

# Initialize tables and ensure production purity on module load
init_database()
purge_demo_accounts()
