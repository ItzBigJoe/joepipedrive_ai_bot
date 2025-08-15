# db.py

import sqlite3
from contextlib import closing

DB_PATH = "lead_responses.db"

def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS replies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                body TEXT,
                your_reply TEXT
            )
        """)

def save_reply(subject: str, body: str, your_reply: str):
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.execute(
            "INSERT INTO replies (subject, body, your_reply) VALUES (?, ?, ?)",
            (subject or "", body or "", your_reply or "")
        )

def get_recent_replies(limit: int = 5):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute(
            "SELECT subject, body, your_reply FROM replies ORDER BY id DESC LIMIT ?",
            (int(limit),)
        )
        return cur.fetchall()
