import sqlite3
DB_PATH = "lead_responses.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS replies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT,
        body TEXT,
        your_reply TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_reply(subject, body, your_reply):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO replies (subject, body, your_reply) VALUES (?, ?, ?)",
              (subject, body, your_reply))
    conn.commit()
    conn.close()

def get_recent_replies(limit=5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT subject, body, your_reply FROM replies ORDER BY id DESC LIMIT ?", (limit,))
    data = c.fetchall()
    conn.close()
    return data