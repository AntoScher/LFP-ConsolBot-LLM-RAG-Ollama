import sqlite3
from datetime import datetime

def init_db():
    """Инициализирует базу данных"""
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS session_logs (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                query TEXT,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()

def log_query(user_id, query, response):
    """Логирует запрос и ответ в базу данных"""
    conn = sqlite3.connect("app.db")
    c = conn.cursor()
    c.execute("INSERT INTO session_logs (user_id, query, response) VALUES (?, ?, ?)",
              (user_id, query, response[:1000]))  # Ограничиваем длину ответа
    conn.commit()
    conn.close()