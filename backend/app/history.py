import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS request_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            task TEXT,
            status TEXT,
            conflict BOOLEAN,
            model TEXT,
            confidence REAL,
            response_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_request(query: str, response: Dict[str, Any]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    task = response.get("task", "unknown")
    status = response.get("status", "ERROR")
    conflict = response.get("conflict", False)
    confidence = response.get("confidence")
    prov = response.get("provenance") or {}
    model = prov.get("model", "unknown")
    
    c.execute('''
        INSERT INTO request_history 
        (timestamp, query, task, status, conflict, model, confidence, response_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.utcnow().isoformat() + "Z",
        query,
        task,
        status,
        conflict,
        model,
        confidence,
        json.dumps(response)
    ))
    conn.commit()
    conn.close()

def get_history() -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM request_history ORDER BY id DESC LIMIT 100')
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_stats() -> Dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM request_history')
    total_requests = c.fetchone()[0]
    
    c.execute('SELECT task, COUNT(*) FROM request_history GROUP BY task')
    task_dist = {row[0]: row[1] for row in c.fetchall()}
    
    c.execute('SELECT status, COUNT(*) FROM request_history GROUP BY status')
    status_dist = {row[0]: row[1] for row in c.fetchall()}
    
    c.execute('SELECT COUNT(*) FROM request_history WHERE conflict = 1')
    conflicts = c.fetchone()[0]
    
    c.execute('SELECT model, COUNT(*) FROM request_history GROUP BY model')
    model_usage = {row[0]: row[1] for row in c.fetchall()}
    
    conn.close()
    
    return {
        "total_requests": total_requests,
        "task_distribution": task_dist,
        "status_distribution": status_dist,
        "conflicts": conflicts,
        "model_usage": model_usage
    }

init_db()
