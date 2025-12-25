import json
import os
import sqlite3
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    load_dotenv("../.env")

# --- Configuration ---
DATA_FILE = 'spider2-lite.jsonl'
DB_DIR = 'resource/databases/spider2-localdb'
OUTPUT_DIR = 'evaluation_suite/tmp/my_agent_results_latest'
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
MAX_WORKERS = 10 # Adjust based on your API Rate Limit

class SqlResponse(BaseModel):
    sql_query: str
    explanation: Optional[str]

# Global Cache and Locks
SCHEMA_CACHE: Dict[str, str] = {}
CACHE_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()

def safe_print(msg):
    with PRINT_LOCK:
        print(msg)

def get_detailed_schema(db_path):
    """Thread-safe schema extraction with caching."""
    with CACHE_LOCK:
        if db_path in SCHEMA_CACHE:
            return SCHEMA_CACHE[db_path]
        
    if not os.path.exists(db_path): return "Error: DB not found."
    
    schema = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        
        for table in tables:
            schema.append(f"Table: \"{table}\"")
            cursor.execute(f"PRAGMA table_info(\"{table}\")")
            cols = cursor.fetchall()
            schema.append(f"  Columns: " + ", ".join([f"{c[1]} ({c[2]})" for c in cols]))
            
            cursor.execute(f"PRAGMA foreign_key_list(\"{table}\")")
            fks = cursor.fetchall()
            for fk in fks:
                schema.append(f"  FK: {fk[3]} -> {fk[2]}({fk[4]})")
            
            try:
                cursor.execute(f"SELECT * FROM \"{table}\" LIMIT 2")
                schema.append(f"  Sample: {cursor.fetchall()}")
            except: pass
            schema.append("-" * 10)
        conn.close()
    except Exception as e: return f"Error reading schema: {e}"
    
    result = "\n".join(schema)
    with CACHE_LOCK:
        SCHEMA_CACHE[db_path] = result
    return result

# Initialize Client
client = genai.Client(api_key=API_KEY)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.INFO)
)
def call_gemini(prompt: str):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': SqlResponse,
        },
    )
    return response.parsed

def process_task(task):
    """Single task worker function."""
    instance_id = task['instance_id']
    db_id = task['db']
    question = task['question']
    db_path = os.path.join(DB_DIR, f"{db_id}.sqlite")
    out_path = os.path.join(OUTPUT_DIR, f"{instance_id}.sql")
    
    if os.path.exists(out_path):
        return instance_id, "Skipped"

    schema = get_detailed_schema(db_path)
    
    current_prompt = f"""You are a SQLite expert. 
Task: Generate a query for this question: {question}

Database Schema:
{schema}

DIALECT RULES:
1. Wrap all identifiers in double quotes: \"table_name\".\"column_name\".
2. UNION ALL: No ORDER BY/LIMIT inside branches. 
3. Dates: Use 'strftime('%Y-%m-%d', col)' and 'julianday()'.
"""

    error_msg = ""
    last_sql = ""
    for attempt in range(2):
        try:
            prompt = current_prompt if attempt == 0 else f"{current_prompt}\n\nERROR: {error_msg}\nSQL was: {last_sql}\nFix it."
            parsed_response = call_gemini(prompt)
            last_sql = parsed_response.sql_query
            
            # Validation
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(f"EXPLAIN {last_sql}")
                with open(out_path, 'w') as f: f.write(last_sql)
                return instance_id, "Success"
            except sqlite3.Error as e:
                error_msg = str(e)
            finally:
                conn.close()
        except Exception as e:
            error_msg = str(e)
            
    # Final fallback write
    with open(out_path, 'w') as f: f.write(last_sql)
    return instance_id, f"Final fallback (Error: {error_msg})"

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    with open(DATA_FILE, 'r') as f:
        tasks = [json.loads(line) for line in f if "instance_id" in line]
    tasks = [t for t in tasks if t['instance_id'].startswith('local')]
    
    print(f"Starting Parallel optimized run for {len(tasks)} tasks with {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_task, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            instance_id, status = future.result()
            safe_print(f"  [{instance_id}] {status}")

if __name__ == "__main__":
    main()