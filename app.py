# app.py
import os
import time
import json
import hashlib
import logging
import random
import sqlite3
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
import streamlit as st

# ------------------ CONFIG ------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("Please set the OPENROUTER_API_KEY environment variable.")

AVAILABLE_MODELS = [
    "mistralai/mixtral-8x7b:free",
    "google/gemma-2-9b-it:free",
    "deepseek/deepseek-chat-v3.1:free"
]
MODEL_NAME = os.getenv("MODEL_NAME", AVAILABLE_MODELS[0])

MAX_API_ATTEMPTS = 6
BASE_BACKOFF = 1.0
MAX_BACKOFF = 30.0
CB_FAILURE_THRESHOLD = 5
CB_COOLDOWN_SECONDS = 60
DEDUP_TTL_SECONDS = 20

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

# ------------------ STRUCTURED LOGGING ------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "chemeng_buddy.log")

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)
        return json.dumps(log_record)

logger = logging.getLogger("chemeng_buddy")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(JSONFormatter())
logger.addHandler(file_handler)

# ------------------ DATABASE ------------------
DB_PATH = "chemeng_buddy.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS dedup_cache (
            key TEXT PRIMARY KEY,
            timestamp REAL,
            response TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS circuit_breaker (
            id INTEGER PRIMARY KEY,
            failure_count INTEGER,
            opened_at REAL
        )
    """)
    c.execute("INSERT OR IGNORE INTO circuit_breaker (id, failure_count, opened_at) VALUES (1, 0, NULL)")
    conn.commit()
    conn.close()

init_db()

def db_execute(query, params=(), fetch=False):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(query, params)
    result = c.fetchall() if fetch else None
    conn.commit()
    conn.close()
    return result

# ------------------ CIRCUIT BREAKER ------------------
@dataclass
class CircuitBreaker:
    cooldown_seconds: int = CB_COOLDOWN_SECONDS
    threshold: int = CB_FAILURE_THRESHOLD

    def record_failure(self):
        data = db_execute("SELECT failure_count, opened_at FROM circuit_breaker WHERE id=1", fetch=True)[0]
        failure_count, opened_at = data
        failure_count += 1
        if failure_count >= self.threshold and opened_at is None:
            opened_at = time.time()
            logger.warning("Circuit opened", extra={"extra_data": {"failure_count": failure_count}})
        db_execute("UPDATE circuit_breaker SET failure_count=?, opened_at=? WHERE id=1",
                   (failure_count, opened_at))

    def record_success(self):
        db_execute("UPDATE circuit_breaker SET failure_count=0, opened_at=NULL WHERE id=1")

    def is_open(self) -> bool:
        data = db_execute("SELECT failure_count, opened_at FROM circuit_breaker WHERE id=1", fetch=True)[0]
        failure_count, opened_at = data
        if opened_at is None:
            return False
        if time.time() - opened_at > self.cooldown_seconds:
            self.record_success()
            logger.info("Circuit closed", extra={"extra_data": {"event": "circuit_closed"}})
            return False
        return True

# ------------------ DEDUP CACHE ------------------
@dataclass
class DedupCache:
    ttl_seconds: int = DEDUP_TTL_SECONDS

    def get(self, key: str) -> str:
        result = db_execute("SELECT timestamp, response FROM dedup_cache WHERE key=?", (key,), fetch=True)
        if not result:
            return None
        ts, resp = result[0]
        if time.time() - ts > self.ttl_seconds:
            db_execute("DELETE FROM dedup_cache WHERE key=?", (key,))
            return None
        return resp

    def set(self, key: str, response: str):
        db_execute("INSERT OR REPLACE INTO dedup_cache (key, timestamp, response) VALUES (?, ?, ?)",
                   (key, time.time(), response))

circuit = CircuitBreaker()
dedup = DedupCache()

# ------------------ OPENROUTER ------------------
def exponential_backoff_with_jitter(attempt: int) -> float:
    cap = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt))
    return random.uniform(0, cap)

SYSTEM_PROMPT = (
    "You are ChemEng Buddy, an expert tutor in Chemical Engineering. "
    "Provide clear, step-by-step reasoning, equations, and practical insights for students. "
    "Focus only on Chemical Engineering concepts."
)

def build_prompt(question: str, last_q: str = None) -> List[Dict[str, str]]:
    """
    Detects follow-up prompts and reuses only the last question (not the model's answer).
    """
    follow_up_keywords = ["elaborate", "explain", "why", "how", "example", "more", "simpler", "detail"]
    lower_q = question.lower()

    if any(k in lower_q for k in follow_up_keywords) and last_q:
        context = (
            f"The user previously asked: '{last_q}'. "
            f"Now they ask a follow-up: '{question}'. "
            f"Please elaborate based only on that last question."
        )
        logger.info("Follow-up detected", extra={"extra_data": {"last_question": last_q, "user_query": question}})
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]
    else:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def call_openrouter_with_retries(model: str, messages: List[Dict[str, str]]) -> Tuple[bool, str]:
    if circuit.is_open():
        return False, "OpenRouter temporarily unavailable due to repeated failures."

    payload = {"model": model, "messages": messages}
    prompt_hash = hash_prompt(json.dumps(payload, sort_keys=True))
    cached = dedup.get(prompt_hash)
    if cached:
        logger.info("Cache hit", extra={"extra_data": {"model": model}})
        return True, cached

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            start = time.time()
            r = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
            latency = round(time.time() - start, 2)

            if r.status_code == 200:
                data = r.json()
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    dedup.set(prompt_hash, content)
                    circuit.record_success()
                    logger.info("API success", extra={"extra_data": {"model": model, "latency": latency}})
                    return True, content
            elif r.status_code in (429, 500, 503):
                logger.warning("Retrying", extra={"extra_data": {"status": r.status_code, "attempt": attempt}})
                time.sleep(exponential_backoff_with_jitter(attempt))
        except Exception as e:
            circuit.record_failure()
            logger.error("API exception", extra={"extra_data": {"error": str(e)}})
            time.sleep(exponential_backoff_with_jitter(attempt))

    circuit.record_failure()
    logger.error("Max retries reached", extra={"extra_data": {"model": model}})
    return False, "⚠️ OpenRouter unreachable after retries."

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="ChemEng Buddy", layout="wide")

st.markdown("""
<style>
body {background-color: #f8fafc; color: #1a1a1a; font-family: 'Inter', sans-serif;}
.main-title {font-size: 42px; color: #004aad; font-weight: 700; text-align: center; margin-bottom: 0px;}
.subtitle {text-align: center; font-size: 18px; color: #555; margin-bottom: 40px;}
.footer {text-align: center; color: #888; font-size: 14px; margin-top: 50px;}
hr.solid {border-top: 1px solid #ccc;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>⚗️ ChemEng Buddy</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your intelligent Chemical Engineering assistant.</p>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask me a question about Chemical Engineering 🌡️")

if user_query:
    last_q = None
    if len(st.session_state.chat_history) >= 2:
        last_q = st.session_state.chat_history[-2]["content"]

    messages = build_prompt(user_query, last_q)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("Thinking like a ChemEng expert..."):
        success, response = call_openrouter_with_retries(MODEL_NAME, messages)
        if not success:
            response = "⚠️ Unable to get response from OpenRouter."

    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display chat
for chat in st.session_state.chat_history:
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        st.markdown(chat["content"])

# ------------------ FOOTER ------------------
st.markdown("""
<hr class="solid">
<div class='footer'>
    Built with ❤️ by <b>Team EKC</b> · BITS Pilani<br>
    📬 <a href="mailto:prakhar.mathur2020@gmail.com">Contact Us</a>
</div>
""", unsafe_allow_html=True)
