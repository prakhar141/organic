# app.py
import os
import time
import json
import hashlib
import logging
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
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

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chemeng_buddy")

# ------------------ STREAMLIT SETUP ------------------
st.set_page_config(page_title="ChemEng Buddy", layout="wide")
st.markdown("""
<style>
body {background-color: #f8fafc; color: #1a1a1a; font-family: 'Inter', sans-serif;}
.main-title {font-size: 42px; color: #004aad; font-weight: 700; text-align: center; margin-bottom: 0px;}
.subtitle {text-align: center; font-size: 18px; color: #555; margin-bottom: 40px;}
.footer {text-align: center; color: #888; font-size: 14px; margin-top: 50px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>⚗️ ChemEng Buddy</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your intelligent Chemical Engineering assistant.</p>", unsafe_allow_html=True)

# ------------------ UTILITIES ------------------
def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

# ------------------ CIRCUIT BREAKER + DEDUP ------------------
@dataclass
class CircuitBreaker:
    failure_count: int = 0
    opened_at: datetime = None
    cooldown_seconds: int = CB_COOLDOWN_SECONDS
    threshold: int = CB_FAILURE_THRESHOLD

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.threshold and not self.is_open():
            self.opened_at = datetime.utcnow()
            logger.warning(f"Circuit opened due to failures at {self.opened_at}")

    def record_success(self):
        self.failure_count = 0
        self.opened_at = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if datetime.utcnow() - self.opened_at > timedelta(seconds=self.cooldown_seconds):
            self.failure_count = 0
            self.opened_at = None
            return False
        return True

@dataclass
class DedupCache:
    store: Dict[str, Tuple[float, str]] = None
    ttl_seconds: int = DEDUP_TTL_SECONDS

    def __post_init__(self):
        self.store = {}

    def get(self, key: str) -> str:
        entry = self.store.get(key)
        if not entry:
            return None
        ts, resp = entry
        if time.time() - ts > self.ttl_seconds:
            self.store.pop(key, None)
            return None
        return resp

    def set(self, key: str, response: str):
        self.store[key] = (time.time(), response)

if "circuit" not in st.session_state:
    st.session_state.circuit = CircuitBreaker()
if "dedup" not in st.session_state:
    st.session_state.dedup = DedupCache()

circuit: CircuitBreaker = st.session_state.circuit
dedup: DedupCache = st.session_state.dedup

# ------------------ OPENROUTER CLIENT ------------------
def exponential_backoff_with_jitter(attempt: int) -> float:
    cap = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt))
    return random.uniform(0, cap)

SYSTEM_PROMPT = (
    "You are ChemEng Buddy, an expert tutor in Chemical Engineering. "
    "Provide clear, step-by-step reasoning, equations, and practical insights for students. "
    "Focus only on Chemical Engineering concepts."
)

def build_prompt(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

def call_openrouter_with_retries(model: str, messages: List[Dict[str, str]]) -> Tuple[bool, str]:
    if circuit.is_open():
        return False, "OpenRouter temporarily unavailable due to repeated failures."

    payload = {"model": model, "messages": messages}
    prompt_hash = hash_prompt(json.dumps(payload, sort_keys=True))
    cached = dedup.get(prompt_hash)
    if cached:
        return True, cached

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and data["choices"]:
                    content = data["choices"][0]["message"]["content"]
                    dedup.set(prompt_hash, content)
                    circuit.record_success()
                    return True, content
            elif resp.status_code in (429, 500, 503):
                time.sleep(exponential_backoff_with_jitter(attempt))
        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            circuit.record_failure()
            time.sleep(exponential_backoff_with_jitter(attempt))

    circuit.record_failure()
    return False, "⚠️ OpenRouter unreachable after retries."

# ------------------ CHAT INTERFACE ------------------
def type_like_chatgpt(text, speed=0.004):
    placeholder = st.empty()
    out = ""
    for c in text:
        out += c
        placeholder.markdown(out + " |")
        time.sleep(speed)
    placeholder.markdown(out)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

if circuit.is_open():
    st.warning("⚠️ External model temporarily paused due to repeated failures.")

user_query = st.chat_input("Ask me about Chemical Engineering 🌡️")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    messages = build_prompt(user_query)

    with st.spinner("Thinking like a ChemEng expert..."):
        success, response = call_openrouter_with_retries(MODEL_NAME, messages)
        if not success:
            response = "⚠️ Unable to get response from OpenRouter."

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.last_answer_animated = True

# Display chat history
for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        if i == len(st.session_state.chat_history) - 1 and chat["role"] == "assistant" and st.session_state.last_answer_animated:
            type_like_chatgpt(chat["content"])
            st.session_state.last_answer_animated = False
        else:
            st.markdown(chat["content"])

# ------------------ FOOTER ------------------
st.markdown("""
<hr class="solid">
<div class='footer'>
    Built with ❤️ by <b>Team EKC</b> · BITS Pilani<br>
    📬 <a href="mailto:prakhar.mathur2020@gmail.com">Contact Us</a>
</div>
""", unsafe_allow_html=True)
