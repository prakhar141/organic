import os
import json
import requests
import streamlit as st
import hashlib
import time
import sqlite3
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ======================================
# CONFIG
# ======================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_MAIN = "deepseek/deepseek-chat-v3.1:free"
MODEL_FALLBACKS = [
    "google/gemma-2-9b-it:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mixtral-8x7b-instruct:free",
]
EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)

# Hugging Face URLs for prebuilt ChemEng dataset (medical dataset as placeholder)
FAISS_INDEX_URL = "https://huggingface.co/datasets/prakhar146/medical/resolve/main/index.faiss"
FAISS_PKL_URL = "https://huggingface.co/datasets/prakhar146/medical/resolve/main/index.pkl"
LOCAL_FAISS_DIR = "./faiss_store"
os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)

# ======================================
# STREAMLIT PAGE SETUP
# ======================================
st.set_page_config(page_title="⚗️ ChemEng Buddy", layout="wide")
st.title("⚗️ ChemEng Buddy")
st.markdown("Your friendly Chemical Engineering study partner")

# ======================================
# CHAT TYPING ANIMATION
# ======================================
def type_like_chatgpt(text, speed=0.004):
    placeholder = st.empty()
    animated = ""
    for c in text:
        animated += c
        placeholder.markdown(animated + "|")
        time.sleep(speed)
    placeholder.markdown(animated)

# ======================================
# FILE DOWNLOADER (Cached)
# ======================================
def download_file(url: str, local_path: str):
    if not os.path.exists(local_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

download_file(FAISS_INDEX_URL, os.path.join(LOCAL_FAISS_DIR, "index.faiss"))
download_file(FAISS_PKL_URL, os.path.join(LOCAL_FAISS_DIR, "index.pkl"))

# ======================================
# VECTORSTORE LOADER
# ======================================
@st.cache_resource
def load_vector_db():
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.load_local(
        LOCAL_FAISS_DIR,
        embedder,
        allow_dangerous_deserialization=True
    )
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

retriever = load_vector_db()

# ======================================
# OPENROUTER SETUP
# ======================================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

# ======================================
# IN-MEMORY + SQLITE CACHE
# ======================================
if "prompt_cache" not in st.session_state:
    st.session_state.prompt_cache = {}

ENABLE_SQL_CACHE = True
SQLITE_PATH = "chemeng_cache.sqlite"

def init_sqlite():
    conn = sqlite3.connect(SQLITE_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    return conn

_sql_conn = init_sqlite() if ENABLE_SQL_CACHE else None

def cache_key(model, messages):
    raw = model + json.dumps(messages, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

def mem_get(key):
    return st.session_state.prompt_cache.get(key)

def mem_set(key, value):
    st.session_state.prompt_cache[key] = value
    if len(st.session_state.prompt_cache) > 2000:
        st.session_state.prompt_cache.pop(next(iter(st.session_state.prompt_cache)))

def sql_get(key):
    if not _sql_conn:
        return None
    cur = _sql_conn.cursor()
    cur.execute("SELECT value FROM cache WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row else None

def sql_set(key, value):
    if not _sql_conn:
        return
    cur = _sql_conn.cursor()
    cur.execute("INSERT OR REPLACE INTO cache (key,value) VALUES (?,?)", (key, value))
    _sql_conn.commit()

# ======================================
# QUERY OPENROUTER WITH BACKOFF & FALLBACKS
# ======================================
def query_openrouter_with_backoff(model: str, messages: List[Dict[str, str]], max_retries=4) -> str:
    key = cache_key(model, messages)

    # 1️⃣ Check memory cache
    cached = mem_get(key)
    if cached:
        return cached

    # 2️⃣ Check SQLite cache
    cached_sql = sql_get(key)
    if cached_sql:
        mem_set(key, cached_sql)
        return cached_sql

    backoff = 2.0
    for attempt in range(max_retries):
        try:
            payload = {"model": model, "messages": messages}
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
            
            if resp.status_code == 429:
                # explicit 429 detection
                retry_after = int(resp.headers.get("Retry-After", backoff))
                time.sleep(retry_after)
                backoff *= 2
                continue
            
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data and data["choices"]:
                answer = data["choices"][0]["message"]["content"]
                mem_set(key, answer)
                sql_set(key, answer)
                return answer
            return json.dumps(data)
        except requests.HTTPError as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            if attempt == max_retries - 1:
                raise e
        except Exception as e:
            if attempt == max_retries - 1:
                return f"⚠️ Error: {e}"
            time.sleep(backoff)
            backoff *= 2
    return "⚠️ Request failed after retries."

# ======================================
# MODEL FALLBACK CHAIN
# ======================================
def query_models_with_fallbacks(messages: List[Dict[str, str]]):
    models_to_try = [MODEL_MAIN] + MODEL_FALLBACKS
    for m in models_to_try:
        try:
            return query_openrouter_with_backoff(m, messages)
        except Exception as e:
            st.warning(f"{m} failed → {e}")
            continue
    return "⚠️ All models are busy. Please retry later."

# ======================================
# VANILLA RAG PIPELINE
# ======================================
def vanilla_rag_answer(question: str, deep_mode=False) -> str:
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

    base_system = (
        "You are ChemEng Buddy, a helpful tutor for Chemical Engineering. "
        "Explain concepts clearly, step by step, with examples and highlight common mistakes. "
        "Stay focused only on chemical engineering topics."
    )

    # Adaptive DeepThink mode triggers only for complex questions
    if deep_mode:
        base_system += " Use deeper reasoning, derive equations symbolically, and cross-check assumptions."

    messages = [
        {"role": "system", "content": base_system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    return query_models_with_fallbacks(messages)

# ======================================
# CHAT UI
# ======================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

if user_query := st.chat_input("Ask me about Chemical Engineering"):
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("Thinking..."):
        deep_mode = len(user_query.split()) > 25  # Adaptive DeepThink
        answer = vanilla_rag_answer(user_query, deep_mode=deep_mode)
    
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.last_answer_animated = True
    st.rerun()

# ======================================
# RENDER CHAT
# ======================================
for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        if (
            i == len(st.session_state.chat_history) - 1
            and chat["role"] == "assistant"
            and st.session_state.last_answer_animated
        ):
            type_like_chatgpt(chat["content"])
            st.session_state.last_answer_animated = False
        else:
            st.markdown(chat["content"])

st.markdown("""<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ❤️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)
