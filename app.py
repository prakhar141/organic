import os
import json
import requests
import streamlit as st
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import sqlite3
import time
import hashlib

# ================== CONFIG ==================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)

# List of free LLMs on OpenRouter (as of now)
LLM_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "qwen/qwen3-4b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1:free"
]

# FAISS URLs
FAISS_INDEX_URL = "https://huggingface.co/datasets/prakhar146/medical/resolve/main/index.faiss"
FAISS_PKL_URL = "https://huggingface.co/datasets/prakhar146/medical/resolve/main/index.pkl"
LOCAL_FAISS_DIR = "./faiss_store"
os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)

# ================== STREAMLIT SETUP ==================
st.set_page_config(page_title="ChemEng Buddy", layout="wide")
st.title("⚗ ChemEng Buddy")
st.markdown("Your friendly Chemical Engineering study partner")

def type_like_chatgpt(text, speed=0.004):
    placeholder = st.empty()
    animated = ""
    for c in text:
        animated += c
        placeholder.markdown(animated + "|")
        time.sleep(speed)
    placeholder.markdown(animated)

def thinking_animation(duration=3):
    placeholder = st.empty()
    for i in range(duration * 4):
        dots = "." * (i % 4)
        placeholder.markdown(f"🤔 Thinking{dots}")
        time.sleep(0.25)
    placeholder.empty()

# ================== DOWNLOAD FAISS ==================
def download_file(url: str, local_path: str):
    if not os.path.exists(local_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

download_file(FAISS_INDEX_URL, os.path.join(LOCAL_FAISS_DIR, "index.faiss"))
download_file(FAISS_PKL_URL, os.path.join(LOCAL_FAISS_DIR, "index.pkl"))

# ================== VECTOR DB ==================
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

# ================== SQLITE CACHE ==================
DB_PATH = "chemeng_cache.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS qa_cache (
    question_hash TEXT PRIMARY KEY,
    question TEXT,
    answer TEXT,
    success INTEGER DEFAULT 1,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def get_question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()

def get_cached_answer(question: str):
    q_hash = get_question_hash(question)
    c.execute("SELECT answer, success FROM qa_cache WHERE question_hash = ?", (q_hash,))
    row = c.fetchone()
    if row:
        answer, success = row
        return answer, success
    return None, None

def set_cached_answer(question: str, answer: str, success: int = 1):
    q_hash = get_question_hash(question)
    c.execute("""
        INSERT OR REPLACE INTO qa_cache (question_hash, question, answer, success)
        VALUES (?, ?, ?, ?)
    """, (q_hash, question, answer, success))
    conn.commit()

# ================== OPENROUTER QUERY WITH FALLBACK ==================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

def query_openrouter_with_fallback(messages: List[Dict[str, str]]) -> str:
    cached_answer, success = get_cached_answer(messages[-1]["content"])
    if cached_answer and success:
        return cached_answer
    if cached_answer and not success:
        return "⚠ Previous API call failed. Skipping repeated attempt."

    for model_name in LLM_MODELS:
        try:
            thinking_animation(duration=2)
            payload = {"model": model_name, "messages": messages}
            r = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "choices" in data and data["choices"]:
                answer = data["choices"][0]["message"]["content"]
                set_cached_answer(messages[-1]["content"], answer, success=1)
                return answer
            set_cached_answer(messages[-1]["content"], json.dumps(data), success=0)
        except Exception as e:
            set_cached_answer(messages[-1]["content"], str(e), success=0)
            continue
    return "⚠ All LLMs failed to respond."

# ================== FOLLOW-UP LOGIC ==================
def build_prompt_with_context(user_question: str):
    # Include previous answer if follow-up detected
    context_text = ""
    if st.session_state.chat_history:
        last_assistant = next(
            (msg for msg in reversed(st.session_state.chat_history) if msg["role"] == "assistant"),
            None
        )
        last_user = next(
            (msg for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"),
            None
        )
        if last_assistant and last_user:
            # Very simple heuristic: follow-up if question is short or contains pronouns
            followup_indicators = ["it", "they", "this", "that", "these", "those", "how", "why", "what"]
            if any(word in user_question.lower() for word in followup_indicators) or len(user_question.split()) < 6:
                context_text = f"Refer to previous Q&A:\nUser: {last_user['content']}\nAssistant: {last_assistant['content']}\n\n"

    docs = retriever.get_relevant_documents(user_question)
    doc_text = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."
    
    prompt = [
        {"role": "system", "content": (
            "You are ChemEng Buddy, a helpful tutor for chemical engineering. "
            "Explain concepts clearly, step by step, with examples and common mistakes. "
            "Stay focused only on chemical engineering topics."
        )},
        {"role": "user", "content": context_text + f"Context:\n{doc_text}\n\nQuestion: {user_question}"}
    ]
    return prompt

# ================== CHAT INTERFACE ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

if user_query := st.chat_input("Ask me about Chemical Engineering"):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.spinner("Thinking..."):
        prompt = build_prompt_with_context(user_query)
        answer = query_openrouter_with_fallback(prompt)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.last_answer_animated = True
    st.rerun()

# Show chat history
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
    Built with ❤ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)
