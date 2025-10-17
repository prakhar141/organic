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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# ================== AUTH: Firebase (Google Sign-In) ==================
# Assumptions:
# - Streamlit secrets contains Firebase config/service account under st.secrets["firebase"]
#   Example .streamlit/secrets.toml:
#   [firebase]
#   apiKey = "..."
#   authDomain = "..."
#   projectId = "..."
#   storageBucket = "..."
#   messagingSenderId = "..."
#   appId = "..."
#   databaseURL = "https://<your-db>.firebaseio.com"
#   serviceAccount = "{ JSON of the service account }"  # optional if using admin features
#
# We use pyrebase4 for client-side Google OAuth redirect flow inside Streamlit.
# Install: pip install pyrebase4
try:
    import pyrebase
except Exception:
    pyrebase = None

def init_firebase():
    if "firebase_app" in st.session_state:
        return st.session_state.firebase_app, st.session_state.firebase_auth
    fb_cfg = st.secrets.get("firebase", {})
    if not fb_cfg:
        st.error("Firebase secrets not found. Add [firebase] in .streamlit/secrets.")
        return None, None
    # pyrebase expects specific keys; map common ones if present
    cfg = {
        "apiKey": fb_cfg.get("apiKey"),
        "authDomain": fb_cfg.get("authDomain"),
        "projectId": fb_cfg.get("projectId"),
        "storageBucket": fb_cfg.get("storageBucket"),
        "messagingSenderId": fb_cfg.get("messagingSenderId"),
        "appId": fb_cfg.get("appId"),
        "databaseURL": fb_cfg.get("databaseURL"),
    }
    app = pyrebase.initialize_app(cfg) if pyrebase else None
    auth = app.auth() if app else None
    st.session_state.firebase_app = app
    st.session_state.firebase_auth = auth
    return app, auth

def render_auth_ui():
    """Renders Google Sign-In/Logout and returns the authenticated user info or None."""
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
        st.session_state.id_token = None

    col1, col2 = st.columns([1,3])
    with col1:
        st.subheader("Login")
    with col2:
        st.caption("Use Google to sign in before using the app")

    app, auth = init_firebase()
    if not pyrebase or not app or not auth:
        st.warning("pyrebase not available or Firebase not configured. Skipping auth in dev mode.")
        return None

    # Simple approach: use 'Sign in with Google' via OAuth using custom token flow.
    # Pyrebase does not provide direct popup-based Google sign-in; in Streamlit, we can
    # open the provider link in a new tab. As a pragmatic approach for Streamlit Cloud,
    # we accept ID token pasted back (or reuse cached session) after redirect.
    # For better UX, consider streamlit-authenticator or custom OAuth redirect handler.

    # Cached session
    if st.session_state.auth_user and st.session_state.id_token:
        with st.sidebar:
            st.success(f"Signed in as {st.session_state.auth_user.get('email', 'Google user')}")
            if st.button("Logout"):
                st.session_state.auth_user = None
                st.session_state.id_token = None
                st.rerun()
        return st.session_state.auth_user

    with st.sidebar:
        st.info("Sign in with Google to continue")
        # Provide a link flow instructions
        st.write("Click the button below to start Google sign-in.")
        start = st.button("Sign in with Google")
        if start:
            st.session_state.show_google_instructions = True

        if st.session_state.get("show_google_instructions"):
            st.markdown(
                "1) Open Firebase Google sign-in page in a new tab.\n\n"
                "2) Complete login and retrieve the ID token (from redirect/callback app).\n\n"
                "3) Paste the ID token here to finish sign-in."
            )
            id_token = st.text_input("Paste Google ID token (JWT)", type="password")
            if st.button("Complete sign-in") and id_token:
                try:
                    user = auth.sign_in_with_custom_token(id_token)  # If using custom token
                except Exception:
                    # Fallback: verify ID token as a bearer with get_account_info
                    try:
                        info = auth.get_account_info(id_token)
                        # emulate user record
                        user = {"email": info.get("users", [{}])[0].get("email", "google-user")}
                    except Exception as e:
                        st.error(f"Failed to verify token: {e}")
                        user = None
                if user:
                    st.session_state.auth_user = user
                    st.session_state.id_token = id_token
                    st.rerun()

    return None

# More pragmatic alternative for Streamlit: use streamlit-oauth or streamlit-authenticator
# Here we keep a simple gate that can be swapped out later.

# ================== UI helpers ==================
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
c.execute(
    """
CREATE TABLE IF NOT EXISTS qa_cache (
    question_hash TEXT PRIMARY KEY,
    question TEXT,
    answer TEXT,
    success INTEGER DEFAULT 1,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""
)
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
    c.execute(
        """
        INSERT OR REPLACE INTO qa_cache (question_hash, question, answer, success)
        VALUES (?, ?, ?, ?)
    """,
        (q_hash, question, answer, success),
    )
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

# ================== FOLLOW-UP LOGIC: LAST QUESTION ONLY ==================

def is_followup_recent(user_question: str, threshold=0.7):
    """
    Checks if the user_question is a follow-up to the last user question in chat history.
    Returns the context string if follow-up detected, else empty string.
    """
    if not st.session_state.chat_history or len(st.session_state.chat_history) < 2:
        return ""

    # Find the last user question and its assistant answer
    last_user_idx = max(i for i, msg in enumerate(st.session_state.chat_history) if msg["role"] == "user")
    last_user_q = st.session_state.chat_history[last_user_idx]["content"]
    last_assistant_a = (
        st.session_state.chat_history[last_user_idx + 1]["content"]
        if last_user_idx + 1 < len(st.session_state.chat_history)
        else ""
    )

    if not last_assistant_a:
        return ""

    # Embed the new question and the last question
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    new_vec = embedder.embed_query(user_question)
    last_vec = embedder.embed_query(last_user_q)

    sim = cosine_similarity([new_vec], [last_vec])[0][0]

    if sim >= threshold:
        return (
            f"Refer to previous Q&A (similarity={sim:.2f}):\n"
            f"User: {last_user_q}\nAssistant: {last_assistant_a}\n\n"
        )

    return ""

def build_prompt_with_context(user_question: str):
    followup_context = is_followup_recent(user_question)

    docs = retriever.get_relevant_documents(user_question)
    doc_text = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

    prompt = [
        {
            "role": "system",
            "content": (
                "You are ChemEng Buddy, a helpful tutor for chemical engineering. "
                "Explain concepts clearly, step by step, with examples and common mistakes. "
                "Stay focused only on chemical engineering topics."
            ),
        },
        {"role": "user", "content": followup_context + f"Context:\n{doc_text}\n\nQuestion: {user_question}"},
    ]
    return prompt

# ================== AUTH GATE ==================
user = render_auth_ui()
if not user:
    st.stop()

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

st.markdown(
    """
    <hr style="margin-top: 40px;"/>
    <div style="text-align: center; color: #888; font-size: 14px;">
        Built with ❤ by Prakhar Mathur · BITS Pilani ·
        <br/>
        📬 Email: <a href=\"mailto:prakhar.mathur2020@gmail.com\">prakhar.mathur2020@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True,
)
