# streamlit_app.py
import os
import time
import json
import requests
import streamlit as st
from typing import List, Dict
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import firebase_admin
from firebase_admin import credentials, db

# ================== CONFIG ==================
st.set_page_config(page_title="⚗ ChemEng Buddy", layout="wide")

# Secrets
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
EMBED_MODEL = st.secrets.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
K_VAL = int(st.secrets.get("K_VAL", 4))
FIREBASE_DATABASE_URL = st.secrets.get("firebase", {}).get("databaseURL", "")

LLM_MODELS = st.secrets.get("LLM_MODELS", [
    "deepseek/deepseek-chat-v3.1:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "qwen/qwen3-4b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1:free"
])

FAISS_INDEX_URL = st.secrets.get("FAISS_INDEX_URL", "https://huggingface.co/datasets/prakhar146/medical/resolve/main/index.faiss")
FAISS_PKL_URL = st.secrets.get("FAISS_PKL_URL", "https://huggingface.co/datasets/prakhar146/medical/resolve/main/index.pkl")
LOCAL_FAISS_DIR = Path(st.secrets.get("LOCAL_FAISS_DIR", "./faiss_store"))
LOCAL_FAISS_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

# ================== UTILITIES ==================
def encode_email(email: str) -> str:
    return email.replace(".", "dot").replace("@", "at")

def now_ts() -> int:
    return int(time.time())

# ================== FIREBASE ==================
def init_firebase():
    if not FIREBASE_DATABASE_URL:
        st.warning("Firebase database URL not configured in st.secrets.")
        return False
    try:
        if not firebase_admin._apps:
            svc_json = st.secrets.get("SERVICE_ACCOUNT", {}).get("key")
            if not svc_json:
                st.warning("SERVICE_ACCOUNT key missing in st.secrets.")
                return False
            svc_dict = json.loads(svc_json)
            if "private_key" in svc_dict:
                svc_dict["private_key"] = svc_dict["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(svc_dict)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        return True
    except Exception as e:
        st.error(f"⚠ Firebase init failed: {e}")
        return False

_FIRE_INIT = init_firebase()

def load_chat_history(email: str) -> List[Dict]:
    if not _FIRE_INIT:
        return []
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        data = ref.get()
        if not data:
            return []
        chats = sorted(list(data.values()), key=lambda x: x.get("timestamp", 0))
        return chats
    except Exception as e:
        st.warning(f"⚠ Failed to load chat history: {e}")
        return []

def save_message(email: str, role: str, content: str):
    if not _FIRE_INIT:
        return
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        ref.push({"role": role, "content": content, "timestamp": now_ts()})
    except Exception as e:
        st.warning(f"⚠ Failed to save message: {e}")

# ================== STATE ==================
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================== FAISS SETUP ==================
def download_file(url: str, local_path: Path):
    try:
        if not local_path.exists() or local_path.stat().st_size == 0:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
    except Exception as e:
        st.warning(f"Failed to download {url}: {e}")
        raise

@st.cache_resource(show_spinner=False)
def load_vector_db():
    try:
        faiss_path = LOCAL_FAISS_DIR / "index.faiss"
        pkl_path = LOCAL_FAISS_DIR / "index.pkl"
        if FAISS_INDEX_URL and not faiss_path.exists():
            download_file(FAISS_INDEX_URL, faiss_path)
        if FAISS_PKL_URL and not pkl_path.exists():
            download_file(FAISS_PKL_URL, pkl_path)
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = FAISS.load_local(str(LOCAL_FAISS_DIR), embedder, allow_dangerous_deserialization=True)
        return vectordb.as_retriever(search_type="similarity", k=K_VAL)
    except Exception as e:
        st.warning(f"Could not load vector DB: {e}")
        return None

with st.spinner("Loading knowledge base..."):
    retriever = load_vector_db()

# ================== LLM QUERY ==================
def query_openrouter(messages: List[Dict[str, str]]) -> str:
    for model in LLM_MODELS:
        try:
            payload = {"model": model, "messages": messages}
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                msg = data["choices"][0]
                return msg.get("message", {}).get("content", "") or msg.get("text", "")
        except Exception:
            continue
    return "⚠️ All LLMs failed to respond. Please try again later."

# ================== PROMPT BUILD ==================
def build_prompt_with_context(user_question: str) -> List[Dict]:
    docs = []
    if retriever:
        try:
            docs = retriever.get_relevant_documents(user_question)
        except Exception as e:
            st.warning(f"Retriever failed: {e}")
    doc_text = "\n".join([d.page_content for d in docs]) if docs else "No relevant context found."
    system = "You are ChemEng Buddy, a friendly Chemical Engineering tutor. Explain clearly, step by step, and give examples."
    user = f"Context:\n{doc_text}\n\nQuestion: {user_question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# ================== LOGIN PAGE ==================
def login_page():
    st.markdown(
        """
        <style>
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 80vh;
            text-align: center;
        }
        .stTextInput>div>div>input {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='centered'>", unsafe_allow_html=True)
    st.title("⚗ ChemEng Buddy")
    st.subheader("Your friendly Chemical Engineering study partner")
    st.markdown("### Please log in to continue")

    with st.form("login_form", clear_on_submit=False):
        name = st.text_input("Full Name", placeholder="John Doe")
        email = st.text_input("Email", placeholder="example@email.com")
        submitted = st.form_submit_button("Login 🔐")
        if submitted:
            if not name or not email:
                st.warning("Please enter both name and email.")
            else:
                st.session_state.auth_user = {"name": name.strip(), "email": email.strip().lower()}
                st.session_state.chat_history = load_chat_history(email.strip().lower())
                st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ================== CHAT PAGE ==================
def chat_page():
    st.sidebar.title("⚙ Controls")
    if st.sidebar.button("Logout 🚪"):
        st.session_state.auth_user = None
        st.session_state.chat_history = []
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Retrieval Info**")
    if retriever:
        st.sidebar.success("Retriever: loaded")
    else:
        st.sidebar.warning("Retriever not available")

    if st.sidebar.button("Refresh Chat History"):
        email = st.session_state.auth_user["email"]
        st.session_state.chat_history = load_chat_history(email)
        st.experimental_rerun()

    # --- Main Chat UI ---
    st.title(f"👋 Hello, {st.session_state.auth_user['name']}")
    st.caption("Start chatting about Chemical Engineering topics below!")

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    user_input = st.chat_input("Ask me anything about Chemical Engineering ⚗")
    if user_input:
        email = st.session_state.auth_user["email"]
        st.session_state.chat_history.append({"role": "user", "content": user_input, "timestamp": now_ts()})
        save_message(email, "user", user_input)

        with st.spinner("Thinking..."):
            messages = build_prompt_with_context(user_input)
            answer = query_openrouter(messages)

        st.session_state.chat_history.append({"role": "assistant", "content": answer, "timestamp": now_ts()})
        save_message(email, "assistant", answer)
        st.experimental_rerun()

# ================== MAIN APP ==================
if st.session_state.auth_user:
    chat_page()
else:
    login_page()
