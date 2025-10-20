# streamlit_app.py
import os
import time
import json
import requests
import streamlit as st
from typing import List, Dict
from pathlib import Path

# LangChain / FAISS imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# ================== CONFIG ==================
st.set_page_config(page_title="⚗ ChemEng Buddy", layout="wide")

# Secrets expected in Streamlit secrets
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
EMBED_MODEL = st.secrets.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
K_VAL = int(st.secrets.get("K_VAL", 4))

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

# Firebase service account
FIREBASE_DATABASE_URL = st.secrets.get("firebase", {}).get("databaseURL", "")

# ================== UTILITIES ==================
def encode_email(email: str) -> str:
    """Encode email to safe Firebase key."""
    return email.replace(".", "dot").replace("@", "at")

def now_ts() -> int:
    return int(time.time())

# ================== FIREBASE INITIALIZATION ==================
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
            # Ensure correct newline formatting for private_key
            if "private_key" in svc_dict:
                svc_dict["private_key"] = svc_dict["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(svc_dict)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        return True
    except Exception as e:
        st.error(f"⚠ Firebase init failed: {e}")
        return False

_FIRE_INITIALIZED = init_firebase()

def load_chat_history_from_firebase(email: str) -> List[Dict]:
    if not _FIRE_INITIALIZED:
        return []
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        data = ref.get()
        if not data:
            return []
        # Convert dict-of-dicts -> list sorted by timestamp
        chats = sorted(list(data.values()), key=lambda x: x.get("timestamp", 0))
        return chats
    except Exception as e:
        st.warning(f"⚠ Failed to load chat history: {e}")
        return []

def save_chat_message_to_firebase(email: str, role: str, content: str):
    if not _FIRE_INITIALIZED:
        st.warning("Firebase not initialized; skipping save.")
        return
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        ref.push({"role": role, "content": content, "timestamp": now_ts()})
    except Exception as e:
        st.warning(f"⚠ Failed to save message: {e}")

# ================== SESSION STATE SETUP ==================
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

# ================== SIDEBAR LOGIN ==================
with st.sidebar:
    st.header("Account")
    if st.session_state.auth_user:
        st.success(f"✅ Logged in as {st.session_state.auth_user['name']} ({st.session_state.auth_user['email']})")
        if st.button("Logout"):
            st.session_state.auth_user = None
            st.session_state.chat_history = []
            st.experimental_rerun()
    else:
        st.markdown("🔐 **Login to ChemEng Buddy**")
        with st.form("login_form", clear_on_submit=False):
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
            submitted = st.form_submit_button("Login")
            if submitted:
                if not full_name or not email:
                    st.warning("Please enter both name and email.")
                else:
                    st.session_state.auth_user = {"name": full_name.strip(), "email": email.strip().lower()}
                    st.session_state.chat_history = load_chat_history_from_firebase(email.strip().lower())
                    st.success("Logged in successfully!")
                    st.experimental_rerun()

# ================== UI: Title + Description ==================
st.title("⚗ ChemEng Buddy")
st.markdown("Your friendly Chemical Engineering study partner. Log in to save and view your chats.")

# ================== ANIMATION HELPERS ==================
def type_like_chatgpt(text: str, speed: float = 0.004):
    placeholder = st.empty()
    animated = ""
    for c in text:
        animated += c
        placeholder.markdown(animated + "|")
        time.sleep(speed)
    placeholder.markdown(animated)

def thinking_animation(duration: int = 2):
    placeholder = st.empty()
    for i in range(duration * 4):
        dots = "." * (i % 4)
        placeholder.markdown(f"🤔 Thinking{dots}")
        time.sleep(0.25)
    placeholder.empty()

# ================== FAISS DOWNLOAD & LOAD ==================
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
    """Download the FAISS files if needed, then load the FAISS index as a retriever."""
    try:
        faiss_path = LOCAL_FAISS_DIR / "index.faiss"
        pkl_path = LOCAL_FAISS_DIR / "index.pkl"
        # download if missing
        if FAISS_INDEX_URL and (not faiss_path.exists()):
            download_file(FAISS_INDEX_URL, faiss_path)
        if FAISS_PKL_URL and (not pkl_path.exists()):
            download_file(FAISS_PKL_URL, pkl_path)
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = FAISS.load_local(str(LOCAL_FAISS_DIR), embedder, allow_dangerous_deserialization=True)
        return vectordb.as_retriever(search_type="similarity", k=K_VAL)
    except Exception as e:
        st.warning(f"Could not load vector DB: {e}")
        return None

with st.spinner("Loading retrieval index..."):
    retriever = load_vector_db()
    if retriever is None:
        st.info("Retriever not available — responses will not include context from the vector store.")

# ================== OPENROUTER QUERY ==================
def query_openrouter(messages: List[Dict[str, str]]) -> str:
    """Try models in LLM_MODELS order, return first successful answer."""
    for model_name in LLM_MODELS:
        try:
            thinking_animation(duration=1)
            payload = {"model": model_name, "messages": messages}
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if isinstance(choice.get("message"), dict):
                    return choice["message"].get("content", "")
                else:
                    return choice.get("text", "")
        except Exception:
            continue
    return "⚠️ All LLMs failed to respond. Please try again later."

# ================== PROMPT / CONTEXT BUILDING ==================
def build_prompt_with_context(user_question: str, chat_history: List[Dict]) -> List[Dict]:
    docs = []
    if retriever:
        try:
            docs = retriever.get_relevant_documents(user_question)
        except Exception as e:
            st.warning(f"Retriever failed: {e}")
            docs = []
    doc_text = "\n".join([d.page_content for d in docs]) if docs else "No relevant context found."
    system_msg = (
        "You are ChemEng Buddy, a friendly Chemical Engineering tutor. "
        "Explain clearly, step by step, show reasoning, and give examples where possible."
    )
    user_msg = f"Context:\n{doc_text}\n\nQuestion: {user_question}"
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

# ================== MAIN CHAT UI ==================
main_col, right_col = st.columns([3, 1])

with main_col:
    if st.session_state.auth_user:
        st.subheader(f"Hello, {st.session_state.auth_user['name']} 👋")
        st.markdown("---")

        if not st.session_state.chat_history:
            st.info("No previous chats found — start a new conversation below!")

        for chat in st.session_state.chat_history:
            role = chat.get("role", "user")
            content = chat.get("content", "")
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
    else:
        st.info("Please log in (sidebar) to load your saved chats and start chatting.")

    st.markdown("---")

    # Chat input — only enabled when logged in
    if st.session_state.auth_user:
        user_query = st.chat_input("Ask me anything about Chemical Engineering ⚗")
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query, "timestamp": now_ts()})
            save_chat_message_to_firebase(st.session_state.auth_user["email"], "user", user_query)

            with st.spinner("Thinking..."):
                prompt = build_prompt_with_context(user_query, st.session_state.chat_history)
                answer = query_openrouter(prompt)

            st.session_state.chat_history.append({"role": "assistant", "content": answer, "timestamp": now_ts()})
            save_chat_message_to_firebase(st.session_state.auth_user["email"], "assistant", answer)
            st.experimental_rerun()
    else:
        st.write("Log in to start chatting and have your chats saved to your account.")

# ================== RIGHT SIDEBAR ==================
with right_col:
    st.header("Controls & Info")
    if retriever:
        st.success("Retriever: loaded")
        st.caption(f"Embedding model: {EMBED_MODEL}")
    else:
        st.warning("Retriever not available")
    st.write("Debug / quick actions:")
    if st.button("Reload vector DB"):
        load_vector_db.clear()
        st.experimental_rerun()
    if st.button("Refresh chat history from server"):
        if st.session_state.auth_user and st.session_state.auth_user.get("email"):
            st.session_state.chat_history = load_chat_history_from_firebase(st.session_state.auth_user["email"])
            st.experimental_rerun()
        else:
            st.warning("You must be logged in to refresh chat history.")

# Footer
st.markdown("---")
st.markdown("**Notes:**\n- Log in with your name and email to access and save your chat history.\n- Chats are stored under your email in Firebase.\n- No Google OAuth required.")
