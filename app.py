import os
import json
import requests
import streamlit as st
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ================== CONFIG ==================
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)
LLM_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "qwen/qwen3-4b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1:free"
]
FAISS_INDEX_URL = "https://huggingface.co/datasets/prakhar146/medical/resolve/main/index.faiss"
FAISS_PKL_URL = "https://huggingface.co/datasets/prakhar146/medical/resolve/main/index.pkl"
LOCAL_FAISS_DIR = "./faiss_store"
os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)

# ================== STREAMLIT SETUP ==================
st.set_page_config(page_title="ChemEng Buddy", layout="wide")
st.title("⚗ ChemEng Buddy")
st.markdown("Your friendly Chemical Engineering study partner")

try:
    import pyrebase
except ImportError:
    pyrebase = None

# ================== FIREBASE INIT ==================
def show_missing_auth_setup():
    st.warning(
        "⚠ Firebase setup missing!\n\n"
        "1. Install pyrebase4 → `pip install pyrebase4`\n"
        "2. Add Firebase config in `.streamlit/secrets.toml` under `[firebase]`."
    )

def init_firebase():
    if "firebase_app" not in st.session_state:
        st.session_state.firebase_app = None
        st.session_state.firebase_auth = None
        st.session_state.firebase_db = None

    if st.session_state.firebase_app and st.session_state.firebase_auth:
        return st.session_state.firebase_app, st.session_state.firebase_auth, st.session_state.firebase_db

    if not pyrebase:
        st.warning("⚠ Please install pyrebase4: pip install pyrebase4")
        return None, None, None

    fb_cfg = st.secrets.get("firebase", None)
    if not fb_cfg:
        st.warning(
            "⚠ Firebase setup missing in secrets.toml!\n"
            "Make sure you have `[firebase]` section with correct keys."
        )
        return None, None, None

    try:
        app = pyrebase.initialize_app(dict(fb_cfg))
        auth = app.auth()
        db = app.database() if "databaseURL" in fb_cfg else None
    except Exception as e:
        st.error(f"Firebase init failed: {e}")
        return None, None, None

    st.session_state.firebase_app = app
    st.session_state.firebase_auth = auth
    st.session_state.firebase_db = db
    return app, auth, db

# ================== AUTH UI ==================
def render_auth_ui():
    """Handles Firebase-like login (manual ID token entry fallback)."""
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None
        st.session_state.id_token = None

    app, auth, db = init_firebase()

    if not app or not auth:
        st.warning("Firebase not configured.")
        return None

    with st.sidebar:
        if st.session_state.auth_user:
            st.success(f"Signed in as {st.session_state.auth_user.get('email', 'User')}")
            if st.button("Logout"):
                st.session_state.auth_user = None
                st.session_state.chat_history = []
                st.session_state.id_token = None
                st.rerun()
            return st.session_state.auth_user
        else:
            st.info("🔐 Sign in with your Firebase ID token.")
            token = st.text_input("Paste your Firebase ID Token (JWT)", type="password")
            if st.button("Sign In"):
                try:
                    info = auth.get_account_info(token)
                    email = info["users"][0].get("email", "unknown@user")
                    user = {"email": email, "idToken": token}
                    st.session_state.auth_user = user
                    st.session_state.id_token = token
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
    return None

# ================== FIREBASE CHAT HISTORY ==================
def encode_email(email: str) -> str:
    return email.replace(".", "_dot_").replace("@", "_at_")

def load_chat_history(email: str):
    _, _, db = init_firebase()
    if not db:
        return []
    try:
        data = db.child("users").child(encode_email(email)).child("chats").get().val()
        if not data:
            return []
        return sorted(list(data.values()), key=lambda x: x.get("timestamp", 0))
    except Exception as e:
        st.warning(f"⚠ Failed to load chat history: {e}")
        return []

def save_chat_message(email: str, role: str, content: str):
    _, _, db = init_firebase()
    if not db:
        return
    try:
        db.child("users").child(encode_email(email)).child("chats").push({
            "role": role,
            "content": content,
            "timestamp": int(time.time())
        })
    except Exception as e:
        st.warning(f"⚠ Failed to save message: {e}")

# ================== UI HELPERS ==================
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
    vectordb = FAISS.load_local(LOCAL_FAISS_DIR, embedder, allow_dangerous_deserialization=True)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

retriever = load_vector_db()

# ================== OPENROUTER ==================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

def query_openrouter(messages: List[Dict[str, str]]) -> str:
    for model_name in LLM_MODELS:
        try:
            thinking_animation(duration=2)
            payload = {"model": model_name, "messages": messages}
            r = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
        except Exception:
            continue
    return "⚠ All LLMs failed to respond."

# ================== CONTEXT BUILDER ==================
def build_prompt_with_context(user_question: str, chat_history: List[Dict]):
    docs = retriever.get_relevant_documents(user_question)
    doc_text = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."

    return [
        {"role": "system", "content": (
            "You are ChemEng Buddy, a helpful tutor for Chemical Engineering. "
            "Explain clearly, step by step, with examples and common mistakes."
        )},
        {"role": "user", "content": f"Context:\n{doc_text}\n\nQuestion: {user_question}"}
    ]

# ================== MAIN APP ==================
user = render_auth_ui()
if not user:
    st.stop()

email = user.get("email")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(email)
if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

if user_query := st.chat_input("Ask me about Chemical Engineering"):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    save_chat_message(email, "user", user_query)

    with st.spinner("Thinking..."):
        prompt = build_prompt_with_context(user_query, st.session_state.chat_history)
        answer = query_openrouter(prompt)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    save_chat_message(email, "assistant", answer)
    st.session_state.last_answer_animated = True
    st.rerun()

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
        📬 Email: <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True,
)
