# app.py
import os
import time
import requests
import streamlit as st
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pyrebase
from urllib.parse import urljoin

# ----------------- Helpers for secrets -----------------
def get_secret(key: str, default=None):
    """Safe wrapper around st.secrets.get"""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def get_nested_secret(section: str):
    """Return dict or None for a nested secret section (like [firebase])"""
    try:
        return st.secrets.get(section, None)
    except Exception:
        return None

# ================== CONFIG ==================
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY", "")
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

# ----------------- Load Firebase config safely -----------------
firebase_config = get_nested_secret("firebase")
redirect_url = get_secret("redirect_url", None)

if not firebase_config:
    st.error(
        "Firebase configuration is missing. Add a `[firebase]` section to your Streamlit secrets "
        "or configure it in the app settings on Streamlit Cloud."
    )
    st.stop()

# minimal keys check
if not firebase_config.get("authDomain"):
    st.error("firebase.authDomain is missing from secrets. Please add it.")
    st.stop()

# ================== FIREBASE CLIENT ==================
try:
    firebase = pyrebase.initialize_app(firebase_config)
    auth = firebase.auth()
    db = firebase.database()
except Exception as e:
    st.error(f"Failed to initialize Firebase client: {e}")
    st.stop()

# ================== GOOGLE SIGN-IN (OAuth via redirect) ==================
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# Read idToken / id_token param from query params safely
query_params = st.experimental_get_query_params()
id_token_list = query_params.get("idToken") or query_params.get("id_token") or []
id_token = id_token_list[0] if id_token_list else None

if id_token and not st.session_state.auth_user:
    try:
        # Note: pyrebase.sign_in_with_custom_token expects a Firebase custom token.
        # If your redirect is providing an OAuth idToken (from Google), you may need to
        # use a different auth method — adjust as needed.
        user_info = auth.sign_in_with_custom_token(id_token)
        st.session_state.auth_user = user_info
        # Clear the query params to keep URL clean
        st.experimental_set_query_params()
        st.experimental_rerun()
    except Exception as e:
        st.error(f"⚠ Failed to sign in with token from redirect: {e}")

with st.sidebar:
    if st.session_state.auth_user:
        # Show email if available
        user_email = st.session_state.auth_user.get("email") if isinstance(st.session_state.auth_user, dict) else None
        st.success(f"Signed in as {user_email or 'user'}")
        if st.button("Logout"):
            st.session_state.auth_user = None
            st.session_state.chat_history = []
            st.experimental_rerun()
    else:
        st.markdown("🔐 Sign in with Google")
        # Build sign-in URL only if redirect_url exists
        auth_domain = firebase_config.get("authDomain", "").rstrip("/")
        if redirect_url and auth_domain:
            # firebase hosted auth handler path
            handler_path = "/__/auth/handler"
            query = (
                f"?mode=signIn&providerId=google.com&redirectUrl={redirect_url}"
            )
            auth_url = f"{auth_domain}{handler_path}{query}"
            st.info("Click below to sign in with Google:")
            st.markdown(f"[Sign in with Google]({auth_url})", unsafe_allow_html=True)
        else:
            # Give clear instruction to the developer/admin how to fix it
            st.warning(
                "OAuth cannot be started because `redirect_url` is not configured in secrets.\n\n"
                "To fix on Streamlit Cloud:\n"
                "1. Open your app → Settings → Secrets.\n"
                "2. Add `redirect_url = \"https://your-app.streamlit.app/\"` and save.\n\n"
                "You can still use the app locally if you configure `secrets.toml` locally."
            )

# Stop app if not logged in (original behavior). If you prefer to allow anonymous access,
# remove this block.
if not st.session_state.auth_user:
    st.stop()

# safe access to email
email = st.session_state.auth_user.get("email") if isinstance(st.session_state.auth_user, dict) else None
if not email:
    st.error("Signed in, but no email was returned from Firebase user info.")
    st.stop()

# ================== FIREBASE CHAT HISTORY ==================
def encode_email(email: str) -> str:
    return email.replace(".", "_dot_").replace("@", "_at_")

def load_chat_history(email: str) -> List[Dict]:
    try:
        data = db.child("users").child(encode_email(email)).child("chats").get().val()
        return sorted(list(data.values()), key=lambda x: x.get("timestamp", 0)) if data else []
    except Exception as e:
        st.warning(f"⚠ Failed to load chat history: {e}")
        return []

def save_chat_message(email: str, role: str, content: str):
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
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            st.warning(f"Could not download {url}: {e}")

download_file(FAISS_INDEX_URL, os.path.join(LOCAL_FAISS_DIR, "index.faiss"))
download_file(FAISS_PKL_URL, os.path.join(LOCAL_FAISS_DIR, "index.pkl"))

# ================== VECTOR DB ==================
@st.cache_resource
def load_vector_db():
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    try:
        vectordb = FAISS.load_local(LOCAL_FAISS_DIR, embedder, allow_dangerous_deserialization=True)
        return vectordb.as_retriever(search_type="similarity", k=K_VAL)
    except Exception as e:
        st.warning(f"Failed to load FAISS vector store: {e}")
        return None

retriever = load_vector_db()
if retriever is None:
    st.warning("Vector DB not loaded — search context will be unavailable.")

# ================== OPENROUTER ==================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

def query_openrouter(messages: List[Dict[str, str]]) -> str:
    if not OPENROUTER_API_KEY:
        return "⚠ OPENROUTER_API_KEY not configured."
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
    doc_text = "No relevant context found."
    if retriever:
        try:
            docs = retriever.get_relevant_documents(user_question)
            doc_text = "\n".join([doc.page_content for doc in docs]) if docs else doc_text
        except Exception as e:
            st.warning(f"Failed retrieving context: {e}")
    return [
        {"role": "system", "content": (
            "You are ChemEng Buddy, a helpful tutor for Chemical Engineering. "
            "Explain clearly, step by step, with examples and common mistakes."
        )},
        {"role": "user", "content": f"Context:\n{doc_text}\n\nQuestion: {user_question}"}
    ]

# ================== MAIN APP ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(email)
if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

# Chat input
if user_query := st.chat_input("Ask me about Chemical Engineering"):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    save_chat_message(email, "user", user_query)

    with st.spinner("Thinking..."):
        prompt = build_prompt_with_context(user_query, st.session_state.chat_history)
        answer = query_openrouter(prompt)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    save_chat_message(email, "assistant", answer)
    st.session_state.last_answer_animated = True
    st.experimental_rerun()

# Display chat history
for i, chat in enumerate(st.session_state.chat_history):
    role = "user" if chat["role"] == "user" else "assistant"
    with st.chat_message(role):
        if (
            i == len(st.session_state.chat_history) - 1
            and chat["role"] == "assistant"
            and st.session_state.last_answer_animated
        ):
            type_like_chatgpt(chat["content"])
            st.session_state.last_answer_animated = False
        else:
            st.markdown(chat["content"])
