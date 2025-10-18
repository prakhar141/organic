import os
import time
import json
import requests
import streamlit as st
from urllib.parse import urlencode
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import firebase_admin
from firebase_admin import credentials, db

# ================== CONFIG ==================
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = 4

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
st.set_page_config(page_title="⚗ ChemEng Buddy", layout="wide")
st.title("⚗ ChemEng Buddy")
st.markdown("Your friendly Chemical Engineering study partner 🧪")

# ================== FIREBASE ADMIN ==================
if not firebase_admin._apps:
    try:
        # Load the service account JSON from the [SERVICE_ACCOUNT] table in secrets.toml
        service_account_dict = json.loads(st.secrets["SERVICE_ACCOUNT"]["key"])
        cred = credentials.Certificate(service_account_dict)
        firebase_admin.initialize_app(cred, {
            "databaseURL": st.secrets["firebase"]["databaseURL"]
        })
    except Exception as e:
        st.error(f"⚠ Firebase init failed: {e}")

# ================== GOOGLE OAUTH SETUP ==================
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

query_params = st.query_params.to_dict()
access_token = query_params.get("access_token")

if access_token and not st.session_state.auth_user:
    try:
        user_info = requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        ).json()

        st.session_state.auth_user = {
            "email": user_info.get("email"),
            "name": user_info.get("name")
        }

        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"⚠ Failed to fetch Google user info: {e}")

# Sidebar login/logout
with st.sidebar:
    if st.session_state.auth_user:
        st.success(f"✅ Logged in as {st.session_state.auth_user['email']}")
        if st.button("Logout"):
            st.session_state.auth_user = None
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown("🔐 **Sign in with Google**")
        client_id = st.secrets["GOOGLE_CLIENT_ID"]
        redirect_url = st.secrets["redirect_url"]

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_url,
            "response_type": "token",
            "scope": "email profile openid"
        }
        google_auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
        st.markdown(f"[👉 Sign in with Google]({google_auth_url})", unsafe_allow_html=True)

# Stop if not logged in
if not st.session_state.auth_user:
    st.stop()

email = st.session_state.auth_user["email"]

# ================== FIREBASE CHAT HISTORY ==================
def encode_email(email: str) -> str:
    """Encode email safely for Firebase keys."""
    return email.replace(".", "_dot_").replace("@", "_at_")

def load_chat_history(email: str) -> List[Dict]:
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        data = ref.get()
        if data:
            # Sort messages by timestamp
            return sorted(data.values(), key=lambda x: x.get("timestamp", 0))
        return []
    except Exception as e:
        st.warning(f"⚠ Failed to load chat history: {e}")
        return []

def save_chat_message(email: str, role: str, content: str):
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        ref.push({
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

# ================== VECTOR DATABASE ==================
@st.cache_resource
def load_vector_db():
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.load_local(LOCAL_FAISS_DIR, embedder, allow_dangerous_deserialization=True)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

retriever = load_vector_db()

# ================== OPENROUTER API ==================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

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
    return "⚠ All LLMs failed to respond. Try again later."

# ================== CONTEXT BUILDER ==================
def build_prompt_with_context(user_question: str, chat_history: List[Dict]):
    docs = retriever.get_relevant_documents(user_question)
    doc_text = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."
    return [
        {"role": "system", "content": (
            "You are ChemEng Buddy, a helpful Chemical Engineering tutor. "
            "Explain concepts clearly, step by step, and give examples where possible."
        )},
        {"role": "user", "content": f"Context:\n{doc_text}\n\nQuestion: {user_question}"}
    ]

# ================== MAIN APP ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(email)

if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

# Chat Input Box
if user_query := st.chat_input("Ask me anything about Chemical Engineering ⚗"):
    # Append user message locally & save to Firebase
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    save_chat_message(email, "user", user_query)

    # Get answer
    with st.spinner("Thinking..."):
        prompt = build_prompt_with_context(user_query, st.session_state.chat_history)
        answer = query_openrouter(prompt)

    # Append assistant message locally & save to Firebase
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    save_chat_message(email, "assistant", answer)
    st.session_state.last_answer_animated = True
    st.rerun()

# Display all previous chats
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
