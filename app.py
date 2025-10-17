import os
import time
import requests
import streamlit as st
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pyrebase4 as pyrebase

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

# ================== FIREBASE CLIENT ==================
firebase_config = st.secrets["firebase"]  # Your Firebase Web config
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

# ================== GOOGLE SIGN-IN ==================
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

with st.sidebar:
    if st.session_state.auth_user:
        st.success(f"Signed in as {st.session_state.auth_user['email']}")
        if st.button("Logout"):
            st.session_state.auth_user = None
            st.session_state.chat_history = []
            st.experimental_rerun()
    else:
        st.markdown("🔐 Sign in with Google")
        if st.button("Sign in with Google"):
            # Display Google OAuth link
            st.info("Copy this link and authenticate with Google in your browser:")
            st.write("https://YOUR_FIREBASE_PROJECT.firebaseapp.com/__/auth/handler")  # Firebase hosting OAuth handler

# Stop app if not logged in
if not st.session_state.auth_user:
    st.stop()

email = st.session_state.auth_user["email"]

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
