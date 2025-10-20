import streamlit as st
import time
import json
import requests
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import firebase_admin
from firebase_admin import credentials, db

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="⚗ ChemEng Buddy", layout="wide")

# -----------------------------------------------------------
# SECRETS AND GLOBAL SETTINGS
# -----------------------------------------------------------
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
FIREBASE_DATABASE_URL = st.secrets.get("firebase", {}).get("databaseURL", "")
EMBED_MODEL = st.secrets.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

LLM_MODELS = [
    "deepseek/deepseek-chat-v3.1:free",
    "mistralai/mistral-small-3.2-24b-instruct:free",
    "qwen/qwen3-4b:free"
]

# -----------------------------------------------------------
# FIREBASE SETUP
# -----------------------------------------------------------
def init_firebase():
    if not FIREBASE_DATABASE_URL:
        return False
    try:
        if not firebase_admin._apps:
            svc_json = st.secrets.get("SERVICE_ACCOUNT", {}).get("key")
            svc_dict = json.loads(svc_json)
            svc_dict["private_key"] = svc_dict["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(svc_dict)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        return True
    except Exception as e:
        st.error(f"Firebase init failed: {e}")
        return False

_FIRE_INIT = init_firebase()

def encode_email(email):
    return email.replace(".", "dot").replace("@", "at")

def save_message(email, role, content):
    if not _FIRE_INIT:
        return
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        ref.push({"role": role, "content": content, "timestamp": int(time.time())})
    except:
        pass

def load_messages(email):
    if not _FIRE_INIT:
        return []
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        data = ref.get()
        if not data:
            return []
        return sorted(list(data.values()), key=lambda x: x.get("timestamp", 0))
    except:
        return []

# -----------------------------------------------------------
# FAISS (Vector Store)
# -----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_retriever():
    try:
        path = Path("./faiss_store")
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = FAISS.load_local(str(path), embedder, allow_dangerous_deserialization=True)
        return vectordb.as_retriever(search_type="similarity", k=4)
    except Exception as e:
        st.warning(f"Could not load vector DB: {e}")
        return None

retriever = load_retriever()

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def query_openrouter(messages):
    for model in LLM_MODELS:
        try:
            payload = {"model": model, "messages": messages}
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
            data = resp.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
        except:
            continue
    return "⚠️ No response from any model."

def make_prompt(question):
    docs = []
    if retriever:
        try:
            docs = retriever.get_relevant_documents(question)
        except:
            pass
    context = "\n".join([d.page_content for d in docs]) if docs else "No context found."
    system = "You are ChemEng Buddy, a helpful Chemical Engineering tutor. Explain clearly and simply."
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# -----------------------------------------------------------
# STATE INITIALIZATION
# -----------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------------------------------------
# LOGIN PAGE
# -----------------------------------------------------------
def login_page():
    st.markdown(
        """
        <style>
        .center-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 85vh;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='center-box'>", unsafe_allow_html=True)
    st.title("⚗ ChemEng Buddy")
    st.subheader("Your friendly Chemical Engineering Study Partner")
    st.markdown("Please sign in to continue 👇")

    name = st.text_input("Full Name")
    email = st.text_input("Email Address")

    if st.button("Sign In 🔐", use_container_width=True):
        if name.strip() == "" or email.strip() == "":
            st.warning("Please fill in all fields.")
        else:
            st.session_state.user = {"name": name.strip(), "email": email.strip().lower()}
            st.session_state.chat = load_messages(email)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# CHAT PAGE
# -----------------------------------------------------------
def chat_page():
    user = st.session_state.user
    st.sidebar.header(f"👋 Hello, {user['name']}")
    if st.sidebar.button("Logout 🚪"):
        st.session_state.user = None
        st.session_state.chat = []
        st.rerun()

    st.title("💬 ChemEng Buddy Chat")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about Chemical Engineering ⚗️")
    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        save_message(user["email"], "user", user_input)

        with st.spinner("Thinking..."):
            messages = make_prompt(user_input)
            answer = query_openrouter(messages)

        st.session_state.chat.append({"role": "assistant", "content": answer})
        save_message(user["email"], "assistant", answer)
        st.rerun()

# -----------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------
if st.session_state.user:
    chat_page()
else:
    login_page()
