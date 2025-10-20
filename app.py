# app.py
import streamlit as st
import time
import json
import requests
from pathlib import Path
import traceback
import os

# -----------------------------------------------------------
# Optional / heavy deps (guarded)
# -----------------------------------------------------------
HAVE_FIREBASE = False
HAVE_FAISS = False
HAVE_LANGCHAIN = False

try:
    import firebase_admin
    from firebase_admin import credentials, db
    HAVE_FIREBASE = True
except Exception:
    HAVE_FIREBASE = False

# Try to import langchain community modules only if available
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAVE_FAISS = True
    HAVE_LANGCHAIN = True
except Exception:
    HAVE_FAISS = False
    HAVE_LANGCHAIN = False

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="⚗ ChemEng Buddy", layout="wide")

# -----------------------------------------------------------
# SECRETS AND GLOBAL SETTINGS
# -----------------------------------------------------------
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "") if st.secrets else ""
FIREBASE_DATABASE_URL = (st.secrets.get("firebase", {}) or {}).get("databaseURL", "") if st.secrets else ""
EMBED_MODEL = st.secrets.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2") if st.secrets else "sentence-transformers/all-MiniLM-L6-v2"

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
# SIMPLE LOCAL CHAT BACKUP (if Firebase not available)
# -----------------------------------------------------------
LOCAL_STORE = Path.home() / ".chemeng_buddy_chats.json"
if not LOCAL_STORE.exists():
    try:
        LOCAL_STORE.write_text(json.dumps({}))
    except Exception:
        pass

# -----------------------------------------------------------
# FIREBASE SETUP (optional)
# -----------------------------------------------------------
def init_firebase():
    if not HAVE_FIREBASE:
        return False, "firebase package not installed"
    if not FIREBASE_DATABASE_URL:
        return False, "no databaseURL in secrets"
    try:
        if not firebase_admin._apps:
            svc_json = (st.secrets.get("SERVICE_ACCOUNT", {}) or {}).get("key", "")
            if not svc_json:
                return False, "no SERVICE_ACCOUNT key in secrets"
            svc_dict = json.loads(svc_json)
            # Fix newline escapes in private key
            if "private_key" in svc_dict:
                svc_dict["private_key"] = svc_dict["private_key"].replace("\\n", "\n")
            cred = credentials.Certificate(svc_dict)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        return True, "initialized"
    except Exception as e:
        return False, f"Firebase init failed: {e}"

_FIRE_INIT, FIRE_INIT_MSG = init_firebase()

def encode_email(email):
    return email.replace(".", "dot").replace("@", "at")

def save_message_firebase(email, role, content):
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        ref.push({"role": role, "content": content, "timestamp": int(time.time())})
    except Exception:
        # don't raise; fallback to local
        raise

def load_messages_firebase(email):
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        data = ref.get()
        if not data:
            return []
        return sorted(list(data.values()), key=lambda x: x.get("timestamp", 0))
    except Exception:
        raise

def save_message_local(email, role, content):
    try:
        data = json.loads(LOCAL_STORE.read_text() or "{}")
        key = encode_email(email)
        data.setdefault(key, []).append({"role": role, "content": content, "timestamp": int(time.time())})
        LOCAL_STORE.write_text(json.dumps(data))
    except Exception:
        pass

def load_messages_local(email):
    try:
        data = json.loads(LOCAL_STORE.read_text() or "{}")
        return sorted(data.get(encode_email(email), []), key=lambda x: x.get("timestamp", 0))
    except Exception:
        return []

def save_message(email, role, content):
    if _FIRE_INIT:
        try:
            save_message_firebase(email, role, content)
            return
        except Exception:
            # fallback to local
            save_message_local(email, role, content)
    else:
        save_message_local(email, role, content)

def load_messages(email):
    if _FIRE_INIT:
        try:
            return load_messages_firebase(email)
        except Exception:
            return load_messages_local(email)
    else:
        return load_messages_local(email)

# -----------------------------------------------------------
# FAISS / Retriever (optional)
# -----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_retriever():
    if not HAVE_FAISS or not HAVE_LANGCHAIN:
        return None, "faiss/langchain not installed"
    try:
        path = Path("./faiss_store")
        if not path.exists():
            return None, "faiss_store path not found"
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = FAISS.load_local(str(path), embedder, allow_dangerous_deserialization=True)
        retriever = vectordb.as_retriever(search_type="similarity", k=4)
        return retriever, "loaded"
    except Exception as e:
        return None, f"Could not load vector DB: {e}"

retriever, RETRIEVER_MSG = load_retriever()

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def query_openrouter(messages):
    # Try multiple models (best-effort). Return a text message or error info.
    for model in LLM_MODELS:
        try:
            payload = {"model": model, "messages": messages}
            resp = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
            data = resp.json()
            # Defensive parsing
            if isinstance(data, dict) and "choices" in data and len(data["choices"]) > 0:
                msg = data["choices"][0].get("message", {}).get("content") or data["choices"][0].get("text")
                if msg:
                    return msg
        except Exception:
            continue
    return "⚠️ No response from any model (or OpenRouter key missing/failed)."

def make_prompt(question):
    docs = []
    if retriever:
        try:
            docs = retriever.get_relevant_documents(question)
        except Exception:
            docs = []
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
# UI - LOGIN
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
            height: 80vh;
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

    with st.form("login_form", clear_on_submit=False):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        submitted = st.form_submit_button("Sign In 🔐")
        if submitted:
            if name.strip() == "" or email.strip() == "":
                st.warning("Please fill in all fields.")
            else:
                st.session_state.user = {"name": name.strip(), "email": email.strip().lower()}
                st.session_state.chat = load_messages(email.strip().lower())
                st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# UI - CHAT
# -----------------------------------------------------------
def chat_page():
    user = st.session_state.user
    # Sidebar with status
    st.sidebar.header(f"👋 Hello, {user['name']}")
    if st.sidebar.button("Logout 🚪"):
        st.session_state.user = None
        st.session_state.chat = []
        st.experimental_rerun()

    st.sidebar.markdown("### System status")
    st.sidebar.text(f"Firebase: {'enabled' if _FIRE_INIT else 'disabled'} ({FIRE_INIT_MSG})")
    st.sidebar.text(f"Vector DB: {'loaded' if retriever else 'not loaded'} ({RETRIEVER_MSG})")
    st.sidebar.text(f"OpenRouter key: {'present' if OPENROUTER_API_KEY else 'missing'}")

    st.title("💬 ChemEng Buddy Chat")

    # Display chat messages
    for msg in st.session_state.chat:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        try:
            with st.chat_message(role):
                st.markdown(content)
        except Exception:
            # fallback to simple write
            st.write(f"{role.upper()}: {content}")

    # Chat input - try to use st.chat_input if available, otherwise fallback
    user_input = None
    if hasattr(st, "chat_input"):
        user_input = st.chat_input("Ask something about Chemical Engineering ⚗️")
    else:
        # Fallback entry with button
        with st.form("ask_form", clear_on_submit=True):
            user_input = st.text_input("Ask something about Chemical Engineering ⚗️")
            send = st.form_submit_button("Send")
            if not send:
                user_input = None

    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        save_message(user["email"], "user", user_input)

        with st.spinner("Thinking..."):
            try:
                messages = make_prompt(user_input)
                answer = query_openrouter(messages)
            except Exception as e:
                answer = f"⚠️ Error when calling model: {e}\n\nTrace:\n{traceback.format_exc()}"

        st.session_state.chat.append({"role": "assistant", "content": answer})
        save_message(user["email"], "assistant", answer)
        # Instead of st.rerun (which can be jarring), just update the page
        st.experimental_rerun()

# -----------------------------------------------------------
# MAIN (wrapped to catch unexpected errors)
# -----------------------------------------------------------
def main():
    try:
        if st.session_state.user:
            chat_page()
        else:
            login_page()
    except Exception as e:
        st.error("An unexpected error occurred — details below.")
        st.exception(e)
        st.write("Traceback (for debugging):")
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
