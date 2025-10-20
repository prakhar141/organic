# streamlit_app.py
import os
import time
import json
import requests
import streamlit as st
import secrets as pysecrets
import urllib.parse
from typing import List, Dict
from pathlib import Path

# LangChain / FAISS imports (same as you used)
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

# OAuth / Google credentials from secrets
CLIENT_ID = st.secrets.get("GOOGLE_ID", "")
CLIENT_SECRET = st.secrets.get("GOOGLE_KEY", "")
REDIRECT_URI = st.secrets.get("redirect_url", "")  # must be registered in Google Cloud Console

# Firebase service account (stored in st.secrets as JSON string under SERVICE_ACCOUNT.key)
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

_FIRED_INITIALIZED = init_firebase()

def load_chat_history_from_firebase(email: str) -> List[Dict]:
    if not _FIRED_INITIALIZED:
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
    if not _FIRED_INITIALIZED:
        st.warning("Firebase not initialized; skipping save.")
        return
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        ref.push({"role": role, "content": content, "timestamp": now_ts()})
    except Exception as e:
        st.warning(f"⚠ Failed to save message: {e}")

# ================== OAUTH (Google same-tab) ==================

# Setup session defaults
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None
if "oauth_state" not in st.session_state or st.session_state.oauth_state is None:
    st.session_state.oauth_state = pysecrets.token_urlsafe(16)
if "auth_code_exchanged" not in st.session_state:
    st.session_state.auth_code_exchanged = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

def google_auth_url():
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
        "state": st.session_state.oauth_state
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)

def exchange_code_for_tokens(code: str) -> Dict:
    payload = {
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post("https://oauth2.googleapis.com/token", data=urllib.parse.urlencode(payload), headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()

# When the app loads, check query params for OAuth `code`
query_code = st.experimental_get_query_params().get("code")
query_state = st.experimental_get_query_params().get("state")
query_error = st.experimental_get_query_params().get("error")

if query_error:
    st.error(f"Google OAuth returned error: {query_error}")

# Only attempt to exchange once per code and when state matches
if query_code and query_state and (query_state[0] == st.session_state.oauth_state) and (not st.session_state.auth_user) and (not st.session_state.auth_code_exchanged):
    st.session_state.auth_code_exchanged = True
    code = query_code[0]
    try:
        tokens = exchange_code_for_tokens(code)
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        if not access_token:
            st.error("No access token returned from Google.")
            st.session_state.auth_code_exchanged = False
        else:
            # fetch userinfo
            user_info_resp = requests.get("https://www.googleapis.com/oauth2/v3/userinfo", headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
            user_info_resp.raise_for_status()
            user_info = user_info_resp.json()
            st.session_state.access_token = access_token
            st.session_state.refresh_token = refresh_token
            st.session_state.auth_user = {"email": user_info.get("email"), "name": user_info.get("name")}
            # load chat history from firebase
            if st.session_state.auth_user and st.session_state.auth_user.get("email"):
                st.session_state.chat_history = load_chat_history_from_firebase(st.session_state.auth_user["email"])
            # clear query params to avoid reusing code
            st.experimental_set_query_params()
            # we can rerun to show logged-in UI
            st.experimental_rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Network error during OAuth token exchange: {e}")
        st.session_state.auth_code_exchanged = False
    except Exception as e:
        st.error(f"Unexpected error in OAuth: {e}")
        st.session_state.auth_code_exchanged = False
elif query_code and query_state and query_state[0] != st.session_state.oauth_state:
    # state mismatch — warn the user
    st.error("State mismatch detected. Please click Sign in again in the same tab.")
    if st.button("Restart sign-in (reset state)"):
        st.session_state.oauth_state = pysecrets.token_urlsafe(16)
        st.session_state.auth_code_exchanged = False
        st.experimental_rerun()

# ================== SIDEBAR (login/logout) ==================
with st.sidebar:
    st.header("Account")
    if st.session_state.auth_user:
        st.success(f"✅ Logged in as {st.session_state.auth_user['email']}")
        if st.button("Logout"):
            st.session_state.auth_user = None
            st.session_state.access_token = None
            st.session_state.refresh_token = None
            st.session_state.chat_history = []
            st.session_state.auth_code_exchanged = False
            st.session_state.oauth_state = pysecrets.token_urlsafe(16)
            st.experimental_rerun()
    else:
        st.markdown("🔐 Sign in with Google")
        url = google_auth_url()
        st.markdown(
            f'<a href="{url}" target="_self">👉 Sign in with Google (same tab)</a>\n\n⚠️ Please do not open in a new tab or reload the page during sign-in.',
            unsafe_allow_html=True,
        )

# ================== UI: Title + Description ==================
st.title("⚗ ChemEng Buddy")
st.markdown("Your friendly Chemical Engineering study partner. Log in with Google to load and save your chats to your account.")

# ================== ANIMATION HELPERS (non-blocking, simple) ==================
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
                # Some OpenRouter responses nest content differently; handle leniently
                choice = data["choices"][0]
                # prefer message.content
                if isinstance(choice.get("message"), dict):
                    return choice["message"].get("content", "")
                else:
                    return choice.get("text", "")
        except Exception:
            # Try next model silently
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
# Keep the main column for conversation
main_col, right_col = st.columns([3, 1])

with main_col:
    # If logged in, show previous chats
    if st.session_state.auth_user:
        st.subheader(f"Hello, {st.session_state.auth_user.get('name') or st.session_state.auth_user.get('email')} 👋")
        st.markdown("---")
        # Render existing chat history from session_state
        if not st.session_state.chat_history:
            st.info("No previous chats found — start a new conversation below!")
        for i, chat in enumerate(st.session_state.chat_history):
            role = chat.get("role", "user")
            content = chat.get("content", "")
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
    else:
        st.info("Please sign in with Google (sidebar) to load your saved chats and start chatting.")

    st.markdown("---")
    # Chat input — only enabled when logged in
    if st.session_state.auth_user:
        user_query = st.chat_input("Ask me anything about Chemical Engineering ⚗")
        if user_query:
            # update chat history immediately in UI & firebase
            st.session_state.chat_history.append({"role": "user", "content": user_query, "timestamp": now_ts()})
            save_chat_message_to_firebase(st.session_state.auth_user["email"], "user", user_query)

            # Build prompt & query LLM
            with st.spinner("Thinking..."):
                prompt = build_prompt_with_context(user_query, st.session_state.chat_history)
                answer = query_openrouter(prompt)

            # Save assistant message
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "timestamp": now_ts()})
            save_chat_message_to_firebase(st.session_state.auth_user["email"], "assistant", answer)

            # Rerun so messages render with chat_message components in order
            st.experimental_rerun()
    else:
        st.write("Sign in to start chatting and have your chats saved to your account.")

# Right column: helper widgets
with right_col:
    st.header("Controls & Info")
    if retriever:
        st.success("Retriever: loaded")
        st.caption(f"Embedding model: {EMBED_MODEL}")
    else:
        st.warning("Retriever not available")
    st.write("Debug / quick actions:")
    if st.button("Reload vector DB"):
        # clear cache by rerunning (force new cache by changing param unlikely)
        load_vector_db.clear()
        st.experimental_rerun()
    if st.button("Refresh chat history from server (Firebase)"):
        if st.session_state.auth_user and st.session_state.auth_user.get("email"):
            st.session_state.chat_history = load_chat_history_from_firebase(st.session_state.auth_user["email"])
            st.experimental_rerun()
        else:
            st.warning("You must be signed in to refresh your chat history.")

# Footer / instructions
st.markdown("---")
st.markdown("**Notes:**\n- Keep your Streamlit `secrets` updated with required keys.\n- The Google OAuth redirect URL must match the one registered in your Google Cloud Console.\n- Firebase saves chats under `users/<encoded-email>/chats` in Realtime Database.")
