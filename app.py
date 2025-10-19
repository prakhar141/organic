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
        service_account_json = st.secrets["SERVICE_ACCOUNT"]["key"]
        service_account_dict = json.loads(service_account_json)
        service_account_dict["private_key"] = service_account_dict["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(service_account_dict)
        firebase_admin.initialize_app(cred, {
            "databaseURL": st.secrets["firebase"]["databaseURL"]
        })
        st.success("✅ Firebase initialized!")
    except Exception as e:
        st.error(f"⚠ Firebase init failed: {e}")

# ================== FIREBASE CHAT HISTORY ==================
def encode_email(email: str) -> str:
    return email.replace(".", "_dot_").replace("@", "_at_")

def load_chat_history(email: str) -> List[Dict]:
    try:
        ref = db.reference(f"users/{encode_email(email)}/chats")
        data = ref.get()
        if data:
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

# ===== REPLACEMENT: Robust Google OAuth block =====
import secrets
import urllib.parse

CLIENT_ID = st.secrets["GOOGLE_ID"]
CLIENT_SECRET = st.secrets["GOOGLE_KEY"]
REDIRECT_URI = st.secrets["redirect_url"]

# init session keys
for key in ["auth_user","access_token","refresh_token","chat_history","last_answer_animated","auth_code_exchanged","oauth_state"]:
    if key not in st.session_state:
        if key == "chat_history":
            st.session_state[key] = []
        elif key == "auth_code_exchanged":
            st.session_state[key] = False
        else:
            st.session_state[key] = None

# generate state if missing
if not st.session_state.oauth_state:
    st.session_state.oauth_state = secrets.token_urlsafe(16)

auth_params = {
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "response_type": "code",
    "scope": "openid email profile",
    "access_type": "offline",
    "prompt": "consent",
    "state": st.session_state.oauth_state
}
google_auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(auth_params)

# parse redirect query parameters
code_list = st.query_params.get("code")
state_list = st.query_params.get("state")
error_list = st.query_params.get("error")

# show errors from redirect
if error_list:
    st.error(f"Google OAuth returned error: {error_list}")

# only attempt exchange if we have a code, the state matches, user not already set, and not exchanged before
elif code_list and state_list and (state_list[0] == st.session_state.oauth_state) and (not st.session_state.auth_user) and (not st.session_state.auth_code_exchanged):
    st.session_state.auth_code_exchanged = True  # mark immediately so reload doesn't reuse
    code = code_list[0]

    # Extra debug: show length/first few chars to detect truncation
    st.debug = lambda *a, **k: None  # no-op if st.debug not available; replace with print if needed
    try:
        # URL-encode payload
        token_payload = urllib.parse.urlencode({
            "code": code,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code"
        })

        token_resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data=token_payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15
        )

        if token_resp.status_code != 200:
            # Very explicit error message for debugging
            st.error(f"Google OAuth token request failed: {token_resp.status_code}\nResponse: {token_resp.text}")
            # Reset flag so user can retry the flow
            st.session_state.auth_code_exchanged = False
        else:
            tokens = token_resp.json()
            access_token = tokens.get("access_token")
            refresh_token = tokens.get("refresh_token")
            if not access_token:
                st.error(f"No access_token in response. Full response:\n{tokens}")
                st.session_state.auth_code_exchanged = False
            else:
                # get userinfo
                user_info_resp = requests.get(
                    "https://www.googleapis.com/oauth2/v3/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10
                )
                if user_info_resp.status_code != 200:
                    st.error(f"Failed to fetch userinfo: {user_info_resp.status_code}\n{user_info_resp.text}")
                    st.session_state.auth_code_exchanged = False
                else:
                    user_info = user_info_resp.json()
                    st.session_state.access_token = access_token
                    st.session_state.refresh_token = refresh_token
                    st.session_state.auth_user = {"email": user_info.get("email"), "name": user_info.get("name")}
                    # clear query params to avoid accidental reuse
                    st.experimental_set_query_params()
                    st.session_state.chat_history = load_chat_history(st.session_state.auth_user["email"])
                    st.rerun()

    except requests.exceptions.RequestException as e:
        st.error(f"Network error during OAuth: {e}")
        st.session_state.auth_code_exchanged = False
    except Exception as e:
        st.error(f"Unexpected OAuth error: {e}")
        st.session_state.auth_code_exchanged = False

elif code_list and state_list and state_list[0] != st.session_state.oauth_state:
    # state mismatch — likely a session / cross-tab problem
    st.error("State mismatch detected. Please click Sign in again (use the same tab).")

# Sidebar UI: same-tab link (no target="_blank")
with st.sidebar:
    if st.session_state.auth_user:
        st.success(f"✅ Logged in as {st.session_state.auth_user['email']}")
        if st.button("Logout"):
            st.session_state.auth_user = None
            st.session_state.access_token = None
            st.session_state.refresh_token = None
            st.session_state.chat_history = []
            st.session_state.auth_code_exchanged = False
            st.session_state.oauth_state = None
            st.rerun()
    else:
        st.markdown("🔐 **Sign in with Google**")
        # IMPORTANT: open in same tab (no target attribute)
        st.markdown(f'<a href="{google_auth_url}">👉 Sign in with Google</a>', unsafe_allow_html=True)
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
if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

# Chat Input
if user_query := st.chat_input("Ask me anything about Chemical Engineering ⚗"):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    save_chat_message(email, "user", user_query)

    with st.spinner("Thinking..."):
        prompt = build_prompt_with_context(user_query, st.session_state.chat_history)
        answer = query_openrouter(prompt)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    save_chat_message(email, "assistant", answer)
    st.session_state.last_answer_animated = True
    st.experimental_rerun()

# Display chat
for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        if i == len(st.session_state.chat_history) - 1 and chat["role"] == "assistant" and st.session_state.last_answer_animated:
            type_like_chatgpt(chat["content"])
            st.session_state.last_answer_animated = False
        else:
            st.markdown(chat["content"])
