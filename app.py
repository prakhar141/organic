import os
import json
import requests
import streamlit as st
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

# ================== CONFIG ==================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-oss-120b:free"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)

# Hugging Face URLs for the prebuilt vector store (swap with Chem Eng dataset)
FAISS_INDEX_URL = "https://huggingface.co/datasets/prakhar146/medical/resolve/main/index.faiss"
FAISS_PKL_URL = "https://huggingface.co/datasets/prakhar146/medical/resolve/main/index.pkl"

# Local directory to store downloaded files
LOCAL_FAISS_DIR = "./faiss_store"
os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)

# ================== STREAMLIT PAGE SETUP ==================
st.set_page_config(page_title="ChemEng Buddy", layout="wide")
st.title("⚗️ ChemEng Buddy")
st.markdown("Your friendly Chemical Engineering study partner")

def type_like_chatgpt(text, speed=0.004):
    placeholder = st.empty()
    animated = ""
    for c in text:
        animated += c
        placeholder.markdown(animated + "|")
        time.sleep(speed)
    placeholder.markdown(animated)

# ================== HELPER: Download Files ==================
def download_file(url: str, local_path: str):
    if not os.path.exists(local_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

# Download FAISS index files
download_file(FAISS_INDEX_URL, os.path.join(LOCAL_FAISS_DIR, "index.faiss"))
download_file(FAISS_PKL_URL, os.path.join(LOCAL_FAISS_DIR, "index.pkl"))

# ================== VECTOR DB LOADING ==================
@st.cache_resource
def load_vector_db():
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.load_local(
        LOCAL_FAISS_DIR,
        embedder,
        allow_dangerous_deserialization=True  # ⚠️ only for trusted .pkl file
    )
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

retriever = load_vector_db()

# ================== OPENROUTER HELPER ==================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

def query_openrouter(model: str, messages: List[Dict[str, str]]) -> str:
    payload = {"model": model, "messages": messages}
    r = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"]
    return json.dumps(data)

# ================== VANILLA RAG PIPELINE ==================
def vanilla_rag_answer(question: str) -> str:
    try:
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."
        
        prompt = [
            {"role": "system", "content": (
                "You are ChemEng Buddy, a tutor for undergraduate and graduate chemical engineering students. "
                "Break down complex concepts in areas like fluid mechanics, heat and mass transfer, thermodynamics, "
                "chemical reaction engineering, process control, and plant design.\n\n"
                "Explain step by step, starting from fundamentals to deeper insights, with examples and common mistakes. "
                "Gamify the experience: include quick quizzes, problem-solving challenges, and badges for progress.\n\n"
                "Offer 'exam hacks' and 'industrial insights' where appropriate (safe, accurate, and ethical). "
                "Always answer in English. Restrict yourself to chemical engineering and closely related topics."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]

        return query_openrouter(MODEL_NAME, prompt)

    except Exception as e:
        return f"⚠️ Servers Are Busy Try after sometime"

# ================== CHAT INTERFACE ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_answer_animated" not in st.session_state:
    st.session_state.last_answer_animated = False

if user_query := st.chat_input("Ask me about Chemical Engineering"):
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("Thinking..."):
        answer = vanilla_rag_answer(user_query)
    
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.session_state.last_answer_animated = True
    st.rerun()

# Show chat history
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

st.markdown("""<hr style="margin-top: 40px;">
<div style='text-align: center; color: #888; font-size: 14px;'>
    Built with ⚗️ by <b>Prakhar Mathur</b> · BITS Pilani · 
    <br>📬 Email: <a href="mailto:prakhar.mathur2020@gmail.com">prakhar.mathur2020@gmail.com</a>
</div>
""", unsafe_allow_html=True)
