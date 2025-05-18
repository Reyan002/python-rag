import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from groq import Groq
from typing import Optional, List

# === Set Groq API Key ===
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# === Groq LLM Wrapper ===
class GroqLLM(LLM):
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = Groq()

    @property
    def _llm_type(self) -> str:
        return "groq-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=stop,
        )
        return response.choices[0].message.content

# === Streamlit App ===
st.set_page_config(page_title="📚 Folder PDF Chatbot", layout="wide")
st.title("📂 Auto-Load PDF Chatbot from Folder")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Hardcoded folder path
FOLDER_PATH = r"C:\Users\ART\Desktop\Content"

@st.cache_data(show_spinner=True)
def load_and_process_folder(folder_path: str):
    all_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(all_docs)

if os.path.exists(FOLDER_PATH):
    with st.spinner("🔄 Processing folder... please wait..."):
        start = time.time()
        documents = load_and_process_folder(FOLDER_PATH)
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
        )
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()
        llm = GroqLLM()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        st.session_state.qa = qa
        st.success(f"✅ Loaded {len(documents)} chunks from folder in {round(time.time()-start, 2)}s")

    # Chat input
    user_input = st.chat_input("Ask something about your documents...")
    if user_input:
        try:
            answer = st.session_state.qa.run(user_input)
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("bot", answer))
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.error("🚫 Folder path not found. Make sure the folder exists on this machine.")

# Display chat history
for role, message in st.session_state.get("chat_history", []):
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)
