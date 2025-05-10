import os
import platform
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from groq import Groq
from typing import Optional, List


# === Set your Groq API Key ===
os.environ["GROQ_API_KEY"] = "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"  # Replace with your own API key


# === Warn if Python 3.13 (some packages may break) ===
if platform.python_version().startswith("3.13"):
    print("⚠️ Python 3.13 is still experimental for many AI packages. Prefer Python 3.10 or 3.11.")


# === Custom GroqLLM that works with LangChain ===
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


# === Ask user for PDF path ===
pdf_path = input("📄 Enter path to your PDF file: ").strip()
if not os.path.exists(pdf_path):
    print("❌ PDF not found. Exiting.")
    exit()

print("🔄 Loading and processing the PDF...")

# === Load and split the PDF ===
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# === Create embeddings ===
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # smaller, faster, compatible
    model_kwargs={"device": "cpu"}  # use "cuda" if you have GPU
)

# === Build FAISS vector store ===
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# === Initialize Groq + RetrievalQA Chain ===
llm = GroqLLM()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# === Interactive Q&A Loop ===
print("\n✅ Ready! Ask questions about your PDF. Type 'exit' to quit.\n")

while True:
    question = input("❓ Your question: ").strip()
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    try:
        answer = qa.run(question)
        print(f"✅ Answer:\n{answer}\n")
    except Exception as e:
        print(f"❌ Error: {e}")