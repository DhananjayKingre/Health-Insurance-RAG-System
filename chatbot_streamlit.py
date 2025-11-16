import streamlit as st
import os
import re
from pathlib import Path
from typing import List, Tuple
import pickle
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import time
from dotenv import load_dotenv 

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Basic configuration values used in the application
PDF_PATH = r"D:\onelab\Health-Insurance-Policy.pdf"
VECTOR_DB_PATH = r"D:\onelab\vectorData"
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Streamlit session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if "show_ready_msg" not in st.session_state:
    st.session_state.show_ready_msg = False
if "ready_msg_time" not in st.session_state:
    st.session_state.ready_msg_time = 0


# PDF PROCESSOR
class PDFProcessor:
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[PAGE {page_num + 1}]\n{page_text}\n"
        except:
            pass
        return text

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Tuple[str, int]]:
        chunks_with_pages = []
        pages = text.split('[PAGE ')

        for page_section in pages[1:]:
            page_match = re.match(r'(\d+)\](.*)', page_section, re.DOTALL)
            if page_match:
                page_num = int(page_match.group(1))
                page_text = page_match.group(2).strip()

                sentences = re.split(r'(?<=[.!?])\s+', page_text)

                current_chunk = []
                current_size = 0

                for sentence in sentences:
                    sentence_words = len(sentence.split())

                    if current_size + sentence_words > chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks_with_pages.append((chunk_text, page_num))
                        overlap_text = ' '.join(current_chunk[-3:])
                        current_chunk = [overlap_text]
                        current_size = len(overlap_text.split())

                    current_chunk.append(sentence)
                    current_size += sentence_words

                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks_with_pages.append((chunk_text, page_num))

        return chunks_with_pages


# VECTOR DATABASE - FIXED CLASS DEFINITION
class VectorDatabase:
    def __init__(self, db_path: str, model_name: str = MODEL_NAME):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.db_path / "faiss_index.bin"
        self.chunks_file = self.db_path / "chunks.pkl"

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        self.index = None
        self.chunks_with_pages = []

    def create_and_save(self, chunks_with_pages: List[Tuple[str, int]]):
        self.chunks_with_pages = chunks_with_pages
        chunks = [chunk for chunk, _ in chunks_with_pages]

        progress_bar = st.progress(0)
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings)
            progress_bar.progress(min((i + batch_size) / len(chunks), 1.0))

        embeddings_array = np.array(all_embeddings).astype('float32')

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_array)

        faiss.write_index(self.index, str(self.index_file))

        with open(self.chunks_file, 'wb') as f:
            pickle.dump(self.chunks_with_pages, f)

    def load(self) -> bool:
        try:
            if not self.index_file.exists() or not self.chunks_file.exists():
                return False

            self.index = faiss.read_index(str(self.index_file))

            with open(self.chunks_file, 'rb') as f:
                self.chunks_with_pages = pickle.load(f)

            return True
        except:
            return False

    def search(self, query: str, k: int = 5):
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks_with_pages):
                chunk, page = self.chunks_with_pages[idx]
                results.append((chunk, page, float(dist)))

        return results


# RAG SYSTEM
class HealthInsuranceRAG:
    def __init__(self, vector_db: VectorDatabase, api_key: str):
        self.vector_db = vector_db
        self.client = Groq(api_key=api_key)
        self.model = GROQ_MODEL

    def answer_question(self, query: str, k: int = 5):
        results = self.vector_db.search(query, k=k)

        if not results:
            return {'answer': "No relevant information found.", 'sources': []}

        context_parts = []
        for i, (chunk, page, score) in enumerate(results, 1):
            context_parts.append(f"[Source {i} - Page {page}]\n{chunk}\n")

        context = "\n".join(context_parts)

        prompt = f"""
You are a helpful assistant.

CONTEXT:
{context}

QUESTION:
{query}

Answer clearly with citations.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )

        answer = response.choices[0].message.content
        return {'answer': answer, 'sources': results}


# SYSTEM INIT
def initialize_system():
    if st.session_state.initialized:
        return

    vector_db = VectorDatabase(VECTOR_DB_PATH)

    if vector_db.load():
        st.session_state.vector_db = vector_db
        st.session_state.initialized = True
        st.session_state.show_ready_msg = True
        st.session_state.ready_msg_time = time.time()
        return

    if not os.path.exists(PDF_PATH):
        st.error(" PDF file not found")
        return

    text = PDFProcessor.extract_text(PDF_PATH)
    chunks_with_pages = PDFProcessor.chunk_text(text)
    vector_db.create_and_save(chunks_with_pages)

    st.session_state.vector_db = vector_db
    st.session_state.initialized = True
    st.session_state.show_ready_msg = True
    st.session_state.ready_msg_time = time.time()


# MAIN UI
def main():
    st.set_page_config(page_title="Health Insurance RAG System", layout="wide")

    st.title("🏥 Health-Insurance RAG System")

    st.markdown("""
<div style="padding:15px; background:#f0f2f6; border-radius:10px; margin-bottom:20px;">
<b>📚 Example Questions</b><br>
• What illnesses are excluded from OPD coverage?<br>
• Is maternity covered? What is the waiting period?<br>
• What is the pre-existing condition waiting period?<br>
• What happens if I miss a premium payment?<br>
• What is covered under day-care procedures?<br>
• What are the permanent exclusions?<br>
</div>
""", unsafe_allow_html=True)

    initialize_system()

    if st.session_state.show_ready_msg:
        if time.time() - st.session_state.ready_msg_time < 2:
            st.success("🤖 Chatbot is ready! Ask your question…")
        else:
            st.session_state.show_ready_msg = False

    with st.sidebar:
        st.header("⚙️ Settings")

        if st.button("Rebuild Database", use_container_width=True):
            db_path = Path(VECTOR_DB_PATH)
            if (db_path / "faiss_index.bin").exists():
                (db_path / "faiss_index.bin").unlink()
            if (db_path / "chunks.pkl").exists():
                (db_path / "chunks.pkl").unlink()
            st.session_state.initialized = False
            st.session_state.vector_db = None
            st.rerun()

    if not st.session_state.initialized:
        return

    if "rag_system" not in st.session_state:
        st.session_state.rag_system = HealthInsuranceRAG(
            st.session_state.vector_db,
            GROQ_API_KEY
        )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "sources" in message:
                with st.expander("📄 View Citations"):
                    for i, (chunk, page, score) in enumerate(message["sources"], 1):
                        st.markdown(f"### Source {i} (Page {page})")
                        st.write(chunk)
                        st.markdown("---")

    if user_query := st.chat_input("Ask about your policy..."):
        with st.chat_message("user"):
            st.markdown(user_query)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("assistant"):
            result = st.session_state.rag_system.answer_question(user_query)
            st.markdown(result["answer"])

            with st.expander("📄 View Citations"):
                for i, (chunk, page, score) in enumerate(result["sources"], 1):
                    st.markdown(f"### Source {i} (Page {page})")
                    st.write(chunk)
                    st.markdown("---")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })

        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()
