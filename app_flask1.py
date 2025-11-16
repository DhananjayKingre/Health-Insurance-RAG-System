from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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
from dotenv import load_dotenv 

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# File paths, model names, and API keys used in this RAG project.
PDF_PATH = r"D:\onelab_flask\Health-Insurance-Policy.pdf"
VECTOR_DB_PATH = r"D:\onelab_flask\vectorData"
TEMPLATE_FOLDER = r"D:\onelab_flask\templates"
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

app = Flask(__name__, template_folder=TEMPLATE_FOLDER)
# Enable CORS for all routes
CORS(app)  

# Global variables for system components
vector_db = None
rag_system = None

# PDF PROCESSOR

class PDFProcessor:
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        # Reads the PDF file and extracts text page by page.
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
        # Splits extracted text into smaller overlapping chunks for better search accuracy.
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
                    # Create chunk once the size exceeds limit
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

# VECTOR DATABASE

class VectorDatabase:
    def __init__(self, db_path: str, model_name: str = MODEL_NAME):
        # Initializes paths and loads the embedding model.
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.db_path / "faiss_index.bin"
        self.chunks_file = self.db_path / "chunks.pkl"

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        self.index = None
        self.chunks_with_pages = []

    def create_and_save(self, chunks_with_pages: List[Tuple[str, int]]):
        # Creates embeddings for chunks and saves FAISS index + chunk data.
        self.chunks_with_pages = chunks_with_pages
        chunks = [chunk for chunk, _ in chunks_with_pages]

        batch_size = 32
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings)
            print(f"Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

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
        # Searches vector DB → builds context → sends to LLM → returns answer.
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



# SYSTEM INITIALIZATION

def initialize_system():
    global vector_db, rag_system
    
    print("Initializing Health Insurance RAG System...")
    
    # Create templates folder if it doesn't exist
    Path(TEMPLATE_FOLDER).mkdir(parents=True, exist_ok=True)
    
    vector_db = VectorDatabase(VECTOR_DB_PATH)

    if vector_db.load():
        print("✅ Vector database loaded successfully!")
    else:
        print("Creating new vector database...")
        
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"❌ PDF file not found at: {PDF_PATH}")

        text = PDFProcessor.extract_text(PDF_PATH)
        chunks_with_pages = PDFProcessor.chunk_text(text)
        vector_db.create_and_save(chunks_with_pages)
        print("✅ Vector database created successfully!")

    rag_system = HealthInsuranceRAG(vector_db, GROQ_API_KEY)
    print("✅ RAG System initialized and ready!")

# FLASK ROUTES

@app.route('/')
def home():
    """Render the main HTML interface"""
    return render_template('index.html')


@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "vector_db_initialized": vector_db is not None,
        "rag_system_initialized": rag_system is not None
    })


@app.route('/Health-insurance-rag-system/askQuestion/<path:question>', methods=['GET'])
def ask_question_get(question):
    # GET request — answers user query and returns JSON response.
    try:
        if not rag_system:
            return jsonify({
                "status": "error",
                "message": "RAG system not initialized"
            }), 500

        result = rag_system.answer_question(question)
        
        # Format sources for JSON response
        sources = []
        for chunk, page, score in result['sources']:
            sources.append({
                "page": page,
                "content": chunk,
                "relevance_score": round(score, 4)
            })

        return jsonify({
            "status": "success",
            "question": question,
            "answer": result['answer'],
            "sources": sources,
            "source_count": len(sources)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/Health-insurance-rag-system/askQuestion', methods=['POST'])
def ask_question_post():
    try:
        if not rag_system:
            return jsonify({
                "status": "error",
                "message": "RAG system not initialized"
            }), 500

        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'question' field in request body"
            }), 400

        question = data['question']
        result = rag_system.answer_question(question)
        
        # Format sources for JSON response
        sources = []
        for chunk, page, score in result['sources']:
            sources.append({
                "page": page,
                "content": chunk,
                "relevance_score": round(score, 4)
            })

        return jsonify({
            "status": "success",
            "question": question,
            "answer": result['answer'],
            "sources": sources,
            "source_count": len(sources)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/Health-insurance-rag-system/rebuild-database', methods=['POST'])
def rebuild_database():
    try:
        global vector_db, rag_system
        
        db_path = Path(VECTOR_DB_PATH)
        if (db_path / "faiss_index.bin").exists():
            (db_path / "faiss_index.bin").unlink()
        if (db_path / "chunks.pkl").exists():
            (db_path / "chunks.pkl").unlink()
        
        initialize_system()
        
        return jsonify({
            "status": "success",
            "message": "Database rebuilt successfully"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# MAIN

if __name__ == '__main__':
    # Initialize the system before starting the server
    initialize_system()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)