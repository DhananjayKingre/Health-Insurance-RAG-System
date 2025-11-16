from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import uvicorn
from dotenv import load_dotenv 

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configuration: File paths, model names, and API keys
PDF_PATH = r"D:\onelab_flask\Health-Insurance-Policy.pdf"
VECTOR_DB_PATH = r"D:\onelab_flask\vectorData"
MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"


# Initialize FastAPI app with project title
app = FastAPI(title="Health Insurance RAG System")

# Enable CORS so frontend apps can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold vector DB and RAG system
vector_db = None
rag_system = None


# Pydantic request/response models for API validation
class QuestionRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    page: int
    content: str
    relevance_score: float


class AnswerResponse(BaseModel):
    status: str
    question: str
    answer: str
    sources: List[SourceResponse]
    source_count: int


class HealthResponse(BaseModel):
    status: str
    vector_db_initialized: bool
    rag_system_initialized: bool


class RebuildResponse(BaseModel):
    status: str
    message: str

# PDF PROCESSOR: Extracts and chunks PDF text

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
    # Splits extracted text into smaller overlapping chunks
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


# Vector Database: Creates, loads, and searches FAISS index
class VectorDatabase:
    def __init__(self, db_path: str, model_name: str = MODEL_NAME):
        # Prepare folder path and filenames for FAISS index and chunks
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


# RAG SYSTEM: Retrieves context + generates answer with Groq LLM
class HealthInsuranceRAG:
    def __init__(self, vector_db: VectorDatabase, api_key: str):
        # Initialize Groq LLM client and set model
        self.vector_db = vector_db
        self.client = Groq(api_key=api_key)
        self.model = GROQ_MODEL
    # Search context, send to LLM, and return final answer + citations
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


# System Initialization: Loads or creates vector DB and RAG system
def initialize_system():
    global vector_db, rag_system
    
    print("Initializing Health Insurance RAG System...")
    
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
      # Initialize RAG system
    rag_system = HealthInsuranceRAG(vector_db, GROQ_API_KEY)
    print("✅ RAG System initialized and ready!")


# STARTUP EVENT Runs when FastAPI launches

@app.on_event("startup")
async def startup_event():
    #Initialize the system when the FastAPI app starts
    initialize_system()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Health Insurance RAG System API",
        "endpoints": {
            "health_check": "/health",
            "ask_question_get": "/Health-insurance-rag-system/askQuestion/{question}",
            "ask_question_post": "/Health-insurance-rag-system/askQuestion",
            "rebuild_database": "/Health-insurance-rag-system/rebuild-database"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_db_initialized": vector_db is not None,
        "rag_system_initialized": rag_system is not None
    }


#ask question using get request
@app.get("/Health-insurance-rag-system/askQuestion/{question}", response_model=AnswerResponse)
async def ask_question_get(question: str):
    
    try:
        if not rag_system:
            raise HTTPException(
                status_code=500,
                detail="RAG system not initialized"
            )

        result = rag_system.answer_question(question)
        
        # Format sources for JSON response
        sources = []
        for chunk, page, score in result['sources']:
            sources.append({
                "page": page,
                "content": chunk,
                "relevance_score": round(score, 4)
            })

        return {
            "status": "success",
            "question": question,
            "answer": result['answer'],
            "sources": sources,
            "source_count": len(sources)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/Health-insurance-rag-system/askQuestion", response_model=AnswerResponse)
async def ask_question_post(request: QuestionRequest):
    
    try:
        if not rag_system:
            raise HTTPException(
                status_code=500,
                detail="RAG system not initialized"
            )

        result = rag_system.answer_question(request.question)
        
        # Format sources for JSON response
        sources = []
        for chunk, page, score in result['sources']:
            sources.append({
                "page": page,
                "content": chunk,
                "relevance_score": round(score, 4)
            })

        return {
            "status": "success",
            "question": request.question,
            "answer": result['answer'],
            "sources": sources,
            "source_count": len(sources)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Rebuild vector database from scratch
@app.post("/Health-insurance-rag-system/rebuild-database", response_model=RebuildResponse)
async def rebuild_database():
    """Rebuild the vector database from scratch"""
    try:
        global vector_db, rag_system
        
        db_path = Path(VECTOR_DB_PATH)
        if (db_path / "faiss_index.bin").exists():
            (db_path / "faiss_index.bin").unlink()
        if (db_path / "chunks.pkl").exists():
            (db_path / "chunks.pkl").unlink()
        
        initialize_system()
        
        return {
            "status": "success",
            "message": "Database rebuilt successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main entry point for running FastAPI with Uvicorn
if __name__ == '__main__':
    # Run FastAPI app with uvicorn
    uvicorn.run(
        "main:app",  
        host='0.0.0.0',
        port=8000,
        reload=True
    )