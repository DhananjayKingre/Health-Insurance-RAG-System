# 🏥 Health Insurance RAG System (Flask + Groq + FAISS)

An end‑to‑end **Healthcare Insurance RAG (Retrieval-Augmented Generation) System** built using:

- **Flask API**  
- **Groq Llama‑3.3 70B Model**  
- **FAISS Vector Database**  
- **Sentence Transformers**  
- **PDF → Chunking → Embedding → Retrieval → Answer Generation**

This system allows users to ask **natural‑language questions** related to a **Health Insurance Policy PDF**, retrieves the most relevant policy chunks using FAISS, and generates answers using Groq LLM with citations.

---

# 📌 Features

### ✅ PDF Processing  
✔ Extracts text from PDF  
✔ Splits text into page‑level chunks  
✔ Uses sentence‑based segmentation  

### ✅ Embedding + Vector DB  
✔ Creates embeddings using **SentenceTransformer (all‑MiniLM‑L6‑v2)**  
✔ Stores vectors in **FAISS**  
✔ Saves chunk/page metadata using `pickle`  
✔ Supports **rebuilding** the vector index

### ✅ RAG Pipeline  
✔ Query → Retrieve → Rerank → Groq LLM Answer  
✔ Llama‑3.3‑70B‑Versatile for high‑quality output  
✔ Sources and page numbers returned in response

### ✅ Flask REST API  
- `GET /health` → System health check  
- `GET /Health-insurance-rag-system/askQuestion/<question>`  
- `POST /Health-insurance-rag-system/askQuestion`  
- `POST /Health-insurance-rag-system/rebuild-database`  
- UI served from `/`

---

# ⚙️ Technology Stack

| Component | Technology |
|----------|------------|
| Backend API | Flask |
| Embeddings | SentenceTransformer |
| Vector DB | FAISS |
| LLM | Groq Llama‑3.3‑70B |
| Frontend | HTML + JS (template/index.html) |
| File Storage | Local PDF + FAISS index |

---

# 🧠 System Architecture

```
                   ┌─────────────────────────┐
                   │   PDF Document (Policy) │
                   └─────────────┬───────────┘
                                 │ Extract & Chunk
                                 ▼
                      ┌──────────────────────┐
                      │   PDF Processor      │
                      └──────────┬───────────┘
                                 │ Embedding
                                 ▼
                       ┌────────────────────┐
                       │ Sentence Transformer│
                       └──────────┬─────────┘
                                 │ Vectors
                                 ▼
                      ┌────────────────────────┐
                      │      FAISS Index       │
                      └──────────┬────────────┘
                                 │ Retrieve Top‑K
                                 ▼
                      ┌────────────────────────┐
                      │     RAG Pipeline       │
                      └──────────┬────────────┘
                                 │
                                 ▼
                         ┌──────────────┐
                         │ Groq LLM API │
                         └───────┬──────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Final Answer JSON │
                        └──────────────────┘
```

---

# 🗃 ER Diagram (Logical Data Flow)

```
┌───────────────────────┐
│        PDF Pages       │
└──────────────┬────────┘
               │1:N
┌──────────────▼────────┐
│     Chunks Table       │
│ chunk_text             │
│ page_number            │
│ embedding_vector       │
└──────────────┬────────┘
               │1:1
┌──────────────▼────────┐
│     FAISS Index        │
│ vector_id              │
│ similarity_score       │
└────────────────────────┘
```

---

# 🚀 How to Run the Flask RAG API

### 1️⃣ Install Dependencies

```sh
pip install flask flask-cors groq sentence-transformers faiss-cpu PyPDF2 numpy
```

### 2️⃣ Update Configuration inside the script

```python
PDF_PATH = r"D:\onelab_flask\Health-Insurance-Policy.pdf"
GROQ_API_KEY = "your-api-key"
```

### 3️⃣ Run the Server

```sh
python app.py
```

Flask will start at:

```
http://127.0.0.1:5000
```

---

# 🧪 API Endpoints

### 🔹 1. Health Check

```
GET /health
```

### 🔹 2. Ask Question (GET)

```
GET /Health-insurance-rag-system/askQuestion/<your question>
```

### 🔹 3. Ask Question (POST)

```json
POST /Health-insurance-rag-system/askQuestion
{
  "question": "What illnesses are excluded from OPD coverage?"
}
```

### 🔹 4. Rebuild Vector Database

```
POST /Health-insurance-rag-system/rebuild-database
```

---

# 🌐 Frontend UI

The root route serves the chatbot UI:

```
/
```

It loads:
- Chat interface  
- Loader animations  
- JS fetch to Flask question API  

---

# 🌟 About Groq (Short Summary)

**Groq** provides an ultra‑fast LPU™ (Language Processing Unit) that accelerates AI model inference at extreme speed and low latency.  
This project uses the **Groq Llama‑3.3‑70B Versatile model** to deliver real‑time responses for insurance‑related Q&A with high accuracy and contextual grounding.

---

# 📄 Project Structure

```
project/
│── app.py
│── templates/
│      └── index.html
│── data/
│      ├── faiss_index.bin
│      └── chunks.pkl
│── Health-Insurance-Policy.pdf
│── README.md
```

---

# 📥 Download

You can download this README.md from this interface.

---

# 🙌 Author  
**Dhananjay Kingre**  
AI/ML Engineer | Python Developer | RAG Systems | Groq | NLP

