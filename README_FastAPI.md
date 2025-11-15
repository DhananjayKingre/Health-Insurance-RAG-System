# Health Insurance RAG Chatbot (FastAPI + Groq LLaMA3)

A production-ready **Health Insurance Question-Answering System** built using:

- **FastAPI** (Backend API)
- **Groq LLaMA3** (Ultra‑fast LLM inference)
- **Vector Database (FAISS)**
- **PDF Text Extraction**
- **Advanced UI Chat Interface**
- **RAG Pipeline (Retrieve → Rerank → Generate)**

This project enables users to ask insurance-related questions and receive accurate answers retrieved from official policy documents.

---

## 🚀 What is Groq? (Short Simple Explanation)

**Groq** is a lightning‑fast AI inference engine built on LPU (Language Processing Units).  
It is **100× faster** than GPUs for LLMs and can generate answers in real‑time with extremely low latency.  
This speed makes Groq perfect for **chatbots, RAG systems, assistants, and real‑time reasoning** applications.

---

# 🧠 System Architecture (FastAPI RAG)

```
User Query → FastAPI API → Embed Query → FAISS Vector Search  
→ Retrieve Top Relevant Chunks → Groq LLaMA3 → Final Answer → UI
```

---

# 🗄️ ER Diagram (Conceptual)

```
+------------------+
|   Documents      |
+------------------+
| doc_id (PK)      |
| title            |
| content          |
| embedding[]      |
+------------------+

          1:N

+----------------------+
|   Vector_Chunks      |
+----------------------+
| chunk_id (PK)        |
| doc_id (FK)          |
| chunk_text           |
| embedding[]          |
+----------------------+
```

---

# 📂 Project Folder Structure

```
project/
│── data/
│     └── policies.pdf
│── vectors/
│     └── faiss.index
│     └── chunks.json
│── app/
│     ├── main.py
│     ├── rag_pipeline.py
│     ├── embeddings.py
│     └── utils.py
│── ui/
│     └── index.html
│── requirements.txt
│── README.md
```

---

# ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment
```
python -m venv venv
venv\Scriptsctivate
```

### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Add Groq API Key  
Create a **.env** file:

```
GROQ_API_KEY=your_api_key_here
```

---

# 🧩 Build Vector Database (FAISS)

Run your preprocessing script:

```
python build_vector_db.py
```

This generates:

- faiss.index  
- chunks.json  

---

# 🚀 Start FastAPI Server

```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

# 🔥 API Endpoints

### **POST /ask**
Ask any health insurance question.

#### Example Request:
```json
{
  "question": "What illnesses are excluded from OPD coverage?"
}
```

#### Example Response:
```json
{
  "answer": "According to the policy document, OPD excludes..."
}
```

---

# 🧠 RAG Flow (Detailed)

1. User asks a question  
2. Query converted to embeddings  
3. FAISS returns top relevant chunks  
4. Chunks passed to Groq LLaMA3  
5. LLM generates final insurance-specific answer  
6. Response sent back to UI  

---

# 🖥️ UI Chatbox Integration

Your frontend uses:

- Modern chat UI  
- Messages centered on screen  
- Auto-dismiss “Chatbot Ready” toast  
- Smooth animation  
- Supports streaming (optional)

---

# 🛠️ Technologies Used

| Component | Technology |
|----------|------------|
| Backend | FastAPI |
| LLM | Groq LLaMA3 |
| Embeddings | HuggingFace |
| Vector DB | FAISS |
| PDF Reader | pdfminer / PyPDF |
| UI | HTML + JS |

---

# 📌 Future Improvements

- Add Streaming Responses  
- Add User Authentication  
- Add Chat History Storage  
- Add Multi‑PDF Support  
- Add Admin Dashboard  

---



---


