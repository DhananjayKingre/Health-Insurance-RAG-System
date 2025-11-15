
# 🏥 Health-Insurance RAG Chatbot  
### **PDF → Embeddings → Vector Search → Groq LLM → Answer with Citations**

This project is a **Retrieval Augmented Generation (RAG)** chatbot that answers questions from your **Health Insurance Policy PDF** using:  
- **FAISS vector search**  
- **SentenceTransformer embeddings**  
- **Groq LLaMA-3.3-70B** (ultra-fast inference)  
- **Streamlit interactive chat UI**  

The system ensures **accurate, grounded, non-hallucinated answers** with **proper citations from the policy**.

---

## ✨ Features

- ⚡ **Super-fast answers** powered by **Groq’s low-latency LLM inference**  
- 🔍 **Accurate responses with citations** retrieved from your PDF  
- 🧠 **Vector search using FAISS** for top-K retrieval  
- 📄 **Automatic PDF → Text → Chunking → Embeddings**  
- 💬 **Beautiful Streamlit chat interface**  
- 🔁 **Rebuild vector database anytime**  
- 🧹 Clear chat history  
- 🎯 Full RAG pipeline implemented end-to-end  

---

# ⚡ Why Groq?

Groq provides **extraordinary inference speed** for large LLaMA models using its custom LPU (Language Processing Unit).  
This means answers from a **70B-parameter model** come back almost instantly — ideal for chatbots, RAG, and real-time systems.  
Groq removes the need for expensive GPUs and delivers **high accuracy + low latency + low cost** in one place.

---

# 🧠 System Architecture (RAG Pipeline)

```
                 ┌─────────────────────────────────┐
                 │       Health Insurance PDF       │
                 └───────────────────┬──────────────┘
                                     │
                          [1] Extract text (PyPDF2)
                                     │
                                     ▼
                 ┌─────────────────────────────────┐
                 │          Text Chunking           │
                 │  (800 tokens with overlap)       │
                 └───────────────────┬──────────────┘
                                     │
                          [2] Create semantic chunks
                                     │
                                     ▼
                 ┌─────────────────────────────────┐
                 │   Embeddings (MiniLM-L6-v2)      │
                 └───────────────────┬──────────────┘
                                     │
                          [3] Vectorize chunks
                                     │
                                     ▼
                 ┌─────────────────────────────────┐
                 │       FAISS Vector Store         │
                 └───────────────────┬──────────────┘
                                     │
                             [4] Similarity search
                                     │
                                     ▼
                 ┌─────────────────────────────────┐
                 │      Groq LLaMA-3.3-70B          │
                 │ (LLM reasoning on retrieved text)│
                 └───────────────────┬──────────────┘
                                     │
                          [5] Answer with citations
                                     │
                                     ▼
                 ┌─────────────────────────────────┐
                 │   Streamlit Chat UI (final)      │
                 └─────────────────────────────────┘
```

---

# 🗂 ER Diagram (Conceptual Data Flow)

```
┌────────────┐       1-to-many        ┌───────────────┐      1-to-1       ┌─────────────┐
│   PDF      │────────────────────────►│   Chunks       │──────────────────►│ Embeddings  │
└────────────┘                        └───────────────┘                   └─────┬───────┘
                                                                                │
                                                                        stored in │
                                                                                ▼
                                                                        ┌────────────┐
                                                                        │   FAISS    │
                                                                        └─────┬──────┘
                                                                              │ retrieves top-K  
                                                                              ▼
                                                                        ┌────────────┐
                                                                        │   Groq LLM │
                                                                        └─────┬──────┘
                                                                              │ final answer
                                                                              ▼
                                                                        ┌────────────┐
                                                                        │   Streamlit│
                                                                        └────────────┘
```

---

# 📁 Project Structure

```
/Health-RAG
│── app.py                     
│── data/
│     ├── faiss_index.bin      
│     └── chunks.pkl           
│── Health-Insurance-Policy.pdf
│── README.md
```

---

# 🛠 Installation

### 1️⃣ Install dependencies

```bash
pip install streamlit sentence-transformers faiss-cpu groq PyPDF2 numpy
```

### 2️⃣ Run the app

```bash
streamlit run app.py
```

---

# 💬 Example Questions

- What illnesses are excluded from OPD coverage?  
- Is maternity covered?  
- What is the waiting period for pre-existing diseases?  
- What are permanent exclusions?  

---

# 🚀 How It Works

1. Extract text from PDF  
2. Chunk text  
3. Generate embeddings  
4. Build FAISS vector DB  
5. Retrieve relevant chunks  
6. Groq LLM generates answer  
7. Response shown in Streamlit  

---

