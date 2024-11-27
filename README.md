## 🏗️ Architecture
- **Resume Parsing:** LlamaParse for structured metadata extraction
- **Vector Database:** Dual embedding strategy in Pinecone with static chunking (200 tokens, 15-char overlap)
- **Language Models:** LLaMA 3.2 & GPT-4 for semantic analysis
- **Retrieval:** RAG-based system with Ada Embeddings
- **Frontend:** Streamlit UI for intuitive interaction

## 🛠️ Tools & Technologies
- **LlamaIndex & LlamaCloud:** for orchestration
- **Pinecone:** for vector storage
- **GPT-4:** for candidate matching
- **LlamaParse:** for document processing
- **Streamlit:** for user interface
- **Python:** for backend development

## ✨ Key Achievement
Reduced recruitment screening time by 5 hours through intelligent automation and semantic search capabilities.

## 🚀 Getting Started

### 1. API Key Configuration
Setup your open AI key in `.env` file

### 2. Starting the Django Server
```bash
python manage.py runserver 9000
```

### 3. Starting the chainlit App
In another terminal, start the chainlit app
```bash
chainlit run app.py -w
```

