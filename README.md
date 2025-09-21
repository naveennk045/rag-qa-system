 RAG Q&A System - Phase 1

A fast, cost-effective Retrieval-Augmented Generation (RAG) system for document Q&A.

## 🏗️ Architecture (Phase 1)

```
Documents → Split → Embed → FAISS Index
```

## 🛠️ Tech Stack

- **Text Splitting**: LangChain RecursiveCharacterTextSplitter
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Storage**: FAISS
- **LLM**: Groq (Phase 2)

## 📦 Installation

```bash
# Clone repository
git clone <your-repo-url>
cd rag-qa-system

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys
```

## 🚀 Usage

### Phase 1: Build Index

1. **Add documents** to `data/raw/` directory (PDF, TXT, DOCX)

2. **Run the indexing pipeline**:
```bash
python scripts/build_index.py
```

3. **Test embeddings**:
```bash
python scripts/test_embeddings.py
```

## 📁 Project Structure

```
rag-qa-system/
├── config/              # Configuration files
├── data/
│   ├── raw/            # Original documents (add your files here)
│   └── processed/      # Processed chunks
├── src/                # Core modules
├── scripts/            # Pipeline scripts
├── vector_db/          # FAISS index files
├── models/             # Downloaded model cache
└── logs/               # Processing logs
```

## ⚙️ Configuration

Edit `config/config.py` or `.env` file to adjust:

- `CHUNK_SIZE=1500` - Size of text chunks
- `CHUNK_OVERLAP=150` - Overlap between chunks  
- `EMBEDDING_MODEL=all-MiniLM-L6-v2` - Embedding model name

## 📊 Features

- ✅ Multiple document format support (PDF, TXT, DOCX)
- ✅ Optimized text chunking for Q&A
- ✅ Local embedding generation (no API costs)
- ✅ Fast FAISS vector similarity search
- ✅ Comprehensive logging and error handling
- ✅ Modular, extensible architecture

## 🔄 Next Steps (Phase 2)

- [ ] Query processing pipeline
- [ ] Groq LLM integration
- [ ] Streamlit web interface
- [ ] Advanced retrieval strategies

## 📝 License

MIT License

---

**Phase 1 Complete**: Document processing, chunking, embedding, and vector storage ready!


## Phase 2: Query Processing & LLM Integration ✅

### New Features Added:
- **Complete RAG Pipeline**: End-to-end question answering
- **Groq LLM Integration**: Fast, cost-effective text generation
- **Smart Context Retrieval**: Relevance-based chunk selection
- **Streaming Responses**: Real-time answer generation
- **Web Interface**: User-friendly Streamlit app
- **CLI Tool**: Command-line interface for testing

### Additional Scripts:
- `scripts/query_cli.py` - Interactive command-line interface
- `scripts/test_rag_pipeline.py` - Pipeline testing utilities
- `app/streamlit_app.py` - Web-based chat interface

### Usage (Phase 2):

#### Command Line Interface:
```bash
# Interactive mode
python scripts/query_cli.py

# Single query mode
python scripts/query_cli.py "What is the main topic discussed?"
```

#### Web Interface:
```bash
# Start Streamlit app
streamlit run app/streamlit_app.py
```

#### Test Pipeline:
```bash
python scripts/test_rag_pipeline.py
```

### Configuration:
Add these to your `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.3
ENABLE_STREAMING=true
```

---

## 🔄 Complete Workflow

### Phase 1: Build Index
```bash
python scripts/build_index.py
```

### Phase 2: Query Documents
```bash
# CLI
python scripts/query_cli.py

# Web App  
streamlit run app/streamlit_app.py
```