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