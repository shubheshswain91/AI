
### Task 3: Environment Setup
uv venv venv

### verify
uv run python -c "import sklearn, pandas, numpy; print('All packages available')"


## Embedding package
uv pip install "sentence-transformers==2.2.2" "huggingface_hub<0.20" openai

### To verify installations

uv run python -c "import sentence_transformers, openai; print('Embedding packages available')"

### Install vector database package:
uv pip install chromadb

### Verify installation of Chroma DB
uv run python -c "import chromadb; print('ChromaDB available')"

## Vector DB

Vector databases are specialized databases designed to store and search high-dimensional vectors (embeddings). Unlike traditional databases that store text, numbers, or structured data, vector databases are optimized for similarity search.

Key benefits of vector databases:

Store millions of embeddings efficiently
Fast similarity search across all vectors
Persistent storage that survives restarts
Can be shared across multiple applications


#### Task 3: Install Chunking Dependencies

Why do we need these tools?

🔧 LangChain: A powerful framework for building RAG applications

Provides RecursiveCharacterTextSplitter for smart document chunking
Handles different chunk sizes, overlaps, and separators
Makes chunking configuration simple and flexible
🧠 spaCy: Advanced natural language processing library

Provides SpacyTextSplitter for sentence-aware chunking
Understands sentence boundaries and linguistic structure
Breaks documents at natural language boundaries (not just characters)

### Install chunking packages

uv pip install langchain spacy