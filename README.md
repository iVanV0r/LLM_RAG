# 🧪 RAG for Scientific Problem Extraction

Local RAG system for extracting scientific problems from PDF articles (RU/EN).

**Pipeline:** PDF → Ollama → FAISS → LLM → BERTScore

---

## Features

- 📄 Parse PDFs in Russian and English
- 🔍 Vector search with FAISS + Ollama embeddings
- 🤖 Two methods: **Base** (LLM only) and **RAG** (with context)
- 📊 Quality evaluation via BERTScore
- 🔄 Compare multiple LLM models
- 💾 Caching for faster re-runs
- 📈 Visualization charts

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Start Ollama
ollama serve
ollama pull gemma2:9b
ollama pull nomic-embed-text

# Add PDFs
data/pdfs/ru/    # Russian articles
data/pdfs/en/    # English articles

# Run
python main.py
