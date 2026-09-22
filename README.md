# 🎓 SIN - School of AI | AI Agent Xcelerator 2026
### Production 6-Pillar RAG Portal & Interactive Practical Laboratory

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-red.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art **Retrieval-Augmented Generation (RAG)** portal and interactive educational laboratory built for **SIN - School of AI** to counsel prospective students for the **AI Agent Xcelerator Program 2026** (*October 1, 2026*).

---

## 🌟 Key Highlights & Features

### 1. 🔬 Complete 6-Pillar Production RAG Pipeline
Directly engineered from the official RAG Knowledge-to-Impact architectural blueprint:
- **Pillar 1: Retrieval Quality**: Multi-Query expansion (3 semantic variations), Hybrid Lexical (BM25) + Dense Vector reranking with tunable $\alpha$, and similarity threshold noise filtering.
- **Pillar 2: Chunking & Indexing**: Semantic section-aware markdown chunker preserving curriculum hierarchies, table structures, and token overlaps.
- **Pillar 3: Memory & Conversation**: **Reverse Conversational Memory Architecture (RMA)** with 0-token semantic cache hits (100% LLM credit savings) and selective multi-turn memory retrieval.
- **Pillar 4: Prompting & Generation**: Dynamic system grounding prompt injecting user custom instructions, answer style controls, and strict anti-hallucination fallback logic.
- **Pillar 5: Knowledge Base Quality**: Verified 2026 curriculum dataset in [`knowledgebase/`](knowledgebase/) covering the 5-month journey (3M training + 2M industry internship) and 25-student cohort exclusivity.
- **Pillar 6: Evaluation & Optimization**: Real-time token consumption ledger, net credit savings calculations, and turn-by-turn groundedness/faithfulness scoring.

### 2. 📚 Interactive Practical RAG Glossary & Lab
A dedicated hands-on laboratory tab covering all 30+ terms across the 6 pillars with live interactive sandbox widgets:
- Live ChromaDB Top-K search against dense embeddings
- Dynamic Similarity Cutoff filter with pass/drop badges
- Multi-Query 3-angle generator
- Hybrid Search Alpha ($\alpha$) slider
- Chunk Size & Overlap character slicer
- RMA Semantic Cache hit simulator
- Sampling Temperature distribution tester ($0.0 - 1.0$)
- Live Groundedness & Faithfulness response scorer

### 3. 💬 ChatGPT-Style Multi-Thread Chat & Custom Instructions
- **Multi-Thread Chat**: Create isolated chat threads with `➕ New Chat Thread`, switch between sessions, or delete threads with one click.
- **Auto-Titling**: First question in a thread automatically titles the conversation based on the student's topic.
- **ChatGPT-Style Custom Instructions**: Set *'What should the AI know about you?'* and *'How should the AI respond?'* in the sidebar settings, persisted across all threads.

### 4. 🔐 User Authentication & ID Creation
- Secure user registration and login backed by persistent SQLite (`rag_portal.db`) with salted SHA-256 password hashing.
- **1-Click Demo Login**: Pre-seeded demo student account (`student` / `sin2026`) for instant evaluation.

---

## 🗂️ Project Structure & Datasets

```
├── app.py                     # Main Streamlit web application & authentication gate
├── auth_manager.py            # SQLite user authentication, threads CRUD, custom instructions
├── rag_engine.py              # ChromaDB vector store, ONNX embeddings, markdown chunker
├── rag_advanced.py            # 6-pillar RAG pipeline (multi-query, hybrid rerank, groundedness)
├── glossary_lab.py            # Interactive practical RAG glossary & 6-pillar sandbox
├── llm_service.py             # OpenRouter LLM streaming with st.secrets & .env support
├── memory_engine.py           # Reverse Conversational Memory Architecture (RMA) & cache
├── token_manager.py           # Tiktoken-based token counting, limits, and credit accounting
├── visualizer.py              # Plotly 3D vector space, Sankey flow, and token charts
│
├── knowledgebase/             # Knowledge base dataset files
│   ├── info.md                # Primary verified syllabus specification (auto-chunked & embedded)
│   ├── Agent Xcelerator...pdf # Program proposal & curriculum breakdown
│   ├── Er_Harshvardhan...pdf  # Lead architect & mentor profile
│   └── SIN_School_of_AI...pdf # Institutional profile
│
├── brand/                     # Official brand assets
│   ├── logo.png               # High-res SIN - School of AI brand logo
│   └── brand.md               # Brand guidelines, color tokens, and typography
│
├── .streamlit/
│   ├── config.toml            # Imperial crimson dark theme & headless container config
│   └── secrets.toml.example   # Template for Streamlit Cloud secrets
│
├── requirements.txt           # Production dependencies for local & Streamlit Cloud
├── .env.example               # Environment variable template
└── .gitignore                 # Excludes secrets, caches, and local pycache
```

---

## 🚀 Quick Start (Local Setup)

### 1. Clone the repository
```bash
git clone https://github.com/Harshvardhan1609/aixcelerator2026.git
cd aixcelerator2026
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your OpenRouter API Key
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and add your OpenRouter API key:
```env
OPEN_ROUTER_API=sk-or-v1-your-actual-key-here
```
*(Get a free or paid key at [openrouter.ai](https://openrouter.ai/keys))*

### 5. Launch the application
```bash
streamlit run app.py
```
Visit `http://localhost:8501` in your browser!

---

## ☁️ Deploying to Streamlit Cloud (via GitHub)

Deploying this app to Streamlit Community Cloud takes less than 2 minutes:

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "feat: complete 6-pillar RAG portal with auth and dataset"
   git push origin main
   ```
2. **Go to [share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. **Click "New app"** and fill in:
   - **Repository**: `Harshvardhan1609/aixcelerator2026` (or your fork)
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. **Configure Secrets**:
   - Expand **Advanced settings...** -> **Secrets**.
   - Paste:
     ```toml
     OPEN_ROUTER_API = "sk-or-v1-your-openrouter-key"
     ```
5. **Click "Deploy!"**
   - Streamlit Cloud will install `requirements.txt`, launch `app.py`, and automatically initialize the vector database from `knowledgebase/info.md`!

For detailed instructions and screenshots, see [`DEPLOYMENT.md`](DEPLOYMENT.md).

---

## 🔑 Demo Login Credentials

For instant access without registering a new ID, use the 1-Click Demo button or enter:
- **User ID**: `student`
- **Password**: `sin2026`

---

## 🛡️ Architecture & Security
- **No Hardcoded Keys**: API keys are loaded via `st.secrets` on cloud or `.env` locally.
- **Salted Password Hashing**: Passwords are never stored in plaintext (salted SHA-256).
- **Session Isolation**: Each user ID has isolated chat threads and custom instructions stored in SQLite.

---

## 📄 License
MIT License. Developed for **SIN - School of AI** AI Agent Xcelerator Program 2026.
