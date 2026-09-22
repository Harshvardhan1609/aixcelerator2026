# 🚀 Step-by-Step Streamlit Cloud Deployment Guide
### SIN - School of AI | AI Agent Xcelerator 2026 RAG Portal

This guide provides the exact steps to deploy the AI Agent Xcelerator RAG Portal to **Streamlit Community Cloud** using your GitHub repository.

---

## 📋 Pre-Deployment Checklist

Before deploying, make sure you have:
- [x] A GitHub account.
- [x] A free account on [Streamlit Community Cloud](https://share.streamlit.io).
- [x] An OpenRouter API Key ([Get one here](https://openrouter.ai/keys)).
- [x] The repository code ready with `requirements.txt`, `.streamlit/config.toml`, and `knowledgebase/`.

---

## 🛠️ Step 1: Push Your Code to GitHub

Open your terminal in `d:\RAG` and run the following commands:

```bash
# 1. Check current status
git status

# 2. Stage all updated files (excluding .env and pycache via .gitignore)
git add .

# 3. Commit your changes
git commit -m "feat: complete 6-pillar RAG portal with auth, threads, custom instructions, and Streamlit Cloud setup"

# 4. Push to your GitHub main branch
git push origin main
```

> **Security Note**: Your `.env` file containing local API keys is ignored by `.gitignore` and will **never** be pushed to GitHub.

---

## ☁️ Step 2: Deploy on Streamlit Community Cloud

1. Navigate to **[share.streamlit.io](https://share.streamlit.io)** in your browser.
2. Sign in with your **GitHub** account.
3. Click the **"Create app"** (or **"New app"**) button in the top right.
4. Fill in the deployment details:
   - **Repository**: `Harshvardhan1609/aixcelerator2026`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (optional): Choose a custom subdomain (e.g. `sin-ai-agent-xcelerator.streamlit.app`).

---

## 🔐 Step 3: Configure Streamlit Secrets (API Key)

**Crucial Step**: Because `.env` is not committed to GitHub, you must provide your OpenRouter API key to Streamlit Cloud securely:

1. On the app creation screen, click **"Advanced settings..."** at the bottom.
   *(Or if the app is already created: click `Settings` (gear icon) -> `Secrets` in the right-hand menu).*
2. In the **Secrets** text box, paste the following:

```toml
# OpenRouter API Key for SIN Student Counsellor
OPEN_ROUTER_API = "sk-or-v1-your-actual-openrouter-api-key-here"
```

3. Click **"Save"**.
4. Click **"Deploy!"**.

---

## ⏳ Step 4: What Happens on First Cloud Boot?

Streamlit Cloud will automatically:
1. **Provision Container**: Boot a clean Debian Linux environment.
2. **Install Dependencies**: Run `pip install -r requirements.txt` (including `chromadb`, `plotly`, `pysqlite3-binary`, and `tiktoken`).
3. **Auto-Initialize SQLite**: `auth_manager.py` creates `rag_portal.db` and seeds the demo account `student` / `sin2026`.
4. **Auto-Ingest Knowledge Base**: `rag_engine.py` reads `knowledgebase/info.md`, chunks the curriculum tables and dates, generates 384-dimensional ONNX dense vectors, and stores them in ChromaDB.
5. **Launch Application**: The portal is live with SSL (`https://...streamlit.app`)!

---

## 🧪 Step 5: Verify Your Live App

1. Visit your live Streamlit Cloud URL.
2. Log in using the **1-Click Demo Login** button (or sign in with `student` / `sin2026`).
3. Try asking:
   - *"What is the duration, batch size, and practical approach?"*
   - *"Tell me about the 2-month internship phase."*
4. Test the **Custom Instructions** in the sidebar:
   - Bio: *"3rd year engineering student interested in autonomous agents."*
   - Output preference: *"Crisp bullet points with project suggestions."*
5. Switch to **Tab 4: 📚 Interactive RAG Glossary & Practical Lab** to explore the live interactive widgets for all 6 pillars!
6. Click **`➕ New Chat Thread`** to verify multi-threading.

---

## ❓ Troubleshooting & FAQs

### Q: Why do I see "OpenRouter API key is missing"?
- Make sure you added `OPEN_ROUTER_API = "sk-or-v1-..."` in **Streamlit Cloud Dashboard -> Settings -> Secrets**.
- Verify there are no trailing spaces or missing quotes around your key.

### Q: Does the app re-index the dataset on every reboot?
- If `chroma_db` is stored, it queries the existing collection. If empty, `rag_engine.ingest_knowledge_base()` re-indexes from `knowledgebase/info.md` in ~2 seconds automatically.
- You can also click the **"🔄 Re-index Knowledge Base"** button in the sidebar at any time.

### Q: Can multiple users use the deployed app simultaneously?
- Yes! Streamlit Cloud handles multi-user sessions. Each user who creates an account gets their own isolated chat threads and custom instructions stored in the SQLite database.

---

🎉 **Congratulations! Your Production RAG Portal is fully configured and ready for cloud deployment!**
