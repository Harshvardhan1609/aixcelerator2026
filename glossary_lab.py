import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Optional
from rag_advanced import (
    generate_multi_queries,
    simulate_chunking,
    evaluate_groundedness,
    compute_lexical_overlap_score
)

def render_interactive_glossary_lab(rag_engine=None):
    """
    Interactive Practical RAG Glossary & Laboratory.
    Covers all 6 pillars from the RAG architecture infographic:
    1. Retrieval Quality
    2. Chunking & Indexing
    3. Memory & Conversation
    4. Prompting & Generation
    5. Knowledge Base Quality
    6. Evaluation & Optimization
    """
    st.markdown("## 📚 Interactive Practical RAG Glossary & Laboratory")
    st.markdown(
        "Master the **6 Core Pillars of Modern Production RAG** through hands-on interactive widgets, "
        "live parameter testing, and real-time architectural simulations."
    )
    st.caption("🧭 *Inspired by the official RAG Knowledge-to-Impact Architecture Blueprint.*")

    pillar_tabs = st.tabs([
        "🔍 1. Retrieval Quality",
        "📦 2. Chunking & Indexing",
        "🧠 3. Memory & Conversation",
        "⚙️ 4. Prompting & Generation",
        "📑 5. Knowledge Base Quality",
        "📊 6. Evaluation & Optimization"
    ])

    # =========================================================================
    # PILLAR 1: RETRIEVAL QUALITY
    # =========================================================================
    with pillar_tabs[0]:
        st.markdown("### 🔍 Pillar 1: Retrieval Quality (Find the Right Context)")
        st.write("Retrieval quality determines whether the system finds the exact, high-signal information needed to answer the student's question.")
        
        # Term 1: Top-K Value
        with st.expander("🎯 **Top-K Value** — Number of Candidate Neighbors Retrieved", expanded=True):
            st.markdown(
                "**Definition**: `Top-K` defines how many nearest-neighbor chunks are extracted from the vector database (ChromaDB) "
                "based on cosine distance. A higher K provides more context but increases prompt token consumption and risk of distractors."
            )
            st.markdown("#### 🧪 Interactive Sandbox: Live Top-K Retrieval Tester")
            col_k1, col_k2 = st.columns([3, 1])
            with col_k1:
                test_q = st.text_input(
                    "Test Query for ChromaDB:",
                    value="What is the cohort size and practical project structure?",
                    key="glo_topk_q"
                )
            with col_k2:
                test_k = st.slider("Select Top-K Chunks:", min_value=1, max_value=8, value=3, key="glo_topk_slider")

            if st.button("🚀 Run Live Top-K Search", key="btn_live_topk"):
                if rag_engine:
                    with st.spinner(f"Retrieving Top-{test_k} chunks from ChromaDB..."):
                        _, retrieved = rag_engine.get_context_string(test_q, n_results=test_k)
                        st.success(f"Retrieved {len(retrieved)} nearest neighbor chunks!")
                        for i, r in enumerate(retrieved, 1):
                            sim_pct = round(max(0.0, (1.0 - (r.get('distance', 0.5) / 2.0))) * 100, 1)
                            st.markdown(f"**Rank #{i}** (Similarity: `{sim_pct}%` | ID: `{r['id']}`):")
                            st.caption(r.get("document", "")[:220] + "...")
                else:
                    st.info("ChromaDB engine is initializing.")

        # Term 2: Similarity Threshold
        with st.expander("⚖️ **Similarity Threshold** — Noise Filtering Cutoff"):
            st.markdown(
                "**Definition**: A minimum vector confidence score that a chunk must exceed to be admitted into the prompt context. "
                "Chunks below this cutoff are dropped to prevent hallucination and save tokens."
            )
            st.markdown("#### 🧪 Interactive Sandbox: Similarity Cutoff Filter")
            cutoff_val = st.slider("Similarity Threshold Cutoff (%):", min_value=30, max_value=85, value=50, step=5, key="glo_thresh_slider")
            sample_candidates = [
                {"title": "Program Overview & Cohort Size", "similarity": 88.5, "doc": "Capped at 25 students for personalized 1-on-1 mentorship."},
                {"title": "30% Theory / 70% Practical Model", "similarity": 74.2, "doc": "Hands-on RAG architectures, n8n automations, and live agent builds."},
                {"title": "2-Month Internship Phase", "similarity": 61.0, "doc": "Work with partner startups on client-facing autonomous agent workflows."},
                {"title": "General AI User Demographics (Distractor)", "similarity": 42.0, "doc": "Overview of general ChatGPT user trends across industries."}
            ]
            
            st.markdown(f"**Filter Results with Cutoff $\\ge {cutoff_val}\\%$:**")
            for c in sample_candidates:
                passed = c["similarity"] >= cutoff_val
                status_badge = "✅ PASSED (Included in Prompt)" if passed else "❌ DROPPED (Noise Filtered Out)"
                color = "#10B981" if passed else "#EF4444"
                st.markdown(f"• **{c['title']}** — Score: `{c['similarity']}%` → <span style='color:{color}; font-weight:700;'>{status_badge}</span>", unsafe_allow_html=True)

        # Term 3: Multi-Query Retrieval
        with st.expander("🔀 **Multi-Query Retrieval** — Overcoming Vocabulary Mismatch"):
            st.markdown(
                "**Definition**: Generates multiple diverse variations of the user query to search from different semantic angles. "
                "This ensures documents using different terminology (e.g. *'batch limit'* vs *'cohort size'*) are never missed."
            )
            st.markdown("#### 🧪 Interactive Sandbox: Live Multi-Query Generator")
            user_raw_q = st.text_input("Enter any student question:", value="How much practice vs lecture is in the course?", key="glo_mq_input")
            if st.button("✨ Generate Multi-Query Variations", key="btn_gen_mq"):
                variations = generate_multi_queries(user_raw_q)
                st.write("**Generated 3 Semantic Search Variations:**")
                for idx, v in enumerate(variations, 1):
                    st.markdown(f"**Variation {idx}:** `{v}`")

        # Term 4: Hybrid Search & Reranking
        with st.expander("⚡ **Hybrid Search & Reranking** — Vector + Lexical Synergy"):
            st.markdown(
                "**Definition**: Combines dense vector semantic similarity (meaning-based) with BM25 / token term matching (keyword-based). "
                "Alpha ($\alpha$) controls the balance between dense and lexical scores."
            )
            st.markdown("#### 🧪 Interactive Sandbox: Hybrid Alpha Tuner")
            alpha_val = st.slider("Dense Vector Weight (Alpha):", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="glo_alpha_slider")
            st.caption(f"Vector Weight: `{round(alpha_val*100)}%` | Keyword Match Weight: `{round((1-alpha_val)*100)}%`")
            
            dense_score = 0.82
            keyword_score = 0.40
            hybrid_res = (alpha_val * dense_score) + ((1 - alpha_val) * keyword_score)
            st.metric("Hybrid Combined Relevance Score", f"{round(hybrid_res * 100, 1)}%", delta=f"Alpha: {alpha_val}")

    # =========================================================================
    # PILLAR 2: CHUNKING & INDEXING
    # =========================================================================
    with pillar_tabs[1]:
        st.markdown("### 📦 Pillar 2: Chunking & Indexing (Structure Data for Better Retrieval)")
        st.write("Chunking transforms raw markdown or PDFs into semantically bite-sized vectors with clear boundary overlap.")

        with st.expander("✂️ **Chunk Size & Chunk Overlap** — Live Interactive Slicer", expanded=True):
            st.markdown(
                "**Definition**: `Chunk Size` is the maximum length of each indexed text block. "
                "`Chunk Overlap` ensures consecutive chunks share overlapping tokens, preventing ideas from being fractured across chunk boundaries."
            )
            
            sample_syllabus_text = (
                "SIN - School of AI: AI Agent Xcelerator Program 2026. "
                "The core educational framework is structured on a strict 30% Theory and 70% Practical model. "
                "Students spend the first 3 months mastering dense RAG vector databases, autonomous workflow engines, "
                "n8n integration, and API integrations with Claude and Gemini. "
                "Following the training phase, every student enters a mandatory 2-Month Industry Internship Phase. "
                "During this internship, students work on client-facing autonomous agent workflows with industry partners. "
                "Cohorts are strictly limited to 25 students to guarantee personalized mentorship from lead architects. "
                "The awareness and demo enrollment phase occurs in September 2026, and the official program start date is 1 October 2026."
            )
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                c_size = st.slider("Chunk Size (characters):", min_value=100, max_value=400, value=220, step=20, key="glo_c_size")
            with col_c2:
                c_overlap = st.slider("Chunk Overlap (characters):", min_value=0, max_value=100, value=40, step=10, key="glo_c_overlap")

            sim_chunks = simulate_chunking(sample_syllabus_text, chunk_size=c_size, chunk_overlap=c_overlap)
            st.markdown(f"**Result: Generated {len(sim_chunks)} Chunks with Overlap**")
            
            for chk in sim_chunks:
                st.markdown(f"📄 **{chk['chunk_id']}** (Length: `{chk['chars']} chars` | Words: `{chk['words']}`):")
                st.info(chk["full_text"])

        with st.expander("🏷️ **Metadata Tagging & Deduplication**"):
            st.markdown(
                "• **Metadata Tagging**: Storing structured key-value pairs (e.g. `category: Internship`, `module: 4`, `updated: 2026`) alongside vectors to enable metadata pre-filtering.\n"
                "• **Deduplication**: Checking hash or cosine equivalence between chunks to avoid indexing redundant text."
            )

    # =========================================================================
    # PILLAR 3: MEMORY & CONVERSATION
    # =========================================================================
    with pillar_tabs[2]:
        st.markdown("### 🧠 Pillar 3: Memory & Conversation (Remember for a Better Experience)")
        st.write("Equips AI agents with conversational continuity without exploding token context budgets.")

        with st.expander("⚡ **Reverse Conversational Memory Architecture (RMA)** — Live Cache Simulator", expanded=True):
            st.markdown(
                "**Definition**: In traditional Forward RAG, each turn appends all previous messages and full RAG chunks, rapidly inflating prompt tokens. "
                "**RMA** checks an in-memory vector index of prior exchanges first:\n"
                "1. **Tier 1 (Semantic Cache Hit)**: Identical or rephrased queries return instantly with **0 LLM tokens consumed (100% credits saved)**.\n"
                "2. **Tier 2 (Relevant Memory Retrieval)**: Follow-up questions retrieve only the top-1 or top-2 relevant past turns, pruning bloated history."
            )
            
            st.markdown("#### 🧪 Test RMA Semantic Cache Simulator")
            sim_q = st.text_input("Enter a test query to match against Conversational Memory:", value="What is the cohort size again?", key="glo_rma_test")
            
            # Pre-indexed reference query in memory
            ref_q = "What is the cohort size of the program?"
            st.caption(f"📌 Stored Memory Turn #1: *\"{ref_q}\"* → Answer: *\"Strictly capped at 25 students for personalized mentorship.\"*")
            
            if st.button("⚡ Test Memory Cache Match", key="btn_test_cache"):
                # Compute term overlap / semantic estimation
                words_a = set(sim_q.lower().split())
                words_b = set(ref_q.lower().split())
                overlap = len(words_a.intersection(words_b)) / max(len(words_a), len(words_b))
                est_sim = min(98.0, max(50.0, overlap * 100 + 30))
                
                if est_sim >= 74:
                    st.success(f"🎯 **SEMANTIC CACHE HIT!** Estimated Similarity: `{est_sim:.1f}%` $\\ge 74\\%$")
                    st.markdown("""
                    <div style="background:rgba(6, 78, 59, 0.7); border:1px solid #10B981; padding:12px; border-radius:8px;">
                        <strong>⚡ Served instantly from local memory!</strong><br>
                        • LLM Tokens Consumed: <strong>0 tokens</strong><br>
                        • Tokens Saved: <strong>~1,850 tokens (100% API Credit Savings)</strong><br>
                        • Latency: <strong>0.02 seconds</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"Estimated similarity: `{est_sim:.1f}%` (< 74%). Routes to Tier 2: Relevant Memory Retrieval.")

        with st.expander("🗄️ **Session Memory vs. Long-Term User Memory**"):
            st.markdown(
                "• **Session Memory**: Ephemeral context maintained during the current browser session.\n"
                "• **Long-Term Memory**: Persistent multi-session memory and ChatGPT-style Custom Instructions stored in our SQLite portal database."
            )

    # =========================================================================
    # PILLAR 4: PROMPTING & GENERATION
    # =========================================================================
    with pillar_tabs[3]:
        st.markdown("### ⚙️ Pillar 4: Prompting & Generation (Guide, Generate, Stay Grounded)")
        st.write("How instructions, temperature, formatting, and grounding constraints steer LLM responses.")

        with st.expander("🌡️ **Temperature Control** — Deterministic vs. Creative Sampling", expanded=True):
            st.markdown(
                "**Definition**: `Temperature` controls the probability distribution over next-token generation. "
                "Low temperature ($0.0 - 0.3$) is strictly deterministic and factual (ideal for academic policies and syllabus verification). "
                "Higher temperature ($0.7 - 1.0$) increases stylistic variance."
            )
            temp_choice = st.slider("Select Sampling Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key="glo_temp_slider")
            
            if temp_choice <= 0.3:
                st.info(f"❄️ **Temperature {temp_choice} (Deterministic)**: The LLM selects the top token with highest probability. Minimal hallucination risk.")
            elif temp_choice <= 0.7:
                st.success(f"⚖️ **Temperature {temp_choice} (Balanced)**: Balanced tone, suitable for engaging counselling conversations.")
            else:
                st.warning(f"🔥 **Temperature {temp_choice} (High Variance)**: Highly creative responses, but elevated hallucination risk for factual course dates.")

        with st.expander("📝 **Grounding & Fallback Logic**"):
            st.markdown(
                "• **Grounding Prompt**: Enforces that the model only cites facts directly present in the retrieved ChromaDB chunks.\n"
                "• **No-Answer Fallback**: Instructs the model to transparently reply: *'Based on the verified syllabus, this detail is not specified'* rather than guessing."
            )

    # =========================================================================
    # PILLAR 5: KNOWLEDGE BASE QUALITY
    # =========================================================================
    with pillar_tabs[4]:
        st.markdown("### 📑 Pillar 5: Knowledge Base Quality (Good Knowledge, Better Answers)")
        st.write("A RAG system is only as good as the ground-truth data ingested into its vector database.")

        with st.expander("💎 **Knowledge Source Pillars**", expanded=True):
            st.markdown("""
            | Pillar Term | Production Best Practice | Implementation in SIN RAG |
            | :--- | :--- | :--- |
            | **Source Quality** | Use verified curriculum briefs and authoritative documents. | Ingested directly from official `info.md` specification. |
            | **Freshness of Data** | Prevent stale dates or deprecated course schedules. | Stamped with 2026 dates (October 1 start date). |
            | **Table Extraction** | Preserve markdown and HTML tabular structures. | Header-aware chunking preserving curriculum tables. |
            | **Trustworthy Sources** | Display clear source metadata with category tags. | Clickable source expanders under every chat response. |
            """)

    # =========================================================================
    # PILLAR 6: EVALUATION & OPTIMIZATION
    # =========================================================================
    with pillar_tabs[5]:
        st.markdown("### 📊 Pillar 6: Evaluation & Optimization (Measure, Learn, Improve)")
        st.write("RAG evaluation metrics measure whether retrieved context is faithfully reflected in generated answers.")

        with st.expander("📏 **Groundedness & Faithfulness Evaluator** — Live Tester", expanded=True):
            st.markdown(
                "**Definition**: Measures the proportion of assertions in an AI response that are directly backed by retrieved context chunks. "
                "High groundedness ($\\ge 85\\%$) indicates reliable grounding."
            )
            st.markdown("#### 🧪 Test Groundedness Evaluator")
            test_resp = st.text_area(
                "Sample AI Counsellor Response to Evaluate:",
                value="The AI Agent Xcelerator 2026 program features a 30% theory and 70% practical model with a 2-month internship.",
                key="glo_eval_resp"
            )
            test_kb = st.text_area(
                "Sample Knowledge Base Context:",
                value="AI Agent Xcelerator 2026 adheres to 30% theory + 70% practical learning. Students complete a 2-month industry internship with partner startups.",
                key="glo_eval_kb"
            )
            
            if st.button("📊 Evaluate Response Groundedness", key="btn_eval_ground"):
                eval_res = evaluate_groundedness(test_resp, [{"document": test_kb}])
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.metric("Groundedness Score", f"{eval_res['groundedness_score']}%")
                with col_e2:
                    st.metric("Status", eval_res["status"])
                st.caption("Algorithm: Evaluates lexical term overlap and verified claim alignment against source context.")
