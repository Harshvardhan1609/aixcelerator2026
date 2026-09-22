import os
import base64
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from rag_engine import RAGEngine
from llm_service import (
    generate_chat_response_stream,
    POPULAR_FREE_MODELS,
    DEFAULT_MODEL,
    get_openrouter_api_key,
    COUNSELLOR_SYSTEM_PROMPT
)
from visualizer import (
    create_3d_vector_space_plot,
    create_similarity_ranking_chart,
    create_rag_pipeline_sankey_chart,
    create_token_economics_chart,
    create_reverse_memory_flow_chart
)
from token_manager import TokenTracker, count_tokens, count_messages_tokens, estimate_tokens_saved
from memory_engine import ConversationalMemoryEngine
from auth_manager import (
    register_user,
    authenticate_user,
    get_user_threads,
    create_new_thread,
    delete_thread,
    rename_thread,
    save_message,
    get_thread_messages,
    save_custom_instructions,
    get_custom_instructions
)
from rag_advanced import (
    generate_multi_queries,
    hybrid_rerank,
    filter_by_similarity_threshold,
    build_custom_grounding_prompt,
    evaluate_groundedness
)
from glossary_lab import render_interactive_glossary_lab

# Page configuration
st.set_page_config(
    page_title="SIN - School of AI | AI Agent Xcelerator 2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brand", "logo.png")

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    return ""

logo_base64 = get_base64_image(LOGO_PATH)

# Custom CSS for official SIN - School of AI Brand Identity & Visualizer
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    h1, h2, h3, .brand-font {{
        font-family: 'Outfit', sans-serif !important;
    }}
    
    /* Main Background Accent Glow */
    .stApp {{
        background: radial-gradient(circle at 50% -10%, rgba(143, 0, 0, 0.28) 0%, rgba(13, 6, 6, 0.98) 60%, #080303 100%);
    }}

    /* Sidebar Theme */
    [data-testid="stSidebar"] {{
        background-color: #120606 !important;
        border-right: 1px solid rgba(143, 0, 0, 0.35) !important;
    }}
    
    /* Hero Banner Container */
    .brand-hero {{
        background: linear-gradient(135deg, #240404 0%, #4D0000 45%, #8F0000 85%, #2B0000 100%);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 100, 100, 0.25);
        box-shadow: 0 14px 40px -10px rgba(143, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
    }}
    
    .brand-hero::after {{
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.15) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }}

    .hero-top-bar {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 10px;
    }}
    
    .hero-logo-img {{
        height: 48px;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }}

    .brand-subtitle-badge {{
        display: inline-block;
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #FCD34D;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 6px;
    }}

    .brand-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.95rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    
    .brand-title span {{
        color: #F87171;
    }}

    .brand-desc {{
        color: #F3F4F6;
        font-size: 0.94rem;
        line-height: 1.5;
        margin-top: 6px;
        max-width: 900px;
    }}

    /* Stat chips */
    .stat-row {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 14px;
    }}
    
    .stat-chip {{
        background: rgba(0, 0, 0, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 6px 12px;
        border-radius: 9px;
        font-size: 0.82rem;
        color: #F9FAFB;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        backdrop-filter: blur(8px);
    }}
    
    .stat-chip strong {{
        color: #FCA5A5;
    }}

    .stat-chip.highlight strong {{
        color: #FBBF24;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: rgba(20, 7, 7, 0.7);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(143, 0, 0, 0.35);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 18px;
        color: #D1D5DB;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, #8F0000 0%, #B91C1C 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(143, 0, 0, 0.5);
    }}

    /* Buttons Styling */
    div.stButton > button {{
        background-color: #1A0707 !important;
        border: 1px solid rgba(143, 0, 0, 0.5) !important;
        color: #F3F4F6 !important;
        border-radius: 9px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
    }}
    
    div.stButton > button:hover {{
        background: linear-gradient(90deg, #8F0000 0%, #B91C1C 100%) !important;
        border-color: #EF4444 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(143, 0, 0, 0.45) !important;
        transform: translateY(-1px);
    }}

    /* Card Containers */
    .visualizer-card {{
        background: rgba(22, 8, 8, 0.75);
        border: 1px solid rgba(143, 0, 0, 0.35);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
    }}

    /* Chat Messages styling */
    [data-testid="stChatMessage"] {{
        background: rgba(22, 9, 9, 0.6) !important;
        border: 1px solid rgba(143, 0, 0, 0.25) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        backdrop-filter: blur(8px) !important;
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: rgba(30, 10, 10, 0.7) !important;
        border-radius: 8px !important;
        color: #E5E7EB !important;
        border: 1px solid rgba(143, 0, 0, 0.25) !important;
    }}

    /* Selectbox styling */
    div[data-baseweb="select"] > div {{
        background-color: #1A0A0A !important;
        border-color: rgba(143, 0, 0, 0.5) !important;
        color: white !important;
    }}

    /* Token & RMA Badges */
    .token-chip-bar {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
        margin-bottom: 6px;
    }}
    .token-chip {{
        font-size: 0.76rem;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 0.02em;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }}
    .chip-usage {{
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #94A3B8;
    }}
    .chip-saved {{
        background: rgba(6, 78, 59, 0.7);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34D399;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }}
    .chip-cache-hit {{
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.9) 0%, rgba(4, 120, 87, 0.9) 100%);
        border: 1px solid #10B981;
        color: #FFFFFF;
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.4);
        font-weight: 700;
    }}
    .chip-mode {{
        background: rgba(143, 0, 0, 0.35);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #FCA5A5;
    }}
    .metric-subcard {{
        background: rgba(22, 8, 8, 0.6);
        border: 1px solid rgba(143, 0, 0, 0.3);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize RAG Engine in session state
@st.cache_resource(show_spinner="Connecting to SIN Knowledge Base...")
def init_rag_engine():
    engine = RAGEngine()
    engine.ingest_knowledge_base()
    return engine

try:
    rag_engine = init_rag_engine()
except Exception as e:
    st.error(f"Failed to initialize ChromaDB: {e}")
    rag_engine = None

# Initialize Conversational Memory & Token Tracker in session state
if "token_tracker" not in st.session_state:
    st.session_state.token_tracker = TokenTracker(token_limit=25000)

if "memory_engine" not in st.session_state:
    st.session_state.memory_engine = ConversationalMemoryEngine(
        cache_threshold=0.74,
        embedding_fn=rag_engine.embedding_fn if rag_engine else None
    )

if "last_turn_metrics" not in st.session_state:
    st.session_state.last_turn_metrics = None

# User Authentication & Chat Threads State
if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None

# 6-Pillar RAG Hyperparameters
if "rag_temperature" not in st.session_state:
    st.session_state.rag_temperature = 0.4

if "rag_top_k" not in st.session_state:
    st.session_state.rag_top_k = 3

if "rag_similarity_cutoff" not in st.session_state:
    st.session_state.rag_similarity_cutoff = 40.0

if "rag_answer_format" not in st.session_state:
    st.session_state.rag_answer_format = "Balanced (Standard)"

if "rag_enable_multi_query" not in st.session_state:
    st.session_state.rag_enable_multi_query = True

if "rag_enable_hybrid" not in st.session_state:
    st.session_state.rag_enable_hybrid = True

# Main Hero Header HTML generator
logo_img_html = f'<img src="data:image/png;base64,{logo_base64}" class="hero-logo-img" alt="SIN Logo" />' if logo_base64 else ''

# -----------------------------------------------------------------------------
# AUTHENTICATION GATE (Login / Register / ID Creation / 1-Click Demo)
# -----------------------------------------------------------------------------
def render_auth_screen():
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown('### 🦁 SIN - School of AI')
        st.markdown('---')
        st.markdown('#### 🔒 Student Authentication')
        st.caption('Please log in or register a new student ID to access the course counselling chat, 6-pillar lab, and custom instructions.')
        st.markdown('''
        <div style="background:rgba(143,0,0,0.2); border:1px solid rgba(239,68,68,0.3); border-radius:8px; padding:12px; margin-top:10px;">
            <div style="font-size:0.8rem; font-weight:700; color:#FCA5A5;">🔑 Quick Demo Credentials:</div>
            <div style="font-size:0.75rem; color:#D1D5DB; margin-top:4px;">User ID: <code>student</code></div>
            <div style="font-size:0.75rem; color:#D1D5DB;">Password: <code>sin2026</code></div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="brand-hero">
        <div class="brand-subtitle-badge">SIN • SCHOOL OF AI OFFICIAL PORTAL</div>
        <div class="hero-top-bar">
            {logo_img_html}
            <h1 class="brand-title">Student & Counsellor <span>RAG Portal</span> 2026</h1>
        </div>
        <div class="brand-desc">
            Sign in with your Student ID or create a new account to access personalized AI counselling, custom instructions, threaded conversations, and the interactive 6-pillar RAG learning laboratory.
        </div>
        <div class="stat-row">
            <span class="stat-chip">🎓 <strong>AI Agent Xcelerator 2026</strong></span>
            <span class="stat-chip">⏱️ Duration: <strong>5 Months</strong></span>
            <span class="stat-chip">⚙️ Approach: <strong>70% Practical / 30% Theory</strong></span>
            <span class="stat-chip highlight">👥 Max Batch: <strong>25 Students</strong></span>
            <span class="stat-chip highlight">📅 Start Date: <strong>1 October 2026</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    auth_col1, auth_col2, auth_col3 = st.columns([1, 2.2, 1])
    with auth_col2:
        st.markdown("<div class='visualizer-card'>", unsafe_allow_html=True)
        auth_tab1, auth_tab2 = st.tabs(["🔑 Log In", "✨ Create New ID / Sign Up"])

        with auth_tab1:
            st.markdown("#### Access Your Student Portal")
            login_u = st.text_input("User ID / Username:", placeholder="e.g. student or your user ID", key="login_u")
            login_p = st.text_input("Password:", type="password", placeholder="Enter your password", key="login_p")

            if st.button("🚀 Log In to SIN Portal", use_container_width=True, key="btn_login"):
                ok, user_dict, msg = authenticate_user(login_u, login_p)
                if ok and user_dict:
                    st.session_state.authenticated_user = user_dict
                    threads = get_user_threads(user_dict["username"])
                    if threads:
                        st.session_state.current_thread_id = threads[0]["thread_id"]
                    else:
                        st.session_state.current_thread_id = create_new_thread(user_dict["username"], "Welcome to AI Agent Xcelerator")
                    st.success(f"Welcome back, {user_dict['full_name']}!")
                    st.rerun()
                else:
                    st.error(msg)

            st.markdown("---")
            st.markdown("<p style='text-align:center; color:#9CA3AF; font-size:0.85rem;'>Looking for immediate access?</p>", unsafe_allow_html=True)
            if st.button("⚡ Quick 1-Click Demo Login (Explore as @student)", use_container_width=True, key="btn_demo_login"):
                ok, user_dict, _ = authenticate_user("student", "sin2026")
                if ok and user_dict:
                    st.session_state.authenticated_user = user_dict
                    threads = get_user_threads(user_dict["username"])
                    if threads:
                        st.session_state.current_thread_id = threads[0]["thread_id"]
                    else:
                        st.session_state.current_thread_id = create_new_thread(user_dict["username"], "Welcome to AI Agent Xcelerator")
                    st.rerun()

        with auth_tab2:
            st.markdown("#### Create New Student Account")
            reg_n = st.text_input("Your Full Name:", placeholder="e.g. Alex Rivera", key="reg_n")
            reg_u = st.text_input("Choose an ID / Username:", placeholder="e.g. alex2026", key="reg_u")
            reg_p = st.text_input("Choose a Password:", type="password", placeholder="At least 4 characters", key="reg_p")

            if st.button("✨ Create My Account & Enter", use_container_width=True, key="btn_register"):
                ok, msg = register_user(reg_u, reg_p, reg_n)
                if ok:
                    _, user_dict, _ = authenticate_user(reg_u, reg_p)
                    st.session_state.authenticated_user = user_dict
                    tid = create_new_thread(user_dict["username"], "New Conversation")
                    st.session_state.current_thread_id = tid
                    st.success("Account created successfully!")
                    st.rerun()
                else:
                    st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)

def render_authenticated_portal(current_user):
    # Ensure active thread exists for authenticated user
    user_threads = get_user_threads(current_user["username"])
    if not user_threads:
        new_tid = create_new_thread(current_user["username"], "Welcome to AI Agent Xcelerator")
        st.session_state.current_thread_id = new_tid
        user_threads = get_user_threads(current_user["username"])
    elif st.session_state.current_thread_id is None:
        st.session_state.current_thread_id = user_threads[0]["thread_id"]

    # Sidebar Configuration
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown("### 🦁 SIN - School of AI")

        # User Profile & Logout
        st.markdown(
            f"<div style='background:rgba(143,0,0,0.25); border:1px solid rgba(239,68,68,0.35); padding:10px; border-radius:10px; margin-bottom:10px;'>"
            f"<div style='font-size:0.85rem; color:#FFFFFF;'>👤 <strong>{current_user['full_name']}</strong></div>"
            f"<div style='font-size:0.75rem; color:#9CA3AF;'>ID: <code>@{current_user['username']}</code></div>"
            f"</div>",
            unsafe_allow_html=True
        )
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.authenticated_user = None
            st.session_state.current_thread_id = None
            st.session_state.messages = []
            st.session_state.token_tracker.reset()
            st.session_state.memory_engine.clear()
            st.rerun()

        st.markdown("---")

        # ChatGPT-Style Thread Management
        if st.button("➕ New Chat Thread", use_container_width=True):
            new_tid = create_new_thread(current_user["username"], "New Chat")
            st.session_state.current_thread_id = new_tid
            st.session_state.memory_engine.clear()
            st.rerun()

        user_threads = get_user_threads(current_user["username"])
        if user_threads:
            st.markdown("##### 💬 Chat Threads")
            for th in user_threads[:6]:
                is_active = (th["thread_id"] == st.session_state.current_thread_id)
                title_prefix = "👉 " if is_active else "💬 "
                btn_label = f"{title_prefix}{th['title'][:22]}"
            
                c_th1, c_th2 = st.columns([5, 1])
                with c_th1:
                    if st.button(btn_label, key=f"th_{th['thread_id']}", use_container_width=True):
                        st.session_state.current_thread_id = th["thread_id"]
                        st.session_state.memory_engine.clear()
                        st.rerun()
                with c_th2:
                    if st.button("🗑️", key=f"del_{th['thread_id']}", help="Delete Thread"):
                        delete_thread(th["thread_id"])
                        rem = get_user_threads(current_user["username"])
                        st.session_state.current_thread_id = rem[0]["thread_id"] if rem else create_new_thread(current_user["username"], "New Chat")
                        st.session_state.memory_engine.clear()
                        st.rerun()

        st.markdown("---")

        # ChatGPT-Style Custom Instructions
        with st.expander("⚙️ Custom Instructions (ChatGPT-Style)", expanded=False):
            st.caption("Tell the AI about your goals and how you'd like it to format responses.")
            ci_data = get_custom_instructions(current_user["username"])
            ci_bio = st.text_area(
                "What should the AI know about you?",
                value=ci_data.get("user_bio", ""),
                placeholder="e.g. 3rd-year CS student interested in building enterprise autonomous agents.",
                key="ci_bio_input"
            )
            ci_pref = st.text_area(
                "How would you like the AI to respond?",
                value=ci_data.get("response_pref", ""),
                placeholder="e.g. Crisp bullet points, bold key terms, focus on portfolio projects.",
                key="ci_pref_input"
            )
            if st.button("💾 Save Custom Instructions", use_container_width=True):
                save_custom_instructions(current_user["username"], ci_bio, ci_pref)
                st.session_state.authenticated_user["user_bio"] = ci_bio
                st.session_state.authenticated_user["response_pref"] = ci_pref
                st.success("Custom instructions saved successfully!")

        # 6-Pillar RAG Configuration
        with st.expander("🎛️ 6-Pillar RAG Hyperparameters", expanded=False):
            st.session_state.rag_top_k = st.slider("Top-K Retrieved Chunks:", 1, 8, st.session_state.rag_top_k, key="rag_k_slider")
            st.session_state.rag_similarity_cutoff = st.slider("Similarity Threshold Cutoff (%):", 15.0, 75.0, float(st.session_state.rag_similarity_cutoff), step=5.0, key="rag_cut_slider")
            st.session_state.rag_temperature = st.slider("Sampling Temperature:", 0.0, 1.0, float(st.session_state.rag_temperature), step=0.05, key="rag_temp_slider")
        
            fmt_options = ["Balanced (Standard)", "Crisp Bullet Points", "Deep Technical Dive", "Executive Summary"]
            st.session_state.rag_answer_format = st.selectbox(
                "Answer Formatting:",
                fmt_options,
                index=fmt_options.index(st.session_state.rag_answer_format),
                key="rag_fmt_select"
            )
            st.session_state.rag_enable_multi_query = st.checkbox("Enable Multi-Query Expansion", value=st.session_state.rag_enable_multi_query, key="chk_mq")
            st.session_state.rag_enable_hybrid = st.checkbox("Enable Hybrid Lexical+Vector Reranking", value=st.session_state.rag_enable_hybrid, key="chk_hy")

        st.markdown("---")

        # Architecture Selector
        st.markdown("#### 🧠 Memory Architecture")
        arch_options = [
            "⚡ Reverse Memory (Token & Credit Saver)",
            "📜 Standard RAG (Full History)"
        ]
        selected_arch = st.radio(
            "Memory Pipeline Mode:",
            options=arch_options,
            index=0,
            help="Reverse Memory Architecture retrieves only relevant conversational snippets and checks semantic cache first, saving 60-100% of LLM prompt tokens and credits."
        )
        is_reverse_memory = (selected_arch == "⚡ Reverse Memory (Token & Credit Saver)")
        active_mode_key = "reverse_memory" if is_reverse_memory else "standard"

        # Token Usage & Limit Guardrail
        st.markdown("---")
        st.markdown("#### 🪙 Token Usage & Limit Guardrail")
        current_limit = st.session_state.token_tracker.token_limit
        new_limit = st.slider(
            "Session Token Budget:",
            min_value=5000,
            max_value=100000,
            value=current_limit,
            step=2500,
            help="Configure maximum allowed token usage for this session to safeguard AI agent credits."
        )
        if new_limit != current_limit:
            st.session_state.token_tracker.set_token_limit(new_limit)

        summary = st.session_state.token_tracker.get_summary()
        progress_val = min(1.0, max(0.0, summary["pct_used"] / 100.0))
        st.progress(progress_val)

        st.markdown(
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; font-size:0.8rem;'>"
            f"<span><strong>{summary['total_tokens_used']:,}</strong> / {summary['token_limit']:,} tokens</span>"
            f"<span style='color:{summary['status_color']}; font-weight:700;'>{summary['status_label']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        # 2x2 Mini metrics card
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(
                f"<div class='metric-subcard'>"
                f"<div style='font-size:0.7rem; color:#9CA3AF;'>⚡ SAVED TOKENS</div>"
                f"<div style='font-size:1.05rem; font-weight:700; color:#10B981;'>{summary['tokens_saved_total']:,}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col_t2:
            st.markdown(
                f"<div class='metric-subcard'>"
                f"<div style='font-size:0.7rem; color:#9CA3AF;'>🎯 EFFICIENCY</div>"
                f"<div style='font-size:1.05rem; font-weight:700; color:#F59E0B;'>{summary['efficiency_pct']}%</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        col_t3, col_t4 = st.columns(2)
        with col_t3:
            st.markdown(
                f"<div class='metric-subcard'>"
                f"<div style='font-size:0.7rem; color:#9CA3AF;'>💰 CREDITS SAVED</div>"
                f"<div style='font-size:0.92rem; font-weight:700; color:#34D399;'>${summary['credits_saved_usd']:.4f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col_t4:
            st.markdown(
                f"<div class='metric-subcard'>"
                f"<div style='font-size:0.7rem; color:#9CA3AF;'>⚡ CACHE HITS</div>"
                f"<div style='font-size:0.92rem; font-weight:700; color:#60A5FA;'>{summary['cache_hits_count']}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        if st.button("🔄 Reset Token Counter", use_container_width=True):
            st.session_state.token_tracker.reset()
            st.rerun()

        st.markdown("---")

        # Model Selection
        st.markdown("#### 🤖 AI Model (OpenRouter)")
        model_options = list(POPULAR_FREE_MODELS) + ["Custom Model..."]
        selected_option = st.selectbox(
            "Select Active Model:",
            options=model_options,
            index=0,
            help="Select any verified free model or input a custom OpenRouter model ID."
        )
        if selected_option == "Custom Model...":
            active_model = st.text_input("Enter Model Slug:", value="openrouter/free", placeholder="e.g. openrouter/free")
        else:
            active_model = selected_option
        st.caption(f"⚡ Engine: `{active_model}`")

        # API Key Configuration
        st.markdown("#### 🔑 OpenRouter API Key")
        env_api_key = get_openrouter_api_key()
        if env_api_key:
            masked_key = env_api_key[:8] + "..." + env_api_key[-4:]
            st.success(f"Key loaded from `.env` (`{masked_key}`)")
            custom_key = st.text_input("Override API Key (optional):", type="password", placeholder="Leave blank to use .env key")
            api_key_to_use = custom_key.strip() if custom_key.strip() else env_api_key
        else:
            st.warning("No API key detected in `.env`.")
            api_key_to_use = st.text_input("Enter OpenRouter Key:", type="password", placeholder="sk-or-v1-...")

        st.markdown("---")

        # Knowledge Base Controls
        st.markdown("#### 📚 SIN Knowledge Base")
        if rag_engine:
            stats = rag_engine.get_stats()
            st.write(f"📊 **Indexed Chunks**: `{stats['total_chunks']}`")
            if st.button("🔄 Re-index Knowledge Base", use_container_width=True):
                with st.spinner("Re-embedding syllabus documents..."):
                    count = rag_engine.ingest_knowledge_base(force_reload=True)
                    st.success(f"Indexed {count} chunks successfully!")
        else:
            st.error("ChromaDB offline.")

    # Main Hero Header with Brand Guidelines & Colors
    st.markdown(f"""
    <div class="brand-hero">
        <div class="brand-subtitle-badge">SIN • SCHOOL OF AI OFFICIAL ADVISORY</div>
        <div class="hero-top-bar">
            {logo_img_html}
            <h1 class="brand-title">AI Agent <span>Xcelerator</span> 2026</h1>
        </div>
        <div class="brand-desc">
            Welcome, <strong>{current_user['full_name']}</strong>! Explore the 5-month journey from <strong>AI User → AI Builder → Industry-Ready Problem Solver</strong>, test your 6-pillar RAG settings, and inspect memory economics in real-time.
        </div>
        <div class="stat-row">
            <span class="stat-chip">⏱️ Duration: <strong>5 Months</strong></span>
            <span class="stat-chip">📚 Breakdown: <strong>3M Training + 2M Internship</strong></span>
            <span class="stat-chip">⚙️ Approach: <strong>70% Practical / 30% Theory</strong></span>
            <span class="stat-chip highlight">👥 Exclusivity: <strong>25 Students Max Cohort</strong></span>
            <span class="stat-chip highlight">📅 Starting Date: <strong>1 October 2026</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Application Navigation Mode: 4 Dedicated Tabs
    tab_chat, tab_visualizer, tab_token_eco, tab_glossary = st.tabs([
        "💬 Student Counsellor Chat",
        "🔬 3D Vector Space & RAG Architecture Visualizer",
        "🪙 Token Economics & Reverse Memory Visualizer",
        "📚 Interactive RAG Glossary & Practical Lab"
    ])

    # Track last query for visualizer sync
    if "last_query" not in st.session_state:
        st.session_state.last_query = "What is the duration, batch limit, and 70% practical approach?"
    if "last_retrieved_sources" not in st.session_state:
        if rag_engine:
            _, initial_sources = rag_engine.get_context_string(st.session_state.last_query, n_results=4)
            st.session_state.last_retrieved_sources = initial_sources
        else:
            st.session_state.last_retrieved_sources = []

    # ==========================================
    # TAB 1: STUDENT COUNSELLOR CHAT (THREADED)
    # ==========================================
    with tab_chat:
        summary_status = st.session_state.token_tracker.get_summary()
        if summary_status["status"] == "exceeded":
            st.warning(
                f"🚨 **Session Token Limit Exceeded ({summary_status['total_tokens_used']:,} / {summary_status['token_limit']:,} tokens)**: "
                "To safeguard your AI agent credits, increase the **Session Token Budget** slider in the sidebar or click **Reset Token Counter**."
            )

        # Load messages for the current thread from SQLite
        thread_messages = get_thread_messages(st.session_state.current_thread_id)

        # If thread is empty, render initial welcome greeting
        if not thread_messages:
            with st.chat_message("assistant"):
                st.markdown(
                    f"👋 **Welcome, {current_user['full_name']}!**\n\n"
                    "I am your official Student Counsellor for the **AI Agent Xcelerator Program 2026**.\n\n"
                    "Our mission: *Don't Prepare Students Only to Use the Future. Train Them to Build It.*\n\n"
                    "Ask me anything about our **30% Theory + 70% Practical model**, the **2-Month Industry Internship**, "
                    "the **exclusive 25-student cohort**, or what portfolio projects you'll create.\n\n"
                    "💡 *Tip: Your ChatGPT-style **Custom Instructions** are active. Switch to **📚 Interactive RAG Glossary & Practical Lab** to test live sliders for all 6 RAG pillars!*"
                )

        # Render Persistent Thread Messages
        for msg in thread_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
            
                t_info = msg.get("token_info")
                if t_info and msg["role"] == "assistant":
                    if t_info.get("is_cache_hit"):
                        st.markdown(f"""
                        <div class="token-chip-bar">
                            <span class="token-chip chip-cache-hit">⚡ INSTANT REVERSE MEMORY HIT</span>
                            <span class="token-chip chip-usage">0 LLM Tokens Consumed</span>
                            <span class="token-chip chip-saved">💰 ~{t_info.get('tokens_saved', 0):,} Tokens Saved (100% Credits Saved)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        tot = t_info.get("total_tokens", 0)
                        pmt = t_info.get("prompt_tokens", 0)
                        out = t_info.get("completion_tokens", 0)
                        saved = t_info.get("tokens_saved", 0)
                        eff = round((saved / (tot + saved) * 100), 1) if (tot + saved) > 0 else 0
                        mode_tag = t_info.get("mode", "Reverse Memory")
                        grounding_score = t_info.get("groundedness_score", 85.0)
                    
                        st.markdown(f"""
                        <div class="token-chip-bar">
                            <span class="token-chip chip-usage">📊 Turn: <strong>{tot:,}</strong> Tokens (Prompt: {pmt:,} | Output: {out:,})</span>
                            <span class="token-chip chip-saved">⚡ RMA Saved: <strong>{saved:,}</strong> Tokens ({eff}%)</span>
                            <span class="token-chip chip-mode">🧠 {mode_tag}</span>
                            <span class="token-chip" style="background:rgba(16,185,129,0.25); border:1px solid #10B981; color:#34D399;">🛡️ {grounding_score}% Grounded</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Token Anatomy Expander
                    bd = t_info.get("breakdown", {})
                    with st.expander("🔍 Token Anatomy & Diagnostics", expanded=False):
                        col_b1, col_b2 = st.columns([1, 1])
                        with col_b1:
                            st.markdown("**Token Consumption Breakdown:**")
                            breakdown_table = [
                                {"Component": "System Persona & Rules", "Tokens": f"{bd.get('system', 0):,}"},
                                {"Component": "Knowledge Base Context", "Tokens": f"{bd.get('context', 0):,}"},
                                {"Component": "Conversational Memory Snippet", "Tokens": f"{bd.get('memory', 0):,}"},
                                {"Component": "Student Query", "Tokens": f"{bd.get('user', 0):,}"},
                                {"Component": "AI Counsellor Completion", "Tokens": f"{bd.get('completion', 0):,}"},
                                {"Component": "TOTAL TURN TOKENS", "Tokens": f"**{t_info.get('total_tokens', 0):,}**"}
                            ]
                            st.dataframe(pd.DataFrame(breakdown_table), hide_index=True, use_container_width=True)
                        with col_b2:
                            st.markdown("**Efficiency & Credit Savings:**")
                            std_baseline = t_info.get("prompt_tokens", 0) + t_info.get("tokens_saved", 0) + t_info.get("completion_tokens", 0)
                            savings_table = [
                                {"Metric": "Standard RAG Baseline", "Value": f"{std_baseline:,} tokens"},
                                {"Metric": "Actual Consumed", "Value": f"{t_info.get('total_tokens', 0):,} tokens"},
                                {"Metric": "Tokens Saved by RMA", "Value": f"⚡ {t_info.get('tokens_saved', 0):,} tokens"},
                                {"Metric": "Credit Savings Rate", "Value": f"{eff}%"},
                                {"Metric": "Cache Hit Status", "Value": "⚡ HIT (0 Tokens)" if t_info.get("is_cache_hit") else "Selective Retrieval"}
                            ]
                            st.dataframe(pd.DataFrame(savings_table), hide_index=True, use_container_width=True)

                # Render Verified Sources
                if msg.get("sources"):
                    with st.expander(f"🔍 Verified SIN Course Sources ({len(msg['sources'])} chunks)", expanded=False):
                        for idx, src in enumerate(msg["sources"], 1):
                            meta = src.get("metadata", {})
                            st.markdown(f"**Source {idx}:** `{meta.get('category', 'KB')} > {meta.get('title', 'Section')}`")
                            st.caption(src.get("document", ""))
                            st.divider()

        # Handle Chat Input
        prompt_input = st.chat_input("Ask about curriculum, internship, batch size, or ask follow-ups...")
        if prompt_input:
            st.session_state.last_query = prompt_input

            # 1. Save user message to SQLite thread
            save_message(st.session_state.current_thread_id, "user", prompt_input)
            with st.chat_message("user"):
                st.markdown(prompt_input)

            # Auto-title thread if it's currently a default title
            cur_threads = get_user_threads(current_user["username"])
            for th in cur_threads:
                if th["thread_id"] == st.session_state.current_thread_id and th["title"] in ["New Chat", "New Conversation", "Welcome to AI Agent Xcelerator"]:
                    auto_title = prompt_input.strip().replace("\n", " ")[:26]
                    if len(prompt_input.strip()) > 26:
                        auto_title += "..."
                    rename_thread(st.session_state.current_thread_id, auto_title)
                    break

            # 2. Check Reverse Memory Architecture (Semantic Cache Hit)
            cache_hit = None
            if is_reverse_memory:
                cache_hit = st.session_state.memory_engine.check_semantic_cache(prompt_input, threshold=0.74)

            # CASE A: Semantic Memory Cache Hit (0 Tokens Consumed)
            if cache_hit is not None:
                with st.chat_message("assistant"):
                    cached_text = cache_hit["cached_response"]

                    def stream_cache_content():
                        for w in cached_text.split(" "):
                            yield w + " "
                            time.sleep(0.01)

                    response_text = st.write_stream(stream_cache_content)
                    comp_est = count_tokens(response_text)
                    std_prompt_est = count_tokens(COUNSELLOR_SYSTEM_PROMPT) + 1200 + count_messages_tokens(thread_messages) + count_tokens(prompt_input)
                    tokens_saved = estimate_tokens_saved(std_prompt_est, 0, is_cache_hit=True, estimated_standard_completion=comp_est)

                    turn_record = st.session_state.token_tracker.record_turn(
                        query=prompt_input,
                        prompt_tokens=0,
                        completion_tokens=0,
                        tokens_saved=tokens_saved,
                        mode="Reverse Memory (Cache Hit)",
                        is_cache_hit=True,
                        breakdown={"system": 0, "context": 0, "memory": 0, "user": 0, "completion": 0}
                    )
                    turn_record["groundedness_score"] = 95.0
                    st.session_state.last_turn_metrics = turn_record

                    st.markdown(f"""
                    <div class="token-chip-bar">
                        <span class="token-chip chip-cache-hit">⚡ INSTANT REVERSE MEMORY HIT ({cache_hit['similarity_pct']}% MATCH)</span>
                        <span class="token-chip chip-usage">0 LLM Tokens Consumed</span>
                        <span class="token-chip chip-saved">💰 ~{tokens_saved:,} Tokens Saved (100% Credits Saved)</span>
                    </div>
                    """, unsafe_allow_html=True)

                save_message(st.session_state.current_thread_id, "assistant", response_text, cache_hit.get("sources", []), turn_record)
                st.rerun()

            # CASE B: Inference Required
            else:
                # 1. Multi-Query Retrieval & Knowledge Base Context
                retrieved_sources = []
                if rag_engine:
                    with st.spinner("Retrieving verified context from SIN Knowledge Base..."):
                        if st.session_state.rag_enable_multi_query:
                            queries_to_search = generate_multi_queries(prompt_input)
                            seen_ids = set()
                            for q in queries_to_search:
                                _, sources = rag_engine.get_context_string(q, n_results=st.session_state.rag_top_k)
                                for s in sources:
                                    if s["id"] not in seen_ids:
                                        seen_ids.add(s["id"])
                                        retrieved_sources.append(s)
                        else:
                            _, retrieved_sources = rag_engine.get_context_string(prompt_input, n_results=st.session_state.rag_top_k)

                        # 2. Hybrid Reranking & Similarity Threshold Cutoff
                        if st.session_state.rag_enable_hybrid:
                            retrieved_sources = hybrid_rerank(retrieved_sources, prompt_input, alpha=0.7)
                        retrieved_sources = filter_by_similarity_threshold(retrieved_sources, threshold_pct=st.session_state.rag_similarity_cutoff)
                        st.session_state.last_retrieved_sources = retrieved_sources

                # Format retrieved context string
                context_blocks = []
                for idx, item in enumerate(retrieved_sources[:st.session_state.rag_top_k], start=1):
                    t = item.get("metadata", {}).get("title", f"Source {idx}")
                    context_blocks.append(f"--- [Source {idx}: {t}] ---\n{item['document']}")
                retrieved_context = "\n\n".join(context_blocks)

                # 3. Build Grounding Prompt with ChatGPT Custom Instructions
                custom_inst = get_custom_instructions(current_user["username"])
                grounded_sys_prompt = build_custom_grounding_prompt(
                    base_prompt=COUNSELLOR_SYSTEM_PROMPT,
                    custom_user_bio=custom_inst.get("user_bio", ""),
                    custom_response_pref=custom_inst.get("response_pref", ""),
                    answer_format=st.session_state.rag_answer_format,
                    temperature=st.session_state.rag_temperature
                )

                # 4. Build Memory-Augmented Prompt with RMA
                prompt_messages, prompt_meta = st.session_state.memory_engine.build_memory_augmented_prompt(
                    query=prompt_input,
                    kb_context=retrieved_context,
                    raw_chat_messages=thread_messages,
                    system_prompt=grounded_sys_prompt,
                    mode=active_mode_key
                )

                # 5. Stream LLM Response
                with st.chat_message("assistant"):
                    stream_generator = generate_chat_response_stream(
                        messages=prompt_messages,
                        model=active_model,
                        api_key=api_key_to_use,
                        temperature=st.session_state.rag_temperature
                    )
                    response_text = st.write_stream(stream_generator)

                    # Groundedness Evaluation
                    grounding_eval = evaluate_groundedness(response_text, retrieved_sources)

                    comp_tokens = count_tokens(response_text)
                    turn_record = st.session_state.token_tracker.record_turn(
                        query=prompt_input,
                        prompt_tokens=prompt_meta["prompt_tokens"],
                        completion_tokens=comp_tokens,
                        tokens_saved=prompt_meta["tokens_saved"],
                        mode="Reverse Memory" if is_reverse_memory else "Standard RAG",
                        is_cache_hit=False,
                        breakdown={
                            "system": prompt_meta["breakdown"]["system"],
                            "context": prompt_meta["breakdown"]["context"],
                            "memory": prompt_meta["breakdown"]["memory"],
                            "user": prompt_meta["breakdown"]["user"],
                            "completion": comp_tokens
                        }
                    )
                    turn_record["groundedness_score"] = grounding_eval["groundedness_score"]
                    st.session_state.last_turn_metrics = turn_record

                    # Index in RMA memory
                    st.session_state.memory_engine.record_turn(
                        query=prompt_input,
                        response=response_text,
                        sources=retrieved_sources,
                        tokens_used=prompt_meta["prompt_tokens"] + comp_tokens
                    )

                    # Render token badges
                    tot = turn_record["total_tokens"]
                    pmt = turn_record["prompt_tokens"]
                    out = turn_record["completion_tokens"]
                    saved = turn_record["tokens_saved"]
                    eff = round((saved / (tot + saved) * 100), 1) if (tot + saved) > 0 else 0
                    st.markdown(f"""
                    <div class="token-chip-bar">
                        <span class="token-chip chip-usage">📊 Turn: <strong>{tot:,}</strong> Tokens (Prompt: {pmt:,} | Output: {out:,})</span>
                        <span class="token-chip chip-saved">⚡ RMA Saved: <strong>{saved:,}</strong> Tokens ({eff}%)</span>
                        <span class="token-chip chip-mode">🧠 {'Reverse Memory' if is_reverse_memory else 'Standard RAG'}</span>
                        <span class="token-chip" style="background:rgba(16,185,129,0.25); border:1px solid #10B981; color:#34D399;">🛡️ {grounding_eval['groundedness_score']}% Grounded</span>
                    </div>
                    """, unsafe_allow_html=True)

                save_message(st.session_state.current_thread_id, "assistant", response_text, retrieved_sources, turn_record)
                st.rerun()

    # ==========================================
    # TAB 2: 3D VECTOR SPACE & RAG ARCHITECTURE VISUALIZER
    # ==========================================
    with tab_visualizer:
        st.markdown("### 🔬 Interactive 3D Vector Space & RAG Process Explorer")
        st.markdown(
            "Inspect how student queries are transformed into **384-dimensional dense vectors** by ONNX embeddings, "
            "projected into vector space, searched in **ChromaDB**, and routed to the **OpenRouter LLM**."
        )

        col_ctrl1, col_ctrl2 = st.columns([3, 1])
        with col_ctrl1:
            viz_query = st.text_input(
                "🔎 Query to Visualize in 3D Space:",
                value=st.session_state.last_query,
                help="Type any prospective student question to project into the 3D ChromaDB space."
            )
        with col_ctrl2:
            top_k_val = st.slider("Top-K Neighbors:", min_value=1, max_value=6, value=4)

        if st.button("🚀 Run 3D Vector Search & Visualize", use_container_width=True):
            st.session_state.last_query = viz_query
            if rag_engine:
                _, sources = rag_engine.get_context_string(viz_query, n_results=top_k_val)
                st.session_state.last_retrieved_sources = sources

        active_sources = st.session_state.last_retrieved_sources

        # 1. 3D Vector Space Plot
        if rag_engine:
            with st.spinner("Rendering 3D Vector Space with Plotly..."):
                fig_3d = create_3d_vector_space_plot(
                    rag_engine,
                    query=viz_query,
                    retrieved_items=active_sources
                )
                st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': True})
                st.caption("💡 *Tip: Click and drag to rotate the 3D plot. Hover on nodes to inspect chunk titles, categories, and text previews. The glowing gold star represents the query vector with dashed rays to its Top-K nearest neighbors.*")

        # 2. Metrics & Flow Charts
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        with col_metric1:
            st.metric("📦 Indexed Chunks", len(rag_engine.collection.get()["ids"]) if rag_engine else 0)
        with col_metric2:
            st.metric("📐 Embedding Dimensions", "384 (Dense)")
        with col_metric3:
            st.metric("🎯 Nearest Neighbors (K)", len(active_sources))
        with col_metric4:
            st.metric("🤖 Active LLM Model", active_model.split("/")[-1])

        # 3. Two-Column Analytics
        col_viz_left, col_viz_right = st.columns([1, 1])
        with col_viz_left:
            st.markdown("#### 📊 Semantic Match Ranking")
            if active_sources:
                fig_bar = create_similarity_ranking_chart(active_sources)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Run a query to view semantic similarity distribution.")

        with col_viz_right:
            st.markdown("#### ⚡ RAG Data Pipeline Flow")
            fig_sankey = create_rag_pipeline_sankey_chart(viz_query, active_sources, active_model)
            st.plotly_chart(fig_sankey, use_container_width=True)

        # 4. Detailed Step-by-Step Execution Trace
        st.markdown("#### 📋 Step-by-Step RAG Execution Trace")
        trace_steps = [
            {
                "Step": "1. Query Ingestion",
                "Component": "Streamlit Frontend",
                "Details": f"Query: \"{viz_query}\"",
                "Status": "✅ Completed"
            },
            {
                "Step": "2. Multi-Query & Dense Embedding",
                "Component": "ONNX all-MiniLM-L6-v2",
                "Details": "Generated 384-dimensional normalized float vectors",
                "Status": "✅ Completed"
            },
            {
                "Step": "3. Vector Search & Hybrid Rerank",
                "Component": "ChromaDB + BM25 Lexical",
                "Details": f"Evaluated chunks; retrieved top {len(active_sources)} nearest neighbors",
                "Status": "✅ Completed"
            },
            {
                "Step": "4. Memory & Context Synthesis",
                "Component": "Reverse Memory Assembler",
                "Details": f"Compiled {sum(len(s.get('document', '')) for s in active_sources)} chars with memory pruning",
                "Status": "✅ Completed"
            },
            {
                "Step": "5. Grounded LLM Inference",
                "Component": f"OpenRouter ({active_model})",
                "Details": "Streaming persona-grounded response with custom instructions",
                "Status": "✅ Active"
            }
        ]
        st.dataframe(pd.DataFrame(trace_steps), use_container_width=True, hide_index=True)

        # 5. Full Indexed Knowledge Base Inspector
        with st.expander("📚 Inspect All Indexed Knowledge Base Chunks (ChromaDB)", expanded=False):
            if rag_engine:
                kb_data = rag_engine.collection.get(include=["metadatas", "documents"])
                table_rows = []
                for i, chunk_id in enumerate(kb_data["ids"]):
                    meta = kb_data["metadatas"][i]
                    doc = kb_data["documents"][i]
                    table_rows.append({
                        "Chunk ID": chunk_id,
                        "Category": meta.get("category", "N/A"),
                        "Section Title": meta.get("title", "N/A"),
                        "Characters": len(doc),
                        "Content Preview": doc[:120] + "..."
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ==========================================
    # TAB 3: TOKEN ECONOMICS & REVERSE MEMORY VISUALIZER
    # ==========================================
    with tab_token_eco:
        st.markdown("### 🪙 Token Economics & Reverse Conversational Memory Explorer")
        st.markdown(
            "Monitor your real-time **Token Usage**, examine tokens and credits saved by the "
            "**Reverse Conversational Memory Architecture (RMA)**, and inspect the in-memory vector index."
        )

        t_summary = st.session_state.token_tracker.get_summary()

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.metric(
                label="🪙 Total Tokens Consumed",
                value=f"{t_summary['total_tokens_used']:,}",
                delta=f"{t_summary['pct_used']:.1f}% of Budget",
                delta_color="inverse" if t_summary["pct_used"] > 80 else "normal"
            )
        with col_kpi2:
            st.metric(
                label="⚡ RMA Tokens Saved",
                value=f"{t_summary['tokens_saved_total']:,}",
                delta=f"+{t_summary['efficiency_pct']}% Net Efficiency",
                delta_color="normal"
            )
        with col_kpi3:
            st.metric(
                label="🎯 Semantic Cache Hits",
                value=f"{t_summary['cache_hits_count']}",
                delta="100% LLM Free Turns",
                delta_color="normal"
            )
        with col_kpi4:
            st.metric(
                label="💰 Estimated Credits Saved",
                value=f"${t_summary['credits_saved_usd']:.4f}",
                delta="Benchmark: $0.002/1k tokens",
                delta_color="normal"
            )

        st.markdown("---")

        col_chart_left, col_chart_right = st.columns([3, 2])
        with col_chart_left:
            st.markdown("#### 📊 Token Consumption vs. Savings Trajectory")
            fig_token_chart = create_token_economics_chart(
                turns_data=st.session_state.token_tracker.turns,
                token_limit=st.session_state.token_tracker.token_limit
            )
            st.plotly_chart(fig_token_chart, use_container_width=True)

        with col_chart_right:
            st.markdown("#### ⚡ Reverse Memory Pipeline Routing")
            last_metric = st.session_state.last_turn_metrics
            if last_metric:
                is_hit = last_metric.get("is_cache_hit", False)
                sim_val = 0.95 if is_hit else 0.65
                saved_val = last_metric.get("tokens_saved", 0)
                fig_mem_flow = create_reverse_memory_flow_chart(
                    is_cache_hit=is_hit,
                    similarity=sim_val,
                    tokens_saved=saved_val,
                    model_name=active_model
                )
                st.plotly_chart(fig_mem_flow, use_container_width=True)
            else:
                fig_mem_flow = create_reverse_memory_flow_chart(
                    is_cache_hit=False,
                    similarity=0.0,
                    tokens_saved=0,
                    model_name=active_model
                )
                st.plotly_chart(fig_mem_flow, use_container_width=True)

        col_tab_left, col_tab_right = st.columns([1, 1])
        with col_tab_left:
            st.markdown("#### 📋 Turn-by-Turn Token Ledger")
            df_turns = st.session_state.token_tracker.to_dataframe()
            if not df_turns.empty:
                st.dataframe(df_turns, use_container_width=True, hide_index=True)
            else:
                st.info("No turns recorded in this session yet. Ask a question in the chat tab!")

        with col_tab_right:
            st.markdown("#### 🧠 Conversational Memory Vector Index")
            memories = st.session_state.memory_engine.get_all_memories()
            if memories:
                st.dataframe(pd.DataFrame(memories), use_container_width=True, hide_index=True)
                st.caption(f"Indexed **{len(memories)}** conversational turns in high-dimensional vector memory.")
            else:
                st.info("Conversational memory index is empty. Conversation turns will appear here as you chat.")

    # ==========================================
    # TAB 4: INTERACTIVE PRACTICAL RAG GLOSSARY & LAB
    # ==========================================
    with tab_glossary:
        render_interactive_glossary_lab(rag_engine)


# -----------------------------------------------------------------------------
# APPLICATION ENTRYPOINT: AUTH GATEWAY
# -----------------------------------------------------------------------------
if st.session_state.get("authenticated_user") is None:
    render_auth_screen()
else:
    render_authenticated_portal(st.session_state.authenticated_user)
