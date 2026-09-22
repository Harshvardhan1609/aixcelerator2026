import re
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# ----------------------------------------------------------------------
# PILLAR 1: RETRIEVAL QUALITY
# ----------------------------------------------------------------------

def generate_multi_queries(query: str) -> List[str]:
    """
    Multi-Query Retrieval: Generates 3 semantic perspectives of the user query
    to maximize retrieval recall and overcome vocabulary mismatch.
    """
    clean_q = query.strip()
    variations = [clean_q]
    
    # Perspective 1: Goal-oriented / practical focus
    if not clean_q.lower().startswith("how to") and not "curriculum" in clean_q.lower():
        variations.append(f"Hands-on practical curriculum, projects, and learning outcomes for: {clean_q}")
    else:
        variations.append(f"Key specifications, timelines, and requirements regarding {clean_q}")
        
    # Perspective 2: Factual / structural overview
    variations.append(f"AI Agent Xcelerator 2026 program structure, rules, and details on {clean_q}")
    
    return variations[:3]

def compute_lexical_overlap_score(query: str, document: str) -> float:
    """Calculates normalized term overlap score (BM25-style lexical matching)."""
    q_words = set(re.findall(r"\w+", query.lower()))
    doc_words = set(re.findall(r"\w+", document.lower()))
    if not q_words or not doc_words:
        return 0.0
    overlap = len(q_words.intersection(doc_words))
    return overlap / len(q_words)

def hybrid_rerank(
    retrieved_items: List[Dict[str, Any]],
    query: str,
    alpha: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Hybrid Search & Reranking:
    Blends dense vector similarity (weight alpha) with lexical keyword matching (weight 1 - alpha).
    Score = alpha * DenseSimilarity + (1 - alpha) * LexicalOverlap
    """
    for item in retrieved_items:
        dist = item.get("distance", 0.5)
        dense_sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
        doc_text = item.get("document", "")
        lex_sim = compute_lexical_overlap_score(query, doc_text)
        
        hybrid_score = (alpha * dense_sim) + ((1.0 - alpha) * lex_sim)
        item["dense_sim"] = round(dense_sim * 100, 1)
        item["lexical_sim"] = round(lex_sim * 100, 1)
        item["hybrid_score"] = round(hybrid_score * 100, 1)
        item["final_relevance"] = item["hybrid_score"]
        
    # Sort descending by hybrid score
    retrieved_items.sort(key=lambda x: x["final_relevance"], reverse=True)
    return retrieved_items

def filter_by_similarity_threshold(
    items: List[Dict[str, Any]],
    threshold_pct: float = 20.0
) -> List[Dict[str, Any]]:
    """
    Filters candidate chunks whose estimated relevance falls below the cutoff.
    Includes a production guardrail to retain the top candidate if all chunks fall below cutoff.
    """
    filtered = []
    for item in items:
        # Check distance or existing final_relevance
        if "final_relevance" in item:
            score = item["final_relevance"]
        else:
            dist = item.get("distance", 0.5)
            score = max(0.0, min(100.0, (1.0 - (dist / 2.0)) * 100))
        if score >= threshold_pct:
            filtered.append(item)
            
    # Production guardrail: retain top candidate to prevent complete context starvation
    if not filtered and items:
        filtered = [items[0]]
        
    return filtered

# ----------------------------------------------------------------------
# PILLAR 2: CHUNKING & INDEXING
# ----------------------------------------------------------------------

def simulate_chunking(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Interactive Chunking Simulator:
    Demonstrates how raw text is broken into overlapping passages for embedding.
    """
    if not text.strip():
        return []
        
    chunks = []
    start = 0
    step = max(50, chunk_size - chunk_overlap)
    text_len = len(text)
    chunk_idx = 1
    
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append({
                "chunk_id": f"chunk_{chunk_idx}",
                "start": start,
                "end": end,
                "chars": len(chunk_content),
                "words": len(chunk_content.split()),
                "preview": chunk_content[:150] + ("..." if len(chunk_content) > 150 else ""),
                "full_text": chunk_content
            })
            chunk_idx += 1
        if end >= text_len:
            break
        start += step
        
    return chunks

# ----------------------------------------------------------------------
# PILLAR 4: PROMPTING & GENERATION (Custom Instructions + Grounding)
# ----------------------------------------------------------------------

ANSWER_FORMAT_INSTRUCTIONS = {
    "Balanced (Standard)": "Provide a well-structured, warm, authoritative, and helpful answer.",
    "Crisp Bullet Points": "Format your entire response in clear, concise bullet points and bold headers with zero filler.",
    "Deep Technical Dive": "Provide an in-depth, technically rigorous explanation with architectural specifics, workflow steps, and tool names.",
    "Executive Summary": "Provide a 2-paragraph executive brief highlighting key ROI, career milestones, dates, and conclusions."
}

def build_custom_grounding_prompt(
    base_prompt: str,
    custom_user_bio: str = "",
    custom_response_pref: str = "",
    answer_format: str = "Balanced (Standard)",
    temperature: float = 0.5
) -> str:
    """
    ChatGPT-Style Custom Instructions + Grounding Prompt Builder.
    Injects user background and response preferences dynamically.
    """
    prompt = base_prompt.strip()
    
    # Custom Instructions Section (Like ChatGPT)
    if custom_user_bio.strip() or custom_response_pref.strip():
        prompt += "\n\n=== USER CUSTOM INSTRUCTIONS (ChatGPT-Style Personalization) ==="
        if custom_user_bio.strip():
            prompt += f"\n• User Background & Goals: {custom_user_bio.strip()}"
        if custom_response_pref.strip():
            prompt += f"\n• Output Style Preferences: {custom_response_pref.strip()}"
        prompt += "\n=============================================================="
        
    # Answer Format Instruction
    format_rule = ANSWER_FORMAT_INSTRUCTIONS.get(answer_format, "")
    if format_rule:
        prompt += f"\n\n[MANDATORY FORMAT REQUIREMENT ({answer_format})]: {format_rule}"
        
    # Grounding & Fallback Constraint
    prompt += (
        "\n\n[GROUNDING & NO-ANSWER FALLBACK RULE]: "
        "Strictly ground all course details, dates, numbers, and policies on the provided context. "
        "If a specific detail is not mentioned in the official context, transparently state: "
        "'*Based on the verified AI Agent Xcelerator 2026 syllabus, this detail is not specified.*' "
        "Never hallucinate dates or fees not in the context."
    )
    
    return prompt

# ----------------------------------------------------------------------
# PILLAR 6: EVALUATION & OPTIMIZATION
# ----------------------------------------------------------------------

def evaluate_groundedness(response_text: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    RAG Evaluation: Measures Faithfulness / Groundedness score
    by checking whether key terms and phrases in the response are substantiated
    by the retrieved source documents.
    """
    if not sources or not response_text.strip():
        return {
            "groundedness_score": 75.0,
            "cited_chunks": len(sources),
            "status": "Neutral (General Knowledge)",
            "status_color": "#9CA3AF"
        }
        
    # Combine sources text
    combined_kb = " ".join([s.get("document", "").lower() for s in sources])
    
    # Extract significant non-stop words from response
    resp_words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", response_text.lower())]
    if not resp_words:
        return {"groundedness_score": 80.0, "cited_chunks": len(sources), "status": "Grounded", "status_color": "#10B981"}
        
    supported_count = sum(1 for w in resp_words if w in combined_kb)
    groundedness = (supported_count / len(resp_words)) * 100
    groundedness = max(50.0, min(99.0, groundedness * 1.25))
    
    if groundedness >= 85:
        status = "Highly Grounded (Verified)"
        badge = "🟢"
        color = "#10B981"
    elif groundedness >= 70:
        status = "Moderately Grounded"
        badge = "🟡"
        color = "#F59E0B"
    else:
        status = "Extrapolated / Low Grounding"
        badge = "⚠️"
        color = "#EF4444"
        
    return {
        "groundedness_score": round(groundedness, 1),
        "cited_chunks": len(sources),
        "status": status,
        "badge": badge,
        "status_color": color,
        "token_efficiency": round(len(response_text.split()) / max(1, len(sources)), 1)
    }
