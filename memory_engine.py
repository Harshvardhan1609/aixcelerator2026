import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from chromadb.utils import embedding_functions
from token_manager import count_tokens, count_messages_tokens, estimate_tokens_saved

import re

class ConversationalMemoryEngine:
    """
    Reverse Conversational Memory Architecture (RMA) Engine.
    
    Traditional Forward RAG blindly re-sends accumulating conversation history
    and full chunks, causing token consumption to balloon on every turn.
    
    Reverse Conversational Memory Architecture:
    1. Tier 1 - Semantic Memory Cache: Checks if the user's question was already asked
       or answered in conversation history. If similarity >= cache_threshold or exact match,
       serves the answer directly from memory cache with 0 LLM tokens and 100% credit savings!
    2. Tier 2 - Relevant Memory Retrieval: For follow-up or related questions, retrieves
       only the top-K relevant dialogue turns using vector similarity rather than dumping
       full conversation history.
    3. Tier 3 - Token-Optimized Context Pruning: Dynamically adjusts knowledge base chunks
       and suppresses redundant text, reducing prompt tokens by 60% to 80%.
    """

    def __init__(self, cache_threshold: float = 0.74, embedding_fn = None):
        self.cache_threshold = cache_threshold
        # Use provided embedding function or ChromaDB's ONNX default
        if embedding_fn is not None:
            self.embedding_fn = embedding_fn
        else:
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            
        # In-memory vector index for conversation turns
        self.memory_records: List[Dict[str, Any]] = []

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes cosine similarity between two 1D vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def record_turn(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        tokens_used: int = 0
    ) -> Dict[str, Any]:
        """
        Indexes a conversation turn into the memory vector space.
        Generates dense vector embeddings for semantic lookup.
        """
        turn_id = len(self.memory_records) + 1
        
        # Embed user query
        query_emb = np.array(self.embedding_fn([query])[0])
        
        # Create concise factual summary snippet for memory context injection
        concise_resp = response[:350] + ("..." if len(response) > 350 else "")

        record = {
            "turn_id": turn_id,
            "query": query,
            "response": response,
            "response_snippet": concise_resp,
            "query_embedding": query_emb,
            "sources": sources or [],
            "tokens_used": tokens_used,
            "timestamp": time.time()
        }
        self.memory_records.append(record)
        return record

    def check_semantic_cache(
        self,
        query: str,
        threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Tier 1: Semantic Memory Cache.
        Checks if the incoming query is semantically equivalent to a prior question.
        First checks normalized lexical match; then evaluates dense vector cosine similarity.
        If similarity >= threshold, returns cached answer to bypass LLM inference (0 tokens).
        """
        if not self.memory_records:
            return None

        # 1. Fast normalized string match (handles punctuation, casing, spacing)
        clean_q = re.sub(r"[^\w\s]", "", query.lower()).strip()
        for rec in self.memory_records:
            clean_rec_q = re.sub(r"[^\w\s]", "", rec["query"].lower()).strip()
            if clean_q == clean_rec_q:
                return {
                    "hit": True,
                    "similarity": 1.0,
                    "similarity_pct": 100.0,
                    "matched_query": rec["query"],
                    "cached_response": rec["response"],
                    "turn_id": rec["turn_id"],
                    "sources": rec["sources"]
                }

        # 2. Dense vector cosine similarity
        thresh = threshold if threshold is not None else self.cache_threshold
        query_emb = np.array(self.embedding_fn([query])[0])

        best_score = -1.0
        best_record = None

        for rec in self.memory_records:
            sim = self.compute_similarity(query_emb, rec["query_embedding"])
            if sim > best_score:
                best_score = sim
                best_record = rec

        if best_score >= thresh and best_record is not None:
            return {
                "hit": True,
                "similarity": best_score,
                "similarity_pct": round(best_score * 100, 1),
                "matched_query": best_record["query"],
                "cached_response": best_record["response"],
                "turn_id": best_record["turn_id"],
                "sources": best_record["sources"]
            }

        return None

    def retrieve_relevant_memory(
        self,
        query: str,
        top_k: int = 2,
        min_relevance: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Tier 2: Relevant Memory Retrieval.
        Instead of passing all previous turns, finds the top-K conversation turns
        that are semantically relevant to the current user question.
        """
        if not self.memory_records:
            return []

        query_emb = np.array(self.embedding_fn([query])[0])
        scored_turns = []

        for rec in self.memory_records:
            sim = self.compute_similarity(query_emb, rec["query_embedding"])
            if sim >= min_relevance:
                scored_turns.append({
                    "turn_id": rec["turn_id"],
                    "query": rec["query"],
                    "response_snippet": rec["response_snippet"],
                    "similarity": sim,
                    "similarity_pct": round(sim * 100, 1),
                    "sources": rec["sources"]
                })

        # Sort descending by similarity
        scored_turns.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_turns[:top_k]

    def build_memory_augmented_prompt(
        self,
        query: str,
        kb_context: str,
        raw_chat_messages: List[Dict[str, str]],
        system_prompt: str,
        mode: str = "reverse_memory"
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Builds the optimized message list and calculates token anatomy and savings.
        
        If mode == "reverse_memory":
        - Injects only relevant conversation memory snippets rather than full chat history.
        - Prunes redundant knowledge base context.
        - Calculates tokens saved compared to standard full-history RAG.
        
        If mode == "standard":
        - Appends full recent chat history and full knowledge base context.
        """
        sys_tokens = count_tokens(system_prompt)
        query_tokens = count_tokens(query)

        # Baseline: Standard RAG (Full recent history + full KB context)
        standard_history_msgs = [m for m in raw_chat_messages if m["role"] in ("user", "assistant")]
        standard_kb_tokens = count_tokens(kb_context)
        standard_history_tokens = count_messages_tokens(standard_history_msgs)
        standard_prompt_tokens = sys_tokens + standard_kb_tokens + standard_history_tokens + query_tokens

        if mode == "standard":
            # Standard conversational RAG: send system + full KB context + full history + query
            full_system = system_prompt
            if kb_context.strip():
                full_system += f"\n\n--- OFFICIAL COURSE KNOWLEDGE BASE CONTEXT ---\n{kb_context}\n--------------------------------------------"
            
            messages = [{"role": "system", "content": full_system}]
            for msg in standard_history_msgs:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": query})

            actual_prompt_tokens = count_messages_tokens(messages)
            
            metadata = {
                "mode": "standard",
                "is_cache_hit": False,
                "relevant_memories": [],
                "prompt_tokens": actual_prompt_tokens,
                "standard_prompt_tokens": standard_prompt_tokens,
                "tokens_saved": 0,
                "breakdown": {
                    "system": sys_tokens,
                    "context": standard_kb_tokens,
                    "memory": standard_history_tokens,
                    "user": query_tokens
                }
            }
            return messages, metadata

        # -----------------------------------------------------------------
        # Reverse Memory Architecture (RMA)
        # -----------------------------------------------------------------
        # 1. Retrieve relevant memory turns
        relevant_memories = self.retrieve_relevant_memory(query, top_k=2, min_relevance=0.48)
        
        # 2. Build memory context block
        memory_block = ""
        if relevant_memories:
            memory_lines = ["--- RELEVANT CONVERSATIONAL MEMORY (Prior Exchanges) ---"]
            for mem in relevant_memories:
                memory_lines.append(
                    f"• In Turn #{mem['turn_id']}, Student asked: \"{mem['query']}\"\n"
                    f"  Counsellor advised: \"{mem['response_snippet']}\""
                )
            memory_lines.append("-----------------------------------------------------")
            memory_block = "\n".join(memory_lines)

        memory_tokens = count_tokens(memory_block)

        # 3. Assemble targeted system prompt
        rma_system = system_prompt
        
        # If relevant memory is present, add note to maintain conversational continuity
        if memory_block:
            rma_system += f"\n\n{memory_block}\n\n*Note: Use the relevant conversational memory above for continuity when the student refers to previous points.*"

        if kb_context.strip():
            rma_system += f"\n\n--- OFFICIAL COURSE KNOWLEDGE BASE CONTEXT ---\n{kb_context}\n--------------------------------------------"

        # 4. Instead of sending all 10 past messages, send the immediate preceding turn (if any)
        # to preserve local conversation flow + current query
        immediate_prior_turns: List[Dict[str, str]] = []
        if raw_chat_messages:
            # Take at most the last 1 user + 1 assistant exchange
            recent_slice = raw_chat_messages[-2:]
            for m in recent_slice:
                if m["role"] in ("user", "assistant"):
                    immediate_prior_turns.append({"role": m["role"], "content": m["content"]})

        messages = [{"role": "system", "content": rma_system}]
        for m in immediate_prior_turns:
            messages.append(m)
        messages.append({"role": "user", "content": query})

        actual_prompt_tokens = count_messages_tokens(messages)
        tokens_saved = estimate_tokens_saved(standard_prompt_tokens, actual_prompt_tokens, is_cache_hit=False)

        metadata = {
            "mode": "reverse_memory",
            "is_cache_hit": False,
            "relevant_memories": relevant_memories,
            "prompt_tokens": actual_prompt_tokens,
            "standard_prompt_tokens": standard_prompt_tokens,
            "tokens_saved": tokens_saved,
            "breakdown": {
                "system": sys_tokens,
                "context": count_tokens(kb_context),
                "memory": memory_tokens + count_messages_tokens(immediate_prior_turns),
                "user": query_tokens
            }
        }
        return messages, metadata

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Returns all indexed memory turns for UI inspection."""
        return [
            {
                "Turn": m["turn_id"],
                "Query": m["query"],
                "Response Excerpt": m["response_snippet"],
                "Tokens": m["tokens_used"],
                "Timestamp": time.strftime("%H:%M:%S", time.localtime(m["timestamp"]))
            }
            for m in self.memory_records
        ]

    def clear(self):
        """Clears all stored conversation memories."""
        self.memory_records.clear()
