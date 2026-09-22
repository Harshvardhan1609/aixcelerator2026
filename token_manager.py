import re
from typing import List, Dict, Any, Optional
import pandas as pd

# Initialize tiktoken if available, else use fallback
try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None

def count_tokens(text: str) -> int:
    """
    Accurately counts the number of tokens in a string using cl100k_base tokenizer.
    Falls back to a subword regex heuristic if tiktoken is unavailable.
    """
    if not text:
        return 0
    if _ENCODER:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    # High-accuracy fallback: split words, numbers, and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return max(1, int(len(tokens) * 1.15))

def count_messages_tokens(messages: List[Dict[str, str]]) -> int:
    """
    Counts total tokens across a list of message dicts (role, content),
    accounting for chat formatting overhead (~4 tokens per message).
    """
    total = 0
    for msg in messages:
        total += 4  # overhead for <|im_start|>{role}\n{content}<|im_end|>
        total += count_tokens(msg.get("content", ""))
    return total

def estimate_tokens_saved(
    standard_prompt_tokens: int,
    actual_prompt_tokens: int,
    is_cache_hit: bool = False,
    estimated_standard_completion: int = 250
) -> int:
    """
    Calculates tokens saved by Reverse Memory Architecture compared to standard RAG.
    If cache hit, saves both the prompt tokens and the completion tokens that would have been generated!
    """
    if is_cache_hit:
        return standard_prompt_tokens + estimated_standard_completion
    saved = standard_prompt_tokens - actual_prompt_tokens
    return max(0, saved)

class TokenTracker:
    """
    Session-level token usage, limit guardrail, and credit economy manager.
    Tracks granular per-turn consumption and cumulative savings.
    """
    def __init__(self, token_limit: int = 25000):
        self.token_limit = token_limit
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        self.total_tokens_used = 0
        self.tokens_saved_total = 0
        self.cache_hits_count = 0
        self.turns: List[Dict[str, Any]] = []

    def set_token_limit(self, limit: int):
        self.token_limit = max(1000, limit)

    def record_turn(
        self,
        query: str,
        prompt_tokens: int,
        completion_tokens: int,
        tokens_saved: int,
        mode: str,
        is_cache_hit: bool,
        breakdown: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Records metrics for an individual conversation turn.
        """
        turn_num = len(self.turns) + 1
        total_turn_tokens = prompt_tokens + completion_tokens

        self.prompt_tokens_total += prompt_tokens
        self.completion_tokens_total += completion_tokens
        self.total_tokens_used += total_turn_tokens
        self.tokens_saved_total += tokens_saved
        if is_cache_hit:
            self.cache_hits_count += 1

        record = {
            "turn": turn_num,
            "query": query[:40] + ("..." if len(query) > 40 else ""),
            "full_query": query,
            "mode": mode,
            "is_cache_hit": is_cache_hit,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_turn_tokens,
            "tokens_saved": tokens_saved,
            "cumulative_tokens": self.total_tokens_used,
            "cumulative_saved": self.tokens_saved_total,
            "breakdown": breakdown or {
                "system": 0,
                "context": 0,
                "memory": 0,
                "user": 0,
                "completion": completion_tokens
            }
        }
        self.turns.append(record)
        return record

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns high-level statistics for UI dashboard rendering.
        """
        pct_used = (self.total_tokens_used / self.token_limit) * 100 if self.token_limit > 0 else 0
        
        # Determine status
        if pct_used >= 100:
            status = "exceeded"
            status_label = "🚨 Limit Exceeded"
            status_color = "#EF4444"
        elif pct_used >= 80:
            status = "warning"
            status_label = "⚠️ Near Limit (>80%)"
            status_color = "#F59E0B"
        else:
            status = "healthy"
            status_label = "🟢 Budget Healthy"
            status_color = "#10B981"

        # Benchmark credit value ($0.002 per 1,000 tokens)
        credits_saved_usd = (self.tokens_saved_total / 1000.0) * 0.002
        credits_used_usd = (self.total_tokens_used / 1000.0) * 0.002

        total_potential_tokens = self.total_tokens_used + self.tokens_saved_total
        efficiency_pct = (self.tokens_saved_total / total_potential_tokens * 100) if total_potential_tokens > 0 else 0.0

        return {
            "token_limit": self.token_limit,
            "total_tokens_used": self.total_tokens_used,
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "tokens_saved_total": self.tokens_saved_total,
            "cache_hits_count": self.cache_hits_count,
            "pct_used": min(100.0, pct_used),
            "actual_pct": pct_used,
            "status": status,
            "status_label": status_label,
            "status_color": status_color,
            "credits_saved_usd": credits_saved_usd,
            "credits_used_usd": credits_used_usd,
            "efficiency_pct": round(efficiency_pct, 1),
            "total_turns": len(self.turns)
        }

    def reset(self):
        """Resets all token metrics."""
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        self.total_tokens_used = 0
        self.tokens_saved_total = 0
        self.cache_hits_count = 0
        self.turns = []

    def to_dataframe(self) -> pd.DataFrame:
        """Converts turns history to a Pandas DataFrame."""
        if not self.turns:
            return pd.DataFrame()
        
        rows = []
        for t in self.turns:
            rows.append({
                "Turn": t["turn"],
                "Query": t["query"],
                "Mode": t["mode"],
                "Prompt Tokens": t["prompt_tokens"],
                "Completion Tokens": t["completion_tokens"],
                "Total Tokens": t["total_tokens"],
                "Tokens Saved": t["tokens_saved"],
                "Cache Hit": "⚡ Yes" if t["is_cache_hit"] else "No"
            })
        return pd.DataFrame(rows)
