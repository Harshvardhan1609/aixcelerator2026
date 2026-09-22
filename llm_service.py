import os
from typing import Generator, List, Dict, Any, Optional
from dotenv import load_dotenv
import httpx

load_dotenv()

# Verified active free models on OpenRouter
POPULAR_FREE_MODELS = [
    "openrouter/free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "liquid/lfm-2.5-2.6b:free",
    "z-ai/glm-5.2:free",
    "poolside/laguna-s-2.1:free"
]

DEFAULT_MODEL = "openrouter/free"

def get_openrouter_api_key() -> str:
    """Retrieve the OpenRouter API key from environment variables."""
    key = os.getenv("OPEN_ROUTER_API") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
    return key.strip() if key else ""

COUNSELLOR_SYSTEM_PROMPT = """You are the official AI Student Counsellor & Advisor for "SIN - School of AI", representing the flagship "AI Agent Xcelerator Program 2026".
Your mission is to inspire, guide, and advise prospective students, understand their background, answer any questions about the program, and help them determine if this program matches their learning and career goals.

Guidelines & Brand Pillars:
1. Institution: SIN - School of AI (Motto: "Don't Prepare Students Only to Use the Future. Train Them to Build It.")
2. Persona: Empathetic, ambitious, authoritative, encouraging, and technically sharp.
3. Grounding: Base your factual answers on the provided Context from the course knowledge base. Always highlight verified details.
4. Key Value Props to highlight:
   - Transition: AI User -> AI Builder -> AI Agent Developer -> Industry-Ready Problem Solver
   - 5-Month Journey: 3 Months AI Training + 2 Months Internship Experience with real industry projects
   - 30% Theory + 70% Practical Learning (Hands-on RAG, AI Agents, Automation workflows, n8n, Claude, Gemini, APIs)
   - 12 comprehensive Modules
   - Limited Cohort: Maximum 25 students for personalized mentorship & individual project guidance
   - Key Dates: September 2026 is Awareness/Demo/Enrollment; Starting Date is 1 October 2026
   - Real portfolio projects & Xcellence ecosystem exposure
5. Formatting: Use crisp markdown, structured bullet points, and bold text for readability.
"""

def generate_chat_response_stream(
    messages: List[Dict[str, str]],
    context: str = "",
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.5
) -> Generator[str, None, None]:
    """
    Stream response from OpenRouter API.
    Yields chunks of text as they arrive.
    """
    key = api_key or get_openrouter_api_key()
    if not key:
        yield "⚠️ **Error**: OpenRouter API key is missing. Please provide it in `.env` as `OPEN_ROUTER_API=...` or in the sidebar."
        return

    # Build prompt messages with RAG context
    if messages and messages[0].get("role") == "system":
        formatted_messages = list(messages)
    else:
        system_content = COUNSELLOR_SYSTEM_PROMPT
        if context.strip():
            system_content += f"\n\n--- OFFICIAL COURSE KNOWLEDGE BASE CONTEXT ---\n{context}\n--------------------------------------------"

        formatted_messages = [{"role": "system", "content": system_content}]
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "AI Agent Xcelerator Student Counsellor RAG"
    }

    payload = {
        "model": model,
        "messages": formatted_messages,
        "temperature": temperature,
        "stream": True
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status_code != 200:
                    error_text = response.read().decode("utf-8", errors="ignore")
                    yield f"⚠️ **OpenRouter API Error (Status {response.status_code})**: {error_text}\n\n*Tip: The free model '{model}' might be rate-limited or busy. Please try selecting another model like `google/gemini-2.0-flash-exp:free` or `qwen/qwen-2.5-72b-instruct:free` from the sidebar.*"
                    return

                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except Exception:
                            continue
    except Exception as e:
        yield f"\n\n⚠️ **Connection Error**: {str(e)}\n\n*Please verify your internet connection and API key or switch to another free model.*"
