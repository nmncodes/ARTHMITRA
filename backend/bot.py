"""
Arth-Mitra AI Bot - RAG-based financial assistant
Uses LangChain + OpenRouter + ChromaDB for document retrieval and response generation
"""

import os
import re
import glob
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Iterable, Any, TYPE_CHECKING
from dotenv import load_dotenv
from cache import MultiLayerCache
from concurrent.futures import ThreadPoolExecutor
import json
import time

if TYPE_CHECKING:
    from langchain_core.documents import Document

load_dotenv()


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except Exception:
        return default


def _env_csv(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


IS_RENDER = _env_flag("RENDER", "false")
LOW_MEMORY_MODE = _env_flag("LOW_MEMORY_MODE", "true" if IS_RENDER else "false")
ENABLE_RAG = _env_flag("ENABLE_RAG", "false" if LOW_MEMORY_MODE else "true")
DEBUG_TOKEN_LOG = _env_flag("DEBUG_TOKEN_LOG", "false")
OPENROUTER_MODEL = (os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo") or "openai/gpt-3.5-turbo").strip()
STRICT_LLM_ERRORS = _env_flag("STRICT_LLM_ERRORS", "false")


class _TextResponse:
    def __init__(self, content: str):
        self.content = content


class LightweightLLM:
    """Low-memory LLM client using direct HTTP calls (no LangChain runtime imports)."""

    def __init__(self, openrouter_key: Optional[str], gemini_key: Optional[str], allow_offline_llm: bool):
        self.openrouter_key = openrouter_key
        self.gemini_key = gemini_key
        self.allow_offline_llm = allow_offline_llm
        self.openrouter_fallback_models = _env_csv(
            "OPENROUTER_FALLBACK_MODELS",
            "",
        )

        if openrouter_key:
            self.provider_name = "openrouter"
            self.model_name = OPENROUTER_MODEL
        elif gemini_key:
            self.provider_name = "gemini"
            self.model_name = "gemini-1.5-flash"
        elif allow_offline_llm:
            self.provider_name = "ollama"
            self.model_name = "gemma3:1b"
        else:
            raise RuntimeError("No valid LLM provider configured")

    def _coerce_prompt(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload

        if isinstance(payload, list):
            lines = []
            for item in payload:
                role = "user"
                content = ""

                if isinstance(item, dict):
                    role = str(item.get("role", "user"))
                    content = str(item.get("content", ""))
                else:
                    content = str(getattr(item, "content", ""))
                    cls_name = item.__class__.__name__.lower()
                    if "system" in cls_name:
                        role = "system"
                    elif "human" in cls_name:
                        role = "user"
                    elif "ai" in cls_name:
                        role = "assistant"

                lines.append(f"{role.title()}: {content}")
            return "\n".join(lines)

        return str(payload)

    def _post_json(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        import urllib.request

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)

    def _generate_openrouter(self, prompt: str, model_name: Optional[str] = None) -> str:
        model_to_use = model_name or self.model_name
        data = self._post_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
            },
            {
                "model": model_to_use,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

    def _generate_gemini(self, prompt: str) -> str:
        data = self._post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}",
            {"Content-Type": "application/json"},
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3},
            },
        )

        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join([str(p.get("text", "")) for p in parts if isinstance(p, dict)])

    def _generate_ollama(self, prompt: str) -> str:
        data = self._post_json(
            "http://localhost:11434/api/generate",
            {"Content-Type": "application/json"},
            {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        return str(data.get("response", ""))

    def _generate(self, prompt: str) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return ""

        if self.provider_name == "openrouter":
            try:
                return self._generate_openrouter(prompt)
            except Exception as openrouter_error:
                print(f"⚠️ OpenRouter primary model failed ({openrouter_error}).")

                # Retry with optional free OpenRouter models before changing provider.
                for fallback_model in self.openrouter_fallback_models:
                    if fallback_model == self.model_name:
                        continue
                    try:
                        print(f"🔁 Trying OpenRouter fallback model: {fallback_model}")
                        text = self._generate_openrouter(prompt, model_name=fallback_model)
                        if text:
                            self.model_name = fallback_model
                            return text
                    except Exception as fallback_error:
                        print(f"⚠️ OpenRouter fallback model failed ({fallback_model}): {fallback_error}")

                if self.gemini_key:
                    print(f"⚠️ OpenRouter failed ({openrouter_error}). Falling back to Gemini...")
                    return self._generate_gemini(prompt)
                raise

        if self.provider_name == "gemini":
            return self._generate_gemini(prompt)

        # Local Ollama fallback
        return self._generate_ollama(prompt)

    def invoke(self, payload: Any) -> _TextResponse:
        prompt = self._coerce_prompt(payload)
        return _TextResponse(self._generate(prompt))

    def stream(self, payload: Any) -> Iterable[_TextResponse]:
        text = self.invoke(payload).content
        if not text:
            return
        chunk_size = max(32, min(140, len(text) // 24 if len(text) > 24 else 32))
        for idx in range(0, len(text), chunk_size):
            yield _TextResponse(text[idx:idx + chunk_size])

# Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
DOCS_DIR = "./documents"  # Pre-loaded knowledge base documents
UPLOADS_DIR = "./uploads"  # Runtime uploaded documents
GOLD_DATA_PATH = os.path.join(DOCS_DIR, "gold_data.csv")

# Performance optimization settings
CACHE_MEMORY_MAX = _env_int("CACHE_MEMORY_MAX", 80 if LOW_MEMORY_MODE else 200)  # Max L1 (in-memory LRU) entries
CACHE_TTL_HOURS = 24    # Cache expiry in hours (applies to both L1 and L2 disk)
OPTIMIZED_CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", 700 if LOW_MEMORY_MODE else 1000)  # Larger chunks preserve more document context
OPTIMIZED_CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", 100 if LOW_MEMORY_MODE else 150)  # Better continuity across chunk boundaries
OPTIMIZED_RETRIEVAL_K = _env_int("RAG_RETRIEVAL_K", 4 if LOW_MEMORY_MODE else 10)  # Retrieve more candidates for document-grounded answers

RAG_WORKERS = _env_int("RAG_WORKERS", 1 if LOW_MEMORY_MODE else 3)

# Thread pool for parallel retrieval operations
_executor = ThreadPoolExecutor(max_workers=max(1, RAG_WORKERS), thread_name_prefix="rag")

# Month name mappings for date parsing
MONTH_NAMES = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12
}


def parse_date_from_query(query: str) -> Optional[datetime]:
    """
    Parse various date formats from a query string.
    Supports: DD/MM/YYYY, DD-MM-YYYY, "25th December 2020", "December 25, 2020", etc.
    Returns datetime object or None if no date found.
    """
    query_lower = query.lower()
    
    # Pattern 1: DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    pattern1 = r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})'
    match = re.search(pattern1, query)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            pass
    
    # Pattern 2: "25th December 2020", "25 December 2020", "25th Dec 2020"
    pattern2 = r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\s+(\d{4})'
    match = re.search(pattern2, query_lower)
    if match:
        day = int(match.group(1))
        month_name = match.group(2)
        year = int(match.group(3))
        month = MONTH_NAMES.get(month_name)
        if month:
            try:
                return datetime(year, month, day)
            except ValueError:
                pass
    
    # Pattern 3: "December 25, 2020" or "Dec 25 2020"
    pattern3 = r'([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})'
    match = re.search(pattern3, query_lower)
    if match:
        month_name = match.group(1)
        day = int(match.group(2))
        year = int(match.group(3))
        month = MONTH_NAMES.get(month_name)
        if month:
            try:
                return datetime(year, month, day)
            except ValueError:
                pass
    
    # Pattern 4: YYYY/MM/DD or YYYY-MM-DD
    pattern4 = r'(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})'
    match = re.search(pattern4, query)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return datetime(year, month, day)
        except ValueError:
            pass
    
    return None


def is_gold_price_query(query: str) -> bool:
    """Check if the query is asking about gold prices."""
    query_lower = query.lower()
    gold_keywords = ['gold', 'sona', 'sonay', 'gold price', 'gold rate', 'gold ki price', 'gold ka rate']
    date_present = parse_date_from_query(query) is not None
    
    for keyword in gold_keywords:
        if keyword in query_lower and date_present:
            return True
    return False


class GoldPriceLookup:
    """Direct CSV lookup for gold prices by date."""
    
    def __init__(self, csv_path: str = GOLD_DATA_PATH):
        self.csv_path = csv_path
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load and parse the gold data CSV."""
        if os.path.exists(self.csv_path):
            try:
                import pandas as pd

                self.df = pd.read_csv(self.csv_path)
                # Parse dates - format is DD/MM/YYYY
                self.df['ParsedDate'] = pd.to_datetime(
                    self.df['Date'], 
                    format='%d/%m/%Y', 
                    errors='coerce'
                )
                self.df = self.df.dropna(subset=['ParsedDate'])
                self.df = self.df.sort_values('ParsedDate')
            except Exception as e:
                print(f"Error loading gold data: {e}")
                self.df = None
    
    def get_price(self, date: datetime) -> Optional[Dict]:
        """Get gold price for exact date."""
        if self.df is None:
            return None
        
        target_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        result = self.df[self.df['ParsedDate'] == target_date]
        
        if not result.empty:
            row = result.iloc[0]
            return {
                'date': row['Date'],
                'price': row['Price'],
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'volume': row.get('Volume', 'N/A'),
                'found': True
            }
        return None
    
    def get_nearest_price(self, date: datetime, max_days: int = 7) -> Tuple[Optional[Dict], str]:
        """
        Get nearest available price if exact date not found.
        Returns (price_data, explanation_string)
        """
        if self.df is None:
            return None, "Gold price data not available."
        
        target_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check for exact match first
        exact = self.get_price(date)
        if exact:
            return exact, f"Exact date found"
        
        # Find nearest dates (before and after)
        before = self.df[self.df['ParsedDate'] < target_date].tail(1)
        after = self.df[self.df['ParsedDate'] > target_date].head(1)
        
        nearest = None
        direction = ""
        
        if not before.empty and not after.empty:
            before_diff = abs((target_date - before.iloc[0]['ParsedDate']).days)
            after_diff = abs((after.iloc[0]['ParsedDate'] - target_date).days)
            
            if before_diff <= after_diff and before_diff <= max_days:
                nearest = before.iloc[0]
                direction = "before"
            elif after_diff <= max_days:
                nearest = after.iloc[0]
                direction = "after"
        elif not before.empty:
            before_diff = abs((target_date - before.iloc[0]['ParsedDate']).days)
            if before_diff <= max_days:
                nearest = before.iloc[0]
                direction = "before"
        elif not after.empty:
            after_diff = abs((after.iloc[0]['ParsedDate'] - target_date).days)
            if after_diff <= max_days:
                nearest = after.iloc[0]
                direction = "after"
        
        if nearest is not None:
            return {
                'date': nearest['Date'],
                'price': nearest['Price'],
                'open': nearest['Open'],
                'high': nearest['High'],
                'low': nearest['Low'],
                'volume': nearest.get('Volume', 'N/A'),
                'found': False,
                'nearest': True
            }, f"Data not available for requested date (possibly a holiday/weekend). Nearest available date ({direction})"
        
        return None, "No gold price data available within the date range."
    
    def get_date_range(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the available date range in the data."""
        if self.df is None or self.df.empty:
            return None, None
        return self.df['Date'].iloc[0], self.df['Date'].iloc[-1]


# Global gold lookup instance
_gold_lookup: Optional[GoldPriceLookup] = None


def get_gold_lookup() -> GoldPriceLookup:
    """Get or create gold lookup singleton."""
    global _gold_lookup
    if _gold_lookup is None:
        _gold_lookup = GoldPriceLookup()
    return _gold_lookup


# ResponseCache replaced by MultiLayerCache from cache.py
# (L1 in-memory + L2 disk, drop-in compatible API)


def format_user_profile(profile: Dict) -> str:
    """Format user profile information for the system prompt."""
    if not profile:
        return ""
    
    profile_text = "**USER PROFILE** (Provide personalized recommendations based on this information):\n"
    
    # Compulsory fields
    profile_text += f"- **Age**: {profile.get('age', 'Not specified')}"
    if profile.get('age', 0) >= 60:
        profile_text += " (Senior Citizen - eligible for senior citizen schemes)"
    elif profile.get('age', 0) >= 55:
        profile_text += " (Near retirement age)"
    profile_text += "\n"
    
    profile_text += f"- **Annual Income**: {profile.get('income', 'Not specified')}\n"
    profile_text += f"- **Employment Status**: {profile.get('employmentStatus', 'Not specified')}\n"
    
    # Add specific notes based on employment
    emp_status = profile.get('employmentStatus')

    EMP_NOTES = {
    "Salaried - Government":
        "Higher NPS employer contribution limit – 14% of salary",

    "Salaried - Private":
        "Optimize 80C + 80D, consider NPS Tier I ₹50,000 under 80CCD(1B)",

    "Self-Employed":
        "Consider presumptive taxation (44ADA), expense deductions, advance tax planning",

    "Business Owner":
        "Explore 44AD/44AE, depreciation benefits, expense optimisation, advance tax",

    "Retired":
        "Focus on senior citizen schemes like SCSS, PMVVY, higher interest exemption",

    "Unemployed":
        "Focus on tax-free interest instruments and capital gains planning"
    }

    note = EMP_NOTES.get(emp_status)
    if note:
        profile_text += f"  (Note: {note})\n"
    
    profile_text += f"- **Tax Regime**: {profile.get('taxRegime', 'Not specified')}\n"
    if profile.get('taxRegime') == 'Old Regime':
        profile_text += "  (Note: Eligible for 80C, 80D, and other deductions)\n"
    elif profile.get('taxRegime') == 'New Regime':
        profile_text += (
        "(Note: Most deductions like 80C/80D not available; "
        "standard deduction and employer NPS contribution still allowed)\n"
        )
    elif profile.get('taxRegime') not in ('Old Regime', 'New Regime'):
        profile_text += (
        "(Note: Tax-saving advice depends heavily on regime selection)\n"
    )

    
    profile_text += f"- **Housing Status**: {profile.get('homeownerStatus', 'Not specified')}\n"
    if 'Loan' in profile.get('homeownerStatus', ''):
        profile_text += "  (Note: Eligible for home loan interest deduction - Section 24)\n"
    elif 'Rented' in profile.get('homeownerStatus', ''):
        profile_text += "  (Note: May be eligible for HRA exemption if salaried)\n"
    
    # Optional fields (only show if provided)
    if profile.get('children'):
        profile_text += f"- **Children**: {profile.get('children')}"
        if profile.get('childrenAges'):
            profile_text += f" (Ages: {profile.get('childrenAges')})"
            # Check for girl child under 10
            try:
                ages = [int(age.strip()) for age in profile.get('childrenAges', '').split(',') if age.strip()]
                if any(age < 10 for age in ages):
                    profile_text += " - Eligible for Sukanya Samriddhi Yojana if girl child"
            except:
                pass
        profile_text += "\n"
    
    if profile.get('parentsAge'):
        profile_text += f"- **Parents Age**: {profile.get('parentsAge')}\n"
        # Check if parents are senior citizens
        if '60' in str(profile.get('parentsAge', '')) or '65' in str(profile.get('parentsAge', '')):
            profile_text += "  (Note: Additional 80D deduction for senior citizen parents - ₹50,000)\n"
    
    if profile.get('investmentCapacity'):
        profile_text += f"- **Investment Capacity**: {profile.get('investmentCapacity')}\n"
    
    if profile.get('riskAppetite'):
        profile_text += f"- **Risk Appetite**: {profile.get('riskAppetite')}\n"
        if profile.get('riskAppetite') == 'Conservative':
            profile_text += "  (Note: Recommend fixed-return instruments like PPF, NSC, SCSS)\n"
        elif profile.get('riskAppetite') == 'Aggressive':
            profile_text += "  (Note: Can suggest ELSS, NPS equity allocation, market-linked returns)\n"
    
    profile_text += "\n"
    return profile_text


def format_chat_history(history: Optional[List[Dict]]) -> str:
    """Format recent chat history for the prompt."""
    if not history:
        return "None"

    lines = []
    for item in history:
        role = item.get("role", "user")
        content = item.get("content", "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")

    return "\n".join(lines) if lines else "None"


# System prompt optimized for faster token generation
SYSTEM_PROMPT = """You are Arth-Mitra, an expert Indian financial advisor.

{user_profile}

Recent chat: {chat_history}

**Guidelines:**
- Use ## headers, **bold** for key terms, tables for comparisons
- Provide specific numbers, amounts, eligibility criteria
- Reference source documents
- If info not in context, state clearly
- Keep responses concise and actionable

**Context:** {context}

**Query:** {question}

**Response:**"""


class ArthMitraBot:
    """RAG-based financial assistant bot"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.rag_chain = None
        self.llm = None
        self._initialized = False
        self._retriever = None
        self._indexed_files = set()
        self._response_cache = MultiLayerCache(
            memory_max=CACHE_MEMORY_MAX,
            ttl_hours=CACHE_TTL_HOURS,
        )

    def _append_sources_section(self, response: str, sources: List[str]) -> str:
        """Append an explicit Sources section to the response body."""
        if not sources:
            sources = ["Knowledge Base"]
        sources_lines = "\n".join([f"- {source}" for source in sources])
        return f"{response}\n\n---\nSources:\n{sources_lines}"

    def _build_model_outage_message(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Return a useful fallback when cloud model calls fail."""
        metadata = metadata or {}
        lines = [
            "I am facing a temporary model provider issue, but here is a quick plan based on your profile and query.",
        ]

        schemes = metadata.get("schemes") or []
        if schemes:
            lines.append("")
            lines.append("Top options to consider right now:")
            for idx, scheme in enumerate(schemes[:3], start=1):
                name = scheme.get("name", "Option")
                score = scheme.get("score", "-")
                reason = scheme.get("reason", "General suitability based on available profile")
                lines.append(f"{idx}. **{name}** ({score}/100): {reason}")

        action_plan = metadata.get("actionPlan") or {}
        steps = action_plan.get("steps") or []
        if steps:
            lines.append("")
            lines.append("Recommended next steps:")
            for step in steps[:3]:
                lines.append(f"- {step}")

        why_this_answer = metadata.get("whyThisAnswer")
        if why_this_answer:
            lines.append("")
            lines.append(f"Why this snapshot: {why_this_answer}")

        lines.append("")
        lines.append(
            "Please retry in 1-2 minutes. For stronger uptime, configure both OPENROUTER_API_KEY and GEMINI_API_KEY."
        )
        return "\n".join(lines)
    
    def clear_cache(self):
        """Clear all caches (response + embedding)"""
        self._response_cache.clear()
        if self.embeddings and hasattr(self.embeddings, 'clear_cache'):
            self.embeddings.clear_cache()
        print("✅ Response and embedding caches cleared")
    
    def initialize(self, auto_index: bool = True):
        """Initialize the bot with embeddings and LLM"""
        # Prefer OpenRouter key first, then Gemini/Google key.
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        allow_offline_llm = os.getenv("ALLOW_OFFLINE_LLM", "false").lower() == "true"
        
        if not openrouter_key and not gemini_key and not allow_offline_llm:
            raise RuntimeError(
                "No cloud LLM key configured. Set OPENROUTER_API_KEY or GEMINI_API_KEY (or GOOGLE_API_KEY). "
                "If you want local Ollama fallback, set ALLOW_OFFLINE_LLM=true."
            )

        if LOW_MEMORY_MODE:
            print("🪶 LOW_MEMORY_MODE enabled (memory-optimized runtime settings active)")

        if LOW_MEMORY_MODE and not ENABLE_RAG:
            print("🤖 Using low-memory HTTP LLM client")
            self.llm = LightweightLLM(openrouter_key, gemini_key, allow_offline_llm)
            self.embeddings = None
            self.vectorstore = None
            self.rag_chain = None
            self._retriever = None
            self._initialized = True
            return self
        
        # Initialize LLM
        if openrouter_key:
            from langchain_openai import ChatOpenAI

            print(f"🤖 Using OpenRouter AI ({OPENROUTER_MODEL})")
            self.llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            temperature=0.3,
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            )
        elif gemini_key:
            from langchain_google_genai import ChatGoogleGenerativeAI

            print("🤖 Using Google Gemini AI (gemini-1.5-flash)")
            self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=gemini_key,
            convert_system_message_to_human=True,
            )
        else:
            from langchain_ollama import ChatOllama

            # Offline fallback LLM
            print("🟡 No API keys found — using offline LLM (gemma3:1b)")
            self.llm = ChatOllama(
            model="gemma3:1b",   # speeeed
            temperature=0.2,
            base_url="http://localhost:11434",
            think=False,
            )

        if not ENABLE_RAG:
            print("⏭️ ENABLE_RAG=false: running in LLM-only mode (vector retrieval disabled)")
            self.embeddings = None
            self.vectorstore = None
            self.rag_chain = None
            self._retriever = None
            self._initialized = True
            return self

        # Initialize ONNX-accelerated embeddings with built-in LRU cache
        # Falls back to standard HuggingFace if ONNX runtime not available
        if self.embeddings is None:
            from onnx_embeddings import OptimizedEmbeddings

            print("🔄 Loading optimized embeddings model...")
            self.embeddings = OptimizedEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        print("✅ Embeddings model loaded")

        from langchain_chroma import Chroma


        # Load or create vector store
        if os.path.exists(CHROMA_PERSIST_DIR):
            self.vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings
            )
        else:
            # Create empty vectorstore if no documents indexed yet
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            self.vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings
            )
        
        self._initialized = True
        
        # Auto-index documents from knowledge base folder
        if auto_index:
            self._auto_index_documents()
        
        # Create RAG chain
        self._create_rag_chain()
        
        return self
    
    def _auto_index_documents(self):
        """Auto-index all documents from documents + uploads folders"""
        for folder in [DOCS_DIR, UPLOADS_DIR]:
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

        # Get all supported files
        supported_extensions = ['*.pdf', '*.csv', '*.txt', '*.md', '*.docx']
        files_to_index = []
        
        for base_dir in [DOCS_DIR, UPLOADS_DIR]:
            for ext in supported_extensions:
                files_to_index.extend(glob.glob(os.path.join(base_dir, '**', ext), recursive=True))
        
        if not files_to_index:
            print(f"📭 No documents found in {DOCS_DIR} or {UPLOADS_DIR}. Add PDF/CSV/TXT/MD/DOCX files for knowledge base.")
            return
        
        # Check which files are already indexed (by checking metadata)
        try:
            existing_sources = set()
            if self.vectorstore._collection.count() > 0:
                results = self.vectorstore.get(include=['metadatas'])
                for meta in results.get('metadatas', []):
                    if meta and 'source' in meta:
                        existing_sources.add(os.path.basename(meta['source']))
        except:
            existing_sources = set()
        
        # Index new files
        new_files = [f for f in files_to_index if os.path.basename(f) not in existing_sources]
        
        if new_files:
            print(f"📚 Indexing {len(new_files)} new document(s)...")
            for file_path in new_files:
                try:
                    result = self.add_documents(file_path)
                    print(f"  ✓ {result['message']}")
                except Exception as e:
                    print(f"  ✗ Failed to index {os.path.basename(file_path)}: {e}")
        else:
            print(f"✓ Knowledge base up to date ({len(existing_sources)} documents indexed)")
    
    def _format_docs(self, docs):
        """Format retrieved documents into a string with source info - optimized for speed"""
        formatted = []
        max_context_length = 12000  # Larger context allows better coverage of uploaded docs
        current_length = 0
        
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '')
            source_info = f"[Source: {os.path.basename(source)}"
            if page:
                source_info += f", Page {page + 1}"
            source_info += "]"
            
            content = doc.page_content
            # Truncate if we're approaching the limit
            if current_length + len(content) > max_context_length:
                remaining = max_context_length - current_length
                if remaining > 100:  # Only add if we have meaningful space left
                    content = content[:remaining] + "..."
                    formatted.append(f"{content}\\n{source_info}")
                break
            
            formatted.append(f"{content}\\n{source_info}")
            current_length += len(content)
        
        return "\n\n---\n\n".join(formatted)
    
    def _create_rag_chain(self):
        """Create the RAG chain using LCEL"""
        try:
            doc_count = self.vectorstore._collection.count()
        except:
            doc_count = 0
            
        if doc_count > 0:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnablePassthrough

            prompt_template = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

            # Use MMR (Maximal Marginal Relevance) for better diversity with fewer docs
            # This retrieves fewer but more relevant documents = faster queries
            self._retriever = self.vectorstore.as_retriever(
                search_type="mmr",  # Changed from similarity to mmr for better relevance
                search_kwargs={
                    "k": OPTIMIZED_RETRIEVAL_K,  
                    "fetch_k": 30,  # Fetch more candidates and select best diverse chunks
                    "lambda_mult": 0.65  # Balance between relevance and diversity
                }
            )
            
            self.rag_chain = (
                {"context": self._retriever | self._format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | self.llm
                | StrOutputParser()
            )

    def _get_source_docs(self, query: str, source_filter: Optional[str] = None):
        """Retrieve documents, optionally constrained to a specific source file."""
        if not self.vectorstore or not self._retriever:
            return []

        if not source_filter:
            return self._retriever.invoke(query)

        def normalize_source_name(name: str) -> str:
            source = os.path.basename((name or "").strip())
            source = re.sub(r"\s+", " ", source)
            return source.lower()

        # Fetch a wider candidate pool and filter by source filename to support
        # both old metadata (full path) and new metadata (basename).
        candidates = self.vectorstore.similarity_search(query, k=max(OPTIMIZED_RETRIEVAL_K * 3, 30))
        target = normalize_source_name(source_filter)

        filtered = []
        for doc in candidates:
            source_meta = doc.metadata.get("source", "")
            source_name = normalize_source_name(source_meta)
            if source_name == target:
                filtered.append(doc)

        if filtered:
            return filtered[:OPTIMIZED_RETRIEVAL_K]

        # Fallback: if semantic retrieval misses, pull chunks directly from the
        # selected source so the model can still answer from that document.
        try:
            results = self.vectorstore.get(include=['documents', 'metadatas'])
            documents = results.get('documents', []) or []
            metadatas = results.get('metadatas', []) or []

            direct_matches = []
            from langchain_core.documents import Document

            for content, metadata in zip(documents, metadatas):
                if not metadata:
                    continue
                source_name = normalize_source_name(metadata.get('source', ''))
                if source_name == target:
                    direct_matches.append(Document(page_content=content, metadata=metadata))

            direct_matches.sort(key=lambda d: d.metadata.get('chunk_index', 10**9))
            if direct_matches:
                return direct_matches[:OPTIMIZED_RETRIEVAL_K]
        except Exception as e:
            print(f"⚠️ Source filter fallback failed: {e}")

        return []
    
    def add_documents(self, file_path: str) -> Dict:
        """Add documents to the knowledge base"""
        if not self._initialized:
            raise RuntimeError("Bot not initialized. Call initialize() first.")

        if not self.vectorstore:
            return {
                "status": "error",
                "message": "RAG is disabled (ENABLE_RAG=false). Enable RAG to index documents."
            }
        
        # Determine loader based on file type
        file_ext = os.path.splitext(file_path)[1].lower()

        from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == ".csv":
            loader = CSVLoader(file_path)
        elif file_ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif file_ext in [".txt", ".md"]:
            loader = TextLoader(file_path)
        else:
            return {"status": "error", "message": f"Unsupported file type: {file_ext}"}
        
        # Load and split documents with optimized chunk sizes
        documents = loader.load()
        
        # Smaller chunks = faster retrieval and less token usage
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=OPTIMIZED_CHUNK_SIZE,
            chunk_overlap=OPTIMIZED_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "]
        )
        
        splits = text_splitter.split_documents(documents)

        # Ensure consistent source metadata for precise citations
        for index, split in enumerate(splits):
            split.metadata["source"] = os.path.basename(file_path)
            split.metadata["chunk_index"] = index
            split.metadata["file_type"] = file_ext
        
        # Add to vector store
        self.vectorstore.add_documents(splits)
        
        # Recreate RAG chain with updated vectorstore
        self._create_rag_chain()
        
        return {
            "status": "success",
            "message": f"Indexed {len(splits)} chunks from {os.path.basename(file_path)}"
        }

    def remove_document(self, filename: str) -> dict:
        """Remove all chunks for a given filename from the vector store."""
        try:
            if not self.vectorstore:
                return {"status": "error", "message": "Vector store not initialised (RAG may be disabled)"}

            collection = self.vectorstore._collection
            # Find all chunk IDs whose source matches the filename
            results = collection.get(include=["metadatas"])
            ids_to_delete = []
            for doc_id, meta in zip(results["ids"], results["metadatas"]):
                if meta and meta.get("source") == filename:
                    ids_to_delete.append(doc_id)

            if not ids_to_delete:
                return {"status": "success", "message": f"No chunks found for {filename}", "removed": 0}

            collection.delete(ids=ids_to_delete)

            # Rebuild the RAG chain so it picks up the reduced collection
            self._create_rag_chain()

            return {
                "status": "success",
                "message": f"Removed {len(ids_to_delete)} chunks for {filename}",
                "removed": len(ids_to_delete),
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to remove document: {str(e)}"}

    def _extract_text(self, content) -> str:
        """Extract text from LLM response content"""
        if DEBUG_TOKEN_LOG:
            print(f"[DEBUG] _extract_text received: type={type(content)}, value={repr(content)[:100]}")
        
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Handle Gemini's list format
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    return item.get('text', '')
                if isinstance(item, str):
                    return item
            return str(content) if content else ""
        # Handle AIMessageChunk or similar objects
        if hasattr(content, 'text'):
            return content.text
        return str(content) if content else ""

    def _extract_query_terms(self, query: str) -> List[str]:
        words = re.findall(r"[a-zA-Z]{3,}", (query or "").lower())
        stop_words = {
            "what", "which", "where", "when", "this", "that", "from", "with", "about",
            "please", "could", "would", "there", "their", "have", "show", "tell", "into"
        }
        terms: List[str] = []
        seen = set()
        for word in words:
            if word in stop_words or word in seen:
                continue
            seen.add(word)
            terms.append(word)
        return terms[:8]

    def _build_source_highlights(self, source_docs: List["Document"], query: str, max_items: int = 3) -> List[Dict[str, str]]:
        terms = self._extract_query_terms(query)
        highlights: List[Dict[str, str]] = []

        for doc in source_docs:
            content = (doc.page_content or "").strip()
            if not content:
                continue

            snippet = ""
            lower_content = content.lower()
            if terms:
                positions = [lower_content.find(term) for term in terms if lower_content.find(term) >= 0]
                if positions:
                    pos = min(positions)
                    start = max(0, pos - 70)
                    end = min(len(content), pos + 180)
                    snippet = content[start:end].replace("\n", " ").strip()

            if not snippet:
                snippet = content[:220].replace("\n", " ").strip()

            if len(snippet) > 220:
                snippet = snippet[:220].rstrip() + "..."

            source_name = os.path.basename(doc.metadata.get("source", "Knowledge Base"))
            page = doc.metadata.get("page")
            if page is not None and str(page).isdigit():
                source_name = f"{source_name} (Page {int(page) + 1})"

            highlights.append({"source": source_name, "snippet": snippet})
            if len(highlights) >= max_items:
                break

        return highlights

    def _calculate_confidence(self, source_docs: List["Document"], query: str, source_filter: Optional[str] = None) -> Tuple[float, str]:
        if not source_docs:
            return 0.32, "low"

        unique_sources = len({os.path.basename(doc.metadata.get("source", "")) for doc in source_docs})
        context_chars = sum(len((doc.page_content or "")) for doc in source_docs)
        terms = self._extract_query_terms(query)

        match_count = 0
        if terms:
            merged = "\n".join([(doc.page_content or "").lower() for doc in source_docs])
            for term in terms:
                if term in merged:
                    match_count += 1

        term_coverage = (match_count / max(len(terms), 1)) if terms else 0.6

        score = 0.35
        score += min(unique_sources, 4) * 0.09
        score += 0.12 if context_chars > 1800 else 0.06
        score += min(term_coverage * 0.25, 0.25)
        if source_filter:
            score += 0.08

        score = round(max(0.2, min(score, 0.97)), 2)
        label = "high" if score >= 0.75 else "medium" if score >= 0.5 else "low"
        return score, label

    def _parse_income_number(self, income_raw: Optional[str]) -> Optional[float]:
        if not income_raw:
            return None
        cleaned = re.sub(r"[^\d.]", "", str(income_raw))
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except Exception:
            return None

    def _build_scheme_rankings(self, profile: Optional[Dict], query: str, max_items: int = 5) -> List[Dict[str, Any]]:
        profile = profile or {}
        query_l = (query or "").lower()

        age = profile.get("age") or 0
        try:
            age = int(age)
        except Exception:
            age = 0

        tax_regime = (profile.get("taxRegime") or "").lower()
        risk = (profile.get("riskAppetite") or "").lower()
        employment = (profile.get("employmentStatus") or "").lower()
        homeowner = (profile.get("homeownerStatus") or "").lower()
        goals = [str(g).lower() for g in (profile.get("financialGoals") or [])]
        children = (profile.get("children") or "").lower()
        income = self._parse_income_number(profile.get("income"))

        has_tax_intent = any(k in query_l for k in ["tax", "deduction", "save", "80c", "80d", "it return"])
        has_pension_intent = any(k in query_l for k in ["pension", "retire", "retirement"])
        has_child_intent = any(k in query_l for k in ["child", "education", "daughter", "sukanya"])

        schemes = [
            {
                "name": "Public Provident Fund (PPF)",
                "base": 72,
                "eligibility": "Resident individual; 15-year lock-in; up to ₹1.5L under 80C (Old Regime)",
                "nextStep": "Open/continue PPF account and set monthly auto-debit.",
            },
            {
                "name": "National Pension System (NPS)",
                "base": 70,
                "eligibility": "Age 18-70; extra ₹50,000 deduction under 80CCD(1B) (Old Regime)",
                "nextStep": "Select active/auto choice and set annual contribution target.",
            },
            {
                "name": "ELSS Tax Saver Funds",
                "base": 66,
                "eligibility": "Equity-linked 80C option with 3-year lock-in (Old Regime)",
                "nextStep": "Start SIP aligned to risk profile and 3+ year horizon.",
            },
            {
                "name": "Senior Citizens’ Savings Scheme (SCSS)",
                "base": 64,
                "eligibility": "Age 60+ (or eligible retirees); quarterly interest payout",
                "nextStep": "Evaluate SCSS for stable retirement income allocation.",
            },
            {
                "name": "Sukanya Samriddhi Yojana (SSY)",
                "base": 63,
                "eligibility": "Girl child account with EEE tax treatment under 80C (Old Regime)",
                "nextStep": "If eligible, open SSY for long-term education/marriage corpus.",
            },
            {
                "name": "PM Jeevan Jyoti Bima Yojana (PMJJBY)",
                "base": 55,
                "eligibility": "Low-cost annual life cover; linked bank account required",
                "nextStep": "Activate PMJJBY through your bank for baseline life cover.",
            },
        ]

        ranked: List[Dict[str, Any]] = []
        for item in schemes:
            score = float(item["base"])
            name_l = item["name"].lower()
            reasons: List[str] = []
            missing_criteria: List[str] = []

            if "ppf" in name_l:
                if risk in ["conservative", "moderate"]:
                    score += 10
                    reasons.append("Matches lower-volatility preference")
                if "old" in tax_regime or has_tax_intent:
                    score += 8
                    reasons.append("Supports 80C tax-saving plan")

            if "nps" in name_l:
                if "salaried" in employment or "self" in employment or "business" in employment:
                    score += 9
                    reasons.append("Works well for long-term retirement corpus")
                if has_pension_intent or "retirement" in " ".join(goals):
                    score += 10
                    reasons.append("Aligned with retirement objective")
                if "old" in tax_regime:
                    score += 7
                    reasons.append("Eligible for additional tax deduction")

            if "elss" in name_l:
                if risk in ["aggressive", "moderate"]:
                    score += 10
                    reasons.append("Suitable for market-linked growth")
                if "old" in tax_regime or has_tax_intent:
                    score += 7
                    reasons.append("Combines wealth creation with 80C")

            if "scss" in name_l:
                if age >= 60:
                    score += 26
                    reasons.append("Strong fit for senior-citizen eligibility")
                else:
                    score -= 20
                    reasons.append("Usually not eligible before 60")
                    missing_criteria.append("Age 60+ generally required")

            if "sukanya" in name_l:
                if has_child_intent or "yes" in children:
                    score += 18
                    reasons.append("Relevant for child-focused planning")
                else:
                    score -= 16
                    reasons.append("Needs girl-child eligibility")
                    missing_criteria.append("Girl child eligibility needed")

            if "pm jeevan" in name_l and (income is None or income < 1500000):
                score += 8
                reasons.append("Low-cost protection layer")

            if "new" in tax_regime and ("80c" in item["eligibility"].lower() or "deduction" in item["eligibility"].lower()):
                score -= 8
                reasons.append("Tax benefit impact lower in New Regime")
                missing_criteria.append("Old Regime gives stronger deduction benefit")

            if "loan" in homeowner and "nps" in name_l:
                score += 3
                reasons.append("Can complement existing home-loan tax planning")

            if not reasons:
                reasons.append("General suitability based on available profile")

            ranked.append({
                "name": item["name"],
                "score": int(max(1, min(round(score), 100))),
                "reason": "; ".join(reasons[:2]),
                "eligibility": item["eligibility"],
                "nextStep": item["nextStep"],
                "missingCriteria": missing_criteria,
            })

        ranked.sort(key=lambda row: row["score"], reverse=True)
        return ranked[:max_items]

    def _extract_document_insights(self, source_docs: List["Document"], max_items: int = 6) -> List[Dict[str, str]]:
        insights: List[Dict[str, str]] = []
        if not source_docs:
            return insights

        def add_insight(field: str, value: str, source: str):
            if not value:
                return
            if any(item["field"] == field and item["value"] == value for item in insights):
                return
            insights.append({"field": field, "value": value, "source": source})

        for doc in source_docs:
            source_name = os.path.basename(doc.metadata.get("source", "Knowledge Base"))
            text = (doc.page_content or "").replace("\n", " ")
            text_l = text.lower()

            rate_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
            if rate_match:
                add_insight("Interest Rate", f"{rate_match.group(1)}%", source_name)

            date_match = re.search(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})\b", text)
            if date_match and ("maturity" in text_l or "tenure" in text_l or "valid" in text_l):
                add_insight("Maturity / Tenure Date", date_match.group(1), source_name)

            if any(keyword in text_l for keyword in ["penalty", "premature", "charges", "fee"]):
                sentence = re.search(r"([^.!?]*(penalty|premature|charges|fee)[^.!?]*[.!?])", text, re.IGNORECASE)
                if sentence:
                    add_insight("Penalties", sentence.group(1).strip()[:180], source_name)

            if any(keyword in text_l for keyword in ["eligible", "eligibility", "who can", "applicable"]):
                sentence = re.search(r"([^.!?]*(eligible|eligibility|who can|applicable)[^.!?]*[.!?])", text, re.IGNORECASE)
                if sentence:
                    add_insight("Eligibility", sentence.group(1).strip()[:180], source_name)

            if len(insights) >= max_items:
                break

        return insights[:max_items]

    def _build_compare_mode(self, query: str, schemes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        query_l = (query or "").lower()
        is_compare = (" vs " in query_l) or (" versus " in query_l) or ("compare" in query_l)
        if not is_compare or len(schemes) < 2:
            return None

        pool = schemes[:5]
        chosen = []
        for scheme in pool:
            key = scheme.get("name", "").lower()
            parts = [p for p in re.split(r"[^a-zA-Z]+", key) if len(p) > 2]
            if any(part in query_l for part in parts):
                chosen.append(scheme)

        unique = []
        seen_names = set()
        for item in chosen + pool:
            name = item.get("name")
            if name in seen_names:
                continue
            seen_names.add(name)
            unique.append(item)
            if len(unique) == 2:
                break

        if len(unique) < 2:
            return None

        a, b = unique[0], unique[1]
        rec = a if a.get("score", 0) >= b.get("score", 0) else b
        return {
            "schemeA": {
                "name": a.get("name"),
                "score": a.get("score"),
                "pros": [a.get("reason", "")],
                "cons": a.get("missingCriteria", [])[:2],
                "fit": a.get("eligibility", ""),
            },
            "schemeB": {
                "name": b.get("name"),
                "score": b.get("score"),
                "pros": [b.get("reason", "")],
                "cons": b.get("missingCriteria", [])[:2],
                "fit": b.get("eligibility", ""),
            },
            "recommendedFit": f"{rec.get('name')} is currently the better fit for this profile/query.",
        }

    def _build_why_this_answer(self, query: str, source_docs: List["Document"], confidence: float, source_filter: Optional[str]) -> str:
        source_count = len({os.path.basename(doc.metadata.get("source", "")) for doc in source_docs})
        chunk_count = len(source_docs)
        terms = self._extract_query_terms(query)[:4]
        terms_text = ", ".join(terms) if terms else "core query intent"
        if source_filter:
            return (
                f"Response grounded on {chunk_count} chunks from {source_count} source(s) within selected file '{source_filter}', "
                f"matching terms like {terms_text}. Confidence is {int(confidence * 100)}%."
            )
        return (
            f"Response grounded on {chunk_count} retrieved chunks from {source_count} source(s), "
            f"matching terms like {terms_text}. Confidence is {int(confidence * 100)}%."
        )

    def _build_tax_action_plan(self, profile: Optional[Dict], query: str) -> Optional[Dict[str, Any]]:
        profile = profile or {}
        query_l = (query or "").lower()
        tax_regime = (profile.get("taxRegime") or "").strip() or "Not selected"
        employment = (profile.get("employmentStatus") or "").lower()

        is_tax_intent = any(k in query_l for k in [
            "tax", "itr", "deduction", "80c", "80d", "advance tax", "refund", "regime"
        ])
        if not is_tax_intent and not profile:
            return None

        today = datetime.now()
        current_year = today.year
        itr_due = datetime(current_year, 7, 31)
        if today > itr_due:
            itr_due = datetime(current_year + 1, 7, 31)

        reminders = [
            {
                "title": "Review Form 26AS and AIS",
                "dueDate": (today + timedelta(days=7)).date().isoformat(),
                "frequency": "once",
                "category": "compliance",
            },
            {
                "title": "ITR filing deadline",
                "dueDate": itr_due.date().isoformat(),
                "frequency": "yearly",
                "category": "filing",
            },
        ]

        if "self" in employment or "business" in employment:
            reminders.extend([
                {"title": "Advance tax installment (Q1)", "dueDate": f"{today.year}-06-15", "frequency": "yearly", "category": "advance-tax"},
                {"title": "Advance tax installment (Q2)", "dueDate": f"{today.year}-09-15", "frequency": "yearly", "category": "advance-tax"},
                {"title": "Advance tax installment (Q3)", "dueDate": f"{today.year}-12-15", "frequency": "yearly", "category": "advance-tax"},
                {"title": "Advance tax installment (Q4)", "dueDate": f"{today.year + 1}-03-15", "frequency": "yearly", "category": "advance-tax"},
            ])

        steps = [
            "Validate income, TDS, and interest entries from Form 26AS/AIS.",
            f"Confirm regime choice ({tax_regime}) and simulate tax outgo before final filing.",
            "Prioritize eligible deductions/exemptions and keep proof documents organized.",
            "Review filing status 2 weeks before deadline and clear any pending tax.",
        ]

        return {
            "title": "Personalized Tax Action Plan",
            "steps": steps,
            "reminders": reminders,
        }

    def _build_response_metadata(
        self,
        query: str,
        profile: Optional[Dict],
        source_docs: List["Document"],
        source_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        confidence, confidence_label = self._calculate_confidence(source_docs, query, source_filter)
        schemes = self._build_scheme_rankings(profile, query)
        compare = self._build_compare_mode(query, schemes)
        return {
            "confidence": confidence,
            "confidenceLabel": confidence_label,
            "highlights": self._build_source_highlights(source_docs, query),
            "whyThisAnswer": self._build_why_this_answer(query, source_docs, confidence, source_filter),
            "schemes": schemes,
            "comparison": compare,
            "documentInsights": self._extract_document_insights(source_docs),
            "actionPlan": self._build_tax_action_plan(profile, query),
        }
    
    def _handle_gold_price_query(self, query: str) -> Optional[Dict]:
        """
        Handle gold price queries with direct CSV lookup.
        Returns formatted response if this is a gold query, None otherwise.
        """
        # Parse date from the query
        parsed_date = parse_date_from_query(query)
        if not parsed_date:
            return None
        
        # Check if query mentions gold
        query_lower = query.lower()
        gold_keywords = ['gold', 'sona', 'sonay', 'gold price', 'gold rate', 'gold ki', 'gold ka']
        is_gold_query = any(kw in query_lower for kw in gold_keywords)
        
        if not is_gold_query:
            return None
        
        # Get gold price lookup
        gold_lookup = get_gold_lookup()
        
        # Format the requested date for display
        requested_date_str = parsed_date.strftime("%d/%m/%Y")
        requested_date_readable = parsed_date.strftime("%d %B %Y")
        
        # Try to get exact price
        price_data = gold_lookup.get_price(parsed_date)
        
        if price_data:
            response = f"""Namaste! I am Arth-Mitra, your AI financial advisor.

Here is the gold price data for **{requested_date_readable}**:

| Metric | Value |
|--------|-------|
| **Date** | {price_data['date']} |
| **Price** | ${price_data['price']:.2f} |
| **Open** | ${price_data['open']:.2f} |
| **High** | ${price_data['high']:.2f} |
| **Low** | ${price_data['low']:.2f} |
| **Volume** | {price_data['volume']} |

*Note: Prices are in USD per troy ounce.*

If you have any questions about investing in gold (like Sovereign Gold Bonds, Gold ETFs, or physical gold) or their tax implications, feel free to ask!"""
            return {
                "response": response,
                "sources": ["gold_data.csv"],
                "confidence": 0.95,
                "confidenceLabel": "high",
                "highlights": [{"source": "gold_data.csv", "snippet": f"Gold price for {requested_date_str} is {price_data['price']} (USD/oz)."}],
                "schemes": [],
                "actionPlan": None,
                "cached": False,
            }

        # Try to get nearest price if exact not found
        nearest_data, explanation = gold_lookup.get_nearest_price(parsed_date)
        
        if nearest_data:
            response = f"""Namaste! I am Arth-Mitra, your AI financial advisor.

I don't have gold price data for **{requested_date_readable}** (this may be a holiday or weekend when markets were closed).

{explanation}: **{nearest_data['date']}**

| Metric | Value |
|--------|-------|
| **Date** | {nearest_data['date']} |
| **Price** | ${nearest_data['price']:.2f} |
| **Open** | ${nearest_data['open']:.2f} |
| **High** | ${nearest_data['high']:.2f} |
| **Low** | ${nearest_data['low']:.2f} |
| **Volume** | {nearest_data['volume']} |

*Note: Prices are in USD per troy ounce.*

If you need information about gold investment options available in India, such as Sovereign Gold Bonds (SGB), Gold ETFs, or Digital Gold, I'd be happy to help!"""
            return {
                "response": response,
                "sources": ["gold_data.csv"],
                "confidence": 0.82,
                "confidenceLabel": "high",
                "highlights": [{"source": "gold_data.csv", "snippet": f"Nearest available gold price date is {nearest_data['date']} with price {nearest_data['price']} (USD/oz)."}],
                "schemes": [],
                "actionPlan": None,
                "cached": False,
            }

        # No data available at all
        date_range = gold_lookup.get_date_range()
        range_info = ""
        if date_range[0] and date_range[1]:
            range_info = f" The available data ranges from {date_range[0]} to {date_range[1]}."
        
        response = f"""Namaste! I am Arth-Mitra, your AI financial advisor.

I don't have gold price data for **{requested_date_readable}**.{range_info}

If you have questions about current gold investment options in India or tax implications of gold investments, I would be happy to assist!"""
        return {
            "response": response,
            "sources": ["gold_data.csv"],
            "confidence": 0.45,
            "confidenceLabel": "low",
            "highlights": [{"source": "gold_data.csv", "snippet": "Requested date not available in the current dataset."}],
            "schemes": [],
            "actionPlan": None,
            "cached": False,
        }
        
    def get_response(self, query: str, profile: Optional[Dict] = None, history: Optional[List[Dict]] = None, source_filter: Optional[str] = None) -> Dict:
        """Get AI response for a user query with caching"""
        if not self._initialized:
            raise RuntimeError("Bot not initialized. Call initialize() first.")
        
        # Check cache first (skip for gold queries which need real-time data)
        if not source_filter and not is_gold_price_query(query):
            cached_response = self._response_cache.get(query, profile)
            if cached_response:
                print("⚡ Cache hit - returning cached response")
                cached_response["cached"] = True
                return cached_response
        
        # Check if this is a gold price query with a specific date
        gold_response = self._handle_gold_price_query(query)
        if gold_response:
            gold_response["response"] = self._append_sources_section(
                gold_response["response"], gold_response.get("sources", [])
            )
            return gold_response
        
        # Format user profile for context
        user_profile_text = format_user_profile(profile) if profile else ""
        chat_history_text = format_chat_history(history)
        
        # Check document count
        try:
            doc_count = self.vectorstore._collection.count()
        except:
            doc_count = 0
        
        # If no documents indexed, use direct LLM response
        if self.rag_chain is None or doc_count == 0:
            prompt = SYSTEM_PROMPT.replace("{user_profile}", user_profile_text).replace("{chat_history}", chat_history_text).replace("{context}", "No specific documents available.").replace("{question}", query)
            sources = ["General Knowledge - No documents indexed yet"]
            metadata = self._build_response_metadata(query, profile, [], source_filter)

            try:
                response = self.llm.invoke(prompt)
                answer_text = self._extract_text(response.content)
            except Exception as e:
                print(f"⚠️ LLM invocation failed in no-doc mode: {e}")
                if STRICT_LLM_ERRORS:
                    raise RuntimeError(f"LLM invocation failed in no-doc mode: {e}") from e
                answer_text = self._build_model_outage_message(metadata)

            response_text = self._append_sources_section(answer_text, sources)
            return {
                "response": response_text,
                "sources": sources,
                **metadata,
                "cached": False,
            }
        
        # Get source documents for citation
        source_docs = self._get_source_docs(query, source_filter=source_filter)

        if source_filter and not source_docs:
            response_text = (
                f"I could not find relevant content in **{source_filter}** for this question. "
                "Please verify the filename and try a more specific query."
            )
            metadata = self._build_response_metadata(query, profile, [], source_filter)
            return {
                "response": response_text,
                "sources": [source_filter],
                **metadata,
                "cached": False,
            }
        
        # Create a custom prompt with profile
        context = "\n\n".join([doc.page_content for doc in source_docs])
        prompt = SYSTEM_PROMPT.replace("{user_profile}", user_profile_text).replace("{chat_history}", chat_history_text).replace("{context}", context).replace("{question}", query)
        
        # PARALLEL: Submit LLM invocation to thread pool
        # While LLM processes (~2-10s), extract sources and build metadata
        future_llm = _executor.submit(self.llm.invoke, prompt)
        
        # Extract sources while LLM is running
        sources = []
        for doc in source_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            source_str = f"{os.path.basename(source)}"
            if page:
                source_str += f" (Page {page + 1})"
            if source_str not in sources:
                sources.append(source_str)
        
        final_sources = sources if sources else ["Knowledge Base"]
        # Build metadata in parallel with LLM
        metadata = self._build_response_metadata(query, profile, source_docs, source_filter)
        
        # Wait for LLM result
        try:
            response = future_llm.result()
            result = self._extract_text(response.content)
        except Exception as e:
            print(f"⚠️ LLM invocation failed, returning grounded fallback: {e}")
            if STRICT_LLM_ERRORS:
                raise RuntimeError(f"LLM invocation failed with retrieved context: {e}") from e
            result = self._build_model_outage_message(metadata)
        
        response_data = {
            "response": self._append_sources_section(result, final_sources),
            "sources": final_sources,
            **metadata,
            "cached": False,
        }
        
        # Cache the response for future queries
        if not source_filter:
            self._response_cache.set(query, response_data, profile)
        
        return response_data

    def stream_response(self, query: str, profile: Optional[Dict] = None, history: Optional[List[Dict]] = None, source_filter: Optional[str] = None) -> Tuple[Iterable[str], List[str], Dict[str, Any]]:
        """Stream AI response tokens for a user query."""
        if not self._initialized:
            raise RuntimeError("Bot not initialized. Call initialize() first.")

        gold_response = self._handle_gold_price_query(query)
        if gold_response:
            def gold_stream():
                yield gold_response["response"]
            return gold_stream(), gold_response.get("sources", [""]), {
                "confidence": gold_response.get("confidence", 0.7),
                "confidenceLabel": gold_response.get("confidenceLabel", "medium"),
                "whyThisAnswer": "Response is based on direct lookup from gold_data.csv date-indexed records.",
                "highlights": gold_response.get("highlights", []),
                "schemes": gold_response.get("schemes", []),
                "comparison": None,
                "documentInsights": [],
                "actionPlan": gold_response.get("actionPlan"),
                "cached": False,
            }

        user_profile_text = format_user_profile(profile) if profile else ""
        chat_history_text = format_chat_history(history)

        try:
            doc_count = self.vectorstore._collection.count()
        except:
            doc_count = 0

        if self.rag_chain is None or doc_count == 0:
            prompt = SYSTEM_PROMPT.replace("{user_profile}", user_profile_text).replace("{chat_history}", chat_history_text).replace("{context}", "No specific documents available.").replace("{question}", query)
            sources = ["General Knowledge - No documents indexed yet"]
            metadata = self._build_response_metadata(query, profile, [], source_filter)
            metadata["cached"] = False

            def no_doc_stream():
                try:
                    for chunk in self.llm.stream(prompt):
                        text = self._extract_text(chunk.content)
                        if text:
                            yield text
                except Exception as e:
                    print(f"⚠️ Streaming failed in no-doc mode: {e}")
                    if STRICT_LLM_ERRORS:
                        raise RuntimeError(f"Streaming failed in no-doc mode: {e}") from e
                    yield self._build_model_outage_message(metadata)

            return no_doc_stream(), sources, metadata

        source_docs = self._get_source_docs(query, source_filter=source_filter)

        if source_filter and not source_docs:
            metadata = self._build_response_metadata(query, profile, [], source_filter)
            metadata["cached"] = False

            def no_match_stream():
                yield (
                    f"I could not find relevant content in **{source_filter}** for this question. "
                    "Please verify the filename and try a more specific query."
                )

            return no_match_stream(), [source_filter], metadata

        # PARALLEL: Submit metadata computation to thread pool while building prompt
        future_meta = _executor.submit(
            self._build_response_metadata, query, profile, source_docs, source_filter
        )

        context = "\n\n".join([doc.page_content for doc in source_docs])
        prompt = SYSTEM_PROMPT.replace("{user_profile}", user_profile_text).replace("{chat_history}", chat_history_text).replace("{context}", context).replace("{question}", query)

        sources = []
        for doc in source_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            source_str = f"{os.path.basename(source)}"
            if page:
                source_str += f" (Page {page + 1})"
            if source_str not in sources:
                sources.append(source_str)

        final_sources = sources if sources else ["Knowledge Base"]
        metadata = future_meta.result()
        metadata["cached"] = False

        def doc_stream():
            try:
                for chunk in self.llm.stream(prompt):
                    text = self._extract_text(chunk.content)
                    if text:
                        yield text
            except Exception as e:
                print(f"⚠️ Streaming failed with retrieved context: {e}")
                if STRICT_LLM_ERRORS:
                    raise RuntimeError(f"Streaming failed with retrieved context: {e}") from e
                yield self._build_model_outage_message(metadata)

        return doc_stream(), final_sources, metadata
    
    def get_status(self) -> Dict:
        """Get bot status and statistics"""
        doc_count = 0
        if self.vectorstore:
            try:
                doc_count = self.vectorstore._collection.count()
            except:
                doc_count = 0
        
        # Determine which AI model is being used
        model_name = None
        if self.llm:
            if hasattr(self.llm, "provider_name"):
                if self.llm.provider_name == "openrouter":
                    openrouter_model = getattr(self.llm, "model_name", OPENROUTER_MODEL)
                    model_name = f"OpenRouter ({openrouter_model})"
                elif self.llm.provider_name == "gemini":
                    model_name = "Google Gemini (gemini-1.5-flash)"
                elif self.llm.provider_name == "ollama":
                    model_name = "Ollama (gemma3:1b)"

            llm_class_name = self.llm.__class__.__name__
            llm_module = self.llm.__class__.__module__

            if model_name is None and (llm_class_name == "ChatOpenAI" or "langchain_openai" in llm_module):
                chatopenai_model = getattr(self.llm, "model_name", OPENROUTER_MODEL)
                model_name = f"OpenRouter ({chatopenai_model})"
            elif model_name is None and (llm_class_name == "ChatGoogleGenerativeAI" or "langchain_google_genai" in llm_module):
                model_name = "Google Gemini (gemini-1.5-flash)"
            elif model_name is None and (llm_class_name == "ChatOllama" or "langchain_ollama" in llm_module):
                 model_name = "Ollama (gemma3:1b)"
        
        return {
            "initialized": self._initialized,
            "documents_indexed": doc_count,
            "model": model_name
        }


# Singleton instance
_bot_instance: Optional[ArthMitraBot] = None


def get_bot() -> ArthMitraBot:
    """Get or create bot singleton"""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = ArthMitraBot()
    return _bot_instance


def initialize_bot(api_key: Optional[str] = None) -> ArthMitraBot:
    """Initialize and return the bot"""
    bot = get_bot()
    if not bot._initialized:
        bot.initialize()
    return bot
