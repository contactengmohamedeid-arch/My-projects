import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# =========================
# API Configuration
# =========================

API_KEY = os.getenv("GROQ_API_KEY")

BASE_URL = "https://api.groq.com/openai/v1"


# =========================
# Model Configuration
# =========================

MODEL = "openai/gpt-oss-20b"

TEMPERATURE = 0.7


# =========================
# Conversation Configuration
# =========================

SYSTEM_PROMPT = "You are a helpful AI assistant."


# =========================
# Storage Configuration
# =========================

CONVERSATION_DIR = r"F:\Coding\Cache\AI-Terminal-Chatbot"

os.makedirs(CONVERSATION_DIR, exist_ok=True)

CONVERSATION_FILE = os.path.join(
    CONVERSATION_DIR,
    "conversations.json"
)


# =========================
# Configuration Validation
# =========================

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Check your .env file."
    )