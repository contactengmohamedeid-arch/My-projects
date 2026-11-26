# AI Terminal Chatbot
Command-line chatbot with persistent memory. It saves the conversation to a JSON file, keeps recent turns within a 4,000-token budget, and when the history passes 3,000 tokens it asks the model to summarize older turns, so context is kept without resending everything.

Commands: `exit` closes the chatbot, `reset` clears the conversation and summary.

**Structure:** `main.py` (loop), `config.py` (settings), `chatbot/` with `client.py`, `conversation.py`, `storage.py`, `summarizer.py`, `token_manager.py`.

**Setup:** `pip install openai python-dotenv tiktoken`, create a `.env` file with `GROQ_API_KEY=your_key`, run `python main.py`. Uses model `openai/gpt-oss-20b` through Groq's OpenAI-compatible API.
**Note:** `config.py` saves chats to a hardcoded Windows folder (`F:\Coding\Cache\AI-Terminal-Chatbot`); change `CONVERSATION_DIR` for your machine.
