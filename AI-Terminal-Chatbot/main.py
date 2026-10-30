import os
from openai import OpenAI
from dotenv import load_dotenv


# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found! Check your .env file.")


# ==========================================
# 2. INITIALIZE GROQ CLIENT
# ==========================================

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


# ==========================================
# 3. INITIALIZE CONVERSATION HISTORY
# ==========================================

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Give clear, accurate, and concise answers."
        )
    }
]


# ==========================================
# 4. CHATBOT INTERFACE
# ==========================================

print("=" * 45)
print("           AI TERMINAL CHATBOT")
print("=" * 45)
print("Type 'exit' to quit.\n")


# ==========================================
# 5. MAIN CHAT LOOP
# ==========================================

while True:

    user_input = input("You: ").strip()

    # Ignore empty messages
    if not user_input:
        continue

    # Exit condition
    if user_input.lower() == "exit":
        print("\nAI: Goodbye!")
        break

    # Add user message to conversation history
    messages.append({
        "role": "user",
        "content": user_input
    })

    try:

        print("\nAI: ", end="", flush=True)

        # ======================================
        # 6. STREAMING API REQUEST
        # ======================================

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7,
            stream=True
        )

        # ======================================
        # 7. RECEIVE STREAMING CHUNKS
        # ======================================

        assistant_response = ""

        for chunk in response:

            content = chunk.choices[0].delta.content

            if content is not None:

                print(content, end="", flush=True)

                assistant_response += content

        print("\n")

        # ======================================
        # 8. UPDATE CONVERSATION HISTORY
        # ======================================

        messages.append({
            "role": "assistant",
            "content": assistant_response
        })

    except Exception as e:

        print(f"\n\nError: {e}\n")

        # Remove the unanswered user message
        messages.pop()


# ==========================================
# END OF PROGRAM
# ==========================================