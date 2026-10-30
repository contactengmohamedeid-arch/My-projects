import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found!")

# 2. Initialize Groq client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# 3. Initialize conversation history
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Give clear and concise answers."
    }
]

# 4. Chatbot interface
print("=" * 40)
print("       AI TERMINAL CHATBOT")
print("=" * 40)
print("Type 'exit' to quit.\n")

while True:

    # Get user input
    user_input = input("You: ").strip()

    if not user_input:
        continue

    # Exit condition
    if user_input.lower() == "exit":
        print("\nAI: Goodbye!")
        break

    # Add user message to history
    messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        print("\nThinking...")

        # Send conversation history to LLM
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.7
        )

        # Extract AI response
        assistant_response = response.choices[0].message.content

        # Add AI response to history
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })

        print(f"\nAI: {assistant_response}\n")

    except Exception as e:
        print(f"\nError: {e}\n")