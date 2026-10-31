from chatbot.conversation import chat
from chatbot.storage import load_conversation, reset_conversation
from chatbot.storage import load_conversation, reset_conversation, clear_summary

def main():
    messages = load_conversation()

    print("=" * 40)
    print("       AI TERMINAL CHATBOT")
    print("=" * 40)
    print("Commands:")
    print("  exit  - Close the chatbot")
    print("  reset - Start a new conversation")
    print("=" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            messages = reset_conversation()
            clear_summary()
            print("Conversation and summary cleared.")
            continue

        messages = chat(messages, user_input)


if __name__ == "__main__":
    main()