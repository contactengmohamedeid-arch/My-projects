import os
import json

from config import CONVERSATION_FILE, SYSTEM_PROMPT


def save_conversation(messages):
    """Save conversation history to a JSON file."""

    try:
        with open(CONVERSATION_FILE, "w", encoding="utf-8") as file:
            json.dump(
                messages,
                file,
                indent=4,
                ensure_ascii=False
            )

        return True

    except OSError as error:
        print(f"Error saving conversation: {error}")
        return False


def load_conversation():
    """Load conversation history from a JSON file."""

    if not os.path.exists(CONVERSATION_FILE):
        print("No previous conversation found.")
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        with open(CONVERSATION_FILE, "r", encoding="utf-8") as file:
            messages = json.load(file)

        # Validate the main structure
        if not isinstance(messages, list):
            raise ValueError("Conversation must be a list.")

        # Validate each message
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Invalid message format.")

            if message.get("role") not in ["system", "user", "assistant"]:
                raise ValueError("Invalid message role.")

            if not isinstance(message.get("content"), str):
                raise ValueError("Invalid message content.")

        # Ensure system prompt exists at the beginning
        if not messages or messages[0]["role"] != "system":
            messages.insert(
                0,
                {"role": "system", "content": SYSTEM_PROMPT}
            )

        return messages

    except (json.JSONDecodeError, ValueError, OSError) as error:
        print(f"Error loading conversation: {error}")
        return [{"role": "system", "content": SYSTEM_PROMPT}]


def reset_conversation():
    """Reset conversation history."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    save_conversation(messages)

    return messages