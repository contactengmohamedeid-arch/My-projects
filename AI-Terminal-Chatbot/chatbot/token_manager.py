import tiktoken

from config import MODEL


def get_encoding():
    """Return the tokenizer encoding for the current model."""

    try:
        return tiktoken.encoding_for_model(MODEL)

    except KeyError:
        # Fallback for models not recognized by tiktoken
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(messages):
    """Count the total number of tokens in a conversation."""

    encoding = get_encoding()

    total_tokens = 0

    for message in messages:
        content = message.get("content", "")
        total_tokens += len(encoding.encode(content))

    return total_tokens

def manage_context(messages, max_tokens=4000):
    """
    Keep the system prompt and the most recent messages
    within the specified token budget.
    """

    system_message = None
    conversation = []

    # Separate system message from conversation
    for message in messages:
        if message["role"] == "system":
            system_message = message
        else:
            conversation.append(message)

    # Start with the system prompt
    result = [system_message] if system_message else []

    # Calculate remaining token budget
    remaining_tokens = max_tokens - count_tokens(result)

    # Keep recent messages first
    recent_messages = []

    for message in reversed(conversation):
        message_tokens = count_tokens([message])

        if message_tokens <= remaining_tokens:
            recent_messages.insert(0, message)
            remaining_tokens -= message_tokens
        else:
            break

    result.extend(recent_messages)

    return result