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
    Keep the system prompt and the most recent complete
    user-assistant conversation turns.
    """

    system_message = None
    conversation = []

    # Separate system prompt from conversation
    for message in messages:
        if message["role"] == "system":
            system_message = message
        else:
            conversation.append(message)

    result = [system_message] if system_message else []

    remaining_tokens = max_tokens - count_tokens(result)

    if remaining_tokens <= 0:
        raise ValueError("System prompt exceeds the token budget.")

    # Group messages into conversation turns
    turns = []
    current_turn = []

    for message in conversation:
        if message["role"] == "user":
            if current_turn:
                turns.append(current_turn)

            current_turn = [message]

        elif message["role"] == "assistant":
            if current_turn:
                current_turn.append(message)
                turns.append(current_turn)
                current_turn = []

    # Ignore incomplete turns for now
    # Process newest complete turns first
    selected_turns = []

    for turn in reversed(turns):
        turn_tokens = count_tokens(turn)

        if turn_tokens <= remaining_tokens:
            selected_turns.insert(0, turn)
            remaining_tokens -= turn_tokens
        else:
            break

    # Flatten turns into a single message list
    for turn in selected_turns:
        result.extend(turn)

    return result