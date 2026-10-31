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
    """Keep recent complete turns and the latest unanswered message."""

    system_message = next(
        (m for m in messages if m["role"] == "system"),
        None
    )

    conversation = [
        m for m in messages
        if m["role"] != "system"
    ]

    result = [system_message] if system_message else []

    remaining_tokens = max_tokens - count_tokens(result)

    if remaining_tokens <= 0:
        raise ValueError("System prompt exceeds token budget.")

    # Identify the latest unanswered user message
    pending_message = None

    if conversation and conversation[-1]["role"] == "user":
        pending_message = conversation.pop()

    # Group historical messages into complete turns
    turns = []
    current_turn = []

    for message in conversation:
        if message["role"] == "user":
            if current_turn:
                turns.append(current_turn)

            current_turn = [message]

        elif message["role"] == "assistant" and current_turn:
            current_turn.append(message)
            turns.append(current_turn)
            current_turn = []

    # Add pending message to the context first
    if pending_message:
        pending_tokens = count_tokens([pending_message])

        if pending_tokens > remaining_tokens:
            raise ValueError("Latest user message exceeds token budget.")

        result.append(pending_message)
        remaining_tokens -= pending_tokens

    # Add recent complete turns
    selected_turns = []

    for turn in reversed(turns):
        turn_tokens = count_tokens(turn)

        if turn_tokens <= remaining_tokens:
            selected_turns.insert(0, turn)
            remaining_tokens -= turn_tokens
        else:
            break

    # Put historical turns before the pending message
    if pending_message:
        result = result[:-1]

    for turn in selected_turns:
        result.extend(turn)

    if pending_message:
        result.append(pending_message)

    return result

def build_context(system_prompt, summary, recent_messages):
    """Build the final context sent to the LLM."""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    if summary:
        messages.append({
            "role": "system",
            "content": (
                "Here is a summary of the earlier conversation. "
                "Use it as background context:\n\n"
                f"{summary}"
            )
        })

    messages.extend(recent_messages)

    return messages


def select_messages_for_summary(messages, keep_recent_turns=3):
    """Separate older complete turns from recent messages."""

    system_message = next(
        (m for m in messages if m["role"] == "system"),
        None
    )

    conversation = [
        m for m in messages
        if m["role"] != "system"
    ]

    turns = []
    current_turn = []

    for message in conversation:
        if message["role"] == "user":
            if current_turn:
                turns.append(current_turn)

            current_turn = [message]

        elif message["role"] == "assistant" and current_turn:
            current_turn.append(message)
            turns.append(current_turn)
            current_turn = []

    # Preserve an unfinished turn (a user message without an AI response)
    if current_turn:
        turns.append(current_turn)

    recent_turns = turns[-keep_recent_turns:]
    old_turns = turns[:-keep_recent_turns]

    old_messages = [
        message
        for turn in old_turns
        for message in turn
    ]

    recent_messages = [
        message
        for turn in recent_turns
        for message in turn
    ]

    return old_messages, recent_messages