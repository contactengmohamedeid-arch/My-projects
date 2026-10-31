from chatbot.client import client
from chatbot.storage import (
    save_conversation,
    load_summary,
    save_summary
)
from chatbot.summarizer import summarize_conversation

from chatbot.token_manager import (
    count_tokens,
    select_messages_for_summary,
    build_context ,
    manage_context
)

from config import (
    MODEL,
    TEMPERATURE,
    SYSTEM_PROMPT,
    SUMMARY_TRIGGER_TOKENS
)

from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError
)

def update_summary(messages):
    """Summarize older conversation turns when needed."""

    old_messages, recent_messages = select_messages_for_summary(
        messages,
        keep_recent_turns=3
    )

    current_summary = load_summary()

    if not old_messages:
        return current_summary, recent_messages

    if count_tokens(old_messages) < SUMMARY_TRIGGER_TOKENS:
        return current_summary, messages

    summarization_input = []

    if current_summary:
        summarization_input.append({
            "role": "system",
            "content": f"Previous summary:\n{current_summary}"
        })

    summarization_input.extend(old_messages)

    new_summary = summarize_conversation(summarization_input)

    if new_summary is None:
        return current_summary, messages

    if save_summary(new_summary):
        return new_summary, recent_messages

    return current_summary, messages

def chat(messages, user_input):
    """Send a message to the AI using summarized conversation context."""

    working_messages = messages.copy()

    working_messages.append({
        "role": "user",
        "content": user_input
    })

    # Update summary and select relevant messages
    summary, recent_messages = update_summary(working_messages)

    # Remove system messages to avoid duplicates
    active_messages = [
        message for message in recent_messages
        if message["role"] != "system"
    ]

    # Build the final context sent to the AI
    context_messages = build_context(
        SYSTEM_PROMPT,
        summary,
        active_messages
    )

    assistant_response = ""

    try:
        print("\nAI: ", end="", flush=True)

        response = client.chat.completions.create(
            model=MODEL,
            messages=context_messages,
            stream=True,
            temperature=TEMPERATURE,
            timeout=30
        )

        for chunk in response:
            content = chunk.choices[0].delta.content

            if content:
                print(content, end="", flush=True)
                assistant_response += content

        print()

        if not assistant_response:
            print("Error: The AI returned an empty response.")
            return messages

        active_messages.append({
            "role": "assistant",
            "content": assistant_response
        })

        updated_messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + active_messages

        save_conversation(updated_messages)

        return updated_messages

    except RateLimitError:
        print("\nError: Rate limit exceeded.")
        print("Please wait before sending another request.")

    except APITimeoutError:
        print("\nError: Request timed out.")
        print("Please try again.")

    except APIConnectionError:
        print("\nError: Could not connect to the API.")
        print("Check your internet connection.")

    except APIStatusError as error:
        print(f"\nAPI Error: Status code {error.status_code}")
        print(error.message)

    except Exception as error:
        print(f"\nUnexpected error: {error}")

    return messages