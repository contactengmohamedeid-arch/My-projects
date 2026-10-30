from chatbot.client import client
from chatbot.storage import save_conversation
from config import MODEL, TEMPERATURE
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError
)

def chat(messages, user_input):
    """Send a message to the AI and stream its response."""

    # Keep the original history safe
    updated_messages = messages.copy()

    updated_messages.append({
        "role": "user",
        "content": user_input
    })

    assistant_response = ""

    try:
        print("\nAI: ", end="", flush=True)

        response = client.chat.completions.create(
            model=MODEL,
            messages=updated_messages,
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

        updated_messages.append({
            "role": "assistant",
            "content": assistant_response
        })

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