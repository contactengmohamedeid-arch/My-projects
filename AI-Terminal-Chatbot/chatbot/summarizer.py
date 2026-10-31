from chatbot.client import client
from config import MODEL


def summarize_conversation(messages):
    """Generate a summary of an older conversation."""

    if not messages:
        return ""

    summarization_prompt = [
        {
            "role": "system",
            "content": (
                "You are a conversation summarizer. "
                "Summarize the important information from the conversation. "
                "Preserve key facts, user preferences, goals, and relevant context. "
                "Be concise and do not invent information."
            )
        },
        {
            "role": "user",
            "content": f"Summarize this conversation:\n{messages}"
        }
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=summarization_prompt,
            temperature=0.3,
            timeout=30
        )

        summary = response.choices[0].message.content

        if not summary:
            raise ValueError("The model returned an empty summary.")

        return summary.strip()

    except Exception as error:
        print(f"Summarization error: {error}")
        return None