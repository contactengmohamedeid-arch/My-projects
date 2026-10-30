from openai import OpenAI

from config import API_KEY, BASE_URL


def create_client():
    """Initialize and return the Groq API client."""

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    return client


# Create a reusable client
client = create_client()