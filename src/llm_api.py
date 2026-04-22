import requests
from typing import Union, Optional

API_KEY = ""  # API key
BASE_URL = ""
MODEL = "gpt-3.5-turbo"
TIMEOUT = 10  # seconds


def chat_completion(message: str) -> Optional[str]:
    """
    Send a single-user message to the chat completion endpoint and return the assistant's reply.

    Args:
        message: The user query string.

    Returns:
        The assistant response content, or None if an error occurred.
    """
    if not API_KEY:
        print("Error: API_KEY is not set.")
        return None

    url = f"{BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        print(f"Request error: {e}")
    except ValueError:  # JSON decode error
        print("Invalid JSON response:", resp.text if 'resp' in locals() else "")
    except (KeyError, IndexError):
        print("Unexpected response format:", data if 'data' in locals() else None)
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None
