from __future__ import annotations

import os

from google import genai


def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    model = os.getenv(
        "RAG_LLM_MODEL",
        "gemini-3.6-flash",
    )

    print("MODEL:", model)

    client = genai.Client(
        api_key=api_key
    )

    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: GEMINI_OK",
    )

    print("GEMINI_OK")
    print(response.text)


if __name__ == "__main__":
    main()