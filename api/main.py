from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from google.genai import errors as genai_errors

from api.dependencies import get_pipeline
from api.schemas import ChatRequest, ChatResponse


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="3GPP RAG Chatbot API",
    description=(
        "Grounded question answering over "
        "3GPP TS 23.501."
    ),
    version="1.0.0",
)


# ============================================================
# Source helpers
# ============================================================

def deduplicate_sources(
    sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove duplicate source entries referring to the same
    specification, version, release, and section.

    Multiple retrieved chunks may belong to the same section.
    The frontend only needs to display that section once.
    """

    unique_sources: List[Dict[str, Any]] = []
    seen = set()

    for source in sources:

        key = (
            source.get(
                "specification",
                "",
            ),
            source.get(
                "version",
                "",
            ),
            source.get(
                "release",
                "",
            ),
            source.get(
                "section",
                "",
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_sources.append(source)

    return unique_sources


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health() -> Dict[str, str]:
    """
    Basic API health check.
    """

    return {
        "status": "ok",
        "service": "3GPP RAG Chatbot",
    }


# ============================================================
# Chat endpoint
# ============================================================

@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Ask a question against the 3GPP RAG pipeline.
    """

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail={
                "error": "empty_question",
                "message": (
                    "Question cannot be empty."
                ),
            },
        )

    # --------------------------------------------------------
    # Run RAG pipeline
    # --------------------------------------------------------

    try:

        pipeline = get_pipeline()

        result = pipeline.run(
            question
        )

    # --------------------------------------------------------
    # Gemini client errors
    # --------------------------------------------------------

    except genai_errors.ClientError as exc:

        message = str(exc)

        # ----------------------------------------------------
        # Gemini quota exhausted
        # ----------------------------------------------------

        if (
            "RESOURCE_EXHAUSTED" in message
            or "quota" in message.lower()
        ):

            raise HTTPException(
                status_code=429,
                detail={
                    "error": (
                        "gemini_quota_exhausted"
                    ),
                    "message": (
                        "Gemini API quota is currently "
                        "exhausted. Please try again "
                        "after the quota resets."
                    ),
                },
            ) from exc

        # ----------------------------------------------------
        # Other Gemini client-side errors
        # ----------------------------------------------------

        raise HTTPException(
            status_code=502,
            detail={
                "error": (
                    "gemini_client_error"
                ),
                "message": message,
            },
        ) from exc

    # --------------------------------------------------------
    # Gemini server-side errors
    # --------------------------------------------------------

    except genai_errors.ServerError as exc:

        raise HTTPException(
            status_code=503,
            detail={
                "error": (
                    "gemini_unavailable"
                ),
                "message": (
                    "Gemini is temporarily "
                    "unavailable. Please try again."
                ),
            },
        ) from exc

    # --------------------------------------------------------
    # Unexpected application error
    # --------------------------------------------------------

    except Exception as exc:

        print(
            "\n" + "=" * 80
        )
        print("RAG PIPELINE ERROR")
        print("=" * 80)
        print(
            f"Type   : {type(exc).__name__}"
        )
        print(
            f"Message: {exc}"
        )
        print(
            "=" * 80
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": (
                    "rag_pipeline_error"
                ),
                "message": str(exc),
            },
        ) from exc

    # --------------------------------------------------------
    # Deduplicate sources
    # --------------------------------------------------------

    sources = deduplicate_sources(
        result.get(
            "sources",
            [],
        )
    )

    # --------------------------------------------------------
    # Return structured response
    # --------------------------------------------------------

    return ChatResponse(
        answer=result.get(
            "answer",
            "",
        ),
        allowed=result.get(
            "allowed",
            False,
        ),
        reason=result.get(
            "reason",
            "",
        ),
        confidence=float(
            result.get(
                "confidence",
                0.0,
            )
        ),
        sources=sources,
        claims=result.get(
            "claims",
            [],
        ),
        claim_validation=result.get(
            "claim_validation",
            {},
        ),
        completeness_validation=result.get(
            "completeness_validation",
            {},
        ),
    )