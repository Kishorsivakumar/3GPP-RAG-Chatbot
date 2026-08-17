from __future__ import annotations

import requests
import streamlit as st


import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/api/v1/chat",
)


st.set_page_config(
    page_title="3GPP RAG Chatbot",
    page_icon="📡",
    layout="wide",
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        margin-bottom: 1.5rem;
    }

    .source-card {
        padding: 0.8rem 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        margin-bottom: 0.6rem;
    }

    .status-good {
        padding: 0.7rem 1rem;
        border-radius: 10px;
        background: #eaf7ee;
        border: 1px solid #b7dfc1;
    }

    .status-bad {
        padding: 0.7rem 1rem;
        border-radius: 10px;
        background: #fff4e5;
        border: 1px solid #f0c36d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">📡 3GPP RAG Chatbot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Ask grounded questions about 3GPP TS 23.501"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("System")

    st.write(
        "Specification: **3GPP TS 23.501**"
    )

    st.write(
        "Release: **Rel-20**"
    )

    st.write(
        "Version: **20.2.0**"
    )

    st.divider()

    st.subheader("Pipeline")

    st.write("✅ Hybrid Retrieval")
    st.write("✅ BM25")
    st.write("✅ FAISS")
    st.write("✅ Reranking")
    st.write("✅ Section-aware Retrieval")
    st.write("✅ Evidence Gate")
    st.write("✅ Claim Validation")

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# Display conversation
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("metadata")
        ):

            metadata = message[
                "metadata"
            ]

            # ----------------------------------------------
            # Grounding status
            # ----------------------------------------------

            if metadata.get(
                "allowed",
                False,
            ):

                st.success(
                    (
                        "Grounded answer • "
                        f"Confidence: "
                        f"{metadata.get('confidence', 0):.1%}"
                    )
                )

            # ----------------------------------------------
            # Sources
            # ----------------------------------------------

            sources = metadata.get(
                "sources",
                [],
            )

            if sources:

                with st.expander(
                    "📚 Sources",
                    expanded=False,
                ):

                    for source in sources:

                        st.markdown(
                            f"""
                            <div class="source-card">
                            <strong>
                            Section {source.get('section', '')}
                            </strong><br>
                            {source.get('section_title', '')}<br>
                            <small>
                            {source.get('specification', '')}
                            &nbsp; V{source.get('version', '')}
                            &nbsp; {source.get('release', '')}
                            </small>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # ----------------------------------------------
            # Claims
            # ----------------------------------------------

            claims = metadata.get(
                "claims",
                [],
            )

            claim_validation = metadata.get(
                "claim_validation",
                {},
            )

            if claims:

                with st.expander(
                    "🔎 Validated Claims",
                    expanded=False,
                ):

                    st.write(
                        f"Validated: "
                        f"{claim_validation.get('valid_claims', 0)}"
                        f"/"
                        f"{claim_validation.get('total_claims', 0)}"
                    )

                    for claim in claims:

                        st.markdown(
                            f"- {claim.get('text', '')} "
                            f"**[Section {claim.get('section', '')}]**"
                        )

            # ----------------------------------------------
            # Completeness
            # ----------------------------------------------

            completeness = metadata.get(
                "completeness_validation",
                {},
            )

            if completeness.get(
                "required",
                False,
            ):

                with st.expander(
                    "📋 Completeness",
                    expanded=False,
                ):

                    coverage = completeness.get(
                        "coverage",
                        0.0,
                    )

                    st.write(
                        f"Coverage: {coverage:.1%}"
                    )

                    missing = completeness.get(
                        "missing_items",
                        [],
                    )

                    if missing:

                        st.write(
                            "Missing items:"
                        )

                        for item in missing:
                            st.write(
                                f"- {item}"
                            )


# ============================================================
# Chat input
# ============================================================

question = st.chat_input(
    "Ask a question about 3GPP TS 23.501..."
)


if question:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

        st.stop()

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Call FastAPI
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching 3GPP evidence..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": question
                    },
                    timeout=180,
                )

                # =================================================
                # 200: successful request
                # =================================================

                if response.status_code == 200:

                    data = response.json()

                    answer = data.get(
                        "answer",
                        "",
                    )

                    allowed = data.get(
                        "allowed",
                        False,
                    )

                    reason = data.get(
                        "reason",
                        "",
                    )

                    confidence = data.get(
                        "confidence",
                        0.0,
                    )

                    # ---------------------------------------------
                    # Main answer
                    # ---------------------------------------------

                    st.markdown(
                        answer
                    )

                    # ---------------------------------------------
                    # Grounding status
                    # ---------------------------------------------

                    if allowed:

                        st.success(
                            (
                                "✅ Grounded answer • "
                                f"Confidence: "
                                f"{confidence:.1%}"
                            )
                        )

                    else:

                        st.warning(
                            (
                                "⚠️ Evidence was not "
                                "sufficient."
                            )
                        )

                    # ---------------------------------------------
                    # Sources
                    # ---------------------------------------------

                    sources = data.get(
                        "sources",
                        [],
                    )

                    if sources:

                        with st.expander(
                            "📚 Sources"
                        ):

                            for source in sources:

                                st.markdown(
                                    f"""
                                    <div class="source-card">
                                    <strong>
                                    Section {source.get('section', '')}
                                    </strong><br>
                                    {source.get('section_title', '')}<br>
                                    <small>
                                    {source.get('specification', '')}
                                    &nbsp; V{source.get('version', '')}
                                    &nbsp; {source.get('release', '')}
                                    </small>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # ---------------------------------------------
                    # Claims
                    # ---------------------------------------------

                    claims = data.get(
                        "claims",
                        [],
                    )

                    claim_validation = data.get(
                        "claim_validation",
                        {},
                    )

                    if claims:

                        with st.expander(
                            "🔎 Validated Claims"
                        ):

                            st.write(
                                f"Validated: "
                                f"{claim_validation.get('valid_claims', 0)}"
                                f"/"
                                f"{claim_validation.get('total_claims', 0)}"
                            )

                            for claim in claims:

                                st.markdown(
                                    f"- {claim.get('text', '')} "
                                    f"**[Section {claim.get('section', '')}]**"
                                )

                    # ---------------------------------------------
                    # Completeness
                    # ---------------------------------------------

                    completeness = data.get(
                        "completeness_validation",
                        {},
                    )

                    if completeness.get(
                        "required",
                        False,
                    ):

                        with st.expander(
                            "📋 Completeness"
                        ):

                            coverage = completeness.get(
                                "coverage",
                                0.0,
                            )

                            st.write(
                                f"Coverage: {coverage:.1%}"
                            )

                            missing = completeness.get(
                                "missing_items",
                                [],
                            )

                            if missing:

                                st.write(
                                    "Missing items:"
                                )

                                for item in missing:
                                    st.write(
                                        f"- {item}"
                                    )

                    # ---------------------------------------------
                    # Save assistant message
                    # ---------------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "metadata": data,
                        }
                    )

                # =================================================
                # 400
                # =================================================

                elif response.status_code == 400:

                    try:
                        error_data = response.json()
                    except ValueError:
                        error_data = {}

                    message = (
                        error_data
                        .get("detail", {})
                        .get(
                            "message",
                            "Invalid request.",
                        )
                        if isinstance(
                            error_data.get(
                                "detail"
                            ),
                            dict,
                        )
                        else "Invalid request."
                    )

                    st.error(
                        f"❌ {message}"
                    )

                # =================================================
                # 429 Gemini quota
                # =================================================

                elif response.status_code == 429:

                    try:
                        error_data = response.json()
                    except ValueError:
                        error_data = {}

                    detail = error_data.get(
                        "detail",
                        {},
                    )

                    message = (
                        detail.get(
                            "message",
                            "Gemini API quota is currently exhausted.",
                        )
                        if isinstance(
                            detail,
                            dict,
                        )
                        else str(detail)
                    )

                    st.warning(
                        f"⚠️ {message}"
                    )

                    st.info(
                        "Your retrieval and evidence "
                        "pipeline are still available. "
                        "Please retry after the Gemini "
                        "quota resets."
                    )

                # =================================================
                # 502
                # =================================================

                elif response.status_code == 502:

                    st.error(
                        "❌ Gemini client error. "
                        "Please check the Gemini API configuration."
                    )

                # =================================================
                # 503
                # =================================================

                elif response.status_code == 503:

                    st.warning(
                        "⚠️ Gemini is temporarily unavailable. "
                        "Please try again later."
                    )

                # =================================================
                # Other errors
                # =================================================

                else:

                    st.error(
                        (
                            f"❌ API request failed "
                            f"with status "
                            f"{response.status_code}."
                        )
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to FastAPI. "
                    "Start the backend with:\n\n"
                    "`uvicorn api.main:app --reload`"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "❌ The request timed out. "
                    "The backend may still be processing."
                )

            except Exception as exc:

                st.error(
                    f"❌ Unexpected error: {exc}"
                )