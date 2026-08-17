from __future__ import annotations

import streamlit as st

from generation.evidence_gate import EvidenceGate
from generation.llm_client import LLMClient
from generation.rag_pipeline import RAGPipeline

from retrieval.bm25_store import BM25Store
from retrieval.embeddings import EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.section_aware_retriever import (
    SectionAwareRetriever,
)
from retrieval.vector_store import VectorStore


# ============================================================
# Page configuration
# ============================================================

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

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Build RAG pipeline once
# ============================================================

@st.cache_resource
def build_pipeline() -> RAGPipeline:
    """
    Build the complete RAG pipeline once and cache it.

    Streamlit reruns the script whenever the user interacts
    with the application, so caching prevents FAISS, BM25,
    embedding and reranker models from being recreated
    for every question.
    """

    embedding_model = EmbeddingModel()

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    vector_store = VectorStore(
        embedding_model
    )

    vector_store.load()

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    bm25_store = BM25Store()
    bm25_store.load()

    # --------------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    # --------------------------------------------------------
    # Reranker
    # --------------------------------------------------------

    reranker = Reranker()

    # --------------------------------------------------------
    # Section-aware retriever
    # --------------------------------------------------------

    retriever = SectionAwareRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )

    # --------------------------------------------------------
    # Evidence gate
    # --------------------------------------------------------

    evidence_gate = EvidenceGate()

    # --------------------------------------------------------
    # Gemini client
    # --------------------------------------------------------

    llm_client = LLMClient()

    # --------------------------------------------------------
    # Complete pipeline
    # --------------------------------------------------------

    return RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        evidence_gate=evidence_gate,
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
# Load pipeline
# ============================================================

try:

    pipeline = build_pipeline()

except Exception as exc:

    st.error(
        "Unable to initialize the RAG pipeline."
    )

    st.exception(exc)

    st.stop()


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
    st.write("✅ Completeness Validation")

    st.divider()

    st.caption(
        "Powered by 3GPP TS 23.501 grounded retrieval."
    )

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# Helper: display sources
# ============================================================

def display_sources(
    sources: list[dict],
) -> None:
    """
    Display deduplicated source sections.
    """

    if not sources:
        return

    # --------------------------------------------------------
    # Deduplicate by specification/version/release/section
    # --------------------------------------------------------

    unique_sources = []

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

        unique_sources.append(
            source
        )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    with st.expander(
        "📚 Sources",
        expanded=False,
    ):

        for source in unique_sources:

            st.markdown(
                f"""
                <div class="source-card">
                    <strong>
                        Section {source.get('section', '')}
                    </strong>
                    <br>
                    {source.get('section_title', '')}
                    <br>
                    <small>
                        {source.get('specification', '')}
                        &nbsp; V{source.get('version', '')}
                        &nbsp; {source.get('release', '')}
                    </small>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# Helper: display claims
# ============================================================

def display_claims(
    claims: list[dict],
    claim_validation: dict,
) -> None:
    """
    Display generated claims and their validation summary.
    """

    if not claims:
        return

    with st.expander(
        "🔎 Validated Claims",
        expanded=False,
    ):

        valid_claims = claim_validation.get(
            "valid_claims",
            0,
        )

        total_claims = claim_validation.get(
            "total_claims",
            0,
        )

        st.write(
            f"Validated: "
            f"{valid_claims}/{total_claims}"
        )

        for claim in claims:

            claim_text = claim.get(
                "text",
                "",
            )

            section = claim.get(
                "section",
                "",
            )

            st.markdown(
                f"- {claim_text} "
                f"**[Section {section}]**"
            )


# ============================================================
# Helper: display completeness
# ============================================================

def display_completeness(
    completeness: dict,
) -> None:
    """
    Display completeness information for exhaustive/list
    questions.
    """

    if not completeness.get(
        "required",
        False,
    ):
        return

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

        expected_items = completeness.get(
            "expected_items",
            0,
        )

        covered_items = completeness.get(
            "covered_items",
            0,
        )

        st.write(
            f"Items covered: "
            f"{covered_items}/{expected_items}"
        )

        missing = completeness.get(
            "missing_items",
            [],
        )

        if missing:

            st.warning(
                "Some expected items were not covered:"
            )

            for item in missing:

                st.write(
                    f"- {item}"
                )

        else:

            st.success(
                "All expected items were covered."
            )


# ============================================================
# Display conversation history
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

            # ------------------------------------------------
            # Grounding status
            # ------------------------------------------------

            if metadata.get(
                "allowed",
                False,
            ):

                confidence = metadata.get(
                    "confidence",
                    0.0,
                )

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
                        "⚠️ Answer not generated "
                        "from sufficient evidence."
                    )
                )

            # ------------------------------------------------
            # Sources
            # ------------------------------------------------

            display_sources(
                metadata.get(
                    "sources",
                    [],
                )
            )

            # ------------------------------------------------
            # Claims
            # ------------------------------------------------

            display_claims(
                metadata.get(
                    "claims",
                    [],
                ),
                metadata.get(
                    "claim_validation",
                    {},
                ),
            )

            # ------------------------------------------------
            # Completeness
            # ------------------------------------------------

            display_completeness(
                metadata.get(
                    "completeness_validation",
                    {},
                )
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
    # User message
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
    # Assistant response
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching 3GPP evidence and generating answer..."
        ):

            try:

                result = pipeline.run(
                    question
                )

            except Exception as exc:

                error_text = str(
                    exc
                )

                # --------------------------------------------
                # Gemini quota
                # --------------------------------------------

                if (
                    "RESOURCE_EXHAUSTED"
                    in error_text
                    or "quota"
                    in error_text.lower()
                ):

                    st.warning(
                        "⚠️ Gemini API quota is "
                        "currently exhausted."
                    )

                    st.info(
                        "The retrieval pipeline is "
                        "available, but answer generation "
                        "requires Gemini. Please try again "
                        "after the Gemini quota resets."
                    )

                    st.stop()

                # --------------------------------------------
                # Gemini temporarily unavailable
                # --------------------------------------------

                if (
                    "503" in error_text
                    or "UNAVAILABLE"
                    in error_text
                ):

                    st.warning(
                        "⚠️ Gemini is temporarily "
                        "unavailable. Please try again later."
                    )

                    st.stop()

                # --------------------------------------------
                # Invalid API key
                # --------------------------------------------

                if (
                    "API_KEY_INVALID"
                    in error_text
                ):

                    st.error(
                        "❌ Gemini API key is invalid. "
                        "Check the Streamlit secret "
                        "configuration."
                    )

                    st.stop()

                # --------------------------------------------
                # Unexpected error
                # --------------------------------------------

                st.error(
                    "❌ Unexpected RAG pipeline error."
                )

                st.exception(
                    exc
                )

                st.stop()

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            answer = result.get(
                "answer",
                "",
            )

            allowed = result.get(
                "allowed",
                False,
            )

            confidence = result.get(
                "confidence",
                0.0,
            )

            reason = result.get(
                "reason",
                "",
            )

            # ------------------------------------------------
            # Answer
            # ------------------------------------------------

            st.markdown(
                answer
            )

            # ------------------------------------------------
            # Grounding status
            # ------------------------------------------------

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
                        "⚠️ Insufficient evidence "
                        "for a grounded answer."
                    )
                )

                if reason:

                    st.caption(
                        f"Reason: {reason}"
                    )

            # ------------------------------------------------
            # Sources
            # ------------------------------------------------

            display_sources(
                result.get(
                    "sources",
                    [],
                )
            )

            # ------------------------------------------------
            # Claims
            # ------------------------------------------------

            display_claims(
                result.get(
                    "claims",
                    [],
                ),
                result.get(
                    "claim_validation",
                    {},
                ),
            )

            # ------------------------------------------------
            # Completeness
            # ------------------------------------------------

            display_completeness(
                result.get(
                    "completeness_validation",
                    {},
                )
            )

            # ------------------------------------------------
            # Save assistant message
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "metadata": result,
                }
            )