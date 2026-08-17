from __future__ import annotations

from retrieval.bm25_store import BM25Store
from retrieval.embeddings import EmbeddingModel
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.section_aware_retriever import (
    SectionAwareRetriever,
)
from retrieval.vector_store import VectorStore

from generation.evidence_gate import EvidenceGate
from generation.llm_client import LLMClient
from generation.rag_pipeline import RAGPipeline


_pipeline = None


def build_pipeline() -> RAGPipeline:
    """
    Build the complete RAG pipeline.
    """

    embedding_model = EmbeddingModel()

    vector_store = VectorStore(
        embedding_model
    )
    vector_store.load()

    bm25_store = BM25Store()
    bm25_store.load()

    hybrid_retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        bm25_store=bm25_store,
    )

    reranker = Reranker()

    retriever = SectionAwareRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
    )

    evidence_gate = EvidenceGate()

    llm_client = LLMClient()

    return RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        evidence_gate=evidence_gate,
    )


def get_pipeline() -> RAGPipeline:
    """
    Return a singleton RAG pipeline.

    This prevents FAISS, BM25, reranker and Gemini
    clients from being recreated for every request.
    """

    global _pipeline

    if _pipeline is None:
        _pipeline = build_pipeline()

    return _pipeline