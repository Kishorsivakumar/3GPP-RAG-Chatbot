from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.section_expander import SectionExpander
from retrieval.relevance import relevance_score
from retrieval.query_expander import expand_query


CHUNKS_PATH = Path(
    r"data\processed\chunks.json"
)


class SectionAwareRetriever:
    """
    Retrieval pipeline:

        User query
            ↓
        Query expansion
            ↓
        Hybrid retrieval
        (FAISS + BM25)
            ↓
        Cross-encoder reranking
            ↓
        Section expansion
            ↓
        Cross-encoder reranking
            ↓
        3GPP relevance scoring
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
    ):

        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

        with CHUNKS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            chunks = json.load(file)

        self.expander = SectionExpander(
            chunks
        )

    def retrieve(
        self,
        query: str,
        candidate_k: int = 20,
        final_k: int = 5,
    ) -> List[Dict]:

        # =====================================================
        # 1. Expand the user's query
        # =====================================================

        expanded_query = expand_query(
            query
        )

        # =====================================================
        # 2. Hybrid retrieval
        #
        # Use expanded query for:
        #     FAISS
        #     BM25
        # =====================================================

        candidates = (
            self.hybrid_retriever.retrieve(
                query=expanded_query,
                top_k_dense=20,
                top_k_bm25=20,
                final_k=candidate_k,
            )
        )

        # =====================================================
        # 3. First reranking
        #
        # IMPORTANT:
        # Use the ORIGINAL user question here.
        # The reranker should judge whether a passage
        # actually answers what the user asked.
        # =====================================================

        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=candidate_k,
        )

        # =====================================================
        # 4. Section expansion
        #
        # Bring neighboring chunks from the same
        # 3GPP section.
        # =====================================================

        expanded_candidates = (
            self.expander.expand(
                reranked,
                max_per_section=3,
            )
        )

        # =====================================================
        # 5. Rerank expanded evidence
        # =====================================================

        final_candidates = (
            self.reranker.rerank(
                query=query,
                candidates=expanded_candidates,
                top_k=len(expanded_candidates),
            )
        )

        # =====================================================
        # 6. Apply 3GPP-specific relevance scoring
        # =====================================================

        for result in final_candidates:

            result["final_score"] = (
                relevance_score(
                    query=query,
                    chunk=result,
                    reranker_score=result[
                        "reranker_score"
                    ],
                )
            )

        # =====================================================
        # 7. Sort by final score
        # =====================================================

        final_candidates.sort(
            key=lambda x: x["final_score"],
            reverse=True,
        )

        return final_candidates[:final_k]