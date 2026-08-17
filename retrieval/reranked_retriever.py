from __future__ import annotations

from typing import Dict, List

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.relevance import relevance_score


class RerankedRetriever:

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        candidate_k: int = 20,
        final_k: int = 5,
    ) -> List[Dict]:

        candidates = (
            self.hybrid_retriever.retrieve(
                query=query,
                top_k_dense=20,
                top_k_bm25=20,
                final_k=candidate_k,
            )
        )

        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=candidate_k,
        )

        for result in reranked:

            result["final_score"] = (
                relevance_score(
                    query=query,
                    chunk=result,
                    reranker_score=result[
                        "reranker_score"
                    ],
                )
            )

        reranked.sort(
            key=lambda x: x["final_score"],
            reverse=True,
        )

        return reranked[:final_k]