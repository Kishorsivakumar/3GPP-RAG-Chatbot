from __future__ import annotations

from typing import Dict, List

from retrieval.bm25_store import BM25Store
from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import VectorStore


class HybridRetriever:
    """
    Combines dense FAISS retrieval and BM25 retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        bm25_store: BM25Store,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    @staticmethod
    def _key(chunk: Dict) -> str:
        return chunk["chunk_id"]

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 20,
        top_k_bm25: int = 20,
        final_k: int = 10,
        rrf_k: int = 60,
    ) -> List[Dict]:
        """
        Retrieve candidates from both systems and merge them
        using Reciprocal Rank Fusion.
        """

        dense_results = self.vector_store.search(
            query,
            top_k=top_k_dense,
        )

        bm25_results = self.bm25_store.search(
            query,
            top_k=top_k_bm25,
        )

        fused = {}

        # Dense ranking
        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            key = self._key(result)

            if key not in fused:
                fused[key] = {
                    "chunk": result,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                }

            fused[key]["rrf_score"] += (
                1.0 / (rrf_k + rank)
            )

            fused[key]["dense_rank"] = rank

        # BM25 ranking
        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            key = self._key(result)

            if key not in fused:
                fused[key] = {
                    "chunk": result,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                }

            fused[key]["rrf_score"] += (
                1.0 / (rrf_k + rank)
            )

            fused[key]["bm25_rank"] = rank

        ranked = sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        results = []

        for item in ranked[:final_k]:

            result = dict(
                item["chunk"]
            )

            result["rrf_score"] = item[
                "rrf_score"
            ]

            result["dense_rank"] = item[
                "dense_rank"
            ]

            result["bm25_rank"] = item[
                "bm25_rank"
            ]

            results.append(result)

        return results