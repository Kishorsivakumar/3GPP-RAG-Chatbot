from __future__ import annotations

from typing import Dict, List

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-encoder reranker for candidate 3GPP chunks.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):
        self.model = CrossEncoder(model_name)

    @staticmethod
    def build_text(chunk: Dict) -> str:
        """
        Build the passage presented to the cross-encoder.
        """

        return (
            f"Specification: {chunk['specification']}\n"
            f"Release: {chunk['release']}\n"
            f"Version: {chunk['version']}\n"
            f"Section: {chunk['section']}\n"
            f"Section title: {chunk['section_title']}\n"
            f"Content type: {chunk['content_type']}\n\n"
            f"{chunk['content']}"
        )

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Rerank candidate chunks against the query.
        """

        if not candidates:
            return []

        pairs = [
            (
                query,
                self.build_text(chunk),
            )
            for chunk in candidates
        ]

        scores = self.model.predict(
            pairs
        )

        ranked = []

        for chunk, score in zip(
            candidates,
            scores,
        ):
            result = dict(chunk)

            result["reranker_score"] = float(
                score
            )

            ranked.append(result)

        ranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        return ranked[:top_k]