from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):
        self.model = SentenceTransformer(
            model_name
        )

    def encode(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Generate normalized embeddings.
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )