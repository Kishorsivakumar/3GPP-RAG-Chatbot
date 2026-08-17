from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.json"
)

# Project root:
# 3gpp-rag-chatbot/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"


CHUNKS_PATH = Path(
    r"data\processed\chunks.json"
)


class BM25Store:
    """
    BM25 lexical retrieval over the canonical 3GPP chunks.
    """

    def __init__(self):
        self.chunks: List[Dict] = []
        self.bm25 = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize technical text while preserving identifiers
        such as AMF, SMF, N11, 5QI, S-NSSAI, etc.
        """

        text = text.lower()

        return re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)*",
            text,
        )

    @staticmethod
    def build_search_text(chunk: Dict) -> str:
        """
        Include metadata because section names themselves are
        highly informative for 3GPP retrieval.
        """

        return (
            f"{chunk['specification']} "
            f"{chunk['release']} "
            f"{chunk['version']} "
            f"Section {chunk['section']} "
            f"{chunk['section_title']} "
            f"{chunk['content']}"
        )

    def build(self, chunks: List[Dict]) -> None:
        """
        Build BM25 index.
        """

        self.chunks = chunks

        corpus = [
            self.tokenize(
                self.build_search_text(chunk)
            )
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(corpus)

        print(
            f"BM25 index built for "
            f"{len(chunks)} chunks."
        )

    def save(self) -> None:
        """
        Save chunk metadata.

        BM25Okapi itself is easier and safer to rebuild from
        chunks.json, so we persist the canonical chunk list.
        """

        CHUNKS_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with CHUNKS_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(self):
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {CHUNKS_PATH}"
            )

    # existing loading logic...

        with CHUNKS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.chunks = json.load(file)

        corpus = [
            self.tokenize(
                self.build_search_text(chunk)
            )
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(corpus)

        print(
            f"BM25 index loaded for "
            f"{len(self.chunks)} chunks."
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict]:
        """
        Return top BM25 results.
        """

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built."
            )

        tokens = self.tokenize(query)

        scores = self.bm25.get_scores(
            tokens
        )

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for index in top_indices:

            result = dict(
                self.chunks[index]
            )

            result["bm25_score"] = float(
                scores[index]
            )

            results.append(result)

        return results