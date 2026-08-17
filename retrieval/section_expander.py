from __future__ import annotations

from collections import defaultdict
from typing import Dict, List


class SectionExpander:
    """
    Expands a retrieved chunk with nearby chunks from the
    same 3GPP section.

    This helps when one answer is split across multiple
    chunks of the same clause.
    """

    def __init__(
        self,
        chunks: List[Dict],
    ):
        self.chunks = chunks

        self.by_section = defaultdict(list)

        for chunk in chunks:
            key = chunk["section"]

            self.by_section[key].append(chunk)

    def expand(
        self,
        candidates: List[Dict],
        max_per_section: int = 3,
    ) -> List[Dict]:
        """
        Add neighboring chunks belonging to the same section.
        """

        results = []
        seen = set()

        for candidate in candidates:

            section = candidate["section"]

            section_chunks = self.by_section.get(
                section,
                [],
            )

            # Put the original candidate first.
            ordered = [
                candidate
            ] + [
                chunk
                for chunk in section_chunks
                if chunk["chunk_id"]
                != candidate["chunk_id"]
            ]

            added = 0

            for chunk in ordered:

                chunk_id = chunk["chunk_id"]

                if chunk_id in seen:
                    continue

                results.append(
                    dict(chunk)
                )

                seen.add(chunk_id)
                added += 1

                if added >= max_per_section:
                    break

        return results