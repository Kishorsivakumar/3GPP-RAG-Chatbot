from __future__ import annotations

from typing import List, Dict


def split_text(text: str, max_chars: int = 2500, overlap: int = 300):
    """
    Split section text into overlapping chunks.

    We use character limits here as a practical approximation.
    Section boundaries are preserved separately.
    """
    text = text.strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        # Prefer breaking at a paragraph boundary.
        if end < len(text):
            boundary = text.rfind("\n", start, end)

            if boundary > start + 500:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_chunks(
    sections: List[Dict],
    spec_number: str,
    version: str,
    release: str = "Rel-18",
):
    """
    Create metadata-rich chunks from parsed sections.
    """

    chunks = []

    for section in sections:
        section_number = section["section"]
        section_title = section["section_title"]
        content = section["content"]

        section_chunks = split_text(content)

        for index, chunk_text in enumerate(section_chunks, start=1):

            chunk_id = (
                f"{spec_number.replace(' ', '')}_"
                f"{section_number}_"
                f"{index:03d}"
            )

            parent_sections = section_number.split(".")

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "spec_number": spec_number,
                    "version": version,
                    "release": release,
                    "section": section_number,
                    "section_title": section_title,
                    "parent_sections": parent_sections[:-1],
                    "content_type": "text",
                    "content": chunk_text,
                    "source": "3GPP",
                }
            )

    return chunks