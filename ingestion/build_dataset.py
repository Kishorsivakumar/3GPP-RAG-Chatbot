from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from docx import Document

from ingestion.chunker import split_text
from ingestion.metadata import extract_document_metadata
from ingestion.unified_parser import parse_document


# ============================================================
# CONFIGURATION
# ============================================================

DOCX_PATH = Path(
    r"data\raw\TS 23.501\23501-k20\23501-k20.docx"
)

OUTPUT_PATH = Path(
    r"data\processed\chunks.json"
)


# ============================================================
# TEXT CHUNKING
# ============================================================

def build_text_chunks(
    section: Dict,
    metadata: Dict[str, str],
) -> List[Dict]:
    """
    Convert paragraphs from one section into searchable
    semantic text chunks.
    """

    content = "\n".join(
        section["paragraphs"]
    ).strip()

    if not content:
        return []

    chunks = split_text(
        content,
        max_chars=2500,
        overlap=300,
    )

    results: List[Dict] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        section_number = section["section"]

        chunk_id = (
            f"{metadata['specification'].replace(' ', '')}_"
            f"{section_number}_"
            f"{index:03d}"
        )

        results.append(
            {
                "chunk_id": chunk_id,
                "specification": metadata["specification"],
                "version": metadata["version"],
                "release": metadata["release"],
                "section": section_number,
                "section_title": section["section_title"],
                "parent_section": section["parent_section"],
                "parent_title": section["parent_title"],
                "content_type": "text",
                "table_index": None,
                "source": "3GPP",
                "content": chunk,
            }
        )

    return results


# ============================================================
# TABLE CHUNKING
# ============================================================

def build_table_chunks(
    section: Dict,
    metadata: Dict[str, str],
) -> List[Dict]:
    """
    Convert technical tables into searchable table chunks.
    """

    results: List[Dict] = []

    for table in section["tables"]:

        content = table["content"].strip()

        if not content:
            continue

        table_index = table["table_index"]

        chunk_id = (
            f"{metadata['specification'].replace(' ', '')}_"
            f"{section['section']}_"
            f"table_{table_index:03d}"
        )

        results.append(
            {
                "chunk_id": chunk_id,
                "specification": metadata["specification"],
                "version": metadata["version"],
                "release": metadata["release"],
                "section": section["section"],
                "section_title": section["section_title"],
                "parent_section": section["parent_section"],
                "parent_title": section["parent_title"],
                "content_type": "table",
                "table_index": table_index,
                "source": "3GPP",
                "content": content,
            }
        )

    return results


# ============================================================
# SECTION FILTERING
# ============================================================

def should_skip_section(
    section: Dict,
) -> bool:
    """
    Decide whether a section should be excluded from the
    primary RAG knowledge base.

    Currently excludes:
        - Change history
        - Annex X change history
    """

    section_number = (
        section["section"]
        .strip()
        .lower()
    )

    section_title = (
        section["section_title"]
        .strip()
        .lower()
    )

    # Exclude change history
    if "change history" in section_title:
        return True

    # Exclude Annex X
    if section_number.startswith("annex x"):
        return True

    return False


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset() -> List[Dict]:
    """
    Build the complete RAG dataset from the 3GPP DOCX.

    The final dataset contains:
        - Text chunks
        - Technical table chunks
        - Section metadata
        - Parent section metadata
        - Specification version
        - Release information
    """

    # Load DOCX
    document = Document(DOCX_PATH)

    # Extract document metadata
    metadata = extract_document_metadata(
        DOCX_PATH
    )

    print("\nDocument metadata:")
    print(
        f"Specification : {metadata['specification']}"
    )
    print(
        f"Version       : {metadata['version']}"
    )
    print(
        f"Release       : {metadata['release']}"
    )

    # Parse document
    sections = parse_document(
        document
    )

    print(
        f"\nSections parsed: {len(sections)}"
    )

    # Final chunk collection
    all_chunks: List[Dict] = []

    skipped_sections = 0

    for section in sections:

        # Skip administrative/history sections
        if should_skip_section(section):
            skipped_sections += 1
            continue

        # ----------------------------------------------------
        # TEXT CHUNKS
        # ----------------------------------------------------

        text_chunks = build_text_chunks(
            section,
            metadata,
        )

        all_chunks.extend(
            text_chunks
        )

        # ----------------------------------------------------
        # TABLE CHUNKS
        # ----------------------------------------------------

        table_chunks = build_table_chunks(
            section,
            metadata,
        )

        all_chunks.extend(
            table_chunks
        )

    print(
        f"Skipped sections : {skipped_sections}"
    )

    return all_chunks


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    chunks: List[Dict],
) -> None:
    """
    Save the canonical RAG dataset as JSON.
    """

    # Create directory if necessary
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# DATASET STATISTICS
# ============================================================

def print_statistics(
    chunks: List[Dict],
) -> None:
    """
    Print useful dataset statistics.
    """

    text_count = sum(
        1
        for chunk in chunks
        if chunk["content_type"] == "text"
    )

    table_count = sum(
        1
        for chunk in chunks
        if chunk["content_type"] == "table"
    )

    unique_sections = len(
        {
            chunk["section"]
            for chunk in chunks
        }
    )

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(
        f"Total chunks      : {len(chunks)}"
    )

    print(
        f"Text chunks       : {text_count}"
    )

    print(
        f"Table chunks      : {table_count}"
    )

    print(
        f"Unique sections   : {unique_sections}"
    )

    print(
        f"Output file       : {OUTPUT_PATH}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BUILDING 3GPP RAG DATASET")
    print("=" * 60)

    # Build
    chunks = build_dataset()

    # Save
    save_dataset(
        chunks
    )

    # Statistics
    print_statistics(
        chunks
    )

    print(
        "\nDataset built successfully."
    )