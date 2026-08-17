from __future__ import annotations

from typing import List, Dict

from docx.document import Document as DocumentType
from docx.table import Table
from docx.text.paragraph import Paragraph

from ingestion.document_iterator import iter_block_items
from ingestion.section_parser import parse_heading
from ingestion.table_parser import extract_table_text


def _has_content(section: Dict) -> bool:
    """
    Return True when a section contains useful text or tables.
    """

    return bool(
        section["paragraphs"]
        or section["tables"]
    )


def parse_document(document: DocumentType) -> List[Dict]:
    """
    Parse a 3GPP DOCX in actual document order.

    The parser preserves:
        - sections
        - paragraphs
        - tables
        - section hierarchy
    """

    sections: List[Dict] = []

    current_section = None
    table_counter = 0

    # Tracks the active hierarchy.
    # Example:
    # {
    #   1: {"number": "4", "title": "..."},
    #   2: {"number": "4.2", "title": "..."},
    #   3: {"number": "4.2.2", "title": "..."}
    # }
    section_stack = {}

    def save_current_section():
        nonlocal current_section

        if current_section is None:
            return

        if _has_content(current_section):
            sections.append(current_section)

    for block_type, block in iter_block_items(document):

        # =========================================================
        # PARAGRAPH
        # =========================================================
        if block_type == "paragraph":

            paragraph: Paragraph = block
            text = paragraph.text.strip()

            if not text:
                continue

            style_name = paragraph.style.name

            # Ignore Table of Contents.
            if style_name.lower().startswith("toc"):
                continue

            heading = parse_heading(
                text,
                style_name,
            )

            # -----------------------------------------------------
            # New real section
            # -----------------------------------------------------
            if heading:

                save_current_section()

                number, title, level = heading

                # Remove deeper hierarchy levels.
                section_stack = {
                    k: v
                    for k, v in section_stack.items()
                    if k < level
                }

                # Determine parent.
                parent = None

                if level > 1 and (level - 1) in section_stack:
                    parent = section_stack[level - 1]

                current_section = {
                    "section": number,
                    "section_title": title,
                    "level": level,
                    "parent_section": (
                        parent["number"]
                        if parent
                        else None
                    ),
                    "parent_title": (
                        parent["title"]
                        if parent
                        else None
                    ),
                    "paragraphs": [],
                    "tables": [],
                }

                section_stack[level] = {
                    "number": number,
                    "title": title,
                }

                continue

            # -----------------------------------------------------
            # Normal paragraph
            # -----------------------------------------------------
            if current_section is not None:
                current_section["paragraphs"].append(text)

        # =========================================================
        # TABLE
        # =========================================================
        elif block_type == "table":

            table: Table = block

            # Ignore front-matter tables appearing before
            # the first actual specification section.
            if current_section is None:
                continue

            table_text = extract_table_text(table)

            if not table_text.strip():
                continue

            current_section["tables"].append(
                {
                    "table_index": table_counter,
                    "content_type": "table",
                    "content": table_text,
                }
            )

            table_counter += 1

    # Save final section.
    save_current_section()

    return sections