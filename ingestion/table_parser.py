from __future__ import annotations

from typing import List


def clean_cell_text(text: str) -> str:
    """
    Normalize text extracted from a table cell.
    """
    return " ".join(
        text.replace("\xa0", " ").split()
    ).strip()


def extract_table_text(table) -> str:
    """
    Convert a DOCX table into searchable text.

    Duplicate cells caused by merged DOCX cells are removed
    within each row.
    """

    rows: List[str] = []

    seen_rows = set()

    for row in table.rows:

        cells = []
        seen_cells = set()

        for cell in row.cells:

            text = clean_cell_text(
                cell.text
            )

            if not text:
                continue

            # Avoid duplicated merged-cell content
            if text in seen_cells:
                continue

            seen_cells.add(text)
            cells.append(text)

        if not cells:
            continue

        row_text = " | ".join(cells)

        # Avoid completely duplicated rows
        if row_text in seen_rows:
            continue

        seen_rows.add(row_text)
        rows.append(row_text)

    return "\n".join(rows)


def extract_tables(document):
    """
    Extract all non-empty tables from a DOCX document.
    """

    tables = []

    for index, table in enumerate(
        document.tables
    ):

        text = extract_table_text(table)

        if not text.strip():
            continue

        tables.append(
            {
                "table_index": index,
                "content_type": "table",
                "content": text,
            }
        )

    return tables