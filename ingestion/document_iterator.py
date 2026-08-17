from __future__ import annotations

from typing import Iterator, Tuple

from docx.document import Document as DocumentType
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_block_items(
    document: DocumentType,
) -> Iterator[Tuple[str, object]]:
    """
    Yield paragraphs and tables in their actual DOCX order.

    Returns:
        ("paragraph", Paragraph)
        ("table", Table)
    """

    body = document.element.body

    for child in body.iterchildren():

        if child.tag.endswith("}p"):
            yield "paragraph", Paragraph(child, document)

        elif child.tag.endswith("}tbl"):
            yield "table", Table(child, document)