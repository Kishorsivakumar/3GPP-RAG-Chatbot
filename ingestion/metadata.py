from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

from docx import Document


RELEASE_PATTERN = re.compile(
    r"\(Release\s+(\d+)\)",
    re.IGNORECASE,
)

VERSION_PATTERN = re.compile(
    r"\bV(\d+\.\d+\.\d+)\b",
    re.IGNORECASE,
)


def extract_document_metadata(docx_path: Path) -> Dict[str, str]:
    """
    Extract metadata from the DOCX core properties and document tables.
    """

    document = Document(docx_path)
    properties = document.core_properties

    title = (properties.title or "").strip()
    subject = (properties.subject or "").strip()

    # Extract Release
    release_match = RELEASE_PATTERN.search(subject)

    release = (
        f"Rel-{release_match.group(1)}"
        if release_match
        else "Unknown"
    )

    # Extract specification number
    spec_match = re.search(
        r"\b3GPP\s+(TS|TR)\s+\d+\.\d+",
        title,
        re.IGNORECASE,
    )

    specification = (
        spec_match.group(0).strip()
        if spec_match
        else title
    )

    # Extract exact version from document tables
    version = "Unknown"

    for table in document.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.replace("\xa0", " ").strip()
                for cell in row.cells
            )

            version_match = VERSION_PATTERN.search(row_text)

            if version_match:
                version = version_match.group(1)
                break

        if version != "Unknown":
            break

    return {
        "specification": specification,
        "title": title,
        "subject": subject,
        "version": version,
        "release": release,
        "author": (properties.author or "").strip(),
        "last_modified_by": (
            properties.last_modified_by or ""
        ).strip(),
    }


if __name__ == "__main__":
    path = Path(
        r"data\raw\TS 23.501\23501-k20\23501-k20.docx"
    )

    metadata = extract_document_metadata(path)

    for key, value in metadata.items():
        print(f"{key}: {value}")