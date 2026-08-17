from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


SECTION_PATTERN = re.compile(
    r"^(?P<number>\d+(?:\.\d+)*[A-Za-z]?)\s*(?:\t|\s{2,}|\s+)(?P<title>.+)$"
)


@dataclass
class Section:
    number: str
    title: str
    level: int
    paragraphs: List[str] = field(default_factory=list)
    start_paragraph: Optional[int] = None
    end_paragraph: Optional[int] = None


def parse_heading(text: str, style_name: str):
    """
    Parse a real document heading.

    We intentionally ignore TOC styles such as:
        toc 1, toc 2, toc 3, ...

    and only consider Word heading styles.
    """
    if not style_name.lower().startswith("heading"):
        return None

    cleaned = " ".join(text.replace("\xa0", " ").split())

    match = SECTION_PATTERN.match(cleaned)

    if not match:
        return None

    number = match.group("number")
    title = match.group("title").strip()

    level_match = re.search(r"(\d+)$", style_name)

    if level_match:
        level = int(level_match.group(1))
    else:
        level = number.count(".") + 1

    return number, title, level


def build_sections(paragraphs):
    """
    Convert DOCX paragraphs into hierarchical sections.
    """
    sections: List[Section] = []

    current: Optional[Section] = None

    for index, paragraph in enumerate(paragraphs):
        text = paragraph.text.strip()
        style_name = paragraph.style.name

        if not text:
            continue

        heading = parse_heading(text, style_name)

        if heading:
            if current is not None:
                current.end_paragraph = index - 1
                sections.append(current)

            number, title, level = heading

            current = Section(
                number=number,
                title=title,
                level=level,
                start_paragraph=index,
            )

        elif current is not None:
            current.paragraphs.append(text)

    if current is not None:
        current.end_paragraph = len(paragraphs) - 1
        sections.append(current)

    return sections