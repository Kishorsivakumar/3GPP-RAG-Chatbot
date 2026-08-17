from pathlib import Path

from docx import Document

from ingestion.unified_parser import parse_document


path = Path(
    r"data\raw\TS 23.501\23501-k20\23501-k20.docx"
)

document = Document(path)

sections = parse_document(document)

print(f"Sections detected: {len(sections)}")

total_tables = sum(
    len(section["tables"])
    for section in sections
)

print(f"Technical tables detected: {total_tables}")


for section in sections:

    if section["section"] in {
        "4.2.2",
        "4.2.3",
    }:

        print("\n" + "=" * 80)

        print(
            f"Section: "
            f"{section['section']} "
            f"{section['section_title']}"
        )

        print(
            f"Paragraphs: "
            f"{len(section['paragraphs'])}"
        )

        print(
            f"Tables: "
            f"{len(section['tables'])}"
        )

        for table in section["tables"]:
            print("\n--- TABLE ---")
            print(
                f"Table index: "
                f"{table['table_index']}"
            )
            print(
                table["content"][:1000]
            )