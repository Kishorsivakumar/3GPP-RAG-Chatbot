from pathlib import Path

from docx import Document

from ingestion.document_iterator import iter_block_items


path = Path(
    r"data\raw\TS 23.501\23501-k20\23501-k20.docx"
)

document = Document(path)

current_heading = "NONE"
table_index = 0

for block_type, block in iter_block_items(document):

    if block_type == "paragraph":

        text = block.text.strip()

        if block.style.name.startswith("Heading") and text:
            current_heading = text

    elif block_type == "table":

        print(
            f"TABLE {table_index}: "
            f"current heading = {current_heading!r}"
        )

        if table_index in {2, 3, 4, 5}:
            print(
                "First row:",
                [
                    cell.text.strip()
                    for cell in block.rows[0].cells
                ],
            )

        table_index += 1

print(f"\nTotal tables: {table_index}")