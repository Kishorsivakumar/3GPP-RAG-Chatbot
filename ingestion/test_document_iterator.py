from pathlib import Path

from docx import Document

from ingestion.document_iterator import iter_block_items


path = Path(
    r"data\raw\TS 23.501\23501-k20\23501-k20.docx"
)

document = Document(path)

for index, (block_type, block) in enumerate(
    iter_block_items(document)
):

    if block_type == "paragraph":
        text = block.text.strip()

        if text:
            print(
                f"[{index}] "
                f"PARAGRAPH | "
                f"style={block.style.name!r} | "
                f"{text[:150]!r}"
            )

    else:
        print(
            f"[{index}] "
            f"TABLE | rows={len(block.rows)} "
            f"cols={len(block.columns)}"
        )

    if index >= 100:
        break