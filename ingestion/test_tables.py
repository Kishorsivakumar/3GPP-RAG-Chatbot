from pathlib import Path

from docx import Document

from ingestion.table_parser import extract_tables


path = Path(
    r"data\raw\TS 23.501\23501-k20\23501-k20.docx"
)

document = Document(path)

tables = extract_tables(document)

print(f"Tables extracted: {len(tables)}")

for table in tables[:3]:
    print("\n" + "=" * 80)
    print(f"Table index: {table['table_index']}")
    print("-" * 80)
    print(table["content"][:1500])