from pathlib import Path
import zipfile
import fitz


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def extract_zip_files():
    """
    Extract all ZIP files from data/raw into their respective folders.
    """
    for spec_dir in RAW_DIR.iterdir():
        if not spec_dir.is_dir():
            continue

        for zip_file in spec_dir.glob("*.zip"):
            extract_dir = spec_dir / zip_file.stem

            if extract_dir.exists():
                print(f"[SKIP] Already extracted: {zip_file.name}")
                continue

            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall(extract_dir)

            print(f"[OK] Extracted: {zip_file}")


def find_pdfs():
    """
    Locate all PDFs under data/raw.
    """
    pdf_files = list(RAW_DIR.rglob("*.pdf"))

    print(f"\nFound {len(pdf_files)} PDF file(s):")

    for pdf in pdf_files:
        print(f"  - {pdf}")

    return pdf_files


def extract_pdf_text(pdf_path: Path):
    """
    Extract text page-by-page from a PDF.
    """
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    document.close()

    return pages


if __name__ == "__main__":
    extract_zip_files()

    pdf_files = find_pdfs()

    for pdf in pdf_files:
        pages = extract_pdf_text(pdf)

        print(
            f"\n{pdf.name}: "
            f"{len(pages)} pages with extracted text"
        )

        if pages:
            print("\nFirst page preview:\n")
            print(pages[0]["text"][:1000])