"""Utility to extract text from all PDFs in the references directory.

Usage:
    python pdf_text_extractor.py [--src DIR] [--dst DIR]

The default source directory is ``references`` at the repository root.  Each PDF
found will be converted to a ``.txt`` file in the destination directory.  If
the destination directory doesn't exist it will be created.

This script depends on ``pdfplumber`` (pip install pdfplumber) which is better at
preserving order than the standard library.  If you prefer a different backend
feel free to modify the ``extract_text`` function.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None


def extract_text_from_pdf(path: Path) -> str:
    """Return the extracted text for ``path``.

    ``pdfplumber`` is used when available; otherwise ``PyPDF2`` from the
    standard library is used as a fallback.  The fallback may produce poorer
    results so installing ``pdfplumber`` is recommended.
    """
    if pdfplumber:
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

    # fallback using PyPDF2
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise RuntimeError(
            "No PDF backend available; install pdfplumber or PyPDF2"
        )

    reader = PdfReader(str(path))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def main(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.is_dir():
        print(f"Source directory {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for pdf in src_dir.glob("*.pdf"):
        output_file = dst_dir / (pdf.stem + ".txt")
        print(f"Processing {pdf.name} -> {output_file.name}")
        try:
            text = extract_text_from_pdf(pdf)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to read {pdf}: {exc}", file=sys.stderr)
            continue

        output_file.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text from all PDFs in a directory."
    )
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--src",
        type=Path,
        default=repo_root / "references",
        help="Source directory containing PDF files (default: repo/references)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=repo_root / "references" / "text",
        help="Destination directory for text files (default: references/text)",
    )
    args = parser.parse_args()
    main(args.src, args.dst)
