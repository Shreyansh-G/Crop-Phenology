## PDF Text Extraction 

A helper script is provided to convert all PDFs in the `references` directory
into plain text files suitable for ingestion by LLMs or other text processing
pipelines.

```bash
# install dependencies
pip install pdfplumber    # optional; PyPDF2 will be used if pdfplumber is absent

# run using defaults (reads from references/, writes to references/text/)
python scripts/helpers/pdf_text_extractor.py

# supply custom source and output directories
python scripts/helpers/pdf_text_extractor.py --src references --dst references/text
```

Each PDF will produce a `.txt` with the same basename in the destination
folder.
