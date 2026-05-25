from __future__ import annotations

from typing import BinaryIO

import pdfplumber
from langchain_core.documents import Document


def load_pdf(file_obj: BinaryIO, source_name: str = "uploaded_pdf") -> list[Document]:
    """Extract text from each page of a PDF and return LangChain documents."""
    documents: list[Document] = []

    with pdfplumber.open(file_obj) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_name,
                        "page": page_number,
                    },
                )
            )

    return documents
