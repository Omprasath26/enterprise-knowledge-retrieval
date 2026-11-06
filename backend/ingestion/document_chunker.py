
from dataclasses import dataclass
from typing import Any

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (HuggingFaceTokenizer)
from docling_core.types.doc import DoclingDocument
from backend.ingestion.document_parser import (ParsedDocument)


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_MAX_TOKENS = 512


class DocumentChunkingError(Exception):
    """Raised when document chunking fails."""


@dataclass
class DocumentChunk:
    """Structured chunk prepared for the embedding stage."""

    chunk_index: int
    text: str
    contextualized_text: str
    headings: list[str]
    page_numbers: list[int]
    source_file: str
    file_extension: str
    access_tags: list[str] | None = None


class DocumentChunker:
    """Generate context-aware chunks using Docling HybridChunker."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")

        try:
            self.tokenizer = HuggingFaceTokenizer.from_pretrained(
                model_name,
                max_tokens=max_tokens,
            )

            self.chunker = HybridChunker(
                tokenizer=self.tokenizer,
                merge_peers=True,
                repeat_table_header=True,
            )

        except Exception as exc:
            raise DocumentChunkingError(
                f"Failed to initialize HybridChunker: {exc}"
            ) from exc

    @staticmethod
    def _extract_page_numbers(chunk: Any) -> list[int]:
        """Extract page numbers from chunk provenance."""

        page_numbers: set[int] = set()
        metadata = getattr(chunk, "meta", None)

        doc_items = getattr(metadata, "doc_items", []) or []

        for doc_item in doc_items:
            provenance = getattr(doc_item, "prov", []) or []

            for item in provenance:
                page_no = getattr(item, "page_no", None)

                if isinstance(page_no, int):
                    page_numbers.add(page_no)

        return sorted(page_numbers)

    @staticmethod
    def _extract_headings(chunk: Any) -> list[str]:
        """Extract heading hierarchy from chunk metadata."""

        metadata = getattr(chunk, "meta", None)
        headings = getattr(metadata, "headings", None) or []

        return [
            heading.strip()
            for heading in headings
            if isinstance(heading, str) and heading.strip()
        ]

    def chunk_document(
        self,
        parsed_document: ParsedDocument,
    ) -> list[DocumentChunk]:
        """Convert a parsed document into structured chunks."""

        if not isinstance(parsed_document, ParsedDocument):
            raise DocumentChunkingError(
                "Input must be a ParsedDocument instance."
            )

        if not parsed_document.document_data:
            raise DocumentChunkingError(
                "Parsed document contains no structured document data."
            )

        try:
            docling_document = DoclingDocument.model_validate(
                parsed_document.document_data
            )

            generated_chunks = self.chunker.chunk(docling_document)

            results: list[DocumentChunk] = []

            for index, chunk in enumerate(generated_chunks):
                chunk_text = getattr(chunk, "text", "")

                if not isinstance(chunk_text, str) or not chunk_text.strip():
                    continue

                contextualized_text = self.chunker.contextualize(chunk)

                results.append(
                    DocumentChunk(
                        chunk_index=len(results),
                        text=chunk_text,
                        contextualized_text=contextualized_text,
                        headings=self._extract_headings(chunk),
                        page_numbers=self._extract_page_numbers(chunk),
                        source_file=parsed_document.file_name,
                        file_extension=parsed_document.file_extension,
                    )
                )

            return results

        except Exception as exc:
            raise DocumentChunkingError(
                f"Failed to chunk '{parsed_document.file_name}': {exc}"
            ) from exc


if __name__ == "__main__":
    print("Document chunker module loaded successfully.")
