"""Generate normalized BGE-M3 dense embeddings for document chunks."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.ingestion.document_chunker import DocumentChunk

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 8


class DocumentEmbeddingError(Exception):
    """Raised when document embedding generation fails."""


@dataclass
class EmbeddedChunk:
    """A document chunk combined with its dense embedding and metadata."""

    chunk_index: int
    text: str
    contextualized_text: str
    embedding: list[float]
    embedding_model: str
    embedding_dimension: int
    source_file: str
    file_extension: str
    headings: list[str]
    page_numbers: list[int]
    access_tags: list[str] | None = None
    document_id: str | None = None
    document_version: str | None = None


class DocumentEmbeddingGenerator:
    """Generate normalized dense embeddings using BGE-M3 on CPU."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model=None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("Model name cannot be empty.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = DEFAULT_DEVICE
        try:
            self.model = model or SentenceTransformer(model_name, device=self.device)
        except Exception as exc:
            raise DocumentEmbeddingError(
                f"Could not load embedding model '{model_name}': {exc}"
            ) from exc
        self.embedding_dimension = self.model.get_embedding_dimension()
        if not self.embedding_dimension:
            raise DocumentEmbeddingError(
                "The model did not provide a valid embedding dimension."
            )

    def generate_embeddings(
        self, chunks: Sequence[DocumentChunk]
    ) -> list[EmbeddedChunk]:
        if not chunks:
            return []
        texts: list[str] = []
        for position, chunk in enumerate(chunks):
            if not isinstance(chunk, DocumentChunk):
                raise DocumentEmbeddingError(
                    f"Item at position {position} is not a DocumentChunk."
                )
            if not chunk.document_id or not chunk.document_version:
                raise DocumentEmbeddingError(
                    f"Chunk {chunk.chunk_index} is missing document_id or document_version."
                )
            text = chunk.contextualized_text
            if not isinstance(text, str) or not text.strip():
                raise DocumentEmbeddingError(
                    f"Chunk {chunk.chunk_index} has empty contextualized text."
                )
            texts.append(text)
        try:
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise DocumentEmbeddingError(
                f"BGE-M3 embedding generation failed: {exc}"
            ) from exc
        vectors = np.asarray(vectors)
        if vectors.ndim != 2:
            raise DocumentEmbeddingError(
                f"Expected a 2D embedding array; received shape {vectors.shape}."
            )
        if vectors.shape[0] != len(chunks):
            raise DocumentEmbeddingError(
                "Embedding count does not match input chunk count."
            )
        if vectors.shape[1] != self.embedding_dimension:
            raise DocumentEmbeddingError(
                f"Expected {self.embedding_dimension} dimensions; received {vectors.shape[1]}."
            )
        if not np.isfinite(vectors).all():
            raise DocumentEmbeddingError(
                "Embedding output contains NaN or infinite values."
            )
        return [
            EmbeddedChunk(
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                contextualized_text=chunk.contextualized_text,
                embedding=vector.astype(float).tolist(),
                embedding_model=self.model_name,
                embedding_dimension=self.embedding_dimension,
                source_file=chunk.source_file,
                file_extension=chunk.file_extension,
                headings=list(chunk.headings),
                page_numbers=list(chunk.page_numbers),
                access_tags=list(chunk.access_tags) if chunk.access_tags is not None else None,
                document_id=chunk.document_id,
                document_version=chunk.document_version,
            )
            for chunk, vector in zip(chunks, vectors)
        ]


if __name__ == "__main__":
    print("BGE-M3 embedding generator module loaded.")
    print(f"Configured model: {DEFAULT_EMBEDDING_MODEL}")
    print(f"Configured device: {DEFAULT_DEVICE}")
