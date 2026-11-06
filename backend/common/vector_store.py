"""Qdrant vector storage integration for versioned embedded chunks."""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from backend.ingestion.embedding_generator import EmbeddedChunk

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "enterprise_knowledge"
DEFAULT_VECTOR_SIZE = 1024


class VectorStoreError(Exception):
    """Raised when Qdrant storage operations fail."""


class QdrantVectorStore:
    """Store and retrieve BGE-M3 dense vectors in Qdrant."""

    def __init__(
        self,
        url: str = DEFAULT_QDRANT_URL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        client: QdrantClient | None = None) -> None:
        if not url or not url.strip():
            raise ValueError("Qdrant URL cannot be empty.")
        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")
        if vector_size <= 0:
            raise ValueError("Vector size must be greater than zero.")
        self.collection_name = collection_name
        self.vector_size = vector_size
        try:
            self.client = client or QdrantClient(url=url)
        except Exception as exc:
            raise VectorStoreError(f"Could not initialize Qdrant client: {exc}") from exc

    def create_collection(self) -> None:
        """Create the collection if it does not already exist."""
        try:
            if self.client.collection_exists(self.collection_name):
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=models.Distance.COSINE
                ),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not create or verify Qdrant collection '{self.collection_name}': {exc}"
            ) from exc

    @staticmethod
    def _create_point_id(chunk: EmbeddedChunk) -> str:
        """Create a deterministic ID using logical document identity and version."""
        if not chunk.document_id or not chunk.document_version:
            raise VectorStoreError(
                "Cannot create a Qdrant point ID without document_id and document_version."
            )
        identity = (
            f"{chunk.document_id}|{chunk.document_version}|{chunk.chunk_index}"
        )
        return str(uuid5(NAMESPACE_URL, identity))

    def _validate_chunk(self, chunk: EmbeddedChunk) -> None:
        if not isinstance(chunk, EmbeddedChunk):
            raise VectorStoreError("Every item must be an EmbeddedChunk.")
        if len(chunk.embedding) != self.vector_size:
            raise VectorStoreError(
                f"Chunk {chunk.chunk_index} has {len(chunk.embedding)} dimensions; expected {self.vector_size}."
            )
        if not np.isfinite(chunk.embedding).all():
            raise VectorStoreError(
                f"Chunk {chunk.chunk_index} contains invalid vector values."
            )
        if not chunk.source_file:
            raise VectorStoreError(f"Chunk {chunk.chunk_index} has no source filename.")
        if not chunk.document_id or not chunk.document_id.strip():
            raise VectorStoreError(f"Chunk {chunk.chunk_index} has no document_id.")
        version = chunk.document_version
        if not isinstance(version, str) or len(version) != 64:
            raise VectorStoreError(
                f"Chunk {chunk.chunk_index} has an invalid SHA-256 document_version."
            )
        try:
            int(version, 16)
        except ValueError as exc:
            raise VectorStoreError(
                f"Chunk {chunk.chunk_index} has a non-hexadecimal document_version."
            ) from exc

    def upsert_chunks(self, chunks: Sequence[EmbeddedChunk]) -> int:
        """Insert or update versioned embedded chunks in Qdrant."""
        if not chunks:
            return 0
        self.create_collection()
        points = []
        for chunk in chunks:
            self._validate_chunk(chunk)
            payload = {
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "contextualized_text": chunk.contextualized_text,
                "embedding_model": chunk.embedding_model,
                "embedding_dimension": chunk.embedding_dimension,
                "source_file": chunk.source_file,
                "file_extension": chunk.file_extension,
                "headings": chunk.headings,
                "page_numbers": chunk.page_numbers,
                "access_tags": chunk.access_tags or [],
                "document_id": chunk.document_id,
                "document_version": chunk.document_version,
            }
            points.append(
                models.PointStruct(
                    id=self._create_point_id(chunk),
                    vector=chunk.embedding,
                    payload=payload,
                )
            )
        try:
            self.client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Could not insert embedded chunks into Qdrant: {exc}"
            ) from exc
        return len(points)

    def retrieve_points(self, point_ids: Sequence[str], with_vectors: bool = True):
        """Retrieve stored points by their Qdrant IDs."""
        if not point_ids:
            return []
        try:
            return self.client.retrieve(
                collection_name=self.collection_name,
                ids=list(point_ids),
                with_payload=True,
                with_vectors=with_vectors,
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not retrieve points from Qdrant: {exc}") from exc

    def close(self) -> None:
        """Close the Qdrant client connection."""
        self.client.close()


if __name__ == "__main__":
    print("Qdrant vector store module loaded.")
    print(f"Default collection: {DEFAULT_COLLECTION_NAME}")
    print(f"Expected vector dimensions: {DEFAULT_VECTOR_SIZE}")
