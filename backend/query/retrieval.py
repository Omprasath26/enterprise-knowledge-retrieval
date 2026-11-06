
from dataclasses import dataclass
from typing import Any, Sequence
import re

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "enterprise_knowledge"
DEFAULT_DEVICE = "cpu"
DEFAULT_RELEVANCE_THRESHOLD = 0.50
DEFAULT_TOP_K = 5


class RetrievalError(Exception):
    """Raised when document retrieval fails."""


@dataclass
class RetrievedChunk:
    chunk_index: int
    text: str
    source_file: str
    headings: list[str]
    page_numbers: list[int]
    relevance_score: float
    document_id: str
    document_version: str
    chunk_id: str


class DocumentRetriever:
    """Retrieve relevant document chunks with access-tag filtering."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        device: str = DEFAULT_DEVICE,
        client: Any = None,
        model: Any = None):
        if not -1.0 <= relevance_threshold <= 1.0:
            raise ValueError("Relevance threshold must be between -1 and 1.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        self.collection_name = collection_name
        self.relevance_threshold = relevance_threshold
        self.top_k = top_k
        self.device = device

        try:
            self.client = client or QdrantClient(url=qdrant_url)
            self.model = model or SentenceTransformer(
                model_name,
                device=device,
            )
            self.embedding_dimension = self.model.get_embedding_dimension()

        except Exception as exc:
            raise RetrievalError(
                f"Failed to initialize retrieval service: {exc}"
            ) from exc

    @staticmethod
    def _validate_access_tags(allowed_access_tags: Sequence[str]) -> list[str]:
        if isinstance(allowed_access_tags, (str, bytes)):
            raise ValueError(
                "Access tags must be a sequence of tag strings."
            )

        if not isinstance(allowed_access_tags, Sequence):
            raise ValueError(
                "Access tags must be a sequence of tag strings."
            )

        tags = list(allowed_access_tags)

        if any(
            not isinstance(tag, str) or not tag.strip()
            for tag in tags
        ):
            raise ValueError("Access tags must be non-empty strings.")

        return list(dict.fromkeys(tags))

    @staticmethod
    def _valid_provenance(
        document_id: Any,
        document_version: Any,
        chunk_id: Any) -> bool:
        """Return False when a point lacks trustworthy version provenance."""
        if not all(
            isinstance(value, str) and value.strip()
            for value in (document_id, document_version, chunk_id)
        ):
            return False

        version = document_version.strip()

        return (
            len(version) == 64
            and re.fullmatch(r"[0-9a-fA-F]{64}", version) is not None
        )

    def _generate_query_embedding(self, question: str) -> list[float]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        try:
            vector = self.model.encode(
                [question.strip()],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            vector = np.asarray(vector)

            if vector.shape != (1, self.embedding_dimension):
                raise RetrievalError(
                    "Query embedding has an unexpected dimension."
                )

            if not np.isfinite(vector).all():
                raise RetrievalError(
                    "Query embedding contains non-finite values."
                )

            return vector[0].astype(float).tolist()

        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Query embedding generation failed: {exc}"
            ) from exc

    def retrieve(
        self,
        question: str,
        allowed_access_tags: Sequence[str]) -> list[RetrievedChunk]:
        """Retrieve authorized chunks with complete document provenance.

        Access tags must come from a trusted authorization layer.
        Do not accept client-supplied tags as proof of authorization.
        """

        tags = self._validate_access_tags(allowed_access_tags)

        # Fail closed: no permissions means no document access.
        if not tags:
            return []

        query_vector = self._generate_query_embedding(question)

        access_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="access_tags",
                    match=models.MatchAny(any=tags),
                )
            ]
        )

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=access_filter,
                limit=self.top_k,
                score_threshold=self.relevance_threshold,
                with_payload=True,
                with_vectors=False,
            )

            results = []

            for point in response.points:
                payload = point.payload or {}

                source_file = payload.get("source_file")
                text = payload.get("contextualized_text")
                document_id = payload.get("document_id")
                document_version = payload.get("document_version")
                chunk_id = point.id

                # Fail closed if source data or provenance is incomplete.
                if not isinstance(source_file, str) or not source_file.strip():
                    continue

                if not isinstance(text, str) or not text.strip():
                    continue

                if not self._valid_provenance(
                    document_id,
                    document_version,
                    chunk_id,
                ):
                    continue

                results.append(
                    RetrievedChunk(
                        chunk_index=payload.get("chunk_index", -1),
                        text=text,
                        source_file=source_file,
                        headings=payload.get("headings") or [],
                        page_numbers=payload.get("page_numbers") or [],
                        relevance_score=float(point.score),
                        document_id=document_id.strip(),
                        document_version=document_version.strip().lower(),
                        chunk_id=str(chunk_id),
                    )
                )

            return results

        except Exception as exc:
            raise RetrievalError(
                f"Qdrant retrieval failed: {exc}"
            ) from exc

    def close(self) -> None:
        """Close the Qdrant client connection."""
        self.client.close()


if __name__ == "__main__":
    print("Permission-aware retrieval module loaded.")
    print("Embedding model:", DEFAULT_EMBEDDING_MODEL)
    print("Vector dimensions: 1024")
    print("Default relevance threshold:", DEFAULT_RELEVANCE_THRESHOLD)
