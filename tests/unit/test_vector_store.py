
"""Unit tests for Qdrant vector storage."""

import numpy as np
import pytest

from backend.ingestion.embedding_generator import EmbeddedChunk
from backend.common.vector_store import (DEFAULT_COLLECTION_NAME,DEFAULT_VECTOR_SIZE,QdrantVectorStore,VectorStoreError)


class FakeQdrantClient:
    def __init__(self, collection_exists=False):
        self.exists = collection_exists
        self.created_collection = None
        self.upserted_points = []
        self.retrieved_ids = None
        self.closed = False

    def collection_exists(self, collection_name):
        return self.exists

    def create_collection(self, **kwargs):
        self.created_collection = kwargs
        self.exists = True
        return True

    def upsert(self, **kwargs):
        self.upserted_points = kwargs["points"]
        return True

    def retrieve(self, **kwargs):
        self.retrieved_ids = kwargs["ids"]
        return []

    def close(self):
        self.closed = True


@pytest.fixture
def sample_embedded_chunks():
    return [
        EmbeddedChunk(
            chunk_index=0,
            text="Employees receive annual leave.",
            contextualized_text=(
                "HR Policies > Leave Policy\n"
                "Employees receive annual leave."
            ),
            embedding=[0.1] * DEFAULT_VECTOR_SIZE,
            embedding_model="BAAI/bge-m3",
            embedding_dimension=DEFAULT_VECTOR_SIZE,
            source_file="HR_Policy.pdf",
            file_extension=".pdf",
            headings=["HR Policies", "Leave Policy"],
            page_numbers=[2],
            access_tags=["HR"],
            document_id="test-hr-policy",
            document_version="a" * 64,
        ),
        EmbeddedChunk(
            chunk_index=1,
            text="Employees must follow security policies.",
            contextualized_text=(
                "HR Policies > Information Security\n"
                "Employees must follow security policies."
            ),
            embedding=[0.2] * DEFAULT_VECTOR_SIZE,
            embedding_model="BAAI/bge-m3",
            embedding_dimension=DEFAULT_VECTOR_SIZE,
            source_file="HR_Policy.pdf",
            file_extension=".pdf",
            headings=["HR Policies", "Information Security"],
            page_numbers=[3],
            access_tags=["HR", "IT"],
            document_id="test-hr-policy",
            document_version="a" * 64,
        ),
    ]


def test_default_configuration():
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)

    assert store.collection_name == DEFAULT_COLLECTION_NAME
    assert store.vector_size == DEFAULT_VECTOR_SIZE


def test_collection_is_created_with_cosine_distance():
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)

    store.create_collection()

    config = client.created_collection["vectors_config"]

    assert config.size == 1024
    assert config.distance.value == "Cosine"


def test_existing_collection_is_not_recreated():
    client = FakeQdrantClient(collection_exists=True)
    store = QdrantVectorStore(client=client)

    store.create_collection()

    assert client.created_collection is None


def test_empty_input_does_not_insert_points():
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)

    assert store.upsert_chunks([]) == 0
    assert client.upserted_points == []


def test_batch_insert_preserves_payload(sample_embedded_chunks):
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)

    inserted = store.upsert_chunks(sample_embedded_chunks)

    assert inserted == 2
    assert len(client.upserted_points) == 2

    payload = client.upserted_points[0].payload

    assert payload["source_file"] == "HR_Policy.pdf"
    assert payload["page_numbers"] == [2]
    assert payload["headings"] == ["HR Policies", "Leave Policy"]
    assert payload["access_tags"] == ["HR"]
    assert payload["embedding_model"] == "BAAI/bge-m3"
    assert payload["document_id"] == "test-hr-policy"
    assert payload["document_version"] == "a" * 64


def test_stored_vectors_have_expected_dimensions(sample_embedded_chunks):
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)

    store.upsert_chunks(sample_embedded_chunks)

    for point in client.upserted_points:
        assert len(point.vector) == DEFAULT_VECTOR_SIZE


def test_point_ids_are_stable(sample_embedded_chunks):
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)

    first_id = store._create_point_id(sample_embedded_chunks[0])
    second_id = store._create_point_id(sample_embedded_chunks[0])

    assert first_id == second_id


def test_duplicate_chunk_identity_uses_same_point_id(sample_embedded_chunks):
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)

    store.upsert_chunks(sample_embedded_chunks)
    first_ids = [point.id for point in client.upserted_points]

    store.upsert_chunks(sample_embedded_chunks)
    second_ids = [point.id for point in client.upserted_points]

    assert first_ids == second_ids


def test_different_document_versions_produce_different_point_ids(
    sample_embedded_chunks,
):
    first_chunk = sample_embedded_chunks[0]

    first_id = QdrantVectorStore._create_point_id(first_chunk)

    first_chunk.document_version = "b" * 64

    second_id = QdrantVectorStore._create_point_id(first_chunk)

    assert first_id != second_id


def test_different_document_ids_produce_different_point_ids(
    sample_embedded_chunks,
):
    first_chunk = sample_embedded_chunks[0]

    first_id = QdrantVectorStore._create_point_id(first_chunk)

    first_chunk.document_id = "another-hr-policy"

    second_id = QdrantVectorStore._create_point_id(first_chunk)

    assert first_id != second_id


def test_rejects_wrong_vector_dimension(sample_embedded_chunks):
    sample_embedded_chunks[0].embedding = [0.1, 0.2]

    store = QdrantVectorStore(client=FakeQdrantClient())

    with pytest.raises(VectorStoreError):
        store.upsert_chunks(sample_embedded_chunks)


def test_rejects_non_finite_vector(sample_embedded_chunks):
    sample_embedded_chunks[0].embedding[0] = float("nan")

    store = QdrantVectorStore(client=FakeQdrantClient())

    with pytest.raises(VectorStoreError):
        store.upsert_chunks(sample_embedded_chunks)


def test_rejects_missing_source_filename(sample_embedded_chunks):
    sample_embedded_chunks[0].source_file = ""

    store = QdrantVectorStore(client=FakeQdrantClient())

    with pytest.raises(VectorStoreError):
        store.upsert_chunks(sample_embedded_chunks)


def test_retrieves_points_by_id():
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)

    result = store.retrieve_points(["point-1", "point-2"])

    assert result == []
    assert client.retrieved_ids == ["point-1", "point-2"]


def test_empty_retrieval_returns_empty_list():
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)

    assert store.retrieve_points([]) == []
    assert client.retrieved_ids is None


def test_closes_client():
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)

    store.close()

    assert client.closed is True


def test_rejects_invalid_vector_size():
    with pytest.raises(ValueError):
        QdrantVectorStore(
            vector_size=0,
            client=FakeQdrantClient(),
        )
