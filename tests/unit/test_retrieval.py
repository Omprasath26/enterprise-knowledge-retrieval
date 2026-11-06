
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import models

from backend.query.retrieval import (DocumentRetriever,RetrievedChunk,RetrievalError)


class FakeEmbeddingModel:
    """Fake BGE-M3 model for deterministic unit tests."""

    def __init__(self, dimension=1024, output=None):
        self.dimension = dimension
        self.output = output
        self.received_texts = None
        self.encode_kwargs = None

    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.received_texts = texts
        self.encode_kwargs = kwargs

        if self.output is not None:
            return self.output

        return np.ones(
            (len(texts), self.dimension),
            dtype=np.float32,
        )


class FakeQdrantClient:
    """Fake Qdrant client for retrieval tests."""

    def __init__(self, points=None, error=None):
        self.points = points or []
        self.error = error
        self.query_arguments = None
        self.closed = False

    def query_points(self, **kwargs):
        self.query_arguments = kwargs

        if self.error:
            raise self.error

        return SimpleNamespace(points=self.points)

    def close(self):
        self.closed = True


@pytest.fixture
def sample_point():
    return SimpleNamespace(
        id="test-point-1",
        score=0.85,
        payload={
            "chunk_index": 2,
            "contextualized_text": (
                "IT Industry HR Policies: "
                "Employees receive annual leave."
            ),
            "source_file": "IT_Industry_HR_Policies.pdf",
            "headings": ["Leave Policy"],
            "page_numbers": [2],
            "access_tags": ["hr"],
            "document_id": "test-hr-policy",
            "document_version": "a" * 64,
        },
    )


@pytest.fixture
def retriever(sample_point):
    client = FakeQdrantClient(points=[sample_point])
    model = FakeEmbeddingModel()

    instance = DocumentRetriever(
        client=client,
        model=model,
    )

    return instance, client, model


def test_retriever_uses_cpu_and_configured_model(retriever):
    instance, _, _ = retriever

    assert instance.device == "cpu"
    assert instance.embedding_dimension == 1024
    assert instance.collection_name == "enterprise_knowledge"


def test_query_embedding_generation(retriever):
    instance, _, model = retriever

    vector = instance._generate_query_embedding(
        "What is the leave policy?"
    )

    assert len(vector) == 1024
    assert np.isfinite(vector).all()
    assert model.received_texts == [
        "What is the leave policy?"
    ]
    assert model.encode_kwargs["normalize_embeddings"] is True


def test_retrieves_relevant_chunks(retriever):
    instance, client, _ = retriever

    results = instance.retrieve(
        "What is the leave policy?",
        ["hr"],
    )

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].chunk_index == 2
    assert results[0].relevance_score == 0.85
    assert client.query_arguments is not None

    # Verify document provenance metadata.
    assert results[0].document_id == "test-hr-policy"
    assert results[0].document_version == "a" * 64
    assert results[0].chunk_id == "test-point-1"


def test_permission_filter_is_applied(retriever):
    instance, client, _ = retriever

    instance.retrieve(
        "What is the leave policy?",
        ["hr", "management"],
    )

    query_filter = client.query_arguments["query_filter"]

    assert isinstance(query_filter, models.Filter)

    condition = query_filter.must[0]

    assert condition.key == "access_tags"
    assert condition.match.any == ["hr", "management"]


def test_empty_permissions_return_no_results(retriever):
    instance, client, model = retriever

    results = instance.retrieve(
        "What is the leave policy?",
        [],
    )

    assert results == []
    assert client.query_arguments is None
    assert model.received_texts is None


def test_relevance_threshold_is_applied(retriever):
    instance, client, _ = retriever

    instance.retrieve(
        "What is the leave policy?",
        ["hr"],
    )

    assert client.query_arguments["score_threshold"] == 0.50


def test_top_k_limit_is_applied(retriever):
    instance, client, _ = retriever

    instance.retrieve(
        "What is the leave policy?",
        ["hr"],
    )

    assert client.query_arguments["limit"] == 5


def test_source_metadata_is_preserved(retriever):
    instance, _, _ = retriever

    results = instance.retrieve(
        "What is the leave policy?",
        ["hr"],
    )

    result = results[0]

    assert result.source_file == "IT_Industry_HR_Policies.pdf"
    assert result.headings == ["Leave Policy"]
    assert result.page_numbers == [2]


def test_document_provenance_metadata_is_preserved(retriever):
    instance, _, _ = retriever

    results = instance.retrieve(
        "What is the leave policy?",
        ["hr"],
    )

    result = results[0]

    assert result.document_id == "test-hr-policy"
    assert result.document_version == "a" * 64
    assert result.chunk_id == "test-point-1"


def test_missing_source_metadata_is_rejected(retriever):
    instance, client, _ = retriever

    client.points = [
        SimpleNamespace(
            id="invalid-point",
            score=0.9,
            payload={
                "contextualized_text": "Some document text.",
                "document_id": "test-doc",
                "document_version": "a" * 64,
            },
        )
    ]

    results = instance.retrieve(
        "Find information",
        ["hr"],
    )

    assert results == []


def test_missing_text_metadata_is_rejected(retriever):
    instance, client, _ = retriever

    client.points = [
        SimpleNamespace(
            id="invalid-point",
            score=0.9,
            payload={
                "source_file": "policy.pdf",
                "document_id": "test-doc",
                "document_version": "a" * 64,
            },
        )
    ]

    results = instance.retrieve(
        "Find information",
        ["hr"],
    )

    assert results == []


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "document_id": None,
            "document_version": "a" * 64,
        },
        {
            "document_id": "test-doc",
            "document_version": None,
        },
        {
            "document_id": "",
            "document_version": "a" * 64,
        },
        {
            "document_id": "test-doc",
            "document_version": "",
        },
        {
            "document_id": "   ",
            "document_version": "a" * 64,
        },
        {
            "document_id": "test-doc",
            "document_version": "invalid",
        },
        {
            "document_id": "test-doc",
            "document_version": "g" * 64,
        },
        {
            "document_id": "test-doc",
            "document_version": "a" * 63,
        },
        {
            "document_id": "test-doc",
            "document_version": "a" * 65,
        },
    ],
)
def test_skips_points_with_invalid_provenance(retriever, metadata):
    instance, client, _ = retriever

    client.points = [
        SimpleNamespace(
            id="invalid-provenance-point",
            score=0.9,
            payload={
                "source_file": "policy.pdf",
                "contextualized_text": "Some document text.",
                "document_id": metadata["document_id"],
                "document_version": metadata["document_version"],
            },
        )
    ]

    results = instance.retrieve(
        "Find information",
        ["hr"],
    )

    assert results == []


def test_skips_points_without_chunk_id(retriever):
    instance, client, _ = retriever

    client.points = [
        SimpleNamespace(
            id=None,
            score=0.9,
            payload={
                "source_file": "policy.pdf",
                "contextualized_text": "Some document text.",
                "document_id": "test-doc",
                "document_version": "a" * 64,
            },
        )
    ]

    results = instance.retrieve(
        "Find information",
        ["hr"],
    )

    assert results == []


def test_preserves_normalized_document_version(retriever):
    instance, client, _ = retriever

    client.points = [
        SimpleNamespace(
            id="test-point-2",
            score=0.9,
            payload={
                "source_file": "policy.pdf",
                "contextualized_text": "Some document text.",
                "document_id": "test-doc",
                "document_version": "A" * 64,
            },
        )
    ]

    results = instance.retrieve(
        "Find information",
        ["hr"],
    )

    assert len(results) == 1
    assert results[0].document_version == "a" * 64


def test_legacy_points_without_version_are_rejected(retriever):
    """Old Qdrant points must not enter evidence-based cache isolation."""

    instance, client, _ = retriever

    client.points = [
        SimpleNamespace(
            id="legacy-point",
            score=0.9,
            payload={
                "source_file": "old-policy.pdf",
                "contextualized_text": "Legacy document text.",
                "document_id": "legacy-doc",
            },
        )
    ]

    results = instance.retrieve(
        "Find information",
        ["hr"],
    )

    assert results == []


@pytest.mark.parametrize(
    "question",
    ["", "   ", None, 123],
)
def test_rejects_invalid_questions(retriever, question):
    instance, _, _ = retriever

    with pytest.raises(ValueError):
        instance.retrieve(question, ["hr"])


@pytest.mark.parametrize(
    "tags",
    [None, "hr", ["", "hr"], [123]],
)
def test_rejects_invalid_access_tags(retriever, tags):
    instance, _, _ = retriever

    with pytest.raises(ValueError):
        instance.retrieve(
            "What is the leave policy?",
            tags,
        )


def test_rejects_invalid_embedding_dimension(retriever):
    instance, _, model = retriever

    model.output = np.ones(
        (1, 768),
        dtype=np.float32,
    )

    with pytest.raises(
        RetrievalError,
        match="unexpected dimension",
    ):
        instance._generate_query_embedding("Test question")


def test_rejects_non_finite_embedding(retriever):
    instance, _, model = retriever

    output = np.ones((1, 1024), dtype=np.float32)
    output[0][0] = np.nan
    model.output = output

    with pytest.raises(
        RetrievalError,
        match="non-finite",
    ):
        instance._generate_query_embedding("Test question")


def test_rejects_invalid_relevance_threshold():
    with pytest.raises(ValueError):
        DocumentRetriever(
            client=FakeQdrantClient(),
            model=FakeEmbeddingModel(),
            relevance_threshold=1.5,
        )


def test_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        DocumentRetriever(
            client=FakeQdrantClient(),
            model=FakeEmbeddingModel(),
            top_k=0,
        )


def test_qdrant_errors_are_wrapped(retriever):
    instance, client, _ = retriever

    client.error = RuntimeError("Qdrant connection failed")

    with pytest.raises(
        RetrievalError,
        match="Qdrant retrieval failed",
    ):
        instance.retrieve(
            "What is the leave policy?",
            ["hr"],
        )


def test_closes_qdrant_client(retriever):
    instance, client, _ = retriever

    instance.close()

    assert client.closed is True


def test_duplicate_access_tags_are_removed(retriever):
    instance, client, _ = retriever

    instance.retrieve(
        "What is the leave policy?",
        ["hr", "hr", "management"],
    )

    condition = client.query_arguments["query_filter"].must[0]

    assert condition.match.any == ["hr", "management"]


def test_query_does_not_return_vectors(retriever):
    instance, client, _ = retriever

    instance.retrieve(
        "What is the leave policy?",
        ["hr"],
    )

    assert client.query_arguments["with_vectors"] is False
    assert client.query_arguments["with_payload"] is True
