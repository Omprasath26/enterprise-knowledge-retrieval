


import numpy as np
import pytest

from backend.ingestion.document_chunker import DocumentChunk
from backend.ingestion.embedding_generator import (DocumentEmbeddingError,DocumentEmbeddingGenerator)



class FakeEmbeddingModel:
    """Deterministic fake model for unit testing."""

    def __init__(self, dimension=4, output=None):
        self.dimension = dimension
        self.output = output
        self.received_texts = None
        self.encode_kwargs = None

    # Match the current SentenceTransformers API.
    def get_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.received_texts = texts
        self.encode_kwargs = kwargs

        if self.output is not None:
            return self.output

        return np.array(
            [[0.1, 0.2, 0.3, 0.4] for _ in texts],
            dtype=np.float32,
        )


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_index=0,
            text="Employees receive annual leave.",
            contextualized_text=(
                "IT Industry HR Policies > Leave Policy\n"
                "Employees receive annual leave."
            ),
            headings=["IT Industry HR Policies", "Leave Policy"],
            page_numbers=[2],
            source_file="HR_Policy.pdf",
            file_extension=".pdf",
            access_tags=["HR"],
        ),
        DocumentChunk(
            chunk_index=1,
            text="Employees must follow security policies.",
            contextualized_text=(
                "IT Industry HR Policies > Information Security\n"
                "Employees must follow security policies."
            ),
            headings=["IT Industry HR Policies", "Information Security"],
            page_numbers=[3],
            source_file="HR_Policy.pdf",
            file_extension=".pdf",
            access_tags=["HR", "IT"],
        ),
    ]


def test_generator_uses_cpu_and_configured_model():
    model = FakeEmbeddingModel()

    generator = DocumentEmbeddingGenerator(
        model_name="BAAI/bge-m3",
        model=model,
    )

    assert generator.device == "cpu"
    assert generator.model_name == "BAAI/bge-m3"


def test_empty_input_returns_empty_list():
    generator = DocumentEmbeddingGenerator(model=FakeEmbeddingModel())

    assert generator.generate_embeddings([]) == []


def test_generates_one_embedding_per_chunk(sample_chunks):
    generator = DocumentEmbeddingGenerator(model=FakeEmbeddingModel())

    results = generator.generate_embeddings(sample_chunks)

    assert len(results) == len(sample_chunks)


def test_embedding_dimension_is_preserved(sample_chunks):
    generator = DocumentEmbeddingGenerator(model=FakeEmbeddingModel())

    results = generator.generate_embeddings(sample_chunks)

    for result in results:
        assert result.embedding_dimension == 4
        assert len(result.embedding) == 4


def test_embedding_values_are_finite(sample_chunks):
    generator = DocumentEmbeddingGenerator(model=FakeEmbeddingModel())

    results = generator.generate_embeddings(sample_chunks)

    for result in results:
        assert all(np.isfinite(value) for value in result.embedding)


def test_chunk_metadata_is_preserved(sample_chunks):
    generator = DocumentEmbeddingGenerator(model=FakeEmbeddingModel())

    results = generator.generate_embeddings(sample_chunks)

    first = results[0]

    assert first.chunk_index == 0
    assert first.source_file == "HR_Policy.pdf"
    assert first.file_extension == ".pdf"
    assert first.headings == ["IT Industry HR Policies", "Leave Policy"]
    assert first.page_numbers == [2]
    assert first.access_tags == ["HR"]
    assert first.text == sample_chunks[0].text
    assert first.contextualized_text == sample_chunks[0].contextualized_text


def test_model_receives_contextualized_text(sample_chunks):
    model = FakeEmbeddingModel()
    generator = DocumentEmbeddingGenerator(model=model)

    generator.generate_embeddings(sample_chunks)

    assert model.received_texts == [
        chunk.contextualized_text for chunk in sample_chunks
    ]
    assert model.encode_kwargs["normalize_embeddings"] is True
    assert model.encode_kwargs["convert_to_numpy"] is True


def test_rejects_invalid_input():
    generator = DocumentEmbeddingGenerator(model=FakeEmbeddingModel())

    with pytest.raises(DocumentEmbeddingError):
        generator.generate_embeddings(["not a DocumentChunk"])


def test_rejects_empty_contextualized_text(sample_chunks):
    sample_chunks[0].contextualized_text = "   "
    generator = DocumentEmbeddingGenerator(model=FakeEmbeddingModel())

    with pytest.raises(DocumentEmbeddingError):
        generator.generate_embeddings(sample_chunks)


def test_rejects_wrong_embedding_dimension(sample_chunks):
    model = FakeEmbeddingModel(
        dimension=4,
        output=np.array([[0.1, 0.2, 0.3] for _ in sample_chunks]),
    )
    generator = DocumentEmbeddingGenerator(model=model)

    with pytest.raises(DocumentEmbeddingError):
        generator.generate_embeddings(sample_chunks)


def test_rejects_non_finite_embedding_values(sample_chunks):
    model = FakeEmbeddingModel(
        output=np.array(
            [[0.1, np.nan, 0.3, 0.4] for _ in sample_chunks]
        ),
    )
    generator = DocumentEmbeddingGenerator(model=model)

    with pytest.raises(DocumentEmbeddingError):
        generator.generate_embeddings(sample_chunks)


def test_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        DocumentEmbeddingGenerator(
            batch_size=0,
            model=FakeEmbeddingModel(),
        )
