
from pathlib import Path
import pytest
from backend.ingestion.document_parser import (DocumentParser,ParsedDocument)
from backend.ingestion.document_chunker import (DocumentChunker,DocumentChunkingError)


@pytest.fixture(scope="module")
def parser():
    return DocumentParser()


@pytest.fixture(scope="module")
def chunker():
    return DocumentChunker(
        model_name="BAAI/bge-m3",
        max_tokens=512,
    )


def test_hr_policy_document_chunking(parser, chunker):
    """Verify chunk generation and source metadata using a real PDF."""

    sample_file = (
        Path(__file__).resolve().parents[1]
        / "IT_Industry_HR_Policies.pdf"
    )

    assert sample_file.exists(), "Sample PDF not found."

    parsed_document = parser.parse(sample_file)
    chunks = chunker.chunk_document(parsed_document)

    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(chunk.contextualized_text.strip() for chunk in chunks)

    assert all(
        chunk.source_file == "IT_Industry_HR_Policies.pdf"
        for chunk in chunks
    )

    assert all(
        chunk.file_extension == ".pdf"
        for chunk in chunks
    )

    assert [
        chunk.chunk_index for chunk in chunks
    ] == list(range(len(chunks)))

    print("\n--- HYBRID CHUNKING RESULTS ---")
    print("Source:", parsed_document.file_name)
    print("Total chunks:", len(chunks))

    for chunk in chunks:
        print("\nChunk index:", chunk.chunk_index)
        print("Headings:", chunk.headings)
        print("Pages:", chunk.page_numbers)
        print("Text:", chunk.text[:300])
        print("Contextualized:", chunk.contextualized_text[:300])


def test_empty_document_data_error(chunker):
    """Verify that empty parsed document data is rejected."""

    empty_document = ParsedDocument(
        file_name="empty.pdf",
        file_extension=".pdf",
        markdown_content="",
        document_data={},
        headings=[],
        page_numbers=[],
        table_count=0,
    )

    with pytest.raises(
        DocumentChunkingError,
        match="no structured document data"):
        chunker.chunk_document(empty_document)


def test_invalid_input_error(chunker):
    """Verify that invalid input types are rejected."""

    with pytest.raises(
        DocumentChunkingError,
        match="ParsedDocument instance"):
        chunker.chunk_document("invalid input")
