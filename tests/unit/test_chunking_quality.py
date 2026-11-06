
import re
from pathlib import Path

import pytest
from docx import Document
from openpyxl import Workbook

from backend.ingestion.document_parser import DocumentParser
from backend.ingestion.document_chunker import DocumentChunker


MAX_TOKENS = 512

SAMPLE_PDF = (
    Path(__file__).resolve().parents[1]
    / "IT_Industry_HR_Policies.pdf"
)


@pytest.fixture(scope="module")
def parser():
    return DocumentParser()


@pytest.fixture(scope="module")
def chunker():
    return DocumentChunker(
        model_name="BAAI/bge-m3",
        max_tokens=MAX_TOKENS,
    )


@pytest.fixture(scope="module")
def parsed_pdf(parser):
    assert SAMPLE_PDF.exists(), (
        f"Sample PDF not found: {SAMPLE_PDF}"
    )
    return parser.parse(SAMPLE_PDF)


@pytest.fixture(scope="module")
def pdf_chunks(chunker, parsed_pdf):
    return chunker.chunk_document(parsed_pdf)


def normalize_text(text):
    """Normalize whitespace for content comparisons."""
    return re.sub(r"\s+", " ", text).strip().lower()



# 1. TOKEN LIMIT VALIDATION


def test_chunk_text_respects_512_token_limit(chunker, pdf_chunks):
    """Verify raw chunk text stays within 512 tokens."""

    for chunk in pdf_chunks:
        token_count = chunker.tokenizer.count_tokens(
            text=chunk.text
        )

        assert token_count <= MAX_TOKENS, (
            f"Chunk {chunk.chunk_index} contains "
            f"{token_count} tokens."
        )


def test_contextualized_text_respects_512_token_limit(chunker, pdf_chunks):
    """Verify contextualized text stays within 512 tokens."""

    for chunk in pdf_chunks:
        token_count = chunker.tokenizer.count_tokens(
            text=chunk.contextualized_text
        )

        assert token_count <= MAX_TOKENS, (
            f"Contextualized chunk {chunk.chunk_index} "
            f"contains {token_count} tokens."
        )



# 2. TABLE CONTENT VALIDATION


def test_hr_policy_table_content_is_preserved(pdf_chunks):
    """Verify important table headers and values survive chunking."""

    combined_text = normalize_text(
        " ".join(chunk.text for chunk in pdf_chunks)
    )

    required_content = [
        "salary structure",
        "basic",
        "hra",
        "performance bonus",
        "sick leave",
        "maternity leave",
        "26 weeks",
        "notice period",
        "60-90 days",
    ]

    for item in required_content:
        assert normalize_text(item) in combined_text, (
            f"Expected table content missing: {item}"
        )



# 3. PAGE PROVENANCE VALIDATION


def test_chunk_page_numbers_are_valid(parsed_pdf, pdf_chunks):
    """Verify chunk page references are within document pages."""

    valid_pages = set(parsed_pdf.page_numbers)

    assert valid_pages, "PDF parser returned no page numbers."

    for chunk in pdf_chunks:
        assert chunk.page_numbers, (
            f"Chunk {chunk.chunk_index} has no page provenance."
        )

        assert set(chunk.page_numbers).issubset(valid_pages), (
            f"Chunk {chunk.chunk_index} contains invalid pages: "
            f"{chunk.page_numbers}"
        )



# 4. HEADING CONTEXT VALIDATION


def test_heading_context_is_preserved(pdf_chunks):
    """Verify heading metadata appears in contextualized text."""

    chunks_with_headings = [
        chunk for chunk in pdf_chunks if chunk.headings
    ]

    assert chunks_with_headings, (
        "No chunk contains heading metadata."
    )

    for chunk in chunks_with_headings:
        contextualized = normalize_text(
            chunk.contextualized_text
        )

        for heading in chunk.headings:
            assert normalize_text(heading) in contextualized, (
                f"Heading '{heading}' missing from "
                f"chunk {chunk.chunk_index} context."
            )



# 5. SOURCE METADATA VALIDATION


def test_chunk_source_metadata_is_preserved(pdf_chunks):
    """Verify filename, extension and chunk indexes."""

    assert pdf_chunks

    assert [
        chunk.chunk_index for chunk in pdf_chunks
    ] == list(range(len(pdf_chunks)))

    for chunk in pdf_chunks:
        assert chunk.source_file == SAMPLE_PDF.name
        assert chunk.file_extension == ".pdf"



# 6. MULTIPLE FORMAT PIPELINE VALIDATION


def create_sample_file(extension, directory):
    """Create a small test document for each supported format."""

    file_path = directory / f"sample{extension}"

    if extension == ".docx":
        document = Document()
        document.add_heading("Employee Policy", level=1)
        document.add_paragraph(
            "Employees must protect company information."
        )
        document.save(file_path)

    elif extension == ".xlsx":
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Employee Data"
        sheet.append(["Employee", "Department"])
        sheet.append(["Employee A", "Engineering"])
        workbook.save(file_path)
        workbook.close()

    elif extension == ".html":
        file_path.write_text(
            "<html><body>"
            "<h1>Employee Policy</h1>"
            "<p>Protect company information.</p>"
            "</body></html>",
            encoding="utf-8",
        )

    elif extension == ".csv":
        file_path.write_text(
            "Employee,Department\n"
            "Employee A,Engineering\n",
            encoding="utf-8",
        )

    elif extension == ".txt":
        file_path.write_text(
            "Employee Policy\n"
            "Protect company information.",
            encoding="utf-8",
        )

    elif extension == ".md":
        file_path.write_text(
            "# Employee Policy\n\n"
            "Protect company information.",
            encoding="utf-8",
        )

    return file_path


@pytest.mark.parametrize(
    "extension",
    [
        ".docx",
        ".xlsx",
        ".html",
        ".csv",
        ".txt",
        ".md",
    ],
)
def test_parser_to_chunker_pipeline_for_formats(parser, chunker, tmp_path, extension):
    """Verify representative formats pass through both stages."""

    file_path = create_sample_file(
        extension,
        tmp_path,
    )

    parsed_document = parser.parse(file_path)
    chunks = chunker.chunk_document(parsed_document)

    assert chunks, (
        f"No chunks generated for {extension}"
    )

    for chunk in chunks:
        assert chunk.source_file == file_path.name
        assert chunk.file_extension == extension
        assert chunk.text.strip()
        assert chunk.contextualized_text.strip()



# 7. VALIDATION SUMMARY


def test_print_chunking_quality_summary(chunker, parsed_pdf, pdf_chunks):
    """Print useful metrics after the quality checks."""

    print("\n--- CHUNKING QUALITY SUMMARY ---")
    print("Source:", parsed_pdf.file_name)
    print("Total chunks:", len(pdf_chunks))
    print("Maximum configured tokens:", MAX_TOKENS)

    for chunk in pdf_chunks:
        raw_tokens = chunker.tokenizer.count_tokens(
            text=chunk.text
        )

        contextualized_tokens = (
            chunker.tokenizer.count_tokens(
                text=chunk.contextualized_text
            )
        )

        print(
            f"Chunk {chunk.chunk_index}: "
            f"raw={raw_tokens}, "
            f"contextualized={contextualized_tokens}, "
            f"pages={chunk.page_numbers}, "
            f"headings={chunk.headings}"
        )
