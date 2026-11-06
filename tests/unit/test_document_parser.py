
from pathlib import Path

import pytest
from docx import Document
from openpyxl import Workbook

from backend.ingestion.document_parser import (DocumentParser,DocumentParsingError)


@pytest.fixture(scope="module")
def parser():
    """Create one parser instance for this test module."""
    return DocumentParser()


def test_hr_policy_pdf_parsing(parser):
    """Verify parsing of the existing HR policy PDF."""

    sample_file = (
        Path(__file__).resolve().parents[1]
        / "IT_Industry_HR_Policies.pdf"
    )

    assert sample_file.exists(), "Sample PDF not found."

    result = parser.parse(sample_file)

    assert result.file_name == "IT_Industry_HR_Policies.pdf"
    assert result.file_extension == ".pdf"
    assert result.markdown_content.strip()
    assert result.page_numbers
    assert result.headings

    print("\n--- PDF PARSING RESULTS ---")
    print("Filename:", result.file_name)
    print("Headings:", result.headings)
    print("Pages:", result.page_numbers)
    print("Tables:", result.table_count)


def test_docx_parsing(parser, tmp_path):
    """Verify DOCX document parsing."""

    file_path = tmp_path / "sample.docx"

    document = Document()
    document.add_heading("Employee Policy", level=1)
    document.add_paragraph(
        "Employees must follow company security policies."
    )
    document.save(file_path)

    result = parser.parse(file_path)

    assert result.file_extension == ".docx"
    assert result.markdown_content.strip()
    assert "Employee Policy" in result.markdown_content


def test_xlsx_parsing(parser, tmp_path):
    """Verify XLSX spreadsheet parsing."""

    file_path = tmp_path / "sample.xlsx"

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Employee Data"

    sheet.append(["Employee", "Department"])
    sheet.append(["Employee A", "Engineering"])

    workbook.save(file_path)

    result = parser.parse(file_path)

    assert result.file_extension == ".xlsx"
    assert result.markdown_content.strip()
    assert "Engineering" in result.markdown_content


def test_xls_parsing(parser, tmp_path):
    """Verify legacy XLS parsing when xlwt is available."""

    xlwt = pytest.importorskip(
        "xlwt",
        reason="xlwt is required to generate a legacy XLS test file.",
    )

    file_path = tmp_path / "sample.xls"

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("Employee Data")

    sheet.write(0, 0, "Employee")
    sheet.write(0, 1, "Department")
    sheet.write(1, 0, "Employee A")
    sheet.write(1, 1, "Engineering")

    workbook.save(str(file_path))

    result = parser.parse(file_path)

    assert result.file_extension == ".xls"
    assert result.markdown_content.strip()


def test_html_parsing(parser, tmp_path):
    """Verify HTML document parsing."""

    file_path = tmp_path / "sample.html"

    file_path.write_text(
        """
        <html>
            <body>
                <h1>Company Policy</h1>
                <p>Employees must protect company information.</p>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = parser.parse(file_path)

    assert result.file_extension == ".html"
    assert result.markdown_content.strip()
    assert "Company Policy" in result.markdown_content


def test_csv_parsing(parser, tmp_path):
    """Verify CSV document parsing."""

    file_path = tmp_path / "sample.csv"

    file_path.write_text(
        "Employee,Department\n"
        "Employee A,Engineering\n"
        "Employee B,Human Resources\n",
        encoding="utf-8",
    )

    result = parser.parse(file_path)

    assert result.file_extension == ".csv"
    assert result.markdown_content.strip()


def test_txt_parsing(parser, tmp_path):
    """Verify plain-text document parsing."""

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "Company Information Security Policy\n"
        "Employees must protect confidential information.",
        encoding="utf-8",
    )

    result = parser.parse(file_path)

    assert result.file_extension == ".txt"
    assert result.markdown_content.strip()
    assert "confidential information" in result.markdown_content


def test_markdown_parsing(parser, tmp_path):
    """Verify Markdown document parsing."""

    file_path = tmp_path / "sample.md"

    file_path.write_text(
        "# Employee Handbook\n\n"
        "## Security Policy\n\n"
        "Employees must protect company information.\n",
        encoding="utf-8",
    )

    result = parser.parse(file_path)

    assert result.file_extension == ".md"
    assert result.markdown_content.strip()
    assert "Employee Handbook" in result.markdown_content


def test_missing_file_error(parser, tmp_path):
    """Verify handling of a missing document."""

    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(
        DocumentParsingError,
        match="File does not exist"):
        parser.parse(missing_file)


def test_unsupported_extension_error(parser, tmp_path):
    """Verify rejection of unsupported extensions."""

    unsupported_file = tmp_path / "sample.exe"
    unsupported_file.write_text(
        "Unsupported document",
        encoding="utf-8",
    )

    with pytest.raises(
        DocumentParsingError,
        match="Unsupported file extension"):
        parser.parse(unsupported_file)









