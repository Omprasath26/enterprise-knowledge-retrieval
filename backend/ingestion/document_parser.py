
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import xlrd
from openpyxl import Workbook

from docling.document_converter import DocumentConverter


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".csv",
    ".txt",
    ".md",
    ".markdown",
}


class DocumentParsingError(Exception):
    """Raised when a document cannot be parsed."""


@dataclass
class ParsedDocument:
    """Structured output produced by the document parser."""

    file_name: str
    file_extension: str
    markdown_content: str
    document_data: dict[str, Any]
    headings: list[str]
    page_numbers: list[int]
    table_count: int


class DocumentParser:
    """Parse supported documents using Docling."""

    def __init__(self) -> None:
        self.converter = DocumentConverter()

    @staticmethod
    def _convert_xls_to_xlsx(
        source_path: Path,
        output_path: Path) -> Path:
        """Convert a legacy XLS workbook into a temporary XLSX workbook."""

        source_workbook = None
        target_workbook = Workbook()

        try:
            source_workbook = xlrd.open_workbook(
                str(source_path),
                formatting_info=False,
            )

            # Remove the default worksheet.
            target_workbook.remove(target_workbook.active)

            for source_sheet in source_workbook.sheets():
                target_sheet = target_workbook.create_sheet(
                    title=source_sheet.name
                )

                for row_index in range(source_sheet.nrows):
                    for col_index in range(source_sheet.ncols):
                        cell = source_sheet.cell(row_index, col_index)

                        if cell.ctype in (
                            xlrd.XL_CELL_EMPTY,
                            xlrd.XL_CELL_BLANK,
                        ):
                            value = None

                        elif cell.ctype == xlrd.XL_CELL_DATE:
                            value = xlrd.xldate_as_datetime(
                                cell.value,
                                source_workbook.datemode,
                            )

                        elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                            value = bool(cell.value)

                        elif cell.ctype == xlrd.XL_CELL_ERROR:
                            value = xlrd.error_text_from_code.get(
                                cell.value,
                                "#ERROR",
                            )

                        else:
                            value = cell.value

                        target_sheet.cell(
                            row=row_index + 1,
                            column=col_index + 1,
                            value=value,
                        )

                # Preserve merged-cell ranges.
                for row_start, row_end, col_start, col_end in (
                    source_sheet.merged_cells
                ):
                    target_sheet.merge_cells(
                        start_row=row_start + 1,
                        end_row=row_end,
                        start_column=col_start + 1,
                        end_column=col_end,
                    )

            target_workbook.save(output_path)

            return output_path

        finally:
            target_workbook.close()

            if source_workbook is not None:
                source_workbook.release_resources()

    @staticmethod
    def _extract_metadata(document_data: dict[str, Any]) -> tuple[list[str], list[int], int]:
        headings: list[str] = []
        page_numbers: set[int] = set()

        def inspect(value: Any) -> None:
            if isinstance(value, dict):
                page_no = value.get("page_no")

                if isinstance(page_no, int):
                    page_numbers.add(page_no)

                label = value.get("label")
                text = value.get("text")

                if label == "section_header" and isinstance(text, str):
                    headings.append(text.strip())

                for nested_value in value.values():
                    inspect(nested_value)

            elif isinstance(value, list):
                for item in value:
                    inspect(item)

        inspect(document_data)

        tables = document_data.get("tables", [])
        table_count = len(tables) if isinstance(tables, list) else 0

        return headings, sorted(page_numbers), table_count

    def _parse_with_docling(
        self,
        parsing_path: Path,
        original_path: Path) -> ParsedDocument:
        """Parse a document with Docling and preserve original file details."""

        conversion_result = self.converter.convert(str(parsing_path))
        document = conversion_result.document

        markdown_content = document.export_to_markdown()
        document_data = document.export_to_dict()

        headings, page_numbers, table_count = self._extract_metadata(
            document_data
        )

        return ParsedDocument(
            file_name=original_path.name,
            file_extension=original_path.suffix.lower(),
            markdown_content=markdown_content,
            document_data=document_data,
            headings=headings,
            page_numbers=page_numbers,
            table_count=table_count,
        )

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)

        if not path.exists():
            raise DocumentParsingError(
                f"File does not exist: {path}"
            )

        if not path.is_file():
            raise DocumentParsingError(
                f"Provided path is not a file: {path}"
            )

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise DocumentParsingError(
                f"Unsupported file extension: {extension}"
            )

        try:
            if extension == ".xls":
                # TemporaryDirectory automatically removes converted files.
                with TemporaryDirectory(
                    prefix="docling_xls_"
                ) as temporary_directory:
                    converted_path = (
                        Path(temporary_directory)
                        / f"{path.stem}_converted.xlsx"
                    )

                    self._convert_xls_to_xlsx(
                        path,
                        converted_path,
                    )

                    return self._parse_with_docling(
                        converted_path,
                        path,
                    )

            return self._parse_with_docling(path, path)

        except Exception as exc:
            raise DocumentParsingError(
                f"Failed to parse '{path.name}': {exc}"
            ) from exc


if __name__ == "__main__":
    print("Document parser module loaded successfully.")
