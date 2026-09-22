"""Unit tests for the demo pipeline; no API key or network access required."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pymupdf

from demo.highlight_pdf import highlight_extraction_boxes
from demo.view_results import REQUIRED_OUTPUT_FILES, discover_output_dirs

from demo.run_demo import (
    DEFAULT_ENV_FILE,
    extract_contract,
    input_pdfs,
    load_environment,
    pdf_to_markdown,
    repair_contexts,
    validate_context,
)
from demo.schema import ContractExtraction


MARKDOWN = """\
# Huurovereenkomst

Huurder: Example Renter
Totaal per maand: € 1.250,00
Adres gehuurde: Example Street 10, 1234 AB Amsterdam
"""


def example_extraction() -> ContractExtraction:
    return ContractExtraction.model_validate(
        {
            "renter_name": {
                "value": "Example Renter",
                "context": "Huurder: Example Renter",
            },
            "total_rent_price": {
                "value": "€ 1.250,00",
                "context": "Totaal per maand: € 1.250,00",
            },
            "property_address": {
                "street_address": {
                    "value": "Example Street 10",
                    "context": "Adres gehuurde: Example Street 10, 1234 AB Amsterdam",
                },
                "postal_code": {
                    "value": "1234 AB",
                    "context": "Adres gehuurde: Example Street 10, 1234 AB Amsterdam",
                },
                "city": {
                    "value": "Amsterdam",
                    "context": "Adres gehuurde: Example Street 10, 1234 AB Amsterdam",
                },
                "country": {"value": None, "context": None},
            },
        }
    )


class DemoTests(unittest.TestCase):
    @patch("demo.run_demo.load_dotenv")
    def test_loads_project_env_without_overriding_shell(self, load_dotenv):
        load_environment()
        load_dotenv.assert_called_once_with(
            dotenv_path=DEFAULT_ENV_FILE,
            override=False,
        )

    def test_pymupdf4llm_conversion_preserves_source_offsets(self):
        extractor = Mock(
            return_value=[
                {
                    "metadata": {"page_number": 1},
                    "text": "Markdown",
                    "page_boxes": [
                        {
                            "index": 0,
                            "class": "text",
                            "bbox": [10, 20, 30, 40],
                            "pos": (0, 8),
                        }
                    ],
                }
            ]
        )
        layout_configurer = Mock()

        conversion = pdf_to_markdown(
            Mock(), extractor=extractor, layout_configurer=layout_configurer
        )

        self.assertEqual(conversion.markdown, "<!-- page: 1 -->\n\nMarkdown")
        layout_configurer.assert_called_once_with(True)
        extractor.assert_called_once()
        request = extractor.call_args.kwargs
        self.assertTrue(request["page_chunks"])
        self.assertFalse(request["use_ocr"])
        box = conversion.source_map["pages"][0]["boxes"][0]
        start = box["markdown_start"]
        end = box["markdown_end"]
        self.assertEqual(conversion.markdown[start:end], "Markdown")

    def test_empty_pymupdf4llm_output_is_rejected(self):
        extractor = Mock(
            return_value=[
                {"metadata": {"page_number": 1}, "text": "  ", "page_boxes": []}
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "image-only"):
            pdf_to_markdown(
                Mock(), extractor=extractor, layout_configurer=Mock()
            )

    def test_local_llm_request_uses_pydantic_json_schema(self):
        expected = example_extraction()
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=expected.model_dump_json()),
                    finish_reason="stop",
                )
            ]
        )

        actual = extract_contract(MARKDOWN, client)

        self.assertEqual(actual, expected)
        request = client.chat.completions.create.call_args.kwargs
        schema = request["extra_body"]["structured_outputs"]["json"]
        self.assertEqual(schema, ContractExtraction.model_json_schema())
        self.assertEqual(
            schema["properties"]["total_rent_price"]["$ref"], "#/$defs/TextEvidence"
        )
        self.assertEqual(request["extra_body"]["thinking_token_budget"], 1536)
        self.assertEqual(request["reasoning_effort"], "high")
        self.assertIn(MARKDOWN, request["messages"][1]["content"])

    def test_context_must_be_verbatim(self):
        extraction = example_extraction()
        validate_context(extraction, MARKDOWN)
        extraction.renter_name.context = "Renter: Example Renter"
        with self.assertRaisesRegex(RuntimeError, "verbatim"):
            validate_context(extraction, MARKDOWN)

    def test_context_markdown_formatting_is_repaired(self):
        extraction = example_extraction()
        markdown = "\n".join(
            [
                "|**Huurder**|Example Renter, geboren in Amsterdam|",
                extraction.total_rent_price.context,
                extraction.property_address.street_address.context,
            ]
        )
        extraction.renter_name.context = (
            "Huurder: Example Renter, geboren in Amsterdam"
        )

        repaired = repair_contexts(extraction, markdown)

        self.assertEqual(repaired, ["renter_name"])
        self.assertEqual(
            extraction.renter_name.context,
            "Huurder**|Example Renter, geboren in Amsterdam",
        )
        validate_context(extraction, markdown)

    def test_ambiguous_formatting_free_context_is_not_repaired(self):
        markdown = "|**Huurder**|Example Renter|\n|**Huurder**|Example Renter|"
        extraction = example_extraction()
        extraction.renter_name.context = "Huurder: Example Renter"

        self.assertEqual(repair_contexts(extraction, markdown), [])
        with self.assertRaisesRegex(RuntimeError, "verbatim"):
            validate_context(extraction, markdown)

    def test_directory_input_finds_pdfs_in_name_order(self):
        with TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            second = directory / "B.PDF"
            first = directory / "a.pdf"
            ignored = directory / "notes.txt"
            second.touch()
            first.touch()
            ignored.touch()

            self.assertEqual(input_pdfs(directory), [first, second])

    def test_directory_input_rejects_directory_without_pdfs(self):
        with TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "No PDF files"):
                input_pdfs(Path(temp_dir))

    def test_viewer_discovers_complete_outputs_in_name_order(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            second = root / "B-contract"
            first = root / "a-contract"
            incomplete = root / "incomplete"
            for directory in (second, first, incomplete):
                directory.mkdir()
            for directory in (second, first):
                for filename in REQUIRED_OUTPUT_FILES:
                    (directory / filename).touch()
            (incomplete / "extraction.json").touch()

            self.assertEqual(discover_output_dirs(root), [first, second])
            self.assertEqual(discover_output_dirs(first), [first])

    def test_highlights_overlapping_source_map_boxes_once(self):
        source_map = {
            "pages": [
                {
                    "page_number": 1,
                    "boxes": [
                        {
                            "index": 0,
                            "class": "text",
                            "bbox": [10, 10, 100, 30],
                            "markdown_start": 0,
                            "markdown_end": len(MARKDOWN),
                        }
                    ],
                }
            ]
        }
        extraction = example_extraction().model_dump(mode="json")

        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_pdf = temp / "input.pdf"
            output_pdf = temp / "highlighted.pdf"
            document = pymupdf.open()
            document.new_page()
            document.save(input_pdf)
            document.close()

            report = highlight_extraction_boxes(
                input_pdf, MARKDOWN, source_map, extraction, output_pdf
            )

            self.assertTrue(output_pdf.is_file())
            self.assertEqual(report["highlight_count"], 1)
            self.assertEqual(report["unmatched_fields"], [])
            highlighted = pymupdf.open(output_pdf)
            try:
                self.assertEqual(len(list(highlighted[0].annots())), 1)
            finally:
                highlighted.close()


if __name__ == "__main__":
    unittest.main()
