"""Request-format and output-validation tests; no inference server required."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import extract_contract_data as extraction
import pdf_to_markdown as conversion


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = {
            'renter_name': 'Example Renter',
            'total_rent_price': 1250,
            'property_address': dict.fromkeys(extraction.ADDRESS_SCHEMA['properties']),
        }
        self.client = Mock()
        self.reply(json.dumps(self.data))

    def reply(self, content, finish_reason='stop'):
        self.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason,
            )],
        )

    def request(self, backend='nuextract', content='Markdown input'):
        return extraction.request_page_extraction(
            self.client, 'contract', 1, content, 'page-001', self.root,
            extraction.ExtractionConfig(backend=backend, model='test-model'),
        )

    def test_nuextract_text_request(self):
        self.assertEqual(self.request(), self.data)
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request['model'], 'test-model')
        self.assertEqual(request['messages'], [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Markdown input'}]},
        ])
        kwargs = request['extra_body']['chat_template_kwargs']
        self.assertEqual(json.loads(kwargs['template']), extraction.NUEXTRACT_TEMPLATE)
        self.assertFalse(kwargs['enable_thinking'])
        self.assertIn('total monthly', kwargs['instructions'])
        self.assertNotIn('structured_outputs', request['extra_body'])

    def test_nuextract_preserves_image_content(self):
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]
        self.request(content=content)
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request['messages'][0]['content'], content)

    def test_standard_backend_preserves_schema_request(self):
        self.request(backend='llm')
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request['messages'][0]['role'], 'system')
        self.assertEqual(request['extra_body'], {'structured_outputs': {'json': extraction.CONTRACT_SCHEMA}})

    def test_invalid_replies_are_saved(self):
        invalid = [
            'not json', json.dumps({**self.data, 'total_rent_price': '1250'}),
            json.dumps({**self.data, 'extra': True}), '{}', '[]',
        ]
        for content in invalid:
            with self.subTest(content=content):
                self.reply(content)
                with self.assertRaises(RuntimeError):
                    self.request()
                self.assertEqual((self.root/'page-001.invalid.txt').read_text(), content)
        self.reply(json.dumps(self.data))
        self.request()
        self.assertFalse((self.root/'page-001.invalid.txt').exists())

    def test_truncated_reply_rejected_even_if_valid_json(self):
        self.reply(json.dumps(self.data), 'length')
        with self.assertRaisesRegex(RuntimeError, 'context limit'):
            self.request()

    def test_cli_routes_all_sources_and_backends(self):
        for source in extraction.INPUT_DIRS:
            for backend in ("llm", "nuextract"):
                with self.subTest(source=source, backend=backend):
                    model = "nuextract3" if backend == "nuextract" else "gemma"
                    args = SimpleNamespace(
                        source=source, backend=backend, model=model,
                        base_url="http://localhost:8001/v1",
                        input_dir=None, output_dir=None,
                    )
                    results_dir = self.root / "results"
                    with patch.object(extraction, "parse_args", return_value=args), \
                         patch.object(extraction, "RESULTS_DIR", results_dir), \
                         patch.object(extraction, "OpenAI") as client, \
                         patch.object(extraction, "process_markdown") as markdown, \
                         patch.object(extraction, "process_pdfs") as pdf:
                        extraction.main()
                        process = pdf if source == "pdf" else markdown
                        process.assert_called_once_with(
                            client.return_value, extraction.INPUT_DIRS[source],
                            results_dir / source / backend,
                            extraction.ExtractionConfig(backend, model),
                        )
                        client.assert_called_once_with(
                            base_url=args.base_url, api_key=extraction.API_KEY
                        )
                        manifest = json.loads(
                            (results_dir / source / backend / "_run.json").read_text()
                        )
                        self.assertEqual(manifest["source"], source)
                        self.assertEqual(manifest["extractor"], backend)
                        self.assertEqual(manifest["model"], model)


class MarkdownConversionTests(unittest.TestCase):
    def page(self):
        pixmap = Mock()
        pixmap.tobytes.return_value = b"png"
        page = Mock()
        page.get_pixmap.return_value = pixmap
        return page

    def response_client(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="# Markdown"))
            ]
        )
        return client

    def test_all_markdown_output_paths_are_distinct(self):
        self.assertEqual(set(conversion.OUTPUT_DIRS), set(conversion.METHODS))
        self.assertEqual(
            {
                method: path.relative_to(conversion.ROOT_DIR).as_posix()
                for method, path in conversion.OUTPUT_DIRS.items()
            },
            {
                "llm": "markdown/llm",
                "markitdown": "markdown/markitdown",
                "nuextract": "markdown/nuextract",
            },
        )

    def test_nuextract_uses_image_to_markdown_mode(self):
        client = self.response_client()
        result = conversion.page_to_markdown(
            client, self.page(), "nuextract", "nuextract3"
        )
        self.assertEqual(result, "# Markdown")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "nuextract3")
        self.assertEqual(
            request["extra_body"]["chat_template_kwargs"],
            {"mode": "markdown", "enable_thinking": False},
        )
        self.assertEqual(
            [item["type"] for item in request["messages"][0]["content"]],
            ["image_url"],
        )

    def test_llm_uses_transcription_prompt_and_image(self):
        client = self.response_client()
        conversion.page_to_markdown(client, self.page(), "llm", "gemma")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "gemma")
        self.assertEqual(
            [item["type"] for item in request["messages"][0]["content"]],
            ["text", "image_url"],
        )
        self.assertEqual(
            request["messages"][0]["content"][0]["text"], conversion.PROMPT
        )


if __name__ == "__main__":
    unittest.main()
