#!/usr/bin/env python3
"""Convert PDFs to per-page Markdown using an LLM, MarkItDown, or NuExtract3.

Dependencies:
    pip install openai pymupdf "markitdown[pdf]"

The defaults target the local vLLM server used for this project. Override them
when needed, for example:

    LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_MODEL=gemma python pdf_to_markdown.py
"""

import argparse
import base64
import json
from io import BytesIO
import os
from pathlib import Path

import fitz  # PyMuPDF
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "contracts"
METHODS = ("llm", "markitdown", "nuextract")
OUTPUT_DIRS = {method: ROOT_DIR / "markdown" / method for method in METHODS}

BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("LLM_API_KEY", "local")
MODEL = os.getenv("LLM_MODEL", "gemma")

PROMPT = """\
Transcribe this single PDF page to Markdown as faithfully as possible.

Rules:
- Return only Markdown; do not use a fenced code block and do not add commentary.
- Preserve the original reading order and document hierarchy.
- Preserve headings, paragraphs, numbered and bulleted lists, emphasis, links,
  footnotes, and page-visible section numbering.
- Recreate tables as Markdown tables.
- Preserve the label and any entered value for form fields. Represent an empty
  field exactly once as [blank]; do not reproduce its visual underline.
- Preserve a checkbox only when it belongs to a labeled option. Use ☐ and ☒
  for unchecked and checked boxes.
- Never reproduce runs of dots, underscores, spaces, HTML spacing entities,
  empty checkboxes, or other repeated layout characters.
- Never output more than three identical consecutive punctuation marks or
  symbols. Represent an empty writing area exactly once as [blank area].
- Do not summarize, translate, correct, or invent text.
- Join words split only because of a line break, but keep meaningful paragraph
  breaks. Do not reproduce decorative headers or footers unless they contain
  information relevant to the document.
- Write [onleesbaar] where text is present but genuinely unreadable.
"""


def page_to_markdown(
    client: OpenAI, page: fitz.Page, method: str, model: str
) -> str:
    """Render one PDF page and ask the selected vision model for Markdown."""
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image_base64 = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
    image = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
    }

    if method == "nuextract":
        request = {
            "model": model,
            "messages": [{"role": "user", "content": [image]}],
            "temperature": 0.2,
            "extra_body": {
                "chat_template_kwargs": {
                    "mode": "markdown",
                    "enable_thinking": False,
                }
            },
        }
    else:
        request = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": PROMPT}, image],
                }
            ],
            "temperature": 0.2,
            "top_p": 0.95,
            "frequency_penalty": 0.2,
            "extra_body": {
                "repetition_penalty": 1.2,
                "top_k": 64,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        }

    response = client.chat.completions.create(**request)
    return (response.choices[0].message.content or "").strip()


def convert_pdf(
    converter, pdf_path: Path, output_dir: Path, method: str, model: str = MODEL
) -> None:
    """Convert each PDF page using the selected backend."""
    document_output_dir = output_dir / pdf_path.stem
    document_output_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            print(f"  page {page_number}/{document.page_count}")
            if method == "markitdown":
                with fitz.open() as single_page:
                    single_page.insert_pdf(
                        document, from_page=page_index, to_page=page_index
                    )
                    with BytesIO(single_page.tobytes()) as stream:
                        markdown = converter.convert_stream(
                            stream, file_extension=".pdf"
                        ).text_content.strip()
            else:
                markdown = page_to_markdown(converter, page, method, model)

            output_path = document_output_dir / f"page-{page_number:03d}.md"
            output_path.write_text(markdown + "\n", encoding="utf-8")


def write_run_manifest(
    output_dir: Path, method: str, input_dir: Path, model: str, base_url: str | None
) -> None:
    manifest = {
        "producer": method,
        "input_dir": str(input_dir.resolve()),
        "model": model,
        "base_url": base_url,
    }
    (output_dir / "_run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="llm")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, help="Override the output folder.")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument(
        "--model",
        help=(
            "Served model name (default: LLM_MODEL/gemma or "
            "NUEXTRACT_MODEL/nuextract3)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_paths = sorted(args.input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {args.input_dir}")

    if args.method == "markitdown":
        try:
            from markitdown import MarkItDown
        except ImportError as error:
            raise SystemExit(
                'Install MarkItDown with: pip install "markitdown[pdf]"'
            ) from error
        converter = MarkItDown(enable_plugins=False)
        model = "markitdown"
    else:
        converter = OpenAI(base_url=args.base_url, api_key=API_KEY)
        if args.method == "nuextract":
            model = args.model or os.getenv("NUEXTRACT_MODEL", "nuextract3")
        else:
            model = args.model or MODEL

    output_dir = args.output_dir or OUTPUT_DIRS[args.method]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        output_dir, args.method, args.input_dir, model,
        None if args.method == "markitdown" else args.base_url,
    )
    for index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{index}/{len(pdf_paths)}] {pdf_path.name}")
        convert_pdf(converter, pdf_path, output_dir, args.method, model)


if __name__ == "__main__":
    main()
