#!/usr/bin/env python3
"""Convert one PDF, or all PDFs in a directory, using a local LLM."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError
import pymupdf4llm

if __package__:
    from .highlight_pdf import highlight_extraction_boxes
    from .prompt import SYSTEM_PROMPT, extraction_prompt
    from .schema import ContractExtraction
else:
    # Support: python demo/run_demo.py contracts/example.pdf
    from highlight_pdf import highlight_extraction_boxes
    from prompt import SYSTEM_PROMPT, extraction_prompt
    from schema import ContractExtraction


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "local"
DEFAULT_MODEL = "gemma"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "demo" / "output"


@dataclass(frozen=True)
class MarkdownConversion:
    """Markdown plus a character-offset map back to PDF layout boxes."""

    markdown: str
    source_map: dict


def load_environment(env_file: Path = DEFAULT_ENV_FILE) -> None:
    """Load project-local settings without overriding shell environment values."""
    load_dotenv(dotenv_path=env_file, override=False)


def pdf_to_markdown(
    pdf_path: Path,
    extractor=pymupdf4llm.to_markdown,
    layout_configurer=pymupdf4llm.use_layout,
) -> MarkdownConversion:
    """Convert a PDF by page and retain Markdown-to-layout-box offsets."""
    layout_configurer(True)
    chunks = extractor(
        str(pdf_path),
        page_chunks=True,
        use_ocr=False,
    )
    if not isinstance(chunks, list):
        raise RuntimeError("PyMuPDF4LLM did not return page chunks.")

    sections = []
    pages = []
    cursor = 0
    for fallback_number, chunk in enumerate(chunks, start=1):
        # Preserve the page text exactly because page-box offsets refer to it.
        page_text = chunk.get("text", "")
        page_number = chunk.get("metadata", {}).get(
            "page_number", fallback_number
        )
        separator = "\n\n" if sections else ""
        marker = f"<!-- page: {page_number} -->\n\n"
        section = separator + marker + page_text
        content_start = cursor + len(separator) + len(marker)
        content_end = content_start + len(page_text)

        boxes = []
        for box in chunk.get("page_boxes", []):
            local_start, local_end = box["pos"]
            boxes.append(
                {
                    "index": box["index"],
                    "class": box["class"],
                    "bbox": list(box["bbox"]),
                    "markdown_start": content_start + local_start,
                    "markdown_end": content_start + local_end,
                }
            )

        sections.append(section)
        pages.append(
            {
                "page_number": page_number,
                "markdown_start": content_start,
                "markdown_end": content_end,
                "boxes": boxes,
            }
        )
        cursor += len(section)

    markdown = "".join(sections)
    if not any(chunk.get("text", "").strip() for chunk in chunks):
        raise RuntimeError(
            "PyMuPDF4LLM returned no text. The PDF may be image-only; OCR is "
            "disabled in this initial demo."
        )
    return MarkdownConversion(
        markdown=markdown,
        source_map={
            "producer": "pymupdf4llm",
            "producer_version": pymupdf4llm.version,
            "layout_model": True,
            "ocr": False,
            "pages": pages,
        },
    )


def extract_contract(
    markdown: str,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
) -> ContractExtraction:
    """Ask the local vLLM server for a Pydantic-schema-constrained extraction."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": extraction_prompt(markdown)},
        ],
        extra_body={
            "structured_outputs": {"json": ContractExtraction.model_json_schema()},
            "thinking_token_budget": 1536,
        },
        reasoning_effort="high",
        temperature=0,
    )
    choice = completion.choices[0]
    content = choice.message.content
    if not content:
        raise RuntimeError("The local LLM returned no extraction result.")
    if choice.finish_reason == "length":
        raise RuntimeError("The local LLM response reached its context limit.")
    try:
        return ContractExtraction.model_validate_json(content)
    except ValidationError as error:
        raise RuntimeError(
            f"The local LLM returned invalid extraction JSON: {error}"
        ) from error


def evidence_fields(extraction: ContractExtraction) -> dict:
    """Return each extractable field's value-and-context object."""
    return {
        "renter_name": extraction.renter_name,
        "total_rent_price": extraction.total_rent_price,
        "property_address.street_address": extraction.property_address.street_address,
        "property_address.postal_code": extraction.property_address.postal_code,
        "property_address.city": extraction.property_address.city,
        "property_address.country": extraction.property_address.country,
    }


def normalized_text_with_positions(text: str) -> tuple[str, list[int]]:
    """Normalize words while retaining their character positions in text."""
    normalized = []
    positions = []
    for position, character in enumerate(text):
        for folded in character.casefold():
            if folded.isalnum():
                normalized.append(folded)
                positions.append(position)
            elif normalized and normalized[-1] != " ":
                normalized.append(" ")
                positions.append(position)
    if normalized and normalized[-1] == " ":
        normalized.pop()
        positions.pop()
    return "".join(normalized), positions


def find_context_ignoring_formatting(context: str, markdown: str) -> str | None:
    """Find one context match while ignoring Markdown and punctuation syntax."""
    normalized_markdown, positions = normalized_text_with_positions(markdown)
    normalized_context, _ = normalized_text_with_positions(context)
    if not normalized_context:
        return None

    starts = []
    cursor = 0
    while True:
        start = normalized_markdown.find(normalized_context, cursor)
        if start == -1:
            break
        starts.append(start)
        cursor = start + 1
    if len(starts) != 1:
        return None

    start = starts[0]
    end = start + len(normalized_context)
    return markdown[positions[start] : positions[end - 1] + 1]


def repair_contexts(extraction: ContractExtraction, markdown: str) -> list[str]:
    """Replace uniquely matched formatting-free contexts with verbatim Markdown."""
    repaired = []
    for field, item in evidence_fields(extraction).items():
        if item.context and item.context not in markdown:
            exact_context = find_context_ignoring_formatting(item.context, markdown)
            if exact_context is not None:
                item.context = exact_context
                repaired.append(field)
    return repaired


def validate_context(extraction: ContractExtraction, markdown: str) -> None:
    """Ensure evidence is present for values and copied from the source text."""
    errors = []
    for field, item in evidence_fields(extraction).items():
        if item.value is None and item.context is not None:
            errors.append(f"{field}: context must be null when value is null")
        elif item.value is not None and not item.context:
            errors.append(f"{field}: extracted value has no context")
        elif item.context is not None and item.context not in markdown:
            errors.append(f"{field}: context is not a verbatim Markdown excerpt")

    if errors:
        raise RuntimeError("Invalid extraction evidence:\n- " + "\n- ".join(errors))


def input_pdfs(input_path: Path) -> list[Path]:
    """Resolve a PDF or a directory containing PDFs into an ordered file list."""
    if input_path.is_file():
        if input_path.suffix.casefold() != ".pdf":
            raise ValueError(f"Expected a PDF file: {input_path}")
        return [input_path]
    if input_path.is_dir():
        pdfs = sorted(
            (
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.casefold() == ".pdf"
            ),
            key=lambda path: path.name.casefold(),
        )
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found in directory: {input_path}")
        return pdfs
    raise FileNotFoundError(f"Input path not found: {input_path}")


def run(
    pdf_path: Path,
    output_root: Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> Path:
    """Run conversion and extraction, returning the JSON output path."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.casefold() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    document_dir = output_root / pdf_path.stem
    document_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {pdf_path.name} with PyMuPDF4LLM...")
    conversion = pdf_to_markdown(pdf_path)
    markdown = conversion.markdown
    markdown_path = document_dir / "document.md"
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    source_map_path = document_dir / "source-map.json"
    source_map_path.write_text(
        json.dumps(conversion.source_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Extracting contract data with local model {model}...")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=300.0,
        max_retries=4,
    )
    extraction = extract_contract(markdown, client, model)
    repaired_fields = repair_contexts(extraction, markdown)
    if repaired_fields:
        print("Restored exact Markdown context for: " + ", ".join(repaired_fields))
    validate_context(extraction, markdown)

    result_path = document_dir / "extraction.json"
    result_path.write_text(
        extraction.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    highlighted_path = document_dir / "highlighted.pdf"
    highlight_report = highlight_extraction_boxes(
        pdf_path=pdf_path,
        markdown=markdown,
        source_map=conversion.source_map,
        extraction=extraction.model_dump(mode="json"),
        output_path=highlighted_path,
    )
    highlights_path = document_dir / "highlights.json"
    highlights_path.write_text(
        json.dumps(highlight_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (document_dir / "run.json").write_text(
        json.dumps(
            {
                "input_pdf": str(pdf_path.resolve()),
                "markdown": str(markdown_path.resolve()),
                "source_map": str(source_map_path.resolve()),
                "highlighted_pdf": str(highlighted_path.resolve()),
                "highlights": str(highlights_path.resolve()),
                "markdown_producer": "pymupdf4llm",
                "markdown_producer_version": pymupdf4llm.version,
                "layout_model": True,
                "ocr": False,
                "model": model,
                "base_url": base_url,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved highlighted PDF to {highlighted_path}")
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Rental-contract PDF or directory of PDFs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        help="Served model name (default: LLM_MODEL or gemma).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible endpoint (default: {DEFAULT_BASE_URL}).",
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()
    api_key = os.getenv("LLM_API_KEY", DEFAULT_API_KEY)

    pdfs = input_pdfs(args.input)
    for index, pdf_path in enumerate(pdfs, start=1):
        if len(pdfs) > 1:
            print(f"\n[{index}/{len(pdfs)}] Processing {pdf_path.name}")
        result_path = run(
            pdf_path, args.output_dir, api_key, args.model, args.base_url
        )
        print(f"Saved extraction to {result_path}")


if __name__ == "__main__":
    main()
