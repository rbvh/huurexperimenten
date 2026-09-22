#!/usr/bin/env python3
"""Extract structured JSON from rental-contract Markdown pages or PDF page images.

Dependencies:
    pip install openai pymupdf jsonschema

The defaults match the local vLLM server used by pdf_to_markdown.py.

For the default llm backend, start vLLM with compact structured JSON enabled; request-level whitespace
control is currently ignored by vLLM's XGrammar backend:
    --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
"""

import argparse
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from jsonschema import Draft202012Validator, ValidationError
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIRS = {
    "markdown_llm": ROOT_DIR / "markdown" / "llm",
    "markdown_markitdown": ROOT_DIR / "markdown" / "markitdown",
    "markdown_nuextract": ROOT_DIR / "markdown" / "nuextract",
    "pdf": ROOT_DIR / "contracts",
}
RESULTS_DIR = ROOT_DIR / "results"

BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("LLM_API_KEY", "local")
MODEL = os.getenv("LLM_MODEL", "gemma")

NULLABLE_STRING = {"type": ["string", "null"]}
NULLABLE_NUMBER = {"type": ["number", "null"]}

ADDRESS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "street_address": NULLABLE_STRING,
        "postal_code": NULLABLE_STRING,
        "city": NULLABLE_STRING,
        "country": NULLABLE_STRING,
    },
    "required": ["street_address", "postal_code", "city", "country"],
}

CONTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "renter_name": NULLABLE_STRING,
        "total_rent_price": NULLABLE_NUMBER,
        "property_address": ADDRESS_SCHEMA,
    },
    "required": ["renter_name", "total_rent_price", "property_address"],
}

# NuExtract uses a typed output template, not JSON Schema, in its chat template.
NUEXTRACT_TEMPLATE = {
    "renter_name": "string",
    "total_rent_price": "number",
    "property_address": {
        field: "verbatim-string" for field in ADDRESS_SCHEMA["properties"]
    },
}
CONTRACT_VALIDATOR = Draft202012Validator(CONTRACT_SCHEMA)


@dataclass(frozen=True)
class ExtractionConfig:
    backend: str = "llm"
    model: str = MODEL


SYSTEM_PROMPT = """\
You extract agreement-specific facts from Dutch residential rental contracts.
Return compact JSON that exactly follows the supplied schema.

Rules:
- The input contains exactly one page. Use only facts visible on that page.
- Extract only values explicitly completed or selected for this agreement.
- Do not treat examples, explanatory text, blank form options, or general legal
  clauses as agreement-specific facts. Never guess; use null for missing values.
- renter_name is the full name of the renter (huurder), not the landlord or agent.
  If multiple renters are explicitly named, list their names in document order
  in this string, separated by a semicolon and a space.
- total_rent_price is the explicitly stated total monthly rental payment,
  including service costs and utilities when included in that stated total.
  Exclude deposits and one-time payments. Do not calculate a total from components.
  Return a JSON number without currency signs or thousands separators.
- property_address is the address of the rented property, not a party's contact
  address. Use null for each missing address component; do not infer the country.
- Do not infer values from other pages, filenames, or general knowledge.
- If a field is ambiguous or contradictory on this page, return null for it.
"""


def page_number(page_path: Path) -> int:
    return int(page_path.stem.rsplit("-", 1)[1])


def request_page_extraction(
    client: OpenAI,
    contract_name: str,
    number: int,
    user_content,
    page_stem: str,
    page_output_dir: Path,
    config: ExtractionConfig = ExtractionConfig(),
) -> dict:
    """Request and validate one schema-constrained page extraction."""
    if config.backend == "nuextract":
        content_parts = (
            [{"type": "text", "text": user_content}]
            if isinstance(user_content, str) else user_content
        )
        messages = [{"role": "user", "content": content_parts}]
        extra_body = {
            "chat_template_kwargs": {
                "template": json.dumps(NUEXTRACT_TEMPLATE),
                "instructions": SYSTEM_PROMPT.replace("supplied schema", "supplied template"),
                "enable_thinking": False,
            },
        }
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        extra_body = {"structured_outputs": {"json": CONTRACT_SCHEMA}}

    response = client.chat.completions.create(
        model=config.model,
        messages=messages,
        extra_body=extra_body,
        temperature=0.2 if config.backend == "nuextract" else 0,
    )

    choice = response.choices[0]
    content = choice.message.content
    if not content:
        raise RuntimeError(f"Model returned no JSON for {contract_name} page {number}")

    diagnostic_path = page_output_dir / f"{page_stem}.invalid.txt"
    if choice.finish_reason == "length":
        diagnostic_path.write_text(content, encoding="utf-8")
        raise RuntimeError(
            f"JSON for {contract_name} page {number} reached the context limit. "
            f"Raw output: {diagnostic_path}"
        )

    try:
        data = json.loads(content)
        CONTRACT_VALIDATOR.validate(data)
    except (json.JSONDecodeError, ValidationError) as error:
        diagnostic_path.write_text(content, encoding="utf-8")
        raise RuntimeError(
            f"Invalid JSON or schema mismatch for {contract_name} page {number} "
            f"(finish_reason={choice.finish_reason!r}). "
            f"Raw output: {diagnostic_path}"
        ) from error

    diagnostic_path.unlink(missing_ok=True)
    return data


def extract_markdown_page(
    client: OpenAI,
    contract_name: str,
    page_path: Path,
    page_output_dir: Path,
    config: ExtractionConfig = ExtractionConfig(),
) -> dict:
    """Extract facts from one Markdown page."""
    markdown = page_path.read_text(encoding="utf-8")
    number = page_number(page_path)
    user_content = (
        f"Contract: {contract_name}; page: {number}"
        + chr(10) * 2
        + "Page Markdown:"
        + chr(10) * 2
        + markdown
    )
    return request_page_extraction(
        client,
        contract_name,
        number,
        user_content,
        page_path.stem,
        page_output_dir,
        config,
    )


def extract_pdf_page(
    client: OpenAI,
    contract_name: str,
    number: int,
    page,
    page_output_dir: Path,
    config: ExtractionConfig = ExtractionConfig(),
) -> dict:
    """Render one PDF page and extract facts directly from its image."""
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image_base64 = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
    user_content = [
        {
            "type": "text",
            "text": (
                f"Contract: {contract_name}; page: {number}. "
                "Extract only facts explicitly visible in this page image."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            },
        },
    ]
    page_stem = f"page-{number:03d}"
    return request_page_extraction(
        client,
        contract_name,
        number,
        user_content,
        page_stem,
        page_output_dir,
        config,
    )


def value_key(value):
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, (int, float)):
        return ("number", float(value))
    if isinstance(value, str):
        return ("string", " ".join(value.casefold().split()))
    return ("json", json.dumps(value, ensure_ascii=False, sort_keys=True))


def display_value(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def resolve_entries(
    entries: list[tuple[int, object]],
    field: str,
    conflicts: list[dict],
    default=None,
    ignored=(),
):
    """Return one unique value or record every conflicting candidate."""
    ignored_keys = {value_key(value) for value in ignored}
    grouped = {}

    for number, value in entries:
        if value is None or value_key(value) in ignored_keys:
            continue
        key = value_key(value)
        if key not in grouped:
            grouped[key] = {"value": value, "pages": []}
        grouped[key]["pages"].append(number)

    if not grouped:
        return default
    if len(grouped) == 1:
        return next(iter(grouped.values()))["value"]

    conflicts.append(
        {
            "field": field,
            "candidates": [
                {
                    "pages": sorted(set(candidate["pages"])),
                    "value": display_value(candidate["value"]),
                }
                for candidate in grouped.values()
            ],
        }
    )
    return default


def nested_value(data: dict, path: str):
    value = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def resolve_pages(
    page_results: list[tuple[int, dict]],
    path: str,
    conflicts: list[dict],
    default=None,
    ignored=(),
):
    entries = [(number, nested_value(data, path)) for number, data in page_results]
    return resolve_entries(entries, path, conflicts, default, ignored)


def merge_contract(contract_name: str, page_results: list[tuple[int, dict]]) -> dict:
    """Combine page facts, leaving conflicting values null."""
    conflicts = []
    combined = {
        "renter_name": resolve_pages(page_results, "renter_name", conflicts),
        "total_rent_price": resolve_pages(page_results, "total_rent_price", conflicts),
        "property_address": {
            field: resolve_pages(page_results, f"property_address.{field}", conflicts)
            for field in ADDRESS_SCHEMA["properties"]
        },
    }
    for conflict in conflicts:
        print(f"  Warning: {contract_name}: conflicting {conflict['field']}; "
              "combined value is null (see per-page JSON).")
    return combined


def write_page_json(page_output_dir: Path, page_stem: str, data: dict) -> None:
    output_path = page_output_dir / f"{page_stem}.json"
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )


def write_combined_json(
    contract_name: str, page_results: list[tuple[int, dict]], output_dir: Path
) -> None:
    combined = merge_contract(contract_name, page_results)
    output_path = output_dir / f"{contract_name}.json"
    output_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )


def process_markdown(
    client: OpenAI, input_dir: Path, output_dir: Path,
    config: ExtractionConfig = ExtractionConfig(),
) -> None:
    contract_dirs = sorted(
        path for path in input_dir.glob("*")
        if path.is_dir() and any(path.glob("page-*.md"))
    )
    if not contract_dirs:
        raise SystemExit(
            f"No per-page Markdown directories found in {input_dir}. "
            "Run pdf_to_markdown.py first or use --source pdf."
        )

    for contract_index, contract_dir in enumerate(contract_dirs, start=1):
        page_paths = sorted(contract_dir.glob("page-*.md"), key=page_number)
        if not page_paths:
            continue

        print(f"[{contract_index}/{len(contract_dirs)}] {contract_dir.name}")
        page_output_dir = output_dir / contract_dir.name / "pages"
        page_output_dir.mkdir(parents=True, exist_ok=True)
        page_results = []

        for page_index, page_path in enumerate(page_paths, start=1):
            number = page_number(page_path)
            print(f"  page {page_index}/{len(page_paths)}")
            data = extract_markdown_page(
                client, contract_dir.name, page_path, page_output_dir, config
            )
            write_page_json(page_output_dir, page_path.stem, data)
            page_results.append((number, data))

        write_combined_json(contract_dir.name, page_results, output_dir)


def process_pdfs(
    client: OpenAI, input_dir: Path, output_dir: Path,
    config: ExtractionConfig = ExtractionConfig(),
) -> None:
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {input_dir}")

    for contract_index, pdf_path in enumerate(pdf_paths, start=1):
        contract_name = pdf_path.stem
        print(f"[{contract_index}/{len(pdf_paths)}] {pdf_path.name}")
        page_output_dir = output_dir / contract_name / "pages"
        page_output_dir.mkdir(parents=True, exist_ok=True)
        page_results = []

        with fitz.open(pdf_path) as document:
            for number, page in enumerate(document, start=1):
                print(f"  page {number}/{document.page_count}")
                data = extract_pdf_page(
                    client, contract_name, number, page, page_output_dir, config
                )
                page_stem = f"page-{number:03d}"
                write_page_json(page_output_dir, page_stem, data)
                page_results.append((number, data))

        write_combined_json(contract_name, page_results, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract structured rental-contract data page by page."
    )
    parser.add_argument(
        "--source",
        choices=tuple(INPUT_DIRS),
        default="markdown_llm",
        help=(
            "Read Markdown produced by an LLM (default), MarkItDown, or NuExtract; or send PDF page "
            "images directly to the vision model."
        ),
    )
    parser.add_argument(
        "--input-dir", type=Path,
        help="Override the selected source directory.",
    )
    parser.add_argument(
        "--backend", choices=("llm", "nuextract"), default="llm",
        help="Request format: standard schema-constrained LLM or NuExtract3 template.",
    )
    parser.add_argument(
        "--model", help="Served model name (default: LLM_MODEL, or gemma/nuextract3 by backend).",
    )
    parser.add_argument("--base-url", default=BASE_URL, help="OpenAI-compatible server URL.")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Override output folder, for example to compare additional models.",
    )
    return parser.parse_args()


def write_run_manifest(
    output_dir: Path, source: str, input_dir: Path, config: ExtractionConfig,
    base_url: str,
) -> None:
    manifest = {
        "source": source,
        "input_dir": str(input_dir.resolve()),
        "extractor": config.backend,
        "model": config.model,
        "base_url": base_url,
    }
    (output_dir / "_run.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or INPUT_DIRS[args.source]
    default_output = RESULTS_DIR / args.source / args.backend
    output_dir = args.output_dir or default_output
    if args.backend == "nuextract":
        model = args.model or os.getenv("NUEXTRACT_MODEL", "nuextract3")
    else:
        model = args.model or MODEL
    config = ExtractionConfig(backend=args.backend, model=model)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        output_dir, args.source, input_dir, config, args.base_url
    )
    client = OpenAI(base_url=args.base_url, api_key=API_KEY)

    if args.source == "pdf":
        process_pdfs(client, input_dir, output_dir, config)
    else:
        process_markdown(client, input_dir, output_dir, config)


if __name__ == "__main__":
    main()
