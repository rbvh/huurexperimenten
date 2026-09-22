#!/usr/bin/env python3
"""Highlight PDF layout boxes that support fields in an extraction result."""

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import pymupdf


def iter_contexts(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield dotted field names and non-empty evidence contexts."""
    if isinstance(value, dict):
        context = value.get("context")
        if isinstance(context, str) and context:
            yield path, context
            return
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            yield from iter_contexts(child, child_path)


def find_occurrences(text: str, substring: str) -> list[tuple[int, int]]:
    """Return all character ranges for substring in text."""
    ranges = []
    cursor = 0
    while True:
        start = text.find(substring, cursor)
        if start == -1:
            return ranges
        end = start + len(substring)
        ranges.append((start, end))
        cursor = start + 1


def boxes_for_range(
    source_map: dict[str, Any], start: int, end: int
) -> Iterator[dict[str, Any]]:
    """Yield source-map boxes whose Markdown range overlaps start:end."""
    for page in source_map.get("pages", []):
        for box in page.get("boxes", []):
            if box["markdown_start"] < end and box["markdown_end"] > start:
                yield {
                    "page_number": page["page_number"],
                    "box_index": box["index"],
                    "class": box["class"],
                    "bbox": box["bbox"],
                    "markdown_start": box["markdown_start"],
                    "markdown_end": box["markdown_end"],
                }


def highlight_extraction_boxes(
    pdf_path: Path,
    markdown: str,
    source_map: dict[str, Any],
    extraction: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Highlight every source-map box overlapping extracted evidence."""
    matches = []
    unmatched_fields = []
    unique_boxes: dict[tuple[int, tuple[float, ...]], dict[str, Any]] = {}

    for field, context in iter_contexts(extraction):
        occurrences = find_occurrences(markdown, context)
        if not occurrences:
            unmatched_fields.append(field)
            continue

        field_boxes = []
        for start, end in occurrences:
            for box in boxes_for_range(source_map, start, end):
                key = (box["page_number"], tuple(box["bbox"]))
                unique = unique_boxes.setdefault(key, {**box, "fields": []})
                if field not in unique["fields"]:
                    unique["fields"].append(field)
                field_boxes.append(box)

        matches.append(
            {
                "field": field,
                "context": context,
                "markdown_ranges": [list(item) for item in occurrences],
                "boxes": field_boxes,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    try:
        for box in unique_boxes.values():
            page_index = int(box["page_number"]) - 1
            if page_index < 0 or page_index >= document.page_count:
                raise ValueError(
                    f"Source map refers to missing PDF page {box['page_number']}."
                )
            page = document[page_index]
            annotation = page.add_rect_annot(pymupdf.Rect(box["bbox"]))
            annotation.set_colors(stroke=(1, 0.8, 0), fill=(1, 0.8, 0))
            annotation.set_border(width=0)
            annotation.set_opacity(0.25)
            annotation.update()
        document.save(output_path)
    finally:
        document.close()

    return {
        "input_pdf": str(pdf_path.resolve()),
        "highlighted_pdf": str(output_path.resolve()),
        "highlight_count": len(unique_boxes),
        "highlighted_boxes": list(unique_boxes.values()),
        "matches": matches,
        "unmatched_fields": unmatched_fields,
    }


def highlight_from_output_dir(
    document_dir: Path,
    pdf_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[Path, Path]:
    """Load an existing demo result directory and create its highlighted PDF."""
    document_dir = document_dir.resolve()
    run_data = json.loads((document_dir / "run.json").read_text(encoding="utf-8"))
    input_pdf = (pdf_path or Path(run_data["input_pdf"])).resolve()
    highlighted_pdf = output_path or document_dir / "highlighted.pdf"

    report = highlight_extraction_boxes(
        pdf_path=input_pdf,
        markdown=(document_dir / "document.md").read_text(encoding="utf-8"),
        source_map=json.loads(
            (document_dir / "source-map.json").read_text(encoding="utf-8")
        ),
        extraction=json.loads(
            (document_dir / "extraction.json").read_text(encoding="utf-8")
        ),
        output_path=highlighted_pdf,
    )
    report_path = document_dir / "highlights.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return highlighted_pdf, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir", type=Path, help="Existing per-document demo output directory."
    )
    parser.add_argument(
        "--pdf", type=Path, help="Override the input PDF stored in run.json."
    )
    parser.add_argument(
        "--output", type=Path, help="Highlighted PDF path (default: highlighted.pdf)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    highlighted_pdf, report_path = highlight_from_output_dir(
        args.output_dir, args.pdf, args.output
    )
    print(f"Saved highlighted PDF to {highlighted_pdf}")
    print(f"Saved highlight report to {report_path}")


if __name__ == "__main__":
    main()
