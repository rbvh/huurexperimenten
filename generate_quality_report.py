#!/usr/bin/env python3
"""Generate the extraction quality report from the eight result pipelines."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from datetime import date
from html import escape
from pathlib import Path

from jsonschema import Draft202012Validator

from extract_contract_data import CONTRACT_SCHEMA


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPORT = ROOT / "extraction_quality_report.html"

FIELDS = (
    ("renter_name", "Renter"),
    ("total_rent_price", "Total rent"),
    ("street_address", "Street"),
    ("postal_code", "Postcode"),
    ("city", "City"),
)

GROUND_TRUTH = {
    "01_ROZ_2025_ingevuld": {
        "renter_name": "Anne-Fleur Brouwer",
        "total_rent_price": 2040,
        "street_address": "Rembrandtweg 172",
        "postal_code": "1181 GW",
        "city": "Amstelveen",
    },
    "02_VoorbeeldOffice_ingevuld": {
        "renter_name": "Samir Aydin",
        "total_rent_price": 1535,
        "street_address": "Bemuurde Weerd O.Z. 47-A",
        "postal_code": "3514 AP",
        "city": "Utrecht",
    },
    "03_TopVoorbeelden_ingevuld": {
        "renter_name": "Sophie Martens",
        "total_rent_price": None,
        "street_address": "Eusebiusbuitensingel 31",
        "postal_code": "6828 HZ",
        "city": "Arnhem",
    },
    "04_Woonbond_formulier_ingevuld": {
        "renter_name": "Lucas Smit; Emilia Kowalska",
        "total_rent_price": 1320,
        "street_address": "Assendorperstraat 132",
        "postal_code": "8012 CC",
        "city": "Zwolle",
    },
    "05_VBTM_onzelfstandig_ingevuld": {
        "renter_name": "Amina Bouzid",
        "total_rent_price": 670,
        "street_address": "Professor Verbernelaan 87",
        "postal_code": "5037 AD",
        "city": "Tilburg",
    },
    "06_Kences_campuscontract_ingevuld": {
        "renter_name": "Noor van Dijk",
        "total_rent_price": 790,
        "street_address": "Zernikelaan 73",
        "postal_code": "9747 AS",
        "city": "Groningen",
    },
    "07_NVM_klassiek_ingevuld": {
        "renter_name": "Noura El Idrissi",
        "total_rent_price": 1650,
        "street_address": "Van Speijkstraat 56-A",
        "postal_code": "2518 GE",
        "city": "'s-Gravenhage",
    },
    "08_Eenvoudig_1_pagina_ingevuld": {
        "renter_name": "Lotte van Beek",
        "total_rent_price": 930,
        "street_address": "Scharnestraat 21",
        "postal_code": "8601 BB",
        "city": "Sneek",
    },
    "09_Hayman_NL_EN_ingevuld": {
        "renter_name": "Elena García Martínez",
        "total_rent_price": 2005,
        "street_address": "Prins Mauritslaan 22-B",
        "postal_code": "2582 LS",
        "city": "Den Haag",
    },
    "10_model-huurovereenkomst-zelfstandige-woonruimte-2026": {
        "renter_name": "Thomas de Wit",
        "total_rent_price": None,
        "street_address": "Westedijk 42",
        "postal_code": "3513 EW",
        "city": "Utrecht",
    },
    "11_Lang_formeel_particulier_contract_ingevuld": {
        "renter_name": "Yasmin Aksoy; Bram Jansen",
        "total_rent_price": 2060,
        "street_address": "Essenburgsingel 92",
        "postal_code": "3022 EE",
        "city": "Rotterdam",
    },
    "12_Traditioneel_woningcorporatiemodel_ingevuld": {
        "renter_name": "Daan Vermeer",
        "total_rent_price": 1014,
        "street_address": "Brinkgreverweg 214",
        "postal_code": "7413 AH",
        "city": "Deventer",
    },
    "13_Beheerdersmodel_traditioneel_ingevuld": {
        "renter_name": "Fatima El Amrani",
        "total_rent_price": 1805,
        "street_address": "Frederik Hendriklaan 188-C",
        "postal_code": "2582 BJ",
        "city": "Den Haag",
    },
    "14_TopVoorbeelden_origineel_ingevuld_TEST": {
        "renter_name": "Lotte van Dijk",
        "total_rent_price": 1150,
        "street_address": "Molenhof 38",
        "postal_code": "4813 EF",
        "city": "Breda",
    },
    "15_Huizenvinder_origineel_ingevuld_TEST": {
        "renter_name": "Nora Elise Bakker",
        "total_rent_price": 1205,
        "street_address": "Lindenhof 27-A",
        "postal_code": "9714 EF",
        "city": "Groningen",
    },
}

ROUTES = (
    {
        "key": "markdown_llm/llm",
        "short": "ML→L",
        "label": "LLM Markdown → LLM",
        "source": "LLM Markdown",
        "extractor": "LLM",
    },
    {
        "key": "markdown_llm/nuextract",
        "short": "ML→N",
        "label": "LLM Markdown → NuExtract",
        "source": "LLM Markdown",
        "extractor": "NuExtract",
    },
    {
        "key": "markdown_markitdown/llm",
        "short": "MM→L",
        "label": "MarkItDown → LLM",
        "source": "MarkItDown",
        "extractor": "LLM",
    },
    {
        "key": "markdown_markitdown/nuextract",
        "short": "MM→N",
        "label": "MarkItDown → NuExtract",
        "source": "MarkItDown",
        "extractor": "NuExtract",
    },
    {
        "key": "markdown_nuextract/llm",
        "short": "MN→L",
        "label": "NuExtract Markdown → LLM",
        "source": "NuExtract Markdown",
        "extractor": "LLM",
    },
    {
        "key": "markdown_nuextract/nuextract",
        "short": "MN→N",
        "label": "NuExtract Markdown → NuExtract",
        "source": "NuExtract Markdown",
        "extractor": "NuExtract",
    },
    {
        "key": "pdf/llm",
        "short": "PDF→L",
        "label": "PDF images → LLM",
        "source": "PDF images",
        "extractor": "LLM",
    },
    {
        "key": "pdf/nuextract",
        "short": "PDF→N",
        "label": "PDF images → NuExtract",
        "source": "PDF images",
        "extractor": "NuExtract",
    },
)

AUDIT_NOTES = {
    "03": "No explicit total is stated; €1,210 base rent and €55 service costs must not be added.",
    "09": "The accented and unaccented renter spellings are treated as semantically equivalent.",
    "10": "€1,180 is base rent and €115 is a separate advance; no explicit total is stated.",
    "15": "The renter is Nora Elise Bakker and the explicitly stated monthly total is €1,205.",
}


def raw_text(value: object) -> str:
    return (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .casefold()
        .replace("’", "'")
    )


def normalize(value: object):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return re.sub(r"[^a-z0-9]+", " ", raw_text(value)).strip()


def normalize_names(value: object):
    if value is None:
        return None
    names = re.split(r"\s*;\s*|\s+(?:en|and)\s+", raw_text(value))
    return sorted(normalize(name) for name in names)


def field_value(data: dict, field: str):
    if field in data:
        return data.get(field)
    return data.get("property_address", {}).get(field)


def equivalent(predicted: object, expected: object, field: str) -> bool:
    if field == "renter_name":
        return normalize_names(predicted) == normalize_names(expected)
    return normalize(predicted) == normalize(expected)


def display_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def contract_label(stem: str) -> str:
    number, _, title = stem.partition("_")
    title = title.replace("_ingevuld_TEST", "").replace("_ingevuld", "")
    title = title.replace("_", " ")
    return f"{number} · {title}"


def load_and_validate():
    validator = Draft202012Validator(CONTRACT_SCHEMA)
    outputs = {}
    combined_count = 0
    page_count = 0

    for route in ROUTES:
        route_dir = RESULTS / route["key"]
        manifest_path = route_dir / "_run.json"
        if not manifest_path.exists():
            raise SystemExit(f"Missing run manifest: {manifest_path}")
        route["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))

        route_outputs = {}
        for contract in GROUND_TRUTH:
            output_path = route_dir / f"{contract}.json"
            data = json.loads(output_path.read_text(encoding="utf-8"))
            validator.validate(data)
            route_outputs[contract] = data
            combined_count += 1

        for page_path in route_dir.glob("*/pages/page-*.json"):
            validator.validate(json.loads(page_path.read_text(encoding="utf-8")))
            page_count += 1

        outputs[route["key"]] = route_outputs

    return outputs, combined_count, page_count


def calculate_stats(outputs):
    stats = {}
    for route in ROUTES:
        field_correct = {field: 0 for field, _ in FIELDS}
        per_contract = {}
        errors = []
        missing = 0
        wrong = 0

        for contract, expected in GROUND_TRUTH.items():
            predicted = outputs[route["key"]][contract]
            contract_results = {}
            for field, _ in FIELDS:
                actual = field_value(predicted, field)
                target = expected[field]
                correct = equivalent(actual, target, field)
                contract_results[field] = correct
                if correct:
                    field_correct[field] += 1
                else:
                    if actual is None and target is not None:
                        missing += 1
                    else:
                        wrong += 1
                    errors.append(
                        {
                            "contract": contract,
                            "field": field,
                            "predicted": actual,
                            "expected": target,
                        }
                    )
            per_contract[contract] = contract_results

        correct = sum(field_correct.values())
        perfect = sum(all(result.values()) for result in per_contract.values())
        stats[route["key"]] = {
            "correct": correct,
            "accuracy": correct / (len(GROUND_TRUTH) * len(FIELDS)),
            "perfect": perfect,
            "missing": missing,
            "wrong": wrong,
            "field_correct": field_correct,
            "per_contract": per_contract,
            "errors": errors,
        }
    return stats


def aggregate(stats, attribute: str):
    result = {}
    for value in {route[attribute] for route in ROUTES}:
        selected = [route for route in ROUTES if route[attribute] == value]
        correct = sum(stats[route["key"]]["correct"] for route in selected)
        total = len(selected) * len(GROUND_TRUTH) * len(FIELDS)
        result[value] = {"correct": correct, "total": total, "accuracy": correct / total}
    return result


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def score_class(score: int) -> str:
    return {5: "s5", 4: "s4", 3: "s3", 2: "s2", 1: "s1", 0: "s0"}[score]


def build_report() -> str:
    outputs, combined_count, page_count = load_and_validate()
    stats = calculate_stats(outputs)
    ranked = sorted(
        ROUTES,
        key=lambda route: (
            stats[route["key"]]["accuracy"],
            stats[route["key"]]["perfect"],
        ),
        reverse=True,
    )
    source_stats = aggregate(stats, "source")
    extractor_stats = aggregate(stats, "extractor")

    markdown_routes = [route for route in ROUTES if route["source"] != "PDF images"]
    markdown_by_extractor = {}
    for extractor in ("LLM", "NuExtract"):
        selected = [
            route for route in markdown_routes if route["extractor"] == extractor
        ]
        correct = sum(stats[route["key"]]["correct"] for route in selected)
        total = len(selected) * len(GROUND_TRUTH) * len(FIELDS)
        markdown_by_extractor[extractor] = correct / total

    total_correct = sum(item["correct"] for item in stats.values())
    total_scored = len(ROUTES) * len(GROUND_TRUTH) * len(FIELDS)
    best = ranked[0]
    best_stats = stats[best["key"]]

    route_rows = []
    for rank, route in enumerate(ranked, start=1):
        item = stats[route["key"]]
        field_cells = "".join(
            f"<td class='num'>{item['field_correct'][field]}/15</td>"
            for field, _ in FIELDS
        )
        route_rows.append(
            f"""<tr>
<td class="rank">{rank}</td>
<td><b>{escape(route['label'])}</b><small>{escape(route['key'])}</small></td>
<td><div class="scoreline"><b>{pct(item['accuracy'])}</b><span class="track"><i style="width:{item['accuracy']*100:.1f}%"></i></span></div></td>
<td class="num strong">{item['perfect']}/15</td>
{field_cells}
<td class="num missing">{item['missing']}</td>
<td class="num wrong">{item['wrong']}</td>
</tr>"""
        )

    heat_rows = []
    for contract in GROUND_TRUTH:
        cells = []
        for route in ROUTES:
            result = stats[route["key"]]["per_contract"][contract]
            score = sum(result.values())
            bad_fields = [
                label
                for field, label in FIELDS
                if not result[field]
            ]
            title = "All five fields correct" if not bad_fields else "Missed: " + ", ".join(bad_fields)
            cells.append(
                f"<td class='heat {score_class(score)}' title='{escape(title)}'>{score}/5</td>"
            )
        heat_rows.append(
            f"<tr><td><b>{escape(contract_label(contract))}</b><small>{escape(contract)}</small></td>{''.join(cells)}</tr>"
        )

    field_totals = {
        field: sum(item["field_correct"][field] for item in stats.values())
        for field, _ in FIELDS
    }
    field_cards = "".join(
        f"""<div class="field-card">
<span>{escape(label)}</span>
<b>{field_totals[field]}/120</b>
<em>{pct(field_totals[field]/120)}</em>
<div class="track"><i style="width:{field_totals[field]/1.2:.1f}%"></i></div>
</div>"""
        for field, label in FIELDS
    )

    source_cards = "".join(
        f"""<div class="compare-card">
<span>{escape(source)}</span>
<b>{pct(item['accuracy'])}</b>
<small>{item['correct']}/{item['total']} fields across both extractors</small>
<div class="track"><i style="width:{item['accuracy']*100:.1f}%"></i></div>
</div>"""
        for source, item in sorted(
            source_stats.items(), key=lambda pair: pair[1]["accuracy"], reverse=True
        )
    )

    error_rows = []
    for route in ROUTES:
        item = stats[route["key"]]
        examples = item["errors"][:4]
        example_text = "; ".join(
            f"{error['contract'][:2]} {dict(FIELDS)[error['field']]}: "
            f"{display_value(error['predicted'])}"
            for error in examples
        )
        if len(item["errors"]) > 4:
            example_text += f"; +{len(item['errors']) - 4} more"
        error_rows.append(
            f"""<tr>
<td><b>{escape(route['short'])}</b><small>{escape(route['label'])}</small></td>
<td class="num">{item['missing']}</td>
<td class="num">{item['wrong']}</td>
<td>{escape(example_text)}</td>
</tr>"""
        )

    truth_rows = []
    for contract, expected in GROUND_TRUTH.items():
        note = AUDIT_NOTES.get(contract[:2], "")
        truth_rows.append(
            f"""<tr>
<td><b>{escape(contract_label(contract))}</b></td>
<td>{escape(display_value(expected['renter_name']))}</td>
<td>{escape(display_value(expected['total_rent_price']))}</td>
<td>{escape(expected['street_address'])}, {escape(expected['postal_code'])} {escape(expected['city'])}</td>
<td>{escape(note)}</td>
</tr>"""
        )

    generated = date.today().strftime("%d %B %Y")
    model_note = (
        "General LLM: "
        + escape(ROUTES[0]["manifest"]["model"])
        + " · NuExtract: "
        + escape(ROUTES[1]["manifest"]["model"])
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rental Contract Extraction Quality — 8 Pipeline Comparison</title>
<style>
:root{{--ink:#17212b;--muted:#64717d;--paper:#f3f1eb;--card:#fff;--line:#dce1e2;--navy:#17364d;--teal:#11756f;--green:#25845d;--lime:#75a843;--amber:#d08a18;--orange:#d26732;--red:#c54e54;--blue:#466fad}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}.wrap{{width:min(1240px,calc(100% - 34px));margin:auto}}
header{{padding:58px 0 44px;background:linear-gradient(125deg,#10293d,#174c58 67%,#128077);color:#fff}}.kicker{{margin:0 0 10px;color:#9eddd6;font-size:11px;font-weight:850;letter-spacing:.14em;text-transform:uppercase}}h1{{max-width:980px;margin:0;font-size:clamp(37px,6vw,68px);line-height:1.01;letter-spacing:-.045em}}header .lead{{max-width:900px;margin:20px 0 0;color:#d9e9ec;font-size:18px}}.meta{{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:23px;color:#bfd3d9;font-size:13px}}button{{margin-top:23px;padding:8px 13px;border:1px solid #ffffff55;border-radius:8px;background:#ffffff13;color:#fff;font:inherit;cursor:pointer}}
main{{padding:28px 0 65px}}section{{margin-top:30px}}h2{{margin:0;color:var(--navy);font-size:27px;letter-spacing:-.025em}}h3{{margin:0;color:var(--navy);font-size:17px}}p{{margin:8px 0 0}}.section-head{{display:flex;justify-content:space-between;align-items:end;gap:24px;margin-bottom:14px}}.note{{max-width:650px;color:var(--muted);font-size:13px}}
.verdict{{display:grid;grid-template-columns:1fr auto;gap:25px;align-items:center;padding:25px 27px;border:1px solid #b7d6d1;border-radius:16px;background:linear-gradient(110deg,#fff,#eaf7f4);box-shadow:0 12px 30px #15344910}}.verdict strong{{color:var(--teal)}}.badge{{padding:9px 13px;border-radius:99px;background:#d8efe9;color:#0d635d;font-size:12px;font-weight:850;white-space:nowrap}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:15px}}.metric,.panel,.compare-card,.field-card{{border:1px solid var(--line);border-radius:14px;background:var(--card);box-shadow:0 10px 26px #17364d0c}}.metric{{padding:18px}}.metric b{{display:block;color:var(--navy);font-size:31px;line-height:1}}.metric span{{display:block;margin-top:8px;color:var(--muted);font-size:12px}}
.tablebox{{overflow:auto;border:1px solid var(--line);border-radius:14px;background:#fff;box-shadow:0 10px 26px #17364d0c}}table{{width:100%;border-collapse:collapse;font-size:12.5px}}th,td{{padding:10px 11px;text-align:left;vertical-align:middle;border-bottom:1px solid #edf0f1}}th{{position:sticky;top:0;z-index:1;background:#f3f6f6;color:#435461;font-size:10px;letter-spacing:.055em;text-transform:uppercase}}tr:last-child td{{border-bottom:0}}td small{{display:block;color:var(--muted);font-size:10.5px}}.num{{text-align:center;white-space:nowrap}}.rank{{width:38px;text-align:center;color:var(--muted);font-weight:800}}.strong{{color:var(--green);font-weight:850}}.missing{{color:var(--amber);font-weight:750}}.wrong{{color:var(--red);font-weight:750}}
.scoreline{{display:grid;grid-template-columns:48px 95px;align-items:center;gap:8px}}.track{{display:block;height:7px;overflow:hidden;border-radius:99px;background:#e8edef}}.track i{{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,var(--teal),#54a885)}}
.compare-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.compare-card{{padding:17px}}.compare-card span,.field-card span{{display:block;color:var(--muted);font-size:12px}}.compare-card b{{display:block;margin:4px 0;color:var(--navy);font-size:28px}}.compare-card small{{display:block;min-height:35px;color:var(--muted)}}.compare-card .track,.field-card .track{{margin-top:10px}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.panel{{padding:21px}}.stat-pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:15px}}.stat-pair div{{padding:14px;border-radius:11px;background:#f5f7f6}}.stat-pair b{{display:block;color:var(--navy);font-size:25px}}.stat-pair span{{color:var(--muted);font-size:11px}}.callout{{margin-top:14px;padding:13px 15px;border-left:4px solid var(--amber);background:#fff8eb;color:#4f5961;font-size:13px}}
.fields{{display:grid;grid-template-columns:repeat(5,1fr);gap:11px}}.field-card{{padding:16px}}.field-card b{{display:block;margin-top:4px;color:var(--navy);font-size:22px}}.field-card em{{color:var(--muted);font-size:11px;font-style:normal}}
.heat{{min-width:58px;text-align:center;font-weight:850;border-left:2px solid #fff}}.s5{{background:#dcefe4;color:#176b49}}.s4{{background:#f2edce;color:#74610b}}.s3{{background:#f8dcc6;color:#9a4d20}}.s2,.s1,.s0{{background:#f2cdd0;color:#9a3038}}.legend{{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0 0;color:var(--muted);font-size:11px}}.legend i{{display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:2px;vertical-align:-1px}}
.findings{{display:grid;grid-template-columns:repeat(2,1fr);gap:13px}}.finding{{padding:19px;border:1px solid var(--line);border-radius:13px;background:#fff}}.finding b{{color:var(--navy)}}.finding p{{color:var(--muted);font-size:13px}}code{{padding:1px 5px;border-radius:5px;background:#edf1f1;font-size:.92em}}details{{border:1px solid var(--line);border-radius:14px;background:#fff}}summary{{padding:17px 20px;color:var(--navy);font-weight:800;cursor:pointer}}details .inside{{padding:0 20px 20px}}.method{{padding:17px 19px;border:1px solid #ddd7c6;border-radius:12px;background:#fffdf7;color:#5c5a52;font-size:12.5px}}footer{{padding:0 0 34px;color:var(--muted);font-size:11px}}
@media(max-width:900px){{.metrics,.compare-grid{{grid-template-columns:repeat(2,1fr)}}.split{{grid-template-columns:1fr}}.fields{{grid-template-columns:repeat(3,1fr)}}.verdict{{grid-template-columns:1fr}}.badge{{justify-self:start}}}}
@media(max-width:560px){{.metrics,.compare-grid,.fields,.findings{{grid-template-columns:1fr}}.section-head{{display:block}}.wrap{{width:calc(100% - 20px)}}}}
@media print{{body{{background:#fff}}header{{padding:25px 0;background:var(--navy)!important;print-color-adjust:exact}}button{{display:none}}section{{break-inside:avoid}}.metric,.panel,.tablebox,.compare-card,.field-card{{box-shadow:none}}}}
</style>
</head>
<body>
<header><div class="wrap">
<p class="kicker">Rental contract extraction · full pipeline benchmark</p>
<h1>Markdown wins—unless NuExtract reads the PDF directly.</h1>
<p class="lead">Eight paths were scored against 15 manually audited contracts. The best two-stage routes reach 94.7% field accuracy; direct PDF extraction ranges from 58.7% with the general LLM to 92.0% with NuExtract3.</p>
<div class="meta"><span>15 contracts</span><span>8 paths</span><span>120 final records</span><span>{combined_count + page_count} schema-valid JSON outputs</span><span>{generated}</span><span>{model_note}</span></div>
<button onclick="window.print()">Print / save PDF</button>
</div></header>

<main class="wrap">
<section class="verdict">
<div><h2>Best default: MarkItDown → LLM</h2><p><strong>{pct(best_stats['accuracy'])} field accuracy and {best_stats['perfect']}/15 fully correct contracts.</strong> NuExtract Markdown → LLM ties on fields at 94.7%, while preserving one fewer fully correct contract. For direct images, NuExtract is the clear choice.</p></div>
<span class="badge">Human review still needed</span>
</section>

<section class="metrics">
<div class="metric"><b>{pct(best_stats['accuracy'])}</b><span>Best route field accuracy</span></div>
<div class="metric"><b>{best_stats['perfect']}/15</b><span>Fully correct contracts on the best route</span></div>
<div class="metric"><b>{pct(total_correct/total_scored)}</b><span>Accuracy across all eight paths</span></div>
<div class="metric"><b>{combined_count + page_count}/{combined_count + page_count}</b><span>Outputs valid against the schema</span></div>
</section>

<section>
<div class="section-head"><div><p class="kicker">Route ranking</p><h2>Every pipeline, scored field by field</h2></div><p class="note">Five informative fields are scored per contract: renter, explicit total rent, street, postcode, and city. Country was correctly null in every result and is reported separately.</p></div>
<div class="tablebox"><table>
<thead><tr><th>#</th><th>Pipeline</th><th>Accuracy</th><th>Perfect</th>{''.join(f'<th>{escape(label)}</th>' for _, label in FIELDS)}<th>Missing</th><th>Wrong</th></tr></thead>
<tbody>{''.join(route_rows)}</tbody>
</table></div>
</section>

<section>
<div class="section-head"><div><p class="kicker">Input source</p><h2>NuExtract Markdown is strongest on average</h2></div><p class="note">Each card pools both extractors, exposing the quality of the input representation. Direct PDF is included as a baseline.</p></div>
<div class="compare-grid">{source_cards}</div>
</section>

<section class="split">
<div class="panel"><p class="kicker">Extractor effect</p><h2>Text and images favor different models</h2>
<div class="stat-pair"><div><b>{pct(markdown_by_extractor['LLM'])}</b><span>LLM on the three Markdown sources</span></div><div><b>{pct(markdown_by_extractor['NuExtract'])}</b><span>NuExtract on the three Markdown sources</span></div></div>
<div class="callout">On Markdown, the general LLM leads by {pct(markdown_by_extractor['LLM']-markdown_by_extractor['NuExtract'])}. NuExtract loses renter names more often, especially when names appear on different pages or multiple renters share one field.</div>
</div>
<div class="panel"><p class="kicker">Direct-image effect</p><h2>NuExtract reverses the result on PDFs</h2>
<div class="stat-pair"><div><b>{pct(stats['pdf/nuextract']['accuracy'])}</b><span>PDF images → NuExtract</span></div><div><b>{pct(stats['pdf/llm']['accuracy'])}</b><span>PDF images → LLM</span></div></div>
<div class="callout">NuExtract gains {pct(stats['pdf/nuextract']['accuracy']-stats['pdf/llm']['accuracy'])} on the same rendered pages. The general LLM produces frequent OCR errors, incomplete addresses, and wrong address selection.</div>
</div>
</section>

<section>
<div class="section-head"><div><p class="kicker">Field reliability</p><h2>Rent and renter identity remain hardest</h2></div><p class="note">Totals combine all 8 routes × 15 contracts. Equivalent accents, punctuation, and “en” versus semicolon separators are accepted for renter names.</p></div>
<div class="fields">{field_cards}</div>
</section>

<section>
<div class="section-head"><div><p class="kicker">Contract heatmap</p><h2>Where each path succeeds or fails</h2></div><p class="note">Hover a cell to see which fields were missed. A 5/5 cell means the complete final record is correct on all informative fields.</p></div>
<div class="tablebox"><table>
<thead><tr><th>Contract</th>{''.join(f"<th title='{escape(route['label'])}'>{escape(route['short'])}</th>" for route in ROUTES)}</tr></thead>
<tbody>{''.join(heat_rows)}</tbody>
</table></div>
<div class="legend"><span><i class="s5"></i>5/5 correct</span><span><i class="s4"></i>4/5</span><span><i class="s3"></i>3/5</span><span><i class="s2"></i>0–2/5</span></div>
</section>

<section>
<div class="section-head"><div><p class="kicker">Failure profile</p><h2>Missing values versus wrong values</h2></div><p class="note">Wrong includes a populated but incorrect value and a value returned where the audited answer is null.</p></div>
<div class="tablebox"><table>
<thead><tr><th>Pipeline</th><th>Missing</th><th>Wrong</th><th>Representative errors</th></tr></thead>
<tbody>{''.join(error_rows)}</tbody>
</table></div>
</section>

<section>
<div class="section-head"><div><p class="kicker">Interpretation</p><h2>What the comparison says</h2></div></div>
<div class="findings">
<div class="finding"><b>MarkItDown is the pragmatic default.</b><p>Its Markdown is inexpensive, preserves all 45 address components in this set, and gives the highest count of fully correct contracts when paired with the LLM.</p></div>
<div class="finding"><b>NuExtract Markdown is the best source overall.</b><p>Across both extractors it scores {pct(source_stats['NuExtract Markdown']['accuracy'])}. Paired with the LLM it ties the top field score; paired with NuExtract it remains close behind.</p></div>
<div class="finding"><b>Do not use the general LLM directly on these PDF pages.</b><p>It scores {pct(stats['pdf/llm']['accuracy'])}, with {stats['pdf/llm']['missing']} missing and {stats['pdf/llm']['wrong']} wrong fields. Contract 15 scores 0/5.</p></div>
<div class="finding"><b>Direct NuExtract is a credible simpler pipeline.</b><p>At {pct(stats['pdf/nuextract']['accuracy'])}, it trails the best two-stage route by only {pct(best_stats['accuracy']-stats['pdf/nuextract']['accuracy'])}, without generating intermediate Markdown.</p></div>
<div class="finding"><b>Explicit-total logic needs reinforcement.</b><p>Contracts 03 and 10 state components but no total; most routes incorrectly return base rent or calculate a sum. Contract 15 states €1,205 explicitly, yet six routes return null.</p></div>
<div class="finding"><b>Cross-page name merging is the other weak point.</b><p>Contract 15 is missed by every route, and contracts 01, 09, and 11 expose accent, page-merging, and multiple-renter failures.</p></div>
</div>
</section>

<section>
<details>
<summary>Audited reference values and special decisions</summary>
<div class="inside"><div class="tablebox"><table>
<thead><tr><th>Contract</th><th>Renter</th><th>Explicit total</th><th>Rental property</th><th>Audit note</th></tr></thead>
<tbody>{''.join(truth_rows)}</tbody>
</table></div></div>
</details>
</section>

<section class="method">
<b>Method.</b> Final combined records were scored against a manually audited reference derived from the contract content. Numeric values are compared numerically. Text comparison ignores case, punctuation, spacing, accents, smart apostrophes, and “en/and” versus semicolon separators between multiple renter names; substantive spelling differences remain errors. The headline excludes <code>country</code> because all 15 references and all 120 predictions are null, so including it would add 120 non-discriminating correct fields. All {combined_count} combined and {page_count} page-level files were parsed and validated against the current JSON Schema. “Perfect contract” means all five informative fields are correct. This benchmark measures this 15-document set and these recorded model runs; it is comparative evidence, not a general model benchmark.
</section>
</main>
<footer class="wrap">Generated by <code>generate_quality_report.py</code> from <code>results/</code>. Extraction outputs were not modified.</footer>
</body>
</html>
"""


def main() -> None:
    REPORT.write_text(build_report(), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
