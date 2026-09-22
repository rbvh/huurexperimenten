# Rental contract extraction

This project has two independent stages:

1. Convert each PDF page to Markdown with an LLM, MarkItDown, or NuExtract3.
2. Extract contract JSON from any of those Markdown variants, or directly from
   PDF page images, with either a general LLM or NuExtract3.

## Directory layout

The directory names record both stages of the pipeline:

```text
contracts/                              # original PDFs
markdown/
  llm/                                  # PDF -> Markdown with general VLM
  markitdown/                           # PDF -> Markdown with MarkItDown
  nuextract/                            # PDF -> Markdown with NuExtract3
results/
  markdown_llm/
    llm/                                # LLM Markdown -> JSON with general LLM
    nuextract/                          # LLM Markdown -> JSON with NuExtract3
  markdown_markitdown/
    llm/
    nuextract/
  markdown_nuextract/
    llm/
    nuextract/
  pdf/
    llm/                                # PDF images -> JSON with general LLM
    nuextract/                          # PDF images -> JSON with NuExtract3
```

Each Markdown producer writes
`markdown/<producer>/<contract>/page-NNN.md`. Each extractor writes combined
JSON to `results/<source>/<extractor>/<contract>.json` and page JSON to
`results/<source>/<extractor>/<contract>/pages/page-NNN.json`.

Each generated root also contains `_run.json`, which records the producer or
extractor, exact served model, input directory, and endpoint. This avoids losing
provenance when `--model`, `--base-url`, or `--output-dir` is overridden.

Existing outputs created with the earlier flat layout have been moved into the
matching locations above.

## Installation

```sh
pip install openai pymupdf jsonschema 'markitdown[pdf]'
```

Both scripts read PDFs from `contracts/` by default. Use `--input-dir PATH`
and `--output-dir PATH` to override their standard locations.

## Produce Markdown

```sh
python pdf_to_markdown.py --method llm
python pdf_to_markdown.py --method markitdown
python pdf_to_markdown.py --method nuextract --model nuextract3
```

The `llm` and `nuextract` methods render PDF pages to images before sending
them to their model. MarkItDown extracts PDF text locally and does not OCR
image-only pages.

The general LLM defaults to `LLM_MODEL`, or `gemma`. NuExtract defaults to
`NUEXTRACT_MODEL`, or `nuextract3`. Both use `LLM_BASE_URL`, defaulting to
`http://127.0.0.1:8000/v1`.

## Extract JSON

Select the input with `--source`:

| Source | Input |
| --- | --- |
| `markdown_llm` | `markdown/llm/` |
| `markdown_markitdown` | `markdown/markitdown/` |
| `markdown_nuextract` | `markdown/nuextract/` |
| `pdf` | rendered pages from `contracts/` |

Select the extraction request format with `--backend llm` or
`--backend nuextract`. Run the four sources while the general LLM server is
active:

```sh
for source in markdown_llm markdown_markitdown markdown_nuextract pdf; do
  python extract_contract_data.py --source "$source" --backend llm
done
```

After starting the NuExtract3 server, run the same four sources with its request
format:

```sh
for source in markdown_llm markdown_markitdown markdown_nuextract pdf; do
  python extract_contract_data.py --source "$source" --backend nuextract
done
```

A single run looks like:

```sh
python extract_contract_data.py \
  --source markdown_markitdown \
  --backend nuextract \
  --model nuextract3
```

That example reads `markdown/markitdown/` and writes
`results/markdown_markitdown/nuextract/`.

The output schema contains only:

```json
{
  "renter_name": "Example Renter",
  "total_rent_price": 1250.0,
  "property_address": {
    "street_address": "Example Street 10",
    "postal_code": "1234 AB",
    "city": "Amsterdam",
    "country": null
  }
}
```

The price is the explicitly stated total monthly payment, including service
costs and utilities when included in that stated total. Deposits and one-time
payments are excluded, and totals are not calculated from components. Missing
values are `null`. Conflicting page values produce a warning and a `null`
combined value; the page JSON retains each extracted value.

Invalid JSON, schema mismatches, and truncated replies stop the run and save the
raw reply as `page-NNN.invalid.txt`.

## Run NuExtract3 with vLLM

NuExtract3 accepts both page images and text such as Markdown. It uses
`mode="markdown"` for image-to-Markdown and a typed extraction template for
JSON. See the
[NuExtract3 model card](https://huggingface.co/numind/NuExtract3).

Stop another model server first if it uses the same GPU and port, then launch:

```sh
source /workspace/gemma-cu129/bin/activate
export HF_HOME=/workspace/huggingface

vllm serve numind/NuExtract3 \
  --served-model-name nuextract3 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --chat-template-content-format openai \
  --generation-config vllm \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":1,"video":0}'
```

Check that it is ready:

```sh
curl --fail http://127.0.0.1:8000/v1/models
```

Then produce NuExtract Markdown, extract directly from images, or extract from
any Markdown source:

```sh
python pdf_to_markdown.py --method nuextract --model nuextract3
python extract_contract_data.py --source pdf --backend nuextract --model nuextract3
python extract_contract_data.py --source markdown_llm --backend nuextract --model nuextract3
```

The general `llm` backend expects vLLM JSON Schema structured output support.
The Gemma launch example in `vastai_setup.txt` enables the required XGrammar
configuration.
