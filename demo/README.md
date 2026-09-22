# Local LLM extraction demo

This directory contains a single-process PDF extraction path:

```text
PDF -> PyMuPDF4LLM -> Markdown + source map -> local LLM -> JSON -> highlighted PDF
```

PyMuPDF4LLM runs locally with its CPU layout model enabled and OCR disabled.
The LLM is called through an OpenAI-compatible local endpoint. Each extracted
field contains both its structured value and a short verbatim
excerpt from the generated Markdown. The conversion also saves page and layout
box coordinates with character offsets into the Markdown, providing the basis
for mapping evidence back to the PDF. The initial highlighting implementation
highlights each complete layout box that overlaps an extracted context.

## Run

Create a `.env` file in the repository root (beside `pyproject.toml`):

```dotenv
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=gemma
LLM_API_KEY=local
```

You can copy `.env.example` as a starting point. Values already set in the shell
take precedence over `.env`. The API key is only a placeholder for local servers
that do not authenticate.

The Pydantic extraction model remains the source of truth. Its JSON Schema is
sent to vLLM through `structured_outputs`, and the returned JSON is validated
back into that same Pydantic model.

If the local model returns the right evidence text but omits Markdown formatting
such as table pipes or bold markers, the demo conservatively restores the exact
Markdown slice when there is one unique formatting-insensitive match. Ambiguous
or paraphrased evidence is still rejected.

Then run one contract from the repository root:

```powershell
uv run python demo/run_demo.py contracts/01_ROZ_2025_ingevuld.pdf
```

Or point it at a directory to process every PDF directly inside it, in filename
order:

```powershell
uv run python demo/run_demo.py contracts
```

Each PDF gets its own directory under `demo/output`. Directory scanning is not
recursive.

On Windows, forward slashes and backslashes both work for these paths. Do not
escape underscores. The module form also remains available:

```powershell
uv run python -m demo.run_demo .\contracts\01_ROZ_2025_ingevuld.pdf
```

The default output is:

```text
demo/output/01_ROZ_2025_ingevuld/
  document.md
  extraction.json
  highlighted.pdf
  highlights.json
  source-map.json
  run.json
```

To add highlights to an output directory that was generated previously, no
model request is needed:

```powershell
uv run python demo/highlight_pdf.py demo/output/01_ROZ_2025_ingevuld
```

`highlights.json` records the Markdown matches, selected PDF boxes, and any
fields that could not be matched. If multiple fields select the same box, that
box is highlighted only once.

## View results

Open a local browser interface for one output directory:

```powershell
uv run python demo/view_results.py demo/output/01_ROZ_2025_ingevuld
```

The left side shows rendered pages from `highlighted.pdf`, and the right side
shows the extracted values. Clicking a field scrolls the PDF pane to its first
matched source-map box. Stop the local server with `Ctrl+C`.

Override the model or output directory when needed:

```powershell
uv run python demo/run_demo.py contract.pdf `
  --model gemma `
  --base-url http://127.0.0.1:8000/v1 `
  --output-dir path/to/output
```

OCR is deliberately disabled in this initial version so clean, digitally
generated PDFs remain fast and deterministic. The demo stops with a clear error
when a PDF contains no extractable text.

## Test

The unit tests mock the local LLM call and do not need a running model server:

```powershell
uv run python -m unittest demo.test_demo -v
```
