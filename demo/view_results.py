#!/usr/bin/env python3
"""Open a local web interface for one demo output directory."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse
import webbrowser

import pymupdf


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Contract extraction viewer</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #eef1f5; color: #172033; }
    main { display: grid; grid-template-columns: minmax(0, 3fr) minmax(320px, 2fr); height: 100vh; }
    #pdf { overflow: auto; padding: 24px; background: #525966; }
    #data { overflow: auto; padding: 24px; border-left: 1px solid #ccd2dc; background: #f7f8fa; }
    h1 { margin: 0 0 6px; font-size: 22px; }
    .hint { margin: 0 0 22px; color: #667085; font-size: 14px; }
    .page { position: relative; margin: 0 auto 24px; background: white; box-shadow: 0 3px 16px #1118; }
    .page img { display: block; width: 100%; height: 100%; }
    .page-number { position: absolute; right: 8px; bottom: 6px; padding: 2px 7px; border-radius: 10px; background: #172033cc; color: white; font-size: 11px; }
    .focus-box { position: absolute; pointer-events: none; border: 3px solid #e11d48; background: #fb718522; opacity: 0; }
    .focus-box.active { animation: focus 1.8s ease-out; }
    @keyframes focus { 0%, 45% { opacity: 1; } 100% { opacity: 0; } }
    .field { width: 100%; margin: 0 0 10px; padding: 13px 14px; border: 1px solid #d7dce5; border-radius: 8px; background: white; text-align: left; }
    button.field { cursor: pointer; }
    button.field:hover { border-color: #2563eb; box-shadow: 0 1px 5px #2563eb22; }
    .field.unmatched { opacity: .55; }
    .label { display: block; margin-bottom: 4px; color: #667085; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .value { display: block; font-size: 16px; font-weight: 600; overflow-wrap: anywhere; }
    @media (max-width: 800px) {
      main { grid-template-columns: 1fr; grid-template-rows: 60vh 40vh; }
      #data { border-left: 0; border-top: 1px solid #ccd2dc; }
    }
  </style>
</head>
<body>
  <main>
    <section id="pdf" aria-label="PDF pages"></section>
    <aside id="data">
      <h1>Extracted data</h1>
      <p class="hint">Click a field to jump to its highlighted source region.</p>
      <div id="fields"></div>
    </aside>
  </main>
  <script>
    const pdf = document.querySelector('#pdf');
    const fields = document.querySelector('#fields');

    function flatten(value, path = '', result = []) {
      if (value && typeof value === 'object' && 'value' in value && 'context' in value) {
        result.push({ field: path, value: value.value });
      } else if (value && typeof value === 'object') {
        for (const [key, child] of Object.entries(value)) {
          flatten(child, path ? `${path}.${key}` : key, result);
        }
      }
      return result;
    }

    function jumpTo(box) {
      const page = document.querySelector(`#page-${box.page_number}`);
      const marker = page.querySelector('.focus-box');
      const size = page.dataset;
      const [x0, y0, x1, y1] = box.bbox;
      marker.style.left = `${100 * x0 / size.width}%`;
      marker.style.top = `${100 * y0 / size.height}%`;
      marker.style.width = `${100 * (x1 - x0) / size.width}%`;
      marker.style.height = `${100 * (y1 - y0) / size.height}%`;
      marker.classList.remove('active');
      void marker.offsetWidth;
      marker.classList.add('active');
      marker.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    fetch('/api/data').then(response => response.json()).then(data => {
      for (const page of data.pages) {
        const element = document.createElement('div');
        element.className = 'page';
        element.id = `page-${page.page_number}`;
        element.dataset.width = page.width;
        element.dataset.height = page.height;
        element.style.aspectRatio = `${page.width} / ${page.height}`;
        element.innerHTML = `<img src="/pages/${page.page_number}.png" alt="PDF page ${page.page_number}"><div class="focus-box"></div><span class="page-number">${page.page_number}</span>`;
        pdf.appendChild(element);
      }

      const matches = new Map(data.highlights.matches.map(item => [item.field, item.boxes]));
      for (const item of flatten(data.extraction)) {
        const boxes = matches.get(item.field) || [];
        const element = document.createElement(boxes.length ? 'button' : 'div');
        element.className = `field${boxes.length ? '' : ' unmatched'}`;
        const label = document.createElement('span');
        label.className = 'label';
        label.textContent = item.field.replaceAll('_', ' ').replaceAll('.', ' › ');
        const value = document.createElement('span');
        value.className = 'value';
        value.textContent = item.value ?? 'Not found';
        element.append(label, value);
        if (boxes.length) element.addEventListener('click', () => jumpTo(boxes[0]));
        fields.appendChild(element);
      }
    });
  </script>
</body>
</html>
"""


def load_output(document_dir: Path) -> tuple[dict, list[bytes]]:
    """Load result JSON and render the annotated PDF pages for the browser."""
    document_dir = document_dir.resolve()
    extraction = json.loads(
        (document_dir / "extraction.json").read_text(encoding="utf-8")
    )
    highlights = json.loads(
        (document_dir / "highlights.json").read_text(encoding="utf-8")
    )
    pdf_path = document_dir / "highlighted.pdf"
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Highlighted PDF not found: {pdf_path}")

    pages = []
    images = []
    document = pymupdf.open(pdf_path)
    try:
        for index, page in enumerate(document, start=1):
            pages.append(
                {
                    "page_number": index,
                    "width": page.rect.width,
                    "height": page.rect.height,
                }
            )
            images.append(
                page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), annots=True).tobytes(
                    "png"
                )
            )
    finally:
        document.close()

    return {"extraction": extraction, "highlights": highlights, "pages": pages}, images


def serve(document_dir: Path, port: int, open_browser: bool = True) -> None:
    """Serve the viewer until interrupted."""
    data, images = load_output(document_dir)
    data_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self.send_content(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/data":
                self.send_content(data_bytes, "application/json; charset=utf-8")
            elif path.startswith("/pages/") and path.endswith(".png"):
                try:
                    page_number = int(Path(path).stem)
                    image = images[page_number - 1]
                    if page_number < 1:
                        raise IndexError
                except (ValueError, IndexError):
                    self.send_error(404)
                else:
                    self.send_content(image, "image/png")
            else:
                self.send_error(404)

        def send_content(self, content: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"Viewing {document_dir.resolve()}")
    print(f"Open {url} (press Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Per-document output directory.")
    parser.add_argument(
        "--port", type=int, default=8000, help="Local port (default: 8000)."
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open the browser automatically."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve(args.output_dir, args.port, not args.no_browser)


if __name__ == "__main__":
    main()
