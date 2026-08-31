"""OCR quality benchmark: production Docling+RapidOCR stack vs Baidu Unlimited-OCR.

Motivation: Baidu open-sourced Unlimited-OCR (3B, MIT, R-SWA long-context)
claiming book-scale parsing. This harness measures whether it actually beats
the pipeline's current scanned-PDF path (Docling + RapidOCR, full-page mode)
on real corpus pages, and whether it merits replacing/augmenting that path.

What it does per sample:
  1. Render the sample page at --dpi (default 300, the vendor-recommended
     scan resolution) from the source PDF.
  2. Run each selected backend over the SAME page:
       docling  - a 1-page PDF slice through the production converter,
                  built from the live config.toml OCR settings, rendered to
                  markdown exactly like ingestion does. This is the baseline
                  a scanned PDF hits today (HybridPdfParser -> DoclingPdfParser).
       unlimited- OCR via the local Ollama host with the vendor prompt
                  ("document parsing."), detection markers stripped.
       qwen25vl - the current LOCAL vision fallback (qwen2.5vl:3b) with a
                  verbatim-transcription prompt; reference point only, it is
                  not the primary OCR path.
  3. Score against the hand-verified ground-truth transcription with CER and
     WER (strict + alphanumeric-only variants), and record wall time.

Usage:
    python scripts/eval_ocr.py --samples eval/ocr/samples.json \
        --backends docling,unlimited --out eval/ocr/results

samples.json entries: {"id": str, "pdf": path, "page": 0-based int, "gt": path
to ground-truth .txt}. Ground truth is transcribed by reading the rendered
page images (kept under <out>/pages/ for audit).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests

from src.ingestion_classes.unlimited_ocr_pdf_parser import strip_detection_markers

UNLIMITED_MODEL = "frob/unlimited-ocr"
QWEN_VL_MODEL = "qwen2.5vl:3b"
UNLIMITED_PROMPT = "document parsing."
QWEN_VL_PROMPT = (
    "Transcribe ALL text on this scanned page verbatim, preserving the "
    "original reading order. Output only the transcription."
)


# ---------------------------------------------------------------- rendering

def render_page(pdf_path: str, page_no: int, dpi: int, png_path: Path) -> None:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_path)
    if page_no >= len(pdf):
        raise IndexError(f"{pdf_path} has {len(pdf)} pages, sample wants #{page_no}")
    bitmap = pdf[page_no].render(scale=dpi / 72)
    bitmap.to_pil().save(png_path)


def slice_single_page_pdf(pdf_path: str, page_no: int, out_path: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    writer.add_page(reader.pages[page_no])
    with open(out_path, "wb") as fh:
        writer.write(fh)


# ----------------------------------------------------------------- backends

def run_docling(page_pdf: Path, dpi: int) -> str:
    from src.config import load_config
    from src.ingestion import _build_docling_converter

    cfg = load_config().ingestion
    converter = _build_docling_converter(
        accelerator=cfg.accelerator,
        num_threads=cfg.num_threads,
        ocr_enabled=True,
        ocr_backend=cfg.ocr_backend,
        ocr_langs=cfg.ocr_langs,
        ocr_force_full_page=cfg.ocr_force_full_page,
        ocr_bitmap_area_threshold=cfg.ocr_bitmap_area_threshold,
        rapidocr_backend=cfg.rapidocr_backend,
        tesseract_cmd=cfg.tesseract_cmd,
        tesseract_data_path=cfg.tesseract_data_path,
        tesseract_psm=cfg.tesseract_psm,
    )
    result = converter.convert(str(page_pdf))
    return result.document.export_to_markdown()


def _ollama_chat(model: str, prompt: str, image_png: Path, num_ctx: int) -> str:
    image_b64 = base64.b64encode(image_png.read_bytes()).decode()
    response = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "options": {"num_ctx": num_ctx},
        },
        timeout=1800,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def run_unlimited(page_png: Path) -> str:
    return strip_detection_markers(_ollama_chat(UNLIMITED_MODEL, UNLIMITED_PROMPT, page_png, num_ctx=16384))


def run_qwen25vl(page_png: Path) -> str:
    return _ollama_chat(QWEN_VL_MODEL, QWEN_VL_PROMPT, page_png, num_ctx=8192)


BACKENDS = {
    "docling": lambda page_pdf, page_png, dpi: run_docling(page_pdf, dpi),
    "unlimited": lambda page_pdf, page_png, dpi: run_unlimited(page_png),
    "qwen25vl": lambda page_pdf, page_png, dpi: run_qwen25vl(page_png),
}


# ------------------------------------------------------------------ metrics

def _normalize_strict(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    # Undo print hyphenation so "trans-\nmitted" == "transmitted".
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", _normalize_strict(text))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = _normalize_alnum(reference), _normalize_alnum(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref = _normalize_strict(reference).split()
    hyp = _normalize_strict(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(" ".join(ref), " ".join(hyp)) / len(" ".join(ref))


# --------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="eval/ocr/samples.json")
    parser.add_argument("--backends", default="docling,unlimited",
                        help="comma list from: " + ", ".join(BACKENDS))
    parser.add_argument("--out", default="eval/ocr/results")
    parser.add_argument("--dpi", type=int, default=300,
                        help="render dpi for vision backends (docling parses the "
                             "PDF itself and is unaffected)")
    parser.add_argument("--unlimited-dpi", type=int, default=None,
                        help="override dpi for the unlimited backend only; the "
                             "Ollama build caps effective resolution, so dense "
                             "pages need ~110-150 dpi while the vendor stack "
                             "uses 300")
    parser.add_argument("--only", default="", help="comma list of sample ids to (re)run")
    args = parser.parse_args()

    samples = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        samples = [s for s in samples if s["id"] in wanted]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for backend in backends:
        if backend not in BACKENDS:
            raise SystemExit(f"Unknown backend '{backend}'. Use: {', '.join(BACKENDS)}")

    out_dir = Path(args.out)
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for sample in samples:
        sample_id = sample["id"]
        gt_path = ROOT / sample["gt"]
        ground_truth = gt_path.read_text(encoding="utf-8")

        # docling consumes the 1-page PDF slice; vision backends consume the
        # rendered PNG at their own dpi.
        wanted_pngs: dict[str, int] = {}
        for backend in backends:
            wanted_pngs[backend] = (
                args.unlimited_dpi if (backend == "unlimited" and args.unlimited_dpi) else args.dpi
            )
        pngs: dict[str, Path] = {}
        page_pdf = pages_dir / f"{sample_id}.1p.pdf"
        if not page_pdf.exists():
            slice_single_page_pdf(sample["pdf"], sample["page"], page_pdf)
        for backend, dpi in wanted_pngs.items():
            png = pages_dir / f"{sample_id}.{dpi}.png"
            if not png.exists():
                print(f"[render] {sample_id} @ {dpi} dpi")
                render_page(sample["pdf"], sample["page"], dpi, png)
            pngs[backend] = png

        for backend in backends:
            result_path = out_dir / backend / f"{sample_id}.md"
            if result_path.exists():
                text = result_path.read_text(encoding="utf-8")
                elapsed = -1.0
                print(f"[cache] {backend}/{sample_id}")
            else:
                started = time.perf_counter()
                text = BACKENDS[backend](page_pdf, pngs[backend], args.dpi)
                elapsed = time.perf_counter() - started
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(text, encoding="utf-8")
                print(f"[run ] {backend}/{sample_id}: {elapsed:.1f}s")

            rows.append({
                "sample": sample_id,
                "backend": backend,
                "cer": round(cer(ground_truth, text), 4),
                "wer": round(wer(ground_truth, text), 4),
                "seconds": round(elapsed, 1),
                "output_chars": len(text.strip()),
                "gt_chars": len(_normalize_alnum(ground_truth)),
            })

    report = out_dir / "report.md"
    write_report(report, rows, args.dpi)
    print(f"\nreport: {report}")


def write_report(report: Path, rows: list[dict], dpi: int) -> None:
    backends = sorted({r["backend"] for r in rows})
    lines = [
        f"# OCR eval ({dpi} dpi)",
        "",
        "CER/WER against hand-verified transcriptions (lower is better).",
        "",
        "| sample | " + " | ".join(f"{b} CER | {b} WER | {b} s" for b in backends) + " |",
        "|---" * (len(backends) * 3 + 1) + "|",
    ]
    samples = sorted({r["sample"] for r in rows})
    for sample_id in samples:
        cells = []
        for backend in backends:
            match = next((r for r in rows if r["sample"] == sample_id and r["backend"] == backend), None)
            cells.extend([
                f"{match['cer']:.3f}" if match else "-",
                f"{match['wer']:.3f}" if match else "-",
                f"{match['seconds']:.0f}" if match and match["seconds"] >= 0 else "-",
            ])
        lines.append(f"| {sample_id} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Means")
    lines.append("")
    lines.append("| backend | mean CER | mean WER | total s |")
    lines.append("|---|---|---|---|")
    for backend in backends:
        subset = [r for r in rows if r["backend"] == backend]
        lines.append(
            f"| {backend} | {sum(r['cer'] for r in subset) / len(subset):.4f} "
            f"| {sum(r['wer'] for r in subset) / len(subset):.4f} "
            f"| {sum(max(r['seconds'], 0) for r in subset):.0f} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
