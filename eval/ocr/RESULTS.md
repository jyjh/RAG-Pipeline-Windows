# OCR engine eval: Baidu Unlimited-OCR vs the production stack (2026-08-30)

Question: Baidu open-sourced Unlimited-OCR (3B, MIT, R-SWA "one-shot long-horizon
parsing"), marketed as able to parse entire books. Should the pipeline's
scanned-PDF path switch from Docling+RapidOCR to it?

**Verdict: NO — do not switch, and no re-ingestion.** Via the only route that
runs on this 4 GB GPU (the `frob/unlimited-ocr` Ollama port), it is not better
than the current production stack: it ties Docling on clean typeset text, wins
on tables, but hallucinates or truncates on 2 of 6 real corpus pages.

Secondary finding: the pipeline's own local vision fallback `qwen2.5vl:3b` —
already installed, currently only used when Docling fails entirely — is by far
the most accurate OCR of the three on these pages, at ~3-4 min/page on the
4 GB GPU.

## Setup

- 6 pages hand-picked from real scanned corpus PDFs (`F:/FSAE Readings`,
  ingested via `.ingest_batches/run_20260826_163219/scan_batch_*`), spanning
  dense typeset text, a two-column light table, journal prose + figure,
  typewriter thesis with equations, and typewriter lecture notes with
  handwriting.
- Ground truth: each page transcribed by hand from the 300 dpi render
  (`ground_truth/*.txt`, single annotator).
- Backends: production Docling 2.105 + RapidOCR (onnxruntime, full-page,
  exact `config.toml` settings) over a 1-page PDF slice; `frob/unlimited-ocr`
  (F16) via Ollama 0.32.9 with the vendor prompt `document parsing.` at
  110 dpi (see "resolution finding" below); `qwen2.5vl:3b` via Ollama at
  300 dpi with a verbatim-transcription prompt.
- Metric: character/word error rate against normalized text
  (NFKC, casefold, hyphenation undone, punctuation-insensitive).

## Results (from `results/report.md`)

| sample (page type)          | docling CER | unlimited CER | qwen25vl CER |
|-----------------------------|------------:|--------------:|-------------:|
| apollo_p5 (dense text)      |      0.074  |        0.069  |       0.001  |
| apollo_p21 (2-col table)    |      0.131  |        0.032  |       0.001  |
| lec11_p8 (typewriter+hand)  |      0.160  |       40.982  |       0.011  |
| lec7_p5 (figure+define)     |      0.090  |        0.113  |       0.127  |
| ship_p3 (journal+figure)    |      0.040  |        0.444  |       0.002  |
| thesis_p9 (equations)       |      0.068  |        0.078  |       0.075  |
| **mean**                    | **0.094**   | **6.953**     | **0.036**    |

Per-page wall time (4 GB GTX 1650, f16 + CPU offload): docling ~3-30 s/page,
unlimited ~35-60 s/page (when well-behaved; up to 7 min when looping),
qwen2.5vl ~3-4 min/page.

## Why Unlimited-OCR fails here (Ollama port findings)

1. **Effective resolution cap.** The Ollama/llama.cpp build resizes every page
   to ~1024 px regardless of input dpi. At the vendor-recommended 300 dpi the
   dense text is illegible after downscale and the model openly hallucinates
   (CER 617% on the dense page: plausible-looking prose, wrong content). At
   110 dpi text survives the downscale.
2. **No anti-repetition guard.** The vendor stack generates with
   `ngram_size=35` repetition blocking; Ollama does not. On the
   handwriting-annotated lecture page the output loops (18 KB emitted for a
   ~350-char page, CER 4098%); on the journal page it stopped mid-word.
3. The raw `<|det|>` markers and `[x1, y1, x2, y2]` line prefixes are stripped
   by the integration (`strip_detection_markers`), so that part is fine.

None of this is the model's ceiling — with transformers/vLLM, tiling ("gundam"
mode) and the anti-repetition settings, the vendor benchmarks are plausible.
But that stack needs torch 2.10/CUDA and >8 GB VRAM (BF16 weights alone are
~7 GB), which this machine does not have. On this hardware the honest
comparison is the one above, and it says keep Docling.

## What was changed anyway

- `scanned_ocr_engine` config option (`docling` default | `unlimited_ocr`):
  `UnlimitedOcrPdfParser` renders pages (pypdfium2), OCRs each via local
  Ollama, strips detection markers, and emits the same `## Page N` structure
  as the vision fallback. HybridPdfParser routes scanned PDFs engine →
  Docling → page-image vision. DocumentProcessor validates the engine;
  `run_ingestion`, the web job queue, and `main.py` thread the new options.
- `scripts/eval_ocr.py` + `eval/ocr/samples.json` + `ground_truth/`: rerunnable
  benchmark (per-backend dpi, CER/WER, cached results).
- 11 new tests in `tests/test_ingestion_parsers.py` covering marker stripping,
  per-page markdown, retry/fallback routing, and config threading.

## Follow-up worth considering (not done)

`qwen2.5vl:3b` at CER 0.001-0.011 on typeset pages is 1-2 orders of magnitude
more accurate than production RapidOCR on this corpus, but ~10x slower per
page and ~4 min/page on this GPU (~2.5 days for a 1000-page book). If OCR
quality matters more than ingest throughput for part of the corpus, promoting
the existing vision fallback to a first-class `scanned_ocr_engine = "vision"`
for those documents — on the SoCLAaS qwen3-vl:32b rather than the 3B local
model — would be the higher-leverage change. Not implemented here.
