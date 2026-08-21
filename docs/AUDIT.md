# Final Audit And Cleanup Plan

Date: 2026-08-18 · Branch: `soc-llm` · Scope: full tree (113 tracked files, ~23.3k lines of Python in `src/`, `scripts/`, `main.py`).

This is the final audit of the codebase after the SoCLAaS-API overhaul. It records
what is healthy, what duplication and cleanliness debt remains, and a phased plan
to pay it down. Item **P0** is already implemented alongside this document; the
remaining phases are planned but not yet executed.

---

## 1. Executive summary

The pipeline is functionally coherent and well tested (19 test modules, all
passing at audit time; `compileall` clean). The dominant structural debt is the
`<module>.py` + `<module>_classes/` **split-class machinery**: 29 child files
import their parent module, borrow its globals through a proxy/namespace-binding
shim, then re-bind the class identity back — roughly 290 lines of pure ceremony
that preserves, rather than removes, the coupling it was meant to reduce. Beyond
that, the debt is ordinary: nine copies of `utcnow`, duplicated config loaders
between `main.py` and `web_app.py`, copy-pasted Docling/parser blocks, a drifted
PBS template, and stale keys in `config.example.toml` that the loader warns
about and drops.

## 2. What is healthy (do not regress)

- **Retry/timeouts centralized**: `src/llm_api.py:retry_with_backoff` is the
  single engine; `src/embeddings.py` and `src/atomic_io.py` are the only
  purpose-built wrappers. No scattered retry copies.
- **Atomic IO + file locking** (`src/atomic_io.py`, `src/file_lock.py`) used
  consistently by registries, ledgers, and the job queue.
- **`src/coerce.py` + `src/defaults.py`** exist as shared constant/coercion
  modules — earlier dedup work landed; the remaining `_as_bool`/`_bool_value`
  aliases in `web_app.py` are residue (see F3).
- **Setup wizard** (`scripts/setup_instance.py`) is stdlib-only, tested
  (`tests/test_setup_instance.py`), and handles SSH keys, TOML surgery with
  backups, content-fingerprinted HPC provisioning, preflight checks, and
  dependency installation. P0 below closes its remaining one-click gaps.
- **No dead modules** — every `src/` module is imported by production code,
  scripts, or tests (`restart_server.py` is launched as a subprocess).

## 3. Findings

### F1 — Split-class machinery (highest impact)

`src/_class_module_support.py` implements `import_split_class`,
`finalize_split_class`, `bind_module_namespace`, and a `_PendingSplitInstance`
sentinel (its docstring admits it guards a "circular-import timing edge").
Every child file under `ingestion_classes/`, `local_rag_classes/`,
`sectioning_classes/`, `vector_store_classes/`, `web_app_classes/` repeats the
same ~10-line ceremony:

```python
from src._class_module_support import bind_module_namespace, finalize_split_class
import src.web_app as _source_module
bind_module_namespace(_source_module, globals(),
                      proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS)
...
cls.__module__ = _source_module.__name__
finalize_split_class(_source_module, cls)
```

Consequences:

- Parent↔child circular imports everywhere, papered over by the sentinel.
- Child classes still read parent globals un-imported (`chat_request.py` uses
  `DEFAULT_LLM_MODEL`, `CHAT_CONFIG`; parsers use `_build_docling_converter`,
  `logger`), so the split is cosmetic: `ingestion.py` (952 lines) still holds
  all the machinery the parsers borrow.
- The hack *creates* duplication: `web_app_classes/queue_job.py:12` keeps a
  local `_utcnow` copy explicitly because borrowed globals "may not be present
  yet" under the timing edge.

Estimated removal target: ~290 ceremony lines plus deletion of
`_class_module_support.py` (~102 lines) once children are normal modules.

### F2 — God modules

| File | Lines | Problem |
|---|---|---|
| `src/web_app.py` | 5,486 | ~202 top-level functions, 41 routes across 12+ feature areas, module-level mutable singletons |
| `src/web_app_classes/rag_job_queue.py` | 1,416 | one 50+-method class doing uploads, subprocess ingestion, indexing, backups, compaction, ledger recovery |
| `src/local_rag.py` | 1,360 | ~70 functions: failover, chat, parsing, token estimation, BM25/RRF, citations, manifests, web search |
| `scripts/setup_instance.py` | 1,278 | config + SSH + rsync + container build + preflight + CLI (still readable; lowest priority) |

### F3 — Duplicated helpers

- **`utcnow` ×9** (identical `datetime.now(timezone.utc).isoformat(timespec="seconds")`):
  `asset_store.py:20`, `pdf_registry.py:28`, `index_overrides.py:16`,
  `job_ledger.py:42`, `restart_server.py:13`, `web_app.py:402`,
  `web_app_classes/queue_job.py:12`, `job_logging.py:37`, plus
  `local_rag.py:1038` and variants in `api_key_auth.py:66-71`.
- **mtime-signature JSON cache ×2 (+1 variant)**: `pdf_registry.py:40-75` vs
  `job_ledger.py:46-80` (its docstring says "same memoization strategy as
  pdf_registry._load_json"); `asset_store.py:40-50` is a third, simpler variant.
- **Config loaders duplicated between `main.py` and `src/web_app.py`**:
  `_load_ingestion_config` (`main.py:104-128` vs `web_app.py:584-608`, ~25 lines,
  with `_as_bool`/`_bool_value` both aliasing `src/coerce.as_bool`),
  `_load_indexing_config` (`main.py:133-140` vs `web_app.py:624-633`),
  `_pipeline_config` (`main.py:97-101` vs `web_app.py:459-463`), and
  `_load_query_config` vs `_load_chat_config` (~15 shared keys). ≈90 duplicated
  lines that belong in `src/config.py`.
- **Legacy alias residue**: `web_app.py:413-422` keeps `_positive_int`,
  `_bool_value`, … "because the split-class modules and tests reference them",
  plus additional local coercers (`_nonnegative_int`, `_nonempty_str`,
  `_positive_int_or`) that bypass `src/coerce.py`.
- **`_ollama_host` ×2 with diverged semantics**: `embeddings.py:31` (env >
  hardcoded) vs `local_rag.py:163` (env > failover host > config).
- **Byte humanizer ×2**: `disk_space.py:31` vs `web_app_classes/rag_job_queue.py:26`.
- **"arg > env > default" resolution repeated ~20×**: ~100 lines in
  `local_query_engine.py:74-180`, more in `local_vector_indexer.py:60-100` and
  `embeddings.py:90-140` (42 `os.environ.get` sites total).

### F4 — Near-identical classes

- **8 Pydantic request models in 8 files** (`web_app_classes/*_request.py`):
  each ~10 boilerplate lines (F1) around a 3–15-line model. The embedding-options
  trio (`embedding_model` / `embedding_batch_size` / `embedding_timeout`) is
  copy-pasted into 4 of them. `BulkDocumentTrustRequest` was left inline in
  `web_app.py:3337` while its sibling was split out.
- **PDF parsers**: the page-isolation block (writer → add_page → temp
  `page_N.pdf`) is duplicated in `docling_pdf_parser.py` and
  `manual_text_pdf_parser.py`; the 14-kwarg `_build_docling_converter(...)` call
  appears twice *inside* `docling_pdf_parser.py`; `document_processor.py:43-127`
  constructs three parsers with near-identical ~14-kwarg lists (~70 lines
  reducible to one builder).
- **LanceDB column list written 3×**: `REQUIRED_LANCEDB_COLUMNS` /
  `LIST_RECORD_COLUMNS` in `vector_store.py:25-55` and the 19 keys of
  `SectionChunk.to_record` must be kept in sync manually.

### F5 — HPC copy-paste with drift

- `src/hpc.py:199-260` embeds a PBS template that duplicates
  `scripts/nus_hpc_ingest_index.pbs` (~55 lines), and the copies have
  **drifted**: the static `.pbs` asks for `ngpus=1` + `--nv` while defaulting to
  the *CPU* image `rag_pipeline_cpu.sif` — internally inconsistent since the GPU
  path was removed.
- SSH option lists re-typed with three incompatible tunings:
  `hpc_backend.py:68-75` (`_SSH_BASE_OPTS`: ConnectTimeout=15, CountMax=4) vs
  `setup_instance.py` at lines 372-375, 505-508, 528-529, 541-544, 598-599,
  662 (ConnectTimeout=20, CountMax=6) vs the `ssh_config` written by the wizard
  (Interval=15, CountMax=3).

### F6 — Dead / stale code

- Dead functions: `reliability.py:61 source_group_label`,
  `reliability.py:69 source_group_is_assignable` (zero references incl. tests).
- Test-only: `progress_protocol.py:94 is_progress_line`.
- `config.example.toml` ships keys the loader warns about and drops:
  `ocr_strategy`, `figure_crop_enabled`, `table_structure`,
  `progress_enabled` (`[ingestion]`) and `web_search.provider` — none are
  dataclass fields. *(Fixed in P0.)*
- `README.md` "Current Capabilities" still lists `src/utils.py`, which no longer
  exists. *(Fixed in P0.)*

### F7 — Dependencies file

`requirements.txt` uses `~=` pins (patch drift allowed; the roadmap's own
"pin after validation" item is still open); `pytest` (dev-only) is mixed into
runtime requirements; `sentence-transformers` is imported by the
`native_embeddings` path (`src/embeddings.py:248`) but is not listed; the
`torch`/`torchvision` entries are CPU-build placeholders the user must override
with CUDA wheels of the same version (easy to desync).

### F8 — Documentation drift

- `docs/HPC_DELEGATION.md` still documents a `[hpc.gpu]` section and files that
  were deleted (`nus_hpc_serve.pbs`, `tunnel_daemon.*`).
- `docs/HPC_RUNBOOK.md` "run the web app" section still says the app talks to
  Ollama at `127.0.0.1:11434` (only the top banner corrects it).
- `docs/IMPROVEMENTS.md` lists several Tier-2 items as open that are already
  fixed in the tree (streamlit removal, unknown-key warnings, etc.).

---

## 4. Cleanup plan

Each phase is independently shippable, ordered by risk-adjusted value. Run
`pytest` + `python -m compileall src` after every phase. No phase changes
external behavior; all are refactors with the test suite as the guard.

### P0 — One-click setup hardening (implemented with this audit)

Deliverable: clone → double-click `setup.cmd` (or `./setup.sh`) → working server.

- `scripts/setup_instance.py`: `--set-api-key` flag + interactive prompt for the
  SoCLAaS key (persisted to `[llm_api].api_key`; env vars still override);
  live key verification against `/v1/models` after configuration (warn-only);
  preflight row for the API key (WARN, not FAIL, when missing on the soclaas
  backend); missing dependencies now auto-install in `--non-interactive` runs
  (`--no-install-deps` opts out); top-level `main()` wrapper converts
  `KeyboardInterrupt` / `EOFError` / unexpected exceptions into concise messages
  with proper exit codes instead of tracebacks; port prompts retry on bad input.
- `setup.sh`: `python3` → `python` fallback with a clear error when absent.
- `setup.cmd`: explicit "Python not found" message instead of a Store stub.
- `config.example.toml`: stale keys removed (F6).
- `README.md`: one-click quick start documented; stale `src/utils.py` reference
  removed.
- `tests/test_setup_instance.py`: coverage for the new pieces.

### P1 — Mechanical dedup (low risk, high certainty)

1. Add `src/timeutil.py` with `utcnow()` / `utcnow_iso()`; delete the 9 copies
   (F3). Import-compatibility shims are unnecessary — replace call sites.
2. Extract the mtime-signature JSON cache (load/memoize/write-atomic) into one
   module (`src/caches.py` already exists as a home); `pdf_registry`,
   `job_ledger`, `asset_store` consume it.
3. Move the four config loaders into `src/config.py` as the single source
   (`load_ingestion_config()`, `load_indexing_config()`, `load_query_config()`,
   `load_chat_config()`); `main.py` and `web_app.py` call them. Delete the
   `_as_bool`/`_bool_value` alias block in `web_app.py` and fold
   `_nonnegative_int`/`_nonempty_str`/`_positive_int_or` into `src/coerce.py`.
4. Delete the two dead functions in `reliability.py` (F6).
5. Unify `_ollama_host` (config-aware version wins; `embeddings.py` calls it)
   and the byte humanizer.

Estimated deletion: ~250 lines. Effort: small. Risk: low (tests cover the
config surface well).

### P2 — Consolidate request models and small dataclasses

1. Merge the 8 `web_app_classes/*_request.py` files into one `requests.py`
   (~80 lines) with a shared `EmbeddingOptions` base carrying the embedding
   trio; move the stray `BulkDocumentTrustRequest` in with them.
2. Merge `sectioning_classes`' three dataclasses into `sectioning.py` (or one
   `models.py`); derive `LIST_RECORD_COLUMNS` from `REQUIRED_LANCEDB_COLUMNS`
   (or from `SectionChunk.to_record` keys) so the column list exists once.
3. Introduce a `resolve(name, env_default, default, coercer)` helper (or a
   declarative settings table) for the ~20 "arg > env > default" sites
   (F3), starting with `local_query_engine.py`.

Estimated deletion: ~150 lines + one full pattern removed. Effort: medium.
Risk: low-medium (Pydantic model identity is asserted in a few tests — update
them in the same change).

### P3 — Parser consolidation

1. Extract `isolate_page(page) -> Path` and `extract_page_text(...)` helpers;
   use them in both docling and manual parsers (F4).
2. One `_docling_converter_kwargs(...)` builder; both call sites in
  `docling_pdf_parser.py` and the three constructor calls in
   `document_processor.py` consume it (`functools.partial` for the
   role-specific instances).

Estimated deletion: ~100 lines. Effort: medium. Risk: medium — parser tests
(`test_ingestion_parsers.py`) are the guard; run a real fixture PDF through
ingest before/after and diff the Markdown.

### P4 — Retire the split-class machinery (structural)

Do it one parent at a time, each step shippable, `web_app.py` last:

1. `sectioning_classes` and `vector_store_classes` (smallest, fewest
   borrowed globals): convert children to normal modules that import what they
   use; delete the ceremony lines and the `_CLASS_MODULE_PROXY_FUNCTIONS` tuple
   from those parents.
2. `ingestion_classes`: move the borrowed helpers
   (`_default_pdf_writer`, `_png_bytes_for_vision`, `_build_docling_converter`,
   …) into the child modules that own them; parents import from children.
3. `local_rag_classes`: split `local_rag.py`'s ~70 functions into coherent
   modules (failover, chat transport, scoring, citations, manifest, web search)
   as part of the conversion.
4. `web_app_classes`: after P2 only `rag_job_queue.py` remains; decompose the
   queue (upload staging / subprocess ingestion / index mutation / backup /
   compaction) into collaborating modules with explicit imports.
5. Delete `src/_class_module_support.py` and all
   `_CLASS_MODULE_PROXY_FUNCTIONS` tuples.

Estimated deletion: ~400 lines and the circular-import sentinel class.
Effort: large. Risk: medium — the sentinel exists because of real timing edges;
each conversion must be verified with a cold-start `python -m src.web_app` and
the full suite, not just pytest.

### P5 — HPC unification

1. Make `scripts/nus_hpc_ingest_index.pbs` a generated artifact of
   `src/hpc.py:generate_pbs_script` (or delete the static file and generate on
   demand); fix the `ngpus=1`/CPU-image inconsistency while there.
2. Single source for SSH options: export the option list from
   `src/hpc_backend.py` and have `scripts/setup_instance.py` build its
   ssh/rsync command lines from it (pick one tuning: ConnectTimeout=20,
   Interval=30, CountMax=6 — the values setup_instance already uses).

Estimated deletion: ~80 lines. Effort: small-medium. Risk: low (covered by
`test_hpc_backend.py`, `test_hpc_integration.py`, `test_setup_instance.py`).

### P6 — Housekeeping (optional, batched)

- `requirements.txt`: split `pytest` into a dev extra; document (or add) the
  optional `sentence-transformers` extra; note the CUDA-wheel override next to
  the torch pins.
- Refresh stale docs (F8): drop `[hpc.gpu]` from `HPC_DELEGATION.md`, update the
  runbook's web-app section, reconcile `IMPROVEMENTS.md` statuses.
- Reconcile `main.py`'s `run_ingestion`/`run_indexing` trampolines with the
  real functions (print-noise wrappers).

### Guardrails for every phase

- `python -m compileall main.py src scripts` clean.
- `pytest` fully green (update tests in-phase, never after).
- Cold-start smoke: `python -m src.web_app` boots and `/api/health` returns.
- One real PDF through `--mode ingest` + `--mode index` producing a byte-stable
  `db/` (for P3/P4).

## 5. Point-in-time metrics

- Python: 113 tracked files, 23,324 lines in `src/` + `scripts/` + `main.py`.
- Top files by size: `web_app.py` 5,486 · `rag_job_queue.py` 1,416 ·
  `local_rag.py` 1,360 · `setup_instance.py` 1,278 · `local_query_engine.py`
  1,164.
- Estimated duplicated lines (sum of F1, F3, F4, F5): ~950, i.e. ~4% of the
  codebase, concentrated in the five `*_classes/` packages and the two
  config-loading surfaces.
