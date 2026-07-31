# Improvement Backlog

A prioritized backlog of performance, reliability, and security findings from a
full codebase review. The **Tier 1** items were implemented in the same pass that
produced this document; **Tier 2** and **Tier 3** items are captured here for a
future hardening pass. Each entry lists the file(s)/symbol(s) and a one-line fix
sketch so the work can be picked up without re-investigation.

Status legend: ✅ done · ⬜ open

---

## Tier 1 — High impact / low risk (DONE)

- ✅ **`.gitignore` hardening** — add `*.sif`, `*.bak`, `*.tmp`, `config.toml.*`.
  A 2.69 GB `rag_pipeline_cpu.sif` was sitting untracked-but-not-ignored; one
  `git add .` would have pushed it into history. `.gitignore`
- ✅ **Scrub real deployment IP from a tracked test** — `tests/test_web_app.py`
  hardcoded a real Tailscale CGNAT address (`100.87.142.5`). Replaced with an
  RFC 5737 documentation address (`192.0.2.10`).
- ✅ **SSH/rsync timeouts + hardening** — `_run_ssh`/`_run_rsync` had no timeout
  and no `BatchMode`/`ConnectTimeout`; a host-key prompt or stall wedged the
  only job worker forever. `src/hpc_backend.py`
- ✅ **Query-wait watchdog** — `_wait_for_no_queries` busy-waited with no upper
  bound; a leaked `active_query_count` permanently blocked all indexing. Now
  bounded by `[server] query_wait_timeout_seconds` (default 1800s).
  `src/web_app_classes/rag_job_queue.py`, `src/web_app.py`, `config.example.toml`
- ✅ **Gate bulk-data GETs behind auth** — `/api/index/stream` (full-corpus
  NDJSON) and `/api/metrics` (server paths/config) were open even with keys
  configured. `src/web_app.py` (`_SENSITIVE_GET_PATHS`)
- ✅ **Atomic key rotation** — `cmd_rotate` did create-then-delete as two locked
  writes; a crash between them left the old secret active (fail-open). New
  `KeyStore.replace_key` + `ApiKeyAuthenticator.rotate_key` do it in one write.
  `src/api_key_auth.py`, `scripts/manage_api_keys.py`

---

## Tier 2 — Medium (architecture / hardening)

### Performance

- ⬜ **Cache the `QueryEngine`/`EmbeddingEngine` as a singleton.** Every chat
  (`/api/chat/stream`) and `/api/index/vector-search` constructs a fresh
  `QueryEngine` which reloads the embedding model (~10 chats = 10 model loads).
  Add a module-level cached singleton keyed on model name; the cache hook
  already exists near the metrics endpoint. `src/web_app.py:~5410`, `~2261`
- ⬜ **Single-hash PDF fast path.** `list_pdf_documents` re-hashes/re-stats every
  PDF in `data/` to resolve one hash; called by `/api/pdfs/{hash}/{trust,download,
  view,reprocess,reindex}` and upload dedupe (`_data_pdf_duplicate_entries` does
  `data_dir.rglob("*.pdf")` + full-file SHA-256 per upload). Add a direct
  registry/source-map lookup for a single hash. `src/web_app.py:~2753,4358`
- ⬜ **Memoize the index-hierarchy tree derivation.** `list_index_summary_rows` /
  `list_index_child_rows` rebuild the `by_id` dict + walk all rows on each poll;
  the snapshot cache copies rows but the derived tree is recomputed.
  `src/web_app.py:~1939,1999`
- ⬜ **Prune empty rate-limiter buckets.** `RateLimiter._buckets` grows per
  identity and is never swept; deques are pruned but dict keys are kept forever
  (slow memory leak on a long-lived multi-user server). `src/api_key_auth.py:~377`
- ⬜ **Drop dead `streamlit` dependency.** `requirements.txt` retains an unused
  `streamlit~=1.58.0` (comment admits it); removes a large transitive tree from
  both venvs and both Singularity images. `requirements.txt:37`, `Singularity.def:58`,
  `Singularity.cpu.def:49`
- ⬜ **Switch GPU image to `cuda:...-runtime-...`.** `Singularity.def` uses the
  multi-GB `-devel` base but installs prebuilt cu121 torch wheels (no CUDA
  compile). `-runtime` is several GB smaller. Verify no transitive dep compiles
  kernels first. `Singularity.def:2`

### Reliability

- ⬜ **Assert single-worker uvicorn at startup.** The auth/rate-limit/usage/queue
  design is single-worker only (documented but unenforced). `--workers N>1`
  silently breaks rate limiting (`N × limit`), usage accounting, the
  `UPLOAD_FORCE_TOKEN_SECRET`, and double-executes jobs. Assert at startup or
  document as unsupported. `src/api_key_auth.py:373`, `src/web_app.py:~5454`
- ⬜ **Evict terminal jobs from in-memory `_jobs`.** Completed/failed/cancelled
  jobs are removed from the durable ledger but never from `_jobs`, so memory and
  `/api/jobs` poll cost grow unbounded over weeks. `src/web_app_classes/rag_job_queue.py:~398`
- ⬜ **Chunked-upload durability across restarts.** `_CHUNK_UPLOADS` is in-memory;
  on restart `chunk_status` loses filename/total_size and a `.part` older than
  24h is pruned, silently losing a long paused upload. `src/web_app.py:~4717,4764`
- ⬜ **`set -e` in Singularity `%post`.** A failed pip install can be silently
  followed by the rest of the build, yielding a broken-but-built image. Add
  `set -e` at the top of `%post` or chain with `&&`. `Singularity.def:15`,
  `Singularity.cpu.def`
- ⬜ **Network-failure retry in `requestJson`.** Transient 5xx/network errors
  throw immediately with no backoff; long HPC jobs behind a flaky tunnel produce
  many failed polls. `web/app.js:~324`
- ⬜ **Chat-stream resume on drop.** A mid-stream connection drop loses the whole
  answer with no resume from the last token. `web/app.js:~4175`
- ⬜ **Temp-config orphan cleanup in estimator.** `estimate_embed_throughput.py`
  writes a temp TOML (possibly containing secrets) into `%TEMP%`; killed mid-run
  it's never unlinked. Add `atexit` cleanup; avoid copying a secret-laden config.
  `scripts/estimate_embed_throughput.py:~343`

### Security

- ⬜ **Enforce `AuthResult.role` on mutating handlers.** `request.state.api_identity`
  is captured but never checked; a user-role key can cancel any job, delete any
  document, restore any backup. Add per-job/per-document ownership or role gates.
  `src/web_app_classes/rag_job_queue.py`, `src/web_app.py:~3612`
- ⬜ **Rate-count failed auth attempts.** Rejected credentials (401) do not count
  against any rate bucket — only successful auths are limited — enabling
  unlimited credential stuffing within connection limits. Count failures per-IP
  or per-supplied-token. `src/api_key_auth.py:~522`
- ⬜ **Warn/fail on `0.0.0.0` bind with no auth.** README documents binding to
  `0.0.0.0`; combined with the fresh-deploy no-op auth, a public deployment with
  no keys created is fully open for mutating actions. Emit a loud startup
  warning or require a token. `src/web_app.py:~3607`, `README.md:~206`
- ⬜ **XSS defense-in-depth.** Rendered chat HTML relies entirely on the single
  `MarkdownIt(html=False)` flag with zero client-side guard; a regression there
  executes injected HTML. (a) add client-side sanitization of `/api/render`
  output (e.g. DOMPurify); (b) add a URL-scheme allowlist for `source.url` /
  `open_url` / `download_url` before `href` (`javascript:` bypasses
  `escapeHtml`); (c) add a server test pinning that
  `render_markdown_text("<script>")` returns escaped text.
  `src/web_app.py:~3454`, `web/app.js:~705,3611`
- ⬜ **Trust `X-Forwarded-For` only behind configured proxies.** `_client_ip`
  trusts XFF unconditionally, so a client can spoof `last_used_ip`. Only honor
  it when a `trusted_proxies` config is set; else `request.client.host`.
  `src/web_app.py:~3550`
- ⬜ **Pin base images by digest.** Both Singularity recipes use floating tags
  (`nvidia/cuda:...-ubuntu22.04`, `python:3.14-slim`); a rebuilt upstream image
  silently changes the build. Pin with `@sha256:...`. `Singularity.def:2`,
  `Singularity.cpu.def`
- ⬜ **Windows SSH-key ACL hardening.** `setup_instance.py` skips `chmod 0600` on
  Windows (`os.name != "nt"` guards), so generated private keys get default
  ACLs on the project's primary platform. Use `icacls` to restrict to the owner.
  `scripts/setup_instance.py:~230,313`
- ⬜ **Document/limit default unlimited uploads.** `max_upload_bytes`/`max_corpus_bytes`
  default to `0` (unlimited); on a shared/LAN deployment one upload can exhaust
  disk. Set non-zero defaults or warn prominently. `config.example.toml:~101`

---

## Tier 3 — Polish / small hardening

- ⬜ **`escapeHtml` should escape `'`.** It escapes `& < > "` but not the single
  quote; safe today only because all attribute contexts use double quotes, but
  any future single-quoted attribute becomes an injection point. Add
  `.replaceAll("'", "&#39;")`. `web/app.js:~245`
- ⬜ **`uploadFormData` `resolvedWithAuthRetry` scope.** Verify it is not an
  implicit global (line assigns with no `let`/`const`/`var` visible) and is
  reset per request; concurrent uploads could race on the 401-retry flag.
  `web/app.js:~351`
- ⬜ **Pause background-tab pollers.** Three `setInterval` pollers (health 60s,
  jobs 2s-when-active, update 5min) run unconditionally when the tab is hidden.
  Add a `visibilitychange` listener. `web/app.js:~2106`
- ⬜ **Config warns on unknown keys.** `_merge_dataclass` silently drops
  misspelled keys (`if not hasattr: continue`); a typo gives no startup warning.
  Either represent all documented keys on dataclasses or warn on unknowns.
  `src/config.py:~229`
- ⬜ **`parse_qstat_job_state` stanza scoping.** It matches `job_state` anywhere
  in qstat output (a `Variable_List` could contain `job_state=...`); scope to the
  job stanza. `src/hpc_backend.py:~103`
- ⬜ **Serve-job ledger.** `submit_serve_job` is fire-and-forget; on restart the
  long-lived Ollama serve job is orphaned on the GPU cluster with no local
  record. `src/hpc_backend.py:~350`
- ⬜ **Heredoc delimiter assertion.** `_write_remote_pbs_script` relies on an
  implicit invariant that `generate_pbs_script` output contains no single quotes
  / the `HPC_PBS_EOF` delimiter. Assert it, or use a randomized delimiter.
  `src/hpc_backend.py:~389`
- ⬜ **`reliability.py` JSON decode.** `source_group_weight` falls back on
  `OSError` but not `JSONDecodeError`; a corrupt weights file raises uncaught.
  `src/reliability.py:~80`
- ⬜ **`reliability.py`/`pdf_registry.py` quadratic scan.** `_supersede_same_processed_paths`
  is O(N²) over all registry entries on every `mark_job_status`. Index by path.
  `src/pdf_registry.py:~269`
- ⬜ **Tracked sample PDFs in `data/`.** `data/*.pdf` are tracked despite `data/`
  being gitignored (committed before the rule). If not intended as fixtures,
  `git rm --cached data/*.pdf`. `data/`
- ⬜ **`_filter.py` stray scratch script.** A 6-line stdin filter at repo root,
  untracked, easy to accidentally commit. Delete or move under a scratch dir.
  `_filter.py`
