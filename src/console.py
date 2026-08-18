"""Console output helpers shared by the ingest/index/query pipelines.

``status`` is the single stderr-print primitive (previously ``_status`` /
``_progress_status`` duplicated in local_rag.py, ingestion.py, indexing.py, and
embeddings.py). ``iter_with_progress`` wraps tqdm uniformly so progress bars
look identical across phases. Both honor ``enabled`` so ``--no_progress`` and
the job queue's subprocess mode can silence them.
"""
from __future__ import annotations

import sys
from typing import Iterable, Iterator


def status(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _tqdm():
    from tqdm import tqdm

    return tqdm


def iter_with_progress(
    iterable: Iterable,
    *,
    enabled: bool,
    total: int | None,
    desc: str,
    unit: str,
) -> Iterator:
    if not enabled:
        return iterable
    return _tqdm()(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=False,
        dynamic_ncols=True,
        ascii=True,
    )
