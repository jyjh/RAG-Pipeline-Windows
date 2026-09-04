"""Category registry ("split databases") and multi-category query union.

Covers the SQLite category store (CRUD, membership, General semantics) and the
MultiVectorStore score-merged search over several category index directories.
"""

import numpy as np
import pytest

from src.categories import (
    GENERAL_CATEGORY_KEY,
    CategoryStore,
    categories_path,
    category_db_dir,
    normalize_category_key,
)
from src.vector_store import LanceDBVectorStore, MultiVectorStore


# --- CategoryStore -----------------------------------------------------------


def _store(tmp_path):
    return CategoryStore(tmp_path / "data")


def test_general_is_implicit(tmp_path):
    store = _store(tmp_path)
    entries = store.list_categories()
    assert [entry["key"] for entry in entries] == [GENERAL_CATEGORY_KEY]
    assert entries[0]["custom"] is False
    assert store.get_category("general") is not None
    assert store.get_category("missing") is None


def test_category_db_dir_resolution(tmp_path):
    anchor = tmp_path / "db"
    assert category_db_dir(anchor, "general") == anchor
    assert category_db_dir(anchor, "Team-1") == anchor / "categories" / "team-1"
    with pytest.raises(ValueError):
        category_db_dir(anchor, "../escape")


def test_create_list_rename_delete_lifecycle(tmp_path):
    store = _store(tmp_path)
    created = store.create_category("fsae_design", "FSAE Design Docs")
    assert created["key"] == "fsae_design"
    assert created["label"] == "FSAE Design Docs"
    # Duplicate and reserved keys are rejected.
    with pytest.raises(ValueError, match="already exists"):
        store.create_category("fsae_design")
    with pytest.raises(ValueError, match="reserved"):
        store.create_category("general")
    assert store.rename_category("fsae_design", "Team Design")["label"] == "Team Design"
    entries = {entry["key"]: entry for entry in store.list_categories()}
    assert set(entries) == {"general", "fsae_design"}
    assert entries["fsae_design"]["label"] == "Team Design"
    assert store.delete_category("fsae_design")["deleted"] is True
    assert [entry["key"] for entry in store.list_categories()] == [GENERAL_CATEGORY_KEY]
    with pytest.raises(ValueError, match="Unknown category"):
        store.delete_category("fsae_design")


def test_category_weights_have_named_defaults_and_are_adjustable(tmp_path):
    store = _store(tmp_path)
    store.create_category("historical-documents", "Historical Documents")
    store.create_category("design-2026-2027", "2026/2027 Design Documentation")

    entries = {entry["key"]: entry for entry in store.list_categories()}
    assert entries["general"]["weight"] == 1.0
    assert entries["historical-documents"]["weight"] == 0.1
    assert entries["design-2026-2027"]["weight"] == 1.5

    updated = store.update_weight("historical-documents", 0.25)
    assert updated["weight"] == 0.25
    assert store.get_category("historical-documents")["weight"] == 0.25
    with pytest.raises(ValueError, match="fixed"):
        store.update_weight("general", 2)


@pytest.mark.parametrize(
    "bad_key",
    ["", "Has Space", "slash/slash", "ümlaut", "-leading", "x" * 65],
)
def test_invalid_keys_rejected(tmp_path, bad_key):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.create_category(bad_key)
    with pytest.raises(ValueError):
        normalize_category_key(bad_key)


def test_delete_blocked_while_members_remain(tmp_path):
    store = _store(tmp_path)
    store.create_category("keep")
    store.set_memberships(["hash-a", "hash-b"], "keep")
    with pytest.raises(ValueError, match="still has 2 document"):
        store.delete_category("keep")
    store.set_memberships(["hash-a", "hash-b"], GENERAL_CATEGORY_KEY)
    assert store.delete_category("keep")["deleted"] is True


def test_membership_assign_move_and_counts(tmp_path):
    store = _store(tmp_path)
    store.create_category("books")
    store.create_category("team")
    assert store.set_memberships(["h1", "h2"], "books") == 2
    assert store.memberships_for(["h1", "h2", "h3"]) == {"h1": "books", "h2": "books"}
    assert store.membership_counts() == {"books": 2}
    # Re-assignment moves, never duplicates.
    store.set_memberships(["h1"], "team")
    assert store.memberships_for(["h1"]) == {"h1": "team"}
    assert store.membership_counts() == {"books": 1, "team": 1}
    # General membership is the ABSENCE of a row.
    store.set_memberships(["h1"], GENERAL_CATEGORY_KEY)
    assert store.memberships_for(["h1"]) == {}
    # GROUP BY omits zero-count categories; callers use .get(key, 0).
    assert store.membership_counts() == {"books": 1}
    # Assigning to an unknown category fails loudly.
    with pytest.raises(ValueError, match="Unknown category"):
        store.set_memberships(["h9"], "ghost")


def test_memberships_for_chunks_large_corpora(tmp_path):
    """A corpus-wide lookup must not trip SQLite's host-parameter ceiling."""
    store = _store(tmp_path)
    store.create_category("big")
    hashes = [f"{index:064x}" for index in range(1200)]
    store.set_memberships(hashes, "big")
    found = store.memberships_for(hashes)
    assert len(found) == 1200
    assert set(found.values()) == {"big"}


def test_state_version_bumps_on_writes(tmp_path):
    store = _store(tmp_path)
    assert store.state_version() == "0"
    store.create_category("one")
    version_after_create = store.state_version()
    assert version_after_create != "0"
    store.set_memberships(["h1"], "one")
    assert store.state_version() != version_after_create


def test_list_categories_resolves_db_dirs(tmp_path):
    store = _store(tmp_path)
    store.create_category("empty-cat")
    anchor = tmp_path / "db"
    (anchor / "categories" / "empty-cat").mkdir(parents=True)
    entries = {entry["key"]: entry for entry in store.list_categories(anchor_db_dir=anchor)}
    assert entries["general"]["db_dir"] == str(anchor)
    assert entries["empty-cat"]["db_dir"] == str(anchor / "categories" / "empty-cat")
    # exists reflects an on-disk lancedb dir; neither has one yet.
    assert entries["general"]["exists"] is False
    assert entries["empty-cat"]["exists"] is False
    (anchor / "lancedb").mkdir()
    entries = {entry["key"]: entry for entry in store.list_categories(anchor_db_dir=anchor)}
    assert entries["general"]["exists"] is True


def test_categories_path_layout(tmp_path):
    assert categories_path(tmp_path / "data") == tmp_path / "data" / ".categories.json"


# --- MultiVectorStore --------------------------------------------------------


DIM = 8


def _records(doc_id, *, base, n_chunks, source_hash):
    """One document_summary + n chunk records with deterministic vectors."""
    rows = [
        {
            "id": f"{doc_id}-summary",
            "doc_id": doc_id,
            "parent_id": "",
            "node_type": "document_summary",
            "file_path": f"{doc_id}.md",
            "chunk_index": -1,
            "content": f"{doc_id} summary text {base}",
            "title": doc_id,
            "section_path": "",
            "page_start": 1,
            "page_end": 2,
            "summary": f"{doc_id} summary",
            "tags": [],
            "source_hash": source_hash,
            "source_pdf_name": f"{doc_id}.pdf",
            "source_pdf_path": f"/tmp/{doc_id}.pdf",
            "embedding_model": "test-model",
            "embedding_dim": DIM,
            "vector": [base] * DIM,
        }
    ]
    for index in range(n_chunks):
        rows.append(
            {
                "id": f"{doc_id}-chunk-{index}",
                "doc_id": doc_id,
                "parent_id": f"{doc_id}-summary",
                "node_type": "chunk",
                "file_path": f"{doc_id}.md",
                "chunk_index": index,
                "content": f"{doc_id} chunk {index} content {base}",
                "title": doc_id,
                "section_path": f"{doc_id}/s{index}",
                "page_start": 1,
                "page_end": 2,
                "summary": "",
                "tags": [],
                "source_hash": source_hash,
                "source_pdf_name": f"{doc_id}.pdf",
                "source_pdf_path": f"/tmp/{doc_id}.pdf",
                "embedding_model": "test-model",
                "embedding_dim": DIM,
                "vector": [base + index * 0.1] * DIM,
            }
        )
    return rows


@pytest.fixture
def two_category_stores(tmp_path):
    general_dir = tmp_path / "db"
    team_dir = tmp_path / "db" / "categories" / "team"
    team_dir.mkdir(parents=True)
    general = LanceDBVectorStore(general_dir)
    team = LanceDBVectorStore(team_dir)
    general.write_records(
        _records("textbook", base=0.9, n_chunks=2, source_hash="hash-general"),
        embedding_model="test-model",
        embedding_dim=DIM,
    )
    team.write_records(
        _records("design-doc", base=0.1, n_chunks=2, source_hash="hash-team"),
        embedding_model="test-model",
        embedding_dim=DIM,
    )
    return general, team


def test_multi_store_search_merges_and_annotates(two_category_stores):
    general, team = two_category_stores
    multi = MultiVectorStore([general, team], labels=["General", "Team"], keys=["general", "team"])
    assert multi.exists() is True
    assert multi.count() == 6
    assert multi.metadata() == ("test-model", DIM)

    query_vector = [1.0] * DIM  # identical to the textbook chunk-1 vector
    results = multi.search(query_vector, top_k=4)
    # All three textbook rows (base 0.9/1.0) outrank the design-doc rows
    # (base 0.1); exact order within the textbook ties is Lance's choice.
    assert {row["id"] for row in results[:3]} == {
        "textbook-summary",
        "textbook-chunk-0",
        "textbook-chunk-1",
    }
    assert results[3]["id"].startswith("design-doc")
    scores = [row["score"] for row in results]
    assert scores == sorted(scores, reverse=True)
    # Every row is annotated with the label of the store it came from.
    by_id = {row["id"]: row for row in results}
    assert by_id["textbook-summary"]["category"] == "General"
    assert by_id["design-doc-summary"]["category"] == "Team"


def test_multi_store_search_dedupes_shared_ids(tmp_path):
    """Mid-transfer a document exists in two categories; the merged search
    must surface it once (record ids are content-derived and identical)."""
    first = LanceDBVectorStore(tmp_path / "a")
    second = LanceDBVectorStore(tmp_path / "b")
    records = _records("shared-doc", base=0.5, n_chunks=1, source_hash="hash-shared")
    for store in (first, second):
        store.write_records(records, embedding_model="test-model", embedding_dim=DIM)
    multi = MultiVectorStore([first, second], labels=["A", "B"], keys=["a", "b"])
    results = multi.search([0.5] * DIM, top_k=10)
    ids = [row["id"] for row in results]
    assert len(ids) == len(set(ids)) == 2
    # First (highest-ranked, deterministic tie) store wins the annotation.
    assert all(row["category"] == "A" for row in results)


def test_multi_store_child_chunks_route_by_category(two_category_stores):
    general, team = two_category_stores
    multi = MultiVectorStore([general, team], labels=["General", "Team"], keys=["general", "team"])
    results = multi.search([0.1] * DIM, top_k=10)
    parent = next(row for row in results if row["id"] == "design-doc-summary")
    children = multi.child_chunks(parent, limit=10)
    assert [row["id"] for row in children] == ["design-doc-chunk-0", "design-doc-chunk-1"]
    assert all(row["category"] == "Team" for row in children)
    # A general-index parent routes back to the general store.
    general_parent = next(row for row in results if row["id"] == "textbook-summary")
    children = multi.child_chunks(general_parent, limit=10)
    assert children and all(row["category"] == "General" for row in children)


def test_multi_store_empty_stores(tmp_path):
    missing = LanceDBVectorStore(tmp_path / "missing")
    multi = MultiVectorStore([missing], labels=["Empty"], keys=["empty"])
    assert multi.exists() is False
    assert multi.count() == 0
    assert multi.search([0.0] * DIM, top_k=5) == []
