"""Engineer-usability endpoints: follow-up suggestions + answer feedback.

Covers POST/GET /api/chat/followups and POST/GET /api/feedback: happy paths,
input validation, auth gating, the query drain-guard pairing, and the
suggestion parser. The front-end contract (chips/thumbs) is exercised by the
browser suite; these tests pin the server behaviour those depend on.
"""
import json

import pytest
from fastapi.testclient import TestClient

import src.web_app as web_app


class RecordingQueue:
    """Stands in for the job queue; records drain-guard pairing."""

    def __init__(self):
        self.events = []

    def begin_query(self):
        self.events.append("begin")

    def finish_query(self):
        self.events.append("finish")


@pytest.fixture
def feedback_log(safe_tmp_path, monkeypatch):
    """Redirect the feedback log away from the developer's real data/ dir."""
    log_path = safe_tmp_path / "feedback.jsonl"
    monkeypatch.setattr(web_app, "FEEDBACK_LOG_PATH", log_path)
    return log_path


def _fake_llm_chat(replies):
    """Install a fake backend-agnostic chat transport; returns a capture list."""
    calls = []

    def fake_llm_chat(*, model, messages, options, stream, timeout=None, **kwargs):
        calls.append(
            {
                "model": model,
                "messages": messages,
                "options": options,
                "stream": stream,
            }
        )
        if isinstance(replies, Exception):
            raise replies
        return replies[len(calls) - 1] if len(replies) > 1 else replies[0]

    return fake_llm_chat, calls


# -- POST /api/chat/followups --------------------------------------------------


def test_followups_endpoint_returns_suggestions(monkeypatch):
    fake_llm_chat, calls = _fake_llm_chat(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '["What cell count fits the 2026 weight budget?", '
                                '"How does pack temperature limit charge rate?", '
                                '"Which rule caps the pack voltage?"]'
                            ),
                        }
                    }
                ]
            }
        ]
    )
    monkeypatch.setattr("src.local_rag._llm_chat", fake_llm_chat)
    queue = RecordingQueue()
    monkeypatch.setattr(web_app, "job_queue", queue)

    client = TestClient(web_app.app)
    response = client.post(
        "/api/chat/followups",
        json={
            "question": "Which battery chemistry should we run?",
            "answer": "The library sources point to Li-ion 21700 cells [S1].",
            "history": [
                {"role": "user", "content": "What cells fit the budget?"},
                {"role": "assistant", "content": "21700 cells are the best value."},
            ],
            "source_titles": ["Accumulator notes.pdf", "Rules 2026.pdf"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "suggestions": [
            "What cell count fits the 2026 weight budget?",
            "How does pack temperature limit charge rate?",
            "Which rule caps the pack voltage?",
        ]
    }
    # The LLM call is grounded in the question, the answer, and the sources.
    prompt = calls[0]["messages"][1]["content"]
    assert "Which battery chemistry should we run?" in prompt
    assert "Li-ion 21700 cells" in prompt
    assert "Accumulator notes.pdf" in prompt
    assert calls[0]["stream"] is False
    # The drain guard saw exactly one paired query.
    assert queue.events == ["begin", "finish"]


def test_followups_endpoint_requires_question_and_answer():
    client = TestClient(web_app.app)
    response = client.post(
        "/api/chat/followups",
        json={"question": "   ", "answer": "An answer."},
    )
    assert response.status_code == 400


def test_followups_endpoint_maps_llm_failure_to_502(monkeypatch):
    fake_llm_chat, _ = _fake_llm_chat(RuntimeError("backend unreachable"))
    monkeypatch.setattr("src.local_rag._llm_chat", fake_llm_chat)
    monkeypatch.setattr(web_app, "job_queue", RecordingQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/chat/followups",
        json={"question": "Q?", "answer": "An answer."},
    )
    assert response.status_code == 502
    assert "backend unreachable" in response.json()["detail"]


def test_followups_endpoint_caps_source_titles(monkeypatch):
    fake_llm_chat, calls = _fake_llm_chat(
        [{"choices": [{"message": {"content": '["One suggestion here?", "Another suggestion?", "And a third?"]'}}]}]
    )
    monkeypatch.setattr("src.local_rag._llm_chat", fake_llm_chat)
    monkeypatch.setattr(web_app, "job_queue", RecordingQueue())

    client = TestClient(web_app.app)
    response = client.post(
        "/api/chat/followups",
        json={
            "question": "Q?",
            "answer": "A",
            # 10 titles: only the first 8 distinct survive the cap.
            "source_titles": [f"Source {index}.pdf" for index in range(10)],
        },
    )
    assert response.status_code == 200
    prompt = calls[0]["messages"][1]["content"]
    assert "Source 7.pdf" in prompt
    assert "Source 8.pdf" not in prompt


# -- suggestion parser ---------------------------------------------------------


def test_parse_followup_suggestions_json_array_with_fallbacks():
    parse = web_app._parse_followup_suggestions
    # Fenced JSON is recovered.
    fenced = 'Here you go:\n```json\n["What mesh does the wing use?", "How is downforce measured?"]\n```'
    assert parse(fenced) == ["What mesh does the wing use?", "How is downforce measured?"]
    # Numbered-line fallback works without JSON.
    numbered = "1. First follow-up question?\n2. Second follow-up question?\n3. Third follow-up question?"
    assert len(parse(numbered)) == 3
    # Bare prose without markers or question marks is rejected.
    assert parse("Here are three ideas you might like.") == []
    # Deduplication is case-insensitive and short/long junk is dropped.
    messy = ["which material fails first?", "Which material fails first?", "x", "ok" * 150]
    assert parse(json.dumps(messy)) == ["which material fails first?"]
    assert parse("") == []


# -- POST/GET /api/feedback ----------------------------------------------------


def test_feedback_roundtrip_and_tally(feedback_log):
    client = TestClient(web_app.app)
    first = client.post(
        "/api/feedback",
        json={
            "rating": "up",
            "question": "What steering geometry do the sources recommend?",
            "answer_excerpt": "Ackermann geometry reduces tyre scrub [S1].",
            "sources_count": 2,
        },
    )
    assert first.status_code == 200
    assert first.json()["ok"] is True

    second = client.post(
        "/api/feedback",
        json={
            "rating": "down",
            "question": "What steering geometry do the sources recommend?",
            "answer_excerpt": "Ackermann geometry reduces tyre scrub [S1].",
            "note": "the 2026 rules changed this",
            "sources_count": 2,
        },
    )
    assert second.status_code == 200

    # The log holds one JSON line per rating, newest record first in the GET.
    lines = feedback_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    listed = client.get("/api/feedback").json()
    assert listed["total"] == 2
    assert listed["up"] == 1
    assert listed["down"] == 1
    assert listed["records"][0]["note"] == "the 2026 rules changed this"
    assert listed["records"][0]["rating"] == "down"
    assert listed["records"][0]["sources_count"] == 2
    assert listed["records"][0]["timestamp"]

    # A GET without any records yet is an empty, valid listing.
    empty_client = TestClient(web_app.app)
    assert empty_client.get("/api/feedback").json()["total"] == 2


def test_feedback_rejects_invalid_rating(feedback_log):
    client = TestClient(web_app.app)
    response = client.post("/api/feedback", json={"rating": "meh", "question": "Q?"})
    assert response.status_code == 422
    assert not feedback_log.exists()


def test_feedback_requires_question(feedback_log):
    client = TestClient(web_app.app)
    response = client.post("/api/feedback", json={"rating": "up"})
    assert response.status_code == 422


@pytest.mark.remote_client
def test_feedback_get_is_gated_for_remote_clients(monkeypatch, feedback_log):
    """GET /api/feedback carries user notes, so it is a sensitive GET."""
    monkeypatch.setattr(web_app, "_API_TOKEN", "secret-token-123")
    client = TestClient(web_app.app)
    assert client.get("/api/feedback").status_code == 401
    assert (
        client.get("/api/feedback", headers={"X-API-Token": "secret-token-123"}).status_code
        == 200
    )


@pytest.mark.remote_client
def test_feedback_post_is_gated_for_remote_clients(monkeypatch, feedback_log):
    monkeypatch.setattr(web_app, "_API_TOKEN", "secret-token-123")
    client = TestClient(web_app.app)
    blocked = client.post("/api/feedback", json={"rating": "up", "question": "Q?"})
    assert blocked.status_code == 401
    allowed = client.post(
        "/api/feedback",
        json={"rating": "up", "question": "Q?"},
        headers={"X-API-Token": "secret-token-123"},
    )
    assert allowed.status_code == 200
