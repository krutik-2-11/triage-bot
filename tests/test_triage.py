"""Step 1.7: tests that run without Ollama.

Fakes replace the model and the search, so we test OUR logic:
schema, retry loop, give-up path, and routing.
"""
import json

import pytest
from pydantic import ValidationError

import triage_graph
from summarize_chain import TriageSummary
from triage_graph import build_graph

GOOD = TriageSummary(pain_point="GPU not used", area="gpu",
                     kind="bug", key_terms=["gpu", "cuda"])


# ---------- Fakes ----------
class FakeChain:
    """Fails the first `fail_times` calls (like broken JSON), then returns GOOD."""
    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.calls = 0

    def invoke(self, inputs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError("broken JSON")
        return GOOD


class FakeRetriever:
    """Always returns the same matches with fixed scores."""
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def search(self, text, k=3):
        self.calls += 1
        return [{"number": 100 + i, "title": f"Issue {i}",
                 "url": f"https://example.com/{i}", "score": s}
                for i, s in enumerate(self.scores)]


def run(chain, retriever):
    graph = build_graph(chain=chain, retriever=retriever, threshold=0.60)
    return graph.invoke({"question": "test question", "attempts": 0})


@pytest.fixture(autouse=True)
def temp_log(tmp_path, monkeypatch):
    """Send failure logs to a temp folder, not the real logs/."""
    path = tmp_path / "failures.jsonl"
    monkeypatch.setattr(triage_graph, "LOG_PATH", path)
    return path


# ---------- Schema ----------
def test_schema_accepts_valid_summary():
    assert GOOD.area == "gpu" and GOOD.kind == "bug"


def test_schema_rejects_unknown_area():
    with pytest.raises(ValidationError):
        TriageSummary(pain_point="x", area="sourdough", kind="bug", key_terms=["x"])


def test_schema_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        TriageSummary(pain_point="x", area="gpu", kind="complaint", key_terms=["x"])


# ---------- Retry loop ----------
def test_retry_then_success():
    chain = FakeChain(fail_times=1)
    out = run(chain, FakeRetriever([0.9]))
    assert chain.calls == 2
    assert out["attempts"] == 2
    assert out["route"] == "suggest_related"


def test_give_up_after_three_failures(temp_log):
    chain = FakeChain(fail_times=99)
    retriever = FakeRetriever([0.9])
    out = run(chain, retriever)
    assert out["attempts"] == 3
    assert out["route"] == "flag_new"
    assert retriever.calls == 0          # retrieve is skipped on give_up
    record = json.loads(temp_log.read_text().strip())
    assert record["attempts"] == 3       # failure was logged


# ---------- Routing ----------
def test_high_score_suggests_related():
    out = run(FakeChain(), FakeRetriever([0.85, 0.70, 0.40]))
    assert out["route"] == "suggest_related"
    assert "#100" in out["reply"] and "#101" in out["reply"]
    assert "#102" not in out["reply"]    # below threshold, hidden


def test_low_score_flags_new():
    out = run(FakeChain(), FakeRetriever([0.44]))
    assert out["route"] == "flag_new"
    assert "below threshold" in out["reply"]