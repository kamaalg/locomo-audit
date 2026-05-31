"""Tests for the v0.1 category metrics (docs/metrics-requirements.md).

StyleClassifier needs scikit-learn; that test is skipped if unavailable so the
suite stays green in a minimal env. Everything else is pure-python.
"""

import pytest

from eval.metrics import (
    bucketed_inference_score,
    claim_recall,
    classify_update_answer,
    date_within_tolerance,
    fama,
    leak_rate,
    multiple_choice_accuracy,
    parse_choice,
    pndr,
    selective_surfacing,
    supporting_doc_recall,
)


# --- T1/T6 dates ---

def test_date_within_tolerance_exact_and_window():
    assert date_within_tolerance("2023-09-10", "2023-09-10")[0] == 1.0
    assert date_within_tolerance("2023-09-12", "2023-09-10", tolerance_days=3)[0] == 1.0
    assert date_within_tolerance("2023-09-20", "2023-09-10", tolerance_days=3)[0] == 0.0


def test_date_granularity_month_and_year():
    # gold is month-precision -> any day in that month counts
    assert date_within_tolerance("2023-09-28", "Sept 2023")[0] == 1.0
    assert date_within_tolerance("2023-10-01", "Sept 2023")[0] == 0.0
    # gold is year-precision
    assert date_within_tolerance("2019-06-15", "2019")[0] == 1.0


def test_date_unparseable():
    score, fm, _ = date_within_tolerance("not a date", "2023-09-10")
    assert score == 0.0 and fm == "unparseable_date"


# --- T2 ---

def test_claim_recall_partial():
    pred = "I moved to Berlin and started a new job as a teacher."
    claims = ["moved to Berlin", "started a new job", "bought a car"]
    recall, fm, extra = claim_recall(pred, claims)
    assert 0.0 < recall < 1.0 and fm == "partial_synthesis"
    assert len(extra["per_claim"]) == 3


def test_supporting_doc_recall():
    assert supporting_doc_recall(["d1", "d2"], ["d1", "d2", "d3"]) == pytest.approx(2 / 3)
    assert supporting_doc_recall([], []) == 1.0


# --- T4/T5 FAMA ---

def test_classify_update_answer():
    assert classify_update_answer("Berlin", "Berlin", ["Paris"]) == "correct_current"
    assert classify_update_answer("Paris", "Berlin", ["Paris"]) == "stale"
    assert classify_update_answer("Rome", "Berlin", ["Paris"]) == "wrong"


def test_fama_penalizes_stale():
    labels = ["correct_current", "correct_current", "stale", "wrong"]
    out = fama(labels)
    assert out["accuracy"] == 0.5
    assert out["stale_rate"] == 0.25
    assert out["fama_1.0"] == (2 - 1) / 4
    assert out["fama_0.5"] == (2 - 0.5) / 4


# --- T5/T7 multiple choice ---

def test_parse_choice_letter_and_text():
    choices = {"A": "coffee", "B": "tea", "C": "unresolved"}
    assert parse_choice("B) tea", choices) == "B"
    assert parse_choice("the answer is tea", choices) == "B"
    assert parse_choice("totally unrelated zzz", choices) is None


def test_multiple_choice_accuracy():
    assert multiple_choice_accuracy("a", "A") == 1.0
    assert multiple_choice_accuracy("B", "A") == 0.0


# --- T7 ordinal ---

def test_bucketed_inference_ordinal_credit():
    buckets = ["A", "B", "C", "D"]  # ordered; E = abstain
    assert bucketed_inference_score("B", "B", buckets)[0] == 1.0
    assert bucketed_inference_score("C", "B", buckets)[0] == 0.5  # one off
    s, fm, _ = bucketed_inference_score("A", "D", buckets)
    assert s == 0.0 and fm == "wrong_direction"


def test_bucketed_inference_abstain():
    buckets = ["A", "B", "C", "D"]
    s, fm, extra = bucketed_inference_score("B", "E", buckets)  # over-inference
    assert s == 0.0 and fm == "over_inference" and extra["gold_abstain"]
    s, fm, _ = bucketed_inference_score("E", "B", buckets)  # no inference
    assert s == 0.0 and fm == "no_inference"


# --- T8 selective surfacing ---

def test_selective_surfacing_leak():
    s, fm, extra = selective_surfacing(
        "Good luck! Remember you've been anxious lately.",
        must_not_mention=["anxious", "sertraline"],
        should_help=True,
        adequacy_keywords=["good", "luck", "interview"],
    )
    assert s == 0.0 and fm == "leak" and "anxious" in extra["leaked_items"]


def test_selective_surfacing_clean_and_helpful():
    s, fm, _ = selective_surfacing(
        "Good luck at your interview today — you've prepared well!",
        must_not_mention=["anxious", "sertraline"],
        should_help=True,
        adequacy_keywords=["good", "luck", "interview"],
    )
    assert s == 1.0 and fm is None


def test_leak_rate_and_pndr():
    results = [
        (0.0, "leak", {}),
        (1.0, None, {}),
        (0.0, "partial_leak", {}),
        (1.0, None, {}),
    ]
    assert leak_rate(results) == 0.5
    assert pndr(results) == 0.5


# --- T3 (skipped without scikit-learn) ---

def test_style_classifier_keys_on_style_not_topic():
    pytest.importorskip("sklearn")
    from eval.metrics import StyleClassifier

    terse = ["Done. Shipped it. Moving on.", "Quick note. Fixed bug. Out.", "Yep. Works. Next."]
    verbose = [
        "I really wanted to take a moment to reflect on how the day unfolded, because honestly it was quite something.",
        "There is so much I could say about this, and I think it deserves a thorough and careful walkthrough indeed.",
        "Let me explain in detail, since the nuances here genuinely matter a great deal to me personally.",
    ]
    clf = StyleClassifier()
    clf.fit({"terse": terse, "verbose": verbose})
    # a new terse-style line should score higher for the terse persona
    assert clf.prob_persona("Done. Fixed. Bye.", "terse") > clf.prob_persona(
        "Done. Fixed. Bye.", "verbose"
    )
