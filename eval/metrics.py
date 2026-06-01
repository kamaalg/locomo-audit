"""Reference scoring metrics for PersonalMem.

Canonical, third-party-agreeable scoring the paper cites. benchmark-designer owns
*which* metric each category uses (docs/benchmark-spec.md); this module owns *how*
each is computed. Contract: docs/metrics-requirements.md (Wave 2).

Design notes:
- The skeleton primitives (`normalize_answer`, `exact_match`, `token_f1`,
  `numeric_within_tolerance`) keep their plain-float signatures — tests and the
  runner depend on them.
- Richer category metrics return `(score, failure_mode, extra)` so the runner can
  populate per-cell diagnostics (spec §5).
- Heavy deps (scikit-learn, rapidfuzz) are imported lazily with pure-python
  fallbacks so this module imports cleanly in a minimal environment.
"""

from __future__ import annotations

import math
import re
import string
from collections.abc import Callable
from typing import Any

# --------------------------------------------------------------------------- #
# Primitives (Wave 0 — kept exactly; do not change signatures)
# --------------------------------------------------------------------------- #


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD-style)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalized strings match exactly, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 over normalized answers."""
    pred_toks = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common: dict[str, int] = {}
    for t in pred_toks:
        if t in gold_toks:
            common[t] = common.get(t, 0) + 1
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def numeric_within_tolerance(prediction: float, gold: float, rel_tol: float = 0.05) -> float:
    """1.0 if within rel_tol of gold, else 0.0."""
    if gold == 0:
        return float(abs(prediction) <= rel_tol)
    return float(abs(prediction - gold) / abs(gold) <= rel_tol)


# --------------------------------------------------------------------------- #
# T1 / T6 — date answers
# --------------------------------------------------------------------------- #


def date_within_tolerance(
    pred_date: str, gold_date: str, tolerance_days: int = 0
) -> tuple[float, str | None, dict]:
    """1.0 if |pred - gold| <= tolerance_days. Granularity-aware.

    Accepts loose formats ('Sept 2023', '2023-09', '09/2023'). Unparseable -> 0.0
    with failure_mode 'unparseable_date'. If gold is month-precision, compares at
    month granularity (tolerance widened to that month).
    """
    from datetime import date, datetime

    try:
        from dateutil import parser as _dtparser
    except Exception:  # pragma: no cover - dateutil is a declared dep
        return 0.0, "unparseable_date", {"reason": "dateutil missing"}

    def _parse(s: str) -> tuple[date | None, str]:
        s = s.strip()
        # crude precision sniff: a bare year or year-month is low precision
        if re.fullmatch(r"\d{4}", s):
            return date(int(s), 1, 1), "year"
        has_ym = re.search(r"\b(\d{4}[-/]\d{1,2}|[A-Za-z]+\s+\d{4})\b", s)
        has_full = re.search(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", s)
        precision = "month" if has_ym and not has_full else "day"
        try:
            return _dtparser.parse(s, default=datetime(2000, 1, 1)).date(), precision
        except (ValueError, OverflowError):
            return None, "day"

    pred, _ = _parse(pred_date)
    gold, gold_prec = _parse(gold_date)
    if pred is None or gold is None:
        return 0.0, "unparseable_date", {}

    if gold_prec in {"year", "month"}:
        # compare at the coarser granularity
        if gold_prec == "year":
            return (float(pred.year == gold.year), None if pred.year == gold.year else "wrong_time", {})
        same_month = (pred.year, pred.month) == (gold.year, gold.month)
        return (float(same_month), None if same_month else "wrong_time", {})

    delta = abs((pred - gold).days)
    ok = delta <= tolerance_days
    return (float(ok), None if ok else "wrong_time", {"delta_days": delta})


# --------------------------------------------------------------------------- #
# T2 — cross-document synthesis
# --------------------------------------------------------------------------- #

JudgeFn = Callable[[str, str], bool]  # (prediction, claim) -> entailed?


def claim_recall(
    prediction: str,
    claim_set: list[str],
    theta: float = 0.5,
    judge: JudgeFn | None = None,
) -> tuple[float, str | None, dict]:
    """Fraction of gold atomic claims entailed by the prediction.

    Per claim: entailed if token_f1(prediction, claim) >= theta. Claims in the
    ambiguous band (0.3, 0.6) defer to `judge` (3-judge majority entailment per
    eval/judge-prompts/t2_claim_entailment.md) when one is supplied and its
    calibration gate has passed; otherwise treated as not-entailed (conservative).
    """
    per_claim: list[dict] = []
    judged: list[str] = []
    entailed = 0
    for claim in claim_set:
        f1 = _best_window_f1(prediction, claim)
        if 0.3 < f1 < 0.6 and judge is not None:
            is_ent = bool(judge(prediction, claim))
            judged.append(claim)
        else:
            is_ent = f1 >= theta
        entailed += int(is_ent)
        per_claim.append({"claim": claim, "f1": round(f1, 4), "entailed": is_ent})
    recall = entailed / len(claim_set) if claim_set else 0.0
    fm = None if recall >= 0.999 else ("partial_synthesis" if recall > 0 else "retrieval_miss")
    return recall, fm, {"per_claim": per_claim, "judged_claims": judged}


def _best_window_f1(prediction: str, claim: str) -> float:
    """Max token-F1 of `claim` against any same-length-ish window of `prediction`.

    Matching a claim against the whole prediction dilutes precision and sinks every
    claim into the conservative ambiguous band; a sliding window restores the
    intended "is this claim locally entailed?" semantics (spec §T2).
    """
    pred_toks = normalize_answer(prediction).split()
    claim_toks = normalize_answer(claim).split()
    if not claim_toks or not pred_toks:
        return token_f1(prediction, claim)
    best = 0.0
    for w in {len(claim_toks), len(claim_toks) + 1, len(claim_toks) + 2}:
        for i in range(0, max(len(pred_toks) - w + 1, 1)):
            window = " ".join(pred_toks[i : i + w])
            best = max(best, token_f1(window, claim))
            if best == 1.0:
                return 1.0
    return best


def supporting_doc_recall(pred_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    """|pred ∩ gold| / |gold|. Separates retrieval failure from synthesis failure."""
    gold = set(gold_doc_ids)
    if not gold:
        return 1.0
    return len(set(pred_doc_ids) & gold) / len(gold)


# --------------------------------------------------------------------------- #
# T3 — style consistency (held-out, NON-LLM stylometric classifier)
# --------------------------------------------------------------------------- #


def _style_features(text: str) -> list[float]:
    """Topic-stripped stylometric feature vector.

    Function-word rates, sentence-length moments, punctuation/casing/emoji rates.
    Deliberately avoids content words so the classifier keys on style, not topic.
    """
    fn_words = (
        "the a an and or but if then of to in on at for with as is are was were be "
        "this that these those i you he she it we they not no so very just"
    ).split()
    toks = re.findall(r"\w+|[^\w\s]", text)
    words = [t for t in toks if t.isalpha()]
    n = max(len(words), 1)
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    slens = [len(s.split()) for s in sents] or [0]
    mean_sl = sum(slens) / len(slens)
    var_sl = sum((x - mean_sl) ** 2 for x in slens) / len(slens)
    emoji = len(re.findall(r"[\U0001F300-\U0001FAFF☀-➿]", text))
    lowered = [w.lower() for w in words]
    feats = [lowered.count(fw) / n for fw in fn_words]
    feats += [
        mean_sl,
        math.sqrt(var_sl),
        text.count(",") / max(len(text), 1),
        text.count("!") / max(len(text), 1),
        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        emoji / max(len(text), 1),
    ]
    return feats


class StyleClassifier:
    """Held-out one-vs-rest stylometric classifier (logistic regression).

    Trained on a persona's style split — documents WITHHELD from the systems —
    so a system cannot win T3 by copying. Requires scikit-learn (lazy import);
    raises a clear error if unavailable.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._persona_ids: list[str] = []

    def _require_sklearn(self):
        try:
            from sklearn.linear_model import LogisticRegression  # noqa: F401
            from sklearn.metrics import roc_auc_score  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "StyleClassifier needs scikit-learn. `pip install scikit-learn`."
            ) from e

    def fit(self, persona_docs: dict[str, list[str]]) -> None:
        self._require_sklearn()
        from sklearn.linear_model import LogisticRegression

        self._persona_ids = list(persona_docs)
        all_items = [(pid, doc) for pid, docs in persona_docs.items() for doc in docs]
        X = [_style_features(doc) for _, doc in all_items]
        for pid in self._persona_ids:
            y = [int(p == pid) for p, _ in all_items]
            if sum(y) == 0 or sum(y) == len(y):
                continue
            self._models[pid] = LogisticRegression(max_iter=1000).fit(X, y)

    def prob_persona(self, text: str, persona_id: str) -> float:
        model = self._models.get(persona_id)
        if model is None:
            return 0.0
        return float(model.predict_proba([_style_features(text)])[0][1])

    def auc(self, samples: list[tuple[str, str, int]]) -> float:
        self._require_sklearn()
        from sklearn.metrics import roc_auc_score

        scores = [self.prob_persona(t, pid) for t, pid, _ in samples]
        labels = [lbl for _, _, lbl in samples]
        if len(set(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))


def style_consistency(
    prediction: str,
    persona_id: str,
    classifier: StyleClassifier,
    reference_doc_texts: list[str],
    adequacy_keywords: list[str],
    verbatim_cap: float = 0.4,
) -> tuple[float, str | None, dict]:
    """Adequacy gate -> verbatim-copy guard -> stylometric persona probability."""
    if not _adequate(prediction, adequacy_keywords):
        return 0.0, "task_fail", {}
    overlap = max((_char_ngram_overlap(prediction, ref) for ref in reference_doc_texts), default=0.0)
    if overlap > verbatim_cap:
        return 0.0, "verbatim_copy", {"overlap": round(overlap, 3)}
    score = classifier.prob_persona(prediction, persona_id)
    return score, None, {"overlap": round(overlap, 3)}


def _char_ngram_overlap(a: str, b: str, n: int = 5) -> float:
    def grams(s: str) -> set[str]:
        s = re.sub(r"\s+", " ", s.lower())
        return {s[i : i + n] for i in range(max(len(s) - n + 1, 0))}
    ga, gb = grams(a), grams(b)
    if not ga:
        return 0.0
    return len(ga & gb) / len(ga)


def _adequate(text: str, keywords: list[str], threshold: float = 0.25) -> bool:
    if not text.strip():
        return False
    if not keywords:
        return True
    norm = normalize_answer(text)
    hits = sum(1 for k in keywords if normalize_answer(k) in norm)
    return hits / len(keywords) >= threshold


# --------------------------------------------------------------------------- #
# T4 / T5 — FAMA (adopted from Memora, arXiv:2604.20006 — DO NOT reinvent)
# --------------------------------------------------------------------------- #


def classify_update_answer(
    prediction: str,
    current_answer: str,
    obsolete_values: list[str],
    answer_type: str = "text",
) -> str:
    """Label an answer as 'correct_current' | 'stale' | 'wrong'."""
    if answer_type == "number":
        try:
            p = float(re.sub(r"[^\d.\-]", "", prediction))
        except ValueError:
            return "wrong"
        if numeric_within_tolerance(p, float(current_answer)):
            return "correct_current"
        if any(numeric_within_tolerance(p, float(o)) for o in obsolete_values):
            return "stale"
        return "wrong"
    if exact_match(prediction, current_answer):
        return "correct_current"
    if any(exact_match(prediction, o) for o in obsolete_values):
        return "stale"
    return "wrong"


def fama(labels: list[str], lam: float = 1.0) -> dict:
    """Forgetting-Aware Memory Accuracy (Memora, arXiv:2604.20006).

    FAMA = (#correct_current - lam * #stale) / N. Report lam=1.0 (primary) and
    lam=0.5 (robustness). `stale_rate` is the headline forgetting diagnostic.
    """
    n = len(labels) or 1
    cc = labels.count("correct_current")
    st = labels.count("stale")
    return {
        "fama_1.0": (cc - 1.0 * st) / n,
        "fama_0.5": (cc - 0.5 * st) / n,
        "stale_rate": st / n,
        "accuracy": cc / n,
    }


# --------------------------------------------------------------------------- #
# T5 / T7 — multiple choice
# --------------------------------------------------------------------------- #


def parse_choice(answer: str, choices: dict[str, str]) -> str | None:
    """Map a free-text answer to a choice letter; None if unparseable."""
    a = answer.strip()
    m = re.match(r"\s*\(?([A-Za-z])\)?[\).:\s]", a) or re.fullmatch(r"\s*\(?([A-Za-z])\)?\s*", a)
    if m and m.group(1).upper() in {k.upper() for k in choices}:
        return m.group(1).upper()
    # fall back to best text overlap with a choice value
    best, best_f1 = None, 0.0
    for letter, text in choices.items():
        f1 = token_f1(answer, text)
        if f1 > best_f1:
            best, best_f1 = letter.upper(), f1
    return best if best_f1 >= 0.5 else None


def multiple_choice_accuracy(pred_choice: str, gold_choice: str) -> float:
    """1.0 if normalized choice letters match."""
    return float(pred_choice.strip().upper() == gold_choice.strip().upper())


# --------------------------------------------------------------------------- #
# T7 — bucketed implicit inference
# --------------------------------------------------------------------------- #


def bucketed_inference_score(
    pred_choice: str,
    gold_choice: str,
    ordered_buckets: list[str],
    abstain_choice: str = "E",
) -> tuple[float, str | None, dict]:
    """Exact credit + ordinal partial credit (0.5 for one bucket off).

    The abstain bucket is not ordinal — no partial credit between a value and
    abstain. Abstention-correctness is tracked separately in `extra`.
    """
    pred, gold = pred_choice.strip().upper(), gold_choice.strip().upper()
    ab = abstain_choice.upper()
    extra = {
        "abstain_correct": (pred == ab) == (gold == ab),
        "pred_abstain": pred == ab,
        "gold_abstain": gold == ab,
    }
    if pred == gold:
        return 1.0, None, extra
    if gold == ab and pred != ab:
        return 0.0, "over_inference", extra
    if pred == ab and gold != ab:
        return 0.0, "no_inference", extra
    order = [b.upper() for b in ordered_buckets]
    if pred in order and gold in order:
        dist = abs(order.index(pred) - order.index(gold))
        if dist == 1:
            return 0.5, None, extra
        return 0.0, "wrong_direction", extra
    return 0.0, "wrong_direction", extra


# --------------------------------------------------------------------------- #
# T8 — selective surfacing (PROGRAMMATIC withhold-check, never an LLM judge)
# --------------------------------------------------------------------------- #


def _fuzzy_ratio(a: str, b: str) -> float:
    try:
        from rapidfuzz.fuzz import token_set_ratio

        return token_set_ratio(a, b) / 100.0
    except Exception:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio()


def selective_surfacing(
    prediction: str,
    must_not_mention: list[str],
    should_help: bool,
    adequacy_keywords: list[str],
    fuzzy_threshold: float = 0.85,
) -> tuple[float, str | None, dict]:
    """Deterministic leak check + help-adequacy gate.

    score = 1.0 iff (no leak) AND (adequacy satisfied when should_help). The alias
    list `must_not_mention` is frozen + human-validated at dataset-build time; this
    function never calls an LLM.
    """
    norm_pred = normalize_answer(prediction)
    leaked: list[str] = []
    exact_leak = False
    for item in must_not_mention:
        ni = normalize_answer(item)
        if not ni:
            continue
        if ni in norm_pred:
            leaked.append(item)
            exact_leak = True
        elif _fuzzy_ratio(ni, norm_pred) >= fuzzy_threshold:
            leaked.append(item)
    if leaked:
        # exact substring match -> hard leak; only-fuzzy match -> partial_leak
        fm = "leak" if exact_leak else "partial_leak"
        return 0.0, fm, {"leaked_items": leaked, "over_suppress": False}
    if should_help and not _adequate(prediction, adequacy_keywords):
        return 0.0, "over_suppress" if not prediction.strip() else "irrelevant", {
            "leaked_items": [],
            "over_suppress": not prediction.strip(),
        }
    return 1.0, None, {"leaked_items": [], "over_suppress": False}


def leak_rate(results: list[tuple[float, str | None, dict]]) -> float:
    """#leak / N — the privacy headline."""
    n = len(results) or 1
    leaks = sum(1 for _, fm, _ in results if fm in {"leak", "partial_leak"})
    return leaks / n


def pndr(results: list[tuple[float, str | None, dict]]) -> float:
    """Privacy Non-Disclosure Rate = 1 - leak_rate."""
    return 1.0 - leak_rate(results)
