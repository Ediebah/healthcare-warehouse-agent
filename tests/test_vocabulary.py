"""Tests for the condition-vocabulary resolver (agent/vocabulary.py).

Pure extraction logic runs with no dependencies; the resolution tests query the committed demo
warehouse and are skipped if it isn't reachable (e.g. a source-only checkout)."""
import pytest

from agent import vocabulary
from agent.vocabulary import _candidates, _refine_phrase, resolve

_HAVE_DB = bool(vocabulary._condition_index(""))
_needs_db = pytest.mark.skipif(not _HAVE_DB, reason="demo warehouse not available")


# ── pure extraction logic (no warehouse) ────────────────────────────────────────────────────────

def test_refine_phrase_trims_analysis_words():
    assert _refine_phrase("risk factors for") is None          # all generic → not a condition
    assert _refine_phrase("for heart attack") == "heart attack"  # drops the connective, keeps the term
    assert _refine_phrase("the flu") == "flu"
    assert _refine_phrase("severe pneumonia") == "pneumonia"    # trims generic 'severe'
    assert _refine_phrase("heart") is None                     # bare anatomy is not a condition
    assert _refine_phrase("how does") is None


def test_candidates_finds_lay_synonym_as_condition():
    cands = _candidates("How does survival differ for heart attack patients?")
    terms = {t for t, _, _ in cands}
    assert "heart attack" in terms
    # the synonym key is flagged is_condition=True (so a zero-match would be reported honestly)
    assert any(t == "heart attack" and is_cond for t, _, is_cond in cands)


def test_candidates_does_not_flag_plain_question():
    # a question with no condition must yield nothing that could be declared "absent"
    cands = _candidates("What are the strongest risk factors for patient mortality?")
    assert all(not is_cond for _, _, is_cond in cands) or cands == []


def test_resolve_is_noop_for_byod():
    # Bring-Your-Own-Data (catalog is not None) has no dim_condition to resolve against
    res = resolve("survival for heart attack patients", catalog={"tables": []})
    assert not res.has_grounding and not res.blocked


# ── resolution against the demo warehouse ───────────────────────────────────────────────────────

@_needs_db
def test_maps_heart_attack_to_myocardial_infarction():
    res = resolve("How does survival differ for heart attack patients?")
    assert res.matched and not res.blocked
    m = res.matched[0]
    assert m.term == "heart attack" and m.pattern == "myocardial infarction"
    assert m.descriptions and m.patients_upper > 0


@_needs_db
def test_maps_copd_abbreviation():
    res = resolve("What is the total encounter cost for COPD patients?")
    assert any("obstructive" in m.pattern for m in res.matched)


@_needs_db
def test_absent_condition_blocks_with_honest_clarification():
    res = resolve("How many patients with the flu were admitted?")
    assert res.blocked and res.absent == ["flu"] and not res.matched
    msg = res.clarification()
    assert "flu" in msg.lower() and res.suggestions          # names the term + offers alternatives


@_needs_db
def test_plain_question_gets_no_grounding_and_is_not_blocked():
    res = resolve("How does patient survival differ by sex?")
    assert not res.matched and not res.blocked and not res.absent


@_needs_db
def test_grounding_block_gives_a_real_ilike_pattern():
    block = resolve("survival for heart attack patients").grounding_block()
    assert "ILIKE" in block and "%myocardial infarction%" in block


@_needs_db
def test_heal_hint_cites_existing_patterns():
    hint = resolve("cost for COPD patients").heal_hint()
    assert "0 rows" in hint and "obstructive" in hint


@_needs_db
def test_nested_condition_is_deduplicated():
    # "chronic kidney disease" must not ALSO list the broader bare-"kidney" match
    res = resolve("What predicts mortality in patients with chronic kidney disease?")
    assert len(res.matched) == 1 and "kidney" in res.matched[0].pattern


@_needs_db
def test_partial_absence_notes_flu_but_still_analyzes_diabetes():
    res = resolve("Compare outcomes for diabetes vs flu patients")
    assert any("diabetes" in m.pattern for m in res.matched)   # diabetes resolves
    assert "flu" in res.absent and not res.blocked             # flu noted, not a hard block
