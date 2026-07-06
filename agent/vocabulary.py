"""Condition-vocabulary resolver — grounds a plain-English condition name in the warehouse's
actual SNOMED descriptions BEFORE the model writes SQL.

Why this exists: the warehouse stores clinical descriptions like `Myocardial infarction (disorder)`
and `Essential hypertension (disorder)`, not the words people type. A user asking about a "heart
attack" or "COPD" would otherwise get `ILIKE '%heart attack%'` → 0 rows, and the agent would
silently analyze an empty cohort. This module maps the lay term to the real descriptions (via a
small, warehouse-grounded synonym map plus direct substring probing), so the grounding context can
tell the model the exact `condition_description` values — and the ILIKE pattern — to filter on.

It is deterministic (no LLM, no network) and touches NO user input to SQL: the full condition index
(≈250 rows) is loaded once and all matching happens in Python, so there is zero injection surface.
Demo-warehouse only — Bring-Your-Own-Data uploads have their own columns and no dim_condition.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .warehouse import QueryError, run_query

# ── lay / abbreviated term → clinical substrings that appear in this warehouse ──────────────────
# Every target below was verified against dim_condition.condition_description. A term whose targets
# match nothing (e.g. "flu") is still useful: it lets the resolver report the absence honestly
# instead of returning a silently-empty cohort. Keys are matched on word boundaries (see _CANDIDATE).
SYNONYMS: dict[str, list[str]] = {
    # cardiovascular
    "heart attack": ["myocardial infarction"],
    "myocardial infarction": ["myocardial infarction"],
    "stemi": ["st segment elevation myocardial infarction"],
    "coronary artery disease": ["ischemic heart disease"],   # bare "coronary" only hits bypass/imaging noise
    "coronary heart disease": ["ischemic heart disease"],
    "cad": ["ischemic heart disease"],
    "heart failure": ["heart failure"],
    "congestive heart failure": ["congestive heart failure"],
    "chf": ["congestive heart failure"],
    "high blood pressure": ["hypertension"],
    "hypertensive": ["hypertension"],
    "high cholesterol": ["hyperlipidemia"],
    "atrial fibrillation": ["atrial fibrillation"],
    "afib": ["atrial fibrillation"],
    "stroke": ["cerebrovascular"],
    "cerebrovascular accident": ["cerebrovascular"],
    # respiratory
    "copd": ["chronic obstructive bronchitis", "emphysema"],
    "chronic obstructive pulmonary disease": ["chronic obstructive bronchitis", "emphysema"],
    "emphysema": ["emphysema"],
    "asthma": ["asthma"],
    # metabolic / endocrine
    "diabetic": ["diabetes"],
    "diabetics": ["diabetes"],
    "type 2 diabetes": ["type 2 diabetes", "type ii diabetes"],
    "type ii diabetes": ["type ii diabetes", "type 2 diabetes"],
    "t2dm": ["type 2 diabetes", "type ii diabetes"],
    "obese": ["obesity"],
    "obesity": ["obesity"],
    # renal
    "chronic kidney disease": ["chronic kidney disease", "chronic kidney"],
    "ckd": ["chronic kidney disease", "chronic kidney"],
    "kidney disease": ["kidney", "renal"],
    "renal failure": ["renal failure", "end-stage renal"],
    "kidney failure": ["renal failure", "end-stage renal", "chronic kidney"],  # else bare "failure" → wrong cohort
    "esrd": ["end-stage renal"],
    # specific arthritis (longest-first, so it wins over the broad "arthritis" key and isn't widened to osteo)
    "rheumatoid arthritis": ["rheumatoid"],
    "rheumatoid": ["rheumatoid"],
    # lay terms for diabetes with no shared substring
    "high blood sugar": ["diabetes"],
    "sugar disease": ["diabetes"],
    # mental health
    "depression": ["depressive"],
    "depressed": ["depressive"],
    "anxiety": ["anxiety"],
    # infectious
    "covid": ["coronavirus"],
    "covid-19": ["coronavirus"],
    "covid19": ["coronavirus"],
    "coronavirus": ["coronavirus"],
    "flu": ["influenza"],           # not present → resolver reports it honestly
    "influenza": ["influenza"],
    # musculoskeletal
    "osteoarthritis": ["osteoarthritis"],
    # "arthritis" is a whole word in "rheumatoid arthritis" but a SUFFIX inside "osteoarthritis"
    # (one word, no boundary), so list both patterns — the union catches every arthritis row.
    "arthritis": ["arthritis", "osteoarthritis"],
    "arthritic": ["arthritis", "osteoarthritis"],
    "osteoporosis": ["osteoporosis"],
    # demonym / adjectival surface forms the warehouse doesn't store literally (it has "asthma",
    # not "asthmatics") — mapped so "among asthmatics" / "anemic patients" still resolve.
    "asthmatic": ["asthma"],
    "asthmatics": ["asthma"],
    "anemic": ["anemia"],
    "anemics": ["anemia"],
    "hypertensives": ["hypertension"],
    "epileptic": ["epilepsy"],
    "epileptics": ["epilepsy"],
    "psoriatic": ["psoriasis"],
    "cirrhotic": ["cirrhosis"],
}

# Clinical abbreviations → real warehouse substrings. Matched CASE-SENSITIVELY (see _candidates) so
# the lowercase words "mi"/"dm"/"af" in ordinary prose don't trip them. A target that isn't present
# (e.g. TIA) simply yields no match — never a fabricated filter.
_ABBREV: dict[str, list[str]] = {
    "MI": ["myocardial infarction"], "HTN": ["hypertension"], "HBP": ["hypertension"],
    "DM": ["diabetes"], "T1DM": ["type 1 diabetes", "diabetes mellitus type 1"],
    "T2DM": ["type 2 diabetes", "type ii diabetes", "diabetes mellitus type 2"],
    "HLD": ["hyperlipidemia"], "CVA": ["cerebrovascular"], "TIA": ["transient ischemic"],
    "IHD": ["ischemic heart disease"], "CAD": ["ischemic heart disease"],
    "CVD": ["cardiovascular", "ischemic heart disease"], "HF": ["heart failure"],
    "CHF": ["congestive heart failure"], "OA": ["osteoarthritis"], "RA": ["rheumatoid"],
    "AF": ["atrial fibrillation"], "AFIB": ["atrial fibrillation"], "CKD": ["chronic kidney"],
    "ESRD": ["end-stage renal"], "COPD": ["chronic obstructive bronchitis", "emphysema"],
    "OSA": ["obstructive sleep apnea", "sleep apnea"], "GERD": ["reflux"], "UTI": ["urinary tract infection"],
    "DVT": ["deep vein", "venous thrombosis"], "PAD": ["peripheral arterial", "peripheral vascular"],
}

# Generic analysis / demographic words that are never a specific condition filter. A bare content
# word (below) is only treated as a candidate condition if it is NOT here AND it actually matches a
# real description, so these keep questions like "which chronic conditions drive cost?" from
# resolving to a spurious filter.
_STOPWORDS = frozenset("""
patient patients survival survive mortality death deaths deceased die dying readmission readmit
readmitted encounter encounters cost costs spend spending average mean median rate rates ratio
prevalence proportion percentage percent count counts number total totals sum risk risks factor
factors predict predicts predictor driving drive drives adjust adjusting adjusted among amongst
differ differs difference differences between compare compared comparison versus group groups grouped
grouping stratify age ages aged sex sexes gender genders race races ethnicity income insurance
insured coverage payer effect effects impact analyze analyzed analysis month months monthly year
years yearly quarter volume volumes forecast trend trends trial trials arm arms power sample samples
size sizes enroll condition conditions disease diseases disorder disorders diagnosis diagnoses
chronic acute severe mild highest lowest higher lower most least common commonly across overall each
every their there these those which what does have having with without people adults individuals
subjects cases cohort clinical medical health healthcare significant strong strongest weak level
levels stage stages type types class classes new standard care margin cure outcome outcomes endpoint
endpoints
history symptom symptoms cause causes caused season seasonal control force forces first primary
complication complications complete record records report reports using review reviews status social
employment employed unemployed education educated housing activity activities behavior behaviour
access medication medications daily occur occurs waiting delivery labor labour order table index
limit limits abuse violence isolation income poverty employment lifestyle physical mental social
determinant determinants environment environmental follow followup adherence time times cell cells
stress crisis related overall main track number associated
""".split())

# Bare anatomy words are excluded from direct-word probing (so "heart" alone doesn't over-match
# "heart failure"/"ischemic heart disease" when the user actually said "heart attack") and a phrase
# made ONLY of them is not a condition. A multi-word term that pairs anatomy with a specifier
# ("heart attack", "heart failure") is kept — those resolve through the synonym map.
_ANATOMY = frozenset("""
heart blood lung lungs kidney kidneys liver brain bone bones skin chest joint joints muscle nerve
nerves back neck head eye eyes ear ears tooth teeth throat body organ
""".split())

# Interrogatives, auxiliaries and connectives that a loose phrase capture can pick up. A phrase made
# only of these (after dropping generic _STOPWORDS) is never a condition, so it is discarded rather
# than mis-reported as an absent condition.
_FUNCTION_WORDS = frozenset("""
how does do did is are was were be been being has have had will would can could should may might
what which who whom whose when where why the a an and or but nor for of to in on at by with from
into onto over under about above below between many much more less than then this that them they
their our your his her its it as if so such just only also very much per vs
""".split())

# "patients with X" / "adults diagnosed with X" — capture the up-to-4 words after the connective;
# _refine_phrase then trims them to a clean condition phrase (or discards it).
_WITH_PAT = re.compile(
    r"\b(?:patient|patients|adult|adults|people|person|persons|individual|individuals|subject|"
    r"subjects|case|cases|those|anyone|everyone|cohort)\s+"
    r"(?:with|who\s+have|who\s+had|having|diagnosed\s+with|suffering\s+from|presenting\s+with)\s+"
    r"((?:[a-z0-9\-]+\s+){0,3}[a-z0-9\-]+)", re.I)
# "X patients" / "X cohort" — capture the up-to-3 words immediately before the noun (e.g. "heart
# attack patients"); _refine_phrase trims connectives like a leading "for".
_PRE_PAT = re.compile(
    r"\b((?:[a-z0-9\-]+\s+){0,2}[a-z0-9\-]+)\s+(?:patient|patients|cohort|cases|population)\b", re.I)


@dataclass
class TermMatch:
    """One user-supplied condition term resolved to real warehouse descriptions. `patterns` is the
    set of ILIKE substrings needed to select the cohort — usually one, but ≥1 when a term spans
    disjoint descriptions (e.g. "COPD" → 'chronic obstructive bronchitis' + 'emphysema')."""
    term: str                                   # what the user said, e.g. "heart attack"
    patterns: list[str]                         # ILIKE substrings to OR together, e.g. ["myocardial infarction"]
    descriptions: list[tuple[str, int]] = field(default_factory=list)  # [(description, n_patients)], desc order

    @property
    def pattern(self) -> str:                   # back-compat: the primary pattern
        return self.patterns[0] if self.patterns else ""

    def ilike(self, col: str = "condition_description") -> str:
        """The filter expression to apply — one ILIKE, or several OR'd when the term is disjoint."""
        return " OR ".join(f"{col} ILIKE '%{p}%'" for p in self.patterns)

    @property
    def n_conditions(self) -> int:
        return len(self.descriptions)

    @property
    def patients_upper(self) -> int:
        """Sum of per-condition patient counts — an UPPER BOUND (a patient with two matching
        conditions is counted twice). Labelled as such wherever shown."""
        return sum(n for _, n in self.descriptions)


@dataclass
class Resolution:
    matched: list[TermMatch] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)             # named conditions with 0 warehouse matches
    suggestions: list[tuple[str, int]] = field(default_factory=list)  # top available conditions

    @property
    def has_grounding(self) -> bool:
        return bool(self.matched or self.absent)

    @property
    def blocked(self) -> bool:
        """The user named a condition, ALL of them are absent → analyzing would fit on an empty
        cohort. The caller should ask a clarifying question instead of proceeding."""
        return bool(self.absent) and not self.matched

    def grounding_block(self) -> str:
        """A compact block appended to the LLM grounding context so both the plain-SQL and the
        model (analytic_sql) paths filter on real descriptions."""
        if not self.has_grounding:
            return ""
        lines = ["", "CONDITION VOCABULARY (resolved against this warehouse — filter on these exact "
                 "values; do NOT invent other spellings):"]
        for m in self.matched:
            shown = ", ".join(f"{d} [{n} pts]" for d, n in m.descriptions[:4])
            more = f", +{m.n_conditions - 4} more" if m.n_conditions > 4 else ""
            lines.append(
                f'• "{m.term}" → use  {m.ilike()}  '
                f"(matches {m.n_conditions} condition(s), up to ~{m.patients_upper} patients): {shown}{more}")
        for term in self.absent:
            sug = ", ".join(d for d, _ in self.suggestions[:6])
            lines.append(
                f'• "{term}" → NOT PRESENT in this warehouse — do not fabricate a filter for it. '
                f"Closest available conditions: {sug}.")
        lines.append("Join fct_conditions to dim_condition on condition_code to apply the filter.")
        return "\n".join(lines)

    def heal_hint(self) -> str:
        """Fed back to the model when a query returns 0 rows — names the real ILIKE patterns that
        DO exist so the retry filters on them instead of guessing again."""
        if self.matched:
            opts = "; ".join(f"({m.ilike()}) (for \"{m.term}\")" for m in self.matched)
            return ("The query returned 0 rows. Filter conditions with ILIKE on "
                    "dim_condition.condition_description using a pattern that EXISTS here: "
                    f"{opts}. Join fct_conditions by condition_code — do not filter a *_code column "
                    "by equality.")
        return ("The query returned 0 rows. Re-check the filters: match a clinical name with ILIKE "
                "on a *_description column (not '=' on a *_code column), and use real category "
                "values (gender is 'M'/'F'). If 0 is genuinely correct, return the same query.")

    def clarification(self) -> str:
        """Honest 'that condition isn't here' message when every named condition is absent."""
        terms = " or ".join(f'"{t}"' for t in self.absent)
        sug = ", ".join(d for d, _ in self.suggestions[:8])
        return (f"I couldn't find {terms} among the conditions recorded in this (synthetic) warehouse, "
                f"so I can't analyze that cohort without inventing data. The most common conditions "
                f"available are: {sug}. Want me to run the same analysis for one of those instead?")


@lru_cache(maxsize=4)
def _condition_index(db_key: str) -> tuple[tuple[str, int], ...]:
    """(lowercased description, distinct patients) for every condition, patient-count desc. Loaded
    once per warehouse and cached. Returns () if the warehouse is unreachable so resolve() degrades
    to a no-op rather than ever breaking an analysis."""
    try:
        df = run_query(
            "SELECT lower(c.condition_description) AS d, COUNT(DISTINCT f.patient_id) AS pts "
            "FROM dim_condition c LEFT JOIN fct_conditions f ON f.condition_code = c.condition_code "
            "GROUP BY 1 ORDER BY pts DESC", db_path=None if db_key == "" else db_key)
        return tuple((str(r.d), int(r.pts)) for r in df.itertuples(index=False))
    except (QueryError, Exception):  # noqa: BLE001 — resolution is best-effort; never fatal
        return ()


@lru_cache(maxsize=4096)
def _pat_re(pat: str) -> re.Pattern:
    """A pattern matches a description only as a WHOLE word/phrase (boundaries on both sides), so
    'order' can't hit dis'order', 'cause' can't hit 'caused', and 'index' can't hit 'body mass
    index'-style fragments. This is the single biggest precision fix from the stress test."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(pat.lower()) + r"(?![a-z0-9])")


def _titlecase_desc(d: str) -> str:
    """Present a lowercased index description readably: strip the SNOMED '(disorder)/(finding)' tag
    and capitalize the first letter (clinical proper nouns keep their internal casing loosely)."""
    d = re.sub(r"\s*\((disorder|finding|situation|procedure|morphologic abnormality)\)\s*$", "", d).strip()
    return d[:1].upper() + d[1:] if d else d


def _refine_phrase(p: str) -> str | None:
    """Trim a loosely-captured phrase to a plausible condition name, or None if it isn't one.
    Drops function words (how/does/for/the…), trims leading/trailing generic terms (chronic/risk…),
    and requires 1–3 remaining content words of length ≥3 — so 'risk factors for' → None but
    'heart attack' and 'chronic kidney disease' survive."""
    words = [w for w in re.sub(r"[^a-z0-9 \-]", " ", p.lower()).split() if w not in _FUNCTION_WORDS]
    while words and words[0] in _STOPWORDS:                  # strip leading generic words ("chronic", "severe")
        words.pop(0)
    while words and words[-1] in _STOPWORDS:                 # strip trailing generic words
        words.pop()
    if not 1 <= len(words) <= 3 or any(len(w) < 3 for w in words):
        return None
    if all(w in _ANATOMY for w in words):                    # bare "heart"/"kidney" is not a condition
        return None
    return " ".join(words)


def _candidates(question: str) -> list[tuple[str, list[str], bool]]:
    """Extract (user_term, ilike_patterns, is_condition) from the question. is_condition=True means
    the phrasing clearly names a condition (a synonym key or a 'patients with X' construction), so a
    zero-match should be reported as absent; False means a bare content word, dropped if it matches
    nothing."""
    ql = question.lower()
    out: list[tuple[str, list[str], bool]] = []
    seen: set[str] = set()

    def add(term: str, patterns: list[str], is_condition: bool) -> None:
        key = term.lower()
        if key and key not in seen:
            seen.add(key)
            out.append((term, patterns, is_condition))

    # 1) synonym / abbreviation keys (longest first so "type 2 diabetes" beats "diabetes")
    for key in sorted(SYNONYMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", ql):
            add(key, SYNONYMS[key], True)

    # 1b) clinical abbreviations, matched CASE-SENSITIVELY on the original text ("MI", "HTN", "CKD")
    # so the lowercase word "mi"/"dm" inside ordinary prose can't false-trigger them.
    for abbr, pats in _ABBREV.items():
        if re.search(rf"\b{re.escape(abbr)}\b", question) and abbr.lower() not in seen:
            add(abbr.lower(), pats, True)

    # 2) explicit "patients with X" / "X patients" constructions. These feed POSITIVE grounding only
    # (is_condition=False): free-text phrases are too noisy to reliably declare a condition "absent"
    # from — the reactive zero-row self-heal is the safety net for an unmapped, truly-absent cohort.
    for rx in (_WITH_PAT, _PRE_PAT):
        for m in rx.finditer(ql):
            phrase = _refine_phrase(m.group(1))
            if phrase:
                add(phrase, SYNONYMS.get(phrase, [phrase]), False)

    # 3) bare significant words (catch conditions named directly, e.g. "anemia", "gout"). Floor is 4
    # chars (was 5) so "gout"/"pain"/"sore" resolve; whole-word matching + the stopword/anatomy
    # filters keep that safe. A trailing-'s' plural also probes its singular ("seizures"→"seizure").
    for w in re.findall(r"[a-z][a-z0-9\-]{3,}", ql):
        if w in _STOPWORDS or w in _FUNCTION_WORDS or w in _ANATOMY or w in seen:
            continue
        pats = list(SYNONYMS.get(w, [w]))
        if w.endswith("s") and len(w) > 4:                  # "seizures" → "seizure"
            pats.append(w[:-1])
        if w.endswith("es") and len(w) > 5:                 # "abscesses" → "abscess"
            pats.append(w[:-2])
        add(w, pats, False)
    return out


def resolve(question: str, catalog: dict | None = None, db_path=None) -> Resolution:
    """Resolve condition terms in `question` against the demo warehouse. No-op (empty Resolution)
    for Bring-Your-Own-Data sessions (catalog is not None) — those have their own columns."""
    res = Resolution()
    if catalog is not None:                                  # BYOD → no dim_condition to resolve against
        return res
    index = _condition_index(str(db_path) if db_path else "")
    if not index:
        return res

    covered: list[frozenset[str]] = []                      # description-sets already claimed, for dedup
    for term, patterns, is_condition in _candidates(question):
        # Union every pattern that adds a new description (COPD → bronchitis + emphysema; "arthritis"
        # → osteo + rheumatoid). Keep only patterns that actually contribute, so the ILIKE isn't
        # redundant. Whole-word matching (via _pat_re) is what stops generic-word false-grounds.
        hitpats: list[str] = []
        descs: dict[str, int] = {}
        for pat in patterns:
            rx = _pat_re(pat)
            new = [(d, n) for d, n in index if d not in descs and rx.search(d)]
            if new:
                hitpats.append(pat)
                descs.update(new)
        if not hitpats:
            if is_condition:                                # clearly a condition, but nothing matches → absent
                res.absent.append(term)
            continue
        best = TermMatch(term=term, patterns=hitpats,
                         descriptions=sorted(((_titlecase_desc(d), n) for d, n in descs.items()),
                                             key=lambda x: -x[1]))
        descset = frozenset(d for d, _ in best.descriptions)
        # skip a term whose descriptions are nested with an already-kept term's (e.g. broad "kidney
        # disease" after specific "chronic kidney disease", or the word "diabetic" after "diabetes").
        # Genuinely distinct cohorts that merely share one description (diabetes vs kidney) are kept.
        if any(descset <= c or c <= descset for c in covered):
            continue
        covered.append(descset)
        res.matched.append(best)

    if res.absent:
        res.suggestions = [(_titlecase_desc(d), n) for d, n in index
                           if "(disorder)" in d and n > 0][:10]
    return res
