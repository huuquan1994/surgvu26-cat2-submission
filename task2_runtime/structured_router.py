"""Robust structured router: answer questions by QUERYING the classifier state, not per-question
templates. State = {present, present_5f, scores, organ}. A question is parsed into a query over
{present tools, their categories, the organ} and answered; anything it can't ground on falls back
to the VLM (trimmed). This generalizes across question variants.

If the question names a tool/subtype outside the 12 classes, or
the parse is unclear, DEFER to the VLM. "Robust" here = graceful degradation, not "covers all".

Deps: only question_router (pure-python) -> importable + testable without torch/GPU.
"""
from __future__ import annotations
import math
import re
from question_router import (  # light, no torch
    commercialize, PURPOSE, CANON_PROCEDURE,
    _VARIANTS, display_name, polar_answer, _polarity, _trim_sentence,
    _anatomy_mirror, _tool_in, POLAR_AUX, UNSUPPORTED_TOOL, short_entity,
)

# 12 generic classes (must match the tool classifier / COMMERCIAL keys)
CLASSES = sorted([
    "needle driver", "monopolar curved scissors", "force bipolar", "clip applier",
    "cadiere forceps", "bipolar forceps", "vessel sealer",
    "permanent cautery hook/spatula", "prograsp forceps", "stapler",
    "grasping retractor", "tip-up fenestrated grasper",
])
_CLASSES = set(CLASSES)

# category (question noun) -> set of classes it groups
CATEGORY = {
    "forceps":   {"bipolar forceps", "cadiere forceps", "prograsp forceps", "force bipolar"},
    "grasper":   {"tip-up fenestrated grasper", "grasping retractor"},
    "scissors":  {"monopolar curved scissors"},
    "retractor": {"grasping retractor"},
    "driver":    {"needle driver"},
    "applier":   {"clip applier"},
    "sealer":    {"vessel sealer"},
    "stapler":   {"stapler"},
    "cautery":   {"permanent cautery hook/spatula"},
}
DISPLAY = {"forceps": "forceps", "grasper": "graspers", "scissors": "scissors",
           "retractor": "retractors", "driver": "needle drivers", "applier": "appliers",
           "sealer": "sealers", "stapler": "staplers", "cautery": "cautery instruments"}
SINGULAR = {"forceps": "forceps", "grasper": "grasper", "scissors": "scissors",
            "retractor": "retractor", "driver": "needle driver", "applier": "applier",
            "sealer": "sealer", "stapler": "stapler", "cautery": "cautery instrument"}

# extra commercial/colloquial aliases -> generic class (augment _VARIANTS)
_ALIASES = {
    "cautery hook": "permanent cautery hook/spatula",
    "cautery spatula": "permanent cautery hook/spatula",
    "permanent cautery hook": "permanent cautery hook/spatula",
    "permanent cautery spatula": "permanent cautery hook/spatula",
}

NEG = re.compile(r"\bno\b|\bnot\b|n't|\bwithout\b|\bexcept\b|\bneither\b")
_BAD_TOKENS = frozenset({"unspecified", "nan"})
_SANITIZER_FALLBACK = "The video shows an endoscopic surgical procedure."
_TOOLISH = re.compile(
    r"\b(tool|tools|instrument|instruments|forceps|grasper|graspers|scissors|"
    r"retractor|retractors|driver|drivers|applier|appliers|sealer|sealers|"
    r"stapler|staplers|cautery|needle|clip|bipolar|cadiere|prograsp)\b"
)
_PURPOSE_CUE = re.compile(r"\b(purpose|why|function|role)\b|used for")


# ---- phrase inventory: (phrase -> atom), longest first so "maryland bipolar forceps" beats "forceps"
def _build_master():
    master = {}
    for gen, variants in _VARIANTS.items():          # commercial variants -> ('tool', generic)
        for v in variants:
            master[v] = ("tool", gen)
    for a, gen in _ALIASES.items():
        master[a] = ("tool", gen)
    for c in CLASSES:                                # generic class names -> ('tool', class)
        master.setdefault(c, ("tool", c))
    for noun, cat in {"forceps": "forceps", "grasper": "grasper", "graspers": "grasper",
                      "scissors": "scissors", "retractor": "retractor", "retractors": "retractor",
                      "appliers": "applier", "applier": "applier", "clip appliers": "applier",
                      "sealers": "sealer", "sealer": "sealer", "vessel sealers": "sealer",
                      "cautery": "cautery", "hook": "cautery", "spatula": "cautery",
                      "staplers": "stapler", "drivers": "driver", "needle drivers": "driver"}.items():
        master.setdefault(noun, ("cat", cat))        # bare/plural category nouns (not full class names)
    phrases = sorted(master, key=len, reverse=True)
    pattern = re.compile(r"(?<![a-z])(" + "|".join(re.escape(p) for p in phrases) + r")(?![a-z])")
    return master, pattern


_MASTER, _PAT = _build_master()


def find_atoms(q: str):
    """Non-overlapping longest matches -> ordered list of ('tool', class) / ('cat', catname)."""
    out, used = [], []
    for m in _PAT.finditer(q):
        s, e = m.span()
        if any(s < ue and us < e for us, ue in used):
            continue
        used.append((s, e)); out.append(_MASTER[m.group(1)])
    return out


def _is_polar(q):
    return q.split()[0] in POLAR_AUX if q.split() else False


def _is_wh(q):
    return q.split()[0] in {"what", "which", "where", "how"} if q.split() else False


def _number(q):
    if q.startswith(("list ", "name ", "identify ")) or re.search(r"\bare\b|\bwere\b", q):
        return "plural"
    return "singular"


def _atom_present(atom, present):
    typ, val = atom
    return (val in present) if typ == "tool" else bool(CATEGORY[val] & present)


def _detect_op(q, n_atoms):
    if n_atoms < 2:
        return "or"
    if re.search(r"\bor\b", q):
        return "or"
    if "both" in q or re.search(r"\band\b", q):
        return "and"
    return "or"


def _join(names):
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"


def _comm_list(subset):
    return [display_name(t) for t in CLASSES if t in subset]


def _format_list(q, subset, noun_key):
    tools = _comm_list(subset)
    if noun_key == "tools":
        return "The tools being used are " + ", ".join(tools) + "."
    if not tools:
        return f"No {DISPLAY.get(noun_key, noun_key)} are being used."
    if len(tools) == 1:
        sing = SINGULAR.get(noun_key, noun_key)
        return (f"The type of {sing} mentioned is {tools[0]}." if "type of" in q
                else f"The {sing} being used is {tools[0]}.")
    return f"The {DISPLAY.get(noun_key, noun_key)} being used are {_join(tools)}."


def _format_singular_surface(q, noun_key, surface):
    sing = SINGULAR.get(noun_key, noun_key)
    return (f"The type of {sing} mentioned is {surface}." if "type of" in q
            else f"The {sing} being used is {surface}.")


# ---- state validity ---------------------------------------------------------
def tool_state_valid(state: dict) -> bool:
    """Structural only; empty present is allowed. Checked before every state route."""
    present = set(state.get("present") or ())
    if not present <= _CLASSES:
        return False
    scores = state.get("scores")
    if not isinstance(scores, dict):
        return False
    for c in CLASSES:
        if c not in scores:
            return False
        try:
            v = float(scores[c])
        except (TypeError, ValueError):
            return False
        if not (0.0 <= v <= 1.0) or math.isnan(v):
            return False
    return True


def organ_valid(organ) -> bool:
    if organ is None:
        return False
    return str(organ).strip().lower() not in {"", "unspecified", "nan", "none", "unknown"}


def det_organ(q, organ):
    o = organ.strip().lower()
    return (f"The organ being manipulated is the {o}." if "organ" in q.lower()
            else f"The location of the surgical procedure is the {o}.")


# ---- routing helpers --------------------------------------------------------
def _vlm_raw(vlm_answer: str):
    return (_trim_sentence(vlm_answer), "vlm_raw")


def _expand_no_match(question: str, answer: str) -> str:
    """Turn a standalone `None` into an explicit relation-preserving negative sentence."""
    if (answer or "").strip().lower().strip(".,!?;:\"' ") != "none":
        return answer
    q = (question or "").strip().lower().strip(" \t\n?.!")
    match = re.fullmatch(
        r"(?:which|what) (?P<noun>tools?|instruments?|forceps|graspers?|scissors|"
        r"retractors?|drivers?|appliers?|sealers?|staplers?|cautery instruments?)"
        r"(?:,? if any,?)? (?P<verb>is|are) (?P<predicate>.+)",
        q,
    )
    if not match:
        return answer
    return (
        f"No {match.group('noun')} {match.group('verb')} "
        f"{match.group('predicate').rstrip(' .!?')}."
    )


def _find_ambiguous_variant(q: str):
    """If q names a commercial variant of a multi-variant class, return (class, variant)."""
    cands = []
    for gen, variants in _VARIANTS.items():
        if len(variants) <= 1:
            continue
        for v in variants:
            if v == gen:
                continue
            cands.append((len(v), v, gen))
    cands.sort(reverse=True)
    for _, v, gen in cands:
        if re.search(rf"(?<![a-z]){re.escape(v)}(?![a-z])", q):
            return gen, v
    return None


def _split_conjuncts(q: str):
    """Split a compound polar on and/or."""
    parts = re.split(r"\s+(?:or|and)\s+", q)
    return [p.strip(" ,.?!") for p in parts if p.strip(" ,.?!")]


def _compound_complete(q: str) -> bool:
    """Every conjunct must contain a recognised atom; else answer nothing from state."""
    parts = _split_conjuncts(q)
    if len(parts) < 2:
        return True
    return all(bool(find_atoms(p)) for p in parts)


def _is_compound_polar(q: str) -> bool:
    return bool(re.search(r"\b(or|and|both|either)\b", q))


def _intent_gate(q: str) -> bool:
    """Return whether the question requires spatial, temporal, camera, or count evidence."""
    if "how many" in q:
        return True
    if re.search(r"\bcamera\b|\bwhen\b|\b(first|last|end|beginning)\b", q):
        return True
    if re.search(r"\b(left|right)\b", q) and _TOOLISH.search(q):
        return True
    if re.search(r"\bhand\b", q) and _TOOLISH.search(q):
        return True
    if q.startswith("where") and _TOOLISH.search(q):
        return True
    return False


# Class presence cannot answer action, target, count, condition, or temporal predicates.
# Mask complete inventory names first (e.g. "grasping retractor" is a NAME, not an action).
# Unknown residual words decline state routing instead of requiring an exhaustive action vocabulary.
_PRESENCE_WORDS = frozenset(
    "a an the any there this that these those is are was were be being been "
    "do does did has have had can could will would tool tools instrument instruments "
    "used use using involved present visible seen shown appear appears appearing "
    "listed mentioned utilized employed included among in during of on for "
    "clip video procedure surgery surgical robotic endoscopic laparoscopic here "
    "what which type types kind kinds list name identify describe "
    "and or both either you tell me if whether true surgeon".split()
)
_UNASSERTED = re.compile(
    r"\b(?:no|not|none|neither|without|unknown|unclear|uncertain|unsure|cannot|can't|"
    r"could|couldn't|may|might|unable|unidentifiable|indeterminate|undetermined|"
    r"ambiguous|indistinguishable|difficult|impossible|maybe|perhaps|possibly|"
    r"probably|likely|either|or)\b|n't\b"
)


def _tool_skeleton(q):
    return _PAT.sub("tool", q).strip(" \t\n?.!")


def _plain_tool_query(q):
    skeleton = _tool_skeleton(q)
    # A flat any()/all() is not a parser for nested Boolean expressions.
    if re.search(r"\band\b", skeleton) and re.search(r"\bor\b", skeleton):
        return False
    # Multiple tools without an explicit Boolean connector usually express a relation
    # ("Cadiere visible during needle-driver use"), not an OR-presence query.
    if len(find_atoms(q)) > 1 and not re.search(r"\b(?:and|or|both|either)\b", q):
        return False
    return all(word in _PRESENCE_WORDS for word in re.findall(r"[\w'-]+", skeleton))


def _generic_purpose(q):
    skeleton = _tool_skeleton(q)
    return bool(re.fullmatch(
        r"(?:what is the (?:purpose|function|role) of (?:using )?(?:the |a )?tool"
        r"(?: in (?:this|the) (?:procedure|surgery))?"
        r"|what is (?:the |a )?tool used for"
        r"|why (?:is|are) (?:the |a )?tool (?:used|being used)"
        r"(?: in (?:this|the) (?:procedure|surgery))?)",
        skeleton,
    ))


def _anatomy_query(q):
    return bool(re.fullmatch(
        r"(?:(?:what|which) organ is (?:being manipulated|being operated on|the surgery performed on)"
        r"|what is the location of (?:the |this )?(?:surgical )?(?:procedure|operation|surgery)"
        r"|where is (?:the |this )?(?:procedure|surgery) (?:being performed|taking place))",
        q.strip(" \t\n?.!"),
    ))


def _identification_request(q):
    return bool(re.match(
        r"(?:can|could|would|will) you (?:please )?(?:tell|list|identify|name|describe|"
        r"show|explain|give)\b", q
    )) and not re.search(r"\b(?:if|whether)\b", q)


def _is_tool_list(q: str) -> bool:
    """Recognize explicit tool-list questions, excluding purpose and action descriptions."""
    if _PURPOSE_CUE.search(q):
        return False
    if not re.search(r"\b(tools|instruments)\b", q):
        return False
    if re.search(r"\b(list|name|identify)\b", q):
        return True
    if re.search(r"\b(what|which)\b", q):
        return True
    if re.search(r"\bdescribe\b", q) and not re.search(r"\bhow\b", q):
        return True
    return False


def _is_procedure(q: str) -> bool:
    """Only generic procedure/summary requests, never an actual task or action."""
    if find_atoms(q) or re.search(r"\b(?:tools?|instruments?)\b", q):
        return False
    return bool(re.fullmatch(
        r"(?:(?:what|which) (?:procedure|surgery) (?:is (?:being performed|this summary describing)"
        r"|does (?:this|the) clip show)"
        r"|what is (?:this|the) summary describing"
        r"|describe (?:the |this )?(?:procedure|surgery)(?: being performed)?)",
        q.strip(" \t\n?.!"),
    ))


def _vlm_named_members(vlm_answer: str, members: set):
    """[(surface, class)] for member tools named in the VLM answer (longest surface first)."""
    text = (vlm_answer or "").lower()
    hits = []
    for gen in members:
        phrases = list(_VARIANTS.get(gen, ())) + [gen]
        for p in sorted(set(phrases), key=len, reverse=True):
            if re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", text):
                hits.append((p, gen))
                break
    return hits


def _is_bad_token(answer: str) -> bool:
    t = (answer or "").strip()
    if not t:
        return True
    return t.lower().strip(".,!?;:\"' ") in _BAD_TOKENS


def _strip_classifier_hint(description: str) -> str:
    """Remove the classifier-only prefix if a full handler description is supplied."""
    lines = str(description or "").splitlines()
    kept = [
        line for line in lines
        if not line.startswith("The organ being manipulated is the ")
        and not line.startswith("Surgical tools in frame ")
    ]
    return "\n".join(kept).strip()


def sanitize_final_answer(answer: str, description: str = ""):
    """Recover missing/invalid output; the string 'None' is a legitimate no-match answer."""
    if not _is_bad_token(answer):
        return answer, None
    desc = _trim_sentence(_strip_classifier_hint(description))
    if not _is_bad_token(desc):
        return desc, "sanitizer"
    return _SANITIZER_FALLBACK, "sanitizer"


def _tool_list_answer(q, state, vlm_answer, valid):
    present = set(state.get("present") or ())
    if not valid:
        return (commercialize(_trim_sentence(vlm_answer)), "vlm_raw")
    subset = present
    if not subset:
        subset = set(state.get("present_5f") or ())
    if not subset:
        return (commercialize(_trim_sentence(vlm_answer)), "vlm_raw")
    return (_format_list(q, subset, "tools"), "state")


def _polar_from_vlm(question, vlm_answer):
    pol = _polarity(vlm_answer)
    if pol is None:
        return _vlm_raw(vlm_answer)
    return (polar_answer(question, pol), "vlm_polarity")


def _ambiguous_variant_polar(question, q, state, vlm_answer, valid):
    """Use state membership for recognized commercial-variant questions. Class present -> Yes;
    class
    absent -> No. Empty state falls back to the VLM instead of fabricating a negative answer."""
    hit = _find_ambiguous_variant(q)
    assert hit is not None
    gen, _variant = hit
    present = set(state.get("present") or ())
    if not valid or not present:
        return _polar_from_vlm(question, vlm_answer)
    return (polar_answer(question, "Yes" if gen in present else "No"), "state")


def route_structured(question, state: dict, vlm_answer: str, description: str = ""):
    """Return (final_answer, source). source ∈ {state, vlm_raw, vlm_polarity, purpose,
    procedure, sanitizer}."""
    # Normalize inputs.
    if not isinstance(question, str):
        ans, src = _vlm_raw(vlm_answer)
        ans, s2 = sanitize_final_answer(ans, description)
        return ans, (s2 or src)

    q = question.strip().lower()
    present = set(state.get("present") or ())
    scores = state.get("scores") if isinstance(state.get("scores"), dict) else {}
    organ = state.get("organ")
    valid = tool_state_valid(state)
    vlm_answer = "" if vlm_answer is None else str(vlm_answer)

    def finish(ans, src):
        if src == "vlm_raw":
            ans = _expand_no_match(question, ans)
        ans2, s2 = sanitize_final_answer(ans, description)
        return ans2, (s2 or src)

    # Negated requests of every answer kind need their full predicate preserved.
    if NEG.search(q) or _identification_request(q):
        return finish(*_vlm_raw(vlm_answer))

    # Some intent families have no grounded state answer even when phrased as a polar. This guard
    # must precede membership routing (e.g. "Is the needle driver in the left hand?").
    if _intent_gate(q):
        return finish(*_vlm_raw(vlm_answer))

    # Inventory names outside the 12 runtime classes are unknown as a whole. Do not reduce them
    # to a recognized suffix ("Potts scissors" -> scissors, "Tenaculum forceps" -> forceps).
    if UNSUPPORTED_TOOL.search(q):
        return finish(*_vlm_raw(vlm_answer))

    atoms = find_atoms(q)
    # Imperative identity requests ask for a selected entity, not every present member of
    # the category. Keep explicit "List ..." as the deliberate full-state list policy.
    if q.startswith(("identify ", "name ")) and any(a[0] == "cat" for a in atoms):
        return finish(*_vlm_raw(vlm_answer))
    if (atoms or re.search(r"\b(?:tools?|instruments?)\b", q)) and (_is_polar(q) or not _PURPOSE_CUE.search(q)):
        if not _plain_tool_query(q):
            # The VLM supplies the action/relation truth, but ordinary polar formatting
            # still applies. Negated questions were already preserved verbatim above.
            routed = _polar_from_vlm(question, vlm_answer) if _is_polar(q) else _vlm_raw(vlm_answer)
            return finish(*routed)

    # Compound polar: every conjunct must have a recognized atom; use bare Yes/No.
    # Only a real multi-conjunct question is compound. A lone "both"/"either"
    # ("Are both hands used?") has one conjunct and must fall through to the single-atom / non-tool
    # polar routes instead of answering a bare No from an empty atom list.
    if _is_polar(q) and _is_compound_polar(q) and len(_split_conjuncts(q)) >= 2:
        if not _compound_complete(q):
            return finish(*_vlm_raw(vlm_answer))
        if not valid:
            return finish(*_polar_from_vlm(question, vlm_answer))
        assert atoms, q                                               # complete => every conjunct has an atom
        op = _detect_op(q, len(atoms))
        res = [_atom_present(a, present) for a in atoms]
        pol = "Yes" if (all(res) if op == "and" else any(res)) else "No"
        return finish(pol, "state")

    # Ambiguous commercial-variant polar.
    if _is_polar(q) and _find_ambiguous_variant(q) is not None:
        return finish(*_ambiguous_variant_polar(question, q, state, vlm_answer, valid))

    # Generic or one-to-one tool/category polar.
    if _is_polar(q) and atoms:
        if not valid or not present:
            return finish(*_polar_from_vlm(question, vlm_answer))
        op = _detect_op(q, len(atoms))
        res = [_atom_present(a, present) for a in atoms]
        pol = "Yes" if (all(res) if op == "and" else any(res)) else "No"
        return finish(polar_answer(question, pol), "state")

    # Non-tool polar, such as an action question.
    if _is_polar(q):
        return finish(*_polar_from_vlm(question, vlm_answer))

    # Explicit tool-list.
    if _is_tool_list(q):
        return finish(*_tool_list_answer(q, state, vlm_answer, valid))

    # Purpose, used-for, function, or role.
    if _PURPOSE_CUE.search(q):
        if not _generic_purpose(q):
            return finish(*_vlm_raw(vlm_answer))
        tool = _tool_in(question)
        if tool and tool in PURPOSE:
            return finish(PURPOSE[tool], "purpose")
        return finish(*_vlm_raw(vlm_answer))

    # Narrow procedure question with an explicit object.
    if _is_procedure(q):
        return finish(CANON_PROCEDURE, "procedure")

    # Only an anatomy identity/location request is answerable by the single organ label.
    if _anatomy_query(q):
        if organ_valid(organ):
            return finish(det_organ(question, organ), "state")
        # Invalid organ state falls back to a compact VLM-derived anatomy answer.
        mirrored = _anatomy_mirror(question, vlm_answer)
        if mirrored:
            return finish(mirrored, "vlm_raw")
        return finish(*_vlm_raw(vlm_answer))

    # Singular or plural category question.
    cats = [a[1] for a in atoms if a[0] == "cat"]
    # A supported noun is not sufficient: the complete question must ask for its identity/list.
    if cats and not _plain_tool_query(q):
        return finish(*_vlm_raw(vlm_answer))
    if cats and (_is_wh(q) or q.startswith(("list", "name", "describe", "identify"))):
        if UNSUPPORTED_TOOL.search(q) or UNSUPPORTED_TOOL.search(vlm_answer.lower()):
            return finish(*_vlm_raw(vlm_answer))
        members = set().union(*(CATEGORY[c] for c in cats))
        noun = cats[0]
        if _UNASSERTED.search(vlm_answer.lower()):
            return finish(*_vlm_raw(vlm_answer))
        if _number(q) == "singular":
            if not valid:
                return finish(commercialize(_trim_sentence(vlm_answer)), "vlm_raw")
            named = _vlm_named_members(vlm_answer, members)
            if named:
                # Keep the VLM surface; classifier scores break ties among VLM-named tools.
                if len(named) == 1 or not scores:
                    surface, gen = named[0]
                else:
                    surface, gen = max(named, key=lambda sg: scores.get(sg[1], 0.0))
                if surface == gen:
                    surface = display_name(gen)
                # Use a short name only when the VLM and verified state agree.
                # Otherwise retain the sentence template.
                if gen in present:
                    return finish(short_entity(surface), "state")
                return finish(_format_singular_surface(q, noun, surface), "state")
            present_m = present & members
            if not present_m or not valid:
                return finish(*_vlm_raw(vlm_answer))
            pick = max(sorted(present_m), key=lambda c: scores.get(c, 0.0))
            return finish(_format_singular_surface(q, noun, display_name(pick)), "state")
        # plural
        if not valid:
            return finish(*_vlm_raw(vlm_answer))
        subset = present & members
        return finish(_format_list(q, subset, noun), "state")

    # Fallback.
    return finish(*_vlm_raw(vlm_answer))
