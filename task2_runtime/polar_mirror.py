"""Minimal question-mirror for polar (yes/no) answers — `polar_mirror(question, polarity)`.

Turns a yes/no QUESTION plus a polarity into the shortest declarative that restates it:

    Is a needle driver involved in the procedure?  + Yes -> "Yes, a needle driver is involved."
    Was a large needle driver used during the surgery? + No -> "No, a large needle driver was not used."
    Is a large needle driver among the listed tools? + No -> "No, a large needle driver is not listed."
    Are there forceps being used here?              + No -> "No, forceps are not being used."

Grammar, not templates: aux + subject + predicate [+ generic context adjunct]. Anything the grammar
cannot parse returns None (caller keeps its previous answer). No case ids, no exact-question matching.
"""
from __future__ import annotations

import re

AUX_COPULA = {"is", "are", "was", "were", "am"}
AUX_DO = {"do", "does", "did"}
AUX_HAVE = {"has", "have", "had"}
AUX_MODAL = {"can", "could", "should", "would", "will", "may", "might", "must"}
AUX = AUX_COPULA | AUX_DO | AUX_HAVE | AUX_MODAL
CONTRACTED = {"isn't": "is", "aren't": "are", "wasn't": "was", "weren't": "were", "doesn't": "does",
              "don't": "do", "didn't": "did", "hasn't": "has", "haven't": "have", "can't": "can",
              "couldn't": "could", "won't": "will", "wouldn't": "would", "shouldn't": "should"}

# politeness / meta wrappers: "can you tell me whether X?", "is it true that X?"
_PREAMBLE = re.compile(
    r"^(?:(?:can|could|would|will) you (?:please )?(?:tell me|confirm|say|check|verify|see)"
    r"(?: if| whether| that)?|is it (?:true|correct|the case|possible) that|"
    r"do you (?:think|believe|know)(?: if| whether| that)?|would you say(?: that)?)\s+", re.I)

# generic CONTEXT adjuncts that carry no content -> dropped from the mirror. Only these nouns:
# a location/object adjunct ("on the uterine horn", "to cut tissue") is content and is kept.
_CTX_NOUN = (r"(?:(?:current|given|entire|whole|surgical|robotic|present|this|particular|specific)\s+)*"
             r"(?:procedure|surgery|surgeries|clip|video|step|operation|scene|frames?|footage|case|"
             r"recording|segment|session|task|sequence|image|images|shot|view|summary|clips|videos)")
_ADJUNCT = re.compile(
    r"(?:^|\s+)(?:in|during|within|throughout|across|over|of)\s+(?:this|the|a|an|these|those)?\s*"
    + _CTX_NOUN + r"\b\s*$|(?:^|\s+)(?:here|currently|right now|at the moment|at this point)\s*$", re.I)
_LISTED = re.compile(r"(?:^|\s+)(?:among|in|on|within|part of)\s+the\s+(?:listed|list of|given|provided)?\s*"
                     r"(?:tools?|instruments?|list|inventory)(?:\s+(?:list|listed|provided|given))?\s*$"
                     r"|(?:^|\s+)(?:among|on)\s+the\s+list\s*$", re.I)

# tokens that START the predicate (everything before = subject)
_PRED_WORDS = {"among", "present", "visible", "in", "on", "inside", "within", "at", "part", "being",
               "used", "involved", "required", "necessary", "needed", "shown", "seen", "mentioned",
               "listed", "performed", "currently", "also", "still", "actively", "already", "utilized",
               "employed", "applied", "deployed", "engaged", "held", "inserted", "installed", "present",
               "available", "included", "there", "made", "done", "cut", "cutting", "open", "been", "be",
               # bare verbs (do-support / modal questions): "Does the surgeon USE ...", "Can it BE seen"
               "use", "see", "appear", "show", "require", "involve", "need", "include", "contain",
               "perform", "hold", "grasp", "have", "employ", "utilize", "feature", "apply", "seem",
               "look", "exist", "occur", "take", "happen", "mention", "list", "get", "make", "belong",
               "come", "go", "work", "operate", "manipulate", "dissect", "coagulate", "retract",
               "handle", "carry"}   # NOT clip/staple/seal/suture: those are tool-name tokens
# -ed/-ing words that are NOT predicates (tool-name adjectives / nouns)
_NOT_PRED = {"fenestrated", "curved", "wristed", "tip-up", "red", "bed", "need", "seed", "feed",
             "thing", "string", "ring", "king", "wing", "spring", "sling", "bleeding", "suturing",
             "clipping", "stapling", "sealing", "grasping", "cutting", "coagulating", "retracting"}
_DET = {"a", "an", "the", "this", "that", "these", "those"}
_QUANT = {"any", "some"}
_WH = {"what", "which", "where", "when", "why", "how", "who", "whom", "whose"}
_PRON = {"this", "that", "it", "they", "these", "those", "he", "she", "we", "you"}


def _tokens(q: str) -> list[str]:
    q = q.strip()
    q = re.sub(r"[?!.]+$", "", q).strip()
    q = _PREAMBLE.sub("", q)
    q = re.sub(r"\s+", " ", q)
    return q.split(" ") if q else []


def _3sg_stem(tok: str) -> str:
    """'appears'->'appear', 'uses'->'use', 'applies'->'apply'; '' when tok is not -s inflected."""
    t = tok.lower()
    if len(t) > 3 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es") and t[-3] in "sxzoh":
        return t[:-2]
    if len(t) > 2 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return ""


def _is_participle(tok: str) -> bool:
    t = tok.lower()
    if t in _NOT_PRED:
        return False
    return len(t) > 4 and (t.endswith("ed") or t.endswith("ing"))


def _split_subject(rest: list[str]) -> tuple[list[str], list[str]]:
    """rest = tokens after the aux (and after 'there'). Return (subject, predicate)."""
    if not rest:
        return [], []
    start = 0 if rest[0].lower() in _PRON else (1 if rest[0].lower() in _DET | _QUANT else 0)
    for i in range(start, len(rest)):
        t = rest[i].lower()
        if i > start and (t in _PRED_WORDS or _is_participle(t) or t in _DET):
            return rest[:i], rest[i:]
        if i == start and (t in _PRED_WORDS or _is_participle(t)) and i == 0:
            # question like "Is cutting happening?" -> subject is the gerund noun; keep scanning
            continue
    return rest, []


def _strip_adjunct(pred: list[str]) -> tuple[list[str], bool]:
    """Drop a trailing generic-context adjunct; map 'among the listed tools' -> 'listed'.
    Returns (predicate_tokens, listed_flag)."""
    s = " ".join(pred)
    if _LISTED.search(s):
        s = _LISTED.sub("", s)
        return (s.split() if s else []) + ["listed"], True
    prev = None
    while prev != s:                       # peel nested adjuncts: "... in this clip here"
        prev = s
        s = _ADJUNCT.sub("", s)
    return (s.split() if s else []), False


def _fix_copular_participle(pred: list[str]) -> list[str]:
    """'being performed an open surgery' -> 'an open surgery' (participial modifier of the subject)."""
    s = " ".join(pred)
    s2 = re.sub(r"^(?:being\s+)?(?:performed|done|carried out|conducted|shown|seen|used)\s+"
                r"(?=(?:a|an|the)\s)", "", s, flags=re.I)
    return s2.split() if s2 else pred


def polar_mirror(question: str, polarity: str | None) -> str | None:
    """Return 'Yes, ...' / 'No, ...' minimal declarative, or None when the question is not parseable."""
    if polarity not in ("Yes", "No"):
        return None
    toks = _tokens(question)
    if not toks:
        return None
    aux_raw = toks[0].lower()
    aux = CONTRACTED.get(aux_raw, aux_raw)
    if aux in AUX:
        rest = toks[1:]
        existential = bool(rest) and rest[0].lower() == "there"
        if existential:
            rest = rest[1:]
        subj, pred = _split_subject(rest)
    else:
        # declarative clause left after a preamble ("...tell me if A STAPLER IS USED"): find the aux
        if toks[0].lower() in _WH:
            return None
        j = next((i for i, t in enumerate(toks) if CONTRACTED.get(t.lower(), t.lower()) in AUX), None)
        if j == 0 or (j is not None and j > 6):
            return None
        if j is None:
            # A preamble can leave a bare declarative whose verb is a main verb
            # ("...that a vessel sealer APPEARS in this clip") -- no aux to move. Find the first
            # finite verb (base form in _PRED_WORDS, or its 3rd-person -s form), mirror directly,
            # with do-support on the negation.
            k = next((i for i in range(1, len(toks))
                      if toks[i].lower() in _PRED_WORDS or _3sg_stem(toks[i]) in _PRED_WORDS), None)
            if k is None:
                return None
            subj, pred = toks[:k], toks[k:]
            if subj[0].lower() in _QUANT:
                subj = subj[1:]
                if not subj:
                    return None
                if _3sg_stem(pred[0]) in _PRED_WORDS and subj[0].lower() not in _DET:
                    subj = ["a"] + subj
            pred, _listed = _strip_adjunct(pred)
            if not pred:
                return None
            subject, verb, rest = " ".join(subj), pred[0], " ".join(pred[1:])
            if polarity == "Yes":
                body = f"{subject} {' '.join(pred)}"
            else:
                stem = _3sg_stem(verb)
                if stem in _PRED_WORDS:                       # 3sg -> does not <lemma>
                    body = f"{subject} does not {stem}" + (f" {rest}" if rest else "")
                else:                                          # base/plural -> do not <verb>
                    body = f"{subject} do not {verb}" + (f" {rest}" if rest else "")
            body = re.sub(r"\s+", " ", body).strip()
            return f"{polarity}, {body}."
        aux = CONTRACTED.get(toks[j].lower(), toks[j].lower())
        existential = toks[0].lower() == "there"
        subj, pred = (toks[1:j] if existential else toks[:j]), toks[j + 1:]
    if not subj:
        return None
    # For "Is <NP> <PP> <NP2>?", _split_subject cuts at the preposition, producing
    # "the instrument is in the left hand a needle driver". When the predicate opens with a
    # locative PP and a NEW determiner-NP follows, the PP belongs to the subject: reattach.
    if (aux in AUX_COPULA and not existential and pred
            and pred[0].lower() in {"in", "on", "at", "inside", "within"}):
        k = next((i for i in range(2, len(pred)) if pred[i].lower() in _DET), None)
        if k is not None:
            subj, pred = subj + pred[:k], pred[k:]
    # drop 'any/some' quantifier; re-article a singular subject ("any needle driver" -> "a needle driver")
    added_article = False
    if subj[0].lower() in _QUANT:
        subj = subj[1:]
        if not subj:
            return None
        if aux in {"is", "was", "does", "has"} and subj[0].lower() not in _DET:
            subj = ["a"] + subj
            added_article = True
    pred, listed = _strip_adjunct(pred)
    if not listed:
        pred = _fix_copular_participle(pred)
    neg = polarity == "No"
    if existential and not pred:
        # "Is there a needle driver?" -> "Yes, there is a needle driver." / "No, there is no needle driver."
        if neg:
            core = subj[1:] if subj[0].lower() in _DET else subj
            body = f"there {aux} no " + " ".join(core)
        else:
            body = f"there {aux} " + " ".join(subj)
        return f"{polarity}, {body}."
    if not pred:
        pred = ["present"]
    subject = " ".join(subj)
    predicate = " ".join(pred)
    if aux in AUX_DO:
        body = f"{subject} {aux} {'not ' if neg else ''}{predicate}"
    elif aux == "can" and neg:
        body = f"{subject} cannot {predicate}"
    else:
        body = f"{subject} {aux} {'not ' if neg else ''}{predicate}"
    body = re.sub(r"\s+", " ", body).strip()
    return f"{polarity}, {body}."
