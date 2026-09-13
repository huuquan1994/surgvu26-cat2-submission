"""Question classification, terminology, and answer-format helpers."""
import re

from polar_mirror import polar_mirror

# auxiliaries that begin a polar (yes/no) question
POLAR_AUX = {"is", "are", "was", "were", "do", "does", "did", "has", "have", "had",
             "can", "could", "should", "would", "will", "am", "isn't", "aren't", "wasn't"}

# Domain-general canonical answers for procedure and tool-purpose questions.
CANON_PROCEDURE = "The summary is describing endoscopic or laparoscopic surgery."
PURPOSE = {
    "monopolar curved scissors": "To cut and dissect tissue using monopolar energy.",
    "bipolar forceps": "To grasp tissue and coagulate vessels using bipolar energy.",
    "cadiere forceps": "To grasp and hold retractors or other instruments during surgery.",
    "prograsp forceps": "To grasp and retract tissue with a firm grip.",
    "grasping retractor": "To grasp and retract tissue for exposure.",
    "needle driver": "To hold and drive the needle when suturing tissue.",
    "vessel sealer": "To seal and divide vessels and tissue bundles.",
    "clip applier": "To apply clips to occlude vessels or ducts.",
    "stapler": "To staple and divide tissue or vessels.",
    "force bipolar": "To grasp tissue and coagulate vessels using bipolar energy.",
    "permanent cautery hook/spatula": "To cut and coagulate tissue using monopolar energy.",
    "tip-up fenestrated grasper": "To grasp and retract tissue for exposure.",
    "scissors": "To cut and dissect tissue.",
    "forceps": "To grasp and hold tissues or objects during the surgery.",   # generic (least specific)
}
_PURPOSE_KEYS = sorted(PURPOSE, key=len, reverse=True)   # match "bipolar forceps" before "forceps"

# Generic class to modal commercial name. Values are lowercase for use inside sentences.
COMMERCIAL = {
    "bipolar forceps": "maryland bipolar forceps",
    "needle driver": "large suturecut needle driver",
    "vessel sealer": "vessel sealer extend",
    "stapler": "sureform stapler 60",
    "grasping retractor": "small grasping retractor",
    "clip applier": "large clip applier",
    "permanent cautery hook/spatula": "permanent cautery hook",
    "prograsp forceps": "prograsp forceps",
    "cadiere forceps": "cadiere forceps",
    "monopolar curved scissors": "monopolar curved scissors",
    "force bipolar": "force bipolar",
    "tip-up fenestrated grasper": "tip-up fenestrated grasper",
}
_COMM_KEYS = sorted(COMMERCIAL, key=len, reverse=True)   # longest first: "bipolar forceps" before none

# Known commercial variants for the twelve runtime classes. Other names fall back to the VLM.
_VARIANTS = {
    "needle driver": ("large suturecut needle driver", "large needle driver",
                      "mega needle driver", "mega suturecut needle driver"),
    "monopolar curved scissors": ("monopolar curved scissors",),
    "force bipolar": ("force bipolar",),
    "clip applier": ("large clip applier", "small clip applier"),
    "cadiere forceps": ("cadiere forceps",),
    "bipolar forceps": ("maryland bipolar forceps", "fenestrated bipolar forceps",
                        "long bipolar grasper", "micro bipolar forceps"),
    "vessel sealer": ("vessel sealer extend",),
    "permanent cautery hook/spatula": ("permanent cautery hook", "permanent cautery spatula"),
    "prograsp forceps": ("prograsp forceps",),
    "stapler": ("sureform stapler 60", "stapler 45", "sureform stapler 45",
                "stapler 30 curved-tip", "stapler 45 curved-tip"),
    "grasping retractor": ("small grasping retractor",),
    "tip-up fenestrated grasper": ("tip-up fenestrated grasper",),
}

# Commercial names that do not belong to the 12-class runtime inventory. They must
# defer to the VLM instead of being reduced to a generic suffix such as "scissors" or "forceps".
UNSUPPORTED_TOOL = re.compile(
    r"\b(potts scissors|tenaculum forceps|maryland dissector|curved bipolar dissector|"
    r"crocodile grasper|suction irrigator|synchroseal)\b|"
    r"\bsingle[ -]?site\b|\bsinglesite\b"
)


def _inside_variant(m, variants) -> bool:
    """True if this whole-word match sits inside an occurrence of a known commercial variant --
    i.e. the model already named a commercial form here, so leave it untouched."""
    s, e, ctx = m.start(), m.end(), m.string.lower()
    for v in variants:
        i = ctx.find(v)
        while i != -1:
            if i <= s and e <= i + len(v):
                return True
            i = ctx.find(v, i + 1)
    return False


def commercialize(answer: str) -> str:
    """Replace a BARE generic tool-class name with the MODAL commercial name (whole-word,
    case-insensitive). Leaves a name the model already made commercial untouched -- its own modal
    (no double-prepend) AND any non-modal variant it named ("fenestrated bipolar forceps" stays).
    The final configuration always emits commercial terminology."""
    out = answer
    for k in _COMM_KEYS:
        comm = COMMERCIAL[k]
        if comm == k:
            continue                                    # no brand to add -> leave model's own casing
        variants = _VARIANTS.get(k, ())
        out = re.sub(rf"(?i)\b{re.escape(k)}\b",
                     lambda m, comm=comm, variants=variants: m.group(0) if _inside_variant(m, variants) else comm,
                     out)
    return (out[:1].upper() + out[1:]) if out else out  # restore the sentence-initial capital


def display_name(tool_class: str) -> str:
    """Answer-facing commercial name for a generic tool class."""
    return COMMERCIAL.get(tool_class, tool_class)


def short_entity(surface: str) -> str:
    """Return a sentence-cased tool entity without trailing punctuation."""
    s = (surface or "").strip().rstrip(".")
    return (s[0].upper() + s[1:]) if s else s


def polar_answer(question: str, polarity: str) -> str:
    """Format a decided Yes/No as a minimal declarative sentence."""
    return polar_mirror(question, polarity) or polarity


def _polarity(ans: str):
    """Extract Yes/No from the model's OWN answer; None if genuinely unclear (then don't fabricate)."""
    a = " " + ans.strip().lower() + " "
    first = re.split(r"[\s,.:;!?]", ans.strip().lower(), 1)[0]
    if first == "yes":
        return "Yes"
    if first == "no":
        return "No"
    if any(t in a for t in (" not ", "n't ", " no ", " absent", " without ", "did not",
                            "does not", " is not", " are not", " was not", " were not")):
        return "No"
    if any(t in a for t in (" yes", "is being used", "was used", "is involved", "is present",
                            "are being used", "was utilized", "is required", "is being performed")):
        return "Yes"
    return None


# Extract a compact location or organ value from the VLM answer.
_PLACE = re.compile(r"(?i)(?:\bon|\bin|\bis)\s+the\s+(.+)$")


def _anatomy_mirror(question: str, answer: str):
    """Re-template the VLM's own place/organ value; None when no clean value can be extracted
    (caller keeps the VLM sentence -- safe floor)."""
    text = (answer or "").strip().rstrip(".")
    m = _PLACE.search(text)
    if not m:
        return None
    place = m.group(1).strip(" .")
    if not place or len(place.split()) > 6:
        return None
    if "organ" in question.lower():
        return f"The organ being manipulated is the {place}."
    return f"The location of the surgical procedure is the {place}."


def _trim_sentence(answer: str) -> str:
    """Keep the first complete sentence; never cut off an answer or negation by word count."""
    text = (answer or "").strip()
    if not text:
        return text
    m = re.search(r"[.!?](?=\s|$)", text)
    return (text[:m.end()] if m else text).strip()


def _tool_in(question: str):
    q = question.lower()
    if UNSUPPORTED_TOOL.search(q):
        return None
    for k in _PURPOSE_KEYS:
        if k in q:
            return k
    return None
