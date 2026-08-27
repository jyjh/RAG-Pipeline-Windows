"""FSAE-EV corpus relevance filter (initial-ingestion plugin).

On initial ingestion, a mixed reading bank contains documents that do not
belong in an FSAE **EV** knowledge base: combustion-powertrain material
(engine/cams/fuel systems), rulebook editions superseded years ago, tutorials
for decade-old software versions, and obsolete battery chemistries. Indexing
them pollutes retrieval and wastes parse hours.

This module is a small rule registry -- every rule is a case-insensitive
regular expression evaluated against the PDF's path (directories included, so
a whole "Engine Related Readings" tree is caught). Each match carries a
human-readable reason. Rules are deliberately CONSERVATIVE: ambiguous names
(engineer, transmission, cooling, dyno) stay in the corpus; only clear
combustion/outdated signals exclude a document.

Usage (driver/CLI integration)::

    from src.fsae_ev_filter import filter_pdf_list
    kept, ignored = filter_pdf_list(pdf_paths, root=corpus_root,
                                    log_file=ROOT / "logs" / "ignored_documents.log")

Ignored documents are appended to the log file, one line each::

    2026-08-26 04:55:01 | Aerodynamics/2008 Rules.pdf | outdated rulebook (2008)

Stdlib only; safe to import before the project venv exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Years at least this old count as outdated (for rulebooks/software).
RULEBOOK_MAX_AGE_YEARS = 3
SOFTWARE_MAX_AGE_YEARS = 5
# Any document whose path carries a year this old is treated as superseded
# course/tutorial material (strict mode requested for ingest speed).
GENERAL_MAX_AGE_YEARS = 14

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Combustion-powertrain signals. Word-boundary anchored on BOTH sides, so
# "engineering"/"engineers" never match the engine rule, and "fuel" exempts
# fuel cells (legitimately EV-relevant).
_COMBUSTION_PATTERNS: list[tuple[str, str]] = [
    (r"\bengines?\b", "combustion powertrain (engine)"),
    (r"\bcombustion", "combustion powertrain"),
    (r"\bcarburet", "combustion powertrain (carburetor)"),
    (r"\bintake\b", "combustion powertrain (intake)"),
    (r"\bexhaust", "combustion powertrain (exhaust)"),
    (r"\bmuffler", "combustion powertrain (exhaust)"),
    (r"\bcamshaft", "combustion powertrain (valvetrain)"),
    (r"\bcrankshaft|\bcrankcase", "combustion powertrain (crank)"),
    (r"\bpiston", "combustion powertrain (piston)"),
    (r"\bturbo", "combustion powertrain (turbo)"),
    (r"\bsupercharg", "combustion powertrain (supercharger)"),
    (r"\bdiesel", "combustion powertrain (diesel)"),
    (r"\bgasoline|\bpetrol", "combustion fuel"),
    (r"(?<!fuel )\bfuel(?![- ]?cell)", "combustion fuel system"),
    (r"\bfuel[- ]injection|\binjectors?\b", "combustion fuel system (injection)"),
    (r"\boil[- ]pan|\boil[- ]system", "combustion lubrication"),
    (r"\bdry[- ]?sump|\bscavenge[- ]?pump", "combustion lubrication (dry sump)"),
    (r"\bignition\b|\bspark[- ]?plug", "combustion ignition system"),
    (r"\bhot\s*rod\s+your\s+buick\s+v6|\bbuick\s+v6\b", "combustion hot-rod material"),
    (r"\bmanifold", "combustion powertrain (manifold)"),
    (r"\bintercooler", "combustion powertrain (intercooler)"),
    (r"\bradiator", "combustion cooling (radiator)"),
    (r"\b[24][- ]stroke|\btwo[- ]stroke|\bfour[- ]stroke", "combustion powertrain (stroke)"),
    (r"\bmethanol|\boctane|\bnitrous", "combustion fuel"),
]

# Non-FSAE material that shows up in shared reading banks.
_JUNK_PATTERNS: list[tuple[str, str]] = [
    (r"\bcustom[- ]pc\b|\bpc[- ]build|\boverclock", "non-FSAE content (PC hardware)"),
    (
        r"\b(?:hollow earth|masonic conspiracy|what nasa isn.t telling you)\b",
        "non-engineering conspiracy material",
    ),
    (r"(?:^|/)popular mechanics\s*-", "low-value consumer repair article"),
    (
        r"(?:^|/)d265[pt]\d+\.pdf$",
        "opaque high-cost document series",
    ),
    (
        r"(?:^|/)(?:viewcmd_[^/]+|viewfn_[^/]+|solidcam_20\d{2}[^/]*)\.pdf$",
        "low-value software help/manual",
    ),
    (
        r"(?:^|/)solver_fortran_[^/]+\.pdf$",
        "low-value software help/manual",
    ),
    (
        r"(?:^|/)(?:\(?solutions?[_ -]manual\)?|university physics.*\[solutions\])",
        "solution-manual title",
    ),
    (
        r"(?:^|/)(?:\d+\s*[-–]\s*)?thermodynamics(?:\s|\.|-|_)*\.pdf$",
        "generic foundation textbook",
    ),
    (
        r"(?:^|/)(?:.*management 101.*|.*six sigma.*|.*project mana(?:ge|g)ment.*|"
        r".*planning and replanning in project and production scheduling.*|"
        r".*principles of research.*)\.pdf$",
        "generic management material",
    ),
    (
        r"(?:^|/).*(?:differential topology|harmonic analysis and partial differential equations|"
        r"introduction to stochastic differential equations).*\.pdf$",
        "pure mathematics material",
    ),
    (
        r"(?:^|/).*(?:quantum(?:.?mechanics|.?fields?.?theory)|atomic.?spectra).*\.pdf$",
        "non-vehicle theoretical physics material",
    ),
    (
        r"(?:^|/).*(?:photoshop|indesign|hacking google maps).*\.pdf$",
        "non-engineering software material",
    ),
    (
        r"(?:^|/).*(?:introduction to simulink|pc1432|pde[_ ]tutorial|"
        r"lecture.notes.(?:the exponential function|max.and.min|big.picture)).*\.pdf$",
        "generic course/tutorial material",
    ),
    (r"(?:^|/)\[nasa[^/]*\]", "generic NASA archive"),
    (
        r"(?:^|/)(?:6\.39 - medical applications of composites|6\.40 - medical devices|"
        r"6\.41 - application of composites in sporting goods)\.pdf$",
        "non-vehicle composites application",
    ),
    (r"(?:^|/)nasa_x[^/]*\.pdf$", "generic NASA aircraft archive"),
    (r"(?:^|/)ship design, construction and operation\.pdf$", "generic ship-design material"),
]

# Large, generic reference-library trees in the FSAE reading bank.  These are
# not bad documents, but indexing thousands of foundation-course chapters,
# answer keys, general handbooks, and unrelated military/ship books makes a
# focused FSAE-EV index slower and less precise.  Match complete path
# components so a useful paper that merely mentions e.g. "mechanics" is kept.
_LOW_VALUE_LIBRARY_PATTERNS: list[tuple[str, str]] = [
    (
        r"(?:^|/)books(?:/|$)",
        "generic book-library collection",
    ),
    (
        r"^knowledge/(?:finite element|mechanics|rai foundation colleges me lecture materials|"
        r"engineering handbook|answer|military|osprey|engineering mathematics|for dummy|"
        r"m\.eng reading|ship|continuum mechanics|physics|engineering in mandrain|financial|"
        r"catalog|tech magazine|nus modules)(?:/|$)",
        "generic reference-library collection",
    ),
    (r"^cornell ansys(?:/|$)", "outdated bulk software tutorial collection"),
    (r"^\[solutions manual\]", "solution-manual collection"),
    (r"^white_fluid_mechanics", "solution-manual collection"),
    (
        r"^shigley.s mechanical engineering design/solutions(?:/|$)",
        "solution-manual collection",
    ),
    (
        r"^reading articles/(?:book collection -matlab-|ansys tutorials collection)(?:/|$)",
        "outdated bulk software tutorial collection",
    ),
    (
        r"^reading articles/engineers_collection_dvd3/(?:management - project & eng|mathematics)(?:/|$)",
        "generic management/mathematics collection",
    ),
    (
        r"^knowledge/(?:matlab|inventor lesson|engineers_collection_dvd3)(?:/|$)",
        "outdated bulk software tutorial collection",
    ),
]

# Obsolete battery chemistry for EV usage.
_CHEMISTRY_PATTERNS: list[tuple[str, str]] = [
    (r"\blead[- ]?acid", "obsolete battery chemistry (lead-acid)"),
    (r"\bni[- ]?cd\b|\bni[- ]?cad\b|\bnicd\b", "obsolete battery chemistry (NiCd)"),
]

# Software whose tutorials age out -- YEAR-SUFFIXED (below) or discontinued
# outright (matched regardless of year).
_SOFTWARE_NAMES = (
    "ansys", "inventor", "alias", "solidworks", "autocad", "matlab",
    "star-ccm", "starccm", "catia", "creo", "unigraphics", "hypermesh",
    "abaqus", "nastran", "patran", "femap", "solid edge", "rhino",
)
_DISCONTINUED_SOFTWARE_RE = re.compile(
    r"\b(alias|unigraphics|solid edge|pro/e|proe)\b"
)
_SOFTWARE_RE = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in _SOFTWARE_NAMES) + r")\b.*"
    r"\b(19|20)\d{2}\b|\b(19|20)\d{2}\b.*\b("
    + "|".join(re.escape(name) for name in _SOFTWARE_NAMES)
    + r")\b"
)

_RULEBOOK_RE = re.compile(r"\brules?\b|\brulebook\b|\bregulations?\b")


@dataclass(frozen=True)
class FilterDecision:
    """Why a document was excluded, for the ignore log."""
    category: str
    reason: str


def _years_in(text: str) -> list[int]:
    return [int(m.group(0)) for m in _YEAR_RE.finditer(text)]


def evaluate_pdf(path: Path, *, root: Path | None = None) -> FilterDecision | None:
    """Return the exclusion decision for one PDF, or None to keep it."""
    rel = path if root is None else (path.relative_to(root) if _is_under(path, root) else path)
    text = str(rel).replace("\\", "/").casefold()
    # Underscores are word characters to Python regex, so a natural-language
    # boundary such as ``\bengine\b`` otherwise misses ``engine_appendix``.
    # Keep the original path text for structural/path rules, but normalize
    # separators for semantic title/category rules.
    word_text = text.replace("_", " ")

    # 1. Outdated rulebook editions.
    if _RULEBOOK_RE.search(text):
        years = _years_in(word_text)
        if years:
            newest = max(years)
            cutoff = datetime.now().year - RULEBOOK_MAX_AGE_YEARS
            if newest <= cutoff:
                return FilterDecision("outdated", f"outdated rulebook ({newest})")

    # 2. Obsolete battery chemistry.
    for pattern, reason in _CHEMISTRY_PATTERNS:
        if re.search(pattern, word_text):
            return FilterDecision("chemistry", reason)

    # 3. Combustion powertrain material.
    for pattern, reason in _COMBUSTION_PATTERNS:
        if re.search(pattern, word_text):
            return FilterDecision("combustion", reason)

    # 4. Non-FSAE material.
    for pattern, reason in _JUNK_PATTERNS:
        if re.search(pattern, text):
            return FilterDecision("junk", reason)

    # 5. Low-signal bulk libraries.  This deliberately runs after the richer
    # chemistry/combustion decisions so the ignore log records the most useful
    # reason when more than one rule matches.
    for pattern, reason in _LOW_VALUE_LIBRARY_PATTERNS:
        if re.search(pattern, text):
            return FilterDecision("scope", reason)

    # 6. Discontinued software (regardless of year), then year-aged software.
    if _DISCONTINUED_SOFTWARE_RE.search(text):
        return FilterDecision("outdated", "discontinued software")
    match = _SOFTWARE_RE.search(word_text)
    if match:
        years = _years_in(word_text)
        cutoff = datetime.now().year - SOFTWARE_MAX_AGE_YEARS
        if years and max(years) <= cutoff:
            return FilterDecision("outdated", f"outdated software tutorial ({max(years)})")

    # 7. Strict general age cutoff: any path carrying a year this old is
    # superseded course/tutorial material (applied AFTER all specific rules
    # so their richer reasons win when they match).
    years = _years_in(word_text)
    if years and max(years) <= datetime.now().year - GENERAL_MAX_AGE_YEARS:
        return FilterDecision("outdated", f"outdated material ({max(years)})")

    return None


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def filter_pdf_list(
    paths: list[Path],
    *,
    root: Path | None = None,
    log_file: Path | None = None,
) -> tuple[list[Path], list[tuple[Path, FilterDecision]]]:
    """Split PDFs into (kept, ignored) and append ignores to ``log_file``.

    The log line format is ``<timestamp> | <relative path> | <reason>``; the
    file is created (with a header) on first use and appended on re-runs, so
    the record accumulates across deployments.
    """
    kept: list[Path] = []
    ignored: list[tuple[Path, FilterDecision]] = []
    for path in paths:
        decision = evaluate_pdf(path, root=root)
        if decision is None:
            kept.append(path)
        else:
            ignored.append((path, decision))
    if ignored and log_file is not None:
        write_ignore_log(ignored, log_file, root=root)
    return kept, ignored


def write_ignore_log(
    ignored: list[tuple[Path, FilterDecision]],
    log_file: Path,
    *,
    root: Path | None = None,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_file.exists()
    with log_file.open("a", encoding="utf-8", newline="\n") as handle:
        if new_file:
            handle.write("# Documents excluded from FSAE-EV ingestion (name | reason)\n")
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for path, decision in ignored:
            rel = path if root is None or not _is_under(path, root) else path.relative_to(root)
            handle.write(f"{stamp} | {rel} | {decision.reason}\n")
