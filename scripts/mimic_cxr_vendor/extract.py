# Per-study findings/impression extraction, ported from the official
# MIT-LCP MIMIC-CXR companion CLI (NOT vendored verbatim — that script is a
# batch CLI over a report directory tree; this is the same per-study logic
# refactored into a function so it can run inside build_mimic_cxr_local.py's
# manifest stage without re-implementing custom_mimic_cxr_rules() ourselves).
#
#   Source:  https://github.com/MIT-LCP/mimic-cxr/blob/master/txt/create_section_files.py
#   Commit:  18cdc41ca483f98659a8e649081f17c10558c3c3 (2019-10-28)
#   Fetched: 2026-08-16
#
# Faithfully reproduces: custom_indices override -> custom_section_names
# override -> section_text() with last-matching-index lookup for each of
# 'impression' and 'findings' (list_rindex in the original). We keep both
# columns separately (the original CLI's study_sections output) rather than
# collapsing to a single "conclusion" section (its patient_studies output),
# per Phase 8F of H100_SCALING_PLAN.md -- report generation conditions on
# FINDINGS specifically, which the collapsed version would have destroyed.

from typing import Dict, Optional, Tuple

from . import section_parser as sp

_CUSTOM_SECTION_NAMES, _CUSTOM_INDICES = sp.custom_mimic_cxr_rules()


def _list_rindex(items, value) -> int:
    """Index of the *last* occurrence of value in items (ported verbatim)."""
    return len(items) - items[-1::-1].index(value) - 1


def extract_findings_impression(text: str, study_stem: str) -> Tuple[str, str]:
    """Return (findings, impression) for one report, official-parser logic.

    study_stem: e.g. "s50414267" (no extension) -- the key custom_mimic_cxr_rules()
    indexes by.
    """
    if study_stem in _CUSTOM_INDICES:
        idx_start, idx_end = _CUSTOM_INDICES[study_stem]
        # The original CLI treats this as the single "conclusion" section with
        # no findings/impression distinction; surface it as impression (the
        # shorter, more citation-worthy section) with findings left empty.
        return "", text[idx_start:idx_end].strip()

    sections, section_names, _ = sp.section_text(text)

    if study_stem in _CUSTOM_SECTION_NAMES:
        sn = _CUSTOM_SECTION_NAMES[study_stem]
        if sn in section_names:
            idx = _list_rindex(section_names, sn)
            return "", sections[idx].strip()
        return "", ""

    findings = ""
    impression = ""
    if "findings" in section_names:
        findings = sections[_list_rindex(section_names, "findings")].strip()
    if "impression" in section_names:
        impression = sections[_list_rindex(section_names, "impression")].strip()

    if not findings and not impression:
        # Official fallback order: last_paragraph, then comparison.
        for sn in ("last_paragraph", "comparison"):
            if sn in section_names:
                impression = sections[_list_rindex(section_names, sn)].strip()
                break

    return findings, impression
