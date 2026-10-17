"""ISBI_BASELINES_PLAN.md B10: section splitting used by the findings-only and impression-only
protocols (CPU, no data)."""
import importlib


def _mod():
    return importlib.import_module("scripts.isbi_published_rrg")


def test_findings_and_impression_parts_split_our_format():
    m = _mod()
    h = "Findings: lungs clear. Impression: no acute process."
    assert m.findings_part(h) == "lungs clear."
    assert m.impression_part(h) == "no acute process."


def test_missing_sections_give_empty_strings():
    m = _mod()
    assert m.impression_part("Findings: lungs clear, report cut at 100 tokens") == ""
    assert m.findings_part("no section labels at all") == ""
    assert m.findings_part("Findings:  Impression: stable.") == ""
