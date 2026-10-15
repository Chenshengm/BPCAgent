"""Versioned unit rules and arithmetic used by the independent validator.

These functions do not process files or decide whether an LLM assessment is true.
Testosterone uses the existing project's molecular-weight convention (288.42
g/mol), not the reciprocal of the rounded clinical factor 0.0347.
"""

from decimal import Decimal, DecimalException
import math
import re


UNIT_SCHEMA_VERSION = "2.1"
UNIT_RULES_VERSION = "2026-10-15.2"
UNIT_RULES = {
    "psa": {"unit": "ng/mL", "factors": {"ng/ml": 1, "ug/l": 1, "ng/dl": 0.01}},
    "testosterone": {
        "unit": "ng/dL",
        "factors": {"ng/dl": 1, "ng/ml": 100, "ug/l": 100, "pg/ml": 0.1, "nmol/l": 28.842},
    },
}
UNIT_REVIEW_RULES = {
    "psa": {
        "confirmation_required": ["ng/dl", "nmol/l"],
        "reason": "Review before conversion. Apply justified inferred unit repairs; PSA has no default molar conversion.",
    },
    "testosterone": {
        "context_review": ["ng/ml", "ug/l", "pg/ml", "nmol/l"],
        "reason": "Check assay identity, reference range and visible history; total/free must remain distinct.",
    },
    "missing_unit": "Infer and apply a supported unit using visible patient evidence; explain the decision. Do not use population majority alone.",
    "unknown_assay": "May infer total/free from the visible assay context, reference range and history. Mark applied inferences explicitly; never merge total/free trajectories.",
    "inference_policy": "Apply justified model inferences as inferred_correction without external approval. Preserve raw fields and report every applied inference. Leave genuinely ambiguous records unresolved.",
    "tmb": "Keep the original value and bin; source unit/scale is unconfirmed.",
}
UNIT_RULE_SOURCES = {
    "mass_concentration": "SI mass/volume scaling; 1 ug/L = 1 ng/mL.",
    "testosterone_molar": {
        "basis": "Project convention: testosterone molecular mass 288.42 g/mol.",
        "factor": 28.842,
        "note": "Retained from the original demo. Not an assay/laboratory verification.",
    },
}
ASSESSMENT_STATUSES = (
    "accepted", "confirmed_correction", "inferred_correction", "suspected_error", "missing_unit", "unresolved",
)
ANALYTE_TYPES = ("psa", "total_testosterone", "free_testosterone", "unknown")
_ANALYTE_ALIASES = {
    "psa": "psa",
    "testosterone": "unknown",
    "total testosterone": "total_testosterone",
    "testosterone, total": "total_testosterone",
    "free testosterone": "free_testosterone",
    "testosterone, free": "free_testosterone",
}


def normalize_unit(value):
    """Normalize known spelling differences, without guessing missing units."""
    return "".join((value or "").replace("μ", "u").replace("µ", "u").lower().split())


def analyte_type(value):
    return _ANALYTE_ALIASES.get((value or "").strip().lower(), "unknown")


def analyte_family(value):
    name = (value or "").strip().lower()
    kind = analyte_type(value)
    if kind == "psa":
        return "psa"
    if kind in {"total_testosterone", "free_testosterone"} or name == "testosterone":
        return "testosterone"
    return None


def expected_normalization(marker, effective_unit=None):
    """Recompute arithmetic only; the caller must first authorize the unit."""
    rule = UNIT_RULES.get(analyte_family(marker.get("analyte_raw")))
    factor = rule["factors"].get(normalize_unit(effective_unit)) if rule else None
    raw = str(marker.get("value_raw") or "").strip()
    numeric = None
    if re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", raw):
        try:
            if factor is not None:
                numeric = float(Decimal(raw) * Decimal(str(factor)))
        except (DecimalException, ValueError, OverflowError):
            pass
    if numeric is None or not math.isfinite(numeric):
        return {"status": "unresolved", "value": None, "unit": None, "factor": None}
    return {"status": "normalized", "value": numeric, "unit": rule["unit"], "factor": factor}
