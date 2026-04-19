"""
guidelines.py
=============
Structured encoding of the four consensus clinical guidelines used as the
normative reference in this study.

Each guideline is expressed as a GuidelineSpec with:
  - required_evidence: the evidence categories the guideline formally requires
  - priority_order:    the recommended sequence (lower index = earlier)
  - diagnostic_rule:   a callable that, given the collected evidence, returns
                       (criterion_fulfilled: bool, missing: list[str])
  - spurious_evidence: evidence categories NOT in the guideline (computed
                       against a canonical list of available test families)

Sources:
  - Appendicitis: Alvarado (1986) + 2020 WSES Jerusalem guidelines
  - Cholecystitis: Tokyo Guidelines 2018 (TG18)
  - Pancreatitis: Revised Atlanta Classification (2012)
  - Diverticulitis: 2020 ASCRS Clinical Practice Guidelines

IMPORTANT: A board-certified clinician co-author should validate these
encodings before submission. The encodings are a faithful reading of the
published guidelines but clinical judgment is required on edge cases.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set, Tuple


# -----------------------------------------------------------------------------
# Canonical evidence categories (what the dataset can serve)
# -----------------------------------------------------------------------------
# These are the high-level "buckets" the guidelines reference. Each maps to
# zero or more concrete dataset fields (lab itemids, imaging modalities, etc.)
# in the EvidenceServer.

EVIDENCE_CATEGORIES = {
    # Clinical / physical
    "history_of_present_illness": "HPI narrative",
    "physical_examination":       "Full physical exam",
    "vital_signs":                "Vitals (T, HR, BP, RR, SpO2)",
    # Laboratory panels
    "cbc":                        "Complete blood count (WBC, neutrophil %, Hb, platelets)",
    "bmp":                        "Basic metabolic panel (Na, K, Cl, HCO3, BUN, Cr, glucose)",
    "cmp_lft":                    "Liver function tests (AST, ALT, ALP, bilirubin, GGT)",
    "lipase":                     "Serum lipase",
    "amylase":                    "Serum amylase",
    "crp":                        "C-reactive protein",
    "procalcitonin":              "Procalcitonin",
    "lactate":                    "Serum lactate",
    "coagulation":                "PT/INR/PTT",
    "urinalysis":                 "Urinalysis",
    "bhcg":                       "Beta-hCG (pregnancy test)",
    # Microbiology
    "blood_culture":              "Blood cultures",
    "urine_culture":              "Urine culture",
    # Imaging
    "abdominal_us":               "Abdominal ultrasound",
    "abdominal_ct":               "Abdominal CT (typically with contrast)",
    "abdominal_xray":             "Abdominal plain radiograph",
    "chest_xray":                 "Chest radiograph",
    "mrcp":                       "MRCP / abdominal MRI",
    "hida_scan":                  "Cholescintigraphy (HIDA)",
}


@dataclass
class GuidelineSpec:
    name: str
    pathology: str
    required_evidence: List[str]
    priority_order: List[str]
    diagnostic_rule_doc: str
    # Returns (satisfies_criterion, list_of_missing_evidence_categories)
    diagnostic_rule: Callable[[Set[str]], Tuple[bool, List[str]]]
    citation: str


# -----------------------------------------------------------------------------
# Appendicitis — Alvarado Score + 2020 WSES Jerusalem Guidelines
# -----------------------------------------------------------------------------
# Alvarado requires: migratory RLQ pain, anorexia, N/V (→ HPI), RLQ
# tenderness + rebound (→ PE), fever (→ vitals), leukocytosis + left-shift
# (→ CBC differential). WSES recommends imaging confirmation (US first-line
# in children/women of reproductive age; CT otherwise) before appendectomy
# in equivocal cases.

def _appendicitis_rule(ev: Set[str]) -> Tuple[bool, List[str]]:
    required_core = {"history_of_present_illness", "physical_examination",
                     "vital_signs", "cbc"}
    required_confirmatory = {"abdominal_us", "abdominal_ct"}
    missing = []
    for e in required_core:
        if e not in ev:
            missing.append(e)
    if not (required_confirmatory & ev):
        missing.append("abdominal_imaging(us_or_ct)")
    return (len(missing) == 0), missing


APPENDICITIS = GuidelineSpec(
    name="Alvarado + WSES Jerusalem 2020",
    pathology="appendicitis",
    required_evidence=[
        "history_of_present_illness", "physical_examination",
        "vital_signs", "cbc", "abdominal_us", "abdominal_ct",
    ],
    priority_order=[
        "history_of_present_illness", "physical_examination",
        "vital_signs", "cbc", "bmp", "crp",
        "abdominal_us", "abdominal_ct", "bhcg", "urinalysis",
    ],
    diagnostic_rule_doc=(
        "Satisfied when core clinical data (HPI, PE, vitals, CBC with diff) "
        "AND at least one confirmatory imaging modality (US or CT) have been "
        "obtained prior to diagnostic commitment."
    ),
    diagnostic_rule=_appendicitis_rule,
    citation="Alvarado A. Ann Emerg Med. 1986;15(5):557-64. "
             "Di Saverio S et al. World J Emerg Surg. 2020;15:27.",
)


# -----------------------------------------------------------------------------
# Cholecystitis — Tokyo Guidelines 2018 (TG18)
# -----------------------------------------------------------------------------
# Diagnostic criteria:
#   A. Local signs of inflammation: Murphy sign, RUQ mass/pain/tenderness (→ PE/HPI)
#   B. Systemic signs of inflammation: fever (→ vitals), elevated CRP,
#      elevated WBC (→ CBC + CRP)
#   C. Imaging findings characteristic of acute cholecystitis
#      (US first-line; HIDA when equivocal)
# Suspected dx: one A + one B.  Definite dx: one A + one B + C.

def _cholecystitis_rule(ev: Set[str]) -> Tuple[bool, List[str]]:
    has_local = "physical_examination" in ev and "history_of_present_illness" in ev
    has_systemic = "vital_signs" in ev and "cbc" in ev  # CRP ideal but WBC acceptable
    has_imaging = bool({"abdominal_us", "hida_scan", "abdominal_ct", "mrcp"} & ev)
    has_lft = "cmp_lft" in ev  # TG18 recommends LFTs to assess severity/complications
    missing = []
    if not has_local: missing.append("local_signs(HPI+PE)")
    if not has_systemic: missing.append("systemic_signs(vitals+CBC)")
    if not has_imaging: missing.append("characteristic_imaging")
    if not has_lft: missing.append("cmp_lft")
    return (len(missing) == 0), missing


CHOLECYSTITIS = GuidelineSpec(
    name="Tokyo Guidelines 2018",
    pathology="cholecystitis",
    required_evidence=[
        "history_of_present_illness", "physical_examination", "vital_signs",
        "cbc", "cmp_lft", "abdominal_us",
    ],
    priority_order=[
        "history_of_present_illness", "physical_examination", "vital_signs",
        "cbc", "cmp_lft", "crp", "lipase",
        "abdominal_us", "abdominal_ct", "mrcp", "hida_scan",
    ],
    diagnostic_rule_doc=(
        "TG18 definite diagnosis requires ≥1 local sign (A) + ≥1 systemic "
        "sign (B) + imaging (C). Operationalized as: HPI, PE, vitals, CBC, "
        "LFTs, AND at least one characteristic imaging study obtained "
        "before diagnostic commitment."
    ),
    diagnostic_rule=_cholecystitis_rule,
    citation="Yokoe M et al. J Hepatobiliary Pancreat Sci. 2018;25(1):41-54.",
)


# -----------------------------------------------------------------------------
# Pancreatitis — Revised Atlanta Classification (2012)
# -----------------------------------------------------------------------------
# Diagnosis requires ≥2 of 3 criteria:
#   1. Abdominal pain consistent with acute pancreatitis (→ HPI/PE)
#   2. Serum lipase OR amylase ≥3× upper limit of normal (→ lipase/amylase)
#   3. Characteristic imaging findings (CECT, MRI, or US)
# Severity stratified by CRP, organ failure, and local complications.

def _pancreatitis_rule(ev: Set[str]) -> Tuple[bool, List[str]]:
    has_pain = "history_of_present_illness" in ev or "physical_examination" in ev
    has_enzyme = "lipase" in ev or "amylase" in ev
    has_imaging = bool({"abdominal_us", "abdominal_ct", "mrcp"} & ev)
    # Revised Atlanta requires ≥2 of 3
    n_criteria = sum([has_pain, has_enzyme, has_imaging])
    missing = []
    if n_criteria < 2:
        if not has_pain: missing.append("pain_assessment(HPI_or_PE)")
        if not has_enzyme: missing.append("pancreatic_enzymes(lipase_or_amylase)")
        if not has_imaging: missing.append("abdominal_imaging")
    # Atlanta also requires severity stratification components
    if "cbc" not in ev: missing.append("cbc")
    if "cmp_lft" not in ev: missing.append("cmp_lft")  # for etiology work-up
    return (len(missing) == 0), missing


PANCREATITIS = GuidelineSpec(
    name="Revised Atlanta Classification 2012",
    pathology="pancreatitis",
    required_evidence=[
        "history_of_present_illness", "physical_examination",
        "lipase", "cbc", "cmp_lft", "abdominal_ct",
    ],
    priority_order=[
        "history_of_present_illness", "physical_examination", "vital_signs",
        "lipase", "amylase", "cbc", "cmp_lft", "crp", "lactate",
        "abdominal_us", "abdominal_ct", "mrcp",
    ],
    diagnostic_rule_doc=(
        "Revised Atlanta requires ≥2 of 3: (1) characteristic pain, "
        "(2) lipase/amylase ≥3× ULN, (3) characteristic imaging. "
        "Etiology work-up additionally requires CBC and LFTs."
    ),
    diagnostic_rule=_pancreatitis_rule,
    citation="Banks PA et al. Gut. 2013;62(1):102-11.",
)


# -----------------------------------------------------------------------------
# Diverticulitis — 2020 ASCRS Clinical Practice Guidelines
# -----------------------------------------------------------------------------
# Diagnostic work-up:
#   - Clinical evaluation: LLQ pain, fever, altered bowel habits (→ HPI, PE, vitals)
#   - Laboratory: CBC (leukocytosis), CRP (correlates with severity)
#   - Imaging: CT abdomen/pelvis with IV contrast is the diagnostic modality
#     of choice for first presentation and to stage severity (Hinchey I–IV).
#     US or MR acceptable alternatives when CT contraindicated.

def _diverticulitis_rule(ev: Set[str]) -> Tuple[bool, List[str]]:
    required = {"history_of_present_illness", "physical_examination",
                "vital_signs", "cbc"}
    imaging_options = {"abdominal_ct", "abdominal_us", "mrcp"}
    missing = [e for e in required if e not in ev]
    if not (imaging_options & ev):
        missing.append("cross_sectional_imaging(ct_preferred)")
    return (len(missing) == 0), missing


DIVERTICULITIS = GuidelineSpec(
    name="ASCRS Clinical Practice Guidelines 2020",
    pathology="diverticulitis",
    required_evidence=[
        "history_of_present_illness", "physical_examination", "vital_signs",
        "cbc", "abdominal_ct",
    ],
    priority_order=[
        "history_of_present_illness", "physical_examination", "vital_signs",
        "cbc", "bmp", "crp", "lactate",
        "abdominal_ct", "abdominal_us", "mrcp", "abdominal_xray",
    ],
    diagnostic_rule_doc=(
        "ASCRS 2020 requires clinical assessment (HPI, PE, vitals), "
        "inflammatory laboratory markers (CBC), and cross-sectional "
        "imaging (CT preferred, US/MR acceptable) for first presentation."
    ),
    diagnostic_rule=_diverticulitis_rule,
    citation="Hall J et al. Dis Colon Rectum. 2020;63(6):728-747.",
)


# -----------------------------------------------------------------------------
# Guideline registry
# -----------------------------------------------------------------------------
GUIDELINES: Dict[str, GuidelineSpec] = {
    "appendicitis":   APPENDICITIS,
    "cholecystitis":  CHOLECYSTITIS,
    "pancreatitis":   PANCREATITIS,
    "diverticulitis": DIVERTICULITIS,
}


def spurious_evidence_for(pathology: str, requested: Set[str]) -> Set[str]:
    """Return the subset of requested evidence that is NOT in the guideline's
    priority order (i.e., not indicated for this presentation)."""
    spec = GUIDELINES[pathology]
    guideline_set = set(spec.priority_order)
    return requested - guideline_set
