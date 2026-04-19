"""
evidence_map.py
===============
Build a patient × evidence_category availability matrix from the real
MIMIC-IV-Ext CDM tables.

For each hadm_id, determine which canonical evidence categories the dataset
*contains* (i.e., what a perfect guideline-follower could possibly obtain).
This is the denominator of the guideline-adherence ceiling analysis.
"""

from pathlib import Path
import json
import re
import pandas as pd
import numpy as np

DATA = Path("/mnt/user-data/uploads")

# --- Canonical categories → keyword patterns on lab_test_mapping.label ---
# Matched against the `label` column of lab_test_mapping.csv (case-insensitive).
LAB_KEYWORDS = {
    "cbc":        [r"white blood", r"wbc", r"hemoglobin", r"hematocrit",
                   r"platelet", r"\brbc\b", r"mch\b", r"mcv", r"mchc", r"rdw",
                   r"neutrophil", r"lymphocyte", r"monocyte", r"eosinophil",
                   r"basophil", r"\bband\b"],
    "bmp":        [r"sodium", r"potassium", r"chloride", r"bicarbonate",
                   r"urea nitrogen", r"creatinine", r"glucose", r"calcium",
                   r"anion gap", r"magnesium", r"phosphate"],
    "cmp_lft":    [r"alanine amin", r"aspartate amin", r"alkaline phos",
                   r"bilirubin", r"albumin", r"total protein",
                   r"\bast\b", r"\balt\b", r"\bggt\b"],
    "lipase":     [r"\blipase\b"],
    "amylase":    [r"\bamylase\b"],
    "crp":        [r"c-reactive", r"c reactive", r"\bcrp\b"],
    "procalcitonin":[r"procalcitonin"],
    "lactate":    [r"\blactate\b", r"lactic"],
    "coagulation":[r"\bpt\b", r"\binr\b", r"\bptt\b", r"prothrombin",
                   r"partial thromboplastin"],
    "urinalysis": [r"leukocyte.*urine", r"specific gravity", r"urine.*ph",
                   r"ketone", r"nitrite", r"urobilinogen"],
    "bhcg":       [r"\bhcg\b", r"pregnancy"],
}

IMAGING_MAP = {
    # (category): list of (modality-prefix, region-prefix) pairs
    "abdominal_us":   [("Ultrasound", "Abdomen")],
    "abdominal_ct":   [("CT", "Abdomen"), ("CTU", "Abdomen")],
    "abdominal_xray": [("Radiograph", "Abdomen")],
    "chest_xray":     [("Radiograph", "Chest")],
    "mrcp":           [("MRCP", "Abdomen"), ("MRI", "Abdomen"), ("MRE", "Abdomen")],
    "ercp":           [("ERCP", "Abdomen")],
}

MICRO_MAP = {
    # Match on the label of spec_itemid (resolved via lab_test_mapping)
    # Blood cultures vs urine cultures, typically
    "blood_culture": [r"blood"],
    "urine_culture": [r"urine"],
}


def load_all():
    hpi   = pd.read_csv(DATA / "history_of_present_illness.csv")
    pe    = pd.read_csv(DATA / "physical_examination.csv")
    labs  = pd.read_csv(DATA / "laboratory_tests.csv")
    micro = pd.read_csv(DATA / "microbiology.csv")
    rad   = pd.read_csv(DATA / "radiology_reports.csv")
    dx    = pd.read_csv(DATA / "discharge_diagnosis.csv")
    proc  = pd.read_csv(DATA / "discharge_procedures.csv")
    lmap  = pd.read_csv(DATA / "lab_test_mapping.csv")
    with open(DATA / "pathology_ids.json") as fp:
        pids = json.load(fp)
    # Patient → pathology
    case_path = {int(h): p for p, ids in pids.items() for h in ids}
    return dict(hpi=hpi, pe=pe, labs=labs, micro=micro, rad=rad, dx=dx,
                proc=proc, lmap=lmap, case_path=case_path)


def build_itemid_to_category(lmap: pd.DataFrame) -> dict:
    """Return {itemid: category_name} for all lab itemids matching our buckets."""
    out = {}
    labels = lmap["label"].fillna("").str.lower()
    for cat, patterns in LAB_KEYWORDS.items():
        mask = pd.Series(False, index=lmap.index)
        for pat in patterns:
            mask |= labels.str.contains(pat, regex=True, na=False)
        for iid in lmap.loc[mask, "itemid"].dropna():
            out[int(iid)] = cat   # first category wins; no duplicates expected
    return out


def evidence_availability(data) -> pd.DataFrame:
    """Return a wide hadm_id × category boolean matrix.
    1 iff the patient's record contains that kind of evidence."""
    lmap = data["lmap"]
    case_ids = sorted(data["case_path"].keys())

    iid2cat = build_itemid_to_category(lmap)

    # HPI / PE: every case has these by construction of the dataset
    has_hpi = set(data["hpi"]["hadm_id"])
    has_pe  = set(data["pe"]["hadm_id"])

    # Labs
    labs = data["labs"]
    labs["category"] = labs["itemid"].map(iid2cat)
    lab_avail = (labs.dropna(subset=["category"])
                     .groupby(["hadm_id", "category"]).size().unstack(fill_value=0) > 0)

    # Imaging
    rad = data["rad"].copy()
    rad["modality"] = rad["modality"].fillna("")
    rad["region"]   = rad["region"].fillna("")
    img_rows = []
    for cat, rules in IMAGING_MAP.items():
        mask = pd.Series(False, index=rad.index)
        for mod, reg in rules:
            mask |= ((rad["modality"] == mod) & (rad["region"] == reg))
        hadm_hit = set(rad.loc[mask, "hadm_id"])
        for h in hadm_hit:
            img_rows.append((h, cat))
    img_avail = pd.DataFrame(img_rows, columns=["hadm_id", "category"]).assign(v=True)
    if len(img_avail):
        img_avail = img_avail.pivot_table(index="hadm_id", columns="category",
                                          values="v", aggfunc="any").fillna(False)

    # Microbiology — map spec_itemid via lmap
    micro = data["micro"].copy()
    micro = micro.merge(lmap[["itemid", "label"]].rename(columns={"itemid": "spec_itemid",
                                                                  "label": "spec_label"}),
                        on="spec_itemid", how="left")
    mic_rows = []
    for cat, patterns in MICRO_MAP.items():
        mask = pd.Series(False, index=micro.index)
        lbl_l = micro["spec_label"].fillna("").str.lower()
        for pat in patterns:
            mask |= lbl_l.str.contains(pat, regex=True, na=False)
        for h in set(micro.loc[mask, "hadm_id"]):
            mic_rows.append((h, cat))
    mic_avail = pd.DataFrame(mic_rows, columns=["hadm_id", "category"]).assign(v=True)
    if len(mic_avail):
        mic_avail = mic_avail.pivot_table(index="hadm_id", columns="category",
                                          values="v", aggfunc="any").fillna(False)

    # Combine everything into a single wide boolean matrix
    all_cats = (["history_of_present_illness", "physical_examination", "vital_signs"]
                + sorted(LAB_KEYWORDS.keys())
                + sorted(IMAGING_MAP.keys())
                + sorted(MICRO_MAP.keys()))

    avail = pd.DataFrame(False, index=case_ids, columns=all_cats, dtype=bool)
    avail.loc[list(has_hpi & set(case_ids)), "history_of_present_illness"] = True
    avail.loc[list(has_pe  & set(case_ids)), "physical_examination"] = True
    # Vital signs: we extract from PE text on demand; treat as present whenever PE is present
    avail["vital_signs"] = avail["physical_examination"]

    if len(lab_avail):
        for col in lab_avail.columns:
            if col in avail.columns:
                idx = lab_avail.index.intersection(avail.index)
                avail.loc[idx, col] = avail.loc[idx, col] | lab_avail.loc[idx, col]
    if "img_avail" in dir() and len(img_avail):
        pass
    try:
        for col in img_avail.columns:
            if col in avail.columns:
                idx = img_avail.index.intersection(avail.index)
                avail.loc[idx, col] = avail.loc[idx, col] | img_avail.loc[idx, col].astype(bool)
    except Exception:
        pass
    try:
        for col in mic_avail.columns:
            if col in avail.columns:
                idx = mic_avail.index.intersection(avail.index)
                avail.loc[idx, col] = avail.loc[idx, col] | mic_avail.loc[idx, col].astype(bool)
    except Exception:
        pass

    # Attach pathology
    avail["pathology"] = pd.Series(data["case_path"])
    return avail


if __name__ == "__main__":
    data = load_all()
    print(f"Cases: {len(data['case_path'])}")
    avail = evidence_availability(data)
    print(f"\nAvailability matrix: {avail.shape}")
    print(f"Pathology counts:\n{avail['pathology'].value_counts()}")
    print(f"\nPer-category availability (fraction of cases):")
    ev_cols = [c for c in avail.columns if c != "pathology"]
    print((avail[ev_cols].mean().sort_values(ascending=False)).round(3).to_string())
    print(f"\nPer-pathology availability of key categories:")
    key = ["cbc", "cmp_lft", "lipase", "abdominal_ct", "abdominal_us", "mrcp"]
    print(avail.groupby("pathology")[key].mean().round(3).to_string())
