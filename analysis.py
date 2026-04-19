"""
analysis.py
===========
Main analysis for the paper. Three sections:

1. Dataset completeness ceiling — for each pathology, what fraction of cases
   have the complete guideline-required evidence set available in MIMIC-IV-Ext CDM?
   This is a structural upper bound: no diagnostic system that follows the
   guideline can exceed this ceiling, because the evidence isn't there.

2. Evidence-category discriminative power — mutual information between
   "category i is abnormal / present" (binary signal extracted from real lab
   values and imaging text) and the true pathology label.

3. Rule-based normative classifier — a "perfect guideline follower" agent
   encoded purely as symbolic rules over extracted features. Reports accuracy,
   per-pathology sensitivity, and confusion matrix as the normative baseline
   against which every LLM benchmark on this dataset should be compared.
"""

from pathlib import Path
import json
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score)

warnings.filterwarnings("ignore")

from evidence_map import load_all, evidence_availability, build_itemid_to_category

DATA = Path("/mnt/user-data/uploads")
OUT  = Path("/home/claude/analysis/out")
OUT.mkdir(exist_ok=True, parents=True)

PATHOLOGIES = ["appendicitis", "cholecystitis", "pancreatitis", "diverticulitis"]

# -----------------------------------------------------------------------------
# Section 1: Dataset completeness ceiling
# -----------------------------------------------------------------------------

# Guideline-required evidence sets (from guidelines.py — condensed here)
REQUIRED = {
    "appendicitis":   ["history_of_present_illness", "physical_examination",
                       "vital_signs", "cbc"],  # + imaging (any of abdominal_us/ct)
    "cholecystitis":  ["history_of_present_illness", "physical_examination",
                       "vital_signs", "cbc", "cmp_lft"],  # + abdominal imaging
    "pancreatitis":   ["history_of_present_illness", "physical_examination",
                       "cbc", "cmp_lft"],  # + lipase/amylase + imaging
    "diverticulitis": ["history_of_present_illness", "physical_examination",
                       "vital_signs", "cbc"],  # + cross-sectional imaging
}

ABDOMINAL_IMAGING = ["abdominal_us", "abdominal_ct", "mrcp", "abdominal_xray"]


def compute_completeness_ceiling(avail: pd.DataFrame) -> pd.DataFrame:
    """For each pathology, fraction of its cases that have complete
    guideline-required evidence present in the dataset."""
    rows = []
    for p in PATHOLOGIES:
        sub = avail[avail["pathology"] == p]
        n = len(sub)
        req = REQUIRED[p]
        # All scalar required categories must be present
        scalar_ok = sub[req].all(axis=1)
        # At least one of the abdominal imaging set
        imaging_ok = sub[ABDOMINAL_IMAGING].any(axis=1)
        # Pathology-specific extras
        if p == "pancreatitis":
            enzyme_ok = sub[["lipase", "amylase"]].any(axis=1)
            complete = scalar_ok & enzyme_ok & imaging_ok
        else:
            complete = scalar_ok & imaging_ok
        rows.append({
            "pathology": p, "n_cases": n,
            "scalar_required_pct": round(100 * scalar_ok.mean(), 1),
            "imaging_available_pct": round(100 * imaging_ok.mean(), 1),
            "enzyme_available_pct": (round(100 * sub[["lipase","amylase"]].any(axis=1).mean(), 1)
                                     if p == "pancreatitis" else None),
            "fully_complete_pct":  round(100 * complete.mean(), 1),
            "ceiling_accuracy":    round(complete.mean(), 3),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Section 2: Feature extraction from raw labs + imaging text
# -----------------------------------------------------------------------------

def parse_numeric(s):
    """Best-effort extraction of a float from a valuestr like '12.4 mg/dL'."""
    if pd.isna(s):
        return np.nan
    m = re.search(r"[-+]?(\d+\.?\d*|\.\d+)", str(s))
    return float(m.group(0)) if m else np.nan


# Names to look for in the `label` column for specific numeric features
SPECIFIC_LABELS = {
    "wbc":        [r"^white blood cells$"],
    "neutrophils_pct": [r"^neutrophils$"],
    "hematocrit": [r"^hematocrit$"],
    "lipase":     [r"^lipase$"],
    "amylase":    [r"^amylase$"],
    "alt":        [r"^alanine amino", r"^alt\b"],
    "ast":        [r"^aspar.{0,3}ate amino", r"^ast\b"],   # MIMIC typo "Asparate"
    "alp":        [r"^alkaline phosphatase"],
    "bilirubin_total": [r"^bilirubin, total$", r"^total bilirubin$"],
    "crp":        [r"c-reactive protein$"],
    "lactate":    [r"^lactate$"],
    "glucose":    [r"^glucose$"],
    "creatinine": [r"^creatinine$"],
}


def build_numeric_features(data) -> pd.DataFrame:
    """Return one row per hadm_id with the latest value for each specific lab."""
    lmap = data["lmap"]
    labs = data["labs"].copy()

    # Map itemid -> label (case preserved)
    lmap_valid = lmap.dropna(subset=["itemid"]).copy()
    lmap_valid["itemid"] = lmap_valid["itemid"].astype(int)
    iid2label = dict(zip(lmap_valid["itemid"], lmap_valid["label"].fillna("")))

    labs["label"] = labs["itemid"].map(iid2label).fillna("")
    labs["label_lc"] = labs["label"].str.lower()
    labs["value_num"] = labs["valuestr"].apply(parse_numeric)

    # For each specific feature, pick matching itemids and keep first value per hadm
    feat = pd.DataFrame(index=sorted(data["case_path"].keys()))
    feat.index.name = "hadm_id"

    for feat_name, patterns in SPECIFIC_LABELS.items():
        mask = pd.Series(False, index=labs.index)
        for pat in patterns:
            mask |= labs["label_lc"].str.match(pat, na=False)
        sub = labs.loc[mask, ["hadm_id", "value_num", "ref_range_upper", "ref_range_lower"]]
        if len(sub) == 0:
            feat[feat_name] = np.nan
            feat[f"{feat_name}_uln"] = np.nan
            continue
        sub = sub.dropna(subset=["value_num"]).groupby("hadm_id").first()
        feat[feat_name] = sub["value_num"]
        feat[f"{feat_name}_uln"] = sub["ref_range_upper"]

    # Abnormality flags (× ULN ratio; NaN if ULN missing)
    for feat_name in ["wbc", "lipase", "amylase", "alt", "ast", "alp",
                      "bilirubin_total", "crp", "lactate"]:
        v = feat[feat_name]
        uln = feat[f"{feat_name}_uln"]
        feat[f"{feat_name}_x_uln"] = v / uln
    return feat


# -----------------------------------------------------------------------------
# Section 2b: Imaging feature extraction from radiology text
# -----------------------------------------------------------------------------

IMAGING_PATTERNS = {
    # Descriptive findings in radiology. Impressions were stripped from the
    # dataset, so we match anatomic/pathologic descriptors, not diagnoses.
    "appendicitis_positive": [
        r"\bappendicitis\b",
        r"appendix.{0,40}(dilat|enlarg|thicken|fluid|inflam|fecalith|distend)",
        r"appendiceal.{0,40}(wall|inflam|dilat|thicken)",
        r"periappendic(eal|ular).{0,40}(fat|strand|fluid|inflam)",
        r"\bfecalith\b",
        r"dilat.{0,10}appendix",
    ],
    "cholecystitis_positive": [
        r"\bcholecystitis\b",
        r"gallbladder.{0,60}(wall.{0,15}thicken|distend|edema|inflam|hydrops)",
        r"pericholecystic.{0,20}(fluid|strand|inflam)",
        r"sonographic murphy",
    ],
    "pancreatitis_positive": [
        r"\bpancreatitis\b",
        r"pancreas.{0,40}(inflam|edema|enlarg|necrosis|peripancreatic)",
        r"peripancreatic.{0,40}(fluid|strand|inflam|fat)",
        r"pancreatic.{0,10}(duct.{0,20}dilat|necrosis|phlegmon|pseudocyst)",
    ],
    "diverticulitis_positive": [
        r"\bdiverticulitis\b",
        # Diverticul* within 300 chars of inflammatory descriptor
        r"diverticul\w*[^.]{0,300}(inflam|wall.{0,10}thicken|stranding|abscess|perfor|phlegmon)",
        r"(inflam|wall.{0,10}thicken|stranding|abscess|perfor|phlegmon)[^.]{0,300}diverticul\w*",
        # "pericolonic" / "pericolic" fat stranding — both spellings
        r"peric(olonic|olic|olic|oloni).{0,60}(fat|strand|inflam|abscess)",
        # Colonic wall thickening with stranding (strong diverticulitis signal)
        r"(sigmoid|colon|colonic).{0,80}wall.{0,15}thicken.{0,150}(strand|inflam|abscess)",
        r"(strand|inflam).{0,150}(sigmoid|colon|colonic).{0,80}wall.{0,15}thicken",
        # Plain diverticulosis mention (still associated with this cohort;
        # less specific but boosts recall)
        r"\bdiverticulosis\b",
    ],
    "gallstones": [
        r"\bcholelithiasis\b",
        r"\bgallstone",
        r"gallbladder.{0,30}(stone|calcul|sludge)",
        r"\bcholedocholithiasis\b",
    ],
}


def build_imaging_features(data) -> pd.DataFrame:
    """Return a wide boolean matrix: hadm_id × imaging-finding flags, extracted
    via regex from radiology report text."""
    rad = data["rad"].copy()
    rad["text_lc"] = rad["text"].fillna("").str.lower()
    case_ids = sorted(data["case_path"].keys())

    feat = pd.DataFrame(False, index=case_ids, columns=list(IMAGING_PATTERNS),
                        dtype=bool)
    feat.index.name = "hadm_id"

    # For each pattern, find any radiology report (for that hadm_id) matching any regex
    for flag, patterns in IMAGING_PATTERNS.items():
        combined = "|".join(patterns)
        hits = set(rad.loc[rad["text_lc"].str.contains(combined, regex=True, na=False),
                           "hadm_id"])
        feat.loc[list(hits & set(case_ids)), flag] = True
    return feat


# -----------------------------------------------------------------------------
# Section 3: Rule-based normative classifier — executes the guidelines
# -----------------------------------------------------------------------------
# Uses extracted numeric + imaging features. For each case, produces a
# differential diagnosis score for each of the four pathologies based strictly
# on guideline criteria, then predicts the argmax.

def classify_case(row) -> str:
    """Symbolic execution of the four diagnostic guidelines.

    Clinical hierarchy respected:
      1. Enzyme ≥ 3× ULN in a pain setting is highly specific for pancreatitis
         (Revised Atlanta); dominates coincident biliary imaging findings
         (gallstone pancreatitis is pancreatitis, not cholecystitis).
      2. Characteristic imaging findings dominate in the absence of enzyme
         elevation.
      3. Gallstones (cholelithiasis) alone without inflammatory wall changes
         do not satisfy TG18 for acute cholecystitis.
    """
    scores = {p: 0.0 for p in PATHOLOGIES}

    lipase_xuln  = row.get("lipase_x_uln") or 0
    amylase_xuln = row.get("amylase_x_uln") or 0
    wbc_xuln     = row.get("wbc_x_uln") or 0
    alp_xuln     = row.get("alp_x_uln") or 0
    alt_xuln     = row.get("alt_x_uln") or 0
    bili_xuln    = row.get("bilirubin_total_x_uln") or 0

    enz_3x = (lipase_xuln >= 3.0) or (amylase_xuln >= 3.0)

    # ---- Pancreatitis: Revised Atlanta ----
    if enz_3x:
        # Enzyme ≥3× ULN with characteristic pain = definite or near-definite
        # pancreatitis by Atlanta criteria (2 of 3 satisfied with pain implicit).
        # Highly specific finding — must outweigh coincident biliary imaging.
        scores["pancreatitis"] += 5.0
        if row.get("pancreatitis_positive"):
            scores["pancreatitis"] += 1.0   # all 3 Atlanta criteria met
    elif row.get("pancreatitis_positive"):
        # Imaging alone (without enzyme elevation): Atlanta 2 of 3 met via pain + imaging
        scores["pancreatitis"] += 3.5

    # ---- Appendicitis ----
    if row.get("appendicitis_positive"):
        scores["appendicitis"] += 3.5
    if wbc_xuln > 1.0:
        scores["appendicitis"] += 0.3

    # ---- Cholecystitis: TG18 ----
    if row.get("cholecystitis_positive"):
        scores["cholecystitis"] += 3.0
    # LFT pattern concerning for biliary obstruction adds weight
    if alp_xuln > 1.0 and alt_xuln > 1.0:
        scores["cholecystitis"] += 0.8
    if bili_xuln > 1.0:
        scores["cholecystitis"] += 0.5
    # Gallstones alone — weak signal that should NEVER trump a pancreatitis enzyme finding
    if row.get("gallstones") and not enz_3x:
        scores["cholecystitis"] += 0.5

    # ---- Diverticulitis ----
    if row.get("diverticulitis_positive"):
        scores["diverticulitis"] += 3.5
    if wbc_xuln > 1.0:
        scores["diverticulitis"] += 0.2

    if all(s == 0 for s in scores.values()):
        return "abstain"
    return max(scores, key=scores.get)


def run_normative_classifier(numeric_feat: pd.DataFrame,
                             imaging_feat: pd.DataFrame,
                             case_path: dict) -> pd.DataFrame:
    all_feat = numeric_feat.join(imaging_feat, how="outer")
    preds = all_feat.apply(classify_case, axis=1)
    labels = pd.Series(case_path).reindex(all_feat.index)
    out = pd.DataFrame({"pred": preds, "true": labels})
    return out


# -----------------------------------------------------------------------------
# Section 4: Mutual information of each evidence category with pathology
# -----------------------------------------------------------------------------

def mutual_info_per_feature(feat_matrix: pd.DataFrame,
                            labels: pd.Series) -> pd.DataFrame:
    """For each binary/numeric feature column in feat_matrix, compute the
    mutual information with the pathology label. Returns sorted DataFrame."""
    from sklearn.feature_selection import mutual_info_classif
    X = feat_matrix.fillna(0).astype(float).values
    y = pd.Categorical(labels).codes
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    return (pd.DataFrame({"feature": feat_matrix.columns, "mi": mi})
            .sort_values("mi", ascending=False).reset_index(drop=True))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("Loading dataset...")
    data = load_all()
    avail = evidence_availability(data)

    print("\n" + "="*68)
    print("SECTION 1 — Dataset completeness ceiling")
    print("="*68)
    ceiling = compute_completeness_ceiling(avail)
    print(ceiling.to_string(index=False))
    ceiling.to_csv(OUT / "sec1_completeness_ceiling.csv", index=False)
    overall_ceiling = (ceiling["ceiling_accuracy"] * ceiling["n_cases"]).sum() / ceiling["n_cases"].sum()
    print(f"\nOverall weighted ceiling: {overall_ceiling:.3f}")

    print("\n" + "="*68)
    print("SECTION 2 — Feature extraction")
    print("="*68)
    numeric_feat = build_numeric_features(data)
    imaging_feat = build_imaging_features(data)
    print(f"Numeric features shape: {numeric_feat.shape}")
    print(f"Imaging features shape: {imaging_feat.shape}")
    print(f"\nImaging finding prevalence (% of cases):")
    prev = (imaging_feat.mean() * 100).round(1)
    print(prev.to_string())
    print(f"\nNumeric feature availability (% non-null):")
    avail_num = ((numeric_feat.notna().mean()) * 100).round(1)
    print(avail_num[[c for c in avail_num.index if not c.endswith("_uln")
                     and not c.endswith("_x_uln")]].to_string())

    # Save intermediate
    numeric_feat.to_csv(OUT / "numeric_features.csv")
    imaging_feat.to_csv(OUT / "imaging_features.csv")

    print("\n" + "="*68)
    print("SECTION 3 — Rule-based normative classifier")
    print("="*68)
    results = run_normative_classifier(numeric_feat, imaging_feat, data["case_path"])
    # Drop abstained for headline accuracy; report separately
    abstain_rate = (results["pred"] == "abstain").mean()
    print(f"Abstention rate: {abstain_rate:.1%}")
    decided = results[results["pred"] != "abstain"]
    overall_acc = (decided["pred"] == decided["true"]).mean()
    print(f"Overall accuracy (decided cases): {overall_acc:.3f}")
    print(f"Overall accuracy (all cases, abstain = wrong): "
          f"{(results['pred'] == results['true']).mean():.3f}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    cm = pd.crosstab(results["true"], results["pred"],
                     margins=True, margins_name="Total")
    print(cm.to_string())
    cm.to_csv(OUT / "sec3_confusion_matrix.csv")

    print("\nPer-pathology classification report:")
    # Map abstain to a dummy label so sklearn doesn't break
    y_true = results["true"].values
    y_pred = results["pred"].replace("abstain", "_abstain_").values
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report).T.round(3)
    print(rep_df.to_string())
    rep_df.to_csv(OUT / "sec3_classification_report.csv")

    print("\n" + "="*68)
    print("SECTION 4 — Mutual information per feature")
    print("="*68)
    all_feat = numeric_feat.join(imaging_feat, how="outer")
    labels = pd.Series(data["case_path"]).reindex(all_feat.index)
    # Drop features with trivial missingness-ness
    mi_df = mutual_info_per_feature(all_feat, labels)
    print(mi_df.head(20).to_string(index=False))
    mi_df.to_csv(OUT / "sec4_mutual_info.csv", index=False)

    # Save overall results summary
    summary = {
        "n_cases": int(len(avail)),
        "pathology_counts": avail["pathology"].value_counts().to_dict(),
        "ceiling_overall": float(overall_ceiling),
        "ceiling_by_pathology": {r["pathology"]: r["ceiling_accuracy"]
                                 for _, r in ceiling.iterrows()},
        "classifier_accuracy_decided":  float(overall_acc),
        "classifier_accuracy_all":      float((results["pred"] == results["true"]).mean()),
        "abstain_rate":                 float(abstain_rate),
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAll outputs → {OUT}")
    return locals()


if __name__ == "__main__":
    res = main()
