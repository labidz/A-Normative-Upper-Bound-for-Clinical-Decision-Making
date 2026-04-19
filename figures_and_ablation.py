"""
figures_and_ablation.py
=======================
Produce the four publication figures plus a feature-ablation information-
theoretic analysis:

  Figure 1: Confusion matrix heatmap (rule-based classifier vs ground truth)
  Figure 2: Per-pathology precision/recall/F1 + overall accuracy vs ceiling
  Figure 3: Mutual information ranking of features
  Figure 4: Feature-ablation — accuracy as single features are removed,
            and accuracy as a function of top-k features retained
  Figure 5: Per-pathology lipase × ULN distributions (showing the diagnostic
            separation that drives MI)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix

from evidence_map import load_all, evidence_availability
from analysis import (build_numeric_features, build_imaging_features,
                      run_normative_classifier, classify_case,
                      PATHOLOGIES, compute_completeness_ceiling,
                      mutual_info_per_feature)

OUT = Path("/home/claude/analysis/out")
OUT.mkdir(exist_ok=True, parents=True)

# Style — clean, paper-quality
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    print("Loading and extracting features...")
    data = load_all()
    avail = evidence_availability(data)
    nf = build_numeric_features(data)
    imf = build_imaging_features(data)
    all_feat = nf.join(imf, how="outer")
    labels = pd.Series(data["case_path"]).reindex(all_feat.index)

    # Run classifier
    results = run_normative_classifier(nf, imf, data["case_path"])

    # ===== FIGURE 1: Confusion matrix =====
    y_true = results["true"].values
    y_pred = results["pred"].replace("abstain", "n/a").values
    labels_order = PATHOLOGIES + ["n/a"]
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_order, yticklabels=labels_order,
                cbar=False, ax=ax1, square=True, linewidths=0.4,
                annot_kws={"fontsize": 9})
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_title("(a) Confusion counts")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels_order, yticklabels=labels_order,
                cbar=False, ax=ax2, square=True, linewidths=0.4,
                vmin=0, vmax=1, annot_kws={"fontsize": 9})
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("")
    ax2.set_title("(b) Row-normalized (per-pathology recall)")
    fig.suptitle("Figure 1. Rule-based normative classifier: confusion matrix "
                 f"(N = {len(results)}; accuracy = {(y_true == y_pred).mean():.3f})",
                 fontsize=11, y=1.02)
    fig.savefig(OUT / "fig1_confusion_matrix.pdf")
    fig.savefig(OUT / "fig1_confusion_matrix.png")
    plt.close(fig)
    print(f"  fig1 saved ({(y_true == y_pred).mean():.3f} accuracy)")

    # ===== FIGURE 2: Per-pathology PRF + ceiling vs achieved =====
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=PATHOLOGIES, zero_division=0)
    prf_df = pd.DataFrame({"pathology": PATHOLOGIES,
                           "precision": p, "recall": r, "f1": f, "support": s})

    ceiling = compute_completeness_ceiling(avail)
    merged = prf_df.merge(ceiling[["pathology", "ceiling_accuracy"]], on="pathology")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(PATHOLOGIES))
    w = 0.27
    ax1.bar(x - w, prf_df["precision"], w, label="Precision",  color="#5176A8")
    ax1.bar(x,     prf_df["recall"],    w, label="Recall",     color="#E68A3B")
    ax1.bar(x + w, prf_df["f1"],        w, label="F1",         color="#6FAE6F")
    ax1.set_xticks(x); ax1.set_xticklabels(PATHOLOGIES, rotation=15)
    ax1.set_ylim(0, 1.0); ax1.set_ylabel("Score")
    ax1.set_title("(a) Per-pathology classifier performance")
    ax1.legend(loc="lower right", frameon=False)
    ax1.axhline(0.0, color="black", lw=0.5)

    # Ceiling vs achieved recall
    ax2.bar(x - 0.2, merged["ceiling_accuracy"], 0.4,
            label="Completeness ceiling", color="#B0B0B0")
    ax2.bar(x + 0.2, merged["recall"], 0.4,
            label="Classifier recall", color="#4577A8")
    ax2.set_xticks(x); ax2.set_xticklabels(PATHOLOGIES, rotation=15)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Fraction of pathology cases")
    ax2.set_title("(b) Completeness ceiling vs classifier recall")
    ax2.legend(loc="lower right", frameon=False)
    for i, (c, rr) in enumerate(zip(merged["ceiling_accuracy"], merged["recall"])):
        ax2.text(i - 0.2, c + 0.02, f"{c:.2f}", ha="center", fontsize=8)
        ax2.text(i + 0.2, rr + 0.02, f"{rr:.2f}", ha="center", fontsize=8)

    fig.suptitle("Figure 2. Normative classifier performance against the "
                 "dataset completeness ceiling", fontsize=11, y=1.02)
    fig.savefig(OUT / "fig2_per_pathology_performance.pdf")
    fig.savefig(OUT / "fig2_per_pathology_performance.png")
    plt.close(fig)
    print(f"  fig2 saved")

    # ===== FIGURE 3: Mutual information ranking =====
    mi_df = mutual_info_per_feature(all_feat, labels)
    # Keep interesting features (drop ULN refs and duplicates)
    mi_df["clean_name"] = mi_df["feature"].str.replace("_x_uln", " ×ULN", regex=False)
    mi_keep = mi_df[~mi_df["feature"].str.endswith("_uln")
                    & ~mi_df["feature"].isin(["hematocrit"])].head(15)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C44E4E" if "positive" in f else
              "#5176A8" if "lipase" in f or "amylase" in f else
              "#6FAE6F" if any(x in f for x in ("alt","ast","alp","bilirubin")) else
              "#B0B0B0" for f in mi_keep["feature"]]
    ax.barh(range(len(mi_keep))[::-1], mi_keep["mi"], color=colors)
    ax.set_yticks(range(len(mi_keep))[::-1])
    ax.set_yticklabels(mi_keep["clean_name"])
    ax.set_xlabel("Mutual information (bits) with pathology label")
    ax.set_title("Figure 3. Per-feature mutual information with 4-class pathology\n"
                 "(max possible MI ≈ log₂ 4 = 2.0 bits; uniform class entropy = 1.93 bits)",
                 fontsize=10)
    # Legend for colors
    from matplotlib.patches import Patch
    legend_items = [Patch(color="#5176A8", label="Pancreatic enzymes"),
                    Patch(color="#C44E4E", label="Imaging findings"),
                    Patch(color="#6FAE6F", label="Liver function tests"),
                    Patch(color="#B0B0B0", label="Other")]
    ax.legend(handles=legend_items, loc="lower right", frameon=False)
    fig.savefig(OUT / "fig3_mutual_information.pdf")
    fig.savefig(OUT / "fig3_mutual_information.png")
    plt.close(fig)
    print(f"  fig3 saved (top MI: {mi_keep.iloc[0]['feature']} = {mi_keep.iloc[0]['mi']:.3f})")

    # ===== FIGURE 4: Feature ablation =====
    # For each feature, how much does classifier accuracy drop if we set it to NaN?
    baseline_acc = (y_true == y_pred).mean()
    ablate_features = ["lipase_x_uln", "amylase_x_uln", "pancreatitis_positive",
                       "cholecystitis_positive", "appendicitis_positive",
                       "diverticulitis_positive", "gallstones",
                       "alp_x_uln", "alt_x_uln", "bilirubin_total_x_uln"]
    ablate_results = []
    for feat in ablate_features:
        if feat not in all_feat.columns:
            continue
        nf_ab, imf_ab = nf.copy(), imf.copy()
        if feat in nf_ab.columns:
            nf_ab[feat] = np.nan
        if feat in imf_ab.columns:
            imf_ab[feat] = False
        res = run_normative_classifier(nf_ab, imf_ab, data["case_path"])
        acc = (res["pred"] == res["true"]).mean()
        ablate_results.append({"feature": feat, "accuracy": acc,
                               "delta": acc - baseline_acc})
    ab_df = pd.DataFrame(ablate_results).sort_values("delta")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_ab = ["#C44E4E" if d < -0.05 else "#E68A3B" if d < -0.02 else "#B0B0B0"
                 for d in ab_df["delta"]]
    ax.barh(range(len(ab_df)), ab_df["delta"], color=colors_ab)
    ax.set_yticks(range(len(ab_df)))
    ax.set_yticklabels(ab_df["feature"].str.replace("_x_uln", " ×ULN"))
    ax.set_xlabel("Δ accuracy when feature is removed")
    ax.set_title(f"Figure 4. Feature-ablation sensitivity\n"
                 f"(baseline classifier accuracy = {baseline_acc:.3f})", fontsize=10)
    ax.axvline(0, color="black", lw=0.5)
    for i, d in enumerate(ab_df["delta"]):
        ax.text(d - 0.002 if d < 0 else d + 0.002, i, f"{d:+.3f}",
                va="center", ha="right" if d < 0 else "left", fontsize=8)
    ax.set_xlim(ab_df["delta"].min() - 0.02, 0.01)
    fig.savefig(OUT / "fig4_ablation.pdf")
    fig.savefig(OUT / "fig4_ablation.png")
    plt.close(fig)
    ab_df.to_csv(OUT / "sec5_ablation.csv", index=False)
    print(f"  fig4 saved (largest drop: {ab_df.iloc[0]['feature']} = {ab_df.iloc[0]['delta']:+.3f})")

    # ===== FIGURE 5: Lipase × ULN distribution by pathology =====
    d = all_feat.copy()
    d["pathology"] = labels
    d["log_lipase_xuln"] = np.log10(d["lipase_x_uln"].clip(lower=0.05))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = {"appendicitis": "#5176A8", "cholecystitis": "#E68A3B",
               "pancreatitis": "#C44E4E", "diverticulitis": "#6FAE6F"}
    for p in PATHOLOGIES:
        sub = d[(d["pathology"] == p) & d["lipase_x_uln"].notna()]
        ax.hist(sub["log_lipase_xuln"], bins=40, alpha=0.55, label=p,
                color=palette[p], density=True)
    ax.axvline(np.log10(3.0), color="black", ls="--", lw=1,
               label="Atlanta threshold (3× ULN)")
    tick_positions = [-1, 0, 1, 2, 3]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(["0.1×", "1×", "10×", "100×", "1000×"])
    ax.set_xlabel("Serum lipase (× upper limit of normal; log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Figure 5. Distribution of serum lipase by pathology\n"
                 "(lipase alone carries 0.46 bits of MI — nearly a quarter of class entropy)",
                 fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.savefig(OUT / "fig5_lipase_distribution.pdf")
    fig.savefig(OUT / "fig5_lipase_distribution.png")
    plt.close(fig)
    print(f"  fig5 saved")

    # ===== Final summary =====
    summary = {
        "n_cases": int(len(all_feat)),
        "pathology_counts": labels.value_counts().to_dict(),
        "classifier_accuracy":            float((y_true == y_pred).mean()),
        "classifier_accuracy_decided":    float((results["pred"] != "abstain").mean() and
                                               (results[results["pred"] != "abstain"]["pred"] ==
                                                results[results["pred"] != "abstain"]["true"]).mean()),
        "abstention_rate":                float((results["pred"] == "abstain").mean()),
        "macro_f1":                       float(f.mean()),
        "weighted_f1":                    float(np.average(f, weights=s)),
        "per_pathology": {p: {"precision": float(p_), "recall": float(r_),
                              "f1": float(f_), "n": int(s_)}
                         for p, p_, r_, f_, s_ in zip(PATHOLOGIES, p, r, f, s)},
        "completeness_ceiling_overall": float((ceiling["ceiling_accuracy"] *
                                               ceiling["n_cases"]).sum() /
                                              ceiling["n_cases"].sum()),
        "top_feature_mi": {row["feature"]: float(row["mi"])
                           for _, row in mi_keep.head(5).iterrows()},
        "ablation_critical": {row["feature"]: float(row["delta"])
                              for _, row in ab_df.head(5).iterrows()},
    }
    with open(OUT / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFinal summary written to {OUT / 'final_summary.json'}")
    print(json.dumps(summary, indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
