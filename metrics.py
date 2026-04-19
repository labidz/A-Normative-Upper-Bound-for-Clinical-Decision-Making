"""
metrics.py
==========
Four deviation metrics for quantifying LLM diagnostic trajectory adherence to
a normative clinical guideline.

Formal definitions:

1. Coverage (Cov).
   Cov(t, g) = |E_t ∩ E_g^req| / |E_g^req|
   where E_t is the set of evidence categories requested in trajectory t,
   and E_g^req is the guideline's required evidence set.  Range [0, 1], 1 = perfect.

2. Spurious-request rate (Spur).
   Spur(t, g) = |E_t \\ E_g^pri| / |E_t|
   fraction of a trajectory's requests that are not in the guideline's
   priority order. Range [0, 1], 0 = perfect.

3. Ordering distance (OrdD).
   Normalized Kendall-tau distance between the trajectory's request sequence
   (filtered to categories present in E_g^pri) and the guideline's priority order.
   OrdD = kendall_tau_distance(σ_t, σ_g) / C(n, 2)
   Range [0, 1], 0 = perfect ordering.

4. Criterion fulfillment at commit (CFC).
   Binary: does the evidence set collected at the moment of diagnostic
   commitment satisfy the guideline's diagnostic rule?
   CFC(t, g) ∈ {0, 1}.

Composite: Guideline Deviation Index (GDI) ∈ [0, 1], lower = better.
   GDI = 0.35*(1 - Cov) + 0.25*Spur + 0.20*OrdD + 0.20*(1 - CFC)

Weights chosen to prioritize coverage and criterion fulfillment (clinically
most consequential) over ordering (least consequential in acute settings).
Sensitivity analysis over weight choice is included in the analysis script.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class TrajectoryRecord:
    """One LLM × one case."""
    hadm_id: int
    pathology: str
    model: str
    requested_sequence: List[str]        # ordered list of evidence categories
    evidence_at_commit: Set[str]         # evidence collected at time of dx
    predicted_diagnosis: str
    correct: bool                        # vs. ground-truth discharge_diagnosis
    n_turns: int


def _kendall_tau_distance(seq_a: List[str], seq_b: List[str]) -> float:
    """Normalized Kendall-tau distance between two orderings over the
    intersection of their element sets. Returns 0 if intersection < 2.
    """
    common = [x for x in seq_a if x in seq_b]
    if len(common) < 2:
        return 0.0
    # positions in seq_b
    pos_b = {x: i for i, x in enumerate(seq_b)}
    n = len(common)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos_b[common[i]] > pos_b[common[j]]:
                inversions += 1
    return inversions / (n * (n - 1) / 2)


def coverage(traj: TrajectoryRecord, required: Set[str]) -> float:
    if not required:
        return 1.0
    return len(set(traj.requested_sequence) & required) / len(required)


def spurious_rate(traj: TrajectoryRecord, priority_order: List[str]) -> float:
    if not traj.requested_sequence:
        return 0.0
    guideline_set = set(priority_order)
    requested_unique = set(traj.requested_sequence)
    return len(requested_unique - guideline_set) / len(requested_unique)


def ordering_distance(traj: TrajectoryRecord, priority_order: List[str]) -> float:
    # Deduplicate trajectory while preserving first-occurrence order
    seen = set()
    seq_t = []
    for e in traj.requested_sequence:
        if e not in seen:
            seen.add(e)
            seq_t.append(e)
    return _kendall_tau_distance(seq_t, priority_order)


def criterion_fulfillment(traj: TrajectoryRecord, rule_fn) -> int:
    fulfilled, _ = rule_fn(traj.evidence_at_commit)
    return int(fulfilled)


def guideline_deviation_index(
    traj: TrajectoryRecord,
    required: Set[str],
    priority_order: List[str],
    rule_fn,
    w: Tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20),
) -> Dict[str, float]:
    cov  = coverage(traj, required)
    spur = spurious_rate(traj, priority_order)
    ordd = ordering_distance(traj, priority_order)
    cfc  = criterion_fulfillment(traj, rule_fn)
    gdi  = w[0]*(1 - cov) + w[1]*spur + w[2]*ordd + w[3]*(1 - cfc)
    return {"coverage": cov, "spurious_rate": spur,
            "ordering_distance": ordd, "criterion_fulfillment": cfc,
            "gdi": gdi}


def compute_all_metrics(traj: TrajectoryRecord, guideline) -> Dict[str, float]:
    return guideline_deviation_index(
        traj,
        required=set(guideline.required_evidence),
        priority_order=guideline.priority_order,
        rule_fn=guideline.diagnostic_rule,
    )
