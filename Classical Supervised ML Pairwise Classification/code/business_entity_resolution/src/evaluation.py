"""
Evaluation and error analysis module.
Computes macro F0.5, precision, recall, singleton performance, candidate recall,
and dissects errors between blocking misses and ML threshold rejections.
"""

from typing import Dict, Set, List, Optional, Tuple
import json
import sys
from pathlib import Path

try:
    from src.model import calculate_f05_score
except ImportError:
    from .model import calculate_f05_score




def evaluate_matching_results(
    gt_mapping: Dict[str, Set[str]],
    pred_mapping: Dict[str, Set[str]],
    candidate_mapping: Optional[Dict[str, Set[str]]] = None
) -> Dict[str, float]:
    """
    Perform evaluation across all evaluation entities.
    Returns dictionary with all official metrics and diagnostic indicators.
    """
    s1_entities = list(gt_mapping.keys())
    n = len(s1_entities)

    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0

    singletons_total = 0
    singletons_correct = 0

    non_singletons_total = 0
    non_singletons_f05 = 0.0

    total_true_links = 0
    recalled_in_candidates = 0
    recalled_in_predictions = 0

    errors_blocking_miss = 0    # True match absent from candidate set
    errors_model_rejection = 0   # True match in candidates but rejected by ML
    false_positives = 0         # Predicted match not in GT

    subset_violations = 0

    for s1_id in s1_entities:
        gt_ids = gt_mapping.get(s1_id, set())
        pred_ids = pred_mapping.get(s1_id, set())
        cand_ids = candidate_mapping.get(s1_id, set()) if candidate_mapping is not None else None

        # Check subset invariant: pred_ids <= cand_ids
        if cand_ids is not None:
            if not pred_ids.issubset(cand_ids):
                subset_violations += 1

        is_singleton = (len(gt_ids) == 0)
        total_true_links += len(gt_ids)

        if is_singleton:
            singletons_total += 1
            if len(pred_ids) == 0:
                singletons_correct += 1
                total_f05 += 1.0
                total_prec += 1.0
                total_rec += 1.0
            else:
                false_positives += len(pred_ids)
                total_f05 += 0.0
                total_prec += 0.0
                total_rec += 0.0
        else:
            non_singletons_total += 1
            tp = len(gt_ids & pred_ids)
            recalled_in_predictions += tp

            if cand_ids is not None:
                cand_tp = len(gt_ids & cand_ids)
                recalled_in_candidates += cand_tp
                # Error breakdown
                for gid in gt_ids:
                    if gid not in cand_ids:
                        errors_blocking_miss += 1
                    elif gid not in pred_ids:
                        errors_model_rejection += 1

            if len(pred_ids) == 0:
                f05 = 0.0
                prec = 0.0
                rec = 0.0
            else:
                prec = tp / len(pred_ids)
                rec = tp / len(gt_ids)
                f05 = calculate_f05_score(prec, rec)
                false_positives += len(pred_ids - gt_ids)

            total_f05 += f05
            total_prec += prec
            total_rec += rec
            non_singletons_f05 += f05

    macro_f05 = total_f05 / n if n > 0 else 0.0
    macro_prec = total_prec / n if n > 0 else 0.0
    macro_rec = total_rec / n if n > 0 else 0.0
    singleton_acc = singletons_correct / singletons_total if singletons_total > 0 else 1.0
    non_singleton_f05 = non_singletons_f05 / non_singletons_total if non_singletons_total > 0 else 0.0

    cand_recall = (recalled_in_candidates / total_true_links) if total_true_links > 0 else 1.0
    pred_recall = (recalled_in_predictions / total_true_links) if total_true_links > 0 else 1.0

    metrics = {
        "macro_f05": macro_f05,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "singleton_accuracy": singleton_acc,
        "non_singleton_f05": non_singleton_f05,
        "candidate_recall": cand_recall,
        "prediction_recall": pred_recall,
        "total_entities": n,
        "singletons_count": singletons_total,
        "non_singletons_count": non_singletons_total,
        "total_true_links": total_true_links,
        "errors_blocking_miss": errors_blocking_miss,
        "errors_model_rejection": errors_model_rejection,
        "false_positives": false_positives,
        "subset_violations": subset_violations
    }

    return metrics


def print_evaluation_report(metrics: Dict[str, float], report_path: Optional[Path] = None) -> None:
    """Print structured metrics table and optionally save to file."""
    lines = [
        "=" * 70,
        "EVALUATION & VALIDATION REPORT",
        "=" * 70,
        f"  Total S1 Entities Evaluated : {int(metrics['total_entities']):,}",
        f"  Singletons (Zero Matches)   : {int(metrics['singletons_count']):,} ({metrics['singletons_count']/metrics['total_entities']*100:.1f}%)",
        f"  Non-Singletons              : {int(metrics['non_singletons_count']):,}",
        f"  Total True Matches in GT    : {int(metrics['total_true_links']):,}",
        "-" * 70,
        f"  Macro F0.5 Score            : {metrics['macro_f05']:.4f}",
        f"  Macro Precision             : {metrics['macro_precision']:.4f}",
        f"  Macro Recall                : {metrics['macro_recall']:.4f}",
        f"  Singleton Accuracy          : {metrics['singleton_accuracy']:.4f}",
        f"  Non-Singleton F0.5          : {metrics['non_singleton_f05']:.4f}",
        "-" * 70,
        f"  Candidate Set Recall        : {metrics['candidate_recall']:.4f}",
        f"  Subset Violations (Pred not in Cand): {int(metrics['subset_violations'])}",
        "-" * 70,
        "  Error Breakdown:",
        f"    - True Matches Missed by Blocking  : {int(metrics['errors_blocking_miss']):,}",
        f"    - True Matches Rejected by ML Model : {int(metrics['errors_model_rejection']):,}",
        f"    - False Positive Predictions        : {int(metrics['false_positives']):,}",
        "=" * 70,
    ]
    report_text = "\n".join(lines)
    print(report_text)

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        json_path = report_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
