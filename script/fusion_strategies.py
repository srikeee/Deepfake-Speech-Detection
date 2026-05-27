"""
Improved Fusion Strategies for Deepfake Audio Detection
=======================================================

PURPOSE:
    Replaces the naive accuracy-weighted probability average which produces
    biased predictions (high fake-recall, near-zero real-recall).

ROOT CAUSE OF ORIGINAL FAILURE:
    - Accuracy weights do not account for class-conditional performance.
    - Models that are biased toward FAKE get high accuracy on FAKE-majority data,
      making their weights large, which amplifies the FAKE bias further.
    - A fixed threshold of 0.5 is miscalibrated for a biased score distribution.

THREE STRATEGIES PROVIDED:
    1. Class-Precision Weighted Fusion   (recommended default)
    2. Calibrated Threshold Fusion       (best when calibration data is available)
    3. Disagreement-Dampened Fusion      (best for academic explainability)

INPUT CONTRACT (must match your evaluate_models.py output):
    results = {
        'wav2vec':  {'accuracy': float,              # 0-100
                     'probabilities': List[float],   # P(fake) per sample, 0-1
                     'labels': List[int]},           # 0=real, 1=fake
        'mel_cnn':  { same structure },
        'mfcc':     { same structure }
    }

OUTPUT CONTRACT (all three strategies return):
    {
        'final_label':      str,    # "REAL" or "FAKE"
        'final_score':      float,  # raw fused score (0-1), higher = more likely FAKE
        'confidence':       float,  # calibrated confidence of the decision (0-1)
        'decision_string':  str,    # e.g. "FAKE (confidence: 0.87)"
        'per_model_weights': dict,  # model → weight used in fusion
        'diagnostics':       dict   # additional info for viva/reporting
    }
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Optional, Tuple


# =============================================================================
#  SHARED UTILITIES
# =============================================================================

MODEL_KEYS = ['wav2vec', 'mel_cnn', 'mfcc']


def _validate_results(results: dict) -> List[str]:
    """Return list of available model keys with valid data."""
    available = []
    for key in MODEL_KEYS:
        r = results.get(key)
        if r and r.get('probabilities') and r.get('labels'):
            available.append(key)
    if not available:
        raise ValueError("No models with valid probabilities found in results dict.")
    return available


def _get_arrays(results, available_models):
    """
    Extract and align probability and label arrays across models.
    Alignment is done by truncating to the minimum available length.
    """

    probs_list = []
    labels_list = []

    for m in available_models:
        probs_list.append(np.array(results[m]['probabilities']))
        labels_list.append(np.array(results[m]['labels']))

    # --------------------------------------------------
    # CRITICAL FIX: ALIGN SAMPLE COUNTS
    # --------------------------------------------------
    min_len = min(len(p) for p in probs_list)

    probs_list = [p[:min_len] for p in probs_list]
    labels_list = [l[:min_len] for l in labels_list]

    probs_matrix = np.stack(probs_list, axis=0)  # (n_models, n_samples)
    labels = labels_list[0]

    return probs_matrix, labels, min_len


def _per_class_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute per-class precision, recall, and confusion matrix for one model.
    """
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, cm[1, 1])

    real_precision = tn / (tn + fn + 1e-9)   # precision for predicting REAL
    real_recall    = tn / (tn + fp + 1e-9)   # recall for REAL class (true negatives)
    fake_precision = tp / (tp + fp + 1e-9)
    fake_recall    = tp / (tp + fn + 1e-9)

    return {
        'real_precision': real_precision,
        'real_recall':    real_recall,
        'fake_precision': fake_precision,
        'fake_recall':    fake_recall,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


def _decision_string(label: str, confidence: float) -> str:
    return f"{label} (confidence: {confidence:.4f})"


def _confidence_from_margin(score: float, threshold: float) -> float:
    """
    Compute confidence as the normalized distance of the fused score from the threshold.
    A score far from the threshold = high confidence.
    A score exactly at the threshold = confidence 0.5.

    Maps the margin linearly into [0, 1]:
        - score = threshold         → confidence = 0.5
        - score = 0.0 or 1.0       → confidence = 1.0
    """
    margin = abs(score - threshold)               # 0 to max(threshold, 1-threshold)
    max_possible = max(threshold, 1.0 - threshold)
    normalized = margin / (max_possible + 1e-9)   # 0 to 1
    # Remap so that max margin → 1.0, zero margin → 0.5
    return 0.5 + 0.5 * normalized


# =============================================================================
#  STRATEGY 1 — CLASS-PRECISION WEIGHTED FUSION
# =============================================================================

def class_precision_weighted_fusion(
    results: dict,
    threshold: float = 0.5,
    sample_idx: Optional[int] = None,
) -> dict:
    """
    CLASS-PRECISION WEIGHTED FUSION
    ================================

    WHY IT FIXES THE BIAS:
        Standard accuracy weighting gives high weight to models that are
        accurate overall, but those models may still be biased toward FAKE.
        This strategy weights each model differently depending on which class
        is currently being decided:

        - For the FAKE side:  weight by the model's FAKE precision
          (how reliable is this model when it says FAKE?)
        - For the REAL side:  weight by the model's REAL precision
          (how reliable is this model when it says REAL?)

        This penalizes models that generate too many false FAKEs and gives
        more trust to models that are precise for each class.

    FUSION FORMULA:
        w_fake_i = fake_precision_i / sum(fake_precision)
        w_real_i = real_precision_i / sum(real_precision)

        fake_score = sum(w_fake_i * P_fake_i)    # weighted P(fake)
        real_score = sum(w_real_i * (1 - P_fake_i))  # weighted P(real)

        # Renormalize to a single unified score
        final_score = fake_score / (fake_score + real_score)

        final_label = "FAKE" if final_score >= threshold else "REAL"

    CONFIDENCE:
        Margin of final_score from threshold, mapped to [0.5, 1.0].
        A score of 0.9 with threshold 0.5 → confidence ≈ 0.9.

    Args:
        results:     dict from evaluate_models.py
        threshold:   decision boundary (default 0.5, tune with calibration)
        sample_idx:  if int, evaluate only that sample; else evaluate all samples

    Returns:
        dict with final_label, final_score, confidence, decision_string,
             per_model_weights, diagnostics
    """
    available = _validate_results(results)
    probs_matrix, labels, _ = _get_arrays(results, available)

    # --- Compute per-class precision for each model ---
    metrics = {}
    for i, key in enumerate(available):
        metrics[key] = _per_class_metrics(probs_matrix[i], labels)

    fake_precisions = np.array([metrics[k]['fake_precision'] for k in available])
    real_precisions = np.array([metrics[k]['real_precision'] for k in available])

    # Normalize to get weights (add epsilon to prevent all-zero)
    w_fake = fake_precisions / (fake_precisions.sum() + 1e-9)
    w_real = real_precisions / (real_precisions.sum() + 1e-9)

    # --- Apply to target sample(s) ---
    if sample_idx is not None:
        probs_vec = probs_matrix[:, sample_idx]  # (n_models,)
        return _apply_class_precision_fusion(
            probs_vec, w_fake, w_real, threshold,
            available, metrics
        )

    # Batch mode: return summary stats
    n_samples = probs_matrix.shape[1]
    all_results = []
    for j in range(n_samples):
        r = _apply_class_precision_fusion(
            probs_matrix[:, j], w_fake, w_real, threshold,
            available, metrics
        )
        all_results.append(r)

    final_scores  = np.array([r['final_score'] for r in all_results])
    final_preds   = np.array([1 if r['final_label'] == 'FAKE' else 0 for r in all_results])
    accuracy      = 100.0 * (final_preds == labels).sum() / len(labels)
    real_recall   = recall_score(labels, final_preds, pos_label=0, zero_division=0)
    fake_recall   = recall_score(labels, final_preds, pos_label=1, zero_division=0)
    real_prec     = precision_score(labels, final_preds, pos_label=0, zero_division=0)
    fake_prec     = precision_score(labels, final_preds, pos_label=1, zero_division=0)

    return {
        'strategy':           'class_precision_weighted',
        'fusion_accuracy':    accuracy,
        'real_recall':        real_recall,
        'fake_recall':        fake_recall,
        'real_precision':     real_prec,
        'fake_precision':     fake_prec,
        'threshold_used':     threshold,
        'per_model_weights':  {'fake_weights': dict(zip(available, w_fake.tolist())),
                               'real_weights': dict(zip(available, w_real.tolist()))},
        'per_model_metrics':  metrics,
        'sample_results':     all_results,
    }


def _apply_class_precision_fusion(
    probs_vec, w_fake, w_real, threshold, model_keys, metrics
):
    """Apply class-precision fusion to a single sample's probabilities."""
    fake_score = float(np.dot(w_fake, probs_vec))
    real_score = float(np.dot(w_real, 1.0 - probs_vec))

    # Normalize so scores sum to 1
    total = fake_score + real_score + 1e-9
    final_score = fake_score / total

    final_label = "FAKE" if final_score >= threshold else "REAL"
    confidence  = _confidence_from_margin(final_score, threshold)

    return {
        'final_label':       final_label,
        'final_score':       round(final_score, 6),
        'confidence':        round(confidence, 6),
        'decision_string':   _decision_string(final_label, confidence),
        'per_model_weights': {'fake_weights': dict(zip(model_keys, w_fake.tolist())),
                              'real_weights': dict(zip(model_keys, w_real.tolist()))},
        'diagnostics': {
            'weighted_fake_score': round(fake_score, 6),
            'weighted_real_score': round(real_score, 6),
            'raw_probs_per_model': dict(zip(model_keys, probs_vec.tolist())),
        }
    }


# =============================================================================
#  STRATEGY 2 — CALIBRATED THRESHOLD FUSION
# =============================================================================

def find_optimal_threshold(
    results: dict,
    metric: str = 'balanced_accuracy',
    n_thresholds: int = 100,
) -> dict:
    available = _validate_results(results)
    probs_matrix, labels, _ = _get_arrays(results, available)

    # 🔧 FIX: derive accuracies correctly
    accuracies = np.array([results[m]['accuracy'] / 100.0 for m in available])
    weights = accuracies / (accuracies.sum() + 1e-9)

    fused_scores = probs_matrix.T @ weights  # (n_samples,)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    metric_values = []

    for t in thresholds:
        preds = (fused_scores >= t).astype(int)

        if metric == 'balanced_accuracy':
            rr = recall_score(labels, preds, pos_label=0, zero_division=0)
            fr = recall_score(labels, preds, pos_label=1, zero_division=0)
            val = (rr + fr) / 2.0
        elif metric == 'f1_fake':
            val = f1_score(labels, preds, pos_label=1, zero_division=0)
        elif metric == 'f1_real':
            val = f1_score(labels, preds, pos_label=0, zero_division=0)
        elif metric == 'f1_macro':
            val = f1_score(labels, preds, average='macro', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        metric_values.append(val)

    best_idx = int(np.argmax(metric_values))

    return {
        'optimal_threshold': float(thresholds[best_idx]),
        'best_metric_value': float(metric_values[best_idx]),
        'metric_used': metric,
        'threshold_sweep': list(zip(thresholds.tolist(), metric_values)),
    }


def calibrated_threshold_fusion(
    results: dict,
    threshold: Optional[float] = None,
    calibration_metric: str = 'balanced_accuracy',
    sample_idx: Optional[int] = None,
) -> dict:
    available = _validate_results(results)
    probs_matrix, labels, _ = _get_arrays(results, available)

    # 🔧 FIX: compute accuracies correctly
    accuracies = np.array([results[m]['accuracy'] / 100.0 for m in available])
    weights = accuracies / (accuracies.sum() + 1e-9)

    # Calibrate threshold if not provided
    if threshold is None:
        calib = find_optimal_threshold(results, metric=calibration_metric)
        threshold = calib['optimal_threshold']
        calib_info = calib
    else:
        calib_info = {'threshold_source': 'user_provided'}

    # --- Single sample mode ---
    if sample_idx is not None:
        probs_vec = probs_matrix[:, sample_idx]
        return _apply_calibrated_fusion(
            probs_vec, weights, threshold, available,
            calib_info, calibration_metric
        )

    # --- Batch mode ---
    n_samples = probs_matrix.shape[1]
    all_results = []
    for j in range(n_samples):
        r = _apply_calibrated_fusion(
            probs_matrix[:, j], weights, threshold,
            available, calib_info, calibration_metric
        )
        all_results.append(r)

    final_preds = np.array([1 if r['final_label'] == 'FAKE' else 0 for r in all_results])
    accuracy     = 100.0 * (final_preds == labels).sum() / len(labels)
    real_recall  = recall_score(labels, final_preds, pos_label=0, zero_division=0)
    fake_recall  = recall_score(labels, final_preds, pos_label=1, zero_division=0)
    real_prec    = precision_score(labels, final_preds, pos_label=0, zero_division=0)
    fake_prec    = precision_score(labels, final_preds, pos_label=1, zero_division=0)

    return {
        'strategy':           'calibrated_threshold',
        'fusion_accuracy':    accuracy,
        'real_recall':        real_recall,
        'fake_recall':        fake_recall,
        'real_precision':     real_prec,
        'fake_precision':     fake_prec,
        'calibrated_threshold': threshold,
        'calibration_info':   calib_info,
        'per_model_weights':  dict(zip(available, weights.tolist())),
        'sample_results':     all_results,
    }


def _apply_calibrated_fusion(probs_vec, weights, threshold, model_keys, calib_info, metric):
    """Apply calibrated-threshold fusion to a single sample."""
    final_score = float(np.dot(weights, probs_vec))
    final_label = "FAKE" if final_score >= threshold else "REAL"

    # Asymmetric confidence: distance from threshold, normalized by available range
    if final_score >= threshold:
        confidence = (final_score - threshold) / (1.0 - threshold + 1e-9)
    else:
        confidence = (threshold - final_score) / (threshold + 1e-9)

    # Remap to [0.5, 1.0] so minimum confidence at boundary = 0.5
    confidence = 0.5 + 0.5 * confidence

    return {
        'final_label':      final_label,
        'final_score':      round(final_score, 6),
        'confidence':       round(confidence, 6),
        'decision_string':  _decision_string(final_label, confidence),
        'per_model_weights': dict(zip(model_keys, weights.tolist())),
        'diagnostics': {
            'calibrated_threshold': threshold,
            'calibration_metric':   metric,
            'raw_probs_per_model':  dict(zip(model_keys, probs_vec.tolist())),
        }
    }


# =============================================================================
#  STRATEGY 3 — DISAGREEMENT-DAMPENED FUSION
# =============================================================================

def disagreement_dampened_fusion(
    results: dict,
    threshold: float = 0.5,
    agreement_boost: float = 0.2,
    sample_idx: Optional[int] = None,
) -> dict:
    available = _validate_results(results)
    probs_matrix, labels, _ = _get_arrays(results, available)

    # 🔧 FIX: compute accuracies correctly
    accuracies = np.array([results[m]['accuracy'] / 100.0 for m in available])
    weights = accuracies / (accuracies.sum() + 1e-9)

    if sample_idx is not None:
        probs_vec = probs_matrix[:, sample_idx]
        return _apply_disagreement_fusion(
            probs_vec, weights, threshold, agreement_boost, available
        )

    # Batch mode
    n_samples = probs_matrix.shape[1]
    all_results = []
    for j in range(n_samples):
        r = _apply_disagreement_fusion(
            probs_matrix[:, j], weights, threshold, agreement_boost, available
        )
        all_results.append(r)

    final_preds = np.array([1 if r['final_label'] == 'FAKE' else 0 for r in all_results])
    accuracy     = 100.0 * (final_preds == labels).sum() / len(labels)
    real_recall  = recall_score(labels, final_preds, pos_label=0, zero_division=0)
    fake_recall  = recall_score(labels, final_preds, pos_label=1, zero_division=0)
    real_prec    = precision_score(labels, final_preds, pos_label=0, zero_division=0)
    fake_prec    = precision_score(labels, final_preds, pos_label=1, zero_division=0)

    return {
        'strategy':          'disagreement_dampened',
        'fusion_accuracy':   accuracy,
        'real_recall':       real_recall,
        'fake_recall':       fake_recall,
        'real_precision':    real_prec,
        'fake_precision':    fake_prec,
        'threshold_used':    threshold,
        'agreement_boost':   agreement_boost,
        'per_model_weights': dict(zip(available, weights.tolist())),
        'sample_results':    all_results,
    }


def _apply_disagreement_fusion(probs_vec, weights, threshold, agreement_boost, model_keys):
    """Apply disagreement-dampened fusion to a single sample."""
    n_models = len(probs_vec)

    # Binary predictions from each model at threshold 0.5
    binary_preds = (probs_vec >= 0.5).astype(int)
    majority_vote = int(np.round(binary_preds.mean()))  # 0 or 1
    n_agreeing    = int((binary_preds == majority_vote).sum())
    agreement_ratio = n_agreeing / n_models  # 1.0 = unanimous

    # Base accuracy-weighted score
    base_score = float(np.dot(weights, probs_vec))

    # Damp toward 0.5 when models disagree
    disagreement = 1.0 - agreement_ratio
    final_score  = base_score * (1.0 - disagreement) + 0.5 * disagreement

    final_label  = "FAKE" if final_score >= threshold else "REAL"

    # Confidence = agreement_ratio × margin from threshold, remapped to [0.5, 1.0]
    margin = abs(final_score - threshold)
    max_margin = max(threshold, 1.0 - threshold)
    normalized_margin = margin / (max_margin + 1e-9)
    confidence = agreement_ratio * (0.5 + 0.5 * normalized_margin)

    # Add agreement boost for unanimous agreement
    if agreement_ratio == 1.0:
        confidence = min(1.0, confidence + agreement_boost)

    return {
        'final_label':      final_label,
        'final_score':      round(final_score, 6),
        'confidence':       round(confidence, 6),
        'decision_string':  _decision_string(final_label, confidence),
        'per_model_weights': dict(zip(model_keys, weights.tolist())),
        'diagnostics': {
            'base_score':         round(base_score, 6),
            'agreement_ratio':    round(agreement_ratio, 4),
            'disagreement_penalty': round(1.0 - agreement_ratio, 4),
            'model_binary_votes': dict(zip(model_keys, binary_preds.tolist())),
            'majority_vote':      majority_vote,
            'raw_probs_per_model': dict(zip(model_keys, probs_vec.tolist())),
        }
    }


# =============================================================================
#  ENSEMBLE RUNNER — run all three and compare
# =============================================================================

def run_all_strategies(
    results: dict,
    calibration_metric: str = 'balanced_accuracy',
    verbose: bool = True,
) -> dict:
    """
    Run all three fusion strategies on the full validation set and print
    a comparative summary.

    Args:
        results:             dict from evaluate_models.py (with probabilities + labels)
        calibration_metric:  threshold calibration target for strategy 2
        verbose:             if True, print a formatted comparison table

    Returns:
        dict keyed by strategy name, each containing full strategy output
    """
    # ------------------------------------------------------------------ #
    # Strategy 1 — Class-Precision Weighted
    # ------------------------------------------------------------------ #
    s1 = class_precision_weighted_fusion(results, threshold=0.5)

    # ------------------------------------------------------------------ #
    # Strategy 2 — Calibrated Threshold (auto-finds optimal threshold)
    # ------------------------------------------------------------------ #
    s2 = calibrated_threshold_fusion(results, calibration_metric=calibration_metric)

    # ------------------------------------------------------------------ #
    # Strategy 3 — Disagreement-Dampened
    # ------------------------------------------------------------------ #
    s3 = disagreement_dampened_fusion(results, threshold=0.5)

    all_strategies = {
        'class_precision_weighted': s1,
        'calibrated_threshold':     s2,
        'disagreement_dampened':    s3,
    }

    if verbose:
        _print_comparison(all_strategies, results, calibration_metric)

    return all_strategies


def _print_comparison(all_strategies: dict, results: dict, calibration_metric: str):
    """Print a formatted comparison table of all fusion strategies."""
    print("\n" + "=" * 70)
    print("  FUSION STRATEGY COMPARISON")
    print("=" * 70)

    # Individual model baselines
    print("\n  Individual Model Performance (Baseline):")
    print("  " + "-" * 50)
    available = _validate_results(results)
    probs_matrix, labels, _ = _get_arrays(results, available)

    for i, key in enumerate(available):
        preds = (probs_matrix[i] >= 0.5).astype(int)
        acc   = 100.0 * (preds == labels).mean()
        rr    = recall_score(labels, preds, pos_label=0, zero_division=0)
        fr    = recall_score(labels, preds, pos_label=1, zero_division=0)
        print(f"  {key.upper():<14} Acc: {acc:5.1f}%  "
              f"Real-Recall: {rr:.3f}  Fake-Recall: {fr:.3f}")

    print("\n  Fused Strategy Performance:")
    print("  " + "-" * 50)
    print(f"  {'Strategy':<32} {'Acc':>6}  {'R-Recall':>9}  {'F-Recall':>9}  {'R-Prec':>7}  {'F-Prec':>7}")
    print("  " + "-" * 70)

    strategy_labels = {
        'class_precision_weighted': 'Class-Precision Weighted',
        'calibrated_threshold':     f'Calibrated ({calibration_metric[:8]})',
        'disagreement_dampened':    'Disagreement-Dampened',
    }

    for key, s in all_strategies.items():
        name  = strategy_labels[key]
        acc   = s['fusion_accuracy']
        rr    = s['real_recall']
        fr    = s['fake_recall']
        rp    = s['real_precision']
        fp    = s['fake_precision']
        print(f"  {name:<32} {acc:>5.1f}%  {rr:>9.3f}  {fr:>9.3f}  {rp:>7.3f}  {fp:>7.3f}")

    print("\n  Notes:")
    if 'calibrated_threshold' in all_strategies:
        t = all_strategies['calibrated_threshold']['calibrated_threshold']
        print(f"    Calibrated threshold = {t:.4f}  "
              f"(optimized for {calibration_metric})")
    print("=" * 70)
