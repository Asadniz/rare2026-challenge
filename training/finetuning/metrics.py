import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_youden_threshold(y_true, y_scores):
    """Select the threshold that maximizes Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    return float(thresholds[best_idx])


def compute_challenge_metrics(y_true, y_scores, threshold=0.5):
    """
    Computes the same metrics as the challenge: AUROC, AUPRC, PPV@90% Recall, Accuracy, Sensitivity, and Specificity.

    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted scores/probabilities

    Returns:
        Dictionary with computed metrics
    """
    # AUROC & AUPRC
    auroc = roc_auc_score(y_true=y_true, y_score=y_scores)
    auprc = average_precision_score(y_true=y_true, y_score=y_scores)

    # Compute Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true=y_true, y_score=y_scores)

    # Find PPV @ 90% Recall
    ppv_90 = np.interp(0.9, recalls[::-1], precisions[::-1])

    # Convert scores to binary predictions at the provided threshold
    y_pred = (y_scores >= threshold).astype(int)

    # Compute Accuracy, Sensitivity, and Specificity
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "PPV@90% Recall": ppv_90,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


# Define metric functions at module level for pickling
def challenge_auroc(y_scores, y_true):
    """Compute AUROC."""
    return roc_auc_score(y_true, y_scores)


def challenge_auprc(y_scores, y_true):
    """Compute AUPRC."""
    return average_precision_score(y_true, y_scores)


def challenge_ppv_at_90_recall(y_scores, y_true):
    """Compute PPV at 90% recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    idx = np.where(recalls >= 0.9)[0][-1]
    return precisions[idx]


def challenge_accuracy(y_scores, y_true):
    """Compute accuracy with 0.5 threshold."""
    y_pred = (y_scores >= 0.5).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return (tp + tn) / (tp + tn + fp + fn)


def challenge_sensitivity(y_scores, y_true):
    """Compute sensitivity (recall) with 0.5 threshold."""
    y_pred = (y_scores >= 0.5).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def challenge_specificity(y_scores, y_true):
    """Compute specificity with 0.5 threshold."""
    y_pred = (y_scores >= 0.5).astype(int)
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def evaluate_with_challenge_metrics(
        y_pred,
        y_true,
        target_ratio=0.01,
        n_repetitions=1000,
        n_jobs=None,
        random_seed=None,
        verbose=False
):
    """
    Complete evaluation using the challenge's metrics and new sampling procedure.

    Args:
        y_pred: Model predictions
        y_true: True labels
        min_positive: Number of positive cases to sample per iteration (default: 1000)
        negative_multiplier: Multiplier for negative cases (default: 100)
        n_repetitions: Number of bootstrap repetitions (default: 1000)
        n_jobs: Number of parallel jobs
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        Dictionary with summary statistics (2.5%, 50%, 97.5% percentiles) for each metric
    """
    from .evaluation import ChallengeEvaluator

    # Create evaluator
    evaluator = ChallengeEvaluator(n_repetitions=n_repetitions, n_jobs=n_jobs, verbose=verbose)

    # Define all challenge metrics using module-level functions (can be pickled)
    challenge_metrics = {
        "AUROC": challenge_auroc,
        "AUPRC": challenge_auprc,
        "PPV@90% Recall": challenge_ppv_at_90_recall,
        "Accuracy": challenge_accuracy,
        "Sensitivity": challenge_sensitivity,
        "Specificity": challenge_specificity,
    }

    # Run evaluation with new sampling
    results = evaluator.evaluate_with_prevalence_correction(
        y_pred, y_true, challenge_metrics, target_ratio, random_seed
    )

    # Convert to challenge's format (2.5%, 50%, 97.5% percentiles)
    summary = {}
    for metric_name, stats in results.items():
        values = np.array(stats['all_values'])
        summary[metric_name] = {
            '2.5%': np.percentile(values, 2.5),
            '50%': np.percentile(values, 50),
            '97.5%': np.percentile(values, 97.5),
            'mean': stats['mean'],
            'std': stats['std']
        }

    return summary


def quick_challenge_evaluation_new_metrics(
        y_pred,
        y_true,
        target_ratio=0.01,
        n_repetitions=1000,
        n_jobs=None,
        random_seed=None,
        verbose=False
):
    """
    Quick evaluation that returns just the median values for all challenge metrics.

    Args:
        y_pred: Model predictions
        y_true: True labels
        min_positive: Number of positive cases to sample per iteration
        negative_multiplier: Multiplier for negative cases
        n_repetitions: Number of bootstrap repetitions
        n_jobs: Number of parallel jobs
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        Dictionary with median values for each metric
    """
    results = evaluate_with_challenge_metrics(
        y_pred, y_true, target_ratio,
        n_repetitions, n_jobs, random_seed, verbose
    )

    return {metric: stats['50%'] for metric, stats in results.items()}

def challenge_ppv(y_pred,
                  y_true,
                  target_ratio=0.01,
                  n_repetitions=1000,
                  n_jobs=None,
                  random_seed=None,
                  verbose=False):
    metrics = quick_challenge_evaluation_new_metrics(
        y_pred, y_true, target_ratio,
        n_repetitions, n_jobs, random_seed, verbose
    )
    return metrics['PPV@90% Recall']


# Example usage and comparison
if __name__ == "__main__":
    import pandas as pd

    # Generate example data
    np.random.seed(42)
    n_samples = 10000
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    y_pred = np.random.random(n_samples)
    y_pred[y_true == 1] += 0.3  # Make predictions correlated with labels
    y_pred = np.clip(y_pred, 0, 1)

    print("Computing challenge metrics with new sampling procedure...")

    # Method 1: Quick single computation using challenge's metric function
    sample_metrics = compute_challenge_metrics(y_true, y_pred)
    print("\nSingle computation (no bootstrap):")
    for metric, value in sample_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Method 2: Quick bootstrap evaluation (median values only)
    print("\nQuick bootstrap evaluation (median values):")
    quick_results = quick_challenge_evaluation_new_metrics(
        y_pred, y_true,
        target_ratio=0.01,
        n_repetitions=1000,  # Reduced for demo
        verbose=True,
        random_seed=42
    )

    for metric, value in quick_results.items():
        print(f"  {metric}: {value:.4f}")

    # Method 3: Full bootstrap evaluation with all challenge metrics
    print("\nFull bootstrap evaluation with confidence intervals:")
    bootstrap_results = evaluate_with_challenge_metrics(
        y_pred, y_true,
        target_ratio=0.01,
        n_repetitions=1000,  # Reduced for demo
        verbose=True,
        random_seed=42
    )

    print("\nBootstrap Results (2.5%, 50%, 97.5% percentiles):")
    for metric_name, stats in bootstrap_results.items():
        print(f"{metric_name}:")
        print(f"  2.5%: {stats['2.5%']:.4f}")
        print(f"  50%:  {stats['50%']:.4f}")
        print(f"  97.5%: {stats['97.5%']:.4f}")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Convert to DataFrame for easy viewing (like the challenge's output)
    df_summary = pd.DataFrame(bootstrap_results).T
    print("\nSummary as DataFrame:")
    print(df_summary.round(4))