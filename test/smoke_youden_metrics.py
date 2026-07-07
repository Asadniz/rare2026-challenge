"""Smoke tests for Youden cutoff and expanded metrics (no training/data required)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training" / "finetuning"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


validation_metrics = _load_module("validation_metrics", ROOT / "validation" / "metrics.py")
evaluation_mod = _load_module("evaluation", ROOT / "training" / "evaluation.py")
finetuning_metrics = _load_module("finetuning_metrics", ROOT / "training" / "finetuning" / "metrics.py")

compute_extended_metrics = validation_metrics.compute_extended_metrics
compute_youden_threshold_gpu = validation_metrics.compute_youden_threshold_gpu
Evaluator = evaluation_mod.Evaluator
compute_challenge_metrics = finetuning_metrics.compute_challenge_metrics
compute_youden_threshold = finetuning_metrics.compute_youden_threshold


class ToyDataset(Dataset):
    def __init__(self, n_samples: int = 40, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, 2, size=n_samples)
        self.features = rng.normal(size=(n_samples, 4)).astype(np.float32)
        self.paths = [f"sample_{i}.png" for i in range(n_samples)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], int(self.labels[idx]), self.paths[idx]


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def make_scores_and_labels(n: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    scores = rng.random(n)
    scores[y_true == 1] += 0.35
    scores = np.clip(scores, 0.0, 1.0)
    return y_true, scores


def test_youden_thresholds_agree():
    y_true, scores = make_scores_and_labels()
    t_numpy = compute_youden_threshold(y_true, scores)
    t_gpu = compute_youden_threshold_gpu(
        torch.tensor(scores, dtype=torch.float32),
        torch.tensor(y_true, dtype=torch.float32),
    )
    assert 0.0 <= t_numpy <= 1.0
    assert abs(t_numpy - t_gpu) < 1e-4, f"numpy={t_numpy}, gpu={t_gpu}"
    print(f"[ok] Youden thresholds agree: {t_numpy:.4f}")


def test_probability_metrics_ignore_threshold():
    y_true, scores = make_scores_and_labels()
    threshold_a = 0.2
    threshold_b = 0.8
    metrics_a = compute_extended_metrics(scores, y_true, threshold_a)
    metrics_b = compute_extended_metrics(scores, y_true, threshold_b)

    for key in ("auroc", "auprc"):
        assert metrics_a[key] == metrics_b[key], f"{key} should not depend on threshold"
    assert metrics_a["f1"] != metrics_b["f1"] or threshold_a == threshold_b
    print("[ok] Probability metrics are threshold-independent")
    print(f"     prob metrics: auroc={metrics_a['auroc']:.4f}, auprc={metrics_a['auprc']:.4f}")


def test_class_metrics_use_threshold():
    y_true, scores = make_scores_and_labels()
    threshold = compute_youden_threshold(y_true, scores)
    metrics = compute_extended_metrics(scores, y_true, threshold)
    y_pred = (scores >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    expected_acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    assert abs(metrics["accuracy"] - expected_acc) < 1e-8
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    print("[ok] Class metrics computed at Youden threshold")
    print(
        f"     class metrics: f1={metrics['f1']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"
    )


def test_finetuning_challenge_metrics():
    y_true, scores = make_scores_and_labels()
    threshold = compute_youden_threshold(y_true, scores)
    metrics = compute_challenge_metrics(y_true, scores, threshold=threshold)

    assert "AUROC" in metrics and "AUPRC" in metrics
    assert "F1" in metrics and "Balanced Accuracy" in metrics
    metrics_other = compute_challenge_metrics(y_true, scores, threshold=threshold + 0.25)
    assert metrics["AUROC"] == metrics_other["AUROC"]
    assert metrics["AUPRC"] == metrics_other["AUPRC"]
    print("[ok] Finetuning challenge metrics shape and prob/class split")


def test_evaluator_validate_epoch_unpacking():
    device = torch.device("cpu")
    evaluator = Evaluator(device)
    model = ToyClassifier().to(device)
    loader = DataLoader(ToyDataset(), batch_size=8, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    result = evaluator.validate_epoch(model, loader, criterion)
    assert len(result) == 8, f"validate_epoch should return 8 values, got {len(result)}"

    loss, acc, ppv, preds, labels, logits, paths, probs = result
    assert len(probs) == len(labels)
    assert len(logits) == len(labels)
    assert len(paths) == len(labels)
    print(f"[ok] Evaluator.validate_epoch returns 8 values (n={len(labels)}, ppv={ppv:.4f})")


def test_evaluator_split_metrics_and_predictions_df():
    device = torch.device("cpu")
    evaluator = Evaluator(device)
    y_true = np.array([0, 0, 1, 1])
    logits = np.array([[1.0, -1.0], [0.5, -0.5], [-0.5, 0.5], [-1.0, 1.0]], dtype=np.float32)
    threshold = 0.5
    metrics = evaluator.compute_split_metrics(logits, y_true, threshold)
    assert "auprc" in metrics and "f1" in metrics

    df = evaluator.create_predictions_dataframe(
        ["a.png", "b.png", "c.png", "d.png"],
        y_true,
        logits,
        fold=0,
        split_type="val",
        youden_threshold=threshold,
    )
    assert "probability" in df.columns
    assert "youden_threshold" in df.columns
    assert "decision" in df.columns
    assert list(df["decision"]) == [0, 0, 1, 1]
    print("[ok] Evaluator split metrics + predictions dataframe")


def test_evaluator_evaluate_split_with_mock_model():
    device = torch.device("cpu")
    evaluator = Evaluator(device)
    model = ToyClassifier().to(device)
    loader = DataLoader(ToyDataset(n_samples=32), batch_size=8, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    scores, labels, metrics = evaluator.evaluate_split(model, loader, criterion, threshold=0.5)
    assert len(scores) == len(labels)
    assert "auroc" in metrics and "balanced_accuracy" in metrics
    print("[ok] Evaluator.evaluate_split runs end-to-end")


def test_evaluator_analyze_predictions_original_vs_youden():
    device = torch.device("cpu")
    evaluator = Evaluator(device)
    y_true, scores = make_scores_and_labels()
    logits = np.stack([1.0 - scores, scores], axis=1).astype(np.float32)

    analysis = evaluator.analyze_predictions(logits, y_true)
    assert "original_metrics" in analysis and "youden_metrics" in analysis
    assert "challenge_ppv" in analysis
    assert analysis["original_threshold"] == 0.5
    assert 0.0 <= analysis["youden_threshold"] <= 1.0

    for key in ("auroc", "auprc"):
        assert analysis["original_metrics"][key] == analysis["youden_metrics"][key]
    print("[ok] analyze_predictions returns original vs youden metric suites")


def test_training_finalize_signature_static():
    source = (ROOT / "training" / "training.py").read_text()
    assert "def _evaluate_epoch_splits(self, model, train_loader, val_loader, criterion, epoch, fold):" in source
    assert "def _finalize_fold_evaluation(self, model, train_loader, val_loader, criterion, fold):" in source
    assert "return val_preds, val_labels_true, val_logits, val_paths, val_ppv, youden_threshold" in source
    print("[ok] Trainer epoch eval + finalize wiring present in training.py")


def main():
    tests = [
        test_youden_thresholds_agree,
        test_probability_metrics_ignore_threshold,
        test_class_metrics_use_threshold,
        test_finetuning_challenge_metrics,
        test_evaluator_validate_epoch_unpacking,
        test_evaluator_split_metrics_and_predictions_df,
        test_evaluator_evaluate_split_with_mock_model,
        test_evaluator_analyze_predictions_original_vs_youden,
        test_training_finalize_signature_static,
    ]

    print("Running Youden/metrics smoke tests...\n")
    for test_fn in tests:
        test_fn()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
