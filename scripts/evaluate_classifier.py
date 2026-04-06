#!/usr/bin/env python3
"""
Extract real classifier metrics from trained model.
Evaluates on test/validation data and generates per-class accuracy.
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from src.classifier.dataset import load_difficulty_dataset


def evaluate_classifier_detailed():
    """Evaluate classifier and return per-class metrics."""
    
    model_path = "models/classifier/best_model.keras"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return None
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading test dataset...")
    _, val_ds, test_ds = load_difficulty_dataset(
        data_dir="data/difficulty_labels",
        image_size=(128, 128),
        batch_size=32,
    )
    
    # Evaluate on test set
    print("\nEvaluating on TEST set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    
    # Evaluate on validation set
    print("\nEvaluating on VALIDATION set...")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.4f} ({val_acc*100:.1f}%)")
    
    # Get per-class metrics
    print("\nComputing per-class accuracy...")
    
    class_names = ["easy", "medium", "hard"]
    per_class_acc = {}
    per_class_samples = {}
    
    # Evaluate on test set with predictions
    all_predictions = []
    all_true_labels = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        all_predictions.extend(pred_classes)
        all_true_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Per-class accuracy
    for class_idx, class_name in enumerate(class_names):
        mask = all_true_labels == class_idx
        if mask.sum() > 0:
            class_acc = (all_predictions[mask] == class_idx).mean()
            per_class_acc[class_name] = class_acc
            per_class_samples[class_name] = int(mask.sum())
    
    # Confusion matrix
    print("\nConfusion Matrix (Test Set):")
    print(f"{'True\\Pred':15s}", end="")
    for cn in class_names:
        print(f"{cn:12s}", end="")
    print()
    print("-" * 55)
    
    confusion = np.zeros((3, 3))
    for true_idx in range(3):
        print(f"{class_names[true_idx]:15s}", end="")
        mask = all_true_labels == true_idx
        if mask.sum() > 0:
            for pred_idx in range(3):
                count = (all_predictions[mask] == pred_idx).sum()
                confusion[true_idx, pred_idx] = count
                print(f"{count:12d}", end="")
        print()
    
    # Report
    print("\n" + "="*70)
    print("CLASSIFIER EVALUATION SUMMARY")
    print("="*70)
    print(f"\nOverall Test Accuracy: {test_acc*100:.1f}%")
    print(f"Overall Val Accuracy: {val_acc*100:.1f}%")
    print(f"\nPer-Class Accuracy (Test Set):")
    
    for class_name in class_names:
        if class_name in per_class_acc:
            acc = per_class_acc[class_name]
            samples = per_class_samples[class_name]
            print(f"  {class_name:8s}: {acc*100:5.1f}% ({samples:4d} samples)")
    
    # Confusion patterns
    print(f"\nKey Confusion Patterns:")
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j and confusion[i, j] > 0:
                total = confusion[i].sum()
                pct = (confusion[i, j] / total * 100) if total > 0 else 0
                print(f"  {true_class:8s} → {pred_class:8s}: {pct:5.1f}% ({int(confusion[i, j]):3d}/{int(total):3d})")
    
    return {
        "test_accuracy": test_acc,
        "val_accuracy": val_acc,
        "test_loss": test_loss,
        "val_loss": val_loss,
        "per_class_accuracy": per_class_acc,
        "per_class_samples": per_class_samples,
        "confusion_matrix": confusion,
    }


if __name__ == "__main__":
    metrics = evaluate_classifier_detailed()
    
    if metrics:
        print("\n" + "="*70)
        print("METRICS FOR TEX REPORT:")
        print("="*70)
        print(f"\nOverall Acc: {metrics['test_accuracy']*100:.1f}%")
        print(f"Easy Acc: {metrics['per_class_accuracy'].get('easy', 0)*100:.1f}%")
        print(f"Medium Acc: {metrics['per_class_accuracy'].get('medium', 0)*100:.1f}%")
        print(f"Hard Acc: {metrics['per_class_accuracy'].get('hard', 0)*100:.1f}%")
