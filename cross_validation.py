"""
Adult (Census Income) Dataset - Model Cross-Validation & Evaluation
Module: Stratified K-Fold Cross-Validation and Hold-Out Evaluation
"""

import time
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


def perform_cross_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    estimator=None,
    n_splits: int = 5,
    random_state: int = 0,
) -> tuple:
  """Executes Stratified K-Fold Cross-Validation on the training dataset.

  Parameters:
  -----------
  X_train : np.ndarray
      Feature matrix for training.
  y_train : np.ndarray
      Target array for training.
  estimator : object, optional
      Scikit-learn estimator instance. Defaults to SVC(kernel='rbf', C=1.0).
  n_splits : int
      Number of cross-validation folds.
  random_state : int
      Random seed for reproducibility.

  Returns:
  --------
  tuple: (cv_scores, elapsed_time)
  """
  if estimator is None:
    estimator = SVC(kernel="rbf", C=1.0, random_state=random_state)

  print(
      f"STEP: Executing Stratified {n_splits}-Fold Cross-Validation ({estimator.__class__.__name__})"
  )
  print("-" * 70)

  kfold = StratifiedKFold(
      n_splits=n_splits, shuffle=True, random_state=random_state
  )
  cv_scores = []

  start_time = time.time()

  for fold, (train_idx, val_idx) in enumerate(
      kfold.split(X_train, y_train), start=1
  ):
    # Fit estimator on the current fold
    estimator.fit(X_train[train_idx], y_train[train_idx])

    # Compute accuracy on validation fold
    score = estimator.score(X_train[val_idx], y_train[val_idx])
    cv_scores.append(score)

    class_dist = np.bincount(y_train[train_idx])
    print(
        f"Fold {fold:02d} | Train Class Dist: {class_dist} | Val Acc:"
        f" {score:.4f}"
    )

  elapsed_time = time.time() - start_time

  print("-" * 70)
  print(
      "Cross-Validation Mean Accuracy:"
      f" {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}"
  )
  print(
      f"Min Accuracy: {np.min(cv_scores):.4f} | Max Accuracy:"
      f" {np.max(cv_scores):.4f}"
  )
  print(f"Total Computation Time: {elapsed_time:.2f} seconds\n")

  return cv_scores, elapsed_time


def evaluate_test_performance(
    estimator, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> dict:
  """Fits the model on full training data and evaluates on hold-out test set."""
  print("STEP: Evaluating Model on Hold-Out Test Set")
  print("-" * 70)

  estimator.fit(X_train, y_train)
  y_pred = estimator.predict(X_test)
  test_accuracy = accuracy_score(y_test, y_pred)

  print(f"Hold-out Test Accuracy: {test_accuracy:.4f}\n")
  print("Classification Report:")
  report = classification_report(
      y_test, y_pred, target_names=["<=50K", ">50K"]
  )
  print(report)

  return {"accuracy": test_accuracy, "y_pred": y_pred, "report": report}