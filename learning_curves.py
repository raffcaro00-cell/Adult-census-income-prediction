"""
Adult (Census Income) Dataset - Model Diagnostics via Learning Curves
Module: Bias-Variance Trade-off Analysis
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def plot_learning_curve(
    estimator,
    X,
    y,
    title: str = "SVM Learning Curve",
    save_path: str = "svm_learning_curve.png",
    cv: int = 5,
    n_jobs: int = -1,
):
  """Computes and plots the learning curve for a given model to analyze bias-variance trade-offs.

  Parameters:
  -----------
  estimator : object
      Trained or instantiated scikit-learn model/pipeline.
  X : array-like
      Feature dataset (e.g., training features).
  y : array-like
      Target labels.
  title : str
      Title for the resulting plot.
  save_path : str
      File path to save the generated figure.
  cv : int
      Number of cross-validation folds.
  n_jobs : int
      Number of jobs to run in parallel.
  """
  sns.set_style("whitegrid")
  print("STEP: Computing Learning Curves across Training Sizes")
  print("-" * 70)

  start_time = time.time()

  train_sizes, train_scores, test_scores = learning_curve(
      estimator=estimator,
      X=X,
      y=y,
      train_sizes=np.linspace(0.1, 1.0, 10),
      cv=cv,
      n_jobs=n_jobs,
      scoring="accuracy",
  )

  elapsed_time = time.time() - start_time
  print(
      f"Learning curve computation completed in {elapsed_time:.2f} seconds.\n"
  )

  # Compute mean and standard deviation across CV folds
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)
  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)

  # Plotting Setup
  plt.figure(figsize=(10, 6))

  # Plot training accuracy
  plt.plot(
      train_sizes,
      train_mean,
      color="blue",
      marker="o",
      markersize=5,
      label="Training Accuracy",
  )
  plt.fill_between(
      train_sizes,
      train_mean + train_std,
      train_mean - train_std,
      alpha=0.15,
      color="blue",
  )

  # Plot validation accuracy
  plt.plot(
      train_sizes,
      test_mean,
      color="green",
      linestyle="--",
      marker="s",
      markersize=5,
      label="Validation Accuracy",
  )
  plt.fill_between(
      train_sizes,
      test_mean + test_std,
      test_mean - test_std,
      alpha=0.15,
      color="green",
  )

  # Formatting and labels
  plt.title(title, fontsize=12)
  plt.xlabel("Number of Training Samples", fontsize=11)
  plt.ylabel("Classification Accuracy", fontsize=11)
  plt.grid(True, linestyle="--", alpha=0.5)
  plt.legend(loc="lower right", fontsize=10)

  # Set dynamic bounds for y-axis
  min_y = max(0.0, np.min(test_mean) - 0.05)
  plt.ylim([min_y, 1.0])

  plt.tight_layout()
  plt.savefig(save_path, dpi=300)
  plt.show()

  print(
      f"Learning curve plot successfully displayed and saved as '{save_path}'."
  )