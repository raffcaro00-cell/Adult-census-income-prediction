import matplotlib.pyplot as plt
import numpy as np


def apply_pca(
    X_train_std,
    X_test_std,
    y_train=None,
    variance_threshold=0.95,
    save_plots=True,
):
  """Applies manual PCA, transforms datasets, optionally generates plots, and

  saves transformed arrays.
  """
  # 1. Covariance matrix and Eigendecomposition
  cov_matrix = np.cov(X_train_std.T)
  eigen_vals, eigen_vecs = np.linalg.eigh(cov_matrix)

  # Sort eigenvalues and eigenvectors in descending order
  eigen_pairs = [
      (eigen_vals[i], eigen_vecs[:, i]) for i in range(len(eigen_vals))
  ]
  eigen_pairs.sort(key=lambda k: k[0], reverse=True)

  # 2. Variance explanation & threshold selection
  total_variance = sum([pair[0] for pair in eigen_pairs])
  var_exp = [(pair[0] / total_variance) for pair in eigen_pairs]
  cum_var_exp = np.cumsum(var_exp)

  n_components = np.argmax(cum_var_exp >= variance_threshold) + 1

  # 3. Projection matrix W & Dataset projection
  w_matrix = np.column_stack([eigen_pairs[i][1] for i in range(n_components)])
  X_train_pca = X_train_std.dot(w_matrix)
  X_test_pca = X_test_std.dot(w_matrix)

  # 4. Save transformed arrays
  np.save("X_train_pca.npy", X_train_pca)
  np.save("X_test_pca.npy", X_test_pca)

  # 5. Optional diagnostic plots
  if save_plots:
    # Explained Variance Plot
    plt.figure(figsize=(10, 5))
    plt.bar(
        range(1, len(var_exp) + 1),
        var_exp,
        alpha=0.5,
        align="center",
        label="Individual Explained Variance",
    )
    plt.step(
        range(1, len(cum_var_exp) + 1),
        cum_var_exp,
        where="mid",
        label="Cumulative Explained Variance",
        color="red",
    )
    plt.axhline(
        y=variance_threshold,
        color="black",
        linestyle="--",
        label=f"{int(variance_threshold*100)}% Threshold",
    )
    plt.ylabel("Explained Variance Ratio")
    plt.xlabel("Principal Component Index")
    plt.title("PCA Explained Variance")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("pca_explained_variance.png")
    plt.close()

    # 2D Projection Plot
    if y_train is not None:
      plt.figure(figsize=(10, 6))
      for label, color, marker in zip([0, 1], ["red", "blue"], ["s", "x"]):
        plt.scatter(
            X_train_pca[y_train == label, 0],
            X_train_pca[y_train == label, 1],
            c=color,
            marker=marker,
            alpha=0.6,
        )
      plt.xlabel("PC1")
      plt.ylabel("PC2")
      plt.title("PCA Projection - Training Set")
      plt.tight_layout()
      plt.savefig("pca_training_projection.png")
      plt.close()

  # Restituiamo 3 elementi per essere compatibili con la riga di main.py:
  # X_train_pca, X_test_pca, pca_model (in questo caso w_matrix)
  return X_train_pca, X_test_pca, w_matrix