import os
from sklearn.metrics import accuracy_score, classification_report

# Direct imports from local modules (flat directory structure)
from Preprocessing import load_and_preprocess_data
from PCA_feature_extraction import apply_pca
from Learning import (
    train_svm_baseline,
    optimize_svm_hyperparameters,
    plot_validation_curve,
)
from cross_validation import perform_cross_validation, evaluate_test_performance
from learning_curves import plot_learning_curve


def run_pipeline():
    # ==========================================
    # STEP 1: DATA PREPROCESSING
    # ==========================================
    print("=== STEP 1: DATA PREPROCESSING ===")

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        file_path='data.pkl',
        target_column='income',
        test_size=0.2,
        random_state=42
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}\n")

    # ==========================================
    # STEP 2: DIMENSIONALITY REDUCTION (PCA)
    # ==========================================
    print("=" * 60)
    print("=== STEP 2: DIMENSIONALITY REDUCTION (PCA) ===")
    print("=" * 60)

    X_train_pca, X_test_pca, pca_model = apply_pca(
        X_train, X_test, y_train=y_train, variance_threshold=0.95
    )
    print("PCA transformation completed successfully.\n")

    # ==========================================
    # STEP 3: MODEL TRAINING AND OPTIMIZATION
    # ==========================================
    print("=" * 60)
    print("=== STEP 3: MODEL TRAINING AND OPTIMIZATION ===")
    print("=" * 60)

    # optimize_svm_hyperparameters returns the best fitted SVC estimator
    best_model = optimize_svm_hyperparameters(X_train_pca, y_train, cv_splits=5)

    # Optional: Generate validation curve for diagnostics
    # plot_validation_curve(X_train_pca, y_train, param_name="gamma")

    # ==========================================
    # STEP 4: MODEL EVALUATION
    # ==========================================
    print("\n" + "=" * 60)
    print("=== STEP 4: MODEL EVALUATION ===")
    print("=" * 60)

    # 1. Stratified Cross-Validation on the best model
    cv_scores, cv_time = perform_cross_validation(
        X_train=X_train_pca, y_train=y_train, estimator=best_model, n_splits=5
    )

    # 2. Final evaluation on the Hold-Out Test Set
    test_results = evaluate_test_performance(
        estimator=best_model,
        X_train=X_train_pca,
        y_train=y_train,
        X_test=X_test_pca,
        y_test=y_test,
    )

    # ==========================================
    # STEP 5: DIAGNOSTIC LEARNING CURVE PLOTTING
    # ==========================================
    print("=" * 60)
    print("=== STEP 5: DIAGNOSTIC LEARNING CURVE PLOTTING ===")
    print("=" * 60)

    # Retrieve parameters directly from the SVC estimator object
    best_c = best_model.C
    best_kernel = best_model.kernel

    plot_learning_curve(
        estimator=best_model,
        X=X_train_pca,
        y=y_train,
        title=f"SVM Learning Curve (Best Params: C={best_c}, Kernel={best_kernel})",
        save_path="svm_learning_curve.png",
    )

    print("\nPipeline execution finished successfully!")


if __name__ == '__main__':
    run_pipeline()