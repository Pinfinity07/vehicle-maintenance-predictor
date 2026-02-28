"""Main entry point orchestrating the full pipeline"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.cleaning import load_and_preprocess_data
from pipeline.encoding import encode_and_split
from pipeline.training import train_model


def main():
    """Run the complete ML pipeline."""
    print("\n" + "=" * 60)
    print(" " * 10 + "VEHICLE MAINTENANCE PREDICTOR")
    print(" " * 15 + "ML PIPELINE")
    print("=" * 60 + "\n")

    # Stage 1: Data Cleaning
    print("[1/3] STAGE: Data Cleaning & Preprocessing")
    print("-" * 60)
    try:
        df = load_and_preprocess_data()
        print(f"✓ Data loaded successfully")
        print(f"  • Dataset shape: {df.shape}")
        print(f"  • Duplicates removed and NAs handled")
        print(f"  • Outliers clipped using IQR method\n")
    except Exception as e:
        print(f"✗ Error in data cleaning: {e}\n")
        sys.exit(1)

    # Stage 2: Encoding and Splitting
    print("[2/3] STAGE: Feature Encoding & Train-Test Split")
    print("-" * 60)
    try:
        X_train_sm, X_test_proc, y_train_sm, y_test, preprocessor = encode_and_split()
        print(f"✓ Encoding completed successfully")
        print(f"  • Feature selection via Mutual Information")
        print(f"  • Ordinal, Nominal, and Numerical features transformed")
        print(f"  • SMOTE applied to training data")
        print(f"  • Training set shape: {X_train_sm.shape}")
        print(f"  • Test set shape: {X_test_proc.shape}\n")
    except Exception as e:
        print(f"✗ Error in encoding: {e}\n")
        sys.exit(1)

    # Stage 3: Model Training
    print("[3/3] STAGE: Model Training & Evaluation")
    print("-" * 60)
    try:
        best_dt, train_acc, test_acc, preprocessor = train_model()
        print("\n✓ Model training completed successfully")
        print(f"  • Best Decision Tree model identified via GridSearchCV")
        print(f"  • 5-fold Stratified Cross-Validation used")
        print(f"  • Optimization metric: F1-Score\n")
    except Exception as e:
        print(f"✗ Error in training: {e}\n")
        sys.exit(1)

    # Final Summary
    print("=" * 60)
    print(" " * 20 + "PIPELINE COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

