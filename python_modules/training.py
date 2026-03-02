import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from .encoding import encode_and_split


def train_model():
    """Train a Decision Tree model using GridSearchCV."""
    
    # Load and process data
    X_train_sm, X_test_proc, y_train_sm, y_test, preprocessor = encode_and_split()
    
    # Stratified K-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Decision Tree parameter grid
    dt_param_grid = {
        "max_depth":        [3, 5, 7, 10, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "criterion":        ["gini", "entropy"],
    }

    # GridSearchCV for hyperparameter tuning
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=dt_param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    dt_grid.fit(X_train_sm, y_train_sm)

    # Get the best model
    best_dt = dt_grid.best_estimator_

    # Make predictions
    y_pred_dt_train = best_dt.predict(X_train_sm)
    y_pred_dt       = best_dt.predict(X_test_proc)

    # Calculate accuracy
    train_acc = accuracy_score(y_train_sm, y_pred_dt_train)
    test_acc  = accuracy_score(y_test,     y_pred_dt)

    # Print results
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test  Accuracy : {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_dt, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_dt)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Maint.", "Maint."])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Decision Tree", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return best_dt, train_acc, test_acc, preprocessor


if __name__ == "__main__":
    print("=" * 50)
    print("Training Decision Tree Model")
    print("=" * 50)
    best_dt, train_acc, test_acc, preprocessor = train_model()
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
