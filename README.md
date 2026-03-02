# Vehicle Maintenance Predictor

An ML-powered web app that predicts whether a vehicle needs maintenance based on its profile, condition, and usage history.

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/kaori02/vehicle-maintenance-predictor)

## Project Structure

```
vehicle-maintenance-predictor/
├── train.py              # Train model and save artifacts
├── app.py                # Gradio web app (loads pre-trained model)
├── requirements.txt      # Python dependencies
│
├── pipeline/             # ML pipeline modules
│   ├── cleaning.py       # Data cleaning & preprocessing
│   ├── encoding.py       # Feature encoding & SMOTE
│   └── training.py       # Model training & evaluation
│
├── data/
│   └── raw_dataset.csv   # Raw dataset
│
└── models/               # Saved model artifacts (gitignored)
    ├── model.joblib       # Trained DecisionTreeClassifier
    ├── preprocessor.joblib# Fitted ColumnTransformer
    └── feature_order.pkl  # Feature column order
```

## Quick Start

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train the Model
```bash
python train.py
```
This runs the full pipeline (cleaning → encoding → training) and saves artifacts to `models/`.

### Run the Web App
```bash
python app.py
```
Loads the pre-trained model and launches a Gradio interface at `http://localhost:7860`.

## ML Pipeline

### 1. Data Cleaning (`pipeline/cleaning.py`)
- Remove duplicates and null values
- Date feature engineering (days since service/warranty)
- Outlier clipping using IQR method

### 2. Feature Engineering (`pipeline/encoding.py`)
- Mutual Information-based feature selection
- Ordinal encoding (maintenance history, tire/brake/battery condition)
- One-hot encoding (vehicle model, fuel type, transmission, owner)
- Robust scaling for numerical features
- SMOTE for class imbalance handling

### 3. Model Training (`pipeline/training.py`)
- Decision Tree Classifier
- 5-fold Stratified Cross-Validation
- GridSearchCV hyperparameter tuning (max_depth, min_samples_leaf, criterion)
- Optimized on F1-Score

## Deployment

The app is deployed on Hugging Face Spaces. To update:

```bash
cd ../hf-space-vehicle
# Copy updated model artifacts
cp ../vehicle-maintenance-predictor/models/*.joblib models/
cp ../vehicle-maintenance-predictor/models/*.pkl models/
# Copy updated app if changed
cp ../vehicle-maintenance-predictor/app.py .
git add -A && git commit -m "Update" && git push
```

## Author

Manan Kapoor
