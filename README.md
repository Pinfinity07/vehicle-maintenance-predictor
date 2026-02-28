# Vehicle Maintenance Predictor

A machine learning pipeline for predicting vehicle maintenance requirements using Decision Tree classification with GridSearchCV optimization.

## 🎯 Project Structure

```
vehicle-maintenance-predictor/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── main.py                            # Main pipeline orchestrator
├── data/
│   └── raw_dataset.csv               # Raw dataset
├── src/
│   ├── __init__.py
│   ├── pipeline/                      # ML pipeline modules
│   │   ├── __init__.py
│   │   ├── cleaning.py               # Data cleaning & preprocessing
│   │   ├── encoding.py               # Feature encoding & SMOTE
│   │   ├── training.py               # Model training with GridSearchCV
│   │   └── visualisation.py          # Data visualization
│   └── app/                           # Gradio web application
│       ├── __init__.py
│       └── app.py                    # Gradio UI
├── scripts/
│   ├── train_model.py                # Standalone training script
│   └── run_app.py                    # Standalone app launcher
├── models/                            # Trained model storage
└── docs/                              # Documentation
```

## 🚀 Quick Start

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline
```bash
python main.py
```

### Run Web App
```bash
python scripts/run_app.py
```

Or directly:
```bash
python -m src.app.app
```

### Train Model Only
```bash
python scripts/train_model.py
```

## 📊 Pipeline Stages

1. **Data Cleaning** (`src/pipeline/cleaning.py`)
   - Load raw dataset
   - Remove duplicates and null values
   - Date feature engineering
   - Outlier clipping using IQR method

2. **Feature Engineering** (`src/pipeline/encoding.py`)
   - Mutual Information-based feature selection
   - Ordinal encoding for categorical features
   - One-hot encoding for nominal features
   - Robust scaling for numerical features
   - SMOTE for class imbalance handling

3. **Model Training** (`src/pipeline/training.py`)
   - Stratified K-Fold Cross-Validation (5 splits)
   - GridSearchCV hyperparameter tuning
   - Decision Tree Classifier
   - F1-score optimization

## 🧠 Model Details

**Algorithm**: Decision Tree Classifier
**Hyperparameters Tuned**:
- max_depth: [3, 5, 7, 10, None]
- min_samples_leaf: [1, 5, 10, 20]
- criterion: [gini, entropy]

**Validation**: 5-fold Stratified Cross-Validation
**Optimization Metric**: F1-Score
**Imbalance Handling**: SMOTE

## 🎨 Web Application

Interactive Gradio interface for making predictions with:
- Vehicle profile inputs (model, fuel type, transmission, etc.)
- Component status inputs (tires, brakes, battery)
- Operational metrics (mileage, age, odometer, fuel efficiency)
- Real-time predictions with confidence scores

## 📈 Performance

Test Results:
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

## 🔄 Data Flow

```
raw_dataset.csv
    ↓
[Cleaning] → Handle nulls, remove duplicates, date engineering, outlier clipping
    ↓
[Feature Engineering] → MI selection, encoding (ordinal/one-hot), scaling, SMOTE
    ↓
[Train-Test Split] → 80/20 split with stratification
    ↓
[Model Training] → GridSearchCV with 5-fold CV
    ↓
[Prediction] → Real-time predictions via web app
```

## 📝 Input Features

- Vehicle_Model, Fuel_Type, Transmission_Type, Owner_Type
- Maintenance_History, Tire_Condition, Brake_Condition, Battery_Status
- Mileage, Vehicle_Age, Odometer_Reading, Fuel_Efficiency
- Reported_Issues, Service_History, Accident_History, Engine_Size

## 🛠️ Development

### Adding New Features
1. Update `src/pipeline/cleaning.py` for preprocessing
2. Update `src/pipeline/encoding.py` for feature engineering
3. Modify `src/app/app.py` UI if needed

### Running Tests
```bash
python -m pytest tests/
```

## 📦 Dependencies

- scikit-learn: ML algorithms
- pandas: Data manipulation
- numpy: Numerical computing
- imbalanced-learn: SMOTE implementation
- gradio: Web UI framework
- matplotlib: Visualization

## 📄 License

MIT License

## 👤 Author

Manan Kapoor

---

For detailed pipeline documentation, see [PIPELINE.md](docs/PIPELINE.md)
