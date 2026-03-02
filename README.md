# Vehicle Maintenance Predictor

A machine learning pipeline for predicting vehicle maintenance requirements using Decision Tree classification with GridSearchCV optimization.

🔗 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/kaori02/vehicle-maintenance-predictor)
📄 **Project Report:** [Google Drive](https://drive.google.com/file/d/16fxy-LBw1LDIxirKwgz1W-qLFJUW7h1c/view?usp=sharing)

## 🎯 Project Structure

```
vehicle-maintenance-predictor/
├── README.md                          # Project documentation
├── data/
│   └── raw_dataset.csv               # Raw dataset
├── scripts/
│   └── train.py                      # Train model & save artifacts
├── app/
│   ├── app.py                        # Gradio web application
│   └── requirements.txt              # Python dependencies
├── pipeline/                          # ML pipeline modules
│   ├── __init__.py
│   ├── cleaning.py                   # Data cleaning & preprocessing
│   ├── encoding.py                   # Feature encoding & SMOTE
│   └── training.py                   # Model training with GridSearchCV
└── models/                            # Saved model artifacts
    ├── model.joblib                   # Trained DecisionTreeClassifier
    ├── preprocessor.joblib            # Fitted ColumnTransformer
    └── feature_order.pkl              # Feature column order
```

## 🚀 Quick Start

### Setup
```bash
# Install dependencies
pip install -r app/requirements.txt
```

### Train the Model
```bash
python scripts/train.py
```
This runs the full pipeline and saves artifacts to `models/`.

### Run Web App
```bash
python app/app.py
```
Loads the pre-trained model and launches a Gradio interface.

## 📊 Pipeline Stages

1. **Data Cleaning** (`pipeline/cleaning.py`)
   - Load raw dataset
   - Remove duplicates and null values
   - Date feature engineering
   - Outlier clipping using IQR method

2. **Feature Engineering** (`pipeline/encoding.py`)
   - Mutual Information-based feature selection
   - Ordinal encoding for categorical features
   - One-hot encoding for nominal features
   - Robust scaling for numerical features
   - SMOTE for class imbalance handling

3. **Model Training** (`pipeline/training.py`)
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
1. Update `pipeline/cleaning.py` for preprocessing
2. Update `pipeline/encoding.py` for feature engineering
3. Modify `app.py` UI if needed

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
