# 💧 Water Potability Prediction

A machine learning pipeline that predicts whether water is safe to drink using physicochemical features. Three classifiers are trained, compared, and explained with SHAP.

---

## 📁 Project Structure

```
├── P1.ipynb           # Main notebook (EDA → preprocessing → training → evaluation → SHAP)
├── water_potability.csv
├── rf_model.pkl       # Saved Random Forest model
└── scaler.pkl         # Saved RobustScaler
```

---

## 📊 Dataset

**File:** `water_potability.csv`  
**Rows:** 3,276 | **Features:** 9 physicochemical properties + 1 binary target

| Feature | Description |
|---|---|
| `ph` | pH of water (0–14) |
| `Hardness` | Capacity of water to precipitate soap (mg/L) |
| `Solids` | Total dissolved solids (ppm) |
| `Chloramines` | Chloramines concentration (ppm) |
| `Sulfate` | Sulfate concentration (mg/L) |
| `Conductivity` | Electrical conductivity (μS/cm) |
| `Organic_carbon` | Total organic carbon (ppm) |
| `Trihalomethanes` | Trihalomethanes concentration (μg/L) |
| `Turbidity` | Cloudiness measure (NTU) |
| `Potability` | **Target** — 1 = potable, 0 = not potable |

**Class distribution:** 1,998 not potable (61%) / 1,278 potable (39%) — imbalanced.

**Missing values:** `ph` (491), `Sulfate` (781), `Trihalomethanes` (162).

---

## 🔧 Pipeline

### 1. EDA
- Distribution histograms for all features
- Correlation heatmap
- Class imbalance check

### 2. Preprocessing
- **Imputation:** KNN Imputer (k=5) for missing values
- **Scaling:** RobustScaler (robust to outliers)
- **Split:** 80/20 train/test, stratified
- **SMOTE:** Applied to training set only to balance classes → 1,598 samples per class

### 3. Models & Hyperparameter Tuning (GridSearchCV, 5-fold, F1 score)

| Model | Hyperparameters Searched |
|---|---|
| Random Forest | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 10, 20} |
| SVM (RBF) | `C` ∈ {0.1, 1, 10}, `gamma` ∈ {scale, auto} |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` |

---

## 📈 Results

| Model | Accuracy | ROC-AUC | F1 (Potable) |
|---|---|---|---|
| **Random Forest** ✅ | **65%** | **0.6698** | **0.51** |
| XGBoost | 58% | 0.6526 | 0.54 |
| SVM | 59% | 0.6334 | 0.49 |

**Best model: Random Forest** (highest ROC-AUC).

---

## 🔍 Interpretability (SHAP)

SHAP `TreeExplainer` is used on the best Random Forest model to explain predictions:

- **Summary beeswarm plot** — feature impact per sample
- **Bar chart** — mean absolute SHAP values (global feature importance)

---

## 🚀 Usage

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
```

### Run the notebook

```bash
jupyter notebook water_quality.ipynb
```

### Load the saved model

```python
import pickle

model  = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Predict on new data (already imputed)
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
```

---

## ⚠️ Notes

- SMOTE is applied **only on the training set** to prevent data leakage.
- Model performance is moderate (~0.67 AUC); water potability is inherently noisy with these features alone.
- For production use, consider adding domain-specific features or ensemble stacking.