# 💧 Water Potability Prediction

A machine learning pipeline that predicts whether water is safe to drink using physicochemical features. Four classifiers are trained, compared, and explained with SHAP. Deployed as an interactive web app via Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://waterqualityclassification-project1.streamlit.app/)

---

## 📁 Project Structure

```
|-- water_quality.ipynb      # Main notebook (EDA → preprocessing → training → evaluation → SHAP)
|-- water_potability.csv     # Dataset (Kaggle — Water Potability, 3,276 samples)
|-- app.py                   # Streamlit deployment app (AquaSafe)
|-- presentation.pdf         # Presentation pdf file
|-- requirements.txt         # Python dependencies
|-- rf_model.pkl             # Saved Random Forest model
|-- scaler.pkl               # Saved RobustScaler
```

---

## 📊 Dataset

**Source:** [Kaggle — Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)  
**File:** `water_potability.csv`  
**Rows:** 3,276 &nbsp;|&nbsp; **Features:** 9 physicochemical properties + 1 binary target

| Feature | Description | Unit |
|---|---|---|
| `ph` | Acidity/alkalinity of water | 0 – 14 |
| `Hardness` | Capacity to precipitate soap | mg/L |
| `Solids` | Total dissolved solids | ppm |
| `Chloramines` | Chloramines concentration | ppm |
| `Sulfate` | Sulfate concentration | mg/L |
| `Conductivity` | Electrical conductivity | μS/cm |
| `Organic_carbon` | Total organic carbon | ppm |
| `Trihalomethanes` | Trihalomethanes concentration | μg/L |
| `Turbidity` | Cloudiness of water | NTU |
| `Potability` | **Target** — 1 = potable, 0 = not potable | — |

**Class distribution:** 1,998 not potable (61%) / 1,278 potable (39%) — imbalanced.  
**Missing values:** `ph` (491), `Sulfate` (781), `Trihalomethanes` (162).

---

## 🔧 Pipeline

### 1. EDA
- Distribution histograms for all 9 features
- Correlation heatmap
- Class imbalance check

### 2. Preprocessing
- **Imputation:** KNN Imputer (k=5) for missing values
- **Scaling:** RobustScaler (robust to outliers)
- **Split:** 80/20 train/test, stratified by target
- **SMOTE:** Applied to training set only → balances to 1,598 samples per class

### 3. Models & Hyperparameter Tuning
GridSearchCV with 5-fold cross-validation, optimising F1 score.

| Model | Hyperparameters Searched |
|---|---|
| Logistic Regression | `solver=liblinear` |
| Random Forest | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 10, 20} |
| Gradient Boosting | Default + random state |
| SVM (RBF) | `C` ∈ {0.1, 1, 10}, `gamma` ∈ {scale, auto} |

---

## 📈 Results

| Model | Accuracy | ROC-AUC | F1 (Potable) |
|---|---|---|---|
| **Random Forest** ✅ | **65%** | **0.6698** | **0.51** |
| Gradient Boosting | 58% | 0.6526 | 0.54 |
| SVM | 59% | 0.6334 | 0.49 |
| Logistic Regression | ~57% | ~0.61 | ~0.47 |

**Best model: Random Forest** (highest ROC-AUC — selected for deployment and SHAP analysis).

---

## 🔍 SHAP Interpretability

SHAP `TreeExplainer` applied to the best Random Forest model reveals which features drive predictions most.

**Key findings from the beeswarm plot:**
- **pH** is the most influential feature — high pH pushes toward potable, low pH toward not potable
- **Sulfate** is the second most important — high values have inconsistent effects
- **Hardness & Chloramines** show moderate influence
- **Turbidity, Trihalomethanes, Conductivity** contribute minimally to predictions

**Plots generated:**
- Beeswarm plot — per-sample feature impact
- Bar chart — mean absolute SHAP values (global importance ranking)
- Waterfall plot — single-sample prediction explanation

---

## 🚀 Usage

### Run the notebook
```bash
jupyter notebook water_quality.ipynb
```

### Run the Streamlit app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Live demo
👉 [waterqualityclassification-project1.streamlit.app](https://waterqualityclassification-project1.streamlit.app/)

---

## ⚠️ Notes & Limitations

- SMOTE is applied **only on the training set** to prevent data leakage into evaluation.
- Model performance is moderate (~0.67 AUC); water potability is inherently noisy with physicochemical features alone.
- The app uses WHO guideline thresholds for contextual warnings — these are advisory, not hard classification rules.
- **Not intended for production use** — always verify with certified laboratory testing before making public health decisions.
