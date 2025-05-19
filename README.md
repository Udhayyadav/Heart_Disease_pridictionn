# 🫀 Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project aims to **predict the presence or absence of heart disease** using various patient health indicators. Using machine learning classification algorithms on the **UCI Heart Disease dataset**, we build, evaluate, and compare different models to determine which performs best at identifying heart disease risk.

---

## 📂 Dataset

- **Source**: UCI Machine Learning Repository  
- **Attributes**: 14 features (like age, sex, chest pain type, cholesterol, etc.)  
- **Target**: Binary classification – `0` (No heart disease), `1` (Presence of heart disease)

---

## 🔧 Project Structure & Steps

1. **Load and Explore Data**
   - Display dataset info, descriptive statistics, null values, and class distribution.
   - Visual EDA using plots for features like age, sex, chest pain type, etc.
   - Correlation heatmap to understand inter-feature relationships.

2. **Data Preprocessing**
   - One-hot encoding for multi-class categorical variables.
   - StandardScaler for numerical features.
   - Train-test split with stratification.
   - ColumnTransformer to apply preprocessing pipelines.

3. **Model Training & Evaluation**
   - Models used:
     - Logistic Regression
     - Random Forest Classifier
     - Support Vector Machine (SVM)
   - Evaluation metrics:
     - Accuracy, Confusion Matrix, Classification Report, ROC AUC
     - Visualized ROC Curves and Confusion Matrices

4. **Ensemble Model**
   - Soft Voting Classifier using Logistic Regression, Random Forest, and SVM
   - Improved predictive performance through model aggregation

5. **Model Comparison**
   - ROC Curve plots for visual comparison
   - Tabular accuracy and AUC scores for each model

---

## 📊 Results Summary

- All models performed reasonably well, indicating strong predictive signals in the dataset.
- **Logistic Regression** performed well in terms of accuracy and interpretability.
- **Ensemble Voting Classifier** offered improved overall robustness.
- Visuals like confusion matrices and ROC curves were used to interpret performance.

---

## 🧠 Future Improvements

- 🔍 **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV`  
- ✨ **Feature Engineering**: Introduce interaction or domain-driven features  
- 🧪 **Cross-Validation**: Apply k-fold CV for robust performance estimates  
- 📈 **Advanced Models**: Try `XGBoost`, `LightGBM`, or Neural Networks  
- 🧐 **Model Explainability**: Use SHAP or LIME to explain predictions

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### Installation
```bash
pip install -r requirements.txt
