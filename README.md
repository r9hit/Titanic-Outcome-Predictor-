# Titanic-Outcome-Predictor-


### 1. Data Cleaning
- Imputed missing values in `Age`, `Embarked`
- Dropped irrelevant columns (`Cabin`, `Ticket`)
- Converted categorical features to numeric

### 2. Logistic Regression
- ✅ **Manual Implementation** using NumPy
- ✅ **Scikit-learn Implementation** for comparison
- 📊 Evaluated using Accuracy, Precision, Recall, and F1 Score

### 3. Feature Engineering
- New Features:
  - `FamilySize = SibSp + Parch`
  - `IsAlone = 1 if FamilySize == 0 else 0`
  - Title extraction from Name
  - Interaction terms (e.g., `Sex_num * Pclass`)
- Resulted in **performance boost** in both versions


## 📈 Model Evaluation

### 🔧 Manual Implementation

#### Basic Logistic Regression
| Metric    | Score   |
|-----------|---------|
| Accuracy  | 0.8101  |
| Precision | 0.7857  |
| Recall    | 0.7432  |
| F1 Score  | 0.7639  |

#### After Feature Engineering
| Metric    | Score   |
|-----------|---------|
| Accuracy  | 0.8436  |
| Precision | 0.8026  |
| Recall    | 0.8243  |
| F1 Score  | 0.8133  |

---

###  Scikit-learn Implementation

#### Basic Logistic Regression
| Metric    | Score   |
|-----------|---------|
| Accuracy  | 0.8101  |
| Precision | 0.7857  |
| Recall    | 0.7432  |
| F1 Score  | 0.7639  |

#### With Interaction Feature (`Sex_num * Pclass`)
| Metric    | Score   |
|-----------|---------|
| Accuracy  | 0.8380  |
| Precision | 0.8000  |
| Recall    | 0.8108  |
| F1 Score  | 0.8054  |

---

## Upcoming Module

### 🎯 Fare Prediction
- Predicting `Fare` using:
  - Manual Linear Regression (Closed Form + Gradient Descent)
  - Scikit-learn Linear Regression
  - Evaluation using MAE, MSE, RMSE

---

## 🛠️ Tools & Libraries
- Python: NumPy, pandas, matplotlib, seaborn
- Scikit-learn
- Jupyter Notebooks
