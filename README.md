# Titanic Outcome & Fare Predictor

This project combines classification and regression techniques to predict survival outcomes and fare values for Titanic passengers using both **manual** and **scikit-learn** implementations.

---

##  Data Cleaning
- Imputed missing values in `Age` and `Embarked`
- Dropped irrelevant columns: `Cabin`, `Ticket`
- Converted categorical features (like `Sex`, `Embarked`) to numeric

---

## (Survival Prediction)

###  Manual Implementation
- Built logistic regression from scratch using NumPy
- Trained using gradient descent with sigmoid activation
- Evaluated using Accuracy, Precision, Recall, and F1 Score

### Scikit-learn Comparison
- Used `LogisticRegression` from `sklearn.linear_model`
- Same features and preprocessing pipeline used for fair benchmarking

---
Logistic Regression 
## Feature Engineering
- Created new features to improve model performance:
  - `FamilySize = SibSp + Parch`
  - `IsAlone = 1 if FamilySize == 0 else 0`
  - Extracted titles from names
  - Created interaction terms like `Sex_num * Pclass`
- Resulted in significant performance improvements in both manual and sklearn models

---

## 📈 Model Evaluation

### 🔧 Manual Logistic Regression

#### Basic Model
| Metric    | Score   |
|-----------|---------|
| Accuracy  | 0.8101  |
| Precision | 0.7857  |
| Recall    | 0.7432  |
| F1 Score  | 0.7639  |

#### With Feature Engineering
| Metric    | Score   |
|-----------|---------|
| Accuracy  | 0.8436  |
| Precision | 0.8026  |
| Recall    | 0.8243  |
| F1 Score  | 0.8133  |

---

### Scikit-learn Logistic Regression

#### Basic Model
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

## Fare Prediction ( Linear Regression) : 

### Manual Linear Regression (Gradient Descent)
Implemented linear regression from scratch to predict passenger fare.

#### Results (Without Log Target Transformation)
| Learning Rate | Final MSE | Final RMSE | R² Score |
|---------------|-----------|------------|-----------|
| 0.01          | 1890.94   | 43.48      | 0.3924    |
| 0.05          | 1891.25   | 43.49      | 0.3923    |
| 0.10          | 1891.25   | 43.49      | 0.3923    |
| 0.15          | 1891.25   | 43.49      | 0.3923    |

#### Results (With Log Target + Feature Engineering)
| Learning Rate | Final MSE | Final RMSE | R² Score |
|---------------|-----------|------------|-----------|
| 0.01          | 1862.44   | 43.16      | 0.4015    |
| 0.05          | 1832.77   | 42.81      | 0.4111    |
| 0.10          | 1830.56   | 42.79      | 0.4118    |
| 0.15          | 1830.45   | 42.78      | 0.4118    |

#### Residual Analysis
- Residuals show random scatter: indicates homoscedasticity
- Lack of tight clustering around zero suggests minor underfitting or non-linear patterns

#### Insight
- Log transformation and feature engineering improved performance
- Learning rate = 0.10 or 0.15 gave best speed-performance balance

---

### Scikit-learn Linear Regression

#### Comparison Table
| Aspect                     | Without Log / Feature Engg  |  With Log + Feature Engg |
|----------------------------|-----------------------------|-----------------------------|
| MSE (↓ better)             | 1891.25                     | **1826.99**                |
| RMSE (↓ better)            | 43.49                       | **42.74**                  |
| R² Score (↑ better)        | 0.3923                      | **0.4129**                 |

#### Interpretation: Why Log + Feature Engineering Wins
Log transformation and feature engineering helped the model generalize better:

**Log Transform Benefits:**
- `Fare` is highly right-skewed with outliers
- `log1p(Fare)` compresses large values, stabilizing variance
- Improves fit for linear models by normalizing the target distribution

**Feature Engineering Benefits:**
- `FamilySize = SibSp + Parch`: captures group travel behavior
- `Sex_Pclass = Sex * Pclass`: indirectly models socio-economic advantage
- These interaction features improve model expressiveness without introducing non-linearity

---


##  Tools & Libraries
- Python
  - NumPy
  - pandas
  - matplotlib
  - seaborn
- Scikit-learn
- Jupyter Notebook
