# Comparing 10 different machine learning models to find the best one for breast cancer classification

## Logistic Regression

Logistic Regression is a machine learning model that is good for categorizing numerical data.

Results from the notebook:

Model: Logistic Regression

Confusion Matrix:
|                 | Predicted Class 0 | Predicted Class 1 |
|-----------------|-------------------|-------------------|
| Actual Class 0  | 62                | 1                 |
| Actual Class 1  | 2                 | 106               |

Classification Report:
|                   | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Class 0           | 0.97      | 0.98   | 0.98     | 63      |
| Class 1           | 0.99      | 0.98   | 0.99     | 108     |
| Accuracy          |           |        | 0.98     | 171     |
| Macro Avg         | 0.98      | 0.98   | 0.98     | 171     |
| Weighted Avg      | 0.98      | 0.98   | 0.98     |         |

AUC Score:
|AUC Score| 0.9980893592004703 | 
|-------------------|-----------|


## K-Nearest Neighbors

**Confusion Matrix:**
|                 | Predicted Class 0 | Predicted Class 1 |
|-----------------|-------------------|-------------------|
| Actual Class 0  | 59                | 4                 |
| Actual Class 1  | 3                 | 105               |

**Classification Report:**
|                   | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Class 0           | 0.95      | 0.94   | 0.94     | 63      |
| Class 1           | 0.96      | 0.97   | 0.97     | 108     |
| Accuracy          |           |        | 0.96     | 171     |
| Macro Avg         | 0.96      | 0.95   | 0.96     | 171     |
| Weighted Avg      | 0.96      | 0.96   | 0.96     |         |

**AUC Score:**
| AUC Score         | 0.9776601998824221 |
|-------------------|---------------------|



Final Rankings by f1

Final Rankings by AUC
