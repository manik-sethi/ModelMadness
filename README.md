# Comparing 10 different machine learning models to find the best one for breast cancer classification

## Logistic Regression

Logistic Regressino is a machine learning model that is good for categorizing numericaldata.

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
|-------------------|-----------|--------|----------|---------|
| AUC Score         |           |        |          | 0.998   |



Logistic Regression has a precision of 0.97, a recal of 0.98 and an f1 score of 0.98


Final Rankings by f1

Final Rankings by AUC
