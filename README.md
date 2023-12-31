# Comparing 10 different machine learning models to find the best one for breast cancer classification

## Logistic Regression

Logistic Regression is a machine learning model that is good for categorizing numerical data.

Results from the notebook:

Model: Logistic Regression

**Confusion Matrix:**
|                 | Predicted Class 0 | Predicted Class 1 |
|-----------------|-------------------|-------------------|
| Actual Class 0  | 62                | 1                 |
| Actual Class 1  | 2                 | 106               |

**Classification Report:**
|                   | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Class 0           | 0.97      | 0.98   | 0.98     | 63      |
| Class 1           | 0.99      | 0.98   | 0.99     | 108     |
| Accuracy          |           |        | 0.98     | 171     |
| Macro Avg         | 0.98      | 0.98   | 0.98     | 171     |
| Weighted Avg      | 0.98      | 0.98   | 0.98     |         |

**AUC Score:**
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

## Model: Support Vector Machine

**Confusion Matrix:**
|                 | Predicted Class 0 | Predicted Class 1 |
|-----------------|-------------------|-------------------|
| Actual Class 0  | 61                | 2                 |
| Actual Class 1  | 3                 | 105               |

**Classification Report:**
|                   | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Class 0           | 0.95      | 0.97   | 0.96     | 63      |
| Class 1           | 0.98      | 0.97   | 0.98     | 108     |
| Accuracy          |           |        | 0.97     | 171     |
| Macro Avg         | 0.97      | 0.97   | 0.97     | 171     |
| Weighted Avg      | 0.97      | 0.97   | 0.97     |         |

**AUC Score:**
| AUC Score         | 0.9964726631393297 |
|-------------------|---------------------|

## Model: Decision Tree

**Confusion Matrix:**
|                 | Predicted Class 0 | Predicted Class 1 |
|-----------------|-------------------|-------------------|
| Actual Class 0  | 59                | 4                 |
| Actual Class 1  | 9                 | 99                |

**Classification Report:**
|                   | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Class 0           | 0.87      | 0.94   | 0.90     | 63      |
| Class 1           | 0.96      | 0.92   | 0.94     | 108     |
| Accuracy          |           |        | 0.92     | 171     |
| Macro Avg         | 0.91      | 0.93   | 0.92     | 171     |
| Weighted Avg      | 0.93      | 0.92   | 0.92     |         |

**AUC Score:**
| AUC Score         | 0.9265873015873015 |
|-------------------|---------------------|



Final Rankings by f1

Final Rankings by AUC
