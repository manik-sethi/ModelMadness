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

## Model: Random Forest

**Confusion Matrix:**
|                 | Predicted Class 0 | Predicted Class 1 |
|-----------------|-------------------|-------------------|
| Actual Class 0  | 59                | 4                 |
| Actual Class 1  | 1                 | 107               |

**Classification Report:**
|                   | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Class 0           | 0.98      | 0.94   | 0.96     | 63      |
| Class 1           | 0.96      | 0.99   | 0.98     | 108     |
| Accuracy          |           |        | 0.97     | 171     |
| Macro Avg         | 0.97      | 0.96   | 0.97     | 171     |
| Weighted Avg      | 0.97      | 0.97   | 0.97     |         |

**AUC Score:**
| AUC Score         | 0.9966931216931216 |
|-------------------|---------------------|


## Model: Gradient Boosting

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
| AUC Score         | 0.9957378012933568 |
|-------------------|---------------------|

## Model: Naive Bayes

**Confusion Matrix:**
|                 | Predicted Class 0 | Predicted Class 1 |
|-----------------|-------------------|-------------------|
| Actual Class 0  | 57                | 6                 |
| Actual Class 1  | 5                 | 103               |

**Classification Report:**
|                   | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Class 0           | 0.92      | 0.90   | 0.91     | 63      |
| Class 1           | 0.94      | 0.95   | 0.95     | 108     |
| Accuracy          |           |        | 0.94     | 171     |
| Macro Avg         | 0.93      | 0.93   | 0.93     | 171     |
| Weighted Avg      | 0.94      | 0.94   | 0.94     |         |

**AUC Score:**
| AUC Score         | 0.9926513815402704 |
|-------------------|---------------------|

## Model: Neural Network (MLP Classifier)

**Confusion Matrix:**
|                 | Predicted Class 0 | Predicted Class 1 |
|-----------------|-------------------|-------------------|
| Actual Class 0  | 61                | 2                 |
| Actual Class 1  | 2                 | 106               |

**Classification Report:**
|                   | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Class 0           | 0.97      | 0.97   | 0.97     | 63      |
| Class 1           | 0.98      | 0.98   | 0.98     | 108     |
| Accuracy          |           |        | 0.98     | 171     |
| Macro Avg         | 0.97      | 0.97   | 0.97     | 171     |
| Weighted Avg      | 0.98      | 0.98   | 0.98     |         |

**AUC Score:**
| AUC Score         | 0.9969135802469136 |
|-------------------|---------------------|

## Model: AdaBoost

**Confusion Matrix:**
|                 | Predicted Class 0 | Predicted Class 1 |
|-----------------|-------------------|-------------------|
| Actual Class 0  | 61                | 2                 |
| Actual Class 1  | 2                 | 106               |

**Classification Report:**
|                   | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Class 0           | 0.97      | 0.97   | 0.97     | 63      |
| Class 1           | 0.98      | 0.98   | 0.98     | 108     |
| Accuracy          |           |        | 0.98     | 171     |
| Macro Avg         | 0.97      | 0.97   | 0.97     | 171     |
| Weighted Avg      | 0.98      | 0.98   | 0.98     |         |

**AUC Score:**
| AUC Score         | 0.9961787184009406 |
|-------------------|---------------------|

## Model: XGBoost

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
| AUC Score         | 0.9944150499706055 |
|-------------------|---------------------|

## Final Rankings by f1
| Rank | Model                   | F1-Score |
|------|-------------------------|----------|
| 1    | Neural Network          | 0.98     |
| 2    | Gradient Boosting       | 0.97     |
| 3    | Random Forest           | 0.97     |
| 4    | XGBoost                 | 0.97     |
| 5    | AdaBoost                | 0.98     |
| 6    | Logistic Regression     | 0.98     |
| 7    | Decision Tree           | 0.92     |
| 8    | Support Vector Machine  | 0.96     |
| 9    | K-Nearest Neighbors     | 0.94     |
| 10   | Naive Bayes             | 0.94     |


## Final Rankings by AUC
| Rank | Model                   | AUC Score |
|------|-------------------------|-----------|
| 1    | Neural Network          | 0.9969    |
| 2    | Gradient Boosting       | 0.9957    |
| 3    | Random Forest           | 0.9967    |
| 4    | XGBoost                 | 0.9944    |
| 5    | AdaBoost                | 0.9962    |
| 6    | Logistic Regression     | 0.9981    |
| 7    | Decision Tree           | N/A       |
| 8    | Support Vector Machine  | 0.9965    |
| 9    | K-Nearest Neighbors     | N/A       |
| 10   | Naive Bayes             | 0.9927    |

