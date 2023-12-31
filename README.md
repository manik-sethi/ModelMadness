# Comparing 10 different machine learning models to find the best one for breast cancer classification

Here are the results from the notebook! Feel free to run in your own.

## Logistic Regression

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
| 1    | Logistic Regression     | 0.98     |
| 2    | Neural Network          | 0.98     |
| 3    | AdaBoost                | 0.98     |
| 4    | XGBoost                 | 0.97     |
| 5    | Gradient Boosting       | 0.97     |
| 6    | Random Forest           | 0.97     |
| 7    | Support Vector Machine  | 0.96     |
| 8    | Naive Bayes             | 0.94     |
| 9    | K-Nearest Neighbors     | 0.94     |
| 10   | Decision Tree           | 0.92     |


## Final Rankings by AUC
| Rank | Model                   | AUC Score |
|------|-------------------------|-----------|
| 1    | Logistic Regression     | 0.9981    |
| 2    | Neural Network          | 0.9969    |
| 3    | AdaBoost                | 0.9962    |
| 4    | Random Forest           | 0.9967    |
| 5    | Gradient Boosting       | 0.9957    |
| 6    | XGBoost                 | 0.9944    |
| 7    | Support Vector Machine  | 0.9965    |
| 8    | Naive Bayes             | 0.9927    |
| 9    | Decision Tree           | N/A       |
| 10   | K-Nearest Neighbors     | N/A       |


# Results

In our analysis of models used for detecting breast cancer, we observed varying performance across different evaluation metrics. **Logistic Regression** demonstrated the highest AUC score, indicating excellent overall model performance. **Neural Network** and **AdaBoost** closely followed with competitive AUC scores. Regarding F1 scores, **Logistic Regression**, **Neural Network**, and **AdaBoost** again showcased top-tier performance. These findings suggest that these models, particularly **Logistic Regression**, **Neural Network**, and **AdaBoost**, are promising candidates for breast cancer detection. However, further considerations, such as interpretability and computational complexity, should inform the final choice of the model for practical implementation in a clinical setting.


