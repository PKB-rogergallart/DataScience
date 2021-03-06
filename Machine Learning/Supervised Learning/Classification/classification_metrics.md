# Classification metrics

## Binary classification
References:

https://towardsdatascience.com/how-to-best-evaluate-a-classification-model-2edb12bcc587
https://www.ritchieng.com/machine-learning-evaluate-classification-model/#
https://towardsdatascience.com/hackcvilleds-4636c6c1ba53

### Confusion Matrix
![Confusion Matrix](/Assets/confusion_matrix_metrics.png)

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

classification_report(y_test, y_pred) #Precision, Recall, F1 score

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.title('Confusion Matrix', size = 15);
```
### Accuracy
Accuracy = {# correct predictions} / {# all predictions}

Need to be compared with the **Null Accuracy** which is the accuracy achieved by always predicting the most frequent class.

```python
# Trick for binary classification (y_test = 0 or 1)
null_accuracy = max(y_test.mean(), 1-y_test.mean())
```

Problem: not useful if imbalanced dataset.

### Errors type I and II
Type I error = FP

Type II error = FN

Classification Error = Misclassification Rate = (FP + FN) / (TP + TN + FP + FN)

### Precision and Recall

- Precision = TP / (TP + FP)
  How good our model is when the prediction is positive.
  Focus on **positive predictions** --> How many positive predictions are correct (true).

- Recall = TP / (TP + FN)
  How good our model is at predicting positive classes.
  Focus on **actual positive classes** --> How many of the positive classes the model can predict correctly.
  
There is a trade-off between Precision and Recall: increasing precision decreases recall and viceversa.

### F1 Score
F1 Score = 2 * Precision * Recall / (Precision + Recall)

It's the harmonic mean of Precision and Recall.

More useful for imbalanced datasets because it takes into account both FP and FN.

### Sensitivity and Specificity

- Sensitivity = True Positive Rate (TPR) = Recall = Proportion of positive class correctly predicted as positive
- Specificity = 1 - FPR = Proportion of negative class correctly predicted as negative

There is a trade-off between Sensitivity and Specificity. In binary logistic regression, threshold can be adjusted to increase sensitivity or specificity.

### ROC and AUC

- ROC = Receiving Operating Characteristics. Plot of FPR vs TPR for different threshold values (in logistic regression)
- AUC = Area under the curve. Allows to compare different models (the higher AUC the better)

x-axis = TPR = Sensitivity = TP / (TP + FN)

y-axis = FPR = 1 - Specificity = FP / (TN + FP)

AUC is useful even when there is high class imbalance.

```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

from sklearn.metrics import auc
auc = roc_auc_score(y_test,logis_pred_prob[:,1])
```

### PRC

- PRC = Precision-Recall Curve

PRCs are better suited for models trained on highly imbalanced datasets

```python
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob[:,1])

from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
f1 = f1_score(y_test, y_pred_prob)
ap = average_precision_score(y_test, y_pred_prob)
```

### Kappa (or Cohen???s kappa)
- Kappa = Classification accuracy normalized by the imbalance of the classes in the data.
https://en.wikipedia.org/wiki/Cohen%27s_kappa
