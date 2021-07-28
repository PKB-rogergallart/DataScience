# Classification metrics

## Binary classification
References:
https://towardsdatascience.com/how-to-best-evaluate-a-classification-model-2edb12bcc587

### Confusion Matrix
![Confusion Matrix](/Assets/confusion_matrix_metrics.png)

### Accuracy
Accuracy = {# correct predictions} / {# all predictions}

Problem: not useful if imbalanced dataset.

### Errors type I and II
Type I error = FP

Type II error = FN

### Precision and Recall

- Precision = TP / (TP + FP)
  How good our model is when the prediction is positive.
  Focus on **positive predictions** --> How many positive predictions are correct (true).

- Recall = TP / (TP + FN)
  How good our model is at predicting positive classes.
  Focus on **actual positive classes** --> How many of the positive classes the model can predict correctly.
  
There is a trade-off between them: increasing precision decreases recall and viceversa.

### F1 Score
F1 Score = 2 * Precision * Recall / (Precision + Recall)

It's the harmonic mean of Precision and Recall.

More useful for imbalanced datasets because it takes into account both FP and FN.

### Sensitivity and Specificity

- Sensitivity = True Positive Rate (TPR) = Recall = Proportion os positive class correctly predicted as positive
- Specificity = 1 - FPR = Proportion of negative class correctly predicted as negative
