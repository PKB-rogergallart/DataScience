# Classification

## Models
### Decision Trees

- Terminology: Root Node, Decision Node, Leaf Node, purity/impurity, pruning

- Attribute Selection Mesaures (ASM) == Splitting criterion :
  - Information Gain (Entropy)
  - Gain Ratio
  - Gini Index

- Main parameters: criterion (== ASM), max_depth (== pre-pruning)

Pros | Cons
-----|-----
Easy to interpret | Sensitive to noiy data <br>(can overfit noisy data)
Little preprocessing needed <br>(no normalization, etc) | Small variation can result in different DT
Can be used for feature engineering <br>(variable selection, predicting missing values) | Biased for imbalanced datasets
Distribution agnostic | -

- Imports:
```python
from sklearn.tree import DecisionTreeClassifier
```

- Visualisation of DTs: 
```python
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())
```

## Classification metrics

References:
https://towardsdatascience.com/how-to-best-evaluate-a-classification-model-2edb12bcc587

### Accuracy
Accuracy = {# correct predictions} / {# all predictions}
Problem: not useful if umbalanced dataset

### Confusion Matrix
![Confusion Matrix](/Assets/confusion_matrix.png)

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
