# Classification Models

## Decision Trees

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
