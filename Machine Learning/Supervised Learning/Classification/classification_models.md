# Classification models

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

## K-Nearest Neighbours (KNN)
- Non-parametric learning algorithm
- Laxy learning algorithm (no specialised training phase)

Pros | Cons
-----|-----
Easy to implement | Does not work well with high dimensional data
Lazy learning (no training phase) | High prediction cost for large datasets
New data can be added seamlessly | Does not work well with categorical data
Distribution agnostic | -
Only 2 hyperparameters<br>(K and distance) | -

## Logistic Regression

- Binary classification. Predicts probability of occurrence using a logit (sigmoid) function
- MLE = Maximum Likelihood Estimation
- Types:
  - Binary Logistic Regression
  - Multinomial Logistic Regression
  - Ordinal Logistic Regression

Pros | Cons
-----|-----
Easy to implement | Not able to handle large number of categorical features
No scaling required | Can't solve non-linear problems (except if non-linear transforms used)
Provides probability score | Not perform well if features not correlated to target and very similar/corerlated to each other.

## Support Vector Machines (SVM)

- The core idea of SVM is to find a maximum marginal hyperplane(MMH) that best divides the dataset into classes.
- Kernel trick when not linearly separable.
- Hyperparameters:
  - Kernel: Linear, Polynomial, Radial Basis Function (RBF)
  - Regularization
  - Gamma (how far points consider in calculating separation line)

Pros | Cons
-----|-----
Good accuracy | Not suitable for large datasets (high training time)
Faster prediction than Naive Bayes | Slower training than NB
Low memory | Works poorly with overlapping classes
Works well with clear spearation and high dimensional space | Sensitive to the type of kernel used
