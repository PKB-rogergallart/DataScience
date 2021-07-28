# Imbalanced Data in classification problems

References:

https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18

## 0. Collect more data

## 1. Change the performance metric
Do not use Accuracy.
Better metrics are: Confusion Matrix, Precision, Recall, F1 Score.

## 2. Change the algorithm
DT/RF usually perform well on imbalanced data.

## 3. Resampling techniques
IMPORTANT: split train/test **BEFORE** oversampling

- Oversampling/Upsampling the minority class:
```python
from sklearn.utils import resample
minority_class_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_sate=RS)
upsampled = pd.concat([majority_class, minority_class_upsampled])
```

- Undersampling/Downsampling the majority class
```python
from sklearn.utils import resample
majority_class_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_sate=RS)
downsampled = pd.concat([minority_class, majority_class_downsampled])
```

- Generate Synthetic Samples

SMOTE = Synthetic Minority Oversampling Technique (uses Nearest Neighbours to generate new data for training)
```python
from imblearn.over_sampling import SMOTE
```

- Ensambling methods (Bagging) with resampling
```python
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#Create an object of the classifier.
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=0)

y_train = credit_df['Class']
X_train = credit_df.drop(['Class'], axis=1, inplace=False)

#Train the classifier.
bbc.fit(X_train, y_train)
preds = bbc.predict(X_test)
````
