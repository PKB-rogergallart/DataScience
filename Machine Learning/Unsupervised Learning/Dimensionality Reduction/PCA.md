# Principal Component Analysis (PCA)

Sources:

https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60


Applications:
- Speed up ML algorithms
- Data visualization

Important:
- Affected by scale, so use StandardScaler()
- Fit only on training set (for ML projects)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2) # or pca = PCA(.95) for 95% variance explained
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

```
