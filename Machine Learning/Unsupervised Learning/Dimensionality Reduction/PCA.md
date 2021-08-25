# Principal Component Analysis (PCA)

Sources:

https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

Applications:
- Speed up ML algorithms
- Data visualization
- Noise filtering

Important:
- Affected by scale, so use StandardScaler()
- Fit only on training set (for ML projects)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(x)
pca = PCA(n_components=2) # or pca = PCA(.95) for 95% variance explained
X_pca = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

pca.components_ # Components
pca.explained_variance_ # Explained variance
np.cumsum(pca.explained_variance_ratio_) # Cumulative explained variance ratio

```

For large datasets, use:  `from sklearn.decomposition import RandomizedPCA`
