# Cluster parameters

## Choose number of clusters

Source: https://realpython.com/k-means-clustering-python/#choosing-the-appropriate-number-of-clusters

### Elbow method
1. Run several k-means incrementing k in each iteration and record SSE for each
2. Plot SEE vs number of clusters k.
3. Find the point where the SEE curve starts to bend down (elbow point) either manually (visually) or programmatically.

```python
!pip install kneed

from kneed import KneeLocator

kl = KneeLocator(range(1, kmax), sse, curve="convex", direction="decreasing")
k1.elbow
```

### Silhouette coefficient
- The silhouette coefficient is a measure of cluster cohesion and separation that depends on two factors: how close a data point is to other points in the cluster and how far it is from points in other clusters.
- The range is between -1 and 1 (the larger the better)
- A score is obtain by averagin the silhouette coefficient of all the points.

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(scaled_features, kmeans.labels_).round(2)
```

### Dendrograms (only in Hierarchical Clustering)
Source: https://towardsdatascience.com/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019

1. Determine the largest vertical distance on the dendrogram that does not intersect any of the other clusters.
2. Draw a horizontal line at both extremities.
3. The optimal number of clusters is equal to the number of vertical lines going through the horizontal line

```python
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
# Visually we determine the optimal number of clusters = 5

model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
model.fit(X)
labels = model.labels_
```

## Evaluating clustering performance
Source: https://realpython.com/k-means-clustering-python/#evaluating-clustering-performance-using-advanced-techniques

### Adjusted Rand Index (ARI) using ground truth labels

- Values range between -1 and 1, being 0 == random assignments and 1 == perfectly labeled clusters.

```python
from sklearn.metrics import adjusted_rand_score
ari_score = adjusted_rand_score(true_labels, kmeans.labels_).round(2)
```
