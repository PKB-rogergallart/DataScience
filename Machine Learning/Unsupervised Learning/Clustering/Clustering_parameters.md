# Cluster parameters

## Choose number of clusters

Source: https://realpython.com/k-means-clustering-python/#choosing-the-appropriate-number-of-clusters

### Elbow method
- Run several k-means incrementing k in each iteration and record SSE for each
- Plot SEE vs number of clusters k.
- Find the point where the SEE curve starts to bend down (elbow point) either manually (visually) or programmatically.

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

## Evaluating clustering performance

### Adjusted Rand Index (ARI) using ground truth labels
- Values range between -1 and 1, being 0 == random assignments and 1 == perfectly labeled clusters.

```python
from sklearn.metrics import adjusted_rand_score
ari_score = adjusted_rand_score(true_labels, kmeans.labels_).round(2)
```
