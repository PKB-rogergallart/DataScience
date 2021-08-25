# Clustering algorithms

## Overview
- Goal of clustering: identify **Meaningfulness** (expand domain knowledge) and **Usefulness** (serve as intermediate step in data pipeline).

- Types of clustering algorithms:
  - Partitional clustering: 
    - Divides all data objects into non-overlapping groups.
    - Examples: k-means, k-medoids

    Pros | Cons
    -----|-----
    Good for sperical shaped clusters | Not well suited for clusters with complex shapes and different sizes
    Scalable w/r to complexity | Non-deterministic (all?)
    _ | You need to select how </br>many groups there are
    _ | Break down when clusters of different densities


  - Hierarchical clustering:
    - Produces a tree-based hierarchy (dendrogram)
    - Two approaches: agglomerative clustering (bottom-up) or divisive clustering (top-down)
    - Examples:

    Pros | Cons
    -----|-----
    Deterministic | High algorithm complexity
    Can reveal relationships | Sensitive to noise and outliers
    _ | You need to select how </br>many groups there are

  - Density-based clustering:
    - Clusters are assigned where there are high densities of data points separated by low-density regions.
    - Does not require the user to determine the number of clusters. 
    - Distance-based parameter acts as tunable threshold
    - Examples: DBSCAN, OPTICS.

    Pros | Cons
    -----|-----
    Does not require user to specify k | Not well suited for high dimensional spaces
    Work with non-sperical shapes | Trouble identifying clusters of varying densities
    Resistant to outliers | You need to select how </br>many groups there are
    
## K-Means clustering
- Uses Expectation-Maximization algorithm (EM).
- We initialize randomly the center points for each group. Then we calculate the distance between each sample and each group centerand we classify the point to be in the group whose center is closest. We recompute the group centers and iterate until the centers don't change much or for a certain fixed number of iterations.
- Quality of the cluster assignments can be determined using Sum of squared error (SSE).
- As it is non-deterministic, usually we run it several times using different initializations, and choose the solution with lowest SSE.

Pros | Cons
-----|-----
Fast, *O(n)* | You need to select how </br>many groups there are
  _ | Results may depend on random initialization </br> (lack of consistency and repeatability)

Alternative:  K-Medians which is less sensitive to outliers but is much slower for larger datasets (because it required sorting in each iteration)

```python
from sklearn.cluster import KMeans

kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)

kmeans.fit(scaled_features)

kmeans.inertia_   # Lowest SSE value
kmeans.cluster_centers_  # Final locations of the centroid
kmeans.n_iter_  # The number of iterations required to converge
kmeans.labels_ # Cluster assignments (numpy array)   
```
## Mean-Shift clustering

- Sliding-window-based centroid-based density-based algorithm.
- Assumes that clusters are convex shaped.
- Parameters: windows size (radius r), number of windows
- We randomly initialize many windows (centers). For each window, we calculate the density of points inside the window and their mean. If centering the window in the mean increases the density, we shift the window and iterate. The windows will tend to go to higher density regions.  When multiple windows overlap, the window with higher density is preserved. We iterate until the density inside the window does no longer increase and until all points lie within some window.    

Pros | Cons
-----|-----
Automatically discovers </br>the number of groups | Window size (radius) </br>selection is not trivial


## Density-Based Spation Clustering of Applications with Noise (DBSCAN)

- Views clusters as areas of high density separated by areas of low density.
- Parameters: distance epsilon &#949; (-> neighborhood), minPoints (-> noise)
- Starting with a random point we extract its neighborhood (epsilon). If not enough points (minPoints), we label it as "noise" and "visited". If enough points, we label it and all its neighbours as belonging to current cluster and "visited". We repeat the same procedure for all the points added to the cluster until all the points in the cluster are labeled and visited. Once done, we repeat the same procedure with a new unvisited point to create a new cluster.

Pros | Cons
-----|-----
Automatically discovers </br>the number of groups | Window size (radius) </br>selection is not trivial
Indentifies outliers as noises | Bad performance when clusters </br>have very different densities
Any size and cluster shape | Bad performance in </br>very high-dimensional data


## Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
- GMM assume the data points are Gaussian distributed (which is less restrictive assumption than saying they are circular lie K-means). Instead of only using mean, now we have mean and standard deviation (for each direction).
- Clusters can have any king of elliptical shape (vs K-means which assume circular shapes)
- EM is the optimization algorithm used to find the mean and SD of the Gaussian for each cluster.
- Procedure very similar to K-means. We select the number of clusters and randomly initialize Gaussian parameters for each. Given the Gaussian distributions, we compute the probability of each point to belong to each cluster. We calculate a weighted sum of the data point positions using the probabilities as weights, and use it to compute the new parameters. We repeat iteratively until convergence.

Pros | Cons
-----|-----
More flexible than K-means </br> in terms of cluster covariance | You need to select how </br>many groups there are
Allows mixed membership | _

## Hierarchical Agglomerative Clustering (HAC)
- Bottom-up algorithm: treats each point as single cluster and successively merges pairs of clusters.
- Parameter: linkage criteria and distance metric (e.g. average linkage = avg distance between points in first cluster and points in second cluster), number of clusters
- We start considering each data point as a cluster. On each iteration we combine the two clusters with the smallest distance metric. We repeat until we reach the top of the tree (we only have one cluster). We can stop building the tree once we have the desired number of clusters.

Linkage criteria (== distance between two clusters):
- Single linkage: the shortest distance between two points in each cluster.
- Complete linkage: the longest distance between two points in each cluster.
- Average linkage: the average distance between each point in one cluster to every point in another cluster.
- Ward linkage: the sum of squared differences within all clusters.

Distance metric:
- Euclidean distance
- Manhattan distance

Pros | Cons
-----|-----
No need to specify number of clusters in advance | Higher complexity *O(n^3)* although </br>some implementations *O(n^2)*
Not sensitive to the choice of distance metric | _
Good when underlying data has hierarchical structure | _

References:
https://towardsdatascience.com/machine-learning-algorithms-part-12-hierarchical-agglomerative-clustering-example-in-python-1e18e0075019
https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/


Sources: 

https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68
