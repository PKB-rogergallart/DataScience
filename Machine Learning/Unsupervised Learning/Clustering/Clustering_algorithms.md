# Clustering algorithms

## K-Means clustering

- We initialize randomly the center points for each group. Then we calculate the distance between each sample and each group centerand we classify the point to be in the group whose center is closest. We recompute the group centers and iterate until the centers don't change much or for a certain fixed number of iterations.

Pros | Cons
-----|-----
Fast, *O(n)* | You need to select how </br>many groups there are
  _ | Results may depend on random initialization </br> (lack of consistency and repeatability)

Alternative:  K-Medians which is less sensitive to outliers but is much slower for larger datasets (because it required sorting in each iteration)

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

## Agglomerative Hierarchical Clustering



Sources: 

https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68
