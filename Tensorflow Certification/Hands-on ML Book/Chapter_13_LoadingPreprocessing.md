# Chapter 13 - Loading and Preprocessing Data with Tensorflow

# The Data API
- The dataset methods do not modify datasets, they create new ones.
- `tf.data.Dataset` methods: 
  - `from_tensor_slices()`, `batch()`  
  - `map()`, `apply()`, `filter()`, `take()`
  - `shuffle(buffer_size=5, seed=42)` where budder_size needs to be big enough but < RAM (or dataset size)
  - 
