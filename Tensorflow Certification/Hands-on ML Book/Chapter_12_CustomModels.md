# Chapter 12 - Custom Models and Training with TF

## Tensorflow as Numpy

- @ = Matrix multiplication (Python >3.5)
- No Automatic type conversion. It raises exception so that you can fix it manually
Use `tf.cast()` to convert types.
- `tf.constant()` values are inmutable. `tf.Variable()` are mutable

## Custom Loss Function
- For better performance use vectorized implementation and whenever possible only TensorFlow operations.
- 2 ways to do it:
  - custom loss function with appropriate input/output
  - subclassing keras.losses.Loss

```python

#Example of custom loss function WITHOUT parameteres

def huber_fn(y_true, y_pred):
  error = y_true - y_pred
  is_small_error = tf.abs(error) < 1
  squared_loss = tf.square(error) / 2
  linear_loss = tf.abs(error) - 0.5
  return tf.where(is_small_error, squared_loss, linear_loss)

model.compile(loss=huber_fn, optimizer="nadam")
model.fit(X_train, y_train, [...])

# Save and load model
model.save("my_model_with_a_custom_loss.h5")
model = keras.models.load_model("my_model_with_a_custom_loss.h5", custom_objects={"huber_fn": huber_fn})
```

```python

#Example of custom loss function WITH hyperparameters

def create_huber(threshold=1.0):
  def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < threshold
    squared_loss = tf.square(error) / 2
    linear_loss = threshold * tf.abs(error) - threshold**2 / 2
    return tf.where(is_small_error, squared_loss, linear_loss)
   return huber_fn

model.compile(loss=create_huber(2.0), optimizer="nadam")
model.fit(X_train, y_train, [...])

# Save and load model (you need to specify the hyperparameteres again)
model.save("my_model_with_a_custom_loss.h5")
model = keras.models.load_model("my_model_with_a_custom_loss.h5", custom_objects={"huber_fn": create_huber(2.0)})
```

```python

#Example of custom loss function WITH hyperparameters USING subclassing

class HuberLoss(keras.losses.Loss):
  def __init__(self, threshold=1.0, **kwargs):
    self.threshold = threshold
    super().__init__(**kwargs)

  def call(self, y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < self.threshold
    squared_loss = tf.square(error) / 2
    linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
    return tf.where(is_small_error, squared_loss, linear_loss)

  def get_config(self):
    base_config = super().get_config()
    return {**base_config, "threshold": self.threshold}

model.compile(loss=HuberLoss(2.), optimizer="nadam")
model.fit(X_train, y_train, [...])

# Save and load model (no need to specify the hyperparameters again, it is saved with the model)
model.save("my_model_with_a_custom_loss.h5")
model = keras.models.load_model("my_model_with_a_custom_loss.h5", custom_objects={"HuberLoss": HuberLoss})
```

## Custom Activation Function, Initializer, Regularizer and Contraints

```python
def my_softplus(z): # return value is just tf.nn.softplus(z)
  return tf.math.log(tf.exp(z) + 1.0)

def my_glorot_initializer(shape, dtype=tf.float32):
  stddev = tf.sqrt(2. / (shape[0] + shape[1]))
  return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
  return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
  return tf.where(weights < 0., tf.zeros_like(weights), weights)

layer = keras.layers.Dense(30, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights)

```

If these custom functions have hyperparameters and you want to save them with the model, use subclassing:
- layers (including activation functions): *keras.layers.Layer* implementing call()
- initializers: *keras.initializers.Initializer* implementing _ _ call() _ _ 
- regularizers: *keras.regularizers.Regularizer* implementing _ _ call() _ _
- constraints: *keras.constraints.Constraint* implementing _ _ call() _ _

## Custom Metrics

Loss | Metric
----- | -----
Used by Gradient Descent to train the model | Used to evaluate the model
Must be differentiable | Can be non-differentiable
Gradients can not be 0 everywhere | Can have 0 gradients everywhere
No need to be easily interpretable by humans | Must be easily interpretable

- We typically use *streaming metrics* (or *stateful metric*) that is metrics that are gradually updated batch after bach. To do so, you need to subclass `keras.metrics.Metric`
- If we define a metric using a simple function, Keras will call it for each batch and keep track of the mean during each epoch which is NOT convenient for some metrics e.g. Precision.

 
```python

#Example of Custom Streaming Metric

class HuberMetric(keras.metrics.Metric):

  def __init__(self, threshold=1.0, **kwargs):
    super().__init__(**kwargs) # handles base args (e.g., dtype)
    self.threshold = threshold
    self.huber_fn = create_huber(threshold)
    self.total = self.add_weight("total", initializer="zeros")
    self.count = self.add_weight("count", initializer="zeros")

  def update_state(self, y_true, y_pred, sample_weight=None):
    metric = self.huber_fn(y_true, y_pred)
    self.total.assign_add(tf.reduce_sum(metric))
    self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

  def result(self):
    return self.total / self.count

  def get_config(self):
    base_config = super().get_config()
    return {**base_config, "threshold": self.threshold}

```

## Custom Layers

### Custom Layers without weights (e.g. `keras.layers.Flatten`, `keras.layers.ReLU`)

Write a function and wrap it in a `keras.layers.Lambda`layer.

```python

exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))

```

### Custom layers with weights (stateful layer)

Create a subclass of the `keras.layers.Layer` class.

```python

class MyDense(keras.layers.Layer):
  def __init__(self, units, activation=None, **kwargs):
    super().__init__(**kwargs)
    self.units = units
    self.activation = keras.activations.get(activation)

  def build(self, batch_input_shape):
    self.kernel = self.add_weight(
          name="kernel", shape=[batch_input_shape[-1], self.units],
          initializer="glorot_normal")
    self.bias = self.add_weight(
          name="bias", shape=[self.units], initializer="zeros")
    super().build(batch_input_shape) # must be at the end

  def call(self, X):
    return self.activation(X @ self.kernel + self.bias)

  def compute_output_shape(self, batch_input_shape):
    return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

def get_config(self):
    base_config = super().get_config()
    return {**base_config, "units": self.units,
          "activation": keras.activations.serialize(self.activation)}

```

- To create a layer with multiple inputs, `call()` should receive as argument a tuple containing all inputs and return a list of outputs. And `compute_output_shape()` should receive as argument a tuple containing each input's batch shape, and return a list of batch output shapes.

```python
class MyMultiLayer(keras.layers.Layer):
  def call(self, X):
     X1, X2 = X
     return [X1 + X2, X1 * X2, X1 / X2]

  def compute_output_shape(self, batch_input_shape):
     b1, b2 = batch_input_shape
     return [b1, b1, b1] # should probably handle broadcasting rules
```

- If the layer needs to have different behavior during training and during testing, add a `training = None` argument to the `call()` method and use this argument to decide what to do (`if training: [...]`)

### Custom models
- Create a subclass of `keras.Model`.
- Create the layers in the constructor `__init__()`.
- Use the layers in the `call()` method.
- To be able to save/load the model, implement the `get_config()` method in the custom model (and in the custom layers if any). Alternatively you can save and load the weights manually using `save_weights()` and `load_weights()`.

## Losses and Metrics based on Model internals
TBD

## Computing gradients using Autodiff
TBD

## Custom training loops
TBD

## TensorFlow Functions and Graphs 
TBD
