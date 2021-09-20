# Chapter 11 - Training Deep Neural Networks

## 1. Vanishing/Exploding Gradient
- DNN suffer from unstable gradients, different layers may learn at widely different speeds. Eg: Vanishing gradients problem in lower layers, Exploding gradients problem in RNN.
- Causes: activation function (e.g. sigmoid) and initialization scheme

### Glorot (Xavier), Lecun and He Initializations
- Terminology: fan_in = number of inputs, fan_out = number of outputs (= number of neurons in MLP)
- We need variance to the ouputs be equal to variance of the inputs, and we need the gradients to have equal variance before and after flowing through a layer in the reverse direction.
- Xavier initialization (or Glorot initialization):
- Lecun initialization:
- He initialization:

```python
# Keras Default = Glorot + uniform distribution

# He initialization:
  keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")
  keras.layers.Dense(10, activation="relu", kernel_initializer="he_uniform")

# He + uniform distribution but based on fan_in:
  he_avg_init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg', distribution='uniform')
  keras.layers.Dense(10, activation="sigmoid", kernel_initializer=he_avg_init)
  ```
  
### Non-saturating activation functions
- LeakyReLU_α(z) = max(αz, z) with α = 0.01(typ.) - 0.2 - 0.3
- Randomized LeakyReLU = RReLU_α(z) where α is randomly picked in a given range at training, and fixed to an average value during testing.
- Parametric Randomized Leaky ReLU = PReLU_α(z) where α is learned during training
- Exponential Linear Unit = ELU: faster convergence but slow to compute, also during testing.
  ![ELU](/Assets/ELU_activation_function.jpg)
- Scaled ELU = SELU: under some conditions, ensures self-normalization of the network.

- Which one to use: in general SELU > ELU > LeakyReLU and variants > tanh > logistic

```python
# LeakyReLU:
model = keras.models.Sequential([
[...]
keras.layers.Dense(10, kernel_initializer="he_normal"),
keras.layers.LeakyReLU(alpha=0.2),
[...]
])

# PReLU:
model = keras.models.Sequential([
[...]
keras.layers.Dense(10, kernel_initializer="he_normal"),
keras.layers.PReLU(),
[...]
])

# SELU:
layer = keras.layers.Dense(10, activation="selu", kernel_initializer="lecun_normal")
```

### Batch Normalization (BN)
- Consists in adding just before or after the activation function for each hidden layer an operation to zero-center and normalize each input, then scales (&#947;) and shifts (&#946;) the results using 2 new parameter vectors per layer (that will be trained too).
- During training:
  - &#947; (output scale vector) and &#946; (output offset vector) are learn using regular backpropagation
  - the batch mean and SD of the current minibatch is calculated and used for training, and the final input mean vector and input SD vector is estimated using exponential moving average
- During testing:
  - we use the estimated mean and SD
- Hyperparameters: momentum (typ. close to 1 e.g. 0.99, 0.999), axis
Pros | Cons
------------ | -------------
Prevents vanishing/exploding gradients | Adds complexity
Faster convergence | Slower predictions (solution: fuse with previous layer)
Reduces the need of other regularization techniques (e.g. droput) | _

```python

# Batch Normalization AFTER activation functions
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.BatchNormalization(),
keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
keras.layers.BatchNormalization(),
keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
keras.layers.BatchNormalization(),
keras.layers.Dense(10, activation="softmax")
])

# Batch Normalization BEFORE activation functions
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.BatchNormalization(),
keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
keras.layers.BatchNormalization(),
keras.layers.Activation("elu"),
keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
keras.layers.BatchNormalization(),
keras.layers.Activation("elu"),
keras.layers.Dense(10, activation="softmax")
])

# Note: mean and SD are considered by Kears as "non-trainable" parameters of the model because they are not trained using backpropagation.

````

- ToRead: Fixup, a possible alternative to BN: https://arxiv.org/abs/1901.09321

### Gradient Clipping:
- Mostly used in RNN (where BN is tricky to use)

```python

optimizer = keras.optimizers.SGD(clipvalue=1.0) #Clips gradient components between -1 and +1
optimizer = keras.optimizers.SGD(clipnorm=1.0) #Clips gradient components using L2 norm, preserving orientation

model.compile(loss="mse", optimizer=optimizer)

```

