# Chapter 11 - Training Deep Neural Networks

## 1. Vanishing/Exploding Gradient
- DNN suffer from unstable gradients, different layers may learn at widely different speeds. Eg: Vanishing gradients problem in lower layers, Exploding gradients problem in RNN.
- Causes: activation function (e.g. sigmoid) and initialization scheme

### 1.1 Glorot (Xavier), Lecun and He Initializations
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
  - LeakyReLU_α(z) = max(αz, z) with α = 0.01(typ.) to 0.2
  - Randomized LeakyReLU = RReLU_α(z) where α is randomly picked in a given range at training, and fixed to an average value during testing.
  - Parametric Randomized Leaky ReLU = PReLU_α(z) where α is learned during training
  - Exponential Linear Unit = ELU
