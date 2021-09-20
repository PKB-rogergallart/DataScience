# Chapter 11 - Training Deep Neural Networks

Techniques to speed up training and reach a better solution:

## 1. Techniques to avoid Vanishing/Exploding Gradient
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

## 2. Reusing Pretrained Layers

### Transfer Learning
- Transfer Learning works best with deep CNN and worst with small dense networks.
- The more similar the tasks are, the more layers you have to reuse (starting from the lower layers).
- Add new layers on top of the reused one. Freeze the reused layers and train the model to see how it performs.
- Try unfreezing reused layers progressively and see how performance improves. Reduce the learning rate to avoid damaging thereused weights. The more data you have the more layers you can unfreeze.
- IMPORTANT: you must always compile your model after you freeze/unfreeze layers

```python
# EXAMPLE:

# clone structure of model A to avoid overwriting and copy weights
model_A = keras.models.load_model("my_model_A.h5")
model_A_clone = keras.models.clone_model(model_A) 
model_A_clone.set_weights(model_A.get_weights()) # copy weights

# Take all layers from A_clone except the output one and add new output layer
model_B_on_A = keras.models.Sequential(model_A_clone.layers[:-1]) 
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

# Freeze all reused layers and compile the model
for layer in model_B_on_A.layers[:-1]:
  layer.trainable = False
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Train the model (output layer) for a few epochs
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4, validation_data=(X_valid_B, y_valid_B))

# Unfreeze reused layers and compile with a lower learning rate
for layer in model_B_on_A.layers[:-1]:
  layer.trainable = True
optimizer = keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-2
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Keep training
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, validation_data=(X_valid_B, y_valid_B))
```


### Unsupervised Pretraining


### Pretraining on an Auxiliary Task
If you do not have much labeled training data, one last option is to train a first neural network on an auxiliary task for which you can easily obtain or generate labeled training data, then reuse the lower layers of that network for your actual task. The
first neural network’s lower layers will learn feature detectors that will likely be reusable by the second neural network.


## 3. Fast Optimizers (alternatives to regular Gradient Descent)

### Regular Gradient Descent:

  ![Gradient_Descent](https://github.com/PKB-rogergallart/DataScience/blob/main/Assets/Gradient_Descent_algorithm.jpg)

### Momentum Optimization

  ![Momentum_Algorithm](https://github.com/PKB-rogergallart/DataScience/blob/main/Assets/Momentum_algorithm.jpg)
- The gradient is used for acceleration, not for speed.
- the hyperparameter &#946; called *momentum* is used as some sort of friction (0=high friction, 1=no friction). Typical value is 0.9 

```python
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
```

### Nesterov Accelerated Gradient (NAG)

  ![Nesterov_Accelerated_Gradient](https://github.com/PKB-rogergallart/DataScience/blob/main/Assets/NAG_algorithm.jpg)

- It measures the gradient of the cost function not at the local position &theta; but slightly ahead in the direction of the momentum

```python
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
```

### AdaGrad (DO NOT USE)

  ![AdaGrad_Algorithm](https://github.com/PKB-rogergallart/DataScience/blob/main/Assets/AdaGrad_algorithm.jpg)
  
  - It corrects its direction earlier to point a bit more toward the global optimium by scaling down the gradient vector along the steepest dimensions. It decays the learning rate faster for steep dimensions than for dimensions with gentler slopes **(adaptive learning rate)**.
  - Works well for simple quadratic problems but often stops too early when trianing neural networks. **It should not be used to train deep neural networks.**

### RMSProp

  ![RMSProp](https://github.com/PKB-rogergallart/DataScience/blob/main/Assets/RMSProp_algorithm.jpg)
  
- Fixes the problems of AdaGrad by accumulating only the gradients from the most recent iterations using an exponential decay.
- **Adaptive learning rate**
- The hyperparameter &beta; (rho, in Keras) is typically set to 0.9.

```python
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
```

### Adam and Nadam Optimization

  ![Adam_algorithm](https://github.com/PKB-rogergallart/DataScience/blob/main/Assets/Adam_algorithm.jpg)
  
- Adam (Adaptive Moment Estimation) combines the ideas of momentum optimization and RMSProp. It keeps tract of an exponentially decaying average of past gradients (like momentum optimization) and also of an exponentially decaying average of past squared gradients (like RMSProp).
- **Adaptive learning rate**

```python
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999) #Typical values

# Epsilon value = keras.backend.epsilon() = 1E-7. 
# To change it use keras.backend.set_epsilon()
```

- Variants:
  - Adamax: uses L_&infinity; norm ('max' norm) in step 2
  - Nadam: Adam + Nesterov trick


### Learning Rate Scheduling
This technique consists in modifying the initial learning rate during training using different strategies, typically starting with a large value and decreasing it later.

- Power scheduling

  ```python
  optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
  # Keras assumes that c=1
  # decay is the inverse of s (number of steps)
  ```
- Exponential scheduling

  ```python
  # function that takes current epoch and return the learning rate
  def exponential_decay_fn(epoch):
    return 0.01 * 0.1**(epoch / 20)

  # Aternative if you don't want to hardcode lr0 and s:
  def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
      return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn
  exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

  # Create a LearningRateScheduler callback and pass it to the fit() method:
  lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
  history = model.fit(X_train_scaled, y_train, [...], callbacks=[lr_scheduler])
  ```
  
- Piecewise constant scheduling

  ```python
  def piecewise_constant_fn(epoch):
    if epoch < 5:
      return 0.01
    elif epoch < 15:
      return 0.005
    else:
      return 0.001

   # Same reasoning a previous strategy but using this schedule function
   ```
 
- Performance scheduling: reduce the learning rate by a factor when the validation error stops dropping.

  ```python
  lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
  ```
  
- 1cycle scheduling: ToRead https://arxiv.org/abs/1803.09820
  To implement it, create a custom callback that modified the LR at each iteration (update the Lr by changing *self.model.optimizer.lr*) 

**ALTERNATIVE WAY TO IMPLEMENT LEARNING SCHEDULES IN KERAS (tf.keras)**
Use one of the schedules available in *keras.optimizers.schedules*, then pass this learning rate to any optimizer

```python
# EXAMPLE: EXPONENTIAL DECAY

s = 20 * len(X_train) // 32 # number of steps in 20 epochs (batch size = 32)
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1) 
optimizer = keras.optimizers.SGD(learning_rate)

```
Pros: simple and when you save the model, the learning rate and its schedule get saved too.
Cons: not part of the Keras API

