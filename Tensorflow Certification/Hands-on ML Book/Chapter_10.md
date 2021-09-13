
# Hyperparameters

## Number of hidden layers
  - Many simple problems can be modelled with 1 or 2 hidden layers
  - Complex problems may require dozens of layer (or hundreds but non fully connected) but in this case you tipically use Transfer Learning and don't need to train all of them.
## Number of neuros per hidden layer
  - Input and Output defined by type of problem
  - Usually better results increasing number layers than number of neurons
  - large neural networks almost never get stuck in local minima, and even when they do these local optima are almost as good as the global optimum. However, they can still get stuck on long plateaus for a long time.
## Hyperparameters
- learning rate: 
    - in general, optimal LR is around 1/2 the maximum LR (when the algorithm diverges)
    - important: it depends on the other hyperparameters, especially the batch size
    - technique: plot Loss vs LR (e.g. LR from 10^-5 to 10)
- optimizer:
    - choose better than mini-batch Gradient Descent
- batch size: 
    - trade-off between training time and instabilities
    - technique: large batch size + learning rate warmup
- activation function:
    - for hidden layers: ReLU is a good default
    - for output layers: it depends on the task
- number of iterations:
    - use Early Stopping technique

To READ: https://arxiv.org/abs/1803.09820
Resources: https://playground.tensorflow.org/
