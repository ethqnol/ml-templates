# Regularization and Normalization

## 1. Underfitting vs. Overfitting

Underfitting happens when both the training and validation loss remain high and flat. The model either isn't complex enough to learn the patterns, or training was cut short. You can fix this by adding more layers or units, reducing dropout, increasing the learning rate, or training longer.

Overfitting happens when the training loss keeps dropping, but the validation loss starts to rise. The model is memorizing the training data instead of learning general patterns. You can fix this by adding dropout, increasing weight decay, simplifying the architecture, or getting more training data.

## 2. Regularization Techniques

Regularization techniques prevent overfitting by constraining the model's capacity. You can use multiple techniques together.

### Dropout (`nn.Dropout(p)`)
During training, dropout randomly turns off a fraction $p$ (usually $0.1$ to $0.5$) of neurons on each step. This forces the model to distribute its learning across the entire network instead of relying heavily on a few specific neurons. During evaluation (`model.eval()`), dropout is turned off, and activations are scaled by $(1 - p)$ to compensate.

### Weight Decay (L2 Regularization)
Weight decay adds a penalty to the loss function based on the size of the weights:

$$
L_{regularized} = L_{data} + \frac{\lambda}{2} \sum w_i^2
$$

This keeps the weights small. Large weights usually mean the model is focusing too hard on specific features of the training data. Keeping weights small creates simpler, smoother decision boundaries. The decay coefficient $\lambda$ is usually set between `1e-4` and `1e-2`.

### Early Stopping
Early stopping stops training when the validation loss stops improving after a set number of epochs (called patience). This stops the model right before it starts memorizing noise in the training set.

## 3. Normalization Techniques

Normalization stabilizes activations across layers, allowing for faster convergence and deeper network architectures.

### Batch Normalization (`nn.BatchNorm1d`)
We can think of batch normalization like grading a class of students on a curve. If different teachers grade different assignments, the raw scores might be on completely different scales. To make them comparable, you adjust the grades of the whole class (the batch) so they have a consistent average and spread. In a neural network, as weights update, the inputs to deeper layers constantly shift around. Batch normalization scales the activations across the entire batch so that the next layer always receives inputs on a stable scale.

In practice, batch normalization normalizes activations across the batch dimension:

$$
\hat{x} = \frac{x - \mu_{batch}}{\sqrt{\sigma^2_{batch} + \epsilon}}
$$

During training, it computes the mean and variance across the mini-batch for each feature independently. During evaluation, it uses running averages accumulated during training. This technique works best for feedforward and convolutional networks trained with stable batch sizes.

### Layer Normalization (`nn.LayerNorm`)
Back to our grading analogy, layer normalization is like grading a single student relative only to their own average. Instead of comparing a student to their classmates (which changes depending on who else is in the batch), you look at their score on one subject compared to their scores in all other subjects. In a neural network, layer normalization normalizes the features within a single sample. Because it doesn't compare across different samples, its behavior doesn't change when you use small or dynamic batch sizes.

In practice, layer normalization normalizes activations across the feature dimension for each sample independently:

$$
\hat{x} = \frac{x - \mu_{sample}}{\sqrt{\sigma^2_{sample} + \epsilon}}
$$

It computes the mean and variance across all features within a single sample. Layer normalization behaves identically during training and evaluation, making it ideal for recurrent neural networks, transformers, and training with small or dynamic batch sizes.
