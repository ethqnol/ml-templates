# Loss Functions

## 1. Classification Loss Functions

### Cross-Entropy Loss (PyTorch `nn.CrossEntropyLoss`)
Use cross-entropy for multi-class classification where each sample belongs to exactly one of $C$ classes.

For a single sample with true class index $c$:

$$
L = -\log \left( \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}} \right) = -z_c + \log\left(\sum_{j=1}^C e^{z_j}\right)
$$

PyTorch expects raw logits as inputs to `nn.CrossEntropyLoss()`. It handles log-softmax and negative log-likelihood together internally for numerical stability. Do not apply a softmax layer to your model outputs when using this loss.

### Binary Cross-Entropy with Logits (PyTorch `nn.BCEWithLogitsLoss`)
Use binary cross-entropy for binary classification ($0$ or $1$) or multi-label classification (predicting multiple independent classes at once).

For target probability $y \in \{0, 1\}$ and predicted logit $z$:

$$
L = -[y \log \sigma(z) + (1 - y) \log(1 - \sigma(z))]
$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation.

`nn.BCEWithLogitsLoss()` takes raw logits directly. This is more stable than applying a Sigmoid followed by standard `nn.BCELoss()`, which can suffer from numerical underflow.

## 2. Regression Loss Functions

### Mean Squared Error / L2 Loss (PyTorch `nn.MSELoss`)
MSE computes the average squared difference between predictions and targets:

$$
L = (y_{pred} - y)^2
$$

This is the default for regression. Because it squares the errors, large errors are penalized heavily, making the model sensitive to outliers.

### Mean Absolute Error / L1 Loss (PyTorch `nn.L1Loss`)
MAE (L1 loss) computes the average absolute difference between predictions and targets:

$$
L = |y_{pred} - y|
$$

It is less sensitive to outliers than MSE, but the gradient is discontinuous at zero, which can cause weight updates to bounce around late in training.

### Huber Loss (PyTorch `nn.HuberLoss`)
Huber loss behaves like MSE when errors are small, and like L1 when errors are large (set by threshold $\delta$, default $1.0$):

$$
L = \begin{cases} 
\frac{1}{2}(y_{pred} - y)^2 & \text{if } |y_{pred} - y| \le \delta \\
\delta \left(|y_{pred} - y| - \frac{1}{2}\delta\right) & \text{otherwise}
\end{cases}
$$

This balances outlier robustness with a smooth gradient as errors approach zero.

## 3. Loss Function Selection Matrix

| Task Description | Output Activation | Expected Model Target Format | PyTorch Loss Class |
| :--- | :--- | :--- | :--- |
| **Multi-class Classification** | Linear (Logits) | Class indexes ($0$ to $C-1$ as Long Tensors) | `nn.CrossEntropyLoss()` |
| **Binary Classification** | Linear (Logits) | Binary values ($0.0$ or $1.0$ as Float Tensors) | `nn.BCEWithLogitsLoss()` |
| **Standard Regression** | Linear (No activation) | Continuous values (Float Tensors) | `nn.MSELoss()` |
| **Outlier-Robust Regression**| Linear (No activation) | Continuous values (Float Tensors) | `nn.HuberLoss()` |
