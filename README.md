# Machine Learning Templates Guide

Templates for classification and regression tasks using PyTorch, PyTorch Lightning, JAX (Flax NNX), and LightGBM (with Optuna).

## Installation

Install all required dependencies using:
```bash
pip install -r requirements.txt
```

## Guides by Framework

* **[PyTorch & PyTorch Lightning](pytorch.md)**: Manual GPU/CPU workflows, mixed precision (AMP) scaling order, and LightningModule abstractions.
* **[JAX / Flax NNX](jax.md)**: Purely functional execution model, XLA compilation boundaries, and NNX state split/merge logic.
* **[LightGBM & Optuna](lgbm.md)**: Tabular model lifecycle, hyperparameter searching, and retraining flows.

---

## Deep Learning Tuning 

To adapt the deep learning templates to a new dataset, the following approach is generally a good way to start tuning your model.

### 1. Choosing a Model Architecture

Start simple, and expand only as needed:

* **The Baseline (Underfitting)**: Always start with a simple linear or 2-layer network. If the training loss stops decreasing early and remains high, your model lacks the capacity to capture the data's patterns (underfitting). In most cases, this is both expected and good, as it justifies increasing model complexity. 
* **Increasing Capacity**: Add hidden layers or increase layer width (e.g., from `64` to `128` or `256` units) to allow the model to learn complex, non-linear relationships.
* **Non-Linearity**: Use non-linear activation functions (e.g. ReLU, GeLU, SwiGLU) between layers to enable the model to learn complex patterns. ReLU is a good default choice.
* **Regularizing (Overfitting)**: If your training loss drops close to zero but your validation loss increases or plateaus high, your model is memorizing the training set (overfitting). Introduce:
  * **Dropout (`nn.Dropout(p)`)**: Randomly drops a fraction `p` (typically `0.1` to `0.5`) of features during each training pass to prevent the network from relying too heavily on any single feature path.
  * **Layer / Batch Normalization**: Normalizes layer inputs to keep signals stable and prevent gradients from exploding or vanishing.


### 2. Tuning Key Hyperparameters

Adjust parameters in this order:

#### Learning Rate (LR)
* **Purpose**: The step size taken at each parameter update towards minimizing loss.
* **How to tune**: 
  * If the loss diverges (goes to `NaN` or spikes upward), the LR is too high.
  * If the loss barely decreases or stays flat, the LR is too low.
  * Typical range is `1e-4` to `1e-2` for AdamW.
* **Scheduling**: Schedulers decrease the learning rate as training progresses:
  * **Cosine Annealing**: Decays the learning rate to near-zero using a cosine curve, letting the model settle into clean minima late in training. This is enabled by default in the templates.

#### Batch Size
* **Purpose**: The number of samples the model processes before updating weights.
* **Trade-offs**:
  * **Larger Batches** (e.g., `128`, `256`): Leverage GPU cores better (faster epochs) and yield cleaner gradients, but require more VRAM and can get stuck in suboptimal local minima.
  * **Smaller Batches** (e.g., `16`, `32`): Noisier gradients act as a regularizer to escape local minima, though training runs slower.
  * **Linear Scaling Rule**: If you double your batch size, you should generally double your learning rate to account for fewer weight updates per epoch.

#### Weight Decay
* **Purpose**: Adds a penalty for large weights to the loss function, encouraging simple decision boundaries.
* **How to tune**: Use values between `1e-4` and `1e-2`. Higher weight decay acts as strong regularization against overfitting.

#### Choosing an Optimizer
* **AdamW** (Default): Decouples weight decay from the adaptive learning rate update. It is the best starting choice for most deep learning models. It converges fast and is robust to non-optimal learning rates.
* **SGD (Stochastic Gradient Descent) with Momentum**: Updates all parameters with a uniform learning rate. While it converges slower than AdamW and requires a careful learning rate decay schedule, it can sometimes lead to slightly better generalization on image/tabular benchmarks.
* **RMSprop**: Divides the learning rate by an exponentially decaying average of squared gradients. It is useful for sequence models (RNNs/LSTMs) or non-stationary reinforcement learning objectives.


### 3. Selecting a Loss Function

Choose a loss function that matches your output labels and distribution:

| Task Type | Objective | Loss Function (PyTorch) | Optax Loss (JAX) |
| :--- | :--- | :--- | :--- |
| **Multi-class Classification** | Target is a single label class index | `nn.CrossEntropyLoss()` | `softmax_cross_entropy_with_integer_labels` |
| **Binary Classification** | Target is `0` or `1` (single output sigmoid/logits) | `nn.BCEWithLogitsLoss()` | `sigmoid_binary_cross_entropy` |
| **Standard Regression** | Minimize squared differences (sensitive to outliers) | `nn.MSELoss()` | Squared Error (`(y_pred - y_true)**2`) |
| **Robust Regression** | Reduce outlier impact (linear penalty for large errors) | `nn.HuberLoss()` or `nn.L1Loss()` | `optax.huber_loss` or L1 loss |


### 4. Loss Curves

Watch training vs validation loss to decide what to change:

* **Underfitting (High Train, High Val)**: 
  * *Fix*: Increase model depth/width, reduce dropout, increase learning rate, or train for more epochs.
* **Overfitting (Low Train, High Val)**:
  * *Fix*: Increase dropout (`0.2` - `0.5`), increase weight decay, simplify the model layers, or gather more training data.
* **Good Fit (Low Train, Low Val, Val tracking slightly above Train)**:
  * *Action*: This is the optimal stopping point.
