# Machine Learning Templates Guide

Templates for classification and regression tasks using PyTorch, PyTorch Lightning, JAX (Flax NNX), and LightGBM (with Optuna).

## Other Resources

These are some good resources for more extensive, in depth learning:
* **[3Blue1Brown Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)**
* **[Andrej Karpathy Neural Networks](https://karpathy.ai/zero-to-hero.html)**
* **[Fast.ai](https://course.fast.ai/)**
* **[Kaggle Learn Tutorials](https://www.kaggle.com/learn)**

## Installation

Install all required dependencies using:
```bash
pip install -r requirements.txt
```

## Guides by Framework

* **[PyTorch & PyTorch Lightning](pytorch.md)**: Manual GPU/CPU workflows, mixed precision (AMP) scaling order, and LightningModule abstractions.
* **[JAX / Flax NNX](jax.md)**: Purely functional execution model, XLA compilation boundaries, and NNX state split/merge logic.
* **[LightGBM & Optuna](lgbm.md)**: Tabular model lifecycle, hyperparameter searching, and retraining flows.

## Knowledge Bank

For detailed concepts, mathematical formulas, and intuitive explanations, see the files in the [knowledge_bank/](knowledge_bank/) directory:
* **[Gradients and Backpropagation](knowledge_bank/1-gradients-backprop.md)**: Computational graphs, the chain rule, and PyTorch autograd behavior.
* **[Optimizers and Schedulers](knowledge_bank/2-optimizers-schedulers.md)**: Math and intuition for SGD, Momentum, RMSprop, Adam, AdamW, and Cosine Annealing.
* **[Regularization and Normalization](knowledge_bank/3-regularization-normalization.md)**: Underfitting vs. overfitting diagnostics, dropout, weight decay, early stopping, Batch Norm, and Layer Norm.
* **[Loss Functions](knowledge_bank/4-loss-funcs.md)**: Choosing loss functions for classification and regression tasks.
* **[MNIST Notebook](knowledge_bank/mnist.ipynb)**: How to build a basic model to classify handwritten digits using PyTorch.

---

## Deep Learning Tuning

To adapt the templates to a new dataset, follow this step-by-step approach to tune your model.

### 1. Choosing a Model Architecture
Start with a simple baseline and scale up as needed:
* **The Baseline (Underfitting)**: Start with a simple linear or 2-layer network. If the training loss remains high and flat, the model lacks capacity (underfitting). 
* **Increasing Capacity**: Add hidden layers or increase layer width (e.g., `64` $\rightarrow$ `128` $\rightarrow$ `256`) to learn complex relationships. Use ReLU as a default activation between layers.
* **Overfitting**: If training loss drops close to zero but validation loss rises or plateaus, the model is memorizing the training set. Resolve this by adding **Dropout** (`0.1` to `0.5`) or **Normalization** (Batch Norm or Layer Norm).

*For a detailed explanation of underfitting, overfitting, and normalization types, see [Regularization and Normalization](knowledge_bank/3-regularization-normalization.md).*

### 2. Tuning Key Hyperparameters
Tune parameters in the following order:
* **Learning Rate (LR)**: If the loss goes to `NaN` or spikes, the learning rate is too high. If the loss remains flat, the learning rate is too low. Start with `1e-3` or `3e-4` for AdamW.
* **Learning Rate Schedulers**: Schedulers like **Cosine Annealing** decrease the step size over time, helping the model settle into narrow minima.
* **Batch Size**: Larger batches (`128`, `256`) train faster but require more VRAM. Smaller batches (`16`, `32`) add noise that acts as a regularizer, though training runs slower. If you double the batch size, you should generally double the learning rate.
* **Weight Decay**: Restricts weight sizes to encourage simpler decision boundaries. Use values between `1e-4` and `1e-2`.
* **Choosing an Optimizer**: Use **AdamW** by default. Use **SGD with Momentum** if you need slightly better generalization on tabular or image benchmarks at the cost of slower convergence.

*For optimizer update math and parameter definitions, see [Optimizers and Schedulers](knowledge_bank/2-optimizers-schedulers.md).*

### 3. Selecting a Loss Function
Match your loss function to your output labels and task type:

| Task Type | Output Activation | Expected Target Format | Loss Function (PyTorch) |
| :--- | :--- | :--- | :--- |
| **Multi-class Classification** | Linear (Logits) | Class index (`LongTensor` from `0` to `C-1`) | `nn.CrossEntropyLoss()` |
| **Binary Classification** | Linear (Logits) | Float targets (`0.0` or `1.0`) | `nn.BCEWithLogitsLoss()` |
| **Standard Regression** | Linear (No activation) | Continuous float values | `nn.MSELoss()` |
| **Robust Regression** | Linear (No activation) | Continuous float values (ignores outliers) | `nn.HuberLoss()` |

*For regression loss trade-offs and classification logits math, see [Loss Functions](knowledge_bank/4-loss-funcs.md).*

### 4. Loss Curves
Monitor training and validation loss curves to decide your next step:
* **Underfitting (High Train, High Val)**: Increase model capacity (layers/units), reduce dropout, or increase the learning rate.
* **Overfitting (Low Train, High Val)**: Increase dropout (`0.2` to `0.5`), increase weight decay, or simplify the model architecture.
* **Good Fit (Low Train, Low Val)**: Stop training when the validation loss begins to flatline or start rising.
