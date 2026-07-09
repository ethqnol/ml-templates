# PyTorch

## PyTorch Loop

Dynamically detect your hardware and switch between a GPU (CUDA) workflow and a CPU (or Apple Silicon MPS) workflow.

### CPU vs. GPU Training

* **On GPU (CUDA)**: Runs with float16 Mixed Precision (AMP) enabled to save VRAM and speed up training. It copies data asynchronously to the GPU using pinned memory (`pin_memory=True` and `non_blocking=True`), scales gradients via `GradScaler` to prevent underflow, unscales them before clipping, and steps the optimizer using the scaler.
* **On CPU (or MPS)**: Runs in standard float32 precision. Gradient scaling is disabled because float32 has enough range to prevent underflow. Pinned memory is turned off since there are no host-to-device bottlenecks, and the optimizer is stepped directly.

```
For each Epoch:
   1. Set model to Train Mode
   2. For each mini-batch (X, Y):
      a. Copy features and labels to target device (asynchronously via pinned memory on CUDA)
      b. Delete existing gradients from memory (set_to_none=True)
      c. Forward Pass (casts math to Float16 under Autocast on CUDA)
      d. Compute Loss
      e. Backpropagate loss to calculate gradients (uses GradScaler on CUDA to prevent underflow)
      f. Clip gradient norms to prevent exploding gradients (unscales gradients first on CUDA)
      g. Update parameters using optimizer step (uses scaler step on CUDA to filter NaNs)
   3. Set model to Eval Mode & disable gradient tracking (torch.no_grad())
   4. For each validation mini-batch:
      a. Run forward pass
      b. Accumulate validation metrics
   5. Step Learning Rate Scheduler
   6. Checkpoint weights if validation loss improved; increment patience otherwise
   7. Stop training if patience limit is reached
```

---

## 2. PyTorch Lightning Loop

PyTorch Lightning abstracts the training loop boilerplate into a state machine managed by a `Trainer` while keeping the mathematical operations inside the `LightningModule`.

### Architectural Flow
```
+-------------------------------------------------------------+
|                     pl.Trainer                              |
|  - Automates hardware selection, loops, epoch metrics       |
|  - Controls Callbacks (EarlyStopping, ModelCheckpoint)      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  pl.LightningModule                         |
|  - training_step(): Forward pass -> Loss -> Log metrics     |
|  - validation_step(): Evaluation metrics                    |
|  - configure_optimizers(): Setup optimizer & LR scheduler   |
+-------------------------------------------------------------+
```

---

## 3. Customization Parameters

To customize the PyTorch templates for a specific dataset or task, modify the initialization values inside the `__main__` execution block.

In all the PyTorch templates (including lightning), customizing the number of features, output target classes, the dataset file path, and target label column can be done in the `if __name__ == "__main__":` block.

For example, in `torch_classification.py` (line 233-255):

```python
if __name__ == "__main__":
    INPUT_FEATURES = 64  # Set to the number of input features in your dataset
    NUM_CLASSES = 2      # Set to the number of target classes
    config = Config()
    set_seed(config.seed) 

    model = Model(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset= ,  # Set to dataset CSV path (e.g. "dataset.csv")
        target= ,   # Set to target column name string
        config=config
    )
```

## 4. Other Customization Options

You can customize the architecture, optimizer, learning rate scheduler, and loss function depending on the dataset and task requirements.

### Model Architecture

The default models use a single linear layer to map inputs to outputs; modifying the Model class will allow you to build more complex, non-linear models. For instance:

* **Standard PyTorch Classification** (`torch_classification.py` lines 43-51):
  ```python
  class Model(nn.Module):
      def __init__(self, input_dim: int, num_classes: int):
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(input_dim, 128),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(128, num_classes)
          )

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          return self.net(x)
  ```
* **PyTorch Regression** (`torch_regression.py` lines 42-49):
  Modify the `Model` class in the same way, setting the final linear layer output size to `output_dim`.
* **PyTorch Lightning Classification** (`lightning_classification.py` lines 44-52):
  Modify the model layers inside `ClassifierModule.__init__` and the operations inside `forward`.

### Schedulers and Optimizers

* **Optimizers**: To use SGD or other optimizers instead of AdamW, update the instantiation in the script execution blocks:
  * In `torch_classification.py` (line 241) and `torch_regression.py` (line 229):
    ```python
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    ```
* **Schedulers**: Schedulers scale the learning rate as epochs progress.
  * To use a step decay scheduler instead of cosine annealing in `torch_classification.py` (line 242) and `torch_regression.py` (line 230):
    ```python
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    ```
  * In PyTorch Lightning, modify the scheduler initialized inside `configure_optimizers` (`lightning_classification.py` lines 79-92 and `lightning_regression.py` lines 75-88).

### Loss Functions (Criterion)

Loss functions are defined at initialization. You can swap them to match your target distribution or task:

* **PyTorch Classification** (`torch_classification.py` line 240):
  * For binary classification, swap `nn.CrossEntropyLoss()` to:
    ```python
    criterion = nn.BCEWithLogitsLoss()
    ```
* **PyTorch Regression** (`torch_regression.py` line 228):
  * For outlier-robust regression, swap `nn.MSELoss()` to:
    ```python
    criterion = nn.HuberLoss()
    ```
* **PyTorch Lightning**: Update the criterion inside `ClassifierModule.__init__` (`lightning_classification.py` line 49) or `RegressorModule.__init__` (`lightning_regression.py` line 49).

### Configurations

Change batch size, epochs, and early stopping limits inside the `Config` dataclass:
* Standard PyTorch (`torch_classification.py` lines 55-73 and `torch_regression.py` lines 51-70).
* PyTorch Lightning (`lightning_classification.py` lines 21-39 and `lightning_regression.py` lines 18-36).

* **Batch Size**: The number of samples processed before updating weights. Larger batches process faster on GPUs because of parallelization, but they result in fewer weight updates per epoch and require more VRAM. Smaller batches update weights more frequently and introduce gradient noise which helps generalize better, but take longer to train.
* **Epochs**: Because early stopping halts training when validation loss stops improving, the exact number of epochs acts primarily as an upper limit.
* **Learning Rate**: Controls the step size taken towards minimizing loss. High learning rates train faster but can overshoot local minima or diverge. Low learning rates converge slowly or get stuck.
* **Weight Decay**: Adds a penalty for large weights to the loss function to prevent overfitting and encourage simpler decision boundaries.
