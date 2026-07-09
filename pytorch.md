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
