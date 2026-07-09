# JAX / Flax NNX

## JAX / Flax NNX Loop

JAX requires purely functional math operations (no in-place mutations of variables). **Flax NNX** bridges this by letting you define stateful OOP modules that are automatically split into stateless data structures before executing compiled JAX math blocks.

The training flow looks different, but methodologically, it is identical to PyTorch: we perform a forward pass, compute a loss, compute gradients, and update parameters. The main difference is that JAX uses just-in-time (JIT) compilation to compile the training step into a fast XLA kernel.  

### Flow
```
           +---------------------------------------------+
           | Stateful Python Objects (Model / Optimizer) |
           +----------------------+----------------------+
                                  |
                                  | (nnx.jit compilation boundary)
                                  v
           +---------------------------------------------+
           | State Split: Graph Def + Parameter PyTree   |
           +----------------------+----------------------+
                                  |
                                  v
           +---------------------------------------------+
           | Pure JAX function compiles via XLA          |
           | Value & Grads traced on parameter inputs    |
           +----------------------+----------------------+
                                  |
                                  v
           +---------------------------------------------+
           | Update: optimizer.update(grads)             |
           +----------------------+----------------------+
                                  |
                                  | (nnx.jit exit)
                                  v
           +---------------------------------------------+
           | New Parameters merged back in-place to OOP  |
           +---------------------------------------------+
```

### Pseudocode training step
```python
# nnx.jit compiles the functional split-run-merge cycle into a fast XLA kernel
@nnx.jit
def train_step(model, optimizer, batch):
    # loss function must take model as the FIRST argument for gradient tracking
    def loss_fn(model):
        logits = model(batch["x"])
        loss = softmax_cross_entropy(logits, batch["y"])
        return loss, logits

    # trace parameters and compute loss + grads
    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    # updates parameters (mutations are safe inside JIT boundary)
    optimizer.update(grads)
    return loss
```

---

## JAX / Flax NNX Customization Parameters

To customize JAX/Flax NNX templates, modify the values inside the `__main__` block.

For example, in [jax_classification.py](jax_classification.py#L213-L236), we can customize features count, target class count, dataset path, and target label column:

```python
if __name__ == "__main__":
    INPUT_FEATURES = 64  # Set to the number of input features in your dataset
    NUM_CLASSES = 2      # Set to the number of target classes
    config = Config()
    set_seed(config.seed) 

    # initialize rngs
    rngs = nnx.Rngs(config.seed)
    model = Model(input_dim=INPUT_FEATURES, num_classes=NUM_CLASSES, rngs=rngs)
    
    # initialize optax adamw optimizer function
    def optimizer_fn(m):
        return nnx.Optimizer(m, optax.adamw(config.learning_rate, config.weight_decay))

    trainer = Trainer(
        model=model,
        optimizer_fn=optimizer_fn,
        dataset= ,  # Set to dataset CSV path (e.g. "dataset.csv")
        target= ,   # Set to target column name string
        config=config
    )
     
    trainer.train()
```

## JAX / Flax NNX Customization Options

You can customize the architecture, optimizer, learning rate scheduler, and loss function depending on the dataset and task requirements.

### Model Architecture

The default JAX templates use a single linear layer. To add hidden layers and activation functions:

* **JAX Classification** (`jax_classification.py` lines 51-57):
  ```python
  class Model(nnx.Module):
      def __init__(self, input_dim: int, num_classes: int, rngs: nnx.Rngs):
          self.linear1 = nnx.Linear(input_dim, 128, rngs=rngs)
          self.linear2 = nnx.Linear(128, num_classes, rngs=rngs)

      def __call__(self, x):
          x = nnx.relu(self.linear1(x))
          return self.linear2(x)
  ```
* **JAX Regression** (`jax_regression.py` lines 49-55):
  Modify the `Model` class in the same way, setting the final linear layer output size to `output_dim`.

### Schedulers and Optimizers

Optimizers and schedules are defined inside the `optimizer_fn` initialization block:

* **Optimizers**: To use SGD instead of AdamW:
  * Update `optimizer_fn` in `jax_classification.py` (line 224) and `jax_regression.py` (line 222) to:
    ```python
    def optimizer_fn(m):
        return nnx.Optimizer(m, optax.sgd(config.learning_rate, momentum=0.9))
    ```
* **Schedulers**: Schedulers scale the learning rate as epochs progress.
  * To chain a cosine learning rate decay schedule instead of a fixed rate:
    ```python
    def optimizer_fn(m):
        schedule = optax.cosine_decay_schedule(init_value=config.learning_rate, decay_steps=config.epochs)
        return nnx.Optimizer(m, optax.adamw(learning_rate=schedule))
    ```

### Loss Functions

Change the loss function calculated inside the JITcompiled steps:

* **JAX Classification** (`jax_classification.py` lines 81-85):
  * For multi-label classification, swap `optax.softmax_cross_entropy_with_integer_labels` inside the inner `loss_fn` to:
    ```python
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch["y"]).mean()
    ```
* **JAX Regression** (`jax_regression.py` lines 79-83):
  * For outlier-robust regression, swap standard squared error inside the inner `loss_fn` to Huber loss:
    ```python
        loss = jnp.mean(optax.huber_loss(preds, batch["y"]))
    ```

### Configurations

Change batch size, epochs, and early stopping limits inside the `Config` dataclass:
* `jax_classification.py` (lines 60-70).
* `jax_regression.py` (lines 58-68).
