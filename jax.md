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
