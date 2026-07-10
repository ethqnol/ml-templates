# Gradients and Backpropagation


## 1. Computational Graphs

Neural networks represent calculations using directed computational graphs:
* **Nodes**: Operations (like multiplication, addition, or activation functions).
* **Edges**: The flow of data. Directed edges carry tensors (multi-dimensional arrays) from one operation to the next.

For example, the linear operation $y = wx + b$ looks like this:
```
  x ---\
        (Multiply) ---> wx ---\
  w ---/                       (Add) ---> y
  b --------------------------/
```

In the **forward pass**, PyTorch runs these calculations to compute outputs. In the **backward pass**, PyTorch traces this graph in reverse to calculate how much each parameter ($w$ and $b$) contributed to the final loss.


## 2. The Chain Rule

Backpropagation is how neural networks compute gradients, and it is built on the calculus **Chain Rule**.

### The Intuition: A Chain Reaction
If $A$ affects $B$, and $B$ affects $C$, the chain rule calculates how a change in $A$ impacts $C$ by multiplying the intermediate rates of change:

$$
\text{Rate of change of } C \text{ relative to } A = (\text{Rate of change of } C \text{ relative to } B) \times (\text{Rate of change of } B \text{ relative to } A)
$$

Think of three connected gears: **Gear A**, **Gear B**, and **Gear C**.
* Turning Gear A once makes Gear B spin twice ($\frac{\partial B}{\partial A} = 2$).
* Turning Gear B once makes Gear C spin three times ($\frac{\partial C}{\partial B} = 3$).

How much does Gear C turn if you spin Gear A once?

$$
\frac{\partial C}{\partial A} = \frac{\partial C}{\partial B} \cdot \frac{\partial B}{\partial A} = 3 \times 2 = 6 \text{ turns}
$$

Multiplying these gear ratios (local derivatives) gives the overall rate of change.

### Math Example
For an input $x$, intermediate step $y$, and loss $L$:

$$
y = 3x \quad \text{and} \quad L = y^2
$$

To find how $L$ changes relative to $x$ ($\frac{\partial L}{\partial x}$):

1. **Direct Substitution**:
   Combine the formulas first: $L = (3x)^2 = 9x^2$.
   Take the derivative:

   $$
   \frac{\partial L}{\partial x} = 18x
   $$

2. **Using the Chain Rule**:
   First, calculate the local derivatives layer-by-layer:
   * How $L$ changes w.r.t $y$:

     $$
     \frac{\partial L}{\partial y} = 2y
     $$

   * How $y$ changes w.r.t $x$:

     $$
     \frac{\partial y}{\partial x} = 3
     $$
   
   Multiply them:

   $$
   \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = (2y) \cdot 3 = 6y
   $$

   Substitute $y = 3x$ back:

   $$
   \frac{\partial L}{\partial x} = 6(3x) = 18x
   $$

Both approaches give the same result. But in a neural network with millions of parameters across hundreds of layers, substituting equations directly is impossible. The chain rule allows PyTorch to compute local derivatives for each layer independently, then multiply them together to get the final gradient. 


## 3. Automatic Differentiation (Autograd)

Instead of requiring you to write out derivative formulas manually, PyTorch uses an automatic differentiation engine called **Autograd** to track and calculate gradients.

### How Autograd Tracks Operations
Setting `requires_grad=True` on a tensor tells PyTorch to track all operations performed on it:

```python
import torch
x = torch.tensor(3.0, requires_grad=True)
```

PyTorch tracks three main attributes for these tensors:
* **`.data`**: The tensor's current value (e.g., `3.0`).
* **`.grad`**: The calculated gradient (starts as `None`, populated by `.backward()`).
* **`.grad_fn`**: The operation that created the tensor (e.g., `<AddBackward0>`, `<MulBackward0>`).

Every calculation on a tracked tensor builds a dynamic **computational graph** in the background.

### Autograd in Code
Here is how PyTorch handles the math example ($y = x^2$ and $z = 2y$ at $x = 3.0$):

```python
import torch

# 1. Initialize input and request gradient tracking
x = torch.tensor(3.0, requires_grad=True)

# 2. Forward pass
y = x ** 2  # y.grad_fn points to <PowBackward0>
z = 2 * y   # z.grad_fn points to <MulBackward0>

# 3. Backward pass
z.backward()

# 4. Read the gradient
print(x.grad)  # Output: 12.0
```

#### How `z.backward()` runs:
1. **Start**: Autograd starts at the output $z$.
2. **Trace Step 1**: It checks $z$'s `.grad_fn` (`MulBackward0`) and calculates the derivative with respect to $y$, which is $2$.
3. **Trace Step 2**: It moves to $y$'s `.grad_fn` (`PowBackward0`) and calculates the derivative with respect to $x$, which is $2x$.
4. **Multiply**: It applies the chain rule: $\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x} = 2 \cdot 2x = 4x$.
5. **Evaluate & Store**: With $x = 3.0$, the final gradient is $4 \times 3 = 12$. This is saved in `x.grad`.


## 4. Zeroing Gradient Buffers

By default, PyTorch adds new gradients to existing ones on each `.backward()` call instead of overwriting them. This is useful for techniques like gradient accumulation (simulating larger batch sizes when memory is low).

For standard training, you only want gradients for the current batch. If you don't clear the old gradients, they will accumulate and corrupt your weight updates.

To prevent this:
* Call `optimizer.zero_grad()` at the start of each training step.
* Or use `optimizer.zero_grad(set_to_none=True)`. This method is faster.