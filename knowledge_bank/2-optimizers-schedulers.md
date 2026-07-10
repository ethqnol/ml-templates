# Optimizers and Schedulers

## 1. Optimization Algorithms

Optimization algorithms use parameter gradients ($\nabla L(\theta)$) to step parameters in the direction of steepest descent.

### Symbol Table
| Symbol | Name | Description |
| :--- | :--- | :--- |
| $\theta$ | Parameters | The model's weights and biases. |
| $\eta$ | Learning Rate | The step size scaling factor for updates. |
| $\nabla L(\theta)$ | Gradient | The derivative of the loss, indicating the direction of steepest increase. |
| $v_t$ | Velocity | Accumulated gradients used to speed up steps in consistent directions. |
| $\beta$ | Momentum/Decay Rate | Coefficient (typically $0.9$) scaling historical velocity or running averages. |
| $s_t$ | Squared Gradient Average | Running average of squared gradients, used to scale learning rate per parameter. |
| $m_t$ | Gradient Average | Running average of gradients, tracking the first moment (direction). |
| $\epsilon$ | Epsilon | A tiny constant (e.g., $10^{-8}$) to avoid dividing by zero. |
| $\lambda$ | Weight Decay | Scaling factor for L2 regularization directly on weights. |

### Stochastic Gradient Descent (SGD)
We can think of this like finding your way down a mountain in the dark. In this analogy, the gradient is the slope of the section of mountain you're standing on. Since you can't see in the dark, if you notice that the ground is sloping downhill in a certain direction then you know that this is the direction you should travel in. And every couple step you re-evaluate your direction to make sure you're still going downhill. 

But how large should your steps be? In our mountain analogy, the size of your steps is the learning rate. If our steps are too large we might accidentally step over the bottom of the mountain and end up on the mountain across from us. But if our steps are too small, we'll die of starvation or something before we reach the bottom. Thus, our learning rate (step size) should be adjusted to have a balance.

In practice, SGD updates parameters by stepping in the opposite direction of the gradient, scaled by the learning rate ($\eta$):
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

#### SGD with Momentum
SGD with Momentum is a more powerful version of our inital idea. The concept of momentum is like picking up speed while traveling down the hill. Where small bumps or holes in the mountain may slow us down in SGD, adding momentum lets us maintain our speed and walk through these obstacles. On the other hand, this can also cause us to overshoot the bottom of the mountain and end up on the mountain across from us. Thus, we need to adjust the momentum coefficient to have a balance.

SGD with Momentum accumulates a velocity vector $v_t$ based on previous gradients, scaled by a momentum coefficient $\beta$ (typically $0.9$). This helps the updates speed up down steep slopes and roll through flat regions:
$$v_{t+1} = \beta v_t + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

### RMSprop
RMSprop adapts the step size for each parameter individually. In our mountain analogy, the terrain isn't uniform. Certain paths are steep cliffs, while others are long, flat stretches of land. If we use the same step size everywhere, we might slip off a cliff or get stuck on a ridge. RMSprop is like having a brain. If you notice that you are on a very steep path with large drops (large gradients), you'd naturally shrink your steps to keep from falling off a cliff. If you are on a flat, slow path (small gradients), you'd naturally take larger steps to get across it faster.

In practice, RMSprop divides the learning rate by a running average of the squared gradients ($s_t$):
$$s_{t+1} = \beta s_t + (1 - \beta) (\nabla L(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_{t+1}} + \epsilon} \nabla L(\theta_t)$$

### Adam (Adaptive Moment Estimation)
Adam combines the concepts of momentum and RMSprop. This makes it the most robust way to find the bottom of the mountain under most conditions, which is why it is the default choice for most machine learning models.

In practice, Adam tracks both the average gradient direction ($m_t$, first moment) and the average gradient variance ($v_t$, second moment):
$$m_{t+1} = \beta_1 m_t + (1 - \beta_1) \nabla L(\theta_t)$$
$$v_{t+1} = \beta_2 v_t + (1 - \beta_2) (\nabla L(\theta_t))^2$$

After bias correction, parameters are updated as:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \hat{m}_{t+1}$$

### AdamW (Adam with Decoupled Weight Decay)
In standard Adam, we penalize large weights (weight decay) by adding a penalty to the loss, but this accidentally alters our running averages of the direction and step size. AdamW fixes this by applying weight decay directly to the weights, keeping the adaptive step calculations clean.

In practice, AdamW decouples weight decay and applies it directly to the parameters:
$$\theta_{t+1} = \theta_t - \eta \left( \lambda \theta_t + \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon} \right)$$
This is the default starting optimizer for most deep learning models.

## 2. Choosing and Scheduling the Learning Rate (LR)

### Selecting the Initial Learning Rate
A learning rate that is too high causes step updates to overshoot the minimum, making the loss function oscillate or return `NaN`. A learning rate that is too low results in tiny updates, causing slow training or getting stuck in local minima.

Common starting baselines:
* AdamW: `1e-3` or `3e-4`
* SGD: `1e-1` or `1e-2`

### Learning Rate Schedulers
Reducing the step size over time helps the model settle into narrow minima late in training.

#### Cosine Annealing (PyTorch `optim.lr_scheduler.CosineAnnealingLR`)
Cosine annealing decreases the learning rate from its initial value to near-zero following a cosine curve:
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)$$

To use this schedule, set $T_{max}$ to the total number of training epochs so the decay completes by the end of training. It is robust and does not require manual decay milestones.
