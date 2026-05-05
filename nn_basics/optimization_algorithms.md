# Optimization Algorithms

## 1. Mini-Batch Gradient Descent

### Notation

| Superscript | Meaning | Example |
|---|---|---|
| $(i)$ | $i$-th training example | $x^{(i)}$ |
| $[l]$ | Layer $l$ | $W^{[l]}$ |
| $\{t\}$ | Mini-batch $t$ | $X^{\{t\}},\, Y^{\{t\}}$ |

### Variants

| Variant | Mini-batch size | Behaviour |
|---|---|---|
| Batch GD | $m$ (whole dataset) | Smooth descent, one update per epoch, very slow on large data |
| Stochastic GD | 1 | One update per example, very noisy, loses vectorization speedup |
| **Mini-batch GD** | $1 < B < m$ | Best of both: vectorized, frequent updates |

### Algorithm (one epoch)

$$\text{for } t = 1, \ldots, \lceil m/B \rceil:$$

1. Forward prop on $X^{\{t\}}$ → compute $\hat{Y}^{\{t\}}$
2. Compute cost: $J^{\{t\}} = \frac{1}{B}\sum_{i=1}^{B}\mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2B}\sum_l \|W^{[l]}\|_F^2$
3. Backprop on $J^{\{t\}}$ → compute $dW^{[l]},\, db^{[l]}$
4. Update: $W^{[l]} \leftarrow W^{[l]} - \alpha\, dW^{[l]},\quad b^{[l]} \leftarrow b^{[l]} - \alpha\, db^{[l]}$

With batch GD, 1 epoch = 1 gradient step. With mini-batch GD, 1 epoch = $\lceil m/B \rceil$ steps.

### Choosing mini-batch size

- $m \leq 2000$: just use batch GD.
- Otherwise, typical sizes: **64, 128, 256, 512** (powers of 2 — fits CPU/GPU memory layout).
- Make sure $X^{\{t\}}, Y^{\{t\}}$ fit in CPU/GPU memory; exceeding memory causes a sharp performance cliff.
- Mini-batch size is a hyperparameter to tune.

### Cost curve behaviour

- Batch GD: $J$ decreases monotonically every iteration.
- Mini-batch GD: $J^{\{t\}}$ oscillates slightly but trends downward overall (each mini-batch is a different sample).

---

## 2. Exponentially Weighted Averages (EWA)

The foundation of momentum, RMSprop, and Adam.

$$\boxed{V_t = \beta\, V_{t-1} + (1-\beta)\,\theta_t}, \qquad V_0 = 0$$

$V_t$ approximates an average over roughly $\frac{1}{1-\beta}$ recent values:

| $\beta$ | Effective window | Curve |
|---|---|---|
| 0.5 | ~2 steps | Very noisy, fast to adapt |
| 0.9 | ~10 steps | Smooth, moderate lag |
| 0.98 | ~50 steps | Very smooth, slow to adapt |

**Intuition:** the weight on $\theta_{t-k}$ decays as $\beta^k$. After $\frac{1}{1-\beta}$ steps the weight has decayed to $\approx e^{-1} \approx 0.37$ of the current weight.

**Memory efficiency:** requires only one stored number; overwrites it in-place each step.

### 2.1 Bias Correction

Because $V_0 = 0$, early estimates are systematically too small. Fix:

$$\hat{V}_t = \frac{V_t}{1 - \beta^t}$$

As $t \to \infty$, $\beta^t \to 0$ so $\hat{V}_t \approx V_t$ — correction only matters in the first few steps. In practice, momentum often skips bias correction; Adam always includes it.

---

## 3. Gradient Descent with Momentum

Almost always faster than plain gradient descent. Smooths oscillations by using an EWA of the gradients.

### Algorithm

Initialize: $V_{dW} = 0,\; V_{db} = 0$

On each iteration $t$, compute $dW, db$ via backprop, then:

$$\boxed{V_{dW} = \beta_1\, V_{dW} + (1-\beta_1)\, dW}$$

$$\boxed{V_{db} = \beta_1\, V_{db} + (1-\beta_1)\, db}$$

$$W \leftarrow W - \alpha\, V_{dW}, \qquad b \leftarrow b - \alpha\, V_{db}$$

### Why it works

On a cost surface with narrow valleys (fast curvature in one direction, slow in another):
- Oscillating directions: positive and negative gradients average out → $V_{dW}$ small → slow movement → dampened oscillations.
- Progress direction: gradients consistently point the same way → $V_{dW}$ large → fast movement.

This allows a **larger learning rate** without diverging, because oscillations are suppressed.

**Physical analogy:** the gradient provides acceleration; $\beta_1$ acts as friction preventing unlimited speed-up.

### Hyperparameters

| Parameter | Typical value | Notes |
|---|---|---|
| $\alpha$ | tune | Most important to tune |
| $\beta_1$ | **0.9** | Rarely needs tuning |

---

## 4. RMSprop (Root Mean Square Prop)

Adapts the learning rate **per parameter** by dividing by the root of a running average of squared gradients.

### Algorithm

Initialize: $S_{dW} = 0,\; S_{db} = 0$

On each iteration $t$:

$$\boxed{S_{dW} = \beta_2\, S_{dW} + (1-\beta_2)\, dW^2}$$

$$\boxed{S_{db} = \beta_2\, S_{db} + (1-\beta_2)\, db^2}$$

$$W \leftarrow W - \alpha\, \frac{dW}{\sqrt{S_{dW}} + \varepsilon}, \qquad b \leftarrow b - \alpha\, \frac{db}{\sqrt{S_{db}} + \varepsilon}$$

The squaring is **elementwise**. $\varepsilon \approx 10^{-8}$ prevents division by zero.

### Why it works

In directions with large gradients (high oscillation), $S_{dW}$ is large → update is scaled down → oscillations dampened.  
In directions with small gradients (slow progress), $S_{dW}$ is small → update is scaled up → faster progress.

Net effect: effectively uses a larger $\alpha$ without diverging.

### Hyperparameters

| Parameter | Typical value |
|---|---|
| $\alpha$ | tune |
| $\beta_2$ | 0.999 |
| $\varepsilon$ | $10^{-8}$ |

*RMSprop was first proposed by Geoffrey Hinton in a Coursera lecture, not a traditional paper.*

---

## 5. Adam (Adaptive Moment Estimation)

Combines momentum (first moment) and RMSprop (second moment). The most widely used optimizer in practice.

### Algorithm

Initialize: $V_{dW} = S_{dW} = V_{db} = S_{db} = 0$

On iteration $t$, compute $dW, db$, then:

**Momentum update (first moment):**

$$V_{dW} = \beta_1\, V_{dW} + (1-\beta_1)\, dW$$

$$V_{db} = \beta_1\, V_{db} + (1-\beta_1)\, db$$

**RMSprop update (second moment):**

$$S_{dW} = \beta_2\, S_{dW} + (1-\beta_2)\, dW^2$$

$$S_{db} = \beta_2\, S_{db} + (1-\beta_2)\, db^2$$

**Bias correction:**

$$\hat{V}_{dW} = \frac{V_{dW}}{1-\beta_1^t}, \qquad \hat{V}_{db} = \frac{V_{db}}{1-\beta_1^t}$$

$$\hat{S}_{dW} = \frac{S_{dW}}{1-\beta_2^t}, \qquad \hat{S}_{db} = \frac{S_{db}}{1-\beta_2^t}$$

**Parameter update:**

$$\boxed{W \leftarrow W - \alpha\,\frac{\hat{V}_{dW}}{\sqrt{\hat{S}_{dW}} + \varepsilon}}$$

$$\boxed{b \leftarrow b - \alpha\,\frac{\hat{V}_{db}}{\sqrt{\hat{S}_{db}} + \varepsilon}}$$

### Hyperparameters

| Parameter | Recommended default | Notes |
|---|---|---|
| $\alpha$ | tune | Most important |
| $\beta_1$ | **0.9** | First moment (momentum) |
| $\beta_2$ | **0.999** | Second moment (RMSprop) |
| $\varepsilon$ | $10^{-8}$ | Never needs tuning |

In practice: fix $\beta_1, \beta_2, \varepsilon$ at defaults; only tune $\alpha$.

**Name origin:** Adam = **Ada**ptive **M**oment estimation. $\beta_1$ estimates the first moment (mean of gradients), $\beta_2$ the second moment (uncentered variance of gradients).

---

## 6. Learning Rate Decay

As training converges, a smaller $\alpha$ helps the algorithm settle near the minimum instead of oscillating around it.

### Common schedules

**Step decay (most common):**

$$\alpha = \frac{\alpha_0}{1 + \text{decayRate} \times \text{epochNum}}$$

**Exponential decay:**

$$\alpha = \alpha_0 \cdot e^{-\text{decayRate} \times \text{epochNum}}$$

**Square-root decay:**

$$\alpha = \frac{\alpha_0}{\sqrt{\text{epochNum}}}$$

**Manual decay:** reduce $\alpha$ by hand when training plateaus (feasible when training takes hours/days).

`decayRate` and $\alpha_0$ are hyperparameters to tune.

---

## 7. The Problem of Local Optima

### Old intuition (incorrect for deep learning)

Low-dimensional plots suggest many local minima where gradient descent could get stuck.

### Modern understanding

In a very high-dimensional space (e.g. 20,000 parameters), a zero-gradient point requires the cost surface to curve **upward in every direction simultaneously**. The probability of this is $\approx 2^{-20000}$ — essentially impossible.

Instead, zero-gradient points are almost always **saddle points**: some directions curve up, others curve down. Gradient descent can escape saddle points.

### Real problem: Plateaus

A **plateau** is a region where $\|\nabla J\| \approx 0$ over many steps, so learning is very slow. The algorithm must wander across the flat region before finding a slope.

Algorithms like momentum, RMSprop, and Adam help escape plateaus faster by accumulating velocity across flat regions.

---

## 8. Algorithm Comparison

| Algorithm | Adapts per-param LR | Uses gradient history | Bias correction | Key hyperparams |
|---|---|---|---|---|
| SGD / Mini-batch GD | No | No | — | $\alpha$ |
| Momentum | No | Yes (1st moment) | Optional | $\alpha, \beta_1$ |
| RMSprop | Yes (2nd moment) | Yes (2nd moment) | Optional | $\alpha, \beta_2$ |
| **Adam** | **Yes (2nd moment)** | **Yes (1st + 2nd)** | **Yes** | $\alpha, \beta_1, \beta_2$ |

**Recommendation:** Use **Adam** with mini-batch gradient descent as the default. It works well across a wide range of architectures and requires minimal tuning beyond the learning rate $\alpha$.

---

## 9. Practical Tips

- **Shuffle** training data before splitting into mini-batches each epoch to avoid systematic ordering effects.
- **Power-of-2** mini-batch sizes (64–512) align with CPU/GPU memory and run faster.
- **Tune $\alpha$ first** — it has the largest impact on convergence speed.
- **Plot $J$ vs. epoch** to diagnose: if $J$ increases or oscillates wildly, $\alpha$ is too large; if $J$ decreases very slowly, $\alpha$ is too small.
- With **dropout**, $J$ is non-deterministic — temporarily set `keep_prob = 1` to verify $J$ decreases monotonically before enabling dropout.
- **Learning rate decay** is more important for fine-grained convergence near the end of training than for early-phase learning.
