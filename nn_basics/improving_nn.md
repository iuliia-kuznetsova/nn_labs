# Improving Deep Neural Networks

## 1. Train / Dev / Test Sets

Applied ML is an iterative process: idea → code → experiment → refine. Splitting data correctly speeds up this cycle.

| Dataset size | Typical split |
|---|---|
| Small (~1k–10k) | 60% train / 20% dev / 20% test |
| Large (~1M+) | 98% train / 1% dev / 1% test |

**Rules of thumb:**
- Dev and test sets must come from the **same distribution**.
- Training data may come from a different distribution (e.g. web-crawled images vs. user uploads).
- It is acceptable to have no test set if an unbiased final estimate is not needed (train + dev only).

---

## 2. Bias and Variance

### 2.1 Mathematical Origin of Bias and Variance

Assume the true relationship is:

$$y = f(x) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2_\varepsilon)$$

where $f(x)$ is the true unknown function and $\varepsilon$ is irreducible noise. You train a model $\hat{f}(x)$ on a dataset $\mathcal{D}$. Because $\mathcal{D}$ is random, $\hat{f}$ is also random — it changes with every different training set you could draw.

The **expected squared error** at a point $x$, averaged over all possible training datasets, decomposes as:

$$\boxed{\mathbb{E}_{\mathcal{D}}\!\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(f(x) - \mathbb{E}[\hat{f}(x)]\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\!\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]}_{\text{Variance}} + \underbrace{\sigma^2_\varepsilon}_{\text{Irreducible noise}}}$$

**Bias²** — how far the average prediction of your model family is from the truth:

$$\text{Bias}^2 = \left(f(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x)]\right)^2$$

This is a property of the **model class**, not any specific training run. A linear model fitting quadratic data will always be systematically wrong — no amount of data fixes this.

**Variance** — how much predictions fluctuate across different training sets:

$$\text{Variance} = \mathbb{E}_{\mathcal{D}}\!\left[\left(\hat{f}(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x)]\right)^2\right]$$

A complex model memorizes noise in whichever training set it sees, so $\hat{f}$ swings wildly across datasets.

**Irreducible noise** $\sigma^2_\varepsilon$ — inherent randomness in $y$ itself. No model can beat this floor.

The total error has a **U-shape** as a function of model complexity:

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \sigma^2_\varepsilon$$

```
Error
  │          Total error
  │        ╲            ╱
  │  Variance╲        ╱
  │            ╲    ╱
  │       Bias²  ╲╱  ← optimal complexity
  │──────────────────────── complexity
```

| Model | Bias | Variance |
|---|---|---|
| Constant $\hat{f}(x) = c$ | High — always predicts the mean | Zero — never changes with data |
| Polynomial of degree $m \to \infty$ | Near zero — fits any $f$ | High — wildly sensitive to training set |

Deep learning escapes the U-curve because with large networks and enough data: **Bias² → 0** (the network can express $f$), and with regularization + more data **Variance stays controlled**. Instead of picking a point on the tradeoff curve, you move the whole curve down.

### 2.2 Bias-Variance Tradeoff in classical ML

**Bias** is the error from a model being too simple to capture the true pattern — it **underfits**. A high-bias model makes the same systematic mistakes regardless of which training data it sees.

**Variance** is the error from a model being too sensitive to the specific training data it was trained on — it **overfits**. A high-variance model fits the training set well but fails to generalize.

In traditional ML, most things you could do to reduce one would worsen the other:

- **Increase model complexity** → bias ↓, variance ↑
- **Decrease model complexity** → bias ↑, variance ↓
- **Add more training data** → helps variance, but not bias
- **Reduce regularization** → bias ↓, variance ↑
- **Increase regularization** → bias ↑, variance ↓

### 2.3 Bias-Variance Tradeoff in the Modern Big-data Deep Models Era

Diagnose model quality by comparing training error and dev error (assuming Bayes/human error ≈ 0%):

| Train error | Dev error | Diagnosis |
|---|---|---|
| Low (~1%) | High (~11%) | High variance (overfitting) |
| High (~15%) | Similar (~16%) | High bias (underfitting) |
| High (~15%) | Even higher (~30%) | High bias **and** high variance |
| Low (~0.5%) | Low (~1%) | Low bias, low variance (ideal) |

**Key insight:** 
In the modern big-data era there are tools that push **one down without hurting the other**:

| Problem | Fix | Effect on bias | Effect on variance |
|---|---|---|---|
| High bias | Train a **bigger network** (with regularization) | ↓ | neutral |
| High variance | Get **more training data** | neutral | ↓ |
| High variance | Add **regularization** | slight ↑ (small) | ↓ |

The workflow becomes a sequential checklist: fix bias first, then fix variance. Each step doesn't undo the other. The tradeoff still exists in theory, but in practice it's much less of a constraint.

In deep learning there is less of a hard bias–variance tradeoff because:
- A bigger (well-regularized) network almost always reduces bias without hurting variance.
- More data reduces variance without hurting bias.
---

## 3. Basic Recipe for Machine Learning

```
Train model
    ↓
High bias?  (look at train error)
    → Bigger network / train longer / new architecture
    ↓
High variance?  (look at dev error)
    → More data / regularization / new architecture
    ↓
Done (low bias + low variance)
```

---

## 4. Regularization

### 4.1 L2 Regularization (Weight Decay)

**Logistic regression cost with L2:**

$$J(w, b) = \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}\|w\|_2^2$$

where $\|w\|_2^2 = \sum_{j=1}^{n_x} w_j^2 = w^\top w$.

**Neural network cost with L2 (Frobenius norm):**

$$J = \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}\sum_{\ell=1}^{L} \left\|W^{[\ell]}\right\|_F^2$$

$$\left\|W^{[\ell]}\right\|_F^2 = \sum_{i=1}^{n^{[\ell]}} \sum_{j=1}^{n^{[\ell-1]}} \left(W^{[\ell]}_{i,j}\right)^2$$

**Modified gradient (backprop + regularization term):**

$$dW^{[\ell]} \leftarrow dW^{[\ell]} + \frac{\lambda}{m} W^{[\ell]}$$

**Weight update (why it's called "weight decay"):**

$$W^{[\ell]} \leftarrow W^{[\ell]} - \alpha \, dW^{[\ell]} = \left(1 - \frac{\alpha\lambda}{m}\right)W^{[\ell]} - \alpha \cdot (\text{backprop term})$$

The factor $\left(1 - \frac{\alpha\lambda}{m}\right)$ is slightly less than 1, so the weights shrink on every step.

**Why it reduces overfitting:**
1. Large $\lambda$ forces $W \approx 0$, effectively simplifying the network toward logistic regression.
2. Small weights → small $z$ values → activations stay in the linear region of tanh → network behaves more like a linear model, less able to overfit.

**L1 regularization** (less common): replaces $\|w\|_2^2$ with $\|w\|_1 = \sum |w_j|$. Produces sparse weights.

$\lambda$ is a hyperparameter tuned on the dev set.

---

### 4.2 Dropout Regularization

Randomly zero out each hidden unit with probability $1 - \text{keep\_prob}$ on every forward pass.

**Inverted dropout (recommended implementation) for layer $\ell$:**

```python
d_l = np.random.rand(*a_l.shape) < keep_prob   # boolean mask
a_l = a_l * d_l                                 # zero out units
a_l = a_l / keep_prob                           # rescale (inverted dropout)
```

Dividing by `keep_prob` ensures the **expected value** of $a^{[\ell]}$ is unchanged, so test-time predictions need no extra scaling.

**At test time:** do not apply dropout (use the full network).

**Why it works:**
- Each unit cannot rely on any single input feature → forced to spread weights → similar effect to L2 regularization.
- Formally, dropout is an adaptive form of L2 regularization with different penalties per weight.

**Practical notes:**
- Apply stronger dropout (lower `keep_prob`) to larger layers (more parameters → more risk of overfitting).
- Dropout is most common in computer vision where data is scarce.
- Dropout makes the cost function $J$ non-deterministic → turn off dropout (set `keep_prob = 1`) when plotting $J$ to verify it decreases.

---

### 4.3 Other Regularization Methods

**Data augmentation:** artificially expand the training set (e.g. flip, crop, rotate images; distort digits). Acts as a regularizer at near-zero extra cost.

**Early stopping:** stop training when dev-set error starts increasing.
- Advantage: tries many values of $\|W\|$ in one training run.
- Disadvantage: couples optimization (minimize $J$) and regularization (reduce variance), making each harder to tune independently. L2 regularization is preferred because it decouples the two.

---

## 5. Normalizing Inputs

### Steps

$$\mu = \frac{1}{m}\sum_{i=1}^{m} x^{(i)}, \qquad x \leftarrow x - \mu$$

$$\sigma^2 = \frac{1}{m}\sum_{i=1}^{m} (x^{(i)})^2, \qquad x \leftarrow \frac{x}{\sigma}$$

Use the **same** $\mu$ and $\sigma^2$ computed on the training set to normalize the test set.

### Why it helps

Unnormalized features on very different scales (e.g. $x_1 \in [0, 1000]$, $x_2 \in [0, 1]$) create a highly elongated cost surface where gradient descent oscillates and needs a tiny learning rate. After normalization the surface is more symmetric and gradient descent converges faster with larger steps.

---

## 6. Vanishing / Exploding Gradients

In a deep network with $L$ layers, the output is (approximately):

$$\hat{y} = W^{[L]} W^{[L-1]} \cdots W^{[1]} x$$

- If $W^{[\ell]} \approx 1.5\,\mathbf{I}$, then activations (and gradients) grow as $\sim 1.5^L$ → **exploding**.
- If $W^{[\ell]} \approx 0.5\,\mathbf{I}$, then activations (and gradients) shrink as $\sim 0.5^L$ → **vanishing**.

Gradient descent becomes very slow or unstable. Careful weight initialization partially addresses this.

---

## 7. Weight Initialization for Deep Networks

To keep $z = \sum_j w_j x_j$ from exploding or vanishing, set:

$$\text{Var}(w_j) = \frac{c}{n^{[\ell-1]}}$$

where $n^{[\ell-1]}$ is the number of inputs to the layer. In practice:

| Activation | Initialization | Constant $c$ | Name |
|---|---|---|---|
| ReLU | `np.random.randn(...) * np.sqrt(2 / n^[l-1])` | 2 | He initialization |
| Tanh | `np.random.randn(...) * np.sqrt(1 / n^[l-1])` | 1 | Xavier initialization |
| Alternative | `np.random.randn(...) * np.sqrt(2 / (n^[l-1] + n^[l]))` | — | Glorot |

The variance parameter can also be treated as a hyperparameter to tune.

---

## 8. Gradient Checking

Used to verify that backprop is implemented correctly. Uses the **two-sided (centered) difference** approximation, which has error $O(\varepsilon^2)$ vs. $O(\varepsilon)$ for one-sided:

$$g(\theta) \approx \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2\varepsilon}$$

### Procedure

1. Reshape and concatenate all parameters $W^{[1]}, b^{[1]}, \ldots, W^{[L]}, b^{[L]}$ into one vector $\theta$.
2. Do the same for all gradients to get $d\theta$.
3. For each component $i$, compute:

$$d\theta_{\text{approx}}^{[i]} = \frac{J(\theta_1, \ldots, \theta_i + \varepsilon, \ldots) - J(\theta_1, \ldots, \theta_i - \varepsilon, \ldots)}{2\varepsilon}$$

4. Check the relative difference:

$$\text{ratio} = \frac{\|d\theta_{\text{approx}} - d\theta\|_2}{\|d\theta_{\text{approx}}\|_2 + \|d\theta\|_2}$$

| Ratio | Verdict |
|---|---|
| $\sim 10^{-7}$ or smaller | Correct |
| $\sim 10^{-5}$ | Examine closely |
| $\sim 10^{-3}$ or larger | Likely a bug |

### Implementation notes
- Only use grad check for **debugging**, not during training (too slow).
- If check fails, inspect individual components of $d\theta_{\text{approx}} - d\theta$ to localize the bug (e.g. all large errors in $db^{[\ell]}$ → bug in that layer's bias gradient).
- **Include the regularization term** in $J$ when checking.
- **Does not work with dropout** (cost is not deterministic). Turn dropout off, verify grad check passes, then re-enable dropout.
- Consider running grad check both at initialization (small $w$) and after some training steps (larger $w$) to catch bugs that only appear away from zero.
