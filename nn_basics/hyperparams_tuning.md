# Hyperparameter Tuning

## 1. Hyperparameter Priority

Not all hyperparameters are equally important. Spend most of your search budget on the high-priority ones.

| Priority | Hyperparameter | Typical default |
|---|---|---|
| 🔴 Most important | Learning rate $\alpha$ | tune |
| 🟠 Second tier | Momentum $\beta_1$ | 0.9 |
| 🟠 Second tier | Mini-batch size $B$ | 64–512 |
| 🟠 Second tier | Hidden units per layer | tune |
| 🟡 Third tier | Number of layers $L$ | tune |
| 🟡 Third tier | Learning rate decay | tune |
| ⬜ Rarely tuned | Adam $\beta_2$ | 0.999 |
| ⬜ Rarely tuned | Adam $\varepsilon$ | $10^{-8}$ |

---

## 2. Search Strategies

### 2.1 Random Search over Grid Search

**Grid search** evaluates all combinations of a fixed set of values per hyperparameter. If one hyperparameter turns out to matter much more than the others, you've wasted most of your compute budget exploring redundant combinations of the unimportant one.

**Random search** samples each trial independently at random. For the same number of trials, it explores more distinct values of every hyperparameter — especially valuable when you don't know in advance which ones will matter most.

> Rule: **always prefer random search** in deep learning.

### 2.2 Coarse-to-Fine Search

1. **Coarse phase:** sample randomly over the full hyperparameter space; identify the promising region.
2. **Fine phase:** zoom into that region and sample more densely.

Repeat until convergence. This focuses compute where it matters without manually guessing ranges.

### 2.3 Pandas vs. Caviar

| Approach | When to use | How it works |
|---|---|---|
| **Panda** (babysit one model) | Limited compute, large dataset | Train one model; monitor it daily and manually nudge hyperparameters |
| **Caviar** (train many in parallel) | Ample compute | Launch many models simultaneously with different configs; pick the best |

Re-evaluate your hyperparameters periodically (at least every few months), even on the same problem — data distributions shift, hardware changes, and what worked before may no longer be optimal.

---

## 3. Sampling on the Right Scale

Sampling uniformly at random is only appropriate when the effect of the hyperparameter is approximately linear across its range.

### 3.1 Linear scale

Appropriate for discrete or bounded-range hyperparameters:
- Number of hidden units: e.g. 50–100 → sample integers uniformly.
- Number of layers $L$: e.g. 2, 3, 4 → grid or uniform sample.

### 3.2 Log scale

Appropriate when the hyperparameter spans several orders of magnitude, or when its effect is multiplicative:

**Learning rate $\alpha \in [10^{-4},\, 1]$:**

```python
r = -4 * np.random.rand()   # r ~ Uniform(-4, 0)
alpha = 10 ** r             # alpha ~ Uniform on log scale
```

**General recipe** for $\alpha \in [10^a,\, 10^b]$:

```python
r = np.random.uniform(a, b)
alpha = 10 ** r
```

### 3.3 Log scale for $\beta$ (momentum / EWA)

$\beta \in [0.9, 0.999]$ looks like a small range but is highly sensitive near 1. Sample via $1 - \beta$ on a log scale:

$$1 - \beta \in [0.001, 0.1] = [10^{-3}, 10^{-1}]$$

```python
r = np.random.uniform(-3, -1)
beta = 1 - 10 ** r
```

**Why:** the effective window is $\frac{1}{1-\beta}$, so $\beta = 0.999$ averages 1000 steps while $\beta = 0.99$ averages 100. A small linear change near 1 causes a massive change in behavior — the log scale distributes samples proportionally to this impact.

---

## 4. Batch Normalization

Batch normalization (Ioffe & Szegedy, 2015) makes hyperparameter search much easier, enables training of very deep networks, and acts as a mild regularizer.

### 4.1 Batch Norm for a single layer

Given pre-activation values $z^{(1)}, \ldots, z^{(m)}$ in layer $\ell$ (on a mini-batch of size $m$):

$$\mu = \frac{1}{m}\sum_{i=1}^{m} z^{(i)}$$

$$\sigma^2 = \frac{1}{m}\sum_{i=1}^{m} (z^{(i)} - \mu)^2$$

$$z^{(i)}_{\text{norm}} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

$$\tilde{z}^{(i)} = \gamma\, z^{(i)}_{\text{norm}} + \beta$$

$\gamma$ and $\beta$ are **learnable parameters** (one per hidden unit per layer), updated by gradient descent (or Adam/RMSprop) just like $W$ and $b$. They allow the network to learn the optimal mean and variance for each layer's activations — setting $\gamma = \sqrt{\sigma^2 + \varepsilon}$ and $\beta = \mu$ recovers the identity map.

> **Note:** the $\beta$ in batch norm is unrelated to the $\beta$ hyperparameters in momentum/Adam.

### 4.2 Integration into a deep network

Because batch norm subtracts the mean of $z^{[\ell]}$, the bias term $b^{[\ell]}$ is cancelled out and can be eliminated. The bias role is taken over by $\beta^{[\ell]}$.

**Forward pass (per layer $\ell$, per mini-batch $\{t\}$):**

$$Z^{[\ell]} = W^{[\ell]} A^{[\ell-1]} \quad\text{(no bias needed)}$$

$$Z^{[\ell]}_{\text{norm}} = \frac{Z^{[\ell]} - \mu^{[\ell]}}{\sqrt{(\sigma^{[\ell]})^2 + \varepsilon}}$$

$$\tilde{Z}^{[\ell]} = \gamma^{[\ell]} \odot Z^{[\ell]}_{\text{norm}} + \beta^{[\ell]}$$

$$A^{[\ell]} = g^{[\ell]}(\tilde{Z}^{[\ell]})$$

**Parameters per layer:** $W^{[\ell]},\, \gamma^{[\ell]},\, \beta^{[\ell]}$ (no $b^{[\ell]}$).

### 4.3 Why batch norm works

**1. Reduces internal covariate shift.**  
From layer $\ell$'s perspective, the distribution of inputs from earlier layers keeps changing as those layers update. Batch norm pins the mean and variance of each layer's input distribution (to values controlled by $\gamma, \beta$), so later layers see a more stable distribution and can learn more independently.

**2. Slight regularization.**  
$\mu$ and $\sigma^2$ are estimated from a mini-batch, not the whole dataset — they carry noise. This adds a small stochastic perturbation to each layer's activations (similar to dropout). The effect is mild and depends on batch size: **larger batches → less noise → less regularization**. Do not rely on batch norm as a primary regularizer.

**3. Allows higher learning rates.**  
By keeping activations in a normalized range, it prevents activations from exploding or vanishing, enabling faster training.

### 4.4 Batch norm at test time

At test time you often process a single example — you cannot compute a meaningful $\mu$ or $\sigma^2$ on a single sample. Instead, maintain a **running (exponentially weighted) average** of $\mu$ and $\sigma^2$ across mini-batches during training:

$$\mu_{\text{run}} \leftarrow \beta_{\text{run}}\, \mu_{\text{run}} + (1-\beta_{\text{run}})\, \mu^{\{t\}}$$

At test time, substitute $\mu_{\text{run}}$ and $\sigma^2_{\text{run}}$ into the normalization formula. Deep learning frameworks handle this automatically.

---

## 5. Softmax Regression (Multi-class Classification)

### 5.1 Softmax activation

For $C$ classes, the output layer has $n^{[L]} = C$ units. Instead of sigmoid, apply:

$$t_i = e^{z^{[L]}_i} \qquad (i = 1, \ldots, C)$$

$$\boxed{a^{[L]}_i = \hat{y}_i = \frac{t_i}{\sum_{j=1}^{C} t_j} = \frac{e^{z^{[L]}_i}}{\sum_{j=1}^{C} e^{z^{[L]}_j}}}$$

The output is a probability vector: $\sum_i \hat{y}_i = 1$, all $\hat{y}_i \geq 0$.

**Special case:** $C = 2$ reduces to logistic regression.

### 5.2 Loss function (cross-entropy)

For a single example with ground-truth one-hot label $y \in \mathbb{R}^C$:

$$\mathcal{L}(\hat{y}, y) = -\sum_{j=1}^{C} y_j \log \hat{y}_j$$

Since $y$ is one-hot (only one $y_k = 1$, rest zero), this simplifies to:

$$\mathcal{L}(\hat{y}, y) = -\log \hat{y}_k$$

where $k$ is the true class. Minimizing the loss maximizes the predicted probability of the correct class.

**Cost over the full training set:**

$$J = \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

### 5.3 Backprop for the output layer

The gradient of the loss with respect to $z^{[L]}$ is:

$$dz^{[L]} = \hat{y} - y$$

This is a $C \times 1$ vector (or $C \times m$ for a mini-batch). All other backprop equations proceed as normal from here.

---

## 6. Advanced Hyperparameter Optimization Methods

Beyond random/grid search, several more principled algorithms exist.

### 6.1 Bayesian Optimization

Builds a **probabilistic surrogate model** (usually a Gaussian Process) of the objective function $f(\lambda)$ (validation loss as a function of hyperparameters $\lambda$). Uses an **acquisition function** to decide where to evaluate next, trading off exploration vs. exploitation.

**Algorithm:**
1. Evaluate $f$ at a few random points.
2. Fit surrogate model to all observations.
3. Maximize acquisition function to pick next $\lambda^*$.
4. Evaluate $f(\lambda^*)$, update model, repeat.

Common acquisition functions:
- **Expected Improvement (EI):** $\mathbb{E}[\max(f(\lambda) - f^*, 0)]$
- **Upper Confidence Bound (UCB):** $\mu(\lambda) + \kappa\,\sigma(\lambda)$
- **Probability of Improvement (PI):** $P(f(\lambda) > f^* + \xi)$

**Pros:** sample-efficient — finds good hyperparameters with far fewer trials than random search.  
**Cons:** surrogate model fitting becomes expensive with many hyperparameters ($\gtrsim 20$); each evaluation must be sequential.  
**Libraries:** `scikit-optimize`, `GPyOpt`, `Ax` (Meta), `SMAC`.

### 6.2 Tree-structured Parzen Estimator (TPE)

Used by **Hyperopt** and **Optuna**. Models $p(\lambda \mid \text{good})$ and $p(\lambda \mid \text{bad})$ separately (instead of the full $p(\text{score} \mid \lambda)$) and maximizes the ratio $\frac{p(\lambda \mid \text{good})}{p(\lambda \mid \text{bad})}$.

Works well on high-dimensional spaces and handles conditional hyperparameters (e.g. momentum only matters if you use SGD).

```python
import optuna

def objective(trial):
    lr    = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.8, 0.99)
    units = trial.suggest_int("units", 32, 512)
    # ... build and train model, return val_loss
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
print(study.best_params)
```

### 6.3 Population-Based Training (PBT)

Trains a **population** of models in parallel. Periodically, low-performing models **copy** the weights of high-performing ones and **perturb** their hyperparameters. Simultaneously optimizes weights and hyperparameters.

**Pros:** adapts hyperparameters during training (e.g. learning rate schedule emerges automatically); very efficient on large clusters.  
**Cons:** requires running many models simultaneously.

### 6.4 Hyperband / ASHA

Extends random search with early stopping. Allocates a small budget (e.g. few epochs) to many configurations, keeps the top fraction, gives them more budget, and repeats — like a tournament bracket.

- **Hyperband:** tries multiple bracket sizes to avoid committing to a fixed early-stop threshold.
- **ASHA (Asynchronous Successive Halving):** asynchronous version; does not wait for all workers to finish before promoting the next cohort.

**Pros:** much faster than random search when poor configs can be identified early.  
**Cons:** some hyperparameters (e.g. batch size, architecture) affect performance trajectories non-monotonically — a config that looks bad at epoch 5 might be best at epoch 100.

### 6.5 Summary

| Method | Parallelism | Sample efficiency | Best for |
|---|---|---|---|
| Grid search | Full | Low | ≤2 hyperparams |
| Random search | Full | Medium | General baseline |
| Bayesian Opt (GP) | Low (sequential) | High | ≤15 hyperparams, expensive eval |
| TPE (Optuna) | Medium | High | General, handles conditionals |
| Hyperband / ASHA | High | High | Fast early-stopping signal |
| PBT | High | High | Large clusters, long training |

---

## 7. Practical Checklist

1. **Start with random search** over the most important hyperparameters ($\alpha$, hidden units, batch size).
2. **Use log scale** for $\alpha$, $\beta$, weight decay $\lambda$.
3. **Use linear/integer scale** for number of layers, units per layer (within a reasonable range).
4. **Coarse-to-fine:** identify the good region first, then refine.
5. **Enable batch norm** by default — it reduces sensitivity to $\alpha$ and initialization.
6. **Re-tune periodically** — as data grows or infrastructure changes, old hyperparameters may go stale.
7. **Fix $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$** for Adam; only tune $\alpha$ (and optionally $\beta_1$).
8. If using **Optuna/Hyperband**, set a wall-time budget and let the scheduler decide; do not hand-pick which trials to kill.
