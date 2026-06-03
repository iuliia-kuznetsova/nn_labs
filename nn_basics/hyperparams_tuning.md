# Hyperparameter Tuning

A practical reference for tuning neural networks: what to tune first, how to search efficiently, advanced schedulers, batch normalization, and interview-style pitfalls.

---

## 1. Hyperparameters

**Parameters** ($W^{[\ell]}$, $b^{[\ell]}$, and learnable batch-norm $\gamma$, $\beta$) are updated by gradient descent during training.

**Hyperparameters** are everything you choose *before* training that controls *how* learning happens. They determine the final learned parameters — hence “hyper.”

| Hyperparameter | Symbol | Effect |
|---|---|---|
| Learning rate | $\alpha$ | Step size; too large → divergence, too small → slow convergence |
| Momentum / Adam $\beta_1$ | $\beta_1$ | EMA of gradients; stabilizes SGD / Adam |
| Mini-batch size | $B$ | Noise in gradient estimates; throughput vs. generalization |
| Hidden units per layer | $n^{[\ell]}$ | Model capacity (width) |
| Number of layers | $L$ | Depth; more abstraction, harder optimization |
| Learning rate schedule / decay | — | Annealing $\alpha$ over training |
| Weight decay | $\lambda$ | L2 penalty strength |
| Adam $\beta_2$, $\varepsilon$ | $\beta_2$, $\varepsilon$ | Second-moment EMA and numerical stability |
| Activation, init, dropout rate | — | Architecture and regularization choices |

**Standard workflow**

1. Pick initial hyperparameters (literature, defaults, or a quick pilot run).
2. Train and evaluate on a **development (validation) set** — never tune on the test set.
3. Adjust and repeat until dev performance is satisfactory; run the test set **once** at the end.

Optimal values are problem- and dataset-specific; there is rarely a universal “best” setting.

### 1.1. Hyperparameter priority

Not all hyperparameters matter equally. Allocate most of your search budget to the high-impact ones.

| Priority | Hyperparameter | Typical default / note |
|---|---|---|
| Most important | Learning rate $\alpha$ | Always tune; often on a **log scale** |
| Second tier | Momentum $\beta_1$ (SGD) or Adam $\beta_1$ | 0.9 is a strong default |
| Second tier | Mini-batch size $B$ | 64–512; affects speed, noise, BN behavior |
| Second tier | Hidden units per layer | Tune width before obsessing over depth |
| Third tier | Number of layers $L$ | Can dominate capacity but costly to search |
| Third tier | Learning rate decay / schedule | Warmup, cosine, step decay |
| Third tier | Weight decay $\lambda$ | Often log-scale; coupled with $\alpha$ |
| Rarely tuned | Adam $\beta_2$ | 0.999 |
| Rarely tuned | Adam $\varepsilon$ | $10^{-8}$ |

**Rule of thumb:** tune $\alpha$ first (and optionally $\beta_1$, $B$, width). Fix Adam’s $\beta_2$ and $\varepsilon$ unless you have strong evidence they matter for your task.

**Panda vs. Caviar** (organizing the search process)

| Approach | When | How |
|---|---|---|
| **Panda** | Limited compute, large dataset | Train one model; monitor daily; manually nudge hyperparameters |
| **Caviar** | Ample parallel compute | Launch many configs at once; pick the best dev result |

Re-tune periodically (every few months on long-lived systems). Data drift, new data volume, and faster hardware can make old settings suboptimal.

### 1.2. Tricky interview questions about hyperparameters

**Q: What is the difference between a parameter and a hyperparameter?**  
A: Parameters are learned from data via gradients ($W$, $b$, …). Hyperparameters are set by the practitioner and define the training procedure (learning rate, architecture depth, batch size, …).

**Q: Why is the learning rate usually the #1 hyperparameter to tune?**  
A: It directly controls step size in parameter space. A wrong $\alpha$ by one order of magnitude often makes training diverge or stall, regardless of other choices.

**Q: Should you tune on the training set or the test set?**  
A: Neither for final selection — use a **validation (dev) set**. The test set must remain untouched until the very end to estimate generalization without optimistic bias.

**Q: If $\beta_2 = 0.999$ works everywhere, why mention it at all?**  
A: Defaults are robust for Adam, but in theory $\beta_2$ controls the EMA window of squared gradients. Interviewers check that you know it exists and that extreme changes near 1 affect effective averaging length — same intuition as momentum $\beta$ near 1.

**Q: Does a larger batch size always mean better training?**  
A: No. Larger $B$ gives lower-variance gradients and higher throughput, but can hurt generalization (sharper minima, less noise). Batch norm’s regularization strength also weakens with large batches.

**Q: Can you “overfit” the validation set?**  
A: Yes, if you run hundreds of trials and keep picking the best dev score, you indirectly fit the dev set. Mitigations: hold out a second validation split, report confidence intervals, or use fewer manual iterations.

---

## 2. Search strategies

Hyperparameter search treats validation loss (or another metric) as a black-box function $f(\lambda)$ where $\lambda$ is the hyperparameter vector. The goal is to find $\lambda^* \approx \arg\min_\lambda f(\lambda)$ within a budget of $N$ training runs.

### 2.1. Grid search

**Definition.** For each hyperparameter $j$, choose a finite set of values $\Lambda_j = \{v_{j,1}, \ldots, v_{j,k_j}\}$. **Grid search** evaluates $f(\lambda)$ on the full Cartesian product:

$$\Lambda_{\text{grid}} = \Lambda_1 \times \Lambda_2 \times \cdots \times \Lambda_d$$

If you use $k$ values per dimension and $d$ hyperparameters, you need **$k^d$** trials — exponential in $d$.

**Example.** $\alpha \in \{10^{-4}, 10^{-3}, 10^{-2}\}$, $B \in \{32, 64, 128\}$, 3 layers × 3 batch sizes = **9** runs. Add a 5th hyperparameter with 5 values each → $5^5 = 3125$ runs.

**Pros**

- Simple, reproducible, embarrassingly parallel.
- Fine coverage when **few** dimensions matter and ranges are small.

**Cons**

- Wastes budget when some dimensions barely affect $f$ (you repeat the same “good” combo with many values of an irrelevant hyperparameter).
- Curse of dimensionality: impractical beyond ~2–3 important continuous hyperparameters.

**When to use**

- Very low dimension ($d \leq 2$).
- Categorical choices with few levels (e.g. $L \in \{2,3,4\}$).
- As a **fine** phase after random search has located a promising region (small local grid).

---

### 2.2. Random search

**Definition.** Sample each trial independently:

$$\lambda^{(i)} \sim p(\lambda), \quad i = 1,\ldots,N$$

where $p(\lambda)$ is a product of per-dimension distributions (uniform on linear or log scale, categorical, etc.).

**Why it beats grid search (Bergstra & Bengio, 2012).**

Suppose only **one** of $d$ hyperparameters strongly affects $f$. Grid search with $N$ trials allocates only about $N^{1/d}$ **distinct** values to that dimension. Random search allocates about **$N$** distinct values to *every* dimension.

Formally, for a 2D grid with $n$ points per axis ($n^2$ trials), each axis gets $n$ values. For $n^2$ random trials, each axis gets $\mathcal{O}(n^2)$ independent samples — far better coverage of the important axis when the other is irrelevant.

**Rule:** In deep learning with many hyperparameters, **prefer random search** over grid search for the coarse exploration phase.

#### Coarse-to-fine search

1. **Coarse:** sample $\lambda$ broadly over the full plausible range (random, often log-scaled).
2. **Identify** a promising region (lowest dev loss cluster).
3. **Fine:** restrict sampling to a smaller box around that region; optionally switch to a small grid.

Repeat until diminishing returns. This concentrates compute without hand-designing a single narrow range upfront.

#### Sampling on the right scale

Uniform sampling in the *original* units is wrong when effects are **multiplicative** or span orders of magnitude.

**Linear scale** — use when the effect is roughly uniform over the range:

- Hidden units: e.g. integers uniform in $[50, 100]$.
- Number of layers: $L \in \{2, 3, 4\}$ (small categorical grid is fine).

**Log scale** — use for $\alpha$, weight decay $\lambda$, and any range spanning decades:

For $\alpha \in [10^a, 10^b]$:

$$r \sim \mathrm{Uniform}(a, b), \qquad \alpha = 10^r$$

```python
r = np.random.uniform(-4, 0)   # log10 range
alpha = 10 ** r
```

**Log scale for momentum / EMA $\beta$ near 1**

Effective memory length is $\frac{1}{1-\beta}$. A linear step in $\beta$ near 1 changes behavior drastically. Sample $1-\beta$ on a log scale:

$$1 - \beta \in [10^{-3}, 10^{-1}] \Rightarrow r \sim \mathrm{Uniform}(-3, -1),\quad \beta = 1 - 10^r$$

```python
r = np.random.uniform(-3, -1)
beta = 1 - 10 ** r
```

**Sensitivity intuition:** $\beta: 0.9 \to 0.9005$ is negligible; $0.999 \to 0.9995$ changes the EMA window from 1000 to 2000 steps — a huge behavioral shift.

---

### 2.3. Bayesian Optimization

**Idea.** Replace expensive evaluations of $f(\lambda)$ with a **surrogate model** that estimates $f$ and uncertainty. Pick the next $\lambda$ by maximizing an **acquisition function** that balances **exploitation** (sample where the surrogate predicts low loss) and **exploration** (sample where uncertainty is high).

**Surrogate (common choice: Gaussian Process).**

Given observations $\mathcal{D} = \{(\lambda^{(i)}, y^{(i)})\}_{i=1}^n$ with $y^{(i)} = f(\lambda^{(i)})$, a GP places a prior on functions and yields posterior mean $\mu(\lambda)$ and variance $\sigma^2(\lambda)$:

$$f(\lambda) \mid \mathcal{D} \sim \mathcal{GP}(\mu(\lambda), \sigma^2(\lambda))$$

Kernel (e.g. Matérn 5/2) encodes smoothness in hyperparameter space.

**Acquisition functions** (maximize to pick next point):

| Name | Formula (concept) | Behavior |
|---|---|---|
| **Expected Improvement (EI)** | $\mathbb{E}[\max(f^* - f(\lambda), 0)]$ with $f^* = \min_i y^{(i)}$ | Targets regions likely to beat the best so far |
| **Upper Confidence Bound (UCB)** | $\mathrm{UCB}(\lambda) = \mu(\lambda) - \kappa\,\sigma(\lambda)$ (minimize loss) | $\kappa$ trades exploration vs exploitation |
| **Probability of Improvement (PI)** | $P(f(\lambda) < f^* - \xi)$ | Greedy; can under-explore |

**Algorithm**

1. Evaluate $f$ at a few random $\lambda$ (initialization).
2. Fit surrogate on $\mathcal{D}$.
3. $\lambda_{\text{next}} = \arg\max_\lambda \alpha(\lambda)$ (acquisition).
4. Evaluate $f(\lambda_{\text{next}})$, append to $\mathcal{D}$, repeat until budget exhausted.

**Pros:** Very **sample-efficient** when each training run is costly and $d$ is moderate ($\lesssim 15$–20).  
**Cons:** GP fitting is $\mathcal{O}(n^3)$ in number of observations; inherently **sequential**; struggles in very high dimensions.  
**Libraries:** `scikit-optimize`, `GPyOpt`, `Ax`, `BoTorch`.

---

### 2.4. Tree-structured Parzen Estimator (TPE)

Used by **Hyperopt** and **Optuna** (default sampler). Instead of modeling $p(y \mid \lambda)$, TPE splits trials into “good” and “bad” and models **two densities** in hyperparameter space.

After $n$ trials, split at a quantile $\gamma$ (e.g. top 20% = good):

$$\ell(\lambda) = p(\lambda \mid y < y^*), \qquad g(\lambda) = p(\lambda \mid y \geq y^*)$$

Parzen estimators (kernel density / tree-structured mixtures) fit $\ell$ and $g$. The next point maximizes **expected improvement** via the ratio:

$$\lambda_{\text{next}} = \arg\max_\lambda \frac{\ell(\lambda)}{g(\lambda)}$$

High $\ell/g$ → $\lambda$ is more common among good runs than bad runs.

**Pros**

- Scales better than GP-Bayes in tens of dimensions.
- Handles **conditional** hyperparameters (e.g. “momentum only if optimizer = SGD”) via nested search spaces.
- Parallel-friendly variants (e.g. constant-liar in Optuna).

**Cons:** Still sequential in spirit; needs enough trials for $\ell$ and $g$ to be meaningful.

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.8, 0.99)
    units = trial.suggest_int("units", 32, 512)
    # build model, train, return val_loss
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

---

### 2.5. Population-Based Training (PBT)

**Idea.** Train a **population** of $K$ models in parallel, each with its own hyperparameters $\lambda_k$ and weights $\theta_k$. Periodically:

1. **Exploit:** weak performers copy weights from strong performers (same architecture).
2. **Explore:** perturb hyperparameters (e.g. $\alpha \leftarrow \alpha \cdot U(0.8, 1.2)$).

So weights and hyperparameters co-evolve during training — learning-rate schedules can **emerge** without hand design.

**Loop (every $T$ steps)**

- Rank population by online metric (validation loss proxy).
- Bottom fraction: $\theta \leftarrow \theta_{\text{winner}}$, $\lambda \leftarrow \mathrm{perturb}(\lambda_{\text{winner}})$.
- Top fraction: continue with current $(\theta, \lambda)$.

**Pros:** Adapts hyperparameters **during** training; excellent on large GPU clusters.  
**Cons:** Needs $K$ simultaneous full trainings; noisy for metrics that only make sense late (e.g. heavy augmentation).

---

### 2.6. Hyperband / ASHA

**Successive Halving (SH).** Start with $n$ configs, each trained for budget $B$. Keep the top $\lfloor \eta^{-1} \rfloor$ fraction, multiply their budget by $\eta$, repeat until one survivor. Poor configs stop early → save compute.

**Hyperband.** Runs multiple SH **brackets** with different $(n, B_{\max})$ trade-offs so you do not commit to a single early-stop aggressiveness. Hyperparameter $\eta$ (default 3) controls elimination rate.

**ASHA (Asynchronous Successive Halving Algorithm).** Workers promote configs to the next rung **asynchronously** when they finish — no global barrier. Fits distributed training (Ray Tune, PyTorch Lightning, etc.).

**Caveat:** Rankings at epoch 5 may not match epoch 100 (batch size, warmup, architecture). Early stopping assumes **fairly consistent** relative ordering across budgets.

**Pros:** Often 10×+ speedup vs. training every config to completion.  
**Cons:** Can discard late bloomers; less ideal when validation curve is non-monotonic.

---

### 2.7. Summary of search strategies

| Method | Parallelism | Sample efficiency | Best for |
|---|---|---|---|
| Grid search | High | Low | $\leq 2$–3 dims, fine local refinement |
| Random search | High | Medium | Default coarse search; many dims |
| Bayesian Opt (GP) | Low (sequential) | High | Expensive trials, $\lesssim 15$ dims |
| TPE (Optuna/Hyperopt) | Medium | High | General; conditional search spaces |
| Hyperband / ASHA | High | High | Clear early signal of bad configs |
| PBT | High | High | Clusters; adapt $\alpha$ during training |

**Practical stack:** random (or TPE) coarse search → optional fine grid → use Hyperband/ASHA inside the tuner to cap epoch budget.

---

### 2.8. Tricky interview questions about search strategies

**Q: Grid search with 5 values per axis and 4 hyperparameters — how many runs?**  
A: $5^4 = 625$ (unless you use random subsampling of the grid).

**Q: Why does random search explore important hyperparameters better?**  
A: Each trial draws independently per dimension, so you get $\mathcal{O}(N)$ samples along every axis, not $\mathcal{O}(N^{1/d})$ as in a high-dimensional grid.

**Q: When is grid search still reasonable?**  
A: Very few dimensions, categorical knobs with few levels, or a **fine** local refinement after random search.

**Q: Bayesian optimization vs. TPE — main difference?**  
A: Classic BO often uses a GP surrogate over $f(\lambda)$ globally. TPE models $\ell(\lambda)$ vs $g(\lambda)$ for good/bad subsets and picks via $\ell/g$ — scales better and handles conditional spaces.

**Q: What is the main weakness of Hyperband?**  
A: Relies on **early** metrics ranking configs correctly; some hyperparameters only show value after long training.

**Q: Can Optuna trials run in parallel?**  
A: Yes, with multiple workers sharing one study storage; TPE uses strategies like constant-liar to suggest points before prior trials finish.

**Q: Log-scale sampling for learning rate — why not uniform on $[10^{-4}, 1]$?**  
A: Uniform wastes ~90% of trials on $[0.1, 1]$ where sensitivity is often lower than in $[10^{-4}, 0.1]$.

---

## 3. Batch normalization

Batch normalization (BN) normalizes layer inputs **per mini-batch** during training, then applies learnable scale and shift. It stabilizes activations, speeds training, and slightly regularizes — which also makes hyperparameter search less brittle.

### 3.1. Idea

Internal layers see inputs whose distribution shifts as earlier layers update (**internal covariate shift**). BN standardizes activations within each batch to roughly zero mean and unit variance, then lets the network learn the optimal mean/variance per unit via $\gamma$ and $\beta$. Optimization becomes smoother; higher learning rates are often possible.

### 3.2. Batch norm for a single layer

Given pre-activations $z^{(1)}, \ldots, z^{(m)}$ in layer $\ell$ on a mini-batch of size $m$:

$$\mu = \frac{1}{m}\sum_{i=1}^{m} z^{(i)}$$

$$\sigma^2 = \frac{1}{m}\sum_{i=1}^{m} \bigl(z^{(i)} - \mu\bigr)^2$$

$$z^{(i)}_{\text{norm}} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

$$\tilde{z}^{(i)} = \gamma\, z^{(i)}_{\text{norm}} + \beta$$

- $\gamma$, $\beta$ are **learnable** (one per feature / hidden unit), updated like $W$.
- Setting $\gamma = \sqrt{\sigma^2 + \varepsilon}$ and $\beta = \mu$ recovers the identity map — BN can always “turn off” if needed.
- $\varepsilon$ (e.g. $10^{-8}$) prevents division by zero.

**Naming trap:** BN’s $\beta$ is **not** momentum/Adam’s $\beta$.

### 3.3. Integration into a deep network

BN is usually applied **before** the nonlinearity (pre-activation BN) or after linear transform — frameworks differ; the principle is the same.

Because BN centers $z^{[\ell]}$, the additive bias $b^{[\ell]}$ is redundant and is often omitted; $\beta^{[\ell]}$ plays the bias role.

**Forward pass (layer $\ell$, mini-batch $\{t\}$):**

$$Z^{[\ell]} = W^{[\ell]} A^{[\ell-1]}$$

$$Z^{[\ell]}_{\text{norm}} = \frac{Z^{[\ell]} - \mu^{[\ell]}}{\sqrt{(\sigma^{[\ell]})^2 + \varepsilon}}$$

$$\tilde{Z}^{[\ell]} = \gamma^{[\ell]} \odot Z^{[\ell]}_{\text{norm}} + \beta^{[\ell]}$$

$$A^{[\ell]} = g^{[\ell]}(\tilde{Z}^{[\ell]})$$

**Parameters per layer:** $W^{[\ell]}, \gamma^{[\ell]}, \beta^{[\ell]}$ (no separate $b^{[\ell]}$ when BN includes $\beta$).

### 3.4. Batch norm at test time

At inference you may have batch size 1 — batch statistics are undefined. Use **running averages** accumulated during training:

$$\mu_{\text{run}} \leftarrow \rho\, \mu_{\text{run}} + (1-\rho)\, \mu_{\text{batch}}$$

$$\sigma^2_{\text{run}} \leftarrow \rho\, \sigma^2_{\text{run}} + (1-\rho)\, \sigma^2_{\text{batch}}$$

At test time, normalize with $\mu_{\text{run}}$ and $\sigma^2_{\text{run}}$. Frameworks (PyTorch `model.eval()`, TensorFlow inference mode) switch automatically.

### 3.5. Pros of batch normalization

| Benefit | Mechanism |
|---|---|
| Faster, stabler training | Smoother loss landscape; less sensitivity to $\alpha$ and init |
| Reduces internal covariate shift | Later layers see more stable input distributions |
| Mild regularization | Batch $\mu$, $\sigma^2$ are noisy estimates → stochastic perturbation (like light dropout) |
| Enables higher learning rates | Activations stay in a reasonable range |
| Easier hyperparameter tuning | Less brittle to $\alpha$ and initialization choices |

**Caveats**

- Large batch size → less noise → **weaker** BN regularization.
- Do not treat BN as a substitute for dropout/L2 when strong regularization is needed.
- Layer norm / group norm are preferred when batch statistics are unreliable (small $B$, RNNs, some NLP/vision transformers).

### 3.6. Tricky interview questions about batch normalization

**Q: What statistics does BN use at train vs test time?**  
A: Train: batch mean/variance. Test: exponential moving average of batch stats from training.

**Q: Why can you drop the bias $b$ when using BN?**  
A: Subtracting $\mu$ removes the effect of a constant offset; learnable $\beta$ provides the shift.

**Q: Is BN’s $\beta$ the same as Adam’s $\beta_1$?**  
A: No — completely different symbols. BN $\beta$ is a per-feature shift; Adam $\beta_1$ is the gradient EMA decay.

**Q: Does BN always improve generalization?**  
A: Not guaranteed; it mainly helps optimization. The regularization effect is mild and batch-size dependent.

**Q: Why does large batch size interact with BN?**  
A: Batch statistics have lower variance → less stochastic regularization from BN.

**Q: Batch norm vs layer norm?**  
A: BN normalizes across the batch dimension (per channel); Layer norm normalizes across features for each example — better when batch size is small or sequences vary.

**Q: Where should $\varepsilon$ appear in the denominator?**  
A: Inside the square root: $\sqrt{\sigma^2 + \varepsilon}$ for numerical stability.

---

## 4. Practical checklist

1. **Split data correctly:** train / dev / test; tune only on dev.
2. **Prioritize $\alpha$**, then $B$, width, depth; fix Adam $\beta_2=0.999$, $\varepsilon=10^{-8}$ unless needed.
3. **Start with random search** (or Optuna TPE) over the important knobs — not full grid search in high dimensions.
4. **Log-scale sample** $\alpha$, weight decay $\lambda$, and momentum-like $\beta$ near 1; **linear/integer** sample for units and layer counts in a sane range.
5. **Coarse-to-fine:** wide random exploration → zoom into the best region → optional small grid.
6. **Enable batch norm** by default in feed-forward CNNs/MLPs when appropriate; remember `train()` vs `eval()` modes.
7. **Use early-stopping schedulers** (Hyperband/ASHA) when trials are expensive; set a wall-clock or trial budget.
8. **Parallelize when possible** (Caviar): many short pilots beat one long guess.
9. **Re-tune** when data, objective, or hardware changes materially.
10. **Document** the search space and best dev run; run the **test set once** for the final report.
