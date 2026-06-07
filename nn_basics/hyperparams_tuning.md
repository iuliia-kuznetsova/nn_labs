# Hyperparameter Tuning

A practical reference for tuning neural networks: 
- Hyperparameters;
- Basic Hyperoptimization Algorithms (Grid Search, Random Search);
- Advanced Hyperoptimization Algorithms (Bayesian Optimization, Tree-structured Parzen Estimator (TPE), Hyperband / ASHA, Population-Based Training (PBT));
- Batch Normalization;
 and interview-style pitfalls.

---

## 1. Hyperparameters

**Parameters** ($W^{[\ell]}$, $b^{[\ell]}$, and learnable batch-norm $\gamma$, $\beta$) are updated by gradient descent during training.

**Hyperparameters** are everything you choose *before* training that controls *how* learning happens. They determine the final learned parameters — hence “hyper.”

### 1.1. Hyperparameters

**Abbreviations used below**

- **EMA** (exponentially weighted average): a running average that gives more weight to recent values and less to older ones — e.g. $v_t = \beta v_{t-1} + (1-\beta) x_t$. Used in momentum, Adam, and batch-norm running statistics.
- **BN** (batch normalization): normalizes each layer’s activations using mini-batch mean and variance during training, then applies learnable scale and shift. Stabilizes training and makes tuning easier — see Section 3.


| Hyperparameter                      | Symbol                   | Effect                                                                                                                                                                                      |
| ----------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Learning rate                       | $\alpha$                 | How large each gradient step is. Too high → loss spikes or divergence; too low → very slow progress.                                                                                        |
| Momentum / Adam $\beta_1$           | $\beta_1$                | EMA decay for **gradients**: blends past gradients with the current one to dampen oscillation and speed up convergence along consistent directions.                                         |
| Mini-batch size                     | $B$                      | How many examples per gradient update. Larger $B$ → faster, lower-noise gradients but often worse generalization; smaller $B$ → noisier updates (mild regularization) and lower throughput. |
| Hidden units per layer              | $n^{[\ell]}$             | Layer **width** — how many neurons per layer. More units → higher capacity and more risk of overfitting; fewer → faster, simpler models.                                                    |
| Number of layers                    | $L$                      | Network **depth**. More layers → richer hierarchical features; also harder optimization, more compute, and longer training.                                                                 |
| Learning rate schedule / decay      | —                        | How $\alpha$ changes over training (warmup, step decay, cosine, …). Often start higher for fast learning, then reduce for stable fine-tuning near the minimum.                              |
| Weight decay                        | $\lambda$                | Strength of the L2 penalty on weights. Larger $\lambda$ → smaller weights, stronger regularization, less overfitting. Usually tuned on a log scale alongside $\alpha$.                      |
| Adam $\beta_2$, $\varepsilon$       | $\beta_2$, $\varepsilon$ | $\beta_2$: EMA decay for **squared** gradients (controls how fast Adam adapts per-parameter step sizes). $\varepsilon$: tiny constant in denominators to prevent division by zero.          |
| Activation, initialization, dropout | —                        | Nonlinearity type (ReLU, …), starting weight scale, and dropout probability. Together they shape whether the network trains reliably, how fast it learns, and how much it regularizes.      |


**Standard workflow**

1. Pick **initial hyperparameters** (literature, defaults, or a quick pilot run).
2. Train on train set and evaluate on a **validation (development) set** — never tune on the test set.
3. Adjust and repeat until dev performance is satisfactory; **run the test set once** at the end.

Optimal values are problem- and dataset-specific; there is rarely a universal “best” setting.

### 1.2. Priorities

Not all hyperparameters matter equally. Allocate most of your search budget to the high-impact ones.


| Priority       | Hyperparameter                             | Typical default / note                                                         |
| -------------- | ------------------------------------------ | ------------------------------------------------------------------------------ |
| Most important | Learning rate $\alpha$                     | Always tune; often on a **log scale**                                          |
| Second tier    | Momentum $\beta_1$ (SGD) or Adam $\beta_1$ | 0.9 is a strong default                                                        |
| Second tier    | Mini-batch size $B$                        | 64–512; affects speed, gradient noise, and batch normalization (BN) statistics |
| Second tier    | Hidden units per layer                     | Tune width before obsessing over depth                                         |
| Third tier     | Number of layers $L$                       | Can dominate capacity but costly to search                                     |
| Third tier     | Learning rate decay / schedule             | Warmup, cosine, step decay                                                     |
| Third tier     | Weight decay $\lambda$                     | Often log-scale; coupled with $\alpha$                                         |
| Rarely tuned   | Adam $\beta_2$                             | 0.999                                                                          |
| Rarely tuned   | Adam $\varepsilon$                         | $10^{-8}$                                                                      |


**Rule of thumb:** tune $\alpha$ first (and optionally $\beta_1$, $B$, width). Fix Adam’s $\beta_2$ and $\varepsilon$ unless you have strong evidence they matter for your task.

**Panda vs. Caviar** (organizing the search process)


| Approach   | When                           | How                                                            |
| ---------- | ------------------------------ | -------------------------------------------------------------- |
| **Panda**  | Limited compute, large dataset | Train one model; monitor daily; manually nudge hyperparameters |
| **Caviar** | Ample parallel compute         | Launch many configs at once; pick the best dev result          |


Re-tune periodically (every few months on long-lived systems). Data drift, new data volume, and faster hardware can make old settings suboptimal.

### 1.3. Tricky interview questions

**Q: What is the difference between a parameter and a hyperparameter?**  
A: Parameters are learned from data via gradients ($W$, $b$, …). Hyperparameters are set by the practitioner and define the training procedure (learning rate, architecture depth, batch size, …).

**Q: Why is the learning rate usually the #1 hyperparameter to tune?**  
A: It directly controls step size in parameter space. A wrong $\alpha$ by one order of magnitude often makes training diverge or stall, regardless of other choices.

**Q: Should you tune on the training set or the test set?**  
A: Neither for final selection — use a **validation (dev) set**. The test set must remain untouched until the very end to estimate generalization without optimistic bias.

**Q: If $\beta_2 = 0.999$ works everywhere, why mention it at all?**  
A: In practice you almost never tune $\beta_2$ — $0.999$ is a safe default for Adam. But you should still know what it does: $\beta_2$ sets how fast Adam’s **EMA of squared gradients** forgets old values. Roughly, $\beta_2 = 0.999$ averages over on the order of $\frac{1}{1-\beta_2} \approx 1000$ recent steps. If you moved $\beta_2$ from $0.99$ to $0.999$, Adam would react much more slowly to changes in gradient magnitude — per-parameter step sizes would adapt sluggishly. The same “near 1 is very sensitive” intuition applies to momentum $\beta_1$. Interviewers ask about $\beta_2$ to check you understand Adam’s mechanics, not because you should grid-search it.

**Q: Does a larger batch size always mean better training?**  
A: No — larger batches help **efficiency**, not always **final quality**. A bigger $B$ means each gradient is averaged over more examples, so updates are less noisy and GPUs are better utilized (more examples per forward/backward pass). But that lower noise can push optimization toward **sharper** minima that fit the training distribution tightly and generalize worse on new data. Small batches inject gradient noise that sometimes acts like mild regularization. Batch size also interacts with **batch normalization (BN)**: BN uses batch mean/variance as a noise source for regularization — with a very large $B$, those statistics are almost deterministic, so that extra regularization weakens. The right $B$ balances hardware speed, stable optimization, and generalization (often 64–512 in practice).

**Q: Can you “overfit” the validation set?**  
A: Yes. The validation set is meant for **one-off** comparison of a few candidate models. If you train 200 models and always keep the one with the best dev score, you are effectively **searching** on the dev set — some configs will look good by random luck on those specific examples, not because they generalize better. Your reported dev score becomes optimistically biased, just like overfitting on training data. Fixes: (1) split data into **train / dev / test**, and touch the test set only once at the very end; (2) use a **second holdout** (dev for tuning, test for final evaluation); (3) limit how many manual tuning rounds you run; (4) when comparing many trials, treat small dev-score differences as noise unless they are consistent across seeds or folds.

---

## 2. Search strategies

Hyperparameter search treats validation loss (or another metric) as a black-box function $f(\lambda)$ where $\lambda$ is the hyperparameter vector. The goal is to find $\lambda^* \approx \arg\min_\lambda f(\lambda)$ within a budget of $N$ training runs.

### 2.1. Basic Hyperopt Methods

#### 2.1.1. **Grid search**


**Core idea**

Try **every combination** of a fixed set of values for each hyperparameter. You pick a list of candidates per knob and train one model for every possible tuple — nothing is skipped.

The total number of trials is the product of all list sizes. With $k$ values per hyperparameter and $d$ hyperparameters: $k^d$ trials — **exponential** growth as you add more knobs.


**Step-by-step workflow**

1. **Define candidates** for each hyperparameter — e.g. lr ∈ {1e-4, 1e-3, 1e-2}, batch size ∈ {32, 64, 128}.
2. **Form the grid** — every combination: (1e-4, 32), (1e-4, 64), (1e-4, 128), (1e-3, 32), …
3. **Train one model** per grid point; evaluate each on the validation set.
4. **Pick** the combination with the best validation score.


**Examples**

**Example (2 hyperparameters).** Grid search tries **every combination**:


|                    | $B = 32$ | $B = 64$ | $B = 128$ |
| ------------------ | -------- | -------- | --------- |
| $\alpha = 10^{-4}$ | run 1    | run 2    | run 3     |
| $\alpha = 10^{-3}$ | run 4    | run 5    | run 6     |
| $\alpha = 10^{-2}$ | run 7    | run 8    | run 9     |


3 learning rates × 3 batch sizes = **9** training runs. You never skip a pair — even if batch size barely matters, you still train all 9 models.

**Example (5 hyperparameters).** Suppose you grid-search 5 knobs (e.g. $\alpha$, $B$, hidden units, layers, weight decay) with **5 values each**. Total runs = $5 \times 5 \times 5 \times 5 \times 5 = 5^5 = \mathbf{3125}$. Adding one more hyperparameter with 5 values multiplies the cost by 5 again ($5^6 = 15{,}625$) — that is why grid search explodes as $d$ grows.


**Pros**

- Simple, reproducible, embarrassingly parallel.
- Fine coverage when **few** dimensions matter and ranges are small.

**Cons**

- Wastes budget when some dimensions barely affect $f$ (you repeat the same combo with many values of an irrelevant hyperparameter).
- Curse of dimensionality: impractical beyond ~2–3 important continuous hyperparameters.


**When to use**

- Very low dimension ($d \leq 2$).
- Categorical choices with few levels (e.g. $L \in 2,3,4$).
- As a **fine** phase after random search has located a promising region (small local grid).



**Usage example**

```python
from itertools import product

lrs = [1e-4, 1e-3, 1e-2]
batch_sizes = [32, 64, 128]

best_loss, best_params = float("inf"), {}
for lr, bs in product(lrs, batch_sizes):        # 3 × 3 = 9 runs
    loss = train_and_evaluate(lr=lr, batch_size=bs)
    if loss < best_loss:
        best_loss, best_params = loss, {"lr": lr, "batch_size": bs}
```

---

#### 2.1.2. **Random search**

**Core idea**

Pick each hyperparameter independently at random for every trial — no fixed grid, no memory of past results. Each trial samples a fresh configuration from your defined distributions, so every run explores a different corner of the space.

The key insight (Bergstra & Bengio, 2012): if only one hyperparameter truly matters, random search effectively allocates all N trials to exploring it. A same-budget grid search spread across d hyperparameters gives only ~N^(1/d) distinct values per axis. With 5 dims and 100 runs that is barely 2-3 values for the most important knob.


**Examples**

**Concrete 2D example (N = 25 runs)**

| | Grid (5x5) | Random (25 pts) |
|---|---|---|
| Distinct lr values | **5** | **~25** |
| Distinct $\varepsilon$ values | **5** | **~25** |

If lr dominates and $\varepsilon$ is irrelevant, random search explores lr **5× more thoroughly** for the same cost.


**How it differs**

Random search vs Grid search: No fixed candidate lists; each trial is independent; explores important axes far more densely when others are irrelevant.
Random search  vs Bayesian Opt / TPE: No memory — does not learn from past trials; fully parallel but may revisit bad regions.
Random search  vs Hyperband / ASHA: Does not early-stop bad runs; each trial trains to completion.


**Step-by-step workflow**

1. **Define distributions** per hyperparameter:
   - Log-uniform for lr, weight decay (multiplicative effects, span decades)
   - Linear integers for hidden units, layers (effect is roughly uniform)
   - Categorical for optimizer type, activation
2. **Set a trial budget** N (e.g. 50 runs).
3. **Sample** a new independent config for each trial.
4. **Train and evaluate** each config on the validation set.
5. **Coarse-to-fine (optional):** after the sweep, identify the best region; narrow ranges and run a second random sweep — or a small local grid — inside that region.
6. **Return** the config with the lowest validation loss.


**Sampling scale cheatsheet**

| Hyperparameter | Scale | Why |
|---|---|---|
| Learning rate, weight decay | **Log** | Spans decades; ×10 matters, not +0.01. Uniform on [1e-4, 1] puts 90% of draws in [0.1, 1]. |
| Momentum $\beta$ near 1 | **Log on $1-\beta$** | Memory window $\approx 1/(1-\beta)$; $\beta: 0.999 \to 0.9995$ doubles window from 1000 to 2000 steps |
| Hidden units, layers | **Linear integer** | Effect is roughly uniform over a reasonable range |
| Optimizer, activation | **Categorical** | Discrete choices; no ordering |

**Log scale for $\alpha$, $\lambda$ (weight decay)**

Sample the exponent, then convert: for $\alpha \in [10^{-4}, 10^0]$:

```python
r     = np.random.uniform(-4, 0)  # exponent
alpha = 10 ** r                   # alpha uniform on log scale
```

**Log scale for momentum $\beta$ near 1**

Sample $1-\beta$ on log scale so each decade of memory window gets equal budget:

```python
r    = np.random.uniform(-3, -1)   # log10(1 - beta)
beta = 1 - 10 ** r                 # beta in [0.9, 0.999]
```

$\beta: 0.9 \to 0.9005$ is negligible (~10 steps window). $\beta: 0.999 \to 0.9995$ doubles it (1000 → 2000 steps).


**Pros**

- Fully parallel — all trials are independent.
- Explores the important axis far more than grid search for the same budget.
- No setup beyond choosing distributions; trivial to implement.
- Coarse-to-fine naturally extends it without extra tooling.

**Cons**

- No memory — does not learn from past results; may waste trials near regions that already failed.
- Does not early-stop bad runs.
- Needs reasonable prior on ranges — if the lr range is completely wrong, more sampling will not help.


**Usage example — scikit-learn (classical ML)**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    "learning_rate": loguniform(1e-4, 1e-1),
    "max_depth":     randint(3, 12),
    "subsample":     loguniform(0.5, 1.0),
}

search = RandomizedSearchCV(model, param_dist, n_iter=50, cv=3, scoring="neg_log_loss")
search.fit(X_train, y_train)
print(search.best_params_)
```

**Coarse-to-fine search**

1. **Coarse:** sample broadly over the full plausible range (random, log-scaled where needed).
2. **Identify** a promising region — the cluster of configs with the lowest dev loss.
3. **Fine:** narrow the ranges to that region and run a second random sweep (or small grid) inside it.

Repeat until diminishing returns. This concentrates compute without hand-designing a single narrow range upfront.


**Typical use** 

Default coarse exploration for any model — fast, parallel, no dependencies. Always run random search before committing to BO or TPE; apply coarse-to-fine if the budget allows a second pass.


**Libraries** 

numpy.random, scipy.stats, sklearn.model_selection.RandomizedSearchCV, optuna.samplers.RandomSampler, Ray Tune tune.uniform / tune.loguniform.

---

### 2.2. Advanced Hyperopt Methods

There are four **advanced hyperparameter optimization (HPO)** methods beyond grid/random search. All treat validation loss as a **black-box** function $f(\lambda)$. Each trial is expensive, and you want the best $\lambda$ within a fixed budget. Think of them as four different **schedulers for expensive experiments**, trading off **sample efficiency**, **parallelism**, and **implementation complexity**:

| Family | Methods | Core question it answers |
|---|---|---|
| **Model-based** | Bayesian Optimization, TPE | Given past trials, where should we try next? |
| **Multi-fidelity** | Hyperband, ASHA | Can we kill bad configs early and spend budget on survivors? |
| **Online / evolutionary** | PBT | Should hyperparameters stay fixed, or adapt during training? |

**One-line summaries**

- **Bayesian Optimization (GP / SMAC-style):** Fit a probabilistic surrogate of $f(\lambda)$; use an acquisition function to pick the next point where promising and uncertain balance best very sample-efficient when each full training is costly.
- **TPE:** A scalable BO variant that models $p(\lambda \mid y)$ (where configs live given their loss) instead of $p(y \mid \lambda)$; splits trials into good/bad groups and samples where good density outweighs bad strong for high-dimensional and conditional spaces (Hyperopt, Optuna).
- **PBT:** A population of models trains in parallel; underperformers copy weights and hyperparameters from winners, then perturb them online search over **schedules** (LR, augmentation), not just static values.
- **Hyperband / ASHA:** HPO as a bandit over training budget (epochs, steps, data fraction); Successive Halving promotes only promising configs to longer runs; ASHA does this asynchronously for massive parallelism.

---

**How the four methods differ (at a glance)**

| | **Bayesian Optimization** | **TPE** | **Hyperband / ASHA** | **PBT** |
|---|---|---|---|---|
| **Learns from** | Past trial results → surrogate model | Past trial results in good vs bad regions | Partial training curves → who to promote | Live training progress → who to copy |
| **When it decides** | Between full (or fixed-budget) trials | Between trials | During training (early stop / promote) | Continuously during training |
| **Picks** | Next static config | Next static config | Which configs deserve more epochs | Evolving weights + hyperparameter schedules |
| **Parallelism** | Low–medium | Medium–high | Very high | High (population) |
| **Best budget type** | Few expensive full runs | Few–many full runs | Many cheap partial runs | Long single jobs |

---

#### 2.2.1. Bayesian Optimization

**Core idea** 

Learn from every trial. After each training run, fit a lightweight **surrogate model** that guesses validation loss at hyperparameter settings you have not tried yet, and flags where it is **uncertain**. The next trial is chosen to balance:

- **Exploitation** — go where loss looks low already.
- **Exploration** — go where the map is blank and surprises are possible.


**How it differs**

Bayesian optimization vs Random search: Remembers history; next trial is informed, not independent.

Bayesian optimization vs TPE: Models the **whole** loss landscape with a smooth surrogate (GP or trees); more principled uncertainty, heavier in high dimensions.

Bayesian optimization vs Hyperband/ASHA: Chooses **what** to try next; does not early-stop bad runs mid-training.

Bayesian optimization vs PBT: Searches **static** hyperparameters; one fixed config per trial.


**Step-by-step workflow**

1. **Define search space** — ranges for lr, depth, regularization, etc.
2. **Bootstrap** — run 5–10 random configs; record validation loss for each.
3. **Fit surrogate** — train GP or tree model on `(hyperparameters → val loss)`.
4. **Propose next config** — pick the point that scores best on an acquisition rule (e.g. Expected Improvement: “likely to beat our best so far”; UCB: “low predicted loss + bonus for uncertainty”).
5. **Train** that config to completion (fixed budget per trial).
6. **Append** result to history; repeat steps 3–6 until trial budget is spent.
7. **Return** the config with the best validation loss seen.


**Pros**

- Very **sample-efficient** when each full training is expensive (hours+).
- Principled **explore/exploit** trade-off.
- Tree-based variants (SMAC) handle discrete and conditional params better than vanilla GP.

**Cons**

- Trials are mostly **sequential** — hard to use 100 GPUs at once without batch extensions.
- GP struggles with **many dimensions** and messy conditional spaces.
- **Fixed budget per config** — no built-in early stopping (pair with Hyperband via BOHB if needed).
- Surrogate fitting slows as history grows.


**Usage example — `scikit-optimize` (GP Bayes)**

```python
from skopt import gp_minimize
from skopt.space import Real, Integer

space = [
    Real(1e-5, 1e-1, prior="log-uniform", name="lr"),
    Integer(32, 512, name="units"),
    Real(1e-6, 1e-2, prior="log-uniform", name="weight_decay"),
]

def objective(params):
    lr, units, weight_decay = params
    val_loss = train_and_evaluate(lr=lr, units=units, weight_decay=weight_decay)
    return val_loss  # minimize

result = gp_minimize(objective, space, n_calls=50, n_initial_points=10)
print("Best params:", result.x)
```

**Typical use** 

CatBoost / XGBoost tuning, medium DL experiments, 20–100 full trainings, limited parallelism.


**Libraries** 

`scikit-optimize`, `Ax`, `BoTorch`, `SMAC`.

---

#### 2.2.2. Tree-structured Parzen Estimator (TPE)

**Core idea** 

Instead of modeling “loss as a function of hyperparameters” globally, ask: **where did good runs cluster vs bad runs?** Split past trials into top performers and the rest; learn two distributions over hyperparameter space; sample next from regions that appear often in **good** runs and rarely in **bad** ones.


**How it differs**

Tree-structured Parzen Estimator vs GP Bayesian Optimization: Lighter surrogate; scales to more dims and mixed/conditional spaces; default in Optuna/Hyperopt.

Tree-structured Parzen Estimator vs Random search: Biased toward regions that already produced low loss.

Tree-structured Parzen Estimator vs Hyperband/ASHA: Picks which hyperparameters to try; does not allocate epoch budget.

Tree-structured Parzen Estimator vs PBT: Static config per trial; no mid-training adaptation.


**Step-by-step workflow**

1. **Define search space** in Optuna/Hyperopt (including conditional rules, e.g. momentum only if optimizer = SGD).
2. **Run trials** — each returns a validation loss.
3. **Label trials** — e.g. bottom 20% loss = “good,” rest = “bad.”
4. **Fit two density models** — where hyperparameters land in good vs bad groups.
5. **Sample next config** from areas with high good/bad density ratio.
6. Repeat until `n_trials` exhausted; return `study.best_params`.


**Example — what TPE learns from history**

| Trial | lr | val loss | Label |
|---|---|---|---|
| 1 | 1e-2 | 0.45 | bad |
| 2 | 3e-3 | 0.31 | good |
| 3 | 1e-4 | 0.52 | bad |
| 4 | 5e-3 | 0.29 | good |

After many rows, TPE skews new suggestions toward lr around $10^{-3}$–$10^{-2}$.


**Pros**

- Default choice for **general HPO** — handles continuous, discrete, and conditional params.
- **Parallel-friendly** (multiple Optuna workers, Ray Tune).
- Easy to combine with early stopping (TPE suggests configs, ASHA prunes bad ones).

**Cons**

- First ~10 trials are effectively random (not enough history).
- Surrogate update cost grows with trial count.
- Each trial usually gets a **fixed training budget** unless you add a pruner.


**Usage example — Optuna (TPE + optional ASHA pruner)**

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    if optimizer == "sgd":
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
    units = trial.suggest_int("units", 32, 512)

    for epoch in range(100):
        val_loss = train_one_epoch(lr, optimizer, units, ...)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()  # ASHA-style early stop
    return val_loss

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner(),
)
study.optimize(objective, n_trials=100, n_jobs=4)  # 4 parallel workers
print(study.best_params)
```

**Typical use** 

First choice for tabular ML and DL when using Optuna or Ray Tune; structured search spaces with dependencies.


**Libraries** 

`Optuna`, `Hyperopt`, `Ray Tune`.

---

### 2.2.4. Population-Based Training (PBT)

**Core idea** 

Do not freeze hyperparameters for the whole run. Train **K models in parallel** (a population). Every few thousand steps, **underperformers copy weights and hyperparameters from winners**, then **mutate** hyperparameters slightly (e.g. lr × random factor in [0.8, 1.2]). Good settings spread; bad ones are replaced — **schedules emerge** without hand design.


**How it differs**

Population-Based Training vs BO / TPE: Optimizes **during** training, not between isolated trials; searches **schedules**, not one static vector.

Population-Based Training vs Hyperband/ASHA: Does not early-stop trials — all population members keep training; weak ones are **reborn** from strong ones.

Population-Based Training vs fixed LR decay: No manual cosine/step schedule — decay can **emerge** from exploit + explore.



**Step-by-step workflow**

1. **Launch population** — e.g. K = 20 workers, each with random initial hyperparameters (lr, weight decay, augmentation strength, …).
2. **Train** all members in parallel for interval T (e.g. 1 000 steps).
3. **Evaluate** — quick validation metric per member.
4. **Rank** population by metric.
5. **Exploit** — bottom ~20%: copy weights + hyperparameters from a random top performer.
6. **Explore** — same bottom members: perturb copied hyperparameters (multiplicative noise).
7. **Continue** training from new checkpoints; top performers unchanged.
8. Repeat steps 2–7 until training budget ends; keep best final checkpoint.

| Action | Who | What happens |
|---|---|---|
| **Exploit** | Bottom ~20% | Copy weights and hyperparameters from a winner |
| **Explore** | Same members | Perturb hyperparameters (e.g. lr × U(0.8, 1.2)) |
| **Hold** | Top performers | Keep training unchanged 


**Pros**

- Discovers **dynamic schedules** (lr, augmentation, dropout over time).
- Adapts to **non-stationary** objectives (RL, long CV/NLP training).
- Excellent when you already run **large parallel training** jobs.

**Cons**

- Needs **K simultaneous trainings** and reliable checkpoint copy.
- Population size and perturbation rates need tuning.
- Ranking noise early in training can cause bad copies.
- Poor fit for **short** or **one-shot** evaluations (CatBoost fit, single-epoch probes).


**Usage example — Ray Tune PBT (conceptual)**

```python
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="val_loss",
    mode="min",
    perturbation_interval=10,           # rebalance every 10 epochs
    hyperparam_mutations={
        "lr": tune.uniform(0.8, 1.2),   # multiply current lr
        "weight_decay": tune.uniform(0.8, 1.2),
    },
)

tune.run(
    train_fn,
    config={"lr": tune.loguniform(1e-5, 1e-2), "weight_decay": tune.loguniform(1e-6, 1e-3)},
    num_samples=20,                     # population size
    scheduler=pbt,
)
```

**Typical use** 

Long RL or large-scale DL where lr / augmentation schedules matter and checkpoint recycling is natural.


**Libraries** 

`Ray Tune` (native PBT), some internal DL platform schedulers.

---

#### 2.2.4. Hyperband / ASHA

**Core idea** 

Most configs are bad — do not train them to completion. Run **many trials cheaply** (few epochs), **drop** the worst, give survivors **more epochs**, repeat until budget is spent. **Hyperband** runs several tournament “brackets” with different aggressiveness; **ASHA** promotes or kills configs **as soon as** partial results arrive — no waiting for slow workers.


**How it differs**

Hyperband / ASHA vs BO / TPE: Does not model the loss surface — mostly **random configs** + **budget allocation**; wins on wall-clock, not trials-to-best.

Hyperband / ASHA vs PBT: **Stops** bad trials instead of recycling them; searches static configs, not evolving schedules.

Hyperband vs ASHA: Hyperband = multiple synchronized bracket strategies; ASHA = async promotion, better for heterogeneous clusters.


**Step-by-step workflow (one Successive Halving bracket)**

1. **Sample** n configs (e.g. 27 random hyperparameter sets).
2. **Rung 0** — train each for B epochs (e.g. 5); record validation loss.
3. **Cut** — keep top 1/3 (if reduction factor η = 3); stop the rest.
4. **Rung 1** — survivors train η×B more epochs (15 total); rank again; keep top 1/3.
5. **Repeat** until one config remains or max budget reached.
6. **Hyperband** — repeat the whole bracket scheme with different (n, max_epochs) trade-offs.
7. **ASHA** — same logic, but promote/kill as workers finish instead of waiting for the full rung.

**Walkthrough** (27 configs, 5 epochs initially, keep top third each round):

| Round | Configs alive | Epochs this round | What happens |
|---|---|---|---|
| 0 | 27 | 5 each | 18 configs eliminated |
| 1 | 9 | 15 each | 6 eliminated |
| 2 | 3 | 45 each | 2 eliminated |
| 3 | 1 | 135 total | winner fully trained |

24 configs never reached full length — large savings when bad runs fail fast.


**Pros**

- Best **anytime** performance when you have many workers and cheap partial training.
- Trivial to scale — each worker is an independent trial.
- Pairs well with TPE (smart suggestions) or random search (simple baseline).

**Cons**

- Assumes **early loss predicts late loss** — “late bloomers” get killed.
- Noisy epoch metrics can mis-rank configs.
- Less sample-efficient than BO when every partial run is still very expensive.


**Usage example — Optuna ASHA pruner**

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    for epoch in range(50):
        val_loss = train_one_epoch(lr, batch_size, ...)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return val_loss

study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.SuccessiveHalvingPruner(),  # or HyperbandPruner
)
study.optimize(objective, n_trials=50, n_jobs=8)
```

**Usage example — Ray Tune ASHA**

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

asha = ASHAScheduler(metric="val_loss", mode="min", max_t=100, grace_period=5)

tune.run(
    train_fn,
    config={"lr": tune.loguniform(1e-5, 1e-1), "layers": tune.randint(2, 6)},
    num_samples=50,
    scheduler=asha,
    resources_per_trial={"gpu": 1},
)
```

**Typical use** 

Multi-epoch DL (CNNs, transformers, time-series forecasters) on clusters; combine with TPE sampler for smarter config proposals.


**Libraries** 

`Optuna` (pruners), `Ray Tune` (ASHA/Hyperband), PyTorch Lightning (optional ASHA integration).

---

### 2.3. Summary and usage

#### Quick comparison (all search methods)

| Method | Parallelism | Sample efficiency | Best for |
|---|---|---|---|
| Grid search | High | Low | $\leq 2$–3 dims; fine local refinement |
| Random search | High | Medium | Default coarse search; many dims |
| Bayesian Opt (GP) | Low (sequential) | High | Expensive trials; $\lesssim 15$ dims |
| TPE (Optuna/Hyperopt) | Medium–high | High | General HPO; conditional spaces |
| Hyperband / ASHA | High | High (wall-clock) | Early-stop signal; large clusters |
| PBT | High | High (online) | Long DL jobs; dynamic schedules |

#### Side-by-side: the four advanced methods

**Core idea and mechanics**

| Method | Underlying idea | How configs are chosen | Resource allocation |
|---|---|---|---|
| **Bayesian Optimization** | Model-based global optimization of black-box $f(\lambda)$ | Optimize acquisition function over surrogate (GP, trees, …) | Fixed budget per config; no built-in early stopping |
| **TPE** | Density-estimation flavored BO; models $p(\lambda \mid y)$ | Sample from KDEs favoring high good/bad density ratio | Fixed budget per eval; combinable with multi-fidelity |
| **Hyperband** | Bandit + early stopping over (mostly) random configs | Random or prior-guided sampling per bracket | Aggressive multi-fidelity via Successive Halving |
| **ASHA** | Async Successive Halving | Same as Hyperband; often random or model-guided | Promote/prune without sync; highly scalable |
| **PBT** | Evolutionary / Lamarckian online search | Exploit (copy best) + explore (perturb $\lambda$) | Long-running population; periodic weight replacement |

**Strengths and weaknesses**

| Method | Strengths | Weaknesses / gotchas |
|---|---|---|
| **Bayesian Optimization** | Very sample-efficient for expensive evals; principled uncertainty | GP-BO weak in high-$d$ and many discrete/conditional params; synchronous; poor at massive parallelism |
| **TPE** | High-$d$, mixed, conditional spaces; easy to parallelize; widely implemented | Model-fit cost grows with history; needs enough trials for stable KDEs |
| **Hyperband / ASHA** | Great anytime wall-clock; cheap partial runs + early stop; easy to parallelize | Mostly random; noisy early metrics; less sample-efficient than BO when each full eval is extremely costly |
| **PBT** | Learns **schedules**, not just static $\lambda$; adapts to non-stationary training | Needs checkpoint infrastructure; population tuning; poor for short/non-iterative evals |

**Typical use cases**

| Method | Good when | Common domains |
|---|---|---|
| **BO / TPE** | One evaluation is very expensive; budget is 20–100 full trainings; modest search space | XGBoost, CatBoost, SVM; medium-sized DL |
| **Hyperband / ASHA** | Multi-epoch training; cheap-to-expensive budget ladder; many workers | Deep nets, transformers, large datasets |
| **PBT** | Long incremental training; dynamic schedules matter | RL, large CV/NLP; production DL at scale |

#### How to choose in practice

| Your situation | Prefer |
|---|---|
| **Tabular / classical ML** (CatBoost, XGBoost, risk models) | **TPE** or generic **BO** first — sample-efficient, structured spaces |
| Cheap trains + many CPU cores | **TPE/BO + early stopping**, or **BOHB** (BO + Hyperband) via Optuna / Ray Tune |
| **Deep learning** — static hyperparameters, multi-epoch training | **ASHA / Hyperband** (Ray Tune, Optuna, Lightning) |
| **Deep learning** — LR / augmentation / dropout **schedules** | **PBT** |
| Limited parallelism, very expensive runs | **TPE / BO** |
| Large heterogeneous cluster, robustness over pure sample efficiency | **ASHA / Hyperband** |
| Long jobs where recycling checkpoints is natural | **PBT** |

**Practical stack (combining methods):** TPE or random coarse search → optional fine local grid → wrap trials in **ASHA/Hyperband** to cap epoch budget → consider **PBT** instead of fixed-$\lambda$ search when training is long and schedules matter. Frameworks like **Optuna** and **Ray Tune** let you mix samplers (TPE) with pruners (ASHA) in one study.

---

### 2.4. Tricky interview questions

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

Batch normalization (BN) is a technique applied inside a neural network that **normalizes** each layer's inputs on the fly during training. After normalizing, it applies a learnable **scale** ($\gamma$) and **shift** ($\beta$) so the network can still represent any distribution it needs. The result: training is more stable, often faster, and less sensitive to hyperparameter choices like learning rate and initialization.

### 3.1. Idea

**The problem it solves.** In a deep network, every time the earlier layers update their weights, the distribution of values flowing into later layers changes — even slightly. Layer 5 has to keep adapting to a shifting input distribution from layers 1–4. This is called **internal covariate shift**, and it makes optimization harder: gradients become unreliable, and you need small learning rates or careful initialization to avoid explosions.

**What BN does.** Before passing values to the next layer, BN:

1. **Normalizes** — subtracts the batch mean and divides by the batch standard deviation, so activations have roughly zero mean and unit variance.
2. **Rescales** — multiplies by $\gamma$ and adds $\beta$ (both learned), so the network can undo the normalization if needed.

This pins each layer's input distribution to a stable range from batch to batch. Later layers see consistent statistics regardless of what the earlier layers are doing, so they can learn more independently and reliably.

**Why it helps optimization.** Normalized inputs mean the loss surface is smoother — gradients point in more consistent directions, and a step that worked last iteration is likely to work again. This allows higher learning rates and makes the network less sensitive to how weights were initialized.

### 3.2. Batch norm for a single layer

Given pre-activations $z^{(1)}, \ldots, z^{(m)}$ in layer $\ell$ on a mini-batch of size $m$:

$$\mu = \frac{1}{m}\sum_{i=1}^{m} z^{(i)}$$

$$\sigma^2 = \frac{1}{m}\sum_{i=1}^{m} \bigl(z^{(i)} - \mu\bigr)^2$$

$$z^{(i)}_{\text{norm}} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

$$\tilde{z}^{(i)} = \gamma z^{(i)}_{\text{norm}} + \beta$$

- $\gamma$, $\beta$ are **learnable** (one per feature / hidden unit), updated like $W$.
- Setting $\gamma = \sqrt{\sigma^2 + \varepsilon}$ and $\beta = \mu$ recovers the identity map — BN can always “turn off” if needed.
- $\varepsilon$ (e.g. $10^{-8}$) prevents division by zero.

**Naming trap:** BN’s $\beta$ is **not** momentum/Adam’s $\beta$.

### 3.3. Integration into a deep network

BN is usually applied **before** the nonlinearity (pre-activation BN) or after linear transform — frameworks differ; the principle is the same.

Because BN centers $z^{[\ell]}$, the additive bias $b^{[\ell]}$ is redundant and is often omitted; $\beta^{[\ell]}$ plays the bias role.

**Forward pass (layer $\ell$, mini-batch $t$):**

$$Z^{[\ell]} = W^{[\ell]} A^{[\ell-1]}$$

$$Z^{[\ell]}_{\text{norm}} = \frac{Z^{[\ell]} - \mu^{[\ell]}}{\sqrt{(\sigma^{[\ell]})^2 + \varepsilon}}$$

$$\tilde{Z}^{[\ell]} = \gamma^{[\ell]} \odot Z^{[\ell]}_{\text{norm}} + \beta^{[\ell]}$$

$$A^{[\ell]} = g^{[\ell]}(\tilde{Z}^{[\ell]})$$

**Parameters per layer:** $W^{[\ell]}, \gamma^{[\ell]}, \beta^{[\ell]}$ (no separate $b^{[\ell]}$ when BN includes $\beta$).

### 3.4. Batch norm at test time

At inference you may have batch size 1 — batch statistics are undefined. Use **running averages** accumulated during training:

$$\mu_{\text{run}} \leftarrow \rho \mu_{\text{run}} + (1-\rho) \mu_{\text{batch}}$$

$$\sigma^2_{\text{run}} \leftarrow \rho \sigma^2_{\text{run}} + (1-\rho) \sigma^2_{\text{batch}}$$

At test time, normalize with $\mu_{\text{run}}$ and $\sigma^2_{\text{run}}$. Frameworks (PyTorch `model.eval()`, TensorFlow inference mode) switch automatically.

### 3.5. Pros of batch normalization


| Benefit                          | Mechanism                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------ |
| Faster, stabler training         | Smoother loss landscape; less sensitivity to $\alpha$ and init                             |
| Reduces internal covariate shift | Later layers see more stable input distributions                                           |
| Mild regularization              | Batch $\mu$, $\sigma^2$ are noisy estimates → stochastic perturbation (like light dropout) |
| Enables higher learning rates    | Activations stay in a reasonable range                                                     |
| Easier hyperparameter tuning     | Less brittle to $\alpha$ and initialization choices                                        |


**Caveats**

- Large batch size → less noise → **weaker** BN regularization.
- Do not treat BN as a substitute for dropout/L2 when strong regularization is needed.
- Layer norm / group norm are preferred when batch statistics are unreliable (small $B$, RNNs, some NLP/vision transformers).

### 3.6. Tricky interview questions

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

