# Deep Neural Network (L-layer Network)

A **deep neural network** is a neural network with **more than one hidden layer**. The word "deep" refers to the number of hidden layers. By convention, a network is called an $L$-layer network when it has $L$ trainable layers (input layer not counted).

---

## 1. Representation

### 1.1 Architecture

A deep network has three kinds of layers:

| Kind | Layer indices | Role |
|------|--------------|------|
| Input layer | $\ell = 0$ | Receives the raw features $\mathbf{x}$; also called $\mathbf{a}^{[0]}$ |
| Hidden layers | $\ell = 1, 2, \ldots, L-1$ | Intermediate representations; not observed in training data |
| Output layer | $\ell = L$ | Produces $\hat{y} = \mathbf{a}^{[L]}$ |

A network with 1 hidden layer is called a 2-layer network, one with 4 hidden layers is called a 5-layer network, and so on (input layer is never counted).

### 1.2 Notation

| Symbol | Meaning |
|--------|---------|
| $L$ | Total number of layers (not counting the input layer) |
| $n^{[\ell]}$ | Number of units in layer $\ell$; $n^{[0]} = n_x$ (number of input features) |
| $\mathbf{W}^{[\ell]}$ | Weight matrix for layer $\ell$ |
| $\mathbf{b}^{[\ell]}$ | Bias vector for layer $\ell$ |
| $\mathbf{z}^{[\ell]}$ | Pre-activation (linear score) for layer $\ell$, single example |
| $\mathbf{a}^{[\ell]}$ | Activation for layer $\ell$, single example; $\mathbf{a}^{[0]} = \mathbf{x}$, $\mathbf{a}^{[L]} = \hat{y}$ |
| $g^{[\ell]}$ | Activation function for layer $\ell$ (may differ per layer) |
| $\mathbf{Z}^{[\ell]},\, \mathbf{A}^{[\ell]}$ | Vectorized versions over all $m$ training examples |

Superscript $[\ell]$ in square brackets denotes the **layer number**; superscript $(i)$ in round brackets denotes the **training example index**.

### 1.3 Parameters and Dimensions

Each layer $\ell$ has a weight matrix and a bias vector whose shapes are fully determined by the layer sizes:

| Parameter | Shape | Rule |
|-----------|-------|------|
| $\mathbf{W}^{[\ell]}$ | $n^{[\ell]} \times n^{[\ell-1]}$ | rows = current layer, cols = previous layer |
| $\mathbf{b}^{[\ell]}$ | $n^{[\ell]} \times 1$ | one entry per neuron in current layer |

**Example.** For a 5-layer network with $n^{[0]}=2,\ n^{[1]}=3,\ n^{[2]}=5,\ n^{[3]}=4,\ n^{[4]}=2,\ n^{[5]}=1$:

| Layer | $\mathbf{W}^{[\ell]}$ shape | $\mathbf{b}^{[\ell]}$ shape |
|-------|------------------------|------------------------|
| 1 | $3 \times 2$ | $3 \times 1$ |
| 2 | $5 \times 3$ | $5 \times 1$ |
| 3 | $4 \times 5$ | $4 \times 1$ |
| 4 | $2 \times 4$ | $2 \times 1$ |
| 5 | $1 \times 2$ | $1 \times 1$ |

**Gradient shapes always match parameter shapes.** $d\mathbf{W}^{[\ell]}$ has the same shape as $\mathbf{W}^{[\ell]}$; $d\mathbf{b}^{[\ell]}$ has the same shape as $\mathbf{b}^{[\ell]}$. This is a useful debugging check.

---

## 2. Computing the Output (Single Example)

### 2.1 Per-Neuron View

Each neuron $i$ in layer $\ell$ performs two computations — identical to a single logistic-regression unit:

$$
z^{[\ell]}_i = \mathbf{w}^{[\ell]\top}_i\,\mathbf{a}^{[\ell-1]} + b^{[\ell]}_i, \qquad
a^{[\ell]}_i = g^{[\ell]}\!\bigl(z^{[\ell]}_i\bigr).
$$

### 2.2 Vectorized Over a Layer

Stacking the $n^{[\ell]}$ weight vectors as rows of $\mathbf{W}^{[\ell]}$:

$$
\mathbf{z}^{[\ell]} = \mathbf{W}^{[\ell]}\,\mathbf{a}^{[\ell-1]} + \mathbf{b}^{[\ell]}, \qquad
\mathbf{a}^{[\ell]} = g^{[\ell]}\!\bigl(\mathbf{z}^{[\ell]}\bigr).
$$

$g^{[\ell]}$ is applied **element-wise**. Shapes: $\mathbf{z}^{[\ell]},\, \mathbf{a}^{[\ell]} \in \mathbb{R}^{n^{[\ell]} \times 1}$.

### 2.3 General Forward-Propagation Equations

The same rule applies at every layer. Starting from $\mathbf{a}^{[0]} = \mathbf{x}$:

$$
\boxed{
\mathbf{z}^{[\ell]} = \mathbf{W}^{[\ell]}\,\mathbf{a}^{[\ell-1]} + \mathbf{b}^{[\ell]}, \qquad
\mathbf{a}^{[\ell]} = g^{[\ell]}\!\bigl(\mathbf{z}^{[\ell]}\bigr), \qquad \ell = 1, 2, \ldots, L.
}
$$

Final output: $\hat{y} = \mathbf{a}^{[L]}$.

This is computed with an explicit **for loop** over $\ell = 1$ to $L$ — there is no way to vectorize across layers, and this loop is perfectly acceptable.

```python
a = x  # a^[0]
for l in range(1, L + 1):
    z = W[l] @ a + b[l]
    a = g[l](z)          # a^[l]
y_hat = a                # a^[L]
```

---

## 3. Vectorizing Across Multiple Examples

### 3.1 Matrix Notation

Stack all $m$ training examples as columns of $\mathbf{X} \in \mathbb{R}^{n^{[0]} \times m}$. Replace lowercase vectors with uppercase matrices:

$$
\boxed{
\mathbf{Z}^{[\ell]} = \mathbf{W}^{[\ell]}\,\mathbf{A}^{[\ell-1]} + \mathbf{b}^{[\ell]}, \qquad
\mathbf{A}^{[\ell]} = g^{[\ell]}\!\bigl(\mathbf{Z}^{[\ell]}\bigr), \qquad \ell = 1, 2, \ldots, L.
}
$$

Initialize with $\mathbf{A}^{[0]} = \mathbf{X}$.

### 3.2 Dimension Table (vectorized)

| Quantity | Shape (single) | Shape (vectorized) |
|----------|---------------|-------------------|
| $\mathbf{W}^{[\ell]}$ | $n^{[\ell]} \times n^{[\ell-1]}$ | same (no change) |
| $\mathbf{b}^{[\ell]}$ | $n^{[\ell]} \times 1$ | same; broadcast over $m$ columns |
| $\mathbf{z}^{[\ell]}$ / $\mathbf{Z}^{[\ell]}$ | $n^{[\ell]} \times 1$ | $n^{[\ell]} \times m$ |
| $\mathbf{a}^{[\ell]}$ / $\mathbf{A}^{[\ell]}$ | $n^{[\ell]} \times 1$ | $n^{[\ell]} \times m$ |

**Index interpretation.** In $\mathbf{Z}^{[\ell]}$ and $\mathbf{A}^{[\ell]}$:
- **Horizontal axis (columns)**: different training examples $1 \to m$.
- **Vertical axis (rows)**: different neurons in that layer.

The bias $\mathbf{b}^{[\ell]}$ (shape $n^{[\ell]} \times 1$) is added to every column via Python/NumPy broadcasting.

---

## 4. Why Deep Representations?

Deep networks can sometimes compute functions much more efficiently than shallow networks. Two key intuitions:

### 4.1 Hierarchical Feature Learning

In an image-recognition network, earlier layers detect low-level features (edges, textures), intermediate layers combine them into parts (eyes, nose), and deeper layers recognize the final objects (faces). Each layer builds on the previous one.

The same hierarchy arises in other domains:

| Domain | Layer 1 (simple) | Layer 2 | Layer 3 (complex) |
|--------|-----------------|---------|-------------------|
| Vision | Edges | Face parts (eye, nose) | Whole faces |
| Speech | Low-level waveforms | Phonemes | Words, phrases |
| Text | Characters / tokens | Morphemes | Sentences, meanings |

### 4.2 Circuit-Theory Argument

Certain functions (e.g., XOR parity of $n$ input bits) can be computed by a **deep** network with $O(\log n)$ layers and $O(n)$ total gates. A **shallow** single-hidden-layer network would need $O(2^n)$ hidden units to compute the same function — exponentially larger. This illustrates that depth enables exponentially more efficient representations for some function classes.

---

## 5. Building Blocks: Forward and Backward Functions

A clean way to think about implementing a deep network is as a sequence of **layer functions**:

### 5.1 Forward Function (layer $\ell$)

- **Input:** $\mathbf{A}^{[\ell-1]}$ (activations from previous layer)
- **Output:** $\mathbf{A}^{[\ell]}$ (activations for this layer)
- **Cache:** $\mathbf{Z}^{[\ell]},\, \mathbf{W}^{[\ell]},\, \mathbf{b}^{[\ell]}$ (stored for use in the backward step)

$$
\mathbf{Z}^{[\ell]} = \mathbf{W}^{[\ell]}\mathbf{A}^{[\ell-1]} + \mathbf{b}^{[\ell]}, \qquad
\mathbf{A}^{[\ell]} = g^{[\ell]}\!\bigl(\mathbf{Z}^{[\ell]}\bigr).
$$

### 5.2 Backward Function (layer $\ell$)

- **Input:** $d\mathbf{A}^{[\ell]}$ (gradient of loss w.r.t. activations in this layer); uses cached $\mathbf{Z}^{[\ell]},\, \mathbf{W}^{[\ell]},\, \mathbf{b}^{[\ell]}$
- **Output:** $d\mathbf{A}^{[\ell-1]}$, $d\mathbf{W}^{[\ell]}$, $d\mathbf{b}^{[\ell]}$

$$
d\mathbf{Z}^{[\ell]} = d\mathbf{A}^{[\ell]} \odot g^{[\ell]\prime}\!\bigl(\mathbf{Z}^{[\ell]}\bigr)
$$

$$
d\mathbf{W}^{[\ell]} = \frac{1}{m}\,d\mathbf{Z}^{[\ell]}\,\mathbf{A}^{[\ell-1]\top}
$$

$$
d\mathbf{b}^{[\ell]} = \frac{1}{m}\,\textstyle\sum_{\text{cols}}\, d\mathbf{Z}^{[\ell]}
$$

$$
d\mathbf{A}^{[\ell-1]} = \mathbf{W}^{[\ell]\top}\,d\mathbf{Z}^{[\ell]}
$$

$\odot$ is element-wise multiplication. The column sum is `np.sum(..., axis=1, keepdims=True)`.

### 5.3 Cache Rationale

The forward pass computes and stores $\mathbf{Z}^{[\ell]}$ (and optionally $\mathbf{W}^{[\ell]},\, \mathbf{b}^{[\ell]}$) in a cache. The backward pass reads those cached values when computing $g^{[\ell]\prime}(\mathbf{Z}^{[\ell]})$ and the weight gradients. This avoids recomputation.

```
Forward:   A^[0] → [forward_1, cache z^[1]] → [forward_2, cache z^[2]] → ... → A^[L] = ŷ
                                                                                    ↓
                                                                              compute loss
Backward:  dA^[0] ← [backward_1, read cache] ← [backward_2, read cache] ← ... ← dA^[L]
           outputs: dW^[1], db^[1], dW^[2], db^[2], ..., dW^[L], db^[L]
```

---

## 6. Gradient Descent

### 6.1 Cost Function

For binary classification with $m$ training examples:

$$
J = \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}\!\bigl(\hat{y}^{(i)},\, y^{(i)}\bigr), \qquad
\mathcal{L}(\hat{y}, y) = -\bigl[y\log\hat{y} + (1-y)\log(1-\hat{y})\bigr].
$$

The parameters across all layers are $\{\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \ldots, \mathbf{W}^{[L]}, \mathbf{b}^{[L]}\}$.

### 6.2 Training Loop

```
Initialize all W^[l] randomly (small values), all b^[l] = 0.
Repeat until convergence:
    1. Forward propagation  →  compute A^[L] = Ŷ
    2. Compute cost J
    3. Backward propagation  →  compute dW^[l] and db^[l] for all l
    4. Update parameters for l = 1, ..., L:
           W^[l] := W^[l] - alpha * dW^[l]
           b^[l] := b^[l] - alpha * db^[l]
```

$\alpha$ is the **learning rate**.

### 6.3 Initialization

Weights **must** be initialized randomly (not to zero) to break symmetry — see `nn_terms.md`, Section 6. Biases can be initialized to zero.

```python
W[l] = np.random.randn(n[l], n[l-1]) * 0.01
b[l] = np.zeros((n[l], 1))
```

---

## 7. Backpropagation

### 7.1 Overview

Backpropagation applies the **chain rule** repeatedly, moving backward from the output layer to the input layer. Each layer's backward function takes the gradient from the layer above and returns the gradient to the layer below, plus the weight gradients for that layer.

### 7.2 Initializing the Backward Pass

For binary cross-entropy with sigmoid output, the gradient at the final layer is:

$$
d\mathbf{A}^{[L]} = -\frac{\mathbf{Y}}{\mathbf{A}^{[L]}} + \frac{1 - \mathbf{Y}}{1 - \mathbf{A}^{[L]}}
$$

(element-wise). This seeds the backward recursion.

### 7.3 Backward Equations for Layer $\ell$ (General, Vectorized)

Given $d\mathbf{A}^{[\ell]}$ and the cached $\mathbf{Z}^{[\ell]},\, \mathbf{W}^{[\ell]},\, \mathbf{A}^{[\ell-1]}$:

$$
\begin{aligned}
d\mathbf{Z}^{[\ell]} &= d\mathbf{A}^{[\ell]} \odot g^{[\ell]\prime}\!\bigl(\mathbf{Z}^{[\ell]}\bigr) & &\in \mathbb{R}^{n^{[\ell]} \times m} \\[4pt]
d\mathbf{W}^{[\ell]} &= \frac{1}{m}\,d\mathbf{Z}^{[\ell]}\,\mathbf{A}^{[\ell-1]\top} & &\in \mathbb{R}^{n^{[\ell]} \times n^{[\ell-1]}} \\[4pt]
d\mathbf{b}^{[\ell]} &= \frac{1}{m}\sum_{\text{cols}}\,d\mathbf{Z}^{[\ell]} & &\in \mathbb{R}^{n^{[\ell]} \times 1} \\[4pt]
d\mathbf{A}^{[\ell-1]} &= \mathbf{W}^{[\ell]\top}\,d\mathbf{Z}^{[\ell]} & &\in \mathbb{R}^{n^{[\ell-1]} \times m}
\end{aligned}
$$

The last equation propagates the error signal one layer back and is used as the input to the backward function of layer $\ell - 1$.

### 7.4 Dimension Sanity Check

Every parameter and its gradient share the same shape:

| Variable | Shape | Gradient | Shape |
|----------|-------|----------|-------|
| $\mathbf{W}^{[\ell]}$ | $n^{[\ell]} \times n^{[\ell-1]}$ | $d\mathbf{W}^{[\ell]}$ | $n^{[\ell]} \times n^{[\ell-1]}$ |
| $\mathbf{b}^{[\ell]}$ | $n^{[\ell]} \times 1$ | $d\mathbf{b}^{[\ell]}$ | $n^{[\ell]} \times 1$ |
| $\mathbf{Z}^{[\ell]}, \mathbf{A}^{[\ell]}$ | $n^{[\ell]} \times m$ | $d\mathbf{Z}^{[\ell]}, d\mathbf{A}^{[\ell]}$ | $n^{[\ell]} \times m$ |

Checking these dimensions when implementing backprop catches the majority of bugs.

### 7.5 NumPy Implementation Sketch

```python
# ---- Forward propagation ----
caches = {}
A = X                             # A^[0] = X
for l in range(1, L + 1):
    A_prev = A
    Z = W[l] @ A_prev + b[l]
    A = g[l](Z)
    caches[l] = (A_prev, Z)       # store for backward pass
Y_hat = A                         # A^[L]

# ---- Initialize backward pass ----
dA = -(Y / Y_hat) + (1 - Y) / (1 - Y_hat)   # dA^[L], binary cross-entropy

# ---- Backward propagation ----
grads = {}
for l in reversed(range(1, L + 1)):
    A_prev, Z = caches[l]
    dZ = dA * g_prime[l](Z)                                 # element-wise
    grads['dW' + str(l)] = (1/m) * dZ @ A_prev.T
    grads['db' + str(l)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA = W[l].T @ dZ                                        # pass to l-1

# ---- Parameter updates ----
for l in range(1, L + 1):
    W[l] -= alpha * grads['dW' + str(l)]
    b[l] -= alpha * grads['db' + str(l)]
```

---

## 8. Hyperparameters

Parameters $\mathbf{W}^{[\ell]}$ and $\mathbf{b}^{[\ell]}$ are **learned** by gradient descent. Everything else that you choose before training is a **hyperparameter** — it controls how learning happens.

| Hyperparameter | Symbol / notation | Effect |
|----------------|-------------------|--------|
| Learning rate | $\alpha$ | Step size per gradient update; too large overshoots, too small trains slowly |
| Number of gradient-descent iterations | — | How many passes through the update loop |
| Number of hidden layers | $L$ | Depth; more layers can learn more abstract features but are harder to train |
| Units per hidden layer | $n^{[1]}, n^{[2]}, \ldots$ | Width; more units increase capacity but also cost |
| Activation functions | $g^{[1]}, \ldots, g^{[L]}$ | Nonlinearity type (ReLU, tanh, sigmoid, etc.) |
| Initialization scale | e.g. $0.01$ | Controls initial weight magnitude |

Hyperparameters determine the final values of the learned parameters — hence the name "hyper." The standard workflow is:

1. Choose initial hyperparameter values based on intuition or literature.
2. Train the network and evaluate on a held-out development (validation) set.
3. Adjust hyperparameters based on results and repeat.

Because optimal hyperparameters vary by problem and dataset, it is common to search over a range of values (e.g., try $\alpha \in \{0.001, 0.01, 0.1\}$ and compare). The number of hidden layers $L$ itself is often treated as a hyperparameter to tune.
