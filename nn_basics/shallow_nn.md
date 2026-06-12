# Shallow Neural Network (One Hidden Layer)

A **shallow neural network** refers to a neural network with exactly **one hidden layer** — making it technically a **2-layer network** (input layer is not counted as an official layer).

Shallow Neural Network basic:
- Representation;
- Computing the Neural Network Output (Single Example);
- Vectorizing Across Multiple Examples;
- Why the Vectorized Equations Are Correct;
- Gradient Descent;
- Backpropagation;
- Softmax Regression (Multi-class Classification)

and interview-style pitfalls.

---

## 1. Representation

### 1.1. Architecture

![Shallow NN graph](graphs\shallow_nn_graph.png)

A shallow neural network with $n_0$ input features, $n_1$ hidden units, and $n_2$ output units has three layers:

| Layer | Index | Description |
|-------|-------|-------------|
| Input layer | $\ell = 0$ | Input features $\mathbf{x} = (x_1, x_2, \ldots, x_{n_0})$; also written $\mathbf{a}^{[0]} = \mathbf{x}$ |
| Hidden layer | $\ell = 1$ | $n_1$ neurons; activations $\mathbf{a}^{[1]} \in \mathbb{R}^{n_1}$; **not observed** in training data |
| Output layer | $\ell = 2$ | $n_2$ neurons (typically $n_2 = 1$ for binary classification); outputs $\hat{y} = \mathbf{a}^{[2]}$ |

**Why "2-layer"?** By convention the input layer is not counted. The hidden layer is layer 1, the output layer is layer 2.

**Why "hidden"?** In a supervised training set you observe inputs $x$ and labels $y$, but never the intermediate values $a^{[1]}$ — hence "hidden."

### 1.2. Parameters and Dimensions

Each layer $\ell$ has a weight matrix $\mathbf{W}^{[\ell]}$ and a bias vector $\mathbf{b}^{[\ell]}$:

| Parameter | Dimension | Meaning |
|-----------|-----------|---------|
| $\mathbf{W}^{[1]}$ | $n_1 \times n_0$ | Each row is the weight vector of one hidden neuron |
| $\mathbf{b}^{[1]}$ | $n_1 \times 1$ | Bias for each hidden neuron |
| $\mathbf{W}^{[2]}$ | $n_2 \times n_1$ | Weight matrix of the output layer |
| $\mathbf{b}^{[2]}$ | $n_2 \times 1$ | Bias for the output neuron(s) |

**Dimension rule of thumb.** $\mathbf{W}^{[\ell]}$ has shape $n_\ell \times n_{\ell-1}$: rows = neurons in the current layer, columns = neurons (or features) in the previous layer.

### 1.3. Notation

- Superscript $[\ell]$ in square brackets denotes the **layer number**.
- Subscript $i$ denotes the **node index** within a layer.
- Superscript $(i)$ in round brackets denotes the **training example index**.
- $g^{[\ell]}$ is the activation function used in layer $\ell$; $g^{[\ell]\prime}$ is its derivative.

### 1.4. Tricky interview questions

**Q1. Why is a network with one hidden layer called a 2-layer network?**  
By convention, only trainable layers are counted, so the hidden layer is layer 1 and the output layer is layer 2.

**Q2. What does "hidden" mean in hidden layer?**  
The hidden activations are not observed directly in the training data; they are internal representations learned by the network.

**Q3. What shape should $\mathbf{W}^{[\ell]}$ have?**  
$\mathbf{W}^{[\ell]}$ has shape $n_\ell \times n_{\ell-1}$: one row per current-layer neuron and one column per previous-layer unit.

---

## 2. Computing the Neural Network Output (Single Example)

### 2.1. Per-Neuron View

Each neuron $i$ in the hidden layer performs two steps — identical to logistic regression:

$$
z^{[1]}_i = \mathbf{w}^{[1]\top}_i\,\mathbf{x} + b^{[1]}_i, \qquad
a^{[1]}_i = g^{[1]}\!\bigl(z^{[1]}_i\bigr).
$$

### 2.2. Vectorized Over the Layer

Stacking the individual weight vectors as rows of $\mathbf{W}^{[1]}$ and computing all hidden neurons simultaneously:

$$
\underbrace{\mathbf{z}^{[1]}}_{n_1 \times 1}
= \underbrace{\mathbf{W}^{[1]}}_{n_1 \times n_0}\,
  \underbrace{\mathbf{x}}_{n_0 \times 1}
+ \underbrace{\mathbf{b}^{[1]}}_{n_1 \times 1},
\qquad
\mathbf{a}^{[1]} = g^{[1]}\!\bigl(\mathbf{z}^{[1]}\bigr).
$$

Here $g^{[1]}$ is applied **element-wise** to the vector $\mathbf{z}^{[1]}$.

The output layer then takes $\mathbf{a}^{[1]}$ as its input:

$$
\underbrace{\mathbf{z}^{[2]}}_{n_2 \times 1}
= \underbrace{\mathbf{W}^{[2]}}_{n_2 \times n_1}\,
  \underbrace{\mathbf{a}^{[1]}}_{n_1 \times 1}
+ \underbrace{\mathbf{b}^{[2]}}_{n_2 \times 1},
\qquad
\hat{y} = \mathbf{a}^{[2]} = g^{[2]}\!\bigl(\mathbf{z}^{[2]}\bigr).
$$

For binary classification $n_2 = 1$ and $g^{[2]} = \sigma$ (sigmoid), so $a^{[2]} \in (0,1)$ is a probability.

### 2.3. Four Forward-Prop Equations (Summary)

$$
\begin{aligned}
\mathbf{z}^{[1]} &= \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]} \\
\mathbf{a}^{[1]} &= g^{[1]}\!\bigl(\mathbf{z}^{[1]}\bigr) \\
\mathbf{z}^{[2]} &= \mathbf{W}^{[2]}\mathbf{a}^{[1]} + \mathbf{b}^{[2]} \\
\mathbf{a}^{[2]} &= g^{[2]}\!\bigl(\mathbf{z}^{[2]}\bigr) = \hat{y}
\end{aligned}
$$

This is conceptually $n_1 + 1$ logistic-regression units, computed all at once.

### 2.4. Tricky interview questions

**Q1. Why is the hidden layer computed with a matrix multiplication?**  
Each row of $\mathbf{W}^{[1]}$ is one neuron's weight vector, so $\mathbf{W}^{[1]}\mathbf{x}$ computes all hidden logits at once.

**Q2. Why is $g^{[1]}$ applied element-wise?**  
Each hidden neuron has its own scalar pre-activation, so the activation function is applied separately to each component.

**Q3. For binary classification, why is the output usually sigmoid?**  
With one output unit, sigmoid maps the output logit to a probability in $(0,1)$.

---

## 3. Vectorizing Across Multiple Examples

### 3.1. From Loops to Matrices

For $m$ training examples, a naive implementation loops over examples:

```python
for i in range(1, m+1):
    z1_i = W1 @ x_i + b1
    a1_i = g1(z1_i)
    z2_i = W2 @ a1_i + b2
    a2_i = g2(z2_i)          # = y_hat_i
```

To eliminate the loop, stack all training examples as **columns** of a matrix:

$$
\mathbf{X} = \begin{bmatrix} | & | & & | \\ \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(m)} \\ | & | & & | \end{bmatrix} \in \mathbb{R}^{n_0 \times m}.
$$

### 3.2. Vectorized Forward Propagation

Replace the lowercase vectors with uppercase matrices ($\mathbf{X}, \mathbf{Z}^{[1]}, \mathbf{A}^{[1]}, \ldots$) formed by stacking the corresponding column vectors for each training example:

$$
\begin{aligned}
\mathbf{Z}^{[1]} &= \mathbf{W}^{[1]}\mathbf{X} + \mathbf{b}^{[1]} & &\in \mathbb{R}^{n_1 \times m} \\
\mathbf{A}^{[1]} &= g^{[1]}\!\bigl(\mathbf{Z}^{[1]}\bigr) & &\in \mathbb{R}^{n_1 \times m} \\
\mathbf{Z}^{[2]} &= \mathbf{W}^{[2]}\mathbf{A}^{[1]} + \mathbf{b}^{[2]} & &\in \mathbb{R}^{n_2 \times m} \\
\mathbf{A}^{[2]} &= g^{[2]}\!\bigl(\mathbf{Z}^{[2]}\bigr) & &\in \mathbb{R}^{n_2 \times m}
\end{aligned}
$$

The bias vectors $\mathbf{b}^{[\ell]}$ are added via **broadcasting**: each bias column is replicated across all $m$ columns.

### 3.3. Index Interpretation

| Axis | Meaning |
|------|---------|
| **Horizontal** (columns) | Different training examples $(1 \to m)$ |
| **Vertical** (rows) | Different nodes / features in that layer |

So the top-left element of $\mathbf{A}^{[1]}$ is the activation of hidden unit 1 on training example 1; moving right gives the same unit on different examples; moving down gives different units on the same example.

### 3.4. Tricky interview questions

**Q1. Why are training examples stored as columns?**  
This convention lets $\mathbf{W}^{[\ell]}\mathbf{A}^{[\ell-1]}$ compute every example in one matrix multiplication.

**Q2. What do columns and rows mean in $\mathbf{A}^{[\ell]}$?**  
Columns correspond to different examples, and rows correspond to different neurons in layer $\ell$.

**Q3. Does vectorization change the math of the network?**  
No. It computes the same per-example equations, but batches them into matrix operations.

---

## 4. Why the Vectorized Equations Are Correct

Consider just the first step, $\mathbf{Z}^{[1]} = \mathbf{W}^{[1]}\mathbf{X} + \mathbf{b}^{[1]}$, and temporarily ignore $\mathbf{b}^{[1]}$.

For a single example $\mathbf{x}^{(i)}$:

$$
\mathbf{W}^{[1]}\mathbf{x}^{(i)} = \mathbf{z}^{[1](i)} \quad \text{(column vector, length } n_1\text{)}.
$$

When all $m$ examples are stacked as columns of $\mathbf{X}$:

$$
\mathbf{W}^{[1]}\mathbf{X}
= \mathbf{W}^{[1]} \begin{bmatrix} \mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(m)} \end{bmatrix}
= \begin{bmatrix} \mathbf{z}^{[1](1)} & \mathbf{z}^{[1](2)} & \cdots & \mathbf{z}^{[1](m)} \end{bmatrix}
= \mathbf{Z}^{[1]}.
$$

This is exactly how matrix multiplication distributes over columns. Python/NumPy broadcasting then adds $\mathbf{b}^{[1]}$ to every column, matching the per-example addition.

The same column-stacking argument applies identically to the other three forward-prop equations (and later to the back-prop equations as well).

**Symmetry note.** Because $\mathbf{x} = \mathbf{a}^{[0]}$, the two pairs of equations can be written uniformly:

$$
\mathbf{Z}^{[\ell]} = \mathbf{W}^{[\ell]}\mathbf{A}^{[\ell-1]} + \mathbf{b}^{[\ell]}, \qquad \mathbf{A}^{[\ell]} = g^{[\ell]}\!\bigl(\mathbf{Z}^{[\ell]}\bigr), \quad \ell = 1, 2.
$$

This pattern extends naturally to deeper networks.

### 4.1. Tricky interview questions

**Q1. Why does $\mathbf{W}^{[1]}\mathbf{X}$ contain all examples' logits?**  
Matrix multiplication applies the same weight matrix to every column of $\mathbf{X}$, producing one output column per example.

**Q2. Why can the same formula work for both layers?**  
Each layer takes the previous layer's activations as input, so $\mathbf{A}^{[\ell-1]}$ plays the same role at every layer.

**Q3. Why is broadcasting the bias valid?**  
The same bias vector should be added to every example's pre-activation, which is exactly what broadcasting does.

---

## 5. Gradient Descent

### 5.1. Cost Function

For binary classification with $m$ training examples, the cost is the average binary cross-entropy loss:

$$
J(\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \mathbf{W}^{[2]}, \mathbf{b}^{[2]})
= \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}\!\bigl(\hat{y}^{(i)},\, y^{(i)}\bigr),
$$

$$
\mathcal{L}(\hat{y}, y) = -\bigl[y \log \hat{y} + (1 - y)\log(1 - \hat{y})\bigr].
$$

### 5.2. Training Loop

```
Initialize parameters randomly
Repeat until convergence:
    1. Forward propagation  →  compute A^[2] = Y_hat
    2. Compute cost J
    3. Backward propagation  →  compute dW1, db1, dW2, db2
    4. Update parameters:
           W1 := W1 - alpha * dW1
           b1 := b1 - alpha * db1
           W2 := W2 - alpha * dW2
           b2 := b2 - alpha * db2
```

where $\alpha$ is the **learning rate**.

### 5.3. Weight Initialization

Weights **must not** be initialized to zero — all neurons would be symmetric and would learn the same function forever (see `nn_terms.md`, Section 6). Instead:

```python
W1 = np.random.randn(n1, n0) * 0.01
b1 = np.zeros((n1, 1))
W2 = np.random.randn(n2, n1) * 0.01
b2 = np.zeros((n2, 1))
```

The small multiplier $0.01$ keeps pre-activations near zero, avoiding saturation of tanh/sigmoid at the start of training.

### 5.4. Tricky interview questions

**Q1. Why are weights initialized randomly?**  
Random weights break symmetry so hidden neurons can learn different functions.

**Q2. Why can biases be initialized to zero?**  
Zero biases do not make hidden neurons identical as long as the weights are randomly initialized.

**Q3. What role does the learning rate $\alpha$ play?**  
It controls how large each parameter update is during gradient descent.

---

## 6. Backpropagation

### 6.1. Intuition

Backprop applies the **chain rule** backwards through the computation graph to obtain the gradient of $J$ with respect to every parameter.

For logistic regression (one layer) the chain was:

$$
\frac{\partial J}{\partial \mathbf{w}} = \frac{\partial J}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}},
$$

resulting in $dz = a - y$, $d\mathbf{w} = dz \cdot \mathbf{x}$, $db = dz$.

A shallow neural network runs the same logic **twice** — once through the output layer, then again through the hidden layer.

### 6.2. Vectorized Backprop Equations

**Forward pass (recap):**

$$
\mathbf{Z}^{[1]} = \mathbf{W}^{[1]}\mathbf{X} + \mathbf{b}^{[1]}, \quad
\mathbf{A}^{[1]} = g^{[1]}\!\bigl(\mathbf{Z}^{[1]}\bigr), \quad
\mathbf{Z}^{[2]} = \mathbf{W}^{[2]}\mathbf{A}^{[1]} + \mathbf{b}^{[2]}, \quad
\mathbf{A}^{[2]} = g^{[2]}\!\bigl(\mathbf{Z}^{[2]}\bigr).
$$

**Backward pass** (binary classification, $g^{[2]} = \sigma$, $\mathbf{Y} \in \mathbb{R}^{1 \times m}$):

$$
\begin{aligned}
d\mathbf{Z}^{[2]} &= \mathbf{A}^{[2]} - \mathbf{Y} & &\in \mathbb{R}^{n_2 \times m} \\[4pt]
d\mathbf{W}^{[2]} &= \frac{1}{m}\,d\mathbf{Z}^{[2]}\,\mathbf{A}^{[1]\top} & &\in \mathbb{R}^{n_2 \times n_1} \\[4pt]
d\mathbf{b}^{[2]} &= \frac{1}{m}\sum_{\text{cols}} d\mathbf{Z}^{[2]} & &\in \mathbb{R}^{n_2 \times 1} \\[4pt]
d\mathbf{Z}^{[1]} &= \mathbf{W}^{[2]\top} d\mathbf{Z}^{[2]} \odot g^{[1]\prime}\!\bigl(\mathbf{Z}^{[1]}\bigr) & &\in \mathbb{R}^{n_1 \times m} \\[4pt]
d\mathbf{W}^{[1]} &= \frac{1}{m}\,d\mathbf{Z}^{[1]}\,\mathbf{X}^{\top} & &\in \mathbb{R}^{n_1 \times n_0} \\[4pt]
d\mathbf{b}^{[1]} &= \frac{1}{m}\sum_{\text{cols}} d\mathbf{Z}^{[1]} & &\in \mathbb{R}^{n_1 \times 1}
\end{aligned}
$$

$\odot$ denotes element-wise (Hadamard) multiplication. The column-sum is `np.sum(..., axis=1, keepdims=True)` in NumPy.

### 6.3. Step-by-Step Derivation

**Step 1 — Output layer ($\ell = 2$).** For the sigmoid output with binary cross-entropy, the chain rule collapses to:

$$
dz^{[2]} = a^{[2]} - y.
$$

Vectorized: $d\mathbf{Z}^{[2]} = \mathbf{A}^{[2]} - \mathbf{Y}$.

**Step 2 — Output layer weights.** $\mathbf{W}^{[2]}$ enters through $\mathbf{Z}^{[2]} = \mathbf{W}^{[2]}\mathbf{A}^{[1]} + \mathbf{b}^{[2]}$, so:

$$
d\mathbf{W}^{[2]} = \frac{1}{m}\,d\mathbf{Z}^{[2]}\,\mathbf{A}^{[1]\top}, \qquad
d\mathbf{b}^{[2]} = \frac{1}{m}\sum_{\text{cols}} d\mathbf{Z}^{[2]}.
$$

(Analogous to logistic regression where $d\mathbf{w} = dz \cdot \mathbf{x}$, with $\mathbf{A}^{[1]}$ playing the role of $\mathbf{x}$.)

**Step 3 — Hidden layer ($\ell = 1$).** To propagate the error back through the hidden layer, apply the chain rule through $\mathbf{Z}^{[2]} = \mathbf{W}^{[2]}\mathbf{A}^{[1]}$ and then through $\mathbf{A}^{[1]} = g^{[1]}(\mathbf{Z}^{[1]})$:

$$
d\mathbf{Z}^{[1]} = \underbrace{\mathbf{W}^{[2]\top} d\mathbf{Z}^{[2]}}_{\text{error from layer 2}} \odot \underbrace{g^{[1]\prime}\!\bigl(\mathbf{Z}^{[1]}\bigr)}_{\text{activation gradient}}.
$$

**Step 4 — Hidden layer weights.**

$$
d\mathbf{W}^{[1]} = \frac{1}{m}\,d\mathbf{Z}^{[1]}\,\mathbf{X}^{\top}, \qquad
d\mathbf{b}^{[1]} = \frac{1}{m}\sum_{\text{cols}} d\mathbf{Z}^{[1]}.
$$

Since $\mathbf{X} = \mathbf{A}^{[0]}$, these are structurally identical to the layer-2 updates with the index shifted by one.

### 6.4. Dimension Check

A useful sanity check: **every parameter and its gradient always have the same shape**.

| Variable | Shape | Gradient | Shape |
|----------|-------|----------|-------|
| $\mathbf{W}^{[1]}$ | $n_1 \times n_0$ | $d\mathbf{W}^{[1]}$ | $n_1 \times n_0$ |
| $\mathbf{b}^{[1]}$ | $n_1 \times 1$ | $d\mathbf{b}^{[1]}$ | $n_1 \times 1$ |
| $\mathbf{W}^{[2]}$ | $n_2 \times n_1$ | $d\mathbf{W}^{[2]}$ | $n_2 \times n_1$ |
| $\mathbf{b}^{[2]}$ | $n_2 \times 1$ | $d\mathbf{b}^{[2]}$ | $n_2 \times 1$ |
| $\mathbf{Z}^{[\ell]}, \mathbf{A}^{[\ell]}$ | $n_\ell \times m$ | $d\mathbf{Z}^{[\ell]}$ | $n_\ell \times m$ |

Verifying these dimensions when implementing backprop catches the majority of bugs.

### 6.5. NumPy Implementation Sketch

```python
# --- Forward propagation ---
Z1 = W1 @ X + b1                          # (n1, m)
A1 = activation_hidden(Z1)                # (n1, m)
Z2 = W2 @ A1 + b2                         # (n2, m)
A2 = sigmoid(Z2)                          # (n2, m)  = Y_hat

# --- Backward propagation ---
dZ2 = A2 - Y                                              # (n2, m)
dW2 = (1/m) * dZ2 @ A1.T                                  # (n2, n1)
db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)          # (n2, 1)

dZ1 = W2.T @ dZ2 * activation_hidden_deriv(Z1)            # (n1, m)
dW1 = (1/m) * dZ1 @ X.T                                   # (n1, n0)
db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)          # (n1, 1)

# --- Parameter updates ---
W1 -= alpha * dW1
b1 -= alpha * db1
W2 -= alpha * dW2
b2 -= alpha * db2
```

### 6.6. Tricky interview questions

**Q1. Why does the output-layer error become $d\mathbf{Z}^{[2]} = \mathbf{A}^{[2]} - \mathbf{Y}$?**  
For sigmoid output with binary cross-entropy, the chain-rule terms simplify to prediction minus label.

**Q2. Why does $d\mathbf{Z}^{[1]}$ include $\mathbf{W}^{[2]\top}$?**  
The hidden layer receives its error signal through the output layer weights, and the transpose maps that signal back to hidden-unit dimensions.

**Q3. What is the easiest way to catch many backprop bugs?**  
Check that every parameter and its gradient have the same shape.

---

## 7. Softmax Regression (Multi-class Classification)

![Softmax graph](graphs\softmax.png)

Softmax is a function that turns a list of raw scores into probabilities that add up to 1, so the model can choose one class out of many. It is commonly used at the output layer of neural networks for multiclass classification.

### 7.1. Softmax activation

For $C$ classes, the output layer has $n^{[L]} = C$ units. Instead of sigmoid, apply:

$$t_i = e^{z^{[L]}_i} \qquad (i = 1, \ldots, C)$$

$$\boxed{a^{[L]}_i = \hat{y}_i = \frac{t_i}{\sum_{j=1}^{C} t_j} = \frac{e^{z^{[L]}_i}}{\sum_{j=1}^{C} e^{z^{[L]}_j}}}$$

The output is a probability vector: $\sum_i \hat{y}_i = 1$, all $\hat{y}_i \geq 0$.

NB!
- Special case of $C = 2$ reduces to logistic regression;
- Softmax is best when each example belongs to exactly one class. If an example can belong to multiple classes at once, you usually use separate sigmoid outputs instead.

### 7.2. Loss function (cross-entropy)

Softmax is useful because it makes the output easy to interpret and works naturally when the classes are mutually exclusive. It is typically paired with cross-entropy loss, which encourages the model to assign high probability to the correct class and low probability to the others.

For a single example with ground-truth one-hot label $y \in \mathbb{R}^C$:

$$\mathcal{L}(\hat{y}, y) = -\sum_{j=1}^{C} y_j \log \hat{y}_j$$

Since $y$ is one-hot (only one $y_k = 1$, rest zero), this simplifies to:

$$\mathcal{L}(\hat{y}, y) = -\log \hat{y}_k$$

where $k$ is the true class. Minimizing the loss maximizes the predicted probability of the correct class.

**Cost over the full training set:**

$$J = \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})$$

### 7.3. Backprop for the output layer

The gradient of the loss with respect to $z^{[L]}$ is:

$$dz^{[L]} = \hat{y} - y$$

This is a $C \times 1$ vector (or $C \times m$ for a mini-batch). All other backprop equations proceed as normal from here.

### 7.4. Tricky interview questions

**Q1. When should softmax be used instead of sigmoid?**  
Use softmax when each example belongs to exactly one of several mutually exclusive classes.

**Q2. Why do softmax outputs sum to 1?**  
Each exponentiated score is divided by the sum of all exponentiated scores, producing a probability distribution.

**Q3. What is the output-layer gradient for softmax with cross-entropy?**  
It has the same simple form: predicted probability vector minus the one-hot label vector.

---