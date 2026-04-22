# Neural Networks: Logistic Regression as the Simplest Network

This note uses **vectorized** notation: one training step updates all parameters using full matrices for $m$ examples at once.

**Notation (used throughout).**

- $n_x$: number of features per example.
- $m$: number of examples (batch size).
- Design matrix $\mathbf{X} \in \mathbb{R}^{n_x \times m}$: each **column** is one example $\mathbf{x}^{(i)} \in \mathbb{R}^{n_x}$.
- Labels $\mathbf{y} \in \mathbb{R}^{1 \times m}$ (row vector), with $y^{(i)} \in \{0,1\}$ for binary classification.
- Weights $\mathbf{w} \in \mathbb{R}^{n_x}$ (column vector) and bias $b \in \mathbb{R}$.
- $\sigma(\cdot)$ is the sigmoid, applied **element-wise** to matrices.
- $\odot$ is the Hadamard (element-wise) product.
- $\mathbf{1}_m \in \mathbb{R}^{m}$ is a column vector of ones (subscript omitted when size is clear).

---



## 1. Logistic regression as the simplest neural network

### 1.1 Binary classification

**Goal.** Predict one of two classes, encoded as $y^{(i)} \in \{0, 1\}$.

**Probabilistic model.** Estimate probability that output $y^{(i)}$ is of a class 1 given input matrix $x^{(i)}$:

$$
\hat{y}^{(i)} = P(y^{(i)} = 1 \mid \mathbf{x}^{(i)})$ with $\hat{y}^{(i)} \in (0,1)
$$

Decision rule: predict class $1$ if $\hat{y}^{(i)} \ge \tfrac{1}{2}$, else class $0$.

---

### 2.2 Logistic regression (vectorized)

For one example,

$$
z^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + b, \qquad
\hat{y}^{(i)} = a^{(i)} = \sigma\bigl(z^{(i)}\bigr) = \frac{1}{1 + e^{-z^{(i)}}}.
$$

**All $m$ examples at once.** Stack columns in $\mathbf{X}$ and define

$$
\mathbf{Z} = \mathbf{w}^\top \mathbf{X} + b \,\mathbf{1}_m^\top \quad \in \mathbb{R}^{1 \times m},
$$

$$
\mathbf{A} = \sigma(\mathbf{Z}) \quad \text{(element-wise)}.
$$

Here 
- $Y$ is output (target) vector of ... size 
$$
add vector here
$$
- $W$ 
$$
add hereis a transponed weight vector of ... size 
$$
- $X$ is input (features) matrix of ... size 
$$
add matrix here
$$
- $b$ is bias vector of ... size  (**broadcasted** across columns). 
$$
add here
$$
- $A$
- $Z$
- $sigma$ is activation function
$$
add formula here
$$

so that 
if Z is large 
if Z is large negative number 

$$
add sigmoid graph
$$


If you prefer column vectors $\mathbf{z}^{(i)}$ stacked into $\mathbf{Z}_{\mathrm{col}} \in \mathbb{R}^{m \times 1}$, equivalently $\mathbf{Z}_{\mathrm{col}} = \mathbf{X}^\top \mathbf{w} + b\,\mathbf{1}_m$.

---

### 2.3 Logistic regression cost function (vectorized)

**Loss function (binary cross-entropy, negative log-likelihood for Bernoulli labels):**

The loss function measures the discrepancy between the prediction (𝑦̂(𝑖)) and the desired output (𝑦(𝑖)) per-example. 
In other words, the loss function computes the error for a single training example x(𝑖) .  

$$
\mathcal{L}^{(i)} = - \Bigl( y^{(i)} \log a^{(i)} + \bigl(1 - y^{(i)}\bigr) \log\bigl(1 - a^{(i)}\bigr) \Bigr).
$$

- If a^(𝑖) = 1: 𝐿(𝑦̂^(𝑖),a^(𝑖)) = −log(𝑦̂(𝑖)) where log(𝑦̂(𝑖)) and 𝑦̂(𝑖)  should be close to 1 
- If a^(𝑖) = 0: 𝐿(𝑦̂(𝑖),a^(𝑖)) = - log(1 − 𝑦̂(𝑖)) where log(1 − 𝑦̂(𝑖)) and 𝑦̂(𝑖) should be close to 0 

**Cost function:**

The cost function is the average of the loss function of the entire training set (over the batch). The goal of NN training is to find the 
parameters 𝑤 𝑎𝑛𝑑 𝑏 that minimize the overall cost function.

$$
J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}^{(i)}
= -\frac{1}{m} \sum_{i=1}^{m} \Bigl( y^{(i)} \log a^{(i)} + \bigl(1 - y^{(i)}\bigr) \log\bigl(1 - a^{(i)}\bigr) \Bigr).
$$

**Vector form** (same value as the sum; $\mathbf{y}, \mathbf{A} \in \mathbb{R}^{1 \times m}$ as row vectors, $\mathbf{1}$ a row of ones of matching size, logs element-wise):

$$
\boldsymbol{\ell} = \mathbf{y} \odot \log \mathbf{A} + (\mathbf{1} - \mathbf{y}) \odot \log(\mathbf{1} - \mathbf{A}) \in \mathbb{R}^{1 \times m},
$$

$$
J = -\frac{1}{m}\,\mathbf{1}_{1 \times m}\,\boldsymbol{\ell}^\top
= -\frac{1}{m}\sum_{j=1}^{m} \ell_j.
$$

Equivalently, with $\mathbf{y}_c = \mathbf{y}^\top$, $\mathbf{A}_c = \mathbf{A}^\top$ in $\mathbb{R}^{m}$,

$$
J = -\frac{1}{m}\,\mathbf{1}_m^\top\Bigl( \mathbf{y}_c \odot \log \mathbf{A}_c + (\mathbf{1}_m - \mathbf{y}_c) \odot \log(\mathbf{1}_m - \mathbf{A}_c) \Bigr).
$$

---

### 2.12 Binary cross-entropy as a cost function

Cross-entropy is a way to measure how different two probability distributions are. In machine learning, it usually measures how well a model’s predicted probabilities match the true labels, so it is commonly used as a loss function for classification.

For discrete distributions 
p
p and 
q
q, cross-entropy is:

H
(
p
,
q
)
=
−
∑
x
p
(
x
)
log
⁡
q
(
x
)
H(p,q)=− 
x
∑
​
 p(x)logq(x)

where 
p
p is the true distribution,

q
q is the predicted distribution

Logistic regression predicts a probability 
a
(
i
)
=
P
(
y
(
i
)
=
1
∣
x
(
i
)
)
a 
(i)
 =P(y 
(i)
 =1∣x 
(i)
 ).
Since the true label 
y
(
i
)
y 
(i)
  is either 0 or 1, it makes sense to measure how likely the model thinks the true answer is. The cost function does exactly that by taking the negative log-likelihood.

For one example:

if 
y
=
1
y=1, the loss becomes 
−
log
⁡
(
a
)
−log(a)

if 
y
=
0
y=0, the loss becomes 
−
log
⁡
(
1
−
a
)
−log(1−a)

The combined form is:

−
(
y
log
⁡
a
+
(
1
−
y
)
log
⁡
(
1
−
a
)
)
−(yloga+(1−y)log(1−a))
This is just a compact way to write both cases at once. If the correct class gets low probability, the loss becomes large.

**Probabilistic view.** Assume $y^{(i)} \sim \mathrm{Bernoulli}(a^{(i)})$ with $a^{(i)} = P(y^{(i)}=1\mid \mathbf{x}^{(i)})$. The **negative log-likelihood** for independent examples is

$$
-\frac{1}{m}\sum_{i=1}^{m} \log P\bigl(y^{(i)} \mid \mathbf{x}^{(i)}\bigr)
= -\frac{1}{m}\sum_{i=1}^{m} \Bigl( y^{(i)}\log a^{(i)} + (1-y^{(i)})\log(1-a^{(i)}) \Bigr),
$$

which is exactly $J$.

**Properties.**

- **Convex** in $(\mathbf{w}, b)$ for logistic regression (sigmoid + cross-entropy), so gradient descent with suitable $\alpha$ finds the global minimum under typical conditions.
For logistic regression, sigmoid plus cross-entropy gives a convex objective in 
(
w
,
b
)
(w,b).
That means optimization is much easier than with many other neural-network losses, because gradient descent is not fighting lots of bad local minima in this case.

- **Penalizes confident mistakes heavily**: if $y^{(i)}=1$ but $a^{(i)}\approx 0$, then $-\log a^{(i)}$ is large.
For example:

if the true label is 1 and the model predicts 
a
=
0.99
a=0.99, the loss is small

if the true label is 1 and the model predicts 
a
=
0.01
a=0.01, the loss is huge

That is good behavior, because being confidently wrong should be punished more than being uncertain.

- **Matches outputs to probabilities** when trained with this loss, so $a^{(i)}$ is calibrated as an estimate of class probability under the model assumptions.
Because the output is interpreted as a probability, the model is trained to make the predicted probability match the observed label.
So after training, 
a
a can be read as the model’s estimate of class probability, not just a raw score.

---

### 2.6 Gradient descent

Gradient descent is used for NN training during backpropogation.

**Idea.** Update NN parameters $W$ and $b$ in the direction opposite to the gradient of $J$.



$$
\mathbf{w} \ := \mathbf{w} - \alpha \,\frac{\partial J}{\partial \mathbf{w}}, \qquad
b \ := b - \alpha \,\frac{\partial J}{\partial b},
$$

where $\alpha > 0$ is the **learning rate**.

For logistic regression, $\frac{\partial J}{\partial \mathbf{w}}$ and $\frac{\partial J}{\partial b}$ are computed from the same backward expressions as in Sections 2.10–2.11.

---



### 2.7 Computation graph

A **computation graph** is a directed acyclic graph of operations: inputs $\to$ intermediate nodes (sums, products, nonlinearities) $\to$ output loss. A computation graph is a step-by-step map of the forward calculation, and backpropagation uses that map in reverse to apply the chain rule and compute gradients efficiently. Each step is a small operation like add, multiply, or apply a function like sigmoid.

For logistic regression (one example), a minimal graph is:

input 
x
x, weights 
w
w, bias 
b
b

compute 
z
=
w
⊤
x
+
b
z=w 
⊤
 x+b

compute 
a
=
σ
(
z
)
a=σ(z)

compute the loss 
L
(
y
,
a
)
L(y,a)

$$
\mathbf{x},\, \mathbf{w},\, b \;\Rightarrow\; z = \mathbf{w}^\top \mathbf{x} + b \;\Rightarrow\; a = \sigma(z) \;\Rightarrow\; \mathcal{L}(y, a).
$$

With batch training, the same graph is repeated for many examples, or written in vector form so you process the whole batch at once. 
Batch training repeats the same structure for each column of $\mathbf{X}$, or uses vectorized nodes for $\mathbf{Z}$ and $\mathbf{A}$. The structure is the same; only the shapes become matrices and vectors.

---

### 2.8 Derivatives with a computation graph

**Backpropagation** means walking backward through the graph and using the chain rule.traverses the graph backward: at each node, multiply local derivatives along paths (chain rule) and sum paths that merge.  If one variable affects the loss through more than one route, all those gradient contributions are added together.

For a node $u = f(v)$, the contribution to $\dfrac{\partial \mathcal{L}}{\partial v}$ is $\dfrac{\partial \mathcal{L}}{\partial u}\dfrac{\partial f}{\partial v}$.

---

### 2.9 Logistic regression: gradient descent (single example)

**Forward:** $z = \mathbf{w}^\top \mathbf{x} + b$, $a = \sigma(z)$,

$$
\mathcal{L} = - \bigl( y \log a + (1-y)\log(1-a) \bigr).
$$

**Useful intermediate:**

$$
\frac{\partial \mathcal{L}}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}, \qquad
\frac{\partial a}{\partial z} = a(1-a).
$$

**Chain rule for $z$:**

$$
\frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial a}\frac{\partial a}{\partial z}
= \Bigl(-\frac{y}{a} + \frac{1-y}{1-a}\Bigr) a(1-a) = a - y.
$$

**Gradients for parameters:**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{\partial \mathcal{L}}{\partial z}\,\mathbf{x} = (a - y)\,\mathbf{x}, \qquad
\frac{\partial \mathcal{L}}{\partial b} = a - y.
$$

---

### 2.10 Gradient descent on $m$ examples (fully vectorized)

The main idea is that the gradient is computed over all examples at once: for many training examples, compute all prediction errors together, turn them into gradients with matrix multiplication, and then update 
w
w and 
b
b in one step.

Define the column vector of errors (same shape as $\mathbf{Z}^\top$ if $\mathbf{Z}$ is $1 \times m$):

$$
\mathbf{dZ} = \mathbf{A} - \mathbf{y} \quad \text{(element-wise; align shapes as } \mathbb{R}^{m} \text{ or } \mathbb{R}^{1\times m} \text{ consistently)},
$$

where
- $Z$ is ...,
- $A$ is ...,
- $y$ is ...

**Gradients of the average cost** $J = \frac{1}{m}\sum_i \mathcal{L}^{(i)}$:

$$
\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m}\,\mathbf{X}\,\mathbf{dZ}^\top
\quad \in \mathbb{R}^{n_x},
\qquad
\frac{\partial J}{\partial b} = \frac{1}{m}\,\mathbf{1}_m^\top \mathbf{dZ}^\top
\quad \in \mathbb{R}.
$$

**Gradient descent step:**

$$
\mathbf{w} \leftarrow \mathbf{w} - \alpha \,\frac{\partial J}{\partial \mathbf{w}}, \qquad
b \leftarrow b - \alpha \,\frac{\partial J}{\partial b}.
$$

(If $\mathbf{dZ}$ is stored as a row vector $\mathbf{dZ} \in \mathbb{R}^{1\times m}$, use $\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m}\,\mathbf{X}\,\mathbf{dZ}^\top$ with matching transpose conventions.)

---





## Summary

| Item | Vectorized form |
|------|------------------|
| Linear logits | $\mathbf{Z} = \mathbf{w}^\top \mathbf{X} + b\,\mathbf{1}_m^\top$ |
| Activations | $\mathbf{A} = \sigma(\mathbf{Z})$ |
| Cost | $J = -\dfrac{1}{m}\sum_{i=1}^{m}\bigl( y^{(i)}\log a^{(i)} + (1-y^{(i)})\log(1-a^{(i)})\bigr)$ |
| Error signal | $\dfrac{\partial \mathcal{L}^{(i)}}{\partial z^{(i)}} = a^{(i)} - y^{(i)}$; batch: $\mathbf{dZ} = \mathbf{A} - \mathbf{y}$ |
| Gradients of $J$ | $\dfrac{\partial J}{\partial \mathbf{w}} = \dfrac{1}{m}\mathbf{X}\mathbf{dZ}^\top$, $\dfrac{\partial J}{\partial b} = \dfrac{1}{m}\mathbf{1}_m^\top\mathbf{dZ}^\top$ |

This is the **simplest** neural network: one layer, one activation, trained by gradient descent on a convex cost—yet it already contains forward pass, loss, backward pass, and vectorization used in deep networks.
