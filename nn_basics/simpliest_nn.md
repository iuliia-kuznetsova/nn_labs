# Neural Networks: Logistic Regression as the Simplest Network

Logistic Regression as the Simplest Neural Network basic:
- Computational graph of Logistic Regression;
- Binary classification;
- Logistic regression (vectorized);
- Logistic regression cost function (vectorized);
- Binary cross-entropy as a cost function;
- Gradient descent;
- Computation graph;
- Derivatives with a computation graph;
- Logistic regression: gradient descent (single example);
- Gradient descent on $m$ examples (fully vectorized);
- Summary

and interview-style pitfalls.

---

This note uses **vectorized** notation: one training step updates all parameters using full matrices for $m$ examples at once.

**Notation**

- $n_x$: number of features per example.
- $m$: number of examples (batch size).
- Design matrix $\mathbf{X} \in \mathbb{R}^{n_x \times m}$: each **column** is one example $\mathbf{x}^{(i)} \in \mathbb{R}^{n_x}$.
- Labels $\mathbf{y} \in \mathbb{R}^{1 \times m}$ (row vector), with $y^{(i)} \in \{0,1\}$ for binary classification.
- Weights $\mathbf{w} \in \mathbb{R}^{n_x}$ (column vector) and bias $b \in \mathbb{R}$.
- $\sigma(\cdot)$ is the sigmoid, applied **element-wise** to matrices.
- $\odot$ is the Hadamard (element-wise) product.
- $\mathbf{1}_m \in \mathbb{R}^{m}$ is a column vector of ones (subscript omitted when size is clear).

---

## 1. Computational graph of Logistic Regression

![Simplest NN graph](graphs\simplest_nn_graph.png)

### 1.1. Tricky interview questions

**Q1. Why can logistic regression be viewed as the simplest neural network?**  
It has one linear unit followed by a sigmoid activation, with no hidden layer.

**Q2. What does the graph show at a high level?**  
It shows the forward flow from inputs and parameters to a logit, then to a sigmoid output and loss.

**Q3. Does logistic regression learn nonlinear decision boundaries by itself?**  
No. It learns a linear decision boundary in the original feature space, unless the input features are transformed first.

---

## 2. Binary classification

**Goal.** Predict one of two classes, encoded as $y^{(i)} \in \{0, 1\}$.

**Probabilistic model.** Estimate the probability that output $y^{(i)}$ is of class $1$ given the input example $\mathbf{x}^{(i)}$:

$$
\hat{y}^{(i)} = P\bigl(y^{(i)} = 1 \mid \mathbf{x}^{(i)}\bigr), \qquad \hat{y}^{(i)} \in (0,1).
$$

**Decision rule.** Predict class $1$ if $\hat{y}^{(i)} \ge \tfrac{1}{2}$, else class $0$.

### 2.1. Tricky interview questions

**Q1. Why is the output written as a probability?**  
Because sigmoid maps the logit to a value in $(0,1)$, which can be interpreted as $P(y=1 \mid \mathbf{x})$.

**Q2. Is the threshold always required to be $0.5$?**  
No. $0.5$ is the default when classes and costs are balanced, but it can be changed for imbalanced data or different error costs.

**Q3. What does $y^{(i)} \in \{0,1\}$ mean?**  
It means each example has one binary label: class $0$ or class $1$.

---

## 3. Logistic regression (vectorized)

For one example,

$$
z^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + b, \qquad
\hat{y}^{(i)} = a^{(i)} = \sigma\bigl(z^{(i)}\bigr) = \frac{1}{1 + e^{-z^{(i)}}}.
$$

**All $m$ examples at once.** Stack columns in $\mathbf{X}$ and define

$$
\mathbf{Z} = \mathbf{w}^\top \mathbf{X} + b \,\mathbf{1}_m^\top \ \in \mathbb{R}^{1 \times m},
$$

$$
\mathbf{A} = \sigma(\mathbf{Z}) \quad \text{(element-wise)},
$$

where

- $\mathbf{Y}$ is the output (target) row vector of size $1 \times m$:

$$
\mathbf{Y} = \bigl[\, y^{(1)},\ y^{(2)},\ \dots,\ y^{(m)} \,\bigr] \in \mathbb{R}^{1 \times m}.
$$

- $\mathbf{w}^\top$ is the **transposed** weight vector of size $1 \times n_x$:

$$
\mathbf{w}^\top = \bigl[\, w_1,\ w_2,\ \dots,\ w_{n_x} \,\bigr] \in \mathbb{R}^{1 \times n_x}.
$$

- $\mathbf{X}$ is the input (features) matrix of size $n_x \times m$ (each column is one example):

$$
\mathbf{X} =
\begin{bmatrix}
\mid & \mid & & \mid \\
\mathbf{x}^{(1)} & \mathbf{x}^{(2)} & \cdots & \mathbf{x}^{(m)} \\
\mid & \mid & & \mid
\end{bmatrix}
\in \mathbb{R}^{n_x \times m}.
$$

- $b$ is a scalar bias **broadcast** across all columns of $\mathbf{w}^\top \mathbf{X}$ (equivalently, add the row vector $b\,\mathbf{1}_m^\top \in \mathbb{R}^{1 \times m}$).

- $\mathbf{Z} \in \mathbb{R}^{1 \times m}$ is the row vector of pre-activations (logits), one entry per example.

- $\mathbf{A} \in \mathbb{R}^{1 \times m}$ is the row vector of predicted probabilities, $\mathbf{A} = \sigma(\mathbf{Z})$.

- $\sigma$ is the sigmoid activation function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma(z) \in (0, 1),
$$

so that 
- If $z$ is a large positive number, $\sigma(z) \approx 1$.
- If $z$ is a large negative number, $\sigma(z) \approx 0$.
- If $z = 0$, $\sigma(z) = 0.5$.

### 3.1. Tricky interview questions

**Q1. Why is $\mathbf{X}$ shaped as $n_x \times m$?**  
Each column is one example, so multiplying $\mathbf{w}^\top \mathbf{X}$ produces one logit per example.

**Q2. Why does $b$ need broadcasting?**  
$b$ is a scalar, but the batch needs one bias term added to each example's logit.

**Q3. What is the shape of $\mathbf{A}$?**  
$\mathbf{A} \in \mathbb{R}^{1 \times m}$, with one predicted probability per training example.

---

## 4. Logistic regression cost function (vectorized)

**Loss function (binary cross-entropy, negative log-likelihood for Bernoulli labels).**

The loss function measures the discrepancy between the prediction $\hat{y}^{(i)}$ and the desired output $y^{(i)}$ **per example**. In other words, the loss function computes the error for a single training example $\mathbf{x}^{(i)}$.

$$
\mathcal{L}^{(i)} = - \Bigl( y^{(i)} \log a^{(i)} + \bigl(1 - y^{(i)}\bigr) \log\bigl(1 - a^{(i)}\bigr) \Bigr).
$$

- If $y^{(i)} = 1$: $\mathcal{L}^{(i)} = -\log a^{(i)}$, which is small when $a^{(i)}$ is close to $1$.
- If $y^{(i)} = 0$: $\mathcal{L}^{(i)} = -\log\bigl(1 - a^{(i)}\bigr)$, which is small when $a^{(i)}$ is close to $0$.

**Cost function.**

The cost function is the **average** of the loss over the entire training set (the whole batch). The goal of NN training is to find the parameters $\mathbf{w}$ and $b$ that minimize the overall cost function.

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

### 4.1. Tricky interview questions

**Q1. What is the difference between loss and cost?**  
Loss is computed for one example; cost is the average loss over the whole training set or batch.

**Q2. Why do we average by $m$?**  
Averaging makes the cost scale independent of the number of examples in the batch.

**Q3. Why are logs applied element-wise in the vector form?**  
Each prediction-label pair contributes its own binary cross-entropy term.

---

## 5. Binary cross-entropy as a cost function

**Cross-entropy** is a way to measure how different two probability distributions are. In machine learning, it usually measures how well a model's predicted probabilities match the true labels, so it is commonly used as a loss function for classification.

For discrete distributions $p$ and $q$, cross-entropy is

$$
H(p, q) = - \sum_{x} p(x) \log q(x),
$$

where $p$ is the **true** distribution and $q$ is the **predicted** distribution.

Logistic regression predicts a probability $a^{(i)} = P\bigl(y^{(i)} = 1 \mid \mathbf{x}^{(i)}\bigr)$. Since the true label $y^{(i)}$ is either $0$ or $1$, it makes sense to measure how likely the model thinks the true answer is. The cost function does exactly that by taking the **negative log-likelihood**.

For one example:

- if $y = 1$, the loss becomes $-\log(a)$;
- if $y = 0$, the loss becomes $-\log(1 - a)$.

The combined form is

$$
-\bigl( y \log a + (1 - y) \log(1 - a) \bigr).
$$

This is just a compact way to write both cases at once. If the correct class gets low probability, the loss becomes large.

**Probabilistic view.** Assume $y^{(i)} \sim \mathrm{Bernoulli}(a^{(i)})$ with $a^{(i)} = P(y^{(i)}=1\mid \mathbf{x}^{(i)})$. The **negative log-likelihood** for independent examples is

$$
-\frac{1}{m}\sum_{i=1}^{m} \log P\bigl(y^{(i)} \mid \mathbf{x}^{(i)}\bigr)
= -\frac{1}{m}\sum_{i=1}^{m} \Bigl( y^{(i)}\log a^{(i)} + (1-y^{(i)})\log(1-a^{(i)}) \Bigr),
$$

which is exactly $J$.

**Properties**

- **Convex** in $(\mathbf{w}, b)$ for logistic regression (sigmoid + cross-entropy), so gradient descent with a suitable learning rate $\alpha$ finds the global minimum under typical conditions.
  For logistic regression, sigmoid plus cross-entropy gives a convex objective in $(\mathbf{w}, b)$. That means optimization is much easier than with many other neural-network losses, because gradient descent is not fighting lots of bad local minima in this case.

- **Penalizes confident mistakes heavily**: if $y^{(i)} = 1$ but $a^{(i)} \approx 0$, then $-\log a^{(i)}$ is large. For example:
  - if the true label is $1$ and the model predicts $a = 0.99$, the loss is small;
  - if the true label is $1$ and the model predicts $a = 0.01$, the loss is huge.
  
  That is good behavior, because being confidently wrong should be punished more than being uncertain.

- **Matches outputs to probabilities** when trained with this loss, so $a^{(i)}$ is calibrated as an estimate of class probability under the model assumptions.
  Because the output is interpreted as a probability, the model is trained to make the predicted probability match the observed label. So after training, $a$ can be read as the model's estimate of class probability, not just a raw score.

### 5.1. Tricky interview questions

**Q1. Why not use squared error for logistic regression classification?**  
Binary cross-entropy matches the Bernoulli likelihood and gives a better optimization objective for probabilistic binary labels.

**Q2. Why does binary cross-entropy punish confident mistakes heavily?**  
Because $\log(a)$ or $\log(1-a)$ becomes very negative when the model assigns near-zero probability to the true class.

**Q3. Why is convexity important here?**  
For logistic regression, a convex objective means suitable gradient descent can find the global minimum rather than a bad local minimum.

---

## 6. Gradient descent

Gradient descent is used for NN training during backpropagation.

**Idea.** Update NN parameters $\mathbf{w}$ and $b$ in the direction **opposite** to the gradient of $J$:

$$
\mathbf{w} := \mathbf{w} - \alpha \,\frac{\partial J}{\partial \mathbf{w}}, \qquad
b := b - \alpha \,\frac{\partial J}{\partial b},
$$

where $\alpha > 0$ is the **learning rate**.

### 6.1. Tricky interview questions

**Q1. Why do we subtract the gradient?**  
The gradient points toward steepest increase, so subtracting it moves parameters toward lower cost.

**Q2. What happens if the learning rate is too large?**  
The updates may overshoot good parameter values and make the cost increase or diverge.

**Q3. What happens if the learning rate is too small?**  
Training can become very slow because each update changes the parameters only slightly.

---

## 7. Computation graph

A **computation graph** is a directed acyclic graph of operations: inputs $\to$ intermediate nodes (sums, products, nonlinearities) $\to$ output loss. A computation graph is a step-by-step map of the forward calculation, and backpropagation uses that map **in reverse** to apply the chain rule and compute gradients efficiently. Each step is a small operation like add, multiply, or apply a function like sigmoid.

For logistic regression (one example), a minimal graph is:

1. inputs $\mathbf{x}$, weights $\mathbf{w}$, bias $b$;
2. compute $z = \mathbf{w}^\top \mathbf{x} + b$;
3. compute $a = \sigma(z)$;
4. compute the loss $\mathcal{L}(y, a)$.

$$
\mathbf{x},\, \mathbf{w},\, b \ \Rightarrow\ z = \mathbf{w}^\top \mathbf{x} + b \ \Rightarrow\ a = \sigma(z) \ \Rightarrow\ \mathcal{L}(y, a).
$$

With batch training, the same graph is repeated for many examples, or written in vector form so you process the whole batch at once. Batch training repeats the same structure for each column of $\mathbf{X}$, or uses vectorized nodes for $\mathbf{Z}$ and $\mathbf{A}$. The structure is the same; only the shapes become matrices and vectors.

### 7.1. Tricky interview questions

**Q1. Why is a computation graph useful for backpropagation?**  
It records how each value was computed, so gradients can be propagated backward through the same operations.

**Q2. What are the nodes in the logistic regression graph?**  
Typical nodes are inputs, parameters, the linear score $z$, sigmoid activation $a$, and loss $\mathcal{L}$.

**Q3. Does vectorization change the computation graph conceptually?**  
No. It processes many examples at once, but the forward and backward dependencies are the same.

---

## 8. Derivatives with a computation graph

**Backpropagation** means walking **backward** through the graph and using the chain rule. At each node, multiply local derivatives along paths (chain rule) and **sum** paths that merge. If one variable affects the loss through more than one route, all those gradient contributions are added together.

For a node $u = f(v)$, the contribution to $\dfrac{\partial \mathcal{L}}{\partial v}$ is

$$
\frac{\partial \mathcal{L}}{\partial v} \mathrel{+}= \frac{\partial \mathcal{L}}{\partial u}\,\frac{\partial f}{\partial v}.
$$

### 8.1. Tricky interview questions

**Q1. Why does backpropagation use the chain rule?**  
The loss depends on parameters through intermediate variables, so gradients must be multiplied through those dependencies.

**Q2. Why do gradient contributions get summed?**  
If a variable affects the loss through multiple paths, the total derivative is the sum of all path contributions.

**Q3. What is a local derivative?**  
It is the derivative of one graph operation with respect to one of its direct inputs.

---

## 9. Logistic regression: gradient descent (single example)

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

### 9.1. Tricky interview questions

**Q1. Why does $\frac{\partial \mathcal{L}}{\partial z}$ simplify to $a-y$?**  
For sigmoid plus binary cross-entropy, the derivative terms cancel neatly, leaving prediction minus label.

**Q2. What does $a-y$ represent intuitively?**  
It is the prediction error for one example in probability space.

**Q3. Why is the weight gradient proportional to $\mathbf{x}$?**  
Each weight controls the logit through its matching input feature, so the error is scaled by that feature value.

---

## 10. Gradient descent on $m$ examples (fully vectorized)

The main idea is that the gradient is computed over **all examples at once**: compute all prediction errors together, turn them into gradients with matrix multiplication, and then update $\mathbf{w}$ and $b$ in one step.

Define the row vector of errors (same shape as $\mathbf{Z}$, i.e. $1 \times m$):

$$
\mathbf{dZ} = \mathbf{A} - \mathbf{y} \ \in \mathbb{R}^{1 \times m},
$$

where

- $\mathbf{Z} \in \mathbb{R}^{1 \times m}$ is the row vector of logits $z^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + b$,
- $\mathbf{A} = \sigma(\mathbf{Z}) \in \mathbb{R}^{1 \times m}$ is the row vector of predicted probabilities,
- $\mathbf{y} \in \mathbb{R}^{1 \times m}$ is the row vector of true labels.

**Gradients of the average cost** $J = \dfrac{1}{m}\sum_i \mathcal{L}^{(i)}$:

$$
\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m}\,\mathbf{X}\,\mathbf{dZ}^\top
\ \in \mathbb{R}^{n_x},
\qquad
\frac{\partial J}{\partial b} = \frac{1}{m}\,\mathbf{dZ}\,\mathbf{1}_m
= \frac{1}{m}\sum_{i=1}^{m} \bigl(a^{(i)} - y^{(i)}\bigr)
\ \in \mathbb{R}.
$$

**Gradient descent step:**

$$
\mathbf{w} \leftarrow \mathbf{w} - \alpha \,\frac{\partial J}{\partial \mathbf{w}}, \qquad
b \leftarrow b - \alpha \,\frac{\partial J}{\partial b}.
$$

### 10.1. Tricky interview questions

**Q1. Why is $\mathbf{dZ} = \mathbf{A} - \mathbf{y}$?**  
It stacks the single-example derivative $a^{(i)} - y^{(i)}$ for all examples in the batch.

**Q2. Why does $\mathbf{X}\mathbf{dZ}^\top$ produce the weight gradient shape?**  
$\mathbf{X}$ has shape $n_x \times m$ and $\mathbf{dZ}^\top$ has shape $m \times 1$, so the result has one gradient per feature.

**Q3. Why is the bias gradient the average of all errors?**  
The same scalar bias contributes to every example's logit, so its gradient sums all example errors and averages them.

---

## 11. Summary

Logistic regression is the **simplest** neural network: one layer, one activation, trained by gradient descent on a convex cost — yet it already contains forward pass, loss, backward pass, and vectorization used in deep networks.

| Item | Vectorized form |
|------|------------------|
| Linear logits | $\mathbf{Z} = \mathbf{w}^\top \mathbf{X} + b\,\mathbf{1}_m^\top$ |
| Activations | $\mathbf{A} = \sigma(\mathbf{Z})$ |
| Cost | $J = -\dfrac{1}{m}\sum_{i=1}^{m}\bigl( y^{(i)}\log a^{(i)} + (1-y^{(i)})\log(1-a^{(i)})\bigr)$ |
| Error signal | $\dfrac{\partial \mathcal{L}^{(i)}}{\partial z^{(i)}} = a^{(i)} - y^{(i)}$; batch: $\mathbf{dZ} = \mathbf{A} - \mathbf{y}$ |
| Gradients of $J$ | $\dfrac{\partial J}{\partial \mathbf{w}} = \dfrac{1}{m}\mathbf{X}\,\mathbf{dZ}^\top$, $\dfrac{\partial J}{\partial b} = \dfrac{1}{m}\mathbf{dZ}\,\mathbf{1}_m$ |

### 11.1. Tricky interview questions

**Q1. What are the four main pieces of logistic regression training?**  
Compute logits, apply sigmoid, compute binary cross-entropy cost, and update parameters using gradients.

**Q2. What is the key error signal used in backpropagation?**  
For each example it is $a^{(i)} - y^{(i)}$; in vectorized form it is $\mathbf{dZ} = \mathbf{A} - \mathbf{y}$.

**Q3. What makes this model a good first neural-network example?**  
It contains forward propagation, loss, backpropagation, gradient descent, and vectorization in the simplest possible setting.


