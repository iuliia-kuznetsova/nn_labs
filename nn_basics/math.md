# Mathematics for Data Science

---

## Table of Contents

1. [Calculus](#1-calculus)
2. [Linear Algebra](#2-linear-algebra)
3. [Probability Theory](#3-probability-theory)
4. [Mathematical Statistics](#4-mathematical-statistics)
5. [Information Theory](#5-information-theory)
6. [Optimization](#6-optimization)

---

## 1. Calculus

### 1.1 Limits

$$\lim_{x \to a} f(x) = L$$

The limit exists if and only if the left-hand and right-hand limits are equal:

$$\lim_{x \to a^-} f(x) = \lim_{x \to a^+} f(x) = L$$

**L'Hôpital's Rule** — for indeterminate forms $\frac{0}{0}$ or $\frac{\infty}{\infty}$:

$$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$

---

### 1.2 Derivative Definition

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

The derivative measures the instantaneous rate of change of $f$ at $x$.

---

### 1.3 Basic Derivative Rules

| Rule | Formula |
|---|---|
| Constant | $(c)' = 0$ |
| Power | $(x^n)' = n x^{n-1}$ |
| Sum | $(f + g)' = f' + g'$ |
| Difference | $(f - g)' = f' - g'$ |
| Product | $(fg)' = f'g + fg'$ |
| Quotient | $\left(\dfrac{f}{g}\right)' = \dfrac{f'g - fg'}{g^2}$ |
| Scalar multiple | $(cf)' = c f'$ |
| **Chain rule** | $(f(g(x)))' = f'(g(x)) \cdot g'(x)$ |

---

### 1.4 Common Derivatives

| Function | Derivative |
|---|---|
| $x^n$ | $n x^{n-1}$ |
| $e^x$ | $e^x$ |
| $e^{ax}$ | $a e^{ax}$ |
| $a^x$ | $a^x \ln a$ |
| $\ln x$ | $\dfrac{1}{x}$ |
| $\log_a x$ | $\dfrac{1}{x \ln a}$ |
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x = \dfrac{1}{\cos^2 x}$ |
| $\sigma(x) = \dfrac{1}{1+e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ |
| $\tanh x$ | $1 - \tanh^2 x$ |
| $\text{ReLU}(x) = \max(0, x)$ | $\mathbf{1}[x > 0]$ |
| $\sqrt{x}$ | $\dfrac{1}{2\sqrt{x}}$ |
| $\dfrac{1}{x}$ | $-\dfrac{1}{x^2}$ |

---

### 1.5 Chain Rule (extended)

For a composite function $f(g(h(x)))$:

$$\frac{d}{dx} f(g(h(x))) = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

This is the backbone of backpropagation in neural networks.

**Example** — derivative of $\sigma(w^\top x + b)$ with respect to $w_j$:

$$\frac{\partial}{\partial w_j} \sigma(w^\top x + b) = \sigma(z)(1 - \sigma(z)) \cdot x_j, \quad z = w^\top x + b$$

---

### 1.6 Partial Derivatives

For $f: \mathbb{R}^n \to \mathbb{R}$, the partial derivative with respect to $x_i$ treats all other variables as constants:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

**Example:** $f(x, y) = x^2 y + e^y$

$$\frac{\partial f}{\partial x} = 2xy, \qquad \frac{\partial f}{\partial y} = x^2 + e^y$$

---

### 1.7 Gradient

The gradient of $f: \mathbb{R}^n \to \mathbb{R}$ is a vector of all partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

- The gradient points in the direction of **steepest ascent**.
- $-\nabla f$ points toward steepest descent — the basis of gradient descent.
- $\nabla f(\mathbf{x}^*) = \mathbf{0}$ is a necessary condition for a local extremum.

---

### 1.8 Jacobian

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is an $m \times n$ matrix of all first-order partial derivatives:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}$$

$J_{ij} = \frac{\partial f_i}{\partial x_j}$. The Jacobian generalizes the chain rule to vector functions.

---

### 1.9 Hessian

The Hessian of $f: \mathbb{R}^n \to \mathbb{R}$ is an $n \times n$ matrix of second-order partial derivatives:

$$H = \nabla^2 f(\mathbf{x}), \qquad H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

The Hessian is symmetric (Schwarz's theorem, under mild conditions). It describes the **curvature** of $f$:

- $H \succ 0$ (positive definite) → local minimum
- $H \prec 0$ (negative definite) → local maximum
- Mixed eigenvalues → saddle point

---

### 1.10 Higher-Order Derivatives

| Order | Notation | Information |
|---|---|---|
| 1st | $f'(x)$, $\frac{df}{dx}$ | Slope / rate of change |
| 2nd | $f''(x)$, $\frac{d^2f}{dx^2}$ | Curvature; convexity if $f'' \geq 0$ |
| $n$-th | $f^{(n)}(x)$ | Used in Taylor expansions |

---

### 1.11 Taylor Series

Approximates a smooth function around a point $a$:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x - a)^n = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

**First-order (linear) approximation** — used to justify gradient descent:

$$f(\mathbf{x} + \delta) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \delta$$

**Second-order (quadratic) approximation:**

$$f(\mathbf{x} + \delta) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \delta + \frac{1}{2} \delta^\top H \delta$$

**Common Taylor expansions** around $x = 0$:

$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$$

$$\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots \quad (|x| < 1)$$

$$\frac{1}{1-x} = 1 + x + x^2 + x^3 + \cdots \quad (|x| < 1)$$

---

### 1.12 Integration

**Definite integral** — area under the curve:

$$\int_a^b f(x)\, dx = F(b) - F(a), \quad \text{where } F'(x) = f(x)$$

**Fundamental Theorem of Calculus:**

$$\frac{d}{dx} \int_a^x f(t)\, dt = f(x)$$

**Common integrals:**

| Function | Integral |
|---|---|
| $x^n$ | $\dfrac{x^{n+1}}{n+1} + C \quad (n \neq -1)$ |
| $\dfrac{1}{x}$ | $\ln|x| + C$ |
| $e^x$ | $e^x + C$ |
| $e^{ax}$ | $\dfrac{1}{a}e^{ax} + C$ |
| $\sin x$ | $-\cos x + C$ |
| $\cos x$ | $\sin x + C$ |

**Integration by parts:**

$$\int u\, dv = uv - \int v\, du$$

**Substitution:** if $x = g(u)$, then $dx = g'(u)\, du$.

---

## 2. Linear Algebra

### 2.1 Vectors

A vector $\mathbf{v} \in \mathbb{R}^n$ is an ordered list of $n$ real numbers:

$$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

**Dot product (inner product):**

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^\top \mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

**Norms:**

| Norm | Formula | Use case |
|---|---|---|
| $L^1$ | $\|\mathbf{v}\|_1 = \sum_i |v_i|$ | Sparsity (Lasso) |
| $L^2$ (Euclidean) | $\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}$ | Standard distance |
| $L^\infty$ | $\|\mathbf{v}\|_\infty = \max_i |v_i|$ | Worst-case bound |
| $L^p$ | $\|\mathbf{v}\|_p = \left(\sum_i |v_i|^p\right)^{1/p}$ | General case |

```python
import numpy as np

v = np.array([1.0, 2.0, 3.0])
u = np.array([4.0, 5.0, 6.0])

dot   = np.dot(v, u)           # dot product
l1    = np.linalg.norm(v, 1)   # L1 norm
l2    = np.linalg.norm(v)      # L2 norm (default)
linf  = np.linalg.norm(v, np.inf)
```

---

### 2.2 Matrices

A matrix $A \in \mathbb{R}^{m \times n}$ has $m$ rows and $n$ columns:

$$A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}$$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)
```

---

### 2.3 Matrix Operations

#### Transpose

$(A^\top)_{ij} = A_{ji}$ — flips rows and columns.

$$A \in \mathbb{R}^{m \times n} \implies A^\top \in \mathbb{R}^{n \times m}$$

Properties: $(AB)^\top = B^\top A^\top$, $(A^\top)^\top = A$.

```python
A.T                    # transpose
np.transpose(A)        # equivalent
```

#### Addition and Scalar Multiplication

$$A + B, \quad cA \quad \text{(elementwise)}$$

```python
A + B
c * A
```

#### Matrix Multiplication

$(AB)_{ij} = \sum_k A_{ik} B_{kj}$ — requires inner dimensions to match: $(m \times k)(k \times n) \to (m \times n)$.

**Not commutative:** $AB \neq BA$ in general.

```python
A @ B                  # matrix multiply (recommended)
np.dot(A, B)           # equivalent
np.matmul(A, B)        # equivalent
```

#### Elementwise (Hadamard) Product

$$C_{ij} = A_{ij} \cdot B_{ij} \quad \text{(same shape required)}$$

```python
A * B                  # elementwise multiply
```

#### Broadcasting

NumPy automatically expands dimensions of size 1 to match:

```python
A = np.ones((3, 4))
b = np.ones((4,))
A + b   # b is broadcast along rows → result shape (3, 4)
```

---

### 2.4 Special Matrices

| Matrix | Definition | Notes |
|---|---|---|
| Square | $m = n$ | |
| Identity $I_n$ | $I_{ij} = \mathbf{1}[i=j]$ | $AI = IA = A$ |
| Diagonal $D$ | $D_{ij} = 0$ for $i \neq j$ | |
| Symmetric | $A = A^\top$ | Covariance matrices are symmetric |
| Orthogonal | $A^\top A = I$, so $A^{-1} = A^\top$ | Columns are orthonormal |
| Positive semi-definite (PSD) | $\mathbf{x}^\top A \mathbf{x} \geq 0$ for all $\mathbf{x}$ | Eigenvalues $\geq 0$ |
| Positive definite (PD) | $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq 0$ | Eigenvalues $> 0$ |

```python
np.eye(n)              # n×n identity
np.diag([1, 2, 3])    # diagonal matrix
np.zeros((m, n))
np.ones((m, n))
```

---

### 2.5 Matrix Inverse

For square $A$, the inverse $A^{-1}$ satisfies $A A^{-1} = A^{-1} A = I$.

Exists iff $A$ is **non-singular** (full rank, $\det A \neq 0$).

Properties:
- $(AB)^{-1} = B^{-1} A^{-1}$
- $(A^\top)^{-1} = (A^{-1})^\top$

```python
np.linalg.inv(A)             # inverse (use carefully — numerically unstable)
np.linalg.solve(A, b)        # solve Ax = b (preferred over inv(A) @ b)
np.linalg.pinv(A)            # Moore-Penrose pseudo-inverse
```

---

### 2.6 Determinant and Trace

**Determinant** — scalar summarizing the matrix; zero for singular matrices:

$$\det(A), \quad |A|$$

$$\det(AB) = \det(A)\det(B), \qquad \det(A^\top) = \det(A), \qquad \det(A^{-1}) = \frac{1}{\det(A)}$$

**Trace** — sum of diagonal elements:

$$\text{tr}(A) = \sum_i a_{ii} = \sum_i \lambda_i \quad \text{(sum of eigenvalues)}$$

$$\text{tr}(AB) = \text{tr}(BA), \qquad \text{tr}(A^\top) = \text{tr}(A)$$

```python
np.linalg.det(A)
np.trace(A)
```

---

### 2.7 Rank and Null Space

**Rank** — number of linearly independent rows (= columns for full-rank matrices):

$$\text{rank}(A) \leq \min(m, n)$$

- Full column rank: columns are linearly independent → $A^\top A$ is invertible.
- Full row rank: rows are linearly independent → $A A^\top$ is invertible.

**Null space (kernel):** $\text{null}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$

```python
np.linalg.matrix_rank(A)
```

---

### 2.8 Eigenvalues and Eigenvectors

For square $A$, $\mathbf{v} \neq \mathbf{0}$ is an eigenvector with eigenvalue $\lambda$ if:

$$A \mathbf{v} = \lambda \mathbf{v}$$

Eigenvalues are roots of the **characteristic polynomial**: $\det(A - \lambda I) = 0$.

**Eigendecomposition** (for diagonalizable $A$):

$$A = Q \Lambda Q^{-1}$$

where $Q$ contains eigenvectors as columns and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

For symmetric $A$: eigenvectors are orthogonal → $A = Q \Lambda Q^\top$.

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
eigenvalues, eigenvectors = np.linalg.eigh(A)   # for symmetric/Hermitian (more stable)
```

---

### 2.9 Singular Value Decomposition (SVD)

Any $A \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$\boxed{A = U \Sigma V^\top}$$

- $U \in \mathbb{R}^{m \times m}$ — orthogonal, left singular vectors (columns are eigenvectors of $AA^\top$)
- $\Sigma \in \mathbb{R}^{m \times n}$ — diagonal with non-negative singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{n \times n}$ — orthogonal, right singular vectors (columns are eigenvectors of $A^\top A$)

**Relationship to eigenvalues:** $\sigma_i = \sqrt{\lambda_i(A^\top A)}$.

**Truncated SVD** (rank-$k$ approximation — best by Frobenius norm):

$$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

Applications: PCA, dimensionality reduction, recommendation systems, image compression.

```python
U, s, Vt = np.linalg.svd(A)               # full SVD; s is 1-D array of singular values
U, s, Vt = np.linalg.svd(A, full_matrices=False)  # economy / thin SVD

# Reconstruct from full SVD
A_reconstructed = U @ np.diag(s) @ Vt

# Rank-k approximation
k = 5
A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

---

### 2.10 Matrix Norms

| Norm | Formula | Notes |
|---|---|---|
| Frobenius | $\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\text{tr}(A^\top A)}$ | Most common in ML |
| Spectral (operator) | $\|A\|_2 = \sigma_{\max}(A)$ | Largest singular value |
| Nuclear | $\|A\|_* = \sum_i \sigma_i$ | Sum of singular values; promotes low rank |

```python
np.linalg.norm(A, 'fro')      # Frobenius
np.linalg.norm(A, 2)          # spectral
np.linalg.norm(A, 'nuc')      # nuclear
```

---

### 2.11 Linear Systems

System $A\mathbf{x} = \mathbf{b}$:

- **Unique solution** if $A$ is square and full rank.
- **Least-squares solution** (overdetermined, $m > n$): minimizes $\|A\mathbf{x} - \mathbf{b}\|_2^2$.

$$\mathbf{x}^* = (A^\top A)^{-1} A^\top \mathbf{b} = A^\dagger \mathbf{b}$$

```python
np.linalg.solve(A, b)          # exact solution (square, full-rank A)
np.linalg.lstsq(A, b, rcond=None)  # least-squares solution
```

---

### 2.12 Useful Matrix Identities

**Matrix inversion lemma (Woodbury):**

$$(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$$

**Quadratic form:** $\mathbf{x}^\top A \mathbf{x} = \sum_{i,j} a_{ij} x_i x_j$

**Gradient of quadratic form** (for symmetric $A$):

$$\nabla_{\mathbf{x}} (\mathbf{x}^\top A \mathbf{x}) = 2A\mathbf{x}$$

$$\nabla_{\mathbf{x}} (\mathbf{a}^\top \mathbf{x}) = \mathbf{a}, \qquad \nabla_{\mathbf{x}} (\mathbf{x}^\top \mathbf{x}) = 2\mathbf{x}$$

---

## 3. Probability Theory

### 3.1 Sample Space and Events

- **Sample space** $\Omega$ — set of all possible outcomes.
- **Event** $A \subseteq \Omega$ — a subset of outcomes.
- **Probability measure** $P$: assigns $P(A) \in [0, 1]$ to events.

**Axioms (Kolmogorov):**
1. $P(A) \geq 0$
2. $P(\Omega) = 1$
3. For mutually exclusive events: $P(A \cup B) = P(A) + P(B)$

**Useful identities:**

$$P(A^c) = 1 - P(A)$$

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

---

### 3.2 Conditional Probability

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Independence:** $A \perp B$ iff $P(A \cap B) = P(A)P(B)$, equivalently $P(A \mid B) = P(A)$.

**Law of Total Probability** — for a partition $\{B_1, \ldots, B_n\}$ of $\Omega$:

$$P(A) = \sum_{i=1}^n P(A \mid B_i) P(B_i)$$

---

### 3.3 Bayes' Theorem

$$\boxed{P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}}$$

In machine learning terms:

$$\text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}$$

**Bayes' theorem (continuous form):**

$$p(\theta \mid x) = \frac{p(x \mid \theta)\, p(\theta)}{p(x)} = \frac{p(x \mid \theta)\, p(\theta)}{\int p(x \mid \theta)\, p(\theta)\, d\theta}$$

---

### 3.4 Random Variables

A **random variable** $X$ is a function $X: \Omega \to \mathbb{R}$.

- **Discrete** — takes countable values; described by a **probability mass function (PMF)**:
$$P(X = x) = p(x), \quad \sum_x p(x) = 1$$

- **Continuous** — takes values in $\mathbb{R}$; described by a **probability density function (PDF)**:
$$P(a \leq X \leq b) = \int_a^b f(x)\, dx, \quad \int_{-\infty}^{\infty} f(x)\, dx = 1$$

**Cumulative distribution function (CDF):**

$$F(x) = P(X \leq x)$$

For continuous: $F'(x) = f(x)$.

---

### 3.5 Expectation and Variance

**Expected value (mean):**

$$\mathbb{E}[X] = \begin{cases} \sum_x x\, p(x) & \text{discrete} \\ \int_{-\infty}^{\infty} x\, f(x)\, dx & \text{continuous} \end{cases}$$

**Properties:**
- $\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b$ (linearity)
- $\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ (always)
- $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ if $X \perp Y$
- $\mathbb{E}[g(X)] = \int g(x) f(x)\, dx$ (law of the unconscious statistician)

**Variance:**

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Properties:**
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ if $X \perp Y$

**Standard deviation:** $\sigma = \sqrt{\text{Var}(X)}$

---

### 3.6 Covariance and Correlation

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

$$\text{Corr}(X, Y) = \rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$

**Covariance matrix** for a random vector $\mathbf{X} = (X_1, \ldots, X_n)^\top$:

$$\Sigma = \text{Cov}(\mathbf{X}) = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top], \qquad \Sigma_{ij} = \text{Cov}(X_i, X_j)$$

$\Sigma$ is symmetric and positive semi-definite.

---

### 3.7 Common Distributions

#### Discrete

| Distribution | PMF | Mean | Variance | Use |
|---|---|---|---|---|
| **Bernoulli** $\text{Ber}(p)$ | $P(X=1)=p$, $P(X=0)=1-p$ | $p$ | $p(1-p)$ | Binary outcome |
| **Binomial** $\text{Bin}(n,p)$ | $\binom{n}{k} p^k (1-p)^{n-k}$ | $np$ | $np(1-p)$ | Count of successes |
| **Poisson** $\text{Poi}(\lambda)$ | $e^{-\lambda}\lambda^k / k!$ | $\lambda$ | $\lambda$ | Event counts, rare events |
| **Geometric** $\text{Geo}(p)$ | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ | Trials until first success |
| **Categorical** | $P(X=k)=p_k$ | — | — | Multi-class classification |

#### Continuous

| Distribution | PDF | Mean | Variance | Use |
|---|---|---|---|---|
| **Uniform** $\mathcal{U}(a,b)$ | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | Priors, sampling |
| **Normal** $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | Noise, CLT, Gaussian models |
| **Standard Normal** $\mathcal{N}(0,1)$ | $\frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ | $0$ | $1$ | Z-scores |
| **Exponential** $\text{Exp}(\lambda)$ | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | Waiting times |
| **Beta** $\text{Beta}(\alpha,\beta)$ | $\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ | $\frac{\alpha}{\alpha+\beta}$ | — | Prior for probability |
| **Gamma** $\Gamma(\alpha, \beta)$ | — | $\alpha/\beta$ | $\alpha/\beta^2$ | Bayesian, wait times |
| **Chi-squared** $\chi^2(k)$ | — | $k$ | $2k$ | Hypothesis testing |
| **Student's $t$** $t(k)$ | — | $0$ | $\frac{k}{k-2}$ | Small-sample inference |
| **Laplace** $\text{Lap}(\mu, b)$ | $\frac{1}{2b}e^{-|x-\mu|/b}$ | $\mu$ | $2b^2$ | L1 regularization prior |

---

### 3.8 Multivariate Normal Distribution

$$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$$

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

Properties:
- Marginals and conditionals of a multivariate normal are normal.
- Linear transformations: if $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$, then $A\mathbf{X} + \mathbf{b} \sim \mathcal{N}(A\boldsymbol{\mu}+\mathbf{b}, A\Sigma A^\top)$.
- $X \perp Y$ iff $\text{Cov}(X, Y) = 0$ (only for jointly normal variables).

---

### 3.9 Moment Generating Function (MGF) and Characteristic Function

**MGF:** $M_X(t) = \mathbb{E}[e^{tX}]$

$$\mathbb{E}[X^n] = M_X^{(n)}(0)$$

**Characteristic function:** $\phi_X(t) = \mathbb{E}[e^{itX}]$ — always exists.

---

### 3.10 Inequalities

**Markov's inequality** ($X \geq 0$):

$$P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}$$

**Chebyshev's inequality:**

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

**Jensen's inequality** (for convex $\phi$):

$$\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)]$$

**Cauchy-Schwarz:**

$$|\mathbb{E}[XY]|^2 \leq \mathbb{E}[X^2]\,\mathbb{E}[Y^2]$$

---

### 3.11 Law of Large Numbers and Central Limit Theorem

**Law of Large Numbers (LLN):** For i.i.d. $X_1, \ldots, X_n$ with mean $\mu$:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{p} \mu \quad \text{as } n \to \infty$$

**Central Limit Theorem (CLT):** For i.i.d. $X_i$ with mean $\mu$ and variance $\sigma^2$:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

The sum of many independent random variables approaches a normal distribution regardless of the original distribution — the foundation for confidence intervals and hypothesis tests.

---

## 4. Mathematical Statistics

### 4.1 Population vs. Sample

| | Population | Sample |
|---|---|---|
| Size | $N$ | $n$ |
| Mean | $\mu$ | $\bar{x} = \frac{1}{n}\sum x_i$ |
| Variance | $\sigma^2$ | $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ |
| Proportion | $p$ | $\hat{p}$ |

The **sample variance** uses $n-1$ (Bessel's correction) to get an unbiased estimator of $\sigma^2$.

---

### 4.2 Estimators

An **estimator** $\hat{\theta}$ is a function of sample data that approximates a population parameter $\theta$.

**Bias:** $\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$

**Mean Squared Error:**

$$\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2] = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2$$

| Property | Definition |
|---|---|
| **Unbiased** | $\mathbb{E}[\hat{\theta}] = \theta$ |
| **Consistent** | $\hat{\theta} \xrightarrow{p} \theta$ as $n \to \infty$ |
| **Efficient** | Achieves minimum variance among unbiased estimators (Cramér-Rao bound) |

**Cramér-Rao Lower Bound:**

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}, \qquad I(\theta) = \mathbb{E}\!\left[\left(\frac{\partial \ln p(X;\theta)}{\partial \theta}\right)^2\right]$$

where $I(\theta)$ is the **Fisher information**.

---

### 4.3 Maximum Likelihood Estimation (MLE)

Given i.i.d. observations $\{x^{(1)}, \ldots, x^{(m)}\}$ from a model $p(x; \theta)$:

**Likelihood:**

$$\mathcal{L}(\theta) = \prod_{i=1}^m p(x^{(i)}; \theta)$$

**Log-likelihood** (easier to maximize; $\log$ is monotone):

$$\ell(\theta) = \ln \mathcal{L}(\theta) = \sum_{i=1}^m \ln p(x^{(i)}; \theta)$$

**MLE:**

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta)$$

Solved by setting $\nabla_\theta \ell(\theta) = 0$ and solving (often via gradient ascent).

**MLE examples:**

- Normal: $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{m}\sum(x_i - \bar{x})^2$
- Bernoulli: $\hat{p} = \bar{x}$

**Connection to loss functions:** minimizing cross-entropy = maximizing log-likelihood for classification.

---

### 4.4 Maximum A Posteriori (MAP) Estimation

Incorporates a prior $p(\theta)$:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\ln p(\mathbf{x} \mid \theta) + \ln p(\theta)\right]$$

- Gaussian prior $\to$ L2 regularization (Ridge)
- Laplace prior $\to$ L1 regularization (Lasso)

---

### 4.5 Confidence Intervals

A $(1-\alpha)$ **confidence interval** for $\mu$ when $\sigma$ is known:

$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

When $\sigma$ is unknown (use sample $s$, $t$-distribution with $n-1$ degrees of freedom):

$$\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

**Common critical values:**

| Confidence level | $z_{\alpha/2}$ |
|---|---|
| 90% | 1.645 |
| 95% | 1.960 |
| 99% | 2.576 |

**Interpretation:** If we repeated the procedure many times, $(1-\alpha)$ of constructed intervals would contain the true $\mu$.

---

### 4.6 Hypothesis Testing

**Setup:**
- $H_0$: null hypothesis (e.g., $\mu = 0$)
- $H_1$: alternative hypothesis (e.g., $\mu \neq 0$)

**Errors:**

| | $H_0$ true | $H_0$ false |
|---|---|---|
| Reject $H_0$ | Type I error (rate $\alpha$) | Correct (power $= 1-\beta$) |
| Fail to reject $H_0$ | Correct | Type II error (rate $\beta$) |

**p-value:** probability of observing a test statistic at least as extreme as the one computed, assuming $H_0$ is true.

$$p\text{-value} < \alpha \implies \text{reject } H_0$$

**Common tests:**

| Test | When to use | Statistic |
|---|---|---|
| One-sample $z$-test | Known $\sigma$, large $n$ | $z = \frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}$ |
| One-sample $t$-test | Unknown $\sigma$ | $t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}$ |
| Two-sample $t$-test | Compare two means | $t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{1/n_1 + 1/n_2}}$ |
| Chi-squared test | Categorical data / independence | $\chi^2 = \sum \frac{(O-E)^2}{E}$ |
| F-test / ANOVA | Compare variances or multiple means | $F = \frac{s_1^2}{s_2^2}$ |

---

### 4.7 Regression

**Simple linear regression:**

$$y = \beta_0 + \beta_1 x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

OLS estimates (minimize $\sum (y_i - \hat{y}_i)^2$):

$$\hat{\beta}_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{\text{Cov}(x,y)}{\text{Var}(x)}, \qquad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

**Multiple linear regression ($\mathbf{X} \in \mathbb{R}^{m \times (n+1)}$, with intercept column):**

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

$$\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

**Regularized regression:**

| Method | Loss | Effect |
|---|---|---|
| Ridge (L2) | $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_2^2$ | Shrinks coefficients; closed form |
| Lasso (L1) | $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_1$ | Sparse coefficients; feature selection |
| Elastic Net | L2 loss + $\lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2\|\boldsymbol{\beta}\|_2^2$ | Combines both |

**Ridge closed form:**

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^\top \mathbf{X} + \lambda I)^{-1} \mathbf{X}^\top \mathbf{y}$$

**Coefficient of determination ($R^2$):**

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} \in [0, 1]$$

---

### 4.8 Evaluation Metrics

#### Regression

$$\text{MSE} = \frac{1}{m}\sum_{i=1}^m (y_i - \hat{y}_i)^2$$

$$\text{RMSE} = \sqrt{\text{MSE}}$$

$$\text{MAE} = \frac{1}{m}\sum_{i=1}^m |y_i - \hat{y}_i|$$

#### Classification (Confusion Matrix)

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\,TP}{2\,TP + FP + FN}$$

$$F_\beta = (1+\beta^2)\frac{\text{Precision} \cdot \text{Recall}}{\beta^2\,\text{Precision} + \text{Recall}}$$

**ROC-AUC:** area under the Receiver Operating Characteristic curve (TPR vs. FPR). AUC = 1 is perfect; AUC = 0.5 is random.

---

### 4.9 Bayesian Inference

Update belief about $\theta$ after observing data $\mathbf{x}$:

$$p(\theta \mid \mathbf{x}) \propto p(\mathbf{x} \mid \theta)\, p(\theta)$$

**Conjugate priors** — posterior is in the same family as the prior:

| Likelihood | Conjugate Prior | Posterior |
|---|---|---|
| Bernoulli/Binomial | Beta | Beta |
| Normal (known $\sigma$) | Normal | Normal |
| Poisson | Gamma | Gamma |

---

## 5. Information Theory

### 5.1 Entropy

**Shannon entropy** — measures average uncertainty (bits when $\log_2$, nats when $\ln$):

$$H(X) = -\sum_x p(x) \log p(x) = -\mathbb{E}[\log p(X)]$$

For continuous $X$ (**differential entropy**):

$$H(X) = -\int f(x) \ln f(x)\, dx$$

Properties:
- $H(X) \geq 0$
- $H(X)$ is maximized by uniform distribution
- For a Gaussian $\mathcal{N}(\mu, \sigma^2)$: $H = \frac{1}{2}\ln(2\pi e \sigma^2)$

---

### 5.2 Cross-Entropy

$$H(p, q) = -\sum_x p(x) \log q(x) = -\mathbb{E}_p[\log q(X)]$$

In classification with true labels $y$ and predicted probabilities $\hat{y}$:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})\right] \quad \text{(binary)}$$

$$\mathcal{L}_{\text{CE}} = -\frac{1}{m}\sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log \hat{y}_k^{(i)} \quad \text{(multi-class)}$$

Minimizing cross-entropy = maximizing log-likelihood.

---

### 5.3 KL Divergence

**Kullback-Leibler divergence** — measures how much distribution $q$ differs from $p$:

$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_p\!\left[\log\frac{p(X)}{q(X)}\right]$$

Properties:
- $D_{\text{KL}}(p \| q) \geq 0$ (Gibbs' inequality)
- $D_{\text{KL}}(p \| q) = 0 \iff p = q$
- **Not symmetric:** $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ in general

**Relationship:** $H(p, q) = H(p) + D_{\text{KL}}(p \| q)$

---

### 5.4 Mutual Information

$$I(X; Y) = D_{\text{KL}}(p(X,Y) \| p(X)p(Y)) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)$$

$I(X; Y) = 0$ iff $X \perp Y$.

---

## 6. Optimization

### 6.1 Convexity

A function $f$ is **convex** if for all $\mathbf{x}, \mathbf{y}$ and $t \in [0,1]$:

$$f(t\mathbf{x} + (1-t)\mathbf{y}) \leq t f(\mathbf{x}) + (1-t) f(\mathbf{y})$$

**Equivalently** (for twice-differentiable $f$): $H \succeq 0$ everywhere.

**Strict convexity:** $<$ above → unique global minimum.

**Convex functions in ML:**

| Function | Convex? |
|---|---|
| $\|w\|_2^2$ | Yes |
| $\|w\|_1$ | Yes |
| Cross-entropy loss | Yes (in outputs, not weights) |
| Hinge loss | Yes |
| Neural network loss | No (in general) |

---

### 6.2 Gradient Descent

Update rule:

$$\boldsymbol{\theta} := \boldsymbol{\theta} - \alpha \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$$

**Convergence** for convex $J$ with Lipschitz gradient (constant $L$):

$$J(\boldsymbol{\theta}^{(t)}) - J^* \leq \frac{L \|\boldsymbol{\theta}^{(0)} - \boldsymbol{\theta}^*\|^2}{2t}$$

**Learning rate selection:** step size $\alpha \leq 1/L$ guarantees descent.

---

### 6.3 Stochastic and Mini-Batch Gradient Descent

$$\boldsymbol{\theta} := \boldsymbol{\theta} - \alpha \nabla_{\boldsymbol{\theta}} J^{\{t\}}(\boldsymbol{\theta})$$

Uses an unbiased estimate of the full gradient: $\mathbb{E}[\nabla J^{\{t\}}] = \nabla J$.

---

### 6.4 Constrained Optimization — Lagrangian

Minimize $f(\mathbf{x})$ subject to $g_i(\mathbf{x}) = 0$:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x})$$

**KKT conditions** (for inequality constraints $h_j(\mathbf{x}) \leq 0$):

$$\nabla f + \sum_i \lambda_i \nabla g_i + \sum_j \mu_j \nabla h_j = 0$$
$$\mu_j h_j(\mathbf{x}) = 0, \quad \mu_j \geq 0, \quad h_j(\mathbf{x}) \leq 0$$

---

### 6.5 Coordinate Descent

Update one coordinate at a time while holding others fixed:

$$\theta_j := \arg\min_{\theta_j} J(\theta_1, \ldots, \theta_j, \ldots, \theta_n)$$

Used in Lasso (coordinate-wise soft-thresholding).

---

### 6.6 Newton's Method

Uses second-order information (Hessian) for faster convergence near optima:

$$\boldsymbol{\theta} := \boldsymbol{\theta} - H^{-1} \nabla J(\boldsymbol{\theta})$$

**Quadratic convergence** near the minimum. Expensive for large $n$ because $H \in \mathbb{R}^{n \times n}$.

**Quasi-Newton (L-BFGS):** approximates $H^{-1}$ using gradient history — popular for non-neural ML.

---

### 6.7 Softmax and Log-Sum-Exp

**Softmax** converts raw scores $\mathbf{z}$ to probabilities:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

**Numerical stability:** subtract $\max_j z_j$ before exponentiating (does not change result):

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k - c}}{\sum_j e^{z_j - c}}, \quad c = \max_j z_j$$

**Log-sum-exp:**

$$\text{LSE}(\mathbf{z}) = \ln \sum_j e^{z_j} = c + \ln \sum_j e^{z_j - c}$$

Arises in cross-entropy loss: $-\ln \text{softmax}(\mathbf{z})_y = -z_y + \text{LSE}(\mathbf{z})$.

---

### 6.8 Common Loss Functions

| Task | Loss | Formula |
|---|---|---|
| Regression | MSE | $\frac{1}{m}\sum(y_i - \hat{y}_i)^2$ |
| Regression | MAE | $\frac{1}{m}\sum|y_i - \hat{y}_i|$ |
| Regression | Huber | MSE if $|e|<\delta$, else MAE |
| Binary classification | Log loss / BCE | $-\frac{1}{m}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| Multi-class | Categorical cross-entropy | $-\frac{1}{m}\sum_i\sum_k y_k^{(i)}\log\hat{y}_k^{(i)}$ |
| SVM | Hinge | $\frac{1}{m}\sum\max(0, 1 - y_i \hat{y}_i)$ |
| Generative | KL divergence | $D_{\text{KL}}(p \| q)$ |

---

### 6.9 Useful NumPy / SciPy Cheatsheet

```python
import numpy as np
from scipy import linalg, stats

# --- Array creation ---
np.zeros((m, n)); np.ones((m, n)); np.eye(n)
np.random.randn(m, n)          # standard normal
np.random.rand(m, n)           # uniform [0, 1)
np.arange(start, stop, step)
np.linspace(start, stop, num)

# --- Shape manipulation ---
A.shape; A.ndim; A.size
A.reshape(m, n)
A.flatten()                    # → 1-D copy
A.ravel()                      # → 1-D view if possible
np.expand_dims(A, axis=0)      # add dimension
A.squeeze()                    # remove size-1 dims
np.concatenate([A, B], axis=0)
np.vstack([A, B]); np.hstack([A, B])

# --- Linear algebra ---
np.dot(A, B); A @ B
np.linalg.inv(A)
np.linalg.solve(A, b)
np.linalg.lstsq(A, b, rcond=None)
np.linalg.det(A)
np.linalg.matrix_rank(A)
np.trace(A)
np.linalg.eig(A)               # eigenvalues, eigenvectors
np.linalg.eigh(A)              # symmetric matrix
np.linalg.svd(A)
np.linalg.norm(v)              # L2
np.linalg.norm(v, 1)           # L1
np.linalg.norm(A, 'fro')       # Frobenius

# --- Statistics ---
np.mean(A, axis=0)             # column means
np.std(A, axis=0, ddof=1)      # unbiased std
np.var(A, axis=0, ddof=1)      # unbiased variance
np.median(A)
np.percentile(A, 75)
np.corrcoef(X)                 # correlation matrix
np.cov(X)                      # covariance matrix (each row is a variable)

# --- Element-wise math ---
np.exp(A); np.log(A); np.sqrt(A); np.abs(A)
np.sum(A, axis=1)              # row sums
np.prod(A); np.cumsum(A)
np.clip(A, a_min, a_max)
np.maximum(A, 0)               # ReLU

# --- Sorting / argmax ---
np.sort(A, axis=0)
np.argsort(A)
np.argmax(A, axis=1)
np.max(A); np.min(A)

# --- Probability distributions (scipy) ---
from scipy.stats import norm, binom, poisson, t, chi2

norm.pdf(x, loc=0, scale=1)    # PDF of N(0,1)
norm.cdf(x)                    # CDF
norm.ppf(0.975)                # inverse CDF (quantile)
t.ppf(0.975, df=n-1)           # t critical value
chi2.ppf(0.95, df=k)
```

---

*This document covers the core mathematical foundations needed in data science and machine learning: calculus for understanding optimization and backpropagation, linear algebra for data representation and transformations, probability and statistics for modeling uncertainty and evaluating models, information theory for loss functions, and optimization for training.*
