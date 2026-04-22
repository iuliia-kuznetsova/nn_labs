# Neural Networks: Terms

**Notation (used throughout).**

- $n_x$: number of features per example.
- $m$: number of examples (batch size).
- Input (features) matrix $\mathbf{X} \in \mathbb{R}^{n_x \times m}$: each **column** is one example $\mathbf{x}^{(i)} \in \mathbb{R}^{n_x}$.
- Output (target) $\mathbf{y} \in \mathbb{R}^{1 \times m}$ (row vector), with $y^{(i)} \in \{0,1\}$ for binary classification.
- Weights $\mathbf{w} \in \mathbb{R}^{n_x}$ (column vector) and bias $b \in \mathbb{R}$.
- Activation function $g$.
- $\sigma(\cdot)$ is the sigmoid, applied **element-wise** to matrices.
- $\odot$ is the Hadamard (element-wise) product.
- $\mathbf{1}_m \in \mathbb{R}^{m}$ is a column vector of ones (subscript omitted when size is clear).

---

## 1. Neural networks: basic terms and architecture

**Neural network (general).** A neural network is a type of machine learning model that uses layers of interconnected artificial neurons to learn a function mapping inputs to outputs, often capturing complex, non-linear patterns. **Depth** usually means how many (trainable) layers the network has; **width** is how many neurons a layer has. Inputs and outputs of the full network are often vectors or higher-dimensional **tensors**; one "neuron" still refers to one scalar output node inside a layer.

**Neuron (one unit).** Computes a **linear** score (logit) $z = \mathbf{w}^\top \mathbf{x} + b$, then a **nonlinear** **activation** $a = g(z)$. The pair $(\mathbf{w}, b)$ are **parameters** (learned by training). A neuron outputs a **single** scalar $a$ for that step.

**Layer.** A collection of neurons that share the same input vector (or tensor) and produce a vector of activations. A **fully connected** (dense) layer maps $\mathbf{x} \mapsto g(\mathbf{W}\mathbf{x} + \mathbf{b})$: each row of $\mathbf{W}$ and matching entry in $\mathbf{b}$ is one neuron. Often the same $g$ is applied **element-wise** to the vector $\mathbf{W}\mathbf{x} + \mathbf{b}$ (i.e. once per component).

**Neuron vs layer.** A **neuron** is one such unit: one $\mathbf{w}$, one $b$, one $g$ after one dot product, so one output $a$. A **layer** is **several** neurons **in parallel** on the **same** input $\mathbf{x}$: each has its own $\mathbf{w}$ and $b$ (or, stacked, a matrix $\mathbf{W}$ and vector $\mathbf{b}$), and you collect their outputs into a **vector** $\mathbf{a}$. So: **1 neuron → 1 number; 1 layer → 1 vector** (as many numbers as there are neurons in that layer). The "layer" is the whole block; a "neuron" is one of the units inside it.

**Training.** Training a neural network is the process of adjusting its weights and biases to **minimize a loss** on data so it makes better predictions on new data. Training usually uses forward propagation to produce outputs, backpropagation to compute how each weight contributed to the error, and an optimizer like gradient descent to update the weights.

**Activation function.** An **activation function** $g$ (or **nonlinearity**) is a fixed, usually nonlinear, map from one real number to one real number. It **turns a linear function of the input into a nonlinear response**. Examples: sigmoid $\sigma$, ReLU, tanh, identity (linear output layer). You pick $g$ when you design the model. Sometimes different neurons use different $g$'s, but a dense layer often uses the same $g$ for all units in that layer.

**Epoch.** One **epoch** is one **complete** pass over the **training** dataset: every training example is used at least once in that cycle (e.g. one presentation of the full set in a fixed or shuffled order, or a sequence of mini-batches that together cover all data once). Training is typically run for many epochs; the loss and metrics are often reported **per epoch** or averaged within an epoch.

**Loss.** A scalar $\mathcal{L}$ (or average over examples) measuring how wrong predictions are. Training **minimizes** expected loss over data.

**Forward pass.** Compute predictions from inputs and current parameters.

**Backward pass (backpropagation).** Apply the chain rule to obtain gradients of the loss with respect to all parameters, moving from the output layer back toward the input.

**Linear regression** is a **single neuron** with identity activation and no hidden layer.

**Logistic regression** is a **single neuron** with sigmoid activation and no hidden layer.

---

## 2. Activation Functions

An activation function maps the pre-activation $z$ (a real number) to the activation $a = g(z)$. Its job is to introduce non-linearity — without it, a stack of layers would collapse into one big linear map.

### 2.1 Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma(z) \in (0, 1).
$$

- **Shape.** S-curve that squashes any real number into the open interval $(0, 1)$.
- **Limits.** $\sigma(z) \to 1$ as $z \to +\infty$, $\sigma(z) \to 0$ as $z \to -\infty$, and $\sigma(0) = 0.5$.
- **Derivative.** $\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)$.
- **Use.** Output layer of **binary classification** (its output is a probability). Rarely used in hidden layers of deep networks because gradients **vanish** when $|z|$ is large (the curve is almost flat there).

---

### 2.2 ReLU (Rectified Linear Unit)

$$
\mathrm{ReLU}(z) = \max(0, z) = \begin{cases} z, & z > 0 \\ 0, & z \le 0 \end{cases}
$$

- **Shape.** Linear for positive inputs, zero for negative inputs.
- **Derivative.** $\mathrm{ReLU}'(z) = 1$ if $z > 0$, else $0$ (undefined at $z = 0$; in practice set to $0$ or $1$).
- **Use.** Default choice for hidden layers of deep networks. It is cheap to compute and does not saturate for $z > 0$, which helps gradients flow.
- **Caveat.** **Dying ReLU**: if a neuron always outputs $0$ (pre-activation stays negative), its gradient is $0$ and it stops learning. Variants like **Leaky ReLU** ($\max(\alpha z, z)$ with small $\alpha$) mitigate this.

---

### 2.3 Tanh

$$
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}, \qquad \tanh(z) \in (-1, 1).
$$

- **Shape.** S-curve like sigmoid, but **zero-centered** (output ranges from $-1$ to $1$).
- **Derivative.** $\tanh'(z) = 1 - \tanh^2(z)$.
- **Relation to sigmoid.** $\tanh(z) = 2\sigma(2z) - 1$.
- **Use.** Often preferred over sigmoid for hidden layers (zero-centered activations help optimization), but still saturates for large $|z|$, so ReLU-family functions usually win in deep nets.

---

## 3. Gradients and Derivatives

### 3.1 Partial derivative and gradient

**Partial derivative.** A partial derivative is the slope of a multivariable function in one chosen direction, while keeping the other variables fixed. For example, $\partial f / \partial x$ tells you how steep the surface is if you move **only** in the $x$-direction.

**Gradient.** The gradient is the vector made from **all** the partial derivatives. So for $f(x, y)$,

$$
\nabla f = \left( \frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y} \right).
$$

Each component is a slope in one coordinate direction, and together they tell you the **direction of steepest increase**.

**Intuition.** If a partial derivative is "the slope along one axis," then the gradient is "the full slope information" at that point. Its direction points uphill most steeply, and its length tells you how steep that uphill direction is.

---

### 3.2 Chain rule and common derivatives

**Chain rule.** The chain rule is a calculus rule for differentiating a function that is inside another function. It says: take the derivative of the outer function, then multiply by the derivative of the inner function. If $y = f(u)$ and $u = g(x)$, then

$$
\frac{dy}{dx} = \frac{dy}{du}\,\frac{du}{dx}.
$$

**Main derivative rules.**

- $\dfrac{d}{dz} z^2 = 2z$.
- $\dfrac{d}{dz} \log z = \dfrac{1}{z}$ (for $z > 0$).
- If $u = \mathbf{w}^\top \mathbf{x} + b$, then $\dfrac{\partial u}{\partial \mathbf{w}} = \mathbf{x}$ and $\dfrac{\partial u}{\partial b} = 1$.

**Sigmoid derivative.** With $\sigma(z) = \dfrac{1}{1+e^{-z}}$,

$$
\frac{d\sigma}{dz} = \sigma(z)\bigl(1 - \sigma(z)\bigr).
$$

This identity is used everywhere $\sigma$ appears inside a loss.

---

## 4. Gradient Descent

### 4.1 Gradient and gradient descent

For a scalar function $f(x, y, z)$ of multiple variables, the gradient is the vector

$$
\nabla f = \left( \frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y},\ \frac{\partial f}{\partial z} \right),
$$

whose value at a point $p$ gives the **direction and the rate of fastest increase**.

Suppose we have a function $z = 2x^2 + 3y^2$. Then its gradient is

$$
\nabla z = \left( \frac{\partial z}{\partial x},\ \frac{\partial z}{\partial y} \right) = (4x,\ 6y).
$$

**Gradient descent** is an optimization algorithm that iteratively minimizes a function by moving **opposite** to its gradient. Starting from initial weights $\theta$, the algorithm updates them as

$$
\theta := \theta - \alpha\,\nabla J(\theta),
$$

where:

- $\alpha$: **learning rate** (step size);
- $J(\theta)$: cost function evaluated at $\theta$;
- $\nabla J(\theta)$: gradient of the cost function.

**Worked example.** For our function $z = 2x^2 + 3y^2$ the gradient is $\nabla J(\theta) = (4x, 6y)$. Let's choose $\alpha = 0.05$.

**Step 0.** Start at the random point

$$
\theta_0 = (2, -3).
$$

The cost function value:

$$
J(\theta_0) = 2 \cdot 2^2 + 3 \cdot (-3)^2 = 8 + 27 = 35.
$$

The gradient at $\theta_0$:

$$
\nabla J(\theta_0) = (4x,\ 6y) = (8,\ -18).
$$

**Step 1.** New point:

$$
\theta_1 = \theta_0 - \alpha\,\nabla J(\theta_0) = (2, -3) - 0.05\,(8, -18) = (2 - 0.4,\ -3 + 0.9) = (1.6,\ -2.1).
$$

New cost:

$$
J(\theta_1) = 2\,(1.6)^2 + 3\,(-2.1)^2 = 5.12 + 13.23 = 18.35 \quad (\text{decreased } \downarrow).
$$

### 4.2 Gradient descent in neural networks

Gradient descent is an optimization algorithm used to **minimize a loss function** by repeatedly moving the model parameters in the direction that reduces error the most.  Gradient descent is what helps neural networks learn. Backpropagation computes the gradients, and gradient descent uses them to update weights and biases.

**Logic.**

1. Start with initial weights and biases, usually random.
2. Compute the model's prediction (forward pass).
3. Measure the error using a loss function.
4. Compute the gradient of the loss with respect to each parameter (backward pass).
5. Update parameters in the **opposite** direction of the gradient.
6. Repeat until the loss stops improving much.

**Update rule.** A common form is

$$
\theta := \theta - \eta\,\nabla J(\theta),
$$

where $\theta$ is a parameter, $\eta$ is the **learning rate**, and $\nabla J(\theta)$ is the gradient of the loss.

**Learning rate.** The learning rate is a hyperparameter that controls step size: too large can **overshoot** the minimum, and too small can make training very slow.

---

## 5. Tensors

### 5.1 In math

**Definition.** In math, an $n$-th-rank tensor in $m$-dimensional space is a mathematical object that has $n$ indices and $m^n$ components and obeys certain transformation rules.

The following table classifies tensor objects by their number of dimensions (rank):

| Rank $n$ | Object |
|----------|--------|
| $0$      | scalar |
| $1$      | vector |
| $2$      | $m \times m$ matrix |
| $\ge 3$  | tensor |

### 5.2 In computer science

**Definition.** In computer science, a tensor is essentially a **multidimensional array of numbers**.

| Rank | CS term    | Shape (typical) | Use-case example |
|------|------------|-----------------|------------------|
| 0    | Scalar     | `[]`            | The "loss" value of a neural network |
| 1    | Vector     | `[n]`           | Single feature vector (e.g. word embedding) |
| 2    | Matrix     | `[n, m]`        | Grayscale image or weight matrix |
|      | Matrix     | `[t, f]`        | Time series (Time, Features)|
| 3    | 3D tensor  | `[c, h, w]`     | RGB image (Channels, Height, Width) |
|      | 3D tensor  | `[e, t, f]`     | Panel time series (Entities, Time, Features)
| 4    | 4D tensor  | `[b, c, h, w]`  | Batch of RGB images (Batch, Channels, Height, Width) |