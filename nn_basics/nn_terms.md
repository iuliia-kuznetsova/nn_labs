# Neural Networks: Terms

Neural networks basic:
- Terms and Architecture;
- Activation Functions;
- Gradients and Derivatives;
- Gradient Descent;
- Tensors;
- Weights Initialization;
- Model Performance Evaluation
and interview-style pitfalls.

---

**Notation**

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

**Neural network (general)** 

A neural network is a type of machine learning model that uses layers of interconnected artificial neurons to learn a function mapping inputs to outputs, often capturing complex, non-linear patterns. **Depth** usually means how many (trainable) layers the network has; **width** is how many neurons a layer has. Inputs and outputs of the full network are often vectors or higher-dimensional **tensors**; one "neuron" still refers to one scalar output node inside a layer.

**Neuron (one unit)** 

Computes a **linear** score (logit) $z = \mathbf{w}^\top \mathbf{x} + b$, then a **nonlinear** **activation** $a = g(z)$. The pair $(\mathbf{w}, b)$ are **parameters** (learned by training). A neuron outputs a **single** scalar $a$ for that step.

**Layer** 

A collection of neurons that share the same input vector (or tensor) and produce a vector of activations. A **fully connected** (dense) layer maps $\mathbf{x} \mapsto g(\mathbf{W}\mathbf{x} + \mathbf{b})$: each row of $\mathbf{W}$ and matching entry in $\mathbf{b}$ is one neuron. Often the same $g$ is applied **element-wise** to the vector $\mathbf{W}\mathbf{x} + \mathbf{b}$ (i.e. once per component).

**Neuron vs layer** 

A **neuron** is one such unit: one $\mathbf{w}$, one $b$, one $g$ after one dot product, so one output $a$. A **layer** is **several** neurons **in parallel** on the **same** input $\mathbf{x}$: each has its own $\mathbf{w}$ and $b$ (or, stacked, a matrix $\mathbf{W}$ and vector $\mathbf{b}$), and you collect their outputs into a **vector** $\mathbf{a}$. So: **1 neuron â†’ 1 number; 1 layer â†’ 1 vector** (as many numbers as there are neurons in that layer). The "layer" is the whole block; a "neuron" is one of the units inside it.

**Training** 

Training a neural network is the process of adjusting its weights and biases to **minimize a loss** on data so it makes better predictions on new data. Training usually uses forward propagation to produce outputs, backpropagation to compute how each weight contributed to the error, and an optimizer like gradient descent to update the weights.

**Activation function** 

An **activation function** $g$ (or **nonlinearity**) is a fixed, usually nonlinear, map from one real number to one real number. It **turns a linear function of the input into a nonlinear response**. Examples: sigmoid $\sigma$, ReLU, tanh, identity (linear output layer). You pick $g$ when you design the model. Sometimes different neurons use different $g$'s, but a dense layer often uses the same $g$ for all units in that layer.

**Epoch** 

One **epoch** is one **complete** pass over the **training** dataset: every training example is used at least once in that cycle (e.g. one presentation of the full set in a fixed or shuffled order, or a sequence of mini-batches that together cover all data once). Training is typically run for many epochs; the loss and metrics are often reported **per epoch** or averaged within an epoch.

**Loss** 

A scalar $\mathcal{L}$ (or average over examples) measuring how wrong predictions are. Training **minimizes** expected loss over data.

**Forward pass** 

Compute predictions from inputs and current parameters.

**Backward pass (backpropagation)** 

Apply the chain rule to obtain gradients of the loss with respect to all parameters, moving from the output layer back toward the input.

**Linear regression** 

Linear regression is a **single neuron** with identity activation and no hidden layer.

**Logistic regression** 

Logistic regression is a **single neuron** with sigmoid activation and no hidden layer.

### 1.1. Tricky interview questions

**Q1. If a neural network has many layers but no nonlinear activations, is it still more powerful than one linear layer?**  
No. A composition of linear maps is still one linear map, so hidden linear layers do not add representational power.

**Q2. Is logistic regression a neural network?**  
Yes, it can be viewed as a single neuron with sigmoid activation and no hidden layer.

**Q3. What is the difference between a parameter and a hyperparameter?**  
Parameters such as weights and biases are learned during training; hyperparameters such as learning rate, number of layers, and activation choice are set before or around training.

---

## 2. Activation Functions

An activation function maps the pre-activation $z$ (a real number) to the activation $a = g(z)$. Its job is to introduce non-linearity. Without it, a stack of layers would collapse into one big linear map.

### 2.1. Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma(z) \in (0, 1).
$$

- **Shape.** S-curve that squashes any real number into the open interval $(0, 1)$.
- **Limits.** $\sigma(z) \to 1$ as $z \to +\infty$, $\sigma(z) \to 0$ as $z \to -\infty$, and $\sigma(0) = 0.5$.
- **Derivative.** $\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)$.
- **Use.** Output layer of **binary classification** (its output is a probability). Rarely used in hidden layers of deep networks because gradients **vanish** when $|z|$ is large (the curve is almost flat there).

---

### 2.2. ReLU (Rectified Linear Unit)

$$
\mathrm{ReLU}(z) = \max(0, z) = \begin{cases} z, & z > 0 \\ 0, & z \le 0 \end{cases}
$$

- **Shape.** Linear for positive inputs, zero for negative inputs.
- **Derivative.** $\mathrm{ReLU}'(z) = 1$ if $z > 0$, else $0$ (undefined at $z = 0$; in practice set to $0$ or $1$).
- **Use.** Default choice for hidden layers of deep networks. It is cheap to compute and does not saturate for $z > 0$, which helps gradients flow.
- **Caveat.** **Dying ReLU**: if a neuron always outputs $0$ (pre-activation stays negative), its gradient is $0$ and it stops learning. Variants like **Leaky ReLU** ($\max(\alpha z, z)$ with small $\alpha$) mitigate this.

---

### 2.3. Tanh

$$
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}, \qquad \tanh(z) \in (-1, 1).
$$

- **Shape.** S-curve like sigmoid, but **zero-centered** (output ranges from $-1$ to $1$).
- **Derivative.** $\tanh'(z) = 1 - \tanh^2(z)$.
- **Relation to sigmoid.** $\tanh(z) = 2\sigma(2z) - 1$.
- **Use.** Often preferred over sigmoid for hidden layers (zero-centered activations help optimization), but still saturates for large $|z|$, so ReLU-family functions usually win in deep nets.

---

### 2.4. Leaky ReLU

$$
\text{Leaky ReLU}(z) = \max(\alpha z,\; z) = \begin{cases} z, & z > 0 \\ \alpha z, & z \le 0 \end{cases}
$$

where $\alpha$ is a small constant (commonly $0.01$).

- **Shape.** Linear for positive inputs; small negative slope $\alpha$ for negative inputs (unlike the flat zero of ReLU).
- **Derivative.** $1$ if $z > 0$; $\alpha$ if $z < 0$ (undefined at $z = 0$; set to either in practice).
- **Use.** Addresses the **dying ReLU** problem: neurons whose pre-activation is always negative still receive a non-zero gradient and can continue to learn.
- **Note.** $\alpha$ can also be made a learnable parameter (Parametric ReLU / PReLU). Usually works slightly better than plain ReLU but is less commonly seen in practice.

---

### 2.5. Choosing an Activation Function

| Where | Recommended choice | Why |
|-------|--------------------|-----|
| Hidden layers (default) | **ReLU** | Cheap, does not saturate for $z > 0$, fast learning |
| Hidden layers (alternative) | **Tanh** | Zero-centered outputs make the next layer's optimization easier; still saturates for large $\lvert z\rvert$ |
| Hidden layers (anti-dying) | **Leaky ReLU** | Avoids dead neurons; comparable to or slightly better than ReLU |
| Output - binary classification | **Sigmoid** | Outputs a probability in $(0,1)$ |
| Output - regression (real value) | **Linear (identity)** | $\hat y$ can range over all reals |
| Output - non-negative regression | **ReLU** | Guarantees $\hat y \ge 0$ |

- **Sigmoid** is almost never used in hidden layers of modern networks; tanh is strictly superior.
- When unsure, try all candidates on a holdout validation set and keep whichever performs best.

![Main activation functions](graphs\main_activation_functions.jpg)

---

### 2.6. Why Non-Linear Activation Functions Are Necessary

If every layer uses a **linear (identity) activation** $g(z) = z$, composing two layers gives:

$$
a^{[2]} = W^{[2]}\!\bigl(W^{[1]}\mathbf{x} + \mathbf{b}^{[1]}\bigr) + \mathbf{b}^{[2]} = \underbrace{W^{[2]}W^{[1]}}_{W'}\,\mathbf{x} + \underbrace{W^{[2]}\mathbf{b}^{[1]}+\mathbf{b}^{[2]}}_{\mathbf{b}'}.
$$

No matter how many layers are stacked, the result is still $W'\mathbf{x} + \mathbf{b}'$ -- **a single linear map**. Adding hidden layers with linear activations is therefore pointless: the network cannot learn non-linear patterns and is equivalent to logistic/linear regression with no hidden layers.

**One valid exception.** The **output layer** may use a linear activation when predicting a real-valued quantity (regression). All hidden layers must still use non-linear activations.

---

### 2.7. Derivatives of Activation Functions

During **backpropagation** the network must evaluate the derivative (slope) of the activation function at every neuron. This section collects the formulas, explains the notation, and gives sanity checks.

#### Notation

For an activation function $g$, the derivative with respect to its scalar input $z$ is written either as $\dfrac{d}{dz}g(z)$ or, using the prime shorthand, as $g'(z)$.

When the activation value $a = g(z)$ has already been computed during the forward pass, the derivative can often be expressed more cheaply in terms of $a$ — avoiding a second evaluation of $g$.

#### 2.7.1. Sigmoid

$$
g(z) = \sigma(z) = \frac{1}{1+e^{-z}}
$$

$$
\boxed{g'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr) = a(1-a)}
$$

**Derivation sketch.** Using the quotient rule on $\sigma(z)$ yields the product $\sigma(z)(1-\sigma(z))$.

**Sanity checks.**

| $z$ | $a = \sigma(z)$ | $g'(z) = a(1-a)$ | Intuition |
|-----|-----------------|-------------------|-----------|
| $+10$ | $\approx 1$ | $\approx 1 \cdot 0 = 0$ | Flat at right tail — gradient vanishes |
| $-10$ | $\approx 0$ | $\approx 0 \cdot 1 = 0$ | Flat at left tail — gradient vanishes |
| $0$ | $0.5$ | $0.5 \cdot 0.5 = 0.25$ | Maximum slope is at $z=0$ |

**Cached-value form.** If $a$ is already stored from the forward pass: $g'(z) = a(1-a)$.

---

#### 2.7.2. Tanh

$$
g(z) = \tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}
$$

$$
\boxed{g'(z) = 1 - \tanh^2(z) = 1 - a^2}
$$

**Sanity checks.**

| $z$ | $a = \tanh(z)$ | $g'(z) = 1-a^2$ | Intuition |
|-----|----------------|-----------------|-----------|
| $+10$ | $\approx +1$ | $\approx 1 - 1 = 0$ | Saturated — gradient vanishes |
| $-10$ | $\approx -1$ | $\approx 1 - 1 = 0$ | Saturated — gradient vanishes |
| $0$ | $0$ | $1 - 0 = 1$ | Maximum slope, steeper than sigmoid at $z=0$ |

**Cached-value form.** $g'(z) = 1 - a^2$.

---

#### 2.7.3. ReLU

$$
g(z) = \max(0,\,z)
$$

$$
\boxed{g'(z) = \begin{cases} 1, & z > 0 \\ 0, & z < 0 \end{cases}}
$$

- **At $z = 0$:** the derivative is technically undefined (the function has a kink). In practice, setting $g'(0) = 0$ or $g'(0) = 1$ both work; the probability of hitting exactly $z = 0$ in floating-point arithmetic is negligible.
- **Sub-gradient perspective.** From the viewpoint of convex optimization, any value in $[0,1]$ is a valid sub-gradient at $z=0$, so gradient descent remains valid.
- **Key advantage.** For $z > 0$ the gradient is a constant $1$ — gradients do **not** vanish in the positive half-space, which speeds up training compared to sigmoid/tanh.

---

#### 2.7.4. Leaky ReLU

$$
g(z) = \max(\alpha z,\, z), \quad \alpha \ll 1\ (\text{e.g.}\ 0.01)
$$

$$
\boxed{g'(z) = \begin{cases} 1, & z > 0 \\ \alpha, & z < 0 \end{cases}}
$$

- At $z = 0$ the same sub-gradient argument applies; setting $g'(0) = 1$ or $g'(0) = \alpha$ are both acceptable.
- The non-zero slope $\alpha$ for negative $z$ means **dead neurons cannot occur** — every neuron always receives a gradient signal.

---

#### 2.7.5. Summary Table

| Function | $g(z)$ | $g'(z)$ | Cached form |
|----------|--------|---------|-------------|
| Sigmoid | $\sigma(z)$ | $\sigma(z)(1-\sigma(z))$ | $a(1-a)$ |
| Tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ | $1 - a^2$ |
| ReLU | $\max(0,z)$ | $\mathbf{1}[z > 0]$ | same |
| Leaky ReLU | $\max(\alpha z, z)$ | $\mathbf{1}[z > 0] + \alpha\,\mathbf{1}[z \le 0]$ | same |

$\mathbf{1}[\cdot]$ is the indicator function (1 if true, 0 if false).

**Why "cached form" matters** 

In a neural network the forward pass already computes and stores $\mathbf{A}^{[\ell]}$. The backward pass can then reuse those stored values:

```python
# Sigmoid backward
dZ = dA * A * (1 - A)

# Tanh backward
dZ = dA * (1 - A**2)

# ReLU backward
dZ = dA * (Z > 0).astype(float)

# Leaky ReLU backward  (alpha = 0.01)
dZ = dA * np.where(Z > 0, 1, alpha)
```

### 2.8. Tricky interview questions

**Q1. Why is sigmoid usually a bad hidden-layer activation in deep networks?**  
For large positive or negative inputs it saturates, making its derivative close to zero and causing weak gradient flow.

**Q2. ReLU is not differentiable at zero. Why can neural networks still use it?**  
The exact point $z=0$ is rarely hit in practice, and implementations choose a valid convention or sub-gradient there.

**Q3. Can the output layer use a different activation from the hidden layers?**  
Yes. Hidden layers usually need nonlinear activations for representation power, while the output activation should match the prediction target, such as sigmoid for binary classification or identity for regression.

---

## 3. Gradients and Derivatives

### 3.1. Partial derivative and gradient

**Partial derivative** 

A partial derivative is the slope of a multivariable function in one chosen direction, while keeping the other variables fixed. For example, $\partial f / \partial x$ tells you how steep the surface is if you move **only** in the $x$-direction.

**Gradient** 

The gradient is the vector made from **all** the partial derivatives. So for $f(x, y)$,

$$
\nabla f = \left( \frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y} \right).
$$

Each component is a slope in one coordinate direction, and together they tell you the **direction of steepest increase**.

**Intuition** 

If a partial derivative is "the slope along one axis," then the gradient is "the full slope information" at that point. Its direction points uphill most steeply, and its length tells you how steep that uphill direction is.

---

### 3.2. Chain rule and common derivatives

**Chain rule** 

The chain rule is a calculus rule for differentiating a function that is inside another function. It says: take the derivative of the outer function, then multiply by the derivative of the inner function. If $y = f(u)$ and $u = g(x)$, then

$$
\frac{dy}{dx} = \frac{dy}{du}\,\frac{du}{dx}.
$$

**Main derivative rules**

- $\dfrac{d}{dz} z^2 = 2z$.
- $\dfrac{d}{dz} \log z = \dfrac{1}{z}$ (for $z > 0$).
- If $u = \mathbf{w}^\top \mathbf{x} + b$, then $\dfrac{\partial u}{\partial \mathbf{w}} = \mathbf{x}$ and $\dfrac{\partial u}{\partial b} = 1$.

**Sigmoid derivative.** With $\sigma(z) = \dfrac{1}{1+e^{-z}}$,

$$
\frac{d\sigma}{dz} = \sigma(z)\bigl(1 - \sigma(z)\bigr).
$$

This identity is used everywhere $\sigma$ appears inside a loss.

### 3.3. Tricky interview questions

**Q1. Does the gradient point toward the minimum of a function?**  
No. The gradient points in the direction of steepest increase; gradient descent moves in the opposite direction.

**Q2. Why does backpropagation rely on the chain rule?**  
Each layer depends on the previous layer, so the effect of a parameter on the final loss must be computed through a chain of intermediate dependencies.

**Q3. Is a partial derivative the same thing as a gradient?**  
No. A partial derivative is one component; the gradient is the vector containing all partial derivatives.

---

## 4. Gradient Descent

### 4.1. Gradient and gradient descent

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

**Worked example** 

For our function $z = 2x^2 + 3y^2$ the gradient is $\nabla J(\theta) = (4x, 6y)$. Let's choose $\alpha = 0.05$.

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

### 4.2. Gradient descent in neural networks

Gradient descent is an optimization algorithm used to **minimize a loss function** by repeatedly moving the model parameters in the direction that reduces error the most.  Gradient descent is what helps neural networks learn. Backpropagation computes the gradients, and gradient descent uses them to update weights and biases.

**Logic**

1. Start with initial weights and biases, usually random.
2. Compute the model's prediction (forward pass).
3. Measure the error using a loss function.
4. Compute the gradient of the loss with respect to each parameter (backward pass).
5. Update parameters in the **opposite** direction of the gradient.
6. Repeat until the loss stops improving much.

**Update rule** 

A common form is

$$
\theta := \theta - \eta\,\nabla J(\theta),
$$

where $\theta$ is a parameter, $\eta$ is the **learning rate**, and $\nabla J(\theta)$ is the gradient of the loss.

**Learning rate** 

The learning rate is a hyperparameter that controls step size: too large can **overshoot** the minimum, and too small can make training very slow.

### 4.3. Tricky interview questions

**Q1. If the loss increases after a gradient descent step, does that mean the gradient was wrong?**  
Not necessarily. The learning rate may be too large, causing the update to overshoot a better region.

**Q2. Why do we subtract the gradient instead of adding it?**  
The gradient points toward steepest increase, so subtracting it moves the parameters toward steepest local decrease.

**Q3. Can gradient descent get stuck even when implemented correctly?**  
Yes. It can be slowed by plateaus, poor conditioning, bad learning rates, saddle points, or local minima depending on the loss surface.

---

## 5. Tensors

### 5.1. In math

**Definition** 

In math, an $n$-th-rank tensor in $m$-dimensional space is a mathematical object that has $n$ indices and $m^n$ components and obeys certain transformation rules.

The following table classifies tensor objects by their number of dimensions (rank):

| Rank $n$ | Object |
|----------|--------|
| $0$      | scalar |
| $1$      | vector |
| $2$      | $m \times m$ matrix |
| $\ge 3$  | tensor |

### 5.2. In computer science

**Definition** 

In computer science, a tensor is essentially a **multidimensional array of numbers**.

| Rank | CS term    | Shape (typical) | Use-case example |
|------|------------|-----------------|------------------|
| 0    | Scalar     | `[]`            | The "loss" value of a neural network |
| 1    | Vector     | `[n]`           | Single feature vector (e.g. word embedding) |
| 2    | Matrix     | `[n, m]`        | Grayscale image or weight matrix |
|      | Matrix     | `[t, f]`        | Time series (Time, Features)|
| 3    | 3D tensor  | `[c, h, w]`     | RGB image (Channels, Height, Width) |
|      | 3D tensor  | `[e, t, f]`     | Panel time series (Entities, Time, Features)
| 4    | 4D tensor  | `[b, c, h, w]`  | Batch of RGB images (Batch, Channels, Height, Width) |

### 5.3. Tricky interview questions

**Q1. Is every matrix a tensor?**  
Yes. A matrix is a rank-2 tensor in the computer-science sense.

**Q2. Does tensor rank always mean the same thing in math and in machine learning code?**  
Not always. In code, rank usually means the number of axes or dimensions; in math, tensor rank can also refer to more formal transformation properties.

**Q3. For image data, why might the same 4D tensor be written as `[b, c, h, w]` or `[b, h, w, c]`?**  
Different frameworks use different channel conventions. The data can represent the same batch of images, but operations must know which axis stores channels.

---

## 6. Weight Initialization

### 6.1. The Symmetry-Breaking Problem

If all weights are initialized to **zero** (or any identical constant), every hidden neuron in a layer computes exactly the same function of the input:

- All neurons receive the same input and have identical weights, so $a^{[1]}_1 = a^{[1]}_2 = \cdots$.
- Back-propagation produces identical gradients for every neuron.
- After every weight update the neurons remain identical — by induction they stay symmetric for all iterations of training.

Result: no matter how many hidden units exist, the network behaves as if it had only **one hidden unit per layer**. Multiple units buy nothing.

> **Bias terms** b do **not** cause this problem and can safely be initialized to zero, as long as W is initialized randomly.

### 6.2. Random Initialization

Break symmetry by drawing weights from a random distribution and scaling by a small constant:

```python
W1 = np.random.randn(n1, n0) * 0.01   # small random values
b1 = np.zeros((n1, 1))                 # zeros are fine for biases

W2 = np.random.randn(n2, n1) * 0.01
b2 = np.zeros((n2, 1))
```

**Why use a small constant (e.g. 0.01)**

If W is large, the pre-activations z = Wx + b are large in magnitude. For **tanh** or **sigmoid** activations this means the network starts in the **saturated** (flat) region of the curve, where the gradient is near zero, so gradient descent is very slow from the start. Multiplying by a small constant keeps initial z values near zero - in the high-gradient region - so learning starts quickly.

**When to use a different constant**

| Setting | Typical choice |
|---------|----------------|
| Shallow network (1 hidden layer) | 0.01 is usually fine |
| Deep network | Specialized initializations (Xavier / Glorot, He) are preferred; they scale the variance with layer width and are covered in later material |

### 6.3. Tricky interview questions

**Q1. Why is initializing all hidden-layer weights to zero a problem?**  
All neurons in the same layer start identical, receive identical gradients, and remain identical, so the layer fails to learn diverse features.

**Q2. Can biases be initialized to zero?**  
Yes. Zero biases do not cause the same symmetry problem as long as the weights are randomly initialized.

**Q3. Why can very large random weights make training slow with sigmoid or tanh?**  
They can push pre-activations into saturated regions where derivatives are near zero, causing vanishing gradients from the first updates.

---

## 7. Model Performance Evaluation

### 7.1. Training Accuracy vs. Test Accuracy

After training, two key numbers are usually reported:

- **Training accuracy** — how well the model performs on the examples it was trained on.
- **Test accuracy** — how well it performs on held-out examples it has never seen.

The relationship between these two numbers is the primary signal for diagnosing how well a model generalizes.

### 7.2. The Common Pattern and the Common Misconception

It is widely observed that *test accuracy tends to be lower than training accuracy* when a model overfits — and this is true. However, it is **not** a law:

> Test accuracy is not required to be lower than training accuracy.

It can be slightly higher, roughly equal, or clearly lower depending on the situation. Treating "test accuracy > training accuracy" as proof of no overfitting, or "test accuracy ≤ training accuracy" as the only valid outcome, are both mistakes.

### 7.3. Why Test Accuracy Can Be Higher Than Training Accuracy

Several factors can produce test accuracy that equals or exceeds training accuracy even in a correctly functioning model:

**1. Regularization is active during training but not at test time**
Techniques like dropout zero out neurons during the forward pass at training time, which makes training harder and effectively lowers training accuracy. At test time the full network is used, so test accuracy benefits from all neurons — this alone can make the test number higher than the training number.

**2. Train/test split randomness**
With small datasets, a random split can produce a test set that is slightly "easier" than the training set — better-balanced classes, less noisy examples, or fewer hard edge cases. This is a statistical artifact of the split, not evidence of good or bad generalization.

**3. How metrics are computed**
Train accuracy is often measured with dropout on and batch normalization in training mode; test accuracy is measured with dropout off and batch normalization in inference mode. These are not the same computational graph, so a direct numerical comparison needs to account for these differences.

### 7.4. How to Actually Judge Generalization

A small gap between training and test accuracy, with both numbers being high, is the reliable indicator of good generalization — not the sign of the difference.

| Observation | Likely interpretation |
|---|---|
| Train: high, Test: much lower | Overfitting — model memorizes training data |
| Train: low, Test: low | Underfitting — model too simple or undertrained |
| Train: high, Test: similarly high | Good generalization |
| Train: slightly lower than Test | Normal when regularization is active during training; not a problem |
| Train: high, Test: high, gap tiny | Strong signal that the model generalizes well |

**Better diagnostic checklist:**
1. Both train and test accuracy are reasonably high.
2. The gap between them is small.
3. Results are stable across different random seeds and splits (not a lucky split artifact).
4. Dev-set performance guided hyperparameter choices (test set was not touched until the end).

### 7.5. Overfitting, Underfitting, and the Accuracy Gap

**Overfitting** (high variance): the model has learned noise specific to the training set.
- Train accuracy: high.
- Test accuracy: notably lower.
- Fix: regularization (L2, dropout), more data, simpler architecture.

**Underfitting** (high bias): the model is too simple to capture the true pattern.
- Train accuracy: low.
- Test accuracy: similarly low (or even close to train).
- Fix: bigger network, more training, better features.

**Good fit**: the model captures the true pattern without memorizing noise.
- Train accuracy: high.
- Test accuracy: close to train accuracy (may be slightly above or below).

The goal of training is not to achieve the lowest possible training error, but to achieve the smallest possible gap between training and test error while keeping both high.

### 7.6. Tricky interview questions

**Q1. Does test accuracy higher than training accuracy prove the model is not overfitting?**  
No. It can happen because of regularization, an easier test split, metric differences, or random variation.

**Q2. Is very high training accuracy always good?**  
Not by itself. If test accuracy is much lower, the model may be memorizing training data instead of generalizing.

**Q3. Why should the test set not be used repeatedly during tuning?**  
Repeated test-set feedback leaks information into model selection, making the test result too optimistic as an estimate of real generalization.

---