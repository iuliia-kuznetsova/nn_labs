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

**Neural network (general).** A neural network is a type of machine learning model that uses layers of interconnected artificial neurons to learn a function mapping inputs to outputs, often capturing complex, non-linear patterns. **Depth** usually means how many (trainable) layers the network has; **width** is how many neurons a layer has. (Inputs and outputs of the full network are often vectors or higher-dimensional **tensors**; one “neuron” still refers to one scalar output node inside a layer.)

**Neuron (one unit).** Computes a **linear** score (logit) $z = \mathbf{w}^\top \mathbf{x} + b$, then a **nonlinear** **activation** $a = g(z)$. The pair $(\mathbf{w}, b)$ are **parameters** (learned by training). A neuron outputs a **single** scalar $a$ for that step.

**Layer.** A collection of neurons that share the same input vector (or tensor) and produce a vector of activations. A **fully connected** (dense) layer maps $\mathbf{x} \mapsto g(\mathbf{W}\mathbf{x} + \mathbf{b})$: each row of $\mathbf{W}$ and matching entry in $\mathbf{b}$ is one neuron. Often the same $g$ is applied **element-wise** to the vector $\mathbf{W}\mathbf{x} + \mathbf{b}$ (i.e. once per component).

**Neuron vs layer.** A **neuron** is one such unit: one $\mathbf{w}$, one $b$, one $g$ after one dot product, so one output $a$. A **layer** is **several** neurons **in parallel** on the **same** input $\mathbf{x}$: each has its own $\mathbf{w}$ and $b$ (or, stacked, a matrix $\mathbf{W}$ and vector $\mathbf{b}$), and you collect their outputs into a **vector** $\mathbf{a}$. So: **1 neuron → 1 number; 1 layer → 1 vector** (as many numbers as there are neurons in that layer). The “layer” is the whole block; a “neuron” is one of the units inside it.

**Training.** Training a neural network is the process of adjusting its weights and biases to **minimize a loss** on data so it makes better predictions on data. Training usually uses forward propagation to produce outputs, backpropagation to compute how each weight contributed to the error, and an optimizer like gradient descent to update the weights

**Activation function.** An **activation function** $g$ (or **nonlinearity**): a fixed, usually nonlinear, map from one real number to one real number. It **turns a linear function of the input into a nonlinear response**. Examples: sigmoid $\sigma$, ReLU, tanh, identity (linear output layer). You pick $g$ when you design the model. Sometimes different neurons use different $g$'s, but a dense layer often uses the same $g$ for all units in that layer.

**Epoch.** One **epoch** is one **complete** pass over the **training** dataset: every training example is used at least once in that cycle (e.g. one presentation of the full set in a fixed or shuffled order, or a sequence of mini-batches that together cover all data once). Training is typically run for many epochs; the loss and metrics are often reported **per epoch** or averaged within an epoch.

**Loss.** A scalar $L$ (or average over examples) measuring how wrong predictions are. Training **minimizes** expected loss over data.

**Forward pass.** Compute predictions from inputs and current parameters.

**Backward pass (backpropagation).** Apply the chain rule to obtain gradients of the loss with respect to all parameters, moving from the output layer back toward the input.

**Linear regression** is a **single neuron** with identity activation and no hidden layer.

**Logistic regression** is a **single neuron** with sigmoid activation and no hidden layer.

---

## 2. Activation Functions

### 2.1 Sigmoid



---

### 2.2 ReLu




---

### 2.3 Tanh


---



## 2. Gradient and derivatives

a partial derivative is the slope of a multivariable function in one chosen direction, while keeping the other variables fixed. For example, 
∂
f
/
∂
x
∂f/∂x tells you how steep the surface is if you move only in the 
x
x-direction.

What the gradient means
The gradient is the vector made from all the partial derivatives. So for 
f
(
x
,
y
)
f(x,y),

∇
f
=
(
∂
f
∂
x
,
∂
f
∂
y
)
.
∇f=( 
∂x
∂f
​
 , 
∂y
∂f
​
 ).
Each component is a slope in one coordinate direction, and together they tell you the direction of steepest increase.

Intuition
If a partial derivative is “the slope along one axis,” then the gradient is “the full slope information” at that point. Its direction points uphill most steeply, and its length tells you how steep that uphill direction is.

### 2.5 Gradient and Derivatives

**Partial derivative** - is how a function changes with respect to one variable while the others stay fixed, so is a slope of function to the axis.

**Gradient** - is the collection of all partial derivatives of a multivariable function, written as a vector.

A gradient is a vector made of partial derivatives, so a partial derivative is one component of the gradient, not the whole thing.

Example
For a function 
f
(
x
,
y
)
f(x,y):

∂
f
/
∂
x
∂f/∂x is the partial derivative in the 
x
x-direction.

∂
f
/
∂
y
∂f/∂y is the partial derivative in the 
y
y-direction.

The gradient is 
∇
f
=
(
∂
f
/
∂
x
,
 
∂
f
/
∂
y
)
∇f=(∂f/∂x, ∂f/∂y).

**Chain rule.** If $y = f(u)$ and $u = g(x)$, then $\dfrac{dy}{dx} = \dfrac{dy}{du}\dfrac{du}{dx}$.

**Main derivative rules**
- $\dfrac{d}{dz} z^2 = 2z$.
- $\dfrac{d}{dz} \log z = \dfrac{1}{z}$ (for $z > 0$).
- If $u = \mathbf{w}^\top \mathbf{x} + b$, then $\dfrac{\partial u}{\partial \mathbf{w}} = \mathbf{x}$, $\dfrac{\partial u}{\partial b} = 1$.

**Sigmoid.** With $\sigma(z) = \dfrac{1}{1+e^{-z}}$,

$$
\frac{d\sigma}{dz} = \sigma(z)\bigl(1 - \sigma(z)\bigr).
$$

This identity is used everywhere $\sigma$ appears inside a loss.

---

## 2. Gradient descent 

Gradient descent is an optimization algorithm used to minimize a loss function by repeatedly moving the model parameters in the direction that reduces error the most.

Intuition
Imagine standing on a hill and trying to reach the lowest point in a valley. Gradient descent looks at the slope at your current position and takes a small step downhill, then repeats this many times.

How it works
Start with initial weights and biases, usually random.

Compute the model’s prediction.

Measure the error using a loss function.

Compute the gradient of the loss with respect to each parameter.

Update parameters in the opposite direction of the gradient.

Repeat until the loss stops improving much.

Update rule
A common form is:

θ
:
=
θ
−
η
∇
J
(
θ
)
θ:=θ−η∇J(θ)
where 
θ
θ is a parameter, 
η
η is the learning rate, and 
∇
J
(
θ
)
∇J(θ) is the gradient of the loss.

In neural networks
Gradient descent is what helps neural networks learn. Backpropagation computes the gradients, and gradient descent uses them to update weights and biases.

Important idea
The learning rate controls step size: too large can overshoot the minimum, and too small can make training very slow.

So, in one sentence: gradient descent is a method for training models by iteratively adjusting parameters to make the loss smaller.

---