---
layout: post
title: "NICE Explained: A Beginner-Friendly Guide to Non-Linear Independent Components Estimation"
date: 2026-08-11
category: pic
---

> **Paper:** *NICE: Non-linear Independent Components Estimation*
> **Authors:** Laurent Dinh, David Krueger, Yoshua Bengio
> **Main idea:** Learn an invertible neural-network transformation that turns complicated data into simple, independent latent variables, while keeping both the probability and the inverse transformation easy to compute.

---

## The big picture before we start

The easiest way to understand NICE is to keep this picture in your head:

```text
                ENCODER
                  f
                  ↓
DATA x  ───────────────────→  LATENT h
 complicated distribution        simple distribution
                                independent variables

                  ↑
                  │
                f⁻¹
                  │
                  ↓

             generated x
```

Suppose we have thousands of handwritten digits.

The raw images are complicated: pixels are correlated, certain pixels tend to be dark together, the shape of one part of a digit tells us something about other parts, etc.

NICE tries to learn a transformation:

$$
h=f(x)
$$

that changes the complicated representation `x` into a representation `h` whose probability distribution is much easier to describe.

The dream is:

$$
\boxed{\text{complicated data }x
\quad\longrightarrow\quad
\text{simple independent latent variables }h}
$$

And because the transformation is **invertible**, we can also go backwards:

$$
\boxed{x=f^{-1}(h)}
$$

That second direction is what allows NICE to **generate new data**.

The paper explicitly describes this as learning a nonlinear bijective transformation that maps training data into a space where its distribution is factorized, while allowing exact log-likelihood computation and easy unbiased sampling.

---

# 1. What is the goal of the paper?

The abstract says:

> “a good representation is one in which the data has a distribution that is easy to model.”

This sounds complicated, but the idea is actually quite intuitive.

## 1.1 What does "representation" mean?

A **representation** is simply another way of describing the same thing.

For example, imagine a photograph of a cat.

The raw representation might be:

```text
pixel 1 = 0.21
pixel 2 = 0.47
pixel 3 = 0.83
...
pixel 100000 = 0.12
```

That is the **data representation**.

But we might instead describe the image using things like:

```text
cat position
cat size
ear shape
fur texture
background
lighting
...
```

That would be another representation.

NICE asks:

> Can we automatically transform the original data into a representation whose probability distribution is much simpler?

The paper calls this new representation the **latent space**.

---

## 1.2 What is "data"?

The data is what we start with.

For example, if we are modeling handwritten digits:

$$
x = \text{one image of a handwritten digit}
$$

For MNIST, an image has:

$$
28\times28=784
$$

pixels.

So we can represent one image as a vector:

$$
x=(x_1,x_2,\ldots,x_{784})
$$

where each $x_i$ represents one pixel.

Therefore:

$$
\boxed{x=\text{one observed example}}
$$

and the whole dataset might be:

$$
D={x^{(1)},x^{(2)},\ldots,x^{(N)}}.
$$

The paper generally considers examples living in a space $\mathcal X$, typically

$$
\mathcal X=\mathbb R^D.
$$

---

## 1.3 What is the latent space?

The **latent space** is the new space in which we represent the same data after applying the learned transformation.

We take:

$$
x
$$

and transform it:

$$
h=f(x).
$$

The resulting vector $h$ lives in the latent space, which we can call $\mathcal H$.

For example:

```text
Original image x

[0.0, 0.2, 0.8, 0.9, ..., 0.1]
             │
             │ f
             ↓
Latent representation h

[0.31, -0.82, 1.14, 0.05, ..., -0.27]
```

The important thing is that NICE wants the components of $h$ to be **independent**.

So instead of having a complicated joint distribution

$$
p_H(h_1,h_2,\ldots,h_D),
$$

we want:

$$
\boxed{
p_H(h)=\prod_{d=1}^{D}p_{H_d}(h_d)
}
$$

In words:

> The probability of the whole latent vector is just the product of the probabilities of its individual components.

This factorization is the central idea of NICE.

---

# 2. What are $x$, $h$, and the latent space?

The paper says:

$$
h=f(x)
$$

and asks the learner to find a transformation such that the resulting distribution factorizes.

Let's carefully define every symbol.

## $x$: the observed data

$$
\boxed{x\in\mathcal X}
$$

For example:

$$
x=\text{one 784-dimensional MNIST image}.
$$

It is what we observe.

---

## $f$: the learned transformation

$$
\boxed{f:\mathcal X\rightarrow\mathcal H}
$$

It takes the complicated data representation and transforms it into the latent representation.

$$
h=f(x).
$$

---

## $h$: the latent representation

$$
\boxed{h\in\mathcal H}
$$

It is the transformed version of $x$.

The crucial requirement is that $h$ should have a simple distribution.

For NICE:

$$
p_H(h)=\prod_d p_{H_d}(h_d).
$$

So the latent variables are independent.

---

## The latent space $\mathcal H$

The latent space is simply the space in which $h$ lives.

If $x\in\mathbb R^D$, then NICE requires $h$ to have the **same dimensionality**:

$$
x\in\mathbb R^D
$$

and

$$
h\in\mathbb R^D.
$$

This is because $f$ needs to be invertible. The paper explicitly assumes that $f$ is invertible and that $h$ and $x$ have the same dimension.

So:

```text
              same number of dimensions

x ∈ R^D  ─────── f ───────→  h ∈ R^D
 complicated                  simple
 distribution                 distribution
```

This is an important difference from many ordinary dimensionality-reduction methods.

NICE is not throwing information away.

It is **rearranging the information**.

---

# 3. Where does this equation come from?

The paper gives:

$$
\boxed{
p_X(x)
=
p_H(f(x))
\left|
\det
\frac{\partial f(x)}{\partial x}
\right|
}
$$

This is the **change-of-variables formula** for probability densities.

At first this equation can look terrifying.

It is actually just a mathematical way of saying:

> If we change coordinates from $x$ to $h=f(x)$, we must account for how much the transformation stretches or compresses space.

Let's build it from intuition.

---

## 3.1 Start with a one-dimensional example

Suppose:

$$
h=2x.
$$

Then:

$$
x=\frac{h}{2}.
$$

Imagine a small interval around $x$.

If the interval has width:

$$
dx,
$$

then after transformation:

$$
dh=2dx.
$$

So the transformation has stretched the space by a factor of 2.

Probability must remain the same:

$$
p_X(x),dx
=
p_H(h),dh.
$$

Since

$$
dh=2dx,
$$

we get:

$$
p_X(x),dx
=
p_H(h)(2dx).
$$

Therefore:

$$
p_X(x)=2p_H(h).
$$

Because:

$$
\frac{dh}{dx}=2,
$$

we can write:

$$
p_X(x)
=
p_H(f(x))
\left|\frac{df(x)}{dx}\right|.
$$

---

# 3.2 What happens in multiple dimensions?

Now imagine that $x$ has many dimensions:

$$
x=(x_1,x_2,\ldots,x_D).
$$

The derivative of a vector-valued function is no longer one number.

It becomes a matrix called the **Jacobian**:

$$
J_f(x)
=
\frac{\partial f(x)}{\partial x}.
$$

The determinant of that matrix tells us how much a tiny volume changes under the transformation.

So:

$$
\boxed{
\left|\det J_f(x)\right|
}
$$

is the volume-scaling factor.

Therefore:

$$
\boxed{
p_X(x)
=
p_H(f(x))
|\det J_f(x)|
}
$$

The paper uses exactly this change-of-variables equation as the foundation for its likelihood calculation.

---

## 3.3 Why does NICE care so much about the Jacobian?

Because NICE wants to calculate:

$$
p_X(x).
$$

To do that, we need:

1. $p_H(f(x))$
2. $|\det J_f(x)|$

The first one is easy because we deliberately choose $p_H$ to be simple.

The second one could be extremely expensive for a general neural network.

So the clever part of NICE is:

> Design the neural network so that its Jacobian determinant is easy to calculate.

That is one of the main architectural ideas of the paper.

---

# 4. How does Bayes' theorem fit into this?

This is an important point because there are **two different things that can be called a "prior"**, and confusing them can make NICE very difficult to understand.

Bayes' theorem is:

$$
\boxed{
p(\theta|D)
=
\frac{p(D|\theta)p(\theta)}
{p(D)}
}
$$

where:

* $p(\theta)$ = prior over model parameters
* $p(D|\theta)$ = likelihood
* $p(\theta|D)$ = posterior over parameters
* $p(D)$ = evidence

But the NICE paper is primarily doing **maximum likelihood**, not Bayesian inference over $\theta$.

This distinction matters.

---

## 4.1 The likelihood in NICE

Suppose our model has parameters $\theta$.

Those parameters determine the transformation:

$$
f_\theta.
$$

Therefore the model defines:

$$
p_\theta(x).
$$

For a dataset:

$$
D={x^{(1)},\ldots,x^{(N)}},
$$

assuming examples are independent:

$$
p_\theta(D)
=
\prod_{n=1}^{N}p_\theta(x^{(n)}).
$$

This is the **likelihood** of the parameters given the observed dataset.

The paper describes learning the transformation using maximum likelihood.

---

## 4.2 The latent prior $p_H(h)$

Here is the source of possible confusion.

The paper calls

$$
p_H(h)
$$

the **prior distribution**. It is the distribution we assume for the latent variables.

For example:

$$
h\sim\mathcal N(0,I)
$$

or a factorized logistic distribution.

The paper says that $p_H$ is a predefined density, for example a standard isotropic Gaussian, and that a factorial prior gives the NICE criterion.

So this is a prior over **latent variables $h$**.

It is **not** the same thing as a Bayesian prior over the neural-network parameters $\theta$.

---

## 4.3 What about a Bayesian prior over $\theta$?

If we wanted to do Bayesian inference over the network parameters, we could introduce:

$$
p(\theta).
$$

Then Bayes' theorem would be:

$$
p(\theta|D)
\propto
p_\theta(D)p(\theta).
$$

But that is **not the main training procedure used by NICE**.

NICE instead maximizes:

$$
\boxed{
\log p_\theta(D)
}
$$

directly.

So, for this paper, it is best to keep the following distinction in your head:

| Object | Meaning |
| ------ | ------- |
| $p_H(h)$ | Prior distribution over latent variables |
| $p_\theta(x)$ | Data distribution produced by the NICE model |
| $\theta$ | Neural-network/model parameters |
| $p_\theta(D)$ | Likelihood of the observed dataset |
| $p(\theta \mid D)$ | Bayesian posterior over parameters, not what NICE directly optimizes |
| $p(D)$ | Bayesian evidence, not the quantity NICE needs to calculate |

---

# 5. What is the neural network actually trying to learn?

Now we can connect all of this.

The paper says it learns a probability density from a parametric family

$$
{p_\theta,\theta\in\Theta}
$$

over a dataset of $N$ examples.

This means:

> We have a family of possible models, and $\theta$ determines which particular model we are using.

---

## 5.1 What are the parameters $\theta$?

Think about an ordinary neural network.

It has:

```text
weights
biases
weights
biases
...
```

All of those numbers are its parameters.

In NICE, the neural networks inside the coupling layers have parameters.

Collect all of them together and call them:

$$
\boxed{\theta}
$$

So:

$$
\theta
\quad\longrightarrow\quad
f_\theta
\quad\longrightarrow\quad
p_\theta(x).
$$

In words:

> The parameters $\theta$ determine the transformation $f$, and therefore determine the probability distribution that NICE assigns to the data.

---

# 5.2 What does $f$ learn?

The encoder $f$ learns how to transform complicated data into a simple latent representation:

$$
\boxed{
h=f_\theta(x)
}
$$

It learns a complicated nonlinear coordinate system.

For example, imagine that real images occupy a complicated curved region of pixel space.

NICE tries to "unfold" that complicated structure into a space where the variables behave independently.

Very roughly:

```text
DATA SPACE

     complicated
       cloud
    █████████
  ███████████
 ████████████
      █████


             f

              ↓


LATENT SPACE

simple independent
coordinates

   •     •
 •   •     •
    •   •
 •     •    •
```

This is an intuition rather than an exact picture, but it captures the goal.

---

# 5.3 What does $f^{-1}$ learn?

Because $f$ is invertible:

$$
h=f(x)
$$

means that we can recover $x$:

$$
\boxed{x=f^{-1}(h)}
$$

The inverse is called the **decoder** in the paper.

Importantly, NICE does **not** train a separate decoder network that approximately reconstructs the input.

The decoder is the exact mathematical inverse of the encoder.

So:

```text
encoder:

x ─────→ h
       f


decoder:

h ─────→ x
       f⁻¹
```

and ideally:

$$
f^{-1}(f(x))=x.
$$

---

# 5.4 Why is the inverse so useful?

Because it gives us a very simple way to generate data.

Suppose we know how to sample from the simple latent distribution:

$$
h\sim p_H(h).
$$

For example, we could generate:

$$
h=(0.3,-1.1,0.2,0.7,\ldots).
$$

Then apply:

$$
x=f^{-1}(h).
$$

The result is a generated data example.

So generation is:

$$
\boxed{
h\sim p_H(h)
\quad\rightarrow\quad
x=f^{-1}(h)
}
$$

The paper explicitly gives this sampling procedure.

---

# 5.5 A simple example tying everything together

Imagine we want to generate pictures of handwritten digits.

### Step 1: Training data

We have:

```text
10,000 handwritten digit images
```

Each image is:

$$
x\in\mathbb R^{784}.
$$

---

### Step 2: NICE learns an encoder

The neural network learns:

$$
h=f_\theta(x).
$$

For one digit:

```text
image x
   ↓
fθ
   ↓
latent vector h
```

---

### Step 3: NICE tries to make $h$ simple

Suppose:

$$
h=(h_1,h_2,\ldots,h_{784}).
$$

We want:

$$
p_H(h)
=
p_{H_1}(h_1)
p_{H_2}(h_2)
\cdots
p_{H_{784}}(h_{784}).
$$

So the latent variables don't need to have complicated dependencies.

---

### Step 4: Train the parameters

The parameters $\theta$ are adjusted so that the observed training images receive high probability:

$$
\max_\theta
\sum_n
\log p_\theta(x^{(n)}).
$$

---

### Step 5: Generate a new digit

Sample independent latent variables:

$$
h_i\sim p_{H_i}.
$$

Then:

$$
x=f_\theta^{-1}(h).
$$

The output is a new image.

So the whole process is:

```text
TRAINING:

digit image x
      ↓
    fθ
      ↓
latent h
      ↓
make h simple + independent
      ↓
maximize likelihood


GENERATION:

sample simple h
      ↓
    fθ⁻¹
      ↓
new digit x
```

---

# 6. Understanding the NICE architecture

Now we can go deeper into the architecture.

---

# 6.1 High-level architecture

The architecture consists conceptually of:

$$
\boxed{
x
\overset{f}{\longrightarrow}
h
}
$$

and

$$
\boxed{
h
\overset{f^{-1}}{\longrightarrow}
x.
}
$$

The paper describes $f$ as the **encoder** and $f^{-1}$ as the **decoder**.

But $f$ itself is not one giant mysterious neural network.

It is composed of multiple simpler transformations:

$$
\boxed{
f=f_L\circ\cdots\circ f_2\circ f_1
}
$$

The paper explains that the forward and backward computations are compositions of the individual layers, while the Jacobian determinant becomes the product of the determinants of those layers.

So:

```text
x
│
▼
f₁
│
▼
f₂
│
▼
f₃
│
▼
...
│
▼
fL
│
▼
h
```

The parameters of the neural networks inside these transformations are what we collectively call $\theta$.

---

# 6.2 Deeper explanation: $x,\mathcal X,h,\mathcal H,\theta$, likelihood, loss, and Jacobians

Let's put all the symbols together.

## $x$

One observed example.

For MNIST:

$$
x\in\mathbb R^{784}.
$$

---

## $\mathcal X$

The space from which the data comes.

For MNIST:

$$
\mathcal X=\mathbb R^{784}.
$$

---

## $h$

The latent representation of $x$:

$$
h=f_\theta(x).
$$

---

## $\mathcal H$

The latent space containing all possible $h$'s.

Because NICE uses an invertible transformation with equal input and output dimensionality:

$$
\mathcal H\approx\mathbb R^D.
$$

---

## $\theta$

All trainable parameters that define the transformation.

In the architecture, these occur inside the neural networks $m^{(1)},m^{(2)},\ldots$ used by the coupling layers, as well as the learned scaling parameters in the final scaling stage.

The paper's experiments, for example, use deep rectified networks for the coupling functions and a learned exponential scaling at the final stage.

---

## Probability density $p_H(h)$

This is the simple distribution we choose for the latent variables.

The paper uses a **factorized** distribution:

$$
p_H(h)
=
\prod_{d=1}^{D}p_{H_d}(h_d).
$$

It uses distributions such as Gaussian or logistic.

---

## Probability density $p_X(x)$

This is the probability density NICE assigns to an actual data point.

Using change of variables:

$$
\boxed{
p_X(x)
=
p_H(f_\theta(x))
\left|
\det J_{f_\theta}(x)
\right|
}
$$

---

## Log-likelihood

Taking the logarithm gives:

$$
\boxed{
\log p_X(x)
=
\log p_H(f_\theta(x))
+
\log
\left|
\det J_{f_\theta}(x)
\right|
}
$$

This is the equation NICE maximizes during training.

For the whole dataset:

$$
\boxed{
\mathcal L(\theta)
=
\sum_{n=1}^{N}
\log p_\theta(x^{(n)})
}
$$

and training tries to maximize this.

---

## Loss function

Machine learning software usually minimizes a loss rather than maximizes an objective.

Therefore we can define:

$$
\boxed{
\text{Loss}
=
-\sum_n\log p_\theta(x^{(n)})
}
$$

So:

```text
maximize log-likelihood
          ≡
minimize negative log-likelihood
```

---

## Where do we calculate Jacobians?

We need the Jacobian of the transformation:

$$
J_f(x)
=
\frac{\partial f(x)}{\partial x}.
$$

But we do **not** want to calculate an expensive determinant of a huge arbitrary matrix.

This is why NICE designs its layers to have special structure.

The paper specifically aims for transformations whose Jacobian determinant is trivial or easy to obtain.

That brings us to coupling layers.

---

# 6.3 Why do we need coupling layers?

This is arguably the cleverest part of NICE.

We want two things simultaneously:

### Requirement 1: The transformation should be powerful

A simple linear transformation isn't enough to model complicated image distributions.

We want something nonlinear and expressive.

Neural networks are excellent for this.

---

### Requirement 2: We need the transformation to be invertible and have an easy Jacobian determinant

A normal neural network does **not** automatically have a convenient inverse.

And calculating:

$$
\det J_f
$$

for an arbitrary high-dimensional neural network can be very expensive.

So the authors design a special building block called a **coupling layer**.

---

## The basic idea

Split $x$ into two parts:

$$
x=(x_1,x_2).
$$

Then define:

$$
\boxed{
y_1=x_1
}
$$

and

$$
\boxed{
y_2=x_2+m(x_1)
}
$$

where $m$ can be a complicated neural network.

This is the additive coupling layer described in the paper.

Notice something interesting:

```text
x1 ───────────────→ y1
 │
 │
 ▼
neural network m
 │
 ▼
m(x1)
 │
x2 ─────── + ─────→ y2
```

The first half is unchanged.

The second half is modified using a neural network that looks at the first half.

---

## Why is the inverse easy?

We have:

$$
y_1=x_1
$$

so immediately:

$$
x_1=y_1.
$$

And:

$$
y_2=x_2+m(x_1).
$$

Therefore:

$$
x_2=y_2-m(x_1).
$$

Since:

$$
x_1=y_1,
$$

we get:

$$
\boxed{
x_2=y_2-m(y_1)
}
$$

So the inverse is:

$$
\boxed{
x_1=y_1,\qquad
x_2=y_2-m(y_1)
}
$$

No neural network inversion is necessary.

The same neural network $m$ is simply evaluated again.

The paper emphasizes that the inverse is therefore as computationally cheap as the forward transformation.

---

## Why is the Jacobian easy?

For:

$$
y_1=x_1
$$

and

$$
y_2=x_2+m(x_1),
$$

the Jacobian has the form:

$$
J=
\begin{bmatrix}
I & 0\
\frac{\partial y_2}{\partial x_1}
&
\frac{\partial y_2}{\partial x_2}
\end{bmatrix}.
$$

Because:

$$
y_2=x_2+m(x_1),
$$

we have:

$$
\frac{\partial y_2}{\partial x_2}=I.
$$

Therefore:

$$
J=
\begin{bmatrix}
I&0\
*&I
\end{bmatrix}.
$$

This is a **triangular matrix**.

The determinant of a triangular matrix is just the product of its diagonal elements.

Every diagonal element here is 1.

Therefore:

$$
\boxed{\det J=1}
$$

This is fantastic computationally.

We have a nonlinear neural network inside the transformation, but the Jacobian determinant is simply:

$$
1.
$$

The paper explicitly derives this triangular structure and unit determinant.

---

# 6.3.1 But there is a problem with coupling layers

You might now think:

> Great! Why don't we just use one coupling layer?

Because one half of the input never changes.

For example:

$$
y_1=x_1.
$$

So $x_1$ does not directly get transformed.

The solution is to use **multiple coupling layers and alternate which half is modified**.

The paper explains that the roles of the two subsets must be exchanged in alternating layers so that every dimension eventually gets modified. It notes that at least three layers are needed for all dimensions to influence one another and that four are generally used.

---

# 6.4 Why do we need rescaling?

There is another subtle problem.

Every additive coupling layer has:

$$
\det J=1.
$$

If we compose many of them:

$$
f=f_L\circ\cdots\circ f_1,
$$

then:

$$
\det J_f
=
\prod_l\det J_{f_l}.
$$

Since every determinant is 1:

$$
\det J_f=1.
$$

Therefore:

$$
\log|\det J_f|=0.
$$

This means the transformation is **volume preserving**.

The paper identifies this as a limitation: a composition of additive coupling layers cannot change the volume.

---

## Why is volume changing important?

Imagine a transformation that needs to stretch one latent direction and compress another.

An additive coupling layer cannot change total volume.

But real data distributions can have very different scales in different directions.

For example:

```text
Data variation:

direction 1: very large
direction 2: small
direction 3: medium
```

We want the model to be able to represent these differences.

So NICE adds a final diagonal scaling transformation:

$$
\boxed{
h_i=S_{ii}h_i^{(4)}
}
$$

where $S$ is diagonal.

The paper introduces this scaling matrix specifically to allow the model to give more weight to some dimensions and less to others.

---

## What does the scaling do to the Jacobian?

Suppose:

$$
h_i=S_{ii}z_i.
$$

Then:

$$
J_S=
\begin{bmatrix}
S_{11}&0&\cdots&0\
0&S_{22}&\cdots&0\
\vdots&&\ddots&\vdots\
0&0&\cdots&S_{DD}
\end{bmatrix}.
$$

The determinant is:

$$
\boxed{
\det J_S
=
\prod_{i=1}^{D}S_{ii}
}
$$

Therefore:

$$
\log|\det J_S|
=
\sum_{i=1}^{D}\log|S_{ii}|.
$$

That is exactly why the final log-likelihood gets the extra scaling term:

$$
\boxed{
\log p_X(x)
=
\sum_i
\left[
\log p_{H_i}(f_i(x))
+
\log|S_{ii}|
\right]
}
$$

as given in the paper.

---

# 6.5 Full breakdown of the NICE architecture

Now let's put everything together.

The architecture used in the experiments consists of:

1. coupling layer 1
2. coupling layer 2
3. coupling layer 3
4. coupling layer 4
5. diagonal scaling

The paper states that the experimental architecture uses four coupling layers followed by a diagonal positive scaling parameterized as $\exp(s)$.

Let's walk through it.

---

## Layer 1

Split the input into two groups:

$$
x=(x_{I_1},x_{I_2}).
$$

Then:

$$
h^{(1)}_{I_1}=x_{I_1}
$$

and

$$
h^{(1)}_{I_2}
=
x_{I_2}
+
m^{(1)}(x_{I_1}).
$$

So the first half controls how the second half changes.

---

## Layer 2

Now switch the roles.

The previously unchanged part gets transformed:

$$
h^{(2)}_{I_2}=h^{(1)}_{I_2}
$$

and

$$
h^{(2)}_{I_1}
=
h^{(1)}_{I_1}
+
m^{(2)}(h^{(1)}_{I_2}).
$$

Now the other half has changed.

---

## Layer 3

Switch again:

$$
h^{(3)}_{I_1}=h^{(2)}_{I_1}
$$

and

$$
h^{(3)}_{I_2}
=
h^{(2)}_{I_2}
+
m^{(3)}(h^{(2)}_{I_1}).
$$

---

## Layer 4

Switch once more:

$$
h^{(4)}_{I_2}=h^{(3)}_{I_2}
$$

and

$$
h^{(4)}_{I_1}
=
h^{(3)}_{I_1}
+
m^{(4)}(h^{(3)}_{I_2}).
$$

These alternating coupling layers allow information from every dimension to influence the others. The paper gives this four-layer structure explicitly in its experimental architecture.

---

## Final scaling layer

Finally:

$$
\boxed{
h=\exp(s)\odot h^{(4)}
}
$$

where:

$$
s=(s_1,\ldots,s_D)
$$

is learned and $\odot$ means element-by-element multiplication.

Therefore:

$$
h_i=e^{s_i}h_i^{(4)}.
$$

The exponential ensures the scaling factors are positive.

The paper uses exactly this form in its experiments.

---

# 7. Where are the neural networks?

This is another point that is easy to misunderstand.

The coupling layer itself is **not** simply a normal neural network.

Instead, it contains a neural network $m$.

For example:

$$
m^{(1)}(x_{I_1})
$$

could be a deep MLP.

It receives one part of the input and outputs a vector that modifies the other part.

So:

```text
                coupling layer

       x1 ────────────────→ y1
        │
        │
        ▼
   Neural network
       m(x1)
        │
        ▼
       (+)
        ▲
        │
       x2
        │
        ▼
       y2
```

The paper emphasizes that $m$ can be arbitrarily complex; in the experiments it is implemented using deep rectified neural networks.

This is the key trick:

> **The neural network can be complicated; the overall transformation remains easy to invert because of the way the neural network is placed inside the coupling layer.**

---

# 8. Why can the model be nonlinear if the Jacobian is so simple?

This is one of the most beautiful ideas in NICE.

Consider:

$$
y_2=x_2+m(x_1).
$$

Suppose:

$$
m(x_1)
$$

is an extremely complicated deep neural network.

Then $y_2$ can be a very complicated nonlinear function of $x_1$.

But:

$$
\frac{\partial y_2}{\partial x_2}=1.
$$

That's all we need for the determinant.

So the model gets:

```text
complex nonlinear transformation
            +
easy inverse
            +
easy Jacobian determinant
```

That is the central architectural insight of NICE.

---

# 9. Putting probability and architecture together

Now let's connect the neural network to the probability equation.

We have:

$$
h=f_\theta(x).
$$

Then calculate the latent probability:

$$
p_H(h).
$$

Because the latent distribution factorizes:

$$
p_H(h)
=
\prod_i p_{H_i}(h_i).
$$

Then calculate the Jacobian determinant:

$$
\left|\det J_{f_\theta}(x)\right|.
$$

Therefore:

$$
\boxed{
p_X(x)
=
p_H(f_\theta(x))
\left|
\det J_{f_\theta}(x)
\right|
}
$$

Taking logs:

$$
\boxed{
\log p_X(x)
=
\log p_H(f_\theta(x))
+
\log|\det J_{f_\theta}(x)|
}
$$

Because the latent distribution factorizes:

$$
\log p_H(h)
=
\sum_i\log p_{H_i}(h_i).
$$

And because the coupling layers have determinant 1, only the final scaling contributes to the determinant.

Therefore, in the NICE architecture:

$$
\boxed{
\log p_X(x)
=
\sum_i
\left[
\log p_{H_i}(h_i)
+
\log|S_{ii}|
\right]
}
$$

This is what makes the exact log-likelihood tractable.

---

# 10. What happens during training?

For each training example $x$:

### Step 1: Encode

Calculate:

$$
h=f_\theta(x).
$$

---

### Step 2: Evaluate the latent probability

Calculate:

$$
\log p_H(h).
$$

Because the latent distribution is factorized:

$$
\log p_H(h)
=
\sum_i\log p_{H_i}(h_i).
$$

---

### Step 3: Calculate the Jacobian contribution

For the coupling layers:

$$
\log|\det J|=0.
$$

For the final scaling:

$$
\log|\det J|
=
\sum_i\log|S_{ii}|.
$$

---

### Step 4: Calculate log-likelihood

$$
\boxed{
\log p_X(x)
=
\log p_H(h)
+
\log|\det J_f(x)|
}
$$

---

### Step 5: Update $\theta$

The model changes its parameters so that the training examples receive higher likelihood.

In the experiments, the authors maximize this log-likelihood using Adam.

---

# 11. What happens during generation?

Generation is almost the reverse.

Instead of starting with $x$, start with $h$.

Because $p_H$ is simple, we can sample:

$$
h\sim p_H(h).
$$

Since the latent dimensions are independent, this is especially easy:

$$
h_1\sim p_{H_1},
$$

$$
h_2\sim p_{H_2},
$$

and so on.

Then:

$$
\boxed{
x=f^{-1}(h)
}
$$

The result is a generated data example.

The paper describes this as ancestral sampling in the graphical model:

$$
H\rightarrow X.
$$

So:

```text
                 TRAINING

              x
              │
              ▼
           encoder
              f
              │
              ▼
              h
              │
              ▼
      simple independent
          distribution


                 GENERATION

      independent h
              │
              ▼
           decoder
             f⁻¹
              │
              ▼
          generated x
```

---

# 12. A complete toy example

Let's make the entire idea concrete with only two dimensions.

Suppose our data is:

$$
x=(x_1,x_2).
$$

Imagine that $x_1$ and $x_2$ are highly dependent.

For example:

$$
x_2\approx x_1^2+\text{noise}.
$$

So the data might lie approximately along a curved parabola:

```text
x2
│
│          •
│       •
│     •
│   •
│ •
└──────────────── x1
```

Modeling this distribution directly is difficult.

---

## Step 1: Transform it

NICE learns something like:

$$
h_1=x_1
$$

and

$$
h_2=x_2-m(x_1).
$$

If the network learns:

$$
m(x_1)\approx x_1^2,
$$

then:

$$
h_2
=
x_2-x_1^2.
$$

Now the curved relationship has been removed.

Instead of:

$$
x_2\approx x_1^2,
$$

we have:

$$
h_2\approx0+\text{noise}.
$$

The complicated distribution has become simpler.

---

## Step 2: Give $h$ a simple distribution

We might model:

$$
h_1\sim\mathcal N(0,1)
$$

and:

$$
h_2\sim\mathcal N(0,1).
$$

And because they factorize:

$$
p_H(h)
=
p_{H_1}(h_1)p_{H_2}(h_2).
$$

---

## Step 3: Generate a new example

Sample:

$$
h_1=0.7
$$

and:

$$
h_2=-0.1.
$$

The inverse transformation is:

$$
x_1=h_1
$$

and:

$$
x_2=h_2+m(h_1).
$$

So approximately:

$$
x_1=0.7
$$

and:

$$
x_2=-0.1+(0.7)^2.
$$

Therefore:

$$
x_2=0.39.
$$

We generated:

$$
\boxed{x=(0.7,0.39)}.
$$

Notice what happened:

```text
simple random variables
       h
       │
       │ f⁻¹
       ▼
complicated data
       x
```

That is generative modeling.

---

# 13. The whole paper in one mathematical story

We can now compress the entire paper into a chain of equations.

We start with observed data:

$$
x\sim p_X(x).
$$

We learn an invertible transformation:

$$
\boxed{h=f_\theta(x)}.
$$

We want the latent distribution to factorize:

$$
\boxed{
p_H(h)=\prod_d p_{H_d}(h_d)
}
$$

The change-of-variables formula gives:

$$
\boxed{
p_X(x)
=
p_H(f_\theta(x))
\left|
\det J_{f_\theta}(x)
\right|
}
$$

Taking logs:

$$
\boxed{
\log p_X(x)
=
\log p_H(f_\theta(x))
+
\log|\det J_{f_\theta}(x)|
}
$$

We choose the architecture so that:

* $f_\theta$ is invertible;
* $f_\theta^{-1}$ is easy to compute;
* the Jacobian determinant is easy to calculate.

The coupling layer:

$$
y_1=x_1
$$

$$
y_2=x_2+m(x_1)
$$

gives:

$$
\det J=1.
$$

Several coupling layers are composed together, alternating the partitions so all dimensions can influence each other.

Then a final scaling layer:

$$
h_i=e^{s_i}h_i^{(4)}
$$

allows the total transformation to change volume.

Finally, training maximizes:

$$
\boxed{
\sum_n\log p_\theta(x^{(n)})
}
$$

and generation is:

$$
\boxed{
h\sim p_H(h),
\qquad
x=f_\theta^{-1}(h).
}
$$

That is NICE.

---
# 14. What Do the Learned Scaling Factors Mean? Connecting NICE to PCA and Manifolds

We have now seen that NICE ends its sequence of coupling layers with a diagonal scaling transformation:

$$
h_i=S_{ii}h_i^{(4)}.
$$

At first, $S_{ii}$ might look like just another technical parameter needed to make the model more flexible.

But the paper gives these numbers an interesting interpretation: they tell us something about **how much variation the model thinks exists in each latent dimension**.

The authors relate these scaling factors to the **eigenspectrum of PCA** and use them to identify the important dimensions of the learned representation.

This section explains that statement carefully.

---

## 14.1 First: what does a scaling factor $S_{ii}$ actually do?

Recall the final scaling layer:

$$
h_i=S_{ii}h_i^{(4)}.
$$

For simplicity, imagine only two dimensions:

$$
h_1=S_{11}h_1^{(4)}
$$

$$
h_2=S_{22}h_2^{(4)}.
$$

Suppose:

$$
S_{11}=10
$$

and:

$$
S_{22}=0.5.
$$

Then the transformation does this:

```text
dimension 1:
h₁ = 10 × h₁⁽⁴⁾

dimension 2:
h₂ = 0.5 × h₂⁽⁴⁾
```

So dimension 1 is stretched strongly, while dimension 2 is compressed.

Geometrically:

```text
Before scaling:

       •
    •     •
   •       •
    •     •
       •


After scaling:

              •
          •       •
       •             •
          •       •
              •
```

The cloud has been stretched more in one direction than another.

Therefore, $S_{ii}$ determines how much the transformation **scales the $i$-th latent direction**.

---

# 14.2 Why is this related to the amount of variation?

Imagine that the data varies enormously in one direction but only slightly in another.

For example, suppose we have images of faces.

Perhaps there is a large amount of variation due to:

* face orientation,
* lighting,
* expression,

while another direction represents a very small detail.

Conceptually:

```text
Large variation

        ←──────────────→
             data
        ←──────────────→


Small variation

             ↑
             │
             │
             ↓
             data
```

A representation that captures the important structure of the data should therefore have some latent dimensions associated with large-scale variation and others associated with small-scale variation.

The scaling factors give us information about this.

The paper states that the scaling factors can be related to the eigenspectrum of PCA and therefore indicate how much variation is present in each latent dimension.

---

# 14.3 A quick reminder: what does PCA do?

To understand the comparison, let's briefly recall PCA.

**Principal Component Analysis $PCA$** tries to find directions in the data along which the data varies the most.

Suppose we have a cloud of points shaped approximately like this:

```text
                  •
               •
            •
         •
      •
   •
```

The data has much more variation along the diagonal direction than in the perpendicular direction.

PCA finds:

$$
\text{first principal direction}
$$

as the direction of greatest variation.

Then it finds:

$$
\text{second principal direction}
$$

as another direction, orthogonal to the first, with the next largest amount of variation.

And so on.

Each principal direction has an associated **eigenvalue**.

Very roughly:

$$
\boxed{
\text{large PCA eigenvalue}
\quad\Longrightarrow\quad
\text{large amount of variation}
}
$$

and:

$$
\boxed{
\text{small PCA eigenvalue}
\quad\Longrightarrow\quad
\text{small amount of variation}.
}
$$

The collection of these eigenvalues is called the **eigenspectrum**.

For example:

$$
\lambda_1=100,
\qquad
\lambda_2=20,
\qquad
\lambda_3=2,
\qquad
\lambda_4=0.1.
$$

This tells us that the first dimension contains much more variation than the fourth.

---

# 14.4 So how does NICE have something similar to a PCA eigenspectrum?

NICE is much more complicated than PCA.

PCA essentially learns a **linear** transformation.

NICE learns a highly **nonlinear** transformation:

$$
h=f(x).
$$

But after the nonlinear transformation, NICE has its latent dimensions:

$$
h_1,h_2,\ldots,h_D.
$$

The scaling factors:

$$
S_{11},S_{22},\ldots,S_{DD}
$$

tell us how the model scales those dimensions.

The paper therefore says that these scaling factors can be related to the eigenspectrum of PCA: they give information about how important the different latent dimensions are.

This is not saying:

> "NICE is secretly doing PCA."

It is saying something more subtle:

> **The scaling factors in NICE play a role analogous to the variance/eigenvalue information that PCA gives us.**

NICE is effectively learning a **nonlinear version of a coordinate system** in which we can examine how important different directions are.

---

# 14.5 The surprising part: larger $S_{ii}$ means less important

This statement from the paper can initially feel backwards:

> "the larger $S_{ii}$ is, the less important the dimension $i$ is."

You might reasonably think:

> If the model stretches dimension $i$ by a lot, surely that dimension must be important?

But we need to remember that $S_{ii}$ belongs to the transformation:

$$
h_i=S_{ii}h_i^{(4)}.
$$

The quantity that the paper uses to visualize the variation is actually related to the **inverse** of the scaling:

$$
\boxed{
\sigma_i=S_{ii}^{-1}.
}
$$

The appendix makes this relationship explicit: when the prior and diagonal scaling are considered together, $\sigma_d=S_{dd}^{-1}$ can be interpreted as the scale parameter of each independent component.

So:

$$
S_{ii}\text{ large}
$$

means:

$$
S_{ii}^{-1}\text{ small}.
$$

And the paper interprets the larger $\sigma_i$ values as corresponding to dimensions with larger variations.

Therefore:

$$
\boxed{
\text{large }S_{ii}
\quad\Longrightarrow\quad
\text{small }S_{ii}^{-1}
\quad\Longrightarrow\quad
\text{less variation}
}
$$

whereas:

$$
\boxed{
\text{small }S_{ii}
\quad\Longrightarrow\quad
\text{large }S_{ii}^{-1}
\quad\Longrightarrow\quad
\text{more variation}.
}
$$

This is the key to understanding the sentence in the paper.

---

# 14.6 Why does the inverse scaling $\sigma_i=S_{ii}^{-1}$ behave like a standard deviation?

Let's see this mathematically.

Suppose the latent variable before the final scaling is:

$$
z_i=h_i^{(4)}.
$$

The final transformation is:

$$
h_i=S_{ii}z_i.
$$

Therefore:

$$
z_i=S_{ii}^{-1}h_i.
$$

Define:

$$
\boxed{
\sigma_i=S_{ii}^{-1}.
}
$$

Then:

$$
z_i=\sigma_i h_i.
$$

Now suppose $h_i$ comes from a standard normal distribution:

$$
h_i\sim\mathcal N(0,1).
$$

Then multiplying it by $\sigma_i$ gives:

$$
z_i\sim\mathcal N(0,\sigma_i^2).
$$

So $\sigma_i$ acts like a **standard deviation** or scale parameter.

For example:

### If

$$
\sigma_i=10,
$$

then:

$$
z_i\sim\mathcal N(0,100),
$$

so there is a lot of variation in that dimension.

### If

$$
\sigma_i=0.1,
$$

then:

$$
z_i\sim\mathcal N(0,0.01),
$$

so there is very little variation.

Therefore:

$$
\boxed{
\text{large }\sigma_i
\Rightarrow
\text{large variation}
}
$$

and because:

$$
\sigma_i=S_{ii}^{-1},
$$

we have:

$$
\boxed{
\text{small }S_{ii}
\Rightarrow
\text{large variation}.
}
$$

This is exactly why the paper says that larger $S_{ii}$ corresponds to a less important dimension.

---

# 14.7 A concrete numerical example

Suppose NICE learns the following scaling factors:

$$
S=
\operatorname{diag}(1,2,10,100).
$$

Then:

$$
S_{11}=1,
$$

$$
S_{22}=2,
$$

$$
S_{33}=10,
$$

$$
S_{44}=100.
$$

Calculate the inverse scales:

$$
\sigma_i=S_{ii}^{-1}.
$$

Therefore:

$$
\sigma_1=1,
$$

$$
\sigma_2=0.5,
$$

$$
\sigma_3=0.1,
$$

$$
\sigma_4=0.01.
$$

So the interpretation is approximately:

| Latent dimension | $S_{ii}$ | $\sigma_i=S_{ii}^{-1}$ | Variation  |
| ---------------- | -------: | ---------------------: | ---------- |
| 1                |        1 |                      1 | high       |
| 2                |        2 |                    0.5 | moderate   |
| 3                |       10 |                    0.1 | small      |
| 4                |      100 |                   0.01 | very small |

Thus:

$$
\boxed{
\text{dimension 1 is more important than dimension 4}
}
$$

even though:

$$
S_{44}>S_{11}.
$$

The reason is that the relevant variation scale is associated with:

$$
S_{ii}^{-1}.
$$

---

# 14.8 What does the "manifold" have to do with this?

Now we reach the next statement in the paper:

> "The important dimensions of the spectrum can be viewed as a manifold learned by the algorithm."

This sounds much more complicated than it really is.

A **manifold** is, informally, a lower-dimensional structure embedded inside a higher-dimensional space.

Consider a sheet of paper sitting in 3D space.

The sheet itself is two-dimensional:

$$
(x,y)
$$

but it is embedded in three-dimensional space:

$$
(x,y,z).
$$

So although we need three coordinates to describe where the sheet sits in the room, the points on the sheet really only have two degrees of freedom.

That sheet is an example of a manifold.

---

# 14.9 How can high-dimensional data have a lower-dimensional manifold?

Consider images.

A $28\times28$ image has:

$$
784
$$

pixel values.

So mathematically:

$$
x\in\mathbb R^{784}.
$$

But not every possible 784-dimensional vector looks like a handwritten digit.

For example, this is mathematically a possible vector:

```text
[0.73, 0.11, 0.92, 0.37, ..., 0.48]
```

but it probably doesn't look like a meaningful handwritten digit.

Real handwritten digits occupy only a tiny structured subset of the enormous 784-dimensional pixel space.

We can imagine:

```text
             784-dimensional space

                 •
              •     •
            •         •
          •             •
           \           /
            \         /
             \_______/

       actual handwritten digits
       occupy a structured region
```

That structured region is what we informally call a **data manifold**.

---

# 14.10 What does NICE do to this manifold?

NICE learns a nonlinear transformation:

$$
h=f(x).
$$

The transformation attempts to organize the data so that its important structure appears in a set of latent dimensions.

Some latent dimensions may contain a lot of variation:

$$
h_1,h_2,h_3,\ldots
$$

while others may contain very little:

$$
h_{100},h_{101},\ldots
$$

The paper suggests that the dimensions associated with substantial variation can be viewed as representing the learned manifold.

So, informally:

$$
\boxed{
\text{important latent dimensions}
\approx
\text{coordinates describing the learned data manifold}.
}
$$

---

# 14.11 This is the nonlinear analogue of PCA

Now the connection to PCA becomes clearer.

PCA might tell us:

$$
\lambda_1\gg\lambda_2\gg\lambda_3\gg\lambda_4.
$$

Therefore:

```text
PC1: very important
PC2: important
PC3: less important
PC4: almost irrelevant
```

NICE can produce a similar ordering through its learned latent scales.

Instead of PCA's eigenvalues, we can examine:

$$
\sigma_i=S_{ii}^{-1}.
$$

For example:

$$
\sigma_1=10,
\qquad
\sigma_2=5,
\qquad
\sigma_3=1,
\qquad
\sigma_4=0.1.
$$

This suggests:

```text
latent dimension 1: lots of variation
latent dimension 2: lots of variation
latent dimension 3: some variation
latent dimension 4: very little variation
```

The authors explicitly describe these $\sigma_d$ values as the nonlinear equivalent of the PCA eigenspectrum.

So a useful mental translation is:

$$
\boxed{
\text{PCA eigenvalues}
\quad\leftrightarrow\quad
\text{NICE latent scales } \sigma_i
}
$$

with the important caveat that NICE's transformation is nonlinear whereas PCA is linear.

---

# 14.12 Why doesn't NICE simply make $S_{ii}=0$?

Here is another subtle point in the paper.

If:

$$
S_{ii}=0,
$$

then:

$$
h_i=S_{ii}h_i^{(4)}=0.
$$

The $i$-th dimension has been completely collapsed.

For example:

$$
z_i
\rightarrow
0.
$$

Many different values of $z_i$ would produce exactly the same output:

$$
z_i=1\rightarrow0
$$

$$
z_i=5\rightarrow0
$$

$$
z_i=-10\rightarrow0.
$$

That means we cannot recover the original value.

The transformation is no longer invertible.

So:

$$
\boxed{
S_{ii}=0
\quad\Rightarrow\quad
\text{information is destroyed}.
}
$$

And that would violate the central requirement of NICE: $f$ must be invertible.

---

# 14.13 But why doesn't the optimization push $S_{ii}$ toward zero?

This is where the two terms in the log-likelihood become important.

Recall:

$$
\log p_X(x)
=
\log p_H(h)
+
\log|\det J_f(x)|.
$$

For the scaling layer:

$$
\log|\det S|
=
\sum_i\log|S_{ii}|.
$$

So the log-likelihood contains terms of the form:

$$
\boxed{
\log p_H(h_i)+\log|S_{ii}|.
}
$$

There are therefore two competing effects.

---

## Effect 1: the prior term

Suppose the latent prior is a standard Gaussian:

$$
p_H(h_i)
\propto
e^{-h_i^2/2}.
$$

Its log-density is:

$$
\log p_H(h_i)
=
-\frac12h_i^2+C.
$$

Recall:

$$
h_i=S_{ii}z_i.
$$

If we make $S_{ii}$ smaller, then for a fixed $z_i$:

$$
h_i=S_{ii}z_i
$$

also becomes smaller.

A smaller $h_i$ is closer to the center of the standard Gaussian.

Therefore the prior term:

$$
\log p_H(h_i)
$$

can become larger.

So the prior term has an incentive to make:

$$
\boxed{S_{ii}\text{ smaller}.}
$$

This is what the paper means when it says:

> "The prior term encourages $S_{ii}$ to be small."

---

# 14.14 Then why doesn't $S_{ii}$ become zero?

Because of the second term:

$$
\log|S_{ii}|.
$$

As:

$$
S_{ii}\rightarrow0^+,
$$

we have:

$$
\log S_{ii}\rightarrow-\infty.
$$

For example:

$$
\log(1)=0
$$

but:

$$
\log(0.1)\approx-2.30
$$

and:

$$
\log(0.01)\approx-4.61.
$$

As we continue toward zero:

$$
\log(0.000001)\approx-13.82.
$$

And mathematically:

$$
\boxed{
\lim_{S_{ii}\rightarrow0^+}\log S_{ii}=-\infty.
}
$$

So the determinant term strongly penalizes $S_{ii}$ becoming zero.

Therefore we have a competition:

```text
prior term
    │
    │ wants Sii smaller
    ▼
  Sii → 0


determinant term
    │
    │ strongly punishes Sii → 0
    ▼
  Sii cannot collapse to zero
```

The paper summarizes exactly this competition by saying that the prior term encourages $S_{ii}$ to be small, while the determinant term prevents it from reaching zero.

---

# 14.15 This is actually an important part of how NICE learns a manifold

Now the pieces fit together.

Suppose a particular dimension of the data contains very little variation.

The model has an incentive to compress that dimension strongly.

So it can learn:

$$
S_{ii}\gg1
$$

which means:

$$
\sigma_i=S_{ii}^{-1}\ll1.
$$

Therefore that latent dimension corresponds to very little variation.

On the other hand, suppose another dimension captures a major source of variation.

The model can learn a smaller:

$$
S_{ii}.
$$

Then:

$$
\sigma_i=S_{ii}^{-1}
$$

is larger, indicating greater variation.

So the model can effectively learn something like:

```text
                latent dimensions

      high variation              low variation
           │                           │
           ▼                           ▼

       h₁       h₂       h₃       h₄       h₅
       ███      ███      ██       █        ▏
```

The dimensions with substantial variation form the important part of the representation.

The paper interprets these important dimensions as relating to the learned manifold.

---

# 14.16 A complete numerical example

Suppose a trained NICE model has:

$$
S_{11}=0.2,
$$

$$
S_{22}=0.5,
$$

$$
S_{33}=2,
$$

$$
S_{44}=20.
$$

Then:

$$
\sigma_i=S_{ii}^{-1}
$$

gives:

$$
\sigma_1=5,
$$

$$
\sigma_2=2,
$$

$$
\sigma_3=0.5,
$$

$$
\sigma_4=0.05.
$$

We can interpret this approximately as:

| Dimension | $S_{ii}$ | $\sigma_i=S_{ii}^{-1}$ | Interpretation       |
| --------- | -------: | ---------------------: | -------------------- |
| $h_1$     |      0.2 |                      5 | Very large variation |
| $h_2$     |      0.5 |                      2 | Large variation      |
| $h_3$     |        2 |                    0.5 | Small variation      |
| $h_4$     |       20 |                   0.05 | Very small variation |

Therefore the first two dimensions are much more important for describing the variation in the data.

We could imagine that:

$$
(h_1,h_2)
$$

approximately describe the important structure of the learned manifold, while $h_3,h_4$ correspond to much smaller variations.

---

# 14.17 Why is this useful?

This gives us a way to inspect what the model has learned.

Instead of looking only at generated images, we can examine:

$$
\sigma_1,\sigma_2,\ldots,\sigma_D.
$$

The paper plots these values after sorting them by size. Large values correspond to dimensions where the model chooses to allow larger variations, thereby highlighting the learned manifold structure.

So we can get something like:

```text
variation
   │
10 │ █
   │ █
 5 │ █ █
   │ █ █
 1 │ █ █ █
   │ █ █ █
0.1│ █ █ █ ▏
   └──────────────────
      1 2 3 4 5 ...
          latent dimension
```

A sharp drop would suggest that only a relatively small number of latent dimensions contain most of the meaningful variation.

This is analogous to looking for a sharp drop in the PCA eigenspectrum.

---

# 14.18 One subtle but important point

We should be careful with the phrase **"important dimension."**

It does **not** necessarily mean:

> "This latent variable corresponds exactly to one human-interpretable feature such as smile, gender, or rotation."

The paper's statement is about the **amount of variation** represented by the dimension.

So a large $\sigma_i$ means that the model assigns a larger scale to that latent component.

It does not automatically mean that a human can look at $h_i$ and say:

> "Ah, this is the smile variable."

The learned dimensions can still be complicated combinations of underlying factors.

---

# 14.19 The connection to the manifold visualization in the paper

The paper also gives a visualization of the learned manifold.

It takes a sphere in the latent space, applies a random rotation, and then maps that structure back to data space using:

$$
f^{-1}.
$$

The resulting structure illustrates part of the manifold learned by the model.

This is another way to understand the idea.

The model has learned a complicated mapping between:

$$
\text{latent space}
$$

and:

$$
\text{data space}.
$$

The latent space gives us relatively simple coordinates, while the inverse transformation bends those coordinates into the complicated structure occupied by real data.

So:

```text
simple latent geometry

        ○
     ○     ○
   ○         ○
     ○     ○
        ○
          │
          │ f⁻¹
          ▼

complicated data manifold

       ╭──────╮
    ╭──╯      ╰──╮
  ╭─╯             ╰─╮
  ╰─────────────────╯
```

The second object can be complicated even though the first is simple.

That is the power of the nonlinear transformation.

---

# 14.20 The big connection: PCA → NICE

We can now summarize the analogy.

### PCA

PCA learns:

$$
\boxed{
\text{linear directions of maximum variance}
}
$$

and gives an eigenspectrum:

$$
\lambda_1,\lambda_2,\ldots,\lambda_D.
$$

Large eigenvalues indicate dimensions containing substantial variation.

---

### NICE

NICE learns:

$$
\boxed{
\text{a nonlinear coordinate transformation}
}
$$

and its scaling factors provide scale information:

$$
S_{11},S_{22},\ldots,S_{DD}.
$$

The paper considers:

$$
\boxed{
\sigma_i=S_{ii}^{-1}
}
$$

as the scale of the corresponding independent component. Large $\sigma_i$ means larger variation.

So, conceptually:

$$
\boxed{
\text{PCA eigenvalues}
\longleftrightarrow
\text{NICE latent scales}
}
$$

with NICE providing a **nonlinear** analogue.

---

# 14.21 The entire idea in one diagram

Here is perhaps the most useful picture to remember:

```text
                         NICE

              complicated data x
                       │
                       │
                       ▼
              nonlinear encoder f
                       │
                       ▼
              coupling layers
                       │
              ┌────────┴────────┐
              │                 │
        latent dimension 1   latent dimension D
              │                 │
              ▼                 ▼
          scale S₁₁          scale SDD
              │                 │
              └────────┬────────┘
                       │
                       ▼
                 latent h


         Examine inverse scales:

              σᵢ = 1 / Sᵢᵢ

                       │
             ┌─────────┴─────────┐
             │                   │
       large σᵢ              small σᵢ
             │                   │
             ▼                   ▼
      lots of variation     little variation
             │                   │
             ▼                   ▼
     important dimensions    less important
             │
             ▼
       learned manifold
```

---

# 14.22 The most important equations from this section

There are four equations worth remembering.

### 1. Final scaling

$$
\boxed{
h_i=S_{ii}h_i^{(4)}
}
$$

---

### 2. Inverse scale

$$
\boxed{
\sigma_i=S_{ii}^{-1}
}
$$

---

### 3. Variation interpretation

$$
\boxed{
\text{large }\sigma_i
\quad\Longrightarrow\quad
\text{large variation in latent dimension }i
}
$$

and therefore:

$$
\boxed{
\text{large }S_{ii}
\quad\Longrightarrow\quad
\text{small variation / less important dimension}.
}
$$

---

### 4. The competing terms in the likelihood

$$
\boxed{
\log p_X(x)
=
\underbrace{\log p_H(h)}_{\text{prior term}}
+
\underbrace{\sum_i\log|S_{ii}|}_{\text{determinant term}}
}
$$

The prior term tends to favor smaller $S_{ii}$, while:

$$
\log S_{ii}\rightarrow-\infty
\quad\text{as}\quad
S_{ii}\rightarrow0^+,
$$

so the determinant term prevents the scaling from collapsing to zero.

---

# 14.23 In plain English

The easiest way to say all of this is:

> **The final scaling layer tells NICE how much variation to assign to each latent direction. The inverse scaling $\sigma_i=1/S_{ii}$ behaves like a scale or standard-deviation parameter: a large $\sigma_i$ means that the corresponding latent dimension contains a lot of variation, while a small $\sigma_i$ means that it contains little variation. This is analogous to PCA, where large eigenvalues indicate directions of high variance. The important latent dimensions can therefore be viewed as describing the lower-dimensional manifold on which the data lies.**
>
> **The model cannot simply make an unimportant dimension disappear by setting $S_{ii}=0$, because that would destroy invertibility. Moreover, the likelihood contains the determinant term $\log S_{ii}$, which goes to $-\infty$ as $S_{ii}$ approaches zero. Thus the prior term pushes toward compression, while the determinant term prevents complete collapse.**

This is why the scaling layer is not just a technical trick for making NICE more expressive. It also gives the model a way to **learn and reveal the relative importance of its latent dimensions**, providing a nonlinear analogue of the variance spectrum that PCA gives us.

---
# 15. The most important intuition to remember

If you are an undergraduate encountering normalizing flows for the first time, I would remember **five ideas**.

### 1. The data is complicated

$$
x
$$

contains correlations and complicated structure.

---

### 2. We transform it

$$
h=f_\theta(x).
$$

The transformation is learned.

---

### 3. We want the latent representation to be simple

$$
p_H(h)=\prod_i p_{H_i}(h_i).
$$

So the latent variables are independent.

---

### 4. We make the transformation invertible

$$
x=f_\theta^{-1}(h).
$$

This means we can go in both directions without losing information.

---

### 5. We carefully design the neural network

We want it to be:

```text
powerful enough to model complicated data
                  +
invertible
                  +
easy to calculate its Jacobian determinant
```

The coupling layers provide exactly this combination.

---

# 16. One final mental picture

Think of NICE as a machine that **untangles a complicated probability distribution**.

Imagine taking a tangled ball of string:

```text
      tangled data

   @@@  @@@
 @@  @@@@@  @@
   @@@@@@@
 @@   @@@  @
```

The encoder $f$ learns how to untangle it:

```text
        f

      ↓↓↓↓↓

   •   •   •
 •   •   •   •
   •   •   •
 •   •   •   •
```

The resulting latent variables are easier to model because they are independent.

Then the decoder $f^{-1}$ can take a point from the simple space and bend it back into the complicated data space.

```text
simple latent point
        h
        │
        │ f⁻¹
        ▼
complicated realistic example
        x
```

And the mathematical trick that makes the whole thing possible is the change-of-variables formula:

$$
\boxed{
p_X(x)
=
p_H(f_\theta(x))
\left|
\det
\frac{\partial f_\theta(x)}{\partial x}
\right|
}
$$

while the architectural trick that makes this computationally practical is the **coupling layer**.

That combination of invertibility, a simple latent distribution, a tractable Jacobian, and neural-network flexibility is the essence of the NICE paper. The authors summarize their contribution as a flexible architecture for learning a highly nonlinear bijective transformation into a factorized space while directly maximizing log-likelihood and retaining efficient unbiased sampling.

---

## Cheat sheet

| Symbol               | Meaning                                                     |
| -------------------- | ----------------------------------------------------------- |
| $x$                  | One observed data example                                   |
| $\mathcal X$         | Data space                                                  |
| $h$                  | Latent representation of $x$                                |
| $\mathcal H$         | Latent space                                                |
| $f$                  | Encoder: $x\rightarrow h$                                   |
| $f^{-1}$             | Decoder: $h\rightarrow x$                                   |
| $\theta$             | Trainable parameters of the model/neural networks           |
| $p_H(h)$             | Simple latent prior distribution                            |
| $p_X(x)$             | Probability density assigned to data                        |
| $J_f$                | Jacobian of $f$                                             |
| $\det J_f$           | Local volume-change factor                                  |
| $p_\theta(D)$        | Likelihood of the training data                             |
| $\log p_\theta(D)$   | Log-likelihood optimized by NICE                            |
| $m(x_1)$             | Neural network inside a coupling layer                      |
| $S$                  | Final learned diagonal scaling                              |
| coupling layer       | Invertible nonlinear building block with tractable Jacobian |
| latent factorization | $p_H(h)=\prod_i p_{H_i}(h_i)$                               |

### The single most important equation

$$
\boxed{
\log p_X(x)
=
\underbrace{\log p_H(f_\theta(x))}_{\text{How plausible is the latent code?}}
+
\underbrace{\log|\det J_{f_\theta}(x)|}_{\text{How does the transformation change volume?}}
}
$$

### The single most important architecture

$$
\boxed{
x
\xrightarrow{\text{coupling }1}
h^{(1)}
\xrightarrow{\text{coupling }2}
h^{(2)}
\xrightarrow{\text{coupling }3}
h^{(3)}
\xrightarrow{\text{coupling }4}
h^{(4)}
\xrightarrow{\text{scaling}}
h
}
$$

and in reverse:

$$
\boxed{
h
\xrightarrow{\text{inverse scaling}}
h^{(4)}
\xrightarrow{\text{inverse coupling }4}
\cdots
\xrightarrow{\text{inverse coupling }1}
x.
}
$$

So if you remember only one sentence from this entire explanation, remember:

> **NICE learns an invertible neural transformation that converts complicated data $x$ into simple independent latent variables $h$, while designing the transformation so that both its inverse and its Jacobian determinant are easy to compute.**
