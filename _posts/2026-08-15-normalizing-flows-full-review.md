---
layout: post
title: "Normalizing Flows: My Study Notes"
date: 2026-08-15
category: pic
---

*A study-first guide to understanding normalizing flows without drowning in notation.*

---

## 0. What am I actually trying to understand?

Before touching the equations, I want to know the problem.

I want a model that can learn a complicated probability distribution over $x$.

For example, suppose

$$
x \in \mathbb R^D
$$

represents an image, and the distribution of natural images is some incredibly complicated density

$$
p_X(x).
$$

The problem is:

> How can I construct a complicated distribution while still being able to calculate its density exactly and generate samples from it efficiently?

Normalizing flows give one elegant answer:

> **Start with a simple distribution and continuously deform it into the complicated distribution I want, while making every deformation invertible and keeping track of how the deformation changes density.**

That sentence is the entire subject in miniature.

Everything else is machinery for making that idea computationally possible.

The original NICE paper describes essentially this goal: learn a nonlinear invertible transformation that maps data into a simpler, factorized latent distribution, while keeping the Jacobian determinant and inverse tractable.

---

# 1. The mental picture first

Imagine that my data distribution looks like this:

```text
                    data distribution

                         ●●
                      ●●●●●
                    ●●●●●
                  ●●●
                ●●
          ●●●
       ●●●
```

It might have:

* curved structure,
* multiple modes,
* correlations,
* complicated geometry.

But my base distribution is something easy:

$$
z\sim\mathcal N(0,I).
$$

Visually:

```text
             Gaussian

               ●
            ●●●●●
          ●●●●●●●
            ●●●●
               ●
```

I want to learn an invertible function

$$
f:X\rightarrow Z
$$

such that

$$
z=f(x).
$$

Then the inverse

$$
x=f^{-1}(z)
$$

takes a simple Gaussian sample and turns it into a sample resembling the data.

So there are really two directions:

### Data → latent

$$
x \xrightarrow{f} z
$$

This is useful for:

* evaluating likelihood,
* inference,
* obtaining latent representations.

### Latent → data

$$
z \xrightarrow{f^{-1}} x
$$

This is useful for:

* generation,
* sampling.

Real NVP explicitly describes this two-way relationship: $z=f(x)$ for inference and $x=f^{-1}(z)$ for generation.

---

# 2. The first load-bearing concept: change of variables

This is the mathematical heart of normalizing flows.

If I understand only one equation initially, it should be this:

$$
p_X(x)
=
p_Z(f(x))
\left|
\det
\frac{\partial f(x)}{\partial x}
\right|.
$$

Or in log form,

$$
\log p_X(x)
=
\log p_Z(f(x))
+
\log
\left|
\det
\frac{\partial f(x)}{\partial x}
\right|.
$$

This is the change-of-variables formula used by Real NVP.

Do not memorize this yet.

Understand what it says.

---

# 3. Why does the Jacobian appear?

This is one of the places where papers can make something elementary sound terrifying.

Forget probability for a moment.

Suppose I stretch a one-dimensional coordinate:

$$
y=2x.
$$

A small interval

$$
[x,x+dx]
$$

becomes

$$
[y,y+dy]
$$

where

$$
dy=2dx.
$$

The same probability mass is now spread over twice the volume.

Therefore the density must become half as large.

So:

$$
p_Y(y)=p_X(x)\frac{1}{2}.
$$

The factor $2$ is the derivative:

$$
\frac{dy}{dx}=2.
$$

In multiple dimensions, the equivalent quantity is the determinant of the Jacobian:

$$
\left|\det J_f(x)\right|.
$$

The determinant tells me:

> **How much the transformation locally expands or contracts volume.**

That's the geometric meaning.

---

# 4. The most important intuition

Whenever I see

$$
\log|\det J|,
$$

I should mentally translate it into:

> **local volume change.**

If

$$
|\det J|>1,
$$

the transformation expands volume.

Probability mass gets spread out, so density decreases.

If

$$
|\det J|<1,
$$

the transformation contracts volume.

Probability mass gets concentrated, so density increases.

The normalizing-flow paper explicitly describes flows in exactly this geometric language: invertible transformations produce local expansions and contractions of the initial density.

This is much more useful than memorizing the determinant formula.

---

# 5. Why invertibility?

Now I ask:

> Why do we insist that $f$ has an inverse?

Because I want to go both ways.

For density evaluation:

$$
x\rightarrow z=f(x).
$$

For generation:

$$
z\rightarrow x=f^{-1}(z).
$$

If the map isn't invertible, a single $z$ could correspond to many different $x$'s, and the clean change-of-variables relationship breaks down.

So a normalizing flow is built from maps that are:

1. differentiable enough for the Jacobian,
2. invertible,
3. computationally cheap in the forward direction,
4. computationally cheap in the inverse direction,
5. equipped with a tractable Jacobian determinant.

The papers repeatedly emphasize this combination of properties. NICE specifically seeks transformations with an easy determinant and easy inverse.

---

# 6. The second load-bearing concept: composition

One simple transformation isn't powerful enough.

So I use many:

$$
z_0
\xrightarrow{f_1}
z_1
\xrightarrow{f_2}
z_2
\xrightarrow{f_3}
\cdots
\xrightarrow{f_K}
z_K.
$$

The total transformation is

$$
f=f_K\circ\cdots\circ f_2\circ f_1.
$$

The beautiful part is that the log-density contributions add:

$$
\log q_K(z_K)
=
\log q_0(z_0)
-
\sum_{k=1}^{K}
\log
\left|
\det
\frac{\partial f_k}{\partial z_{k-1}}
\right|.
$$

This is the key equation for a finite normalizing flow.

This is another place where the notation looks worse than the idea.

The idea is simply:

> **Each layer changes the density. Keep track of every change and add them together.**

---

# 7. Why composition is so powerful

This is the same basic philosophy as neural networks.

One layer:

> simple transformation.

Many layers:

> complicated transformation.

For normalizing flows:

$$
\text{simple invertible maps}
+
\text{composition}
=
\text{complex invertible map}.
$$

The catch is that ordinary neural networks don't generally give us:

* an easy inverse,
* an easy Jacobian determinant.

That is the central engineering/mathematical obstacle.

---

# 8. The central problem

At this point I should stop and formulate the actual research problem.

I want:

$$
\boxed{\text{high expressive power}}
$$

while simultaneously having:

$$
\boxed{\text{easy inverse}}
$$

and

$$
\boxed{\text{easy }\log|\det J|}.
$$

These requirements fight each other.

A completely general neural network can be expressive.

But its inverse may be impossible to compute.

And its Jacobian determinant may be extremely expensive.

The normalizing-flow literature is largely about clever ways of resolving this conflict.

The early flow paper notes that naive invertible parameterizations can make Jacobian computation expensive, motivating transformations with low-cost determinant calculations.

---

# 9. The breakthrough: triangular Jacobians

Here is the mathematical trick that makes everything click.

Suppose

$$
J= \begin{pmatrix} a & 0 \\ b & c \end{pmatrix}.
$$

Then

$$
\det J=ac.
$$

I don't have to calculate a complicated determinant.

For a triangular matrix,

$$
\det J
=
\prod_i J_{ii}.
$$

So the question becomes:

> Can I design a complicated nonlinear transformation whose Jacobian is triangular?

Yes.

That is the idea behind coupling layers.

The NICE paper explicitly motivates the architecture through triangular Jacobians because their determinants are just products of diagonal elements.

---

# 10. The simplest coupling layer

Split the input into two pieces:

$$
x=(x_1,x_2).
$$

Now define

$$
y_1=x_1
$$

and

$$
y_2=x_2+m(x_1).
$$

That's it.

At first glance, this looks almost useless.

But look at what it gives us.

---

# 11. The inverse is trivial

We have

$$
y_1=x_1.
$$

Therefore

$$
x_1=y_1.
$$

And

$$
y_2=x_2+m(x_1).
$$

Therefore

$$
x_2=y_2-m(y_1).
$$

So:

$$
\boxed{
x_1=y_1,\qquad
x_2=y_2-m(y_1)
}
$$

The inverse is just as easy as the forward computation.

And $m$ can be a complicated neural network.

This is the crucial trick.

NICE emphasizes exactly this point: the coupling function $m$ can be arbitrarily complex, while the resulting transformation remains trivially invertible.

---

# 12. Why doesn't the neural network need to be invertible?

This is a very important question.

Suppose

$$
m:\mathbb R^d\rightarrow\mathbb R^{D-d}
$$

is a giant neural network.

It might be impossible to invert $m$.

But I don't care.

The inverse of the whole transformation is

$$
x_2=y_2-m(y_1).
$$

I never need

$$
m^{-1}.
$$

That's the cleverness of the architecture.

The neural network generates a transformation of one part using another part as conditioning information.

---

# 13. The Jacobian

For

$$
y_1=x_1
$$

and

$$
y_2=x_2+m(x_1),
$$

the Jacobian has the structure

$$
J= \begin{pmatrix} I & 0 \\ \frac{\partial y_2}{\partial x_1} & I \end{pmatrix}.
$$

Therefore:

$$
\det J=1.
$$

The complicated derivatives of $m$ appear in the lower-left block.

But the determinant doesn't care.

It only sees the diagonal.

Therefore:

$$
\boxed{\log|\det J|=0}.
$$

This is the mathematical reason the architecture works.

---

# 14. This was NICE

NICE uses the additive coupling transformation

$$
y_1=x_1,
$$

$$
y_2=x_2+m(x_1).
$$

Because its Jacobian determinant is $1$, it is volume-preserving.

But this creates another problem.

If every layer has

$$
|\det J|=1,
$$

then every layer preserves volume.

So the whole flow preserves volume.

That means:

$$
\log|\det J_{\text{total}}|=0.
$$

The model can rearrange the distribution, but cannot locally expand or contract volume.

That's restrictive.

---

# 15. The next idea: affine coupling

Real NVP fixes this.

Instead of

$$
y_2=x_2+m(x_1),
$$

use

$$
y_2
=
x_2\odot\exp(s(x_1))
+
t(x_1).
$$

Here:

* $s$ is a scale network,
* $t$ is a translation network,
* $\odot$ means elementwise multiplication.

The first part remains unchanged:

$$
y_1=x_1.
$$

This is the Real NVP affine coupling layer.

---

# 16. Why the exponential?

Because I need the scaling factor to be positive:

$$
\exp(s(x_1))>0.
$$

That makes the transformation invertible.

The inverse is

$$
x_1=y_1,
$$

$$
x_2
=
\left(y_2-t(y_1)\right)
\odot
\exp(-s(y_1)).
$$

Again:

> I don't need to invert $s$ or $t$.

This is why the networks producing $s$ and $t$ can be arbitrarily complicated.

---

# 17. Now the Jacobian becomes interesting

The Jacobian is triangular:

$$
J= \begin{pmatrix} I & 0 \\ (*) & \operatorname{diag}(\exp(s(x_1))) \end{pmatrix}.
$$

Therefore:

$$
\det J
=
\prod_j \exp(s_j(x_1)).
$$

So:

$$
\log|\det J|
=
\sum_j s_j(x_1).
$$

This is beautiful.

The neural networks can be complicated, but the determinant only requires the outputs of $s$.

Real NVP explicitly derives this triangular-Jacobian determinant and notes that the Jacobian of $s$ or $t$ itself does not need to be computed.

---

# 18. Stop here and understand the trick

This is the point where I should **not** continue reading the paper.

I should be able to answer:

### What is the problem?

We want an expressive invertible transformation with tractable density evaluation.

### Why is the problem difficult?

General nonlinear transformations have expensive inverses and Jacobian determinants.

### What is the trick?

Make the Jacobian triangular.

### How?

Leave half of the variables unchanged and transform the other half conditioned on the unchanged half.

### Why is the inverse easy?

The conditioning variables are unchanged.

### Why is the determinant easy?

The Jacobian is triangular.

### Why can the neural network be complicated?

Because its derivatives don't enter the determinant.

If I understand these six answers, I understand the central architectural idea.

---

# 19. A tiny two-dimensional example

Let

$$
x= \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}.
$$

Suppose

$$
s(x_1)=2x_1,
$$

$$
t(x_1)=x_1^2.
$$

Then

$$
y_1=x_1
$$

and

$$
y_2=x_2e^{2x_1}+x_1^2.
$$

The inverse:

$$
x_1=y_1
$$

and

$$
x_2=(y_2-y_1^2)e^{-2y_1}.
$$

The Jacobian is

$$
J= \begin{pmatrix} 1 & 0 \\ 2x_2e^{2x_1}+2x_1 & e^{2x_1} \end{pmatrix}.
$$

Therefore:

$$
\det J=e^{2x_1}.
$$

Notice something important.

I never needed to calculate the ugly lower-left derivative.

That is the whole trick.

---

# 20. Why one coupling layer isn't enough

There is an obvious problem.

We leave

$$
x_1
$$

unchanged.

So $x_1$ is not directly transformed.

Only $x_2$ changes.

What if we apply another coupling layer that transforms $x_1$ using $x_2$?

Then:

First layer:

$$
(x_1,x_2)
\rightarrow
(x_1,y_2).
$$

Second layer:

$$
(x_1,y_2)
\rightarrow
(z_1,y_2).
$$

Now both variables have participated.

Therefore we alternate the partitions.

Real NVP uses alternating masks so that variables left unchanged by one coupling layer are modified by the next.

---

# 21. The full mental picture

A flow might therefore look like:

$$
x
\rightarrow
f_1(x)
\rightarrow
f_2(f_1(x))
\rightarrow
f_3(\cdots)
\rightarrow
z.
$$

Each layer:

* is invertible,
* has a cheap determinant,
* can contain a powerful neural network.

The composition is therefore:

$$
\boxed{
\text{simple invertible layers}
\rightarrow
\text{complex invertible transformation}
}
$$

while retaining tractable likelihood evaluation.

---

# 22. Why this gives exact likelihood

Now put everything together.

Given an observation $x$:

### Step 1

Transform it:

$$
z=f(x).
$$

### Step 2

Evaluate the simple base density:

$$
\log p_Z(z).
$$

### Step 3

Add the volume correction:

$$
\log|\det J_f(x)|.
$$

Therefore:

$$
\boxed{
\log p_X(x)
=
\log p_Z(f(x))
+
\log|\det J_f(x)|
}
$$

This is exact.

There is no variational approximation to the density itself.

There is no discriminator.

There is no MCMC procedure needed merely to evaluate the likelihood.

Real NVP was explicitly designed around exact and efficient likelihood evaluation, inference, and sampling.

---

# 23. And sampling?

Sampling is almost embarrassingly simple.

Start with

$$
z\sim p_Z.
$$

For example,

$$
z\sim\mathcal N(0,I).
$$

Then apply

$$
x=f^{-1}(z).
$$

That's the generated sample.

So:

```text
Training / likelihood:

x → f → z → evaluate p(z)
      +
   Jacobian correction


Generation:

z ~ simple Gaussian
        ↓
     f⁻¹
        ↓
        x
```

The early NICE paper already highlights this simple sampling procedure: sample $h$ from the prior and apply $f^{-1}$.

---

# 24. One equation that explains the entire model

If I had to reduce normalizing flows to one mathematical statement, it would be:

$$
\boxed{
\log p_X(x)
=
\underbrace{\log p_Z(f(x))}_{\text{make }x\text{ look simple}}
+
\underbrace{\log|\det J_f(x)|}_{\text{correct for volume change}}
}
$$

The first term says:

> Where did this point land in latent space?

The second says:

> How much did the transformation stretch or compress space around it?

That is normalizing flows.

---

# 25. Why do we want a simple latent distribution?

Suppose

$$
p_Z(z)=\mathcal N(0,I).
$$

Then the model is trying to learn an $f$ such that the transformed data

$$
z=f(x)
$$

looks approximately Gaussian.

So the flow is effectively learning a complicated coordinate transformation:

$$
\text{complicated data geometry}
\quad\longrightarrow\quad
\text{simple geometry}.
$$

This connects directly to the motivation of NICE: find a representation in which the distribution becomes easier to model, particularly a factorized distribution.

---

# 26. What exactly is being learned?

The neural networks $s$ and $t$ contain parameters:

$$
\theta.
$$

So really:

$$
f=f_\theta.
$$

Training becomes maximum likelihood:

$$
\max_\theta
\sum_{i=1}^N
\log p_\theta(x_i).
$$

Using the change-of-variables formula:

$$
\max_\theta
\sum_i
\left[
\log p_Z(f_\theta(x_i))
+
\log
|\det J_{f_\theta}(x_i)|
\right].
$$

So the model is learning:

> **How should I warp space so that the observed data becomes probable under a simple base distribution?**

---

# 27. The geometric interpretation of training

This gives me a much better mental model than "the neural network learns a density."

Imagine that the data occupies a complicated high-density region.

The flow learns a deformation of space.

It can:

* stretch some regions,
* compress others,
* rotate and rearrange structure,
* gradually untangle dependencies,
* map complicated regions toward something resembling a Gaussian.

The Jacobian determinant tells us the local volume distortion caused by this deformation.

Therefore the model doesn't merely assign arbitrary probabilities.

It **learns a coordinate system** in which the probability distribution becomes simple.

---

# 28. A useful connection to statistics

Suppose the data has highly correlated coordinates:

$$
x_1\leftrightarrow x_2.
$$

A Gaussian base with independent coordinates assumes:

$$
p_Z(z)
=
\prod_i p(z_i).
$$

So the flow needs to transform the correlated data into approximately independent latent coordinates.

This is conceptually related to the motivation behind independent component analysis.

NICE frames its objective in terms of learning a nonlinear transformation into a factorized latent distribution.

---

# 29. Why normalizing flows are different from VAEs

This distinction is worth understanding conceptually.

A VAE typically introduces an approximate posterior:

$$
q_\phi(z|x).
$$

If $q$ is too simple, it may not resemble the true posterior.

The normalizing-flow paper begins from precisely this limitation and proposes flows as a way to make the approximate distribution progressively more flexible.

The flow idea is:

$$
z_0\sim q_0
$$

then

$$
z_1=f_1(z_0),
$$

$$
z_2=f_2(z_1),
$$

and so on.

The initial distribution can be simple.

The final distribution can be extremely complicated.

This is the important conceptual move:

> **Complexity comes from transforming a simple distribution rather than specifying the complex distribution directly.**

---

# 30. A subtle but important point: not every expectation needs the density

The normalizing-flow paper points out an interesting property.

If

$$
z_K=f_K\circ\cdots\circ f_1(z_0),
$$

then for a function $h$,

$$
\mathbb E_{q_K}[h(z)]
=
\mathbb E_{q_0}\left[h(f_K\circ\cdots\circ f_1(z_0))\right].
$$

So if I only need an expectation and $h$ doesn't itself depend on $q_K$, I can sample from the base and transform the samples without explicitly computing the final density.

This distinction becomes important when comparing different kinds of flows and applications.

---

# 31. What I should NOT memorize

I do not want to memorize:

> "A coupling layer is defined by equations $4$ and $5$."

Instead, I want to remember:

> **Freeze half → use it to transform the other half → triangular Jacobian → cheap determinant → easy inverse.**

That sentence should trigger the equations automatically.

Similarly:

### Change of variables

Don't memorize:

$$
p_X=p_Z|\det J|.
$$

Remember:

> **Same probability mass + changed volume = density correction.**

### Composition

Don't memorize equation $7$.

Remember:

> **Each layer contributes a log-determinant; logs turn products into sums.**

### Affine coupling

Don't memorize the architecture.

Remember:

> **Conditional transformation designed so that the complicated network never needs to be inverted.**

That is mathematical compression.

---

# 32. The three questions I should ask whenever I see a new flow

Whenever I encounter a new architecture, I should immediately ask:

### Question 1: Is it invertible?

Can I compute:

$$
x=f^{-1}(z)?
$$

If not, it isn't a standard normalizing flow of this type.

### Question 2: Can I compute the log determinant?

Can I efficiently calculate:

$$
\log|\det J_f(x)|?
$$

### Question 3: Is it expressive enough?

Can repeated application of these simple transformations represent complicated distributions?

These three questions explain a huge fraction of normalizing-flow architecture design.

---

# 33. The historical progression now makes sense

Instead of memorizing papers chronologically, I can understand the progression as a sequence of problems.

### Problem 1

How do I transform a simple density into a complicated one?

→ Change of variables.

### Problem 2

How do I compose many transformations?

→ Log determinants add.

### Problem 3

How do I make the inverse tractable?

→ Coupling layers.

### Problem 4

How do I make the determinant tractable?

→ Triangular Jacobians.

### Problem 5

How do I make the transformation expressive?

→ Stack many coupling layers and alternate partitions.

### Problem 6

How do I allow volume changes?

→ Affine coupling / scaling.

That is the conceptual history behind NICE → more general normalizing flows → Real NVP.

NICE's additive coupling is volume-preserving, while Real NVP introduces scaling through the affine coupling transformation.

---

# 34. Why the papers initially looked harder than they really are

I think this is worth recording explicitly.

When I first see:

$$
q_K(z_K)
=
q_0(z_0)
\prod_{k=1}^{K}
\left|
\det
\frac{\partial f_k}{\partial z_{k-1}}
\right|^{-1},
$$

my brain sees:

> "Probability theory + multivariable calculus + Jacobians + composition + unfamiliar notation."

But after translation:

> Start with a distribution.
> Stretch/compress it.
> Correct its density.
> Repeat.

That's much simpler.

The notation isn't the idea.

The notation is **the compressed representation of the idea**.

My job as a learner is to decompress it.

---

# 35. My study protocol for normalizing flows

I would study this subject in the following order.

## Stage 1: Change of variables

Before reading more flow papers, make sure I can derive and understand:

$$
p_Y(y)
=
p_X(x)
\left|
\det
\frac{\partial x}{\partial y}
\right|.
$$

I should be able to explain this geometrically.

### Toy exercise

Take

$$
y=2x
$$

and calculate the transformed density.

Then take

$$
y=Ax
$$

and understand why

$$
p_Y(y)
=
p_X(A^{-1}y)|\det A|^{-1}.
$$

---

# 36. Stage 2: Jacobian geometry

Take simple transformations:

$$
f(x,y)=(2x,y)
$$

$$
f(x,y)=(x,3y)
$$

$$
f(x,y)=(x+y,y)
$$

and calculate:

$$
J_f,\qquad \det J_f.
$$

Then ask:

> What does each determinant mean geometrically?

Do not move on until this is intuitive.

---

# 37. Stage 3: Composition

Take

$$
f_1(x)=2x
$$

and

$$
f_2(x)=x+1.
$$

Compute:

$$
f_2(f_1(x)).
$$

Then verify:

$$
\det J_{f_2\circ f_1}
=
\det J_{f_2}\det J_{f_1}.
$$

This tiny exercise makes the flow equations much less mysterious.

---

# 38. Stage 4: Additive coupling

Derive yourself:

$$
y_1=x_1,
$$

$$
y_2=x_2+m(x_1).
$$

Then derive:

1. inverse,
2. Jacobian,
3. determinant,
4. log determinant.

Do not look at NICE while doing it.

If I can derive all four from scratch, I understand the central construction.

---

# 39. Stage 5: Affine coupling

Now replace:

$$
x_2+m(x_1)
$$

with

$$
x_2\odot e^{s(x_1)}+t(x_1).
$$

Again derive:

1. inverse,
2. Jacobian,
3. determinant,
4. log determinant.

Then ask:

> What did this buy us that additive coupling didn't?

Answer:

$$
\boxed{\text{non-unit volume change}}
$$

---

# 40. Stage 6: Stack layers

Construct a toy flow:

$$
f=f_3\circ f_2\circ f_1.
$$

Use alternating masks.

Then calculate the total log determinant:

$$
\log|\det J_f|
=
\sum_{k=1}^3
\log|\det J_{f_k}|.
$$

This is where the entire architecture should finally feel natural.

---

# 41. Stage 7: Only then read Real NVP carefully

At this point I should return to the Real NVP paper.

Now when I see:

> affine coupling layer

I already know why it exists.

When I see:

> triangular Jacobian

I know why.

When I see:

> alternating masks

I know why.

When I see:

> multi-scale architecture

I can ask what computational or representational problem it is solving.

That is dramatically different from reading the paper from page 1 and hoping the notation eventually becomes meaningful.

---

# 42. My paper-reading template

For every new normalizing-flow paper, I should fill this out.

### Problem

What problem is the paper solving?

### Existing obstacle

Why couldn't the previous architecture solve it?

### Key idea

What is the one new trick?

### Mathematical mechanism

What equation makes the trick work?

### Invertibility

Why can I calculate the inverse?

### Jacobian

Why can I calculate the determinant?

### Expressivity

Why is the transformation sufficiently powerful?

### Computational cost

What does it cost?

### Failure mode

What does the architecture sacrifice?

### Toy example

Can I reproduce the mechanism in 1–2 dimensions?

If I can answer these questions, I understand the paper much better than if I have merely read every page.

---

# 43. The proof-pattern library I want to build

Normalizing flows also give me several reusable mathematical patterns.

## Pattern A: Change of variables

$$
p_Y(y)
=
p_X(x)
|\det J_{f^{-1}}(y)|.
$$

Mental translation:

> Transform the density and correct for volume.

---

## Pattern B: Chain rule for Jacobians

$$
J_{g\circ f}(x)
=
J_g(f(x))J_f(x).
$$

Then:

$$
\det J_{g\circ f}
=
\det J_g\det J_f.
$$

Mental translation:

> Composition multiplies local volume changes.

---

## Pattern C: Log turns products into sums

$$
\log\prod_k a_k
=
\sum_k\log a_k.
$$

Mental translation:

> Each flow layer contributes an additive likelihood correction.

---

## Pattern D: Triangular determinant

$$
\det \begin{pmatrix} A & 0 \\ B & C \end{pmatrix}
=
\det(A)\det(C).
$$

Mental translation:

> Put the complicated derivatives where they don't affect the determinant.

This last sentence is almost the entire coupling-layer trick.

---

# 44. The deepest insight I want to remember

The brilliance of normalizing flows is not:

> "Use neural networks to model probability distributions."

That is not particularly surprising.

The cleverness is:

> **Design the architecture so that the neural network can be complicated exactly where the mathematical operations don't need to be complicated.**

That's why the coupling layer is such an elegant construction.

The neural network $s,t$ can be arbitrarily expressive.

But the actual transformation has been structured so that:

$$
\text{inverse}
$$

and

$$
\text{Jacobian determinant}
$$

remain simple.

This is a recurring theme in theoretical machine learning:

> **Don't make the entire object simple. Make the operations you need simple.**

---

# 45. One final mental model

If I forget everything else, I want to remember this:

```text
                    NORMALIZING FLOW

        complicated distribution pX(x)
                     │
                     │  invertible f
                     ▼
              simple distribution pZ(z)
                     │
                     │
             Gaussian / simple prior
```

Forward:

$$
x\rightarrow z=f(x)
$$

and

$$
\boxed{
\log p_X(x)
=
\log p_Z(z)+\log|\det J_f(x)|
}
$$

Backward:

$$
z\rightarrow x=f^{-1}(z).
$$

The entire architectural problem is therefore:

```text
Can I construct f such that:

    f is expressive
        +
    f is invertible
        +
    log |det Jf| is cheap
        +
    f⁻¹ is cheap?
```

Coupling layers answer this through triangular structure.

NICE gives the additive, volume-preserving version.

Real NVP adds learnable scaling.

Stack enough of these simple transformations and I obtain a highly nonlinear invertible transformation capable of modeling complicated distributions.

That is the conceptual core.

---

# 46. What I should be able to do before moving on

I should **not** proceed to more advanced flow architectures until I can do the following without looking at the paper:

* [ ] Explain change of variables geometrically.
* [ ] Derive the 1D density transformation.
* [ ] Explain what a Jacobian determinant means.
* [ ] Derive the Jacobian of an additive coupling layer.
* [ ] Explain why its determinant is $1$.
* [ ] Derive the inverse.
* [ ] Explain why $m$ does not need to be invertible.
* [ ] Derive the affine coupling inverse.
* [ ] Derive its log determinant.
* [ ] Explain why alternating masks are needed.
* [ ] Explain why log determinants add across layers.
* [ ] Explain the complete likelihood calculation.
* [ ] Explain generation using $f^{-1}$.

If I can do those, I don't need to spend another week "understanding normalizing flows."

I have the mechanism.

Now I can move upward.

---

# 47. Next questions I should investigate

Once the above feels comfortable, these are the natural next questions:

1. **Why do autoregressive flows work?**
2. **How is Masked Autoregressive Flow related to Real NVP?**
3. **Why does IAF reverse the computational tradeoff?**
4. **What exactly is Glow changing?**
5. **What makes continuous normalizing flows different?**
6. **Where does the instantaneous change-of-variables formula come from?**
7. **Why does the divergence $\nabla\cdot f$ appear?**
8. **What is the relationship between normalizing flows and optimal transport?**
9. **What expressive-power limitations do coupling architectures have?**
10. **Why are normalizing flows awkward for high-dimensional image likelihoods?**

Those are no longer ten unrelated topics.

They are all variations on the same question:

> **How can I construct a complicated transformation while retaining enough mathematical structure to compute what I need?**

And that is the question I should carry into every new normalizing-flow paper.
