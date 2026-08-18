---
layout: post
title: "Simulator-in-the-Loop Conditional Generative Inverse Design for Photonic Systems"
date: 2026-08-18
category: pic
---

## 1. The problem we are trying to solve

Our system consists of a photonics simulator with **42 tunable parameters**, or knobs:

$$
\theta\in\mathbb{R}^{42}.
$$

When we choose a particular set of knob values $\theta$, the simulator produces an output spectrum containing $D=400$ spectral samples:

$$
y=S(\theta),
$$

where

$$
S:\mathbb{R}^{42}\rightarrow\mathbb{R}^{400}.
$$

The simulator is deterministic: if we give it the same $\theta$, it produces the same spectrum $y$.

Our forward problem is therefore straightforward:

$$
\boxed{\theta\longrightarrow S(\theta)=y}
$$

The problem we care about is the inverse problem.

Given a desired target spectrum

$$
y^\star,
$$

we want to find knob configurations $\theta$ for which

$$
S(\theta)\approx y^\star.
$$

At first glance, this looks like a standard optimization problem:

$$
\theta^\star
=

\arg\min_\theta
d(S(\theta),y^\star).
$$

For example, if we use squared Euclidean error,

$$
d(S(\theta),y^\star)
=

|S(\theta)-y^\star|_2^2.
$$

This would give us **one** solution.

However, this is not actually the problem we want to solve.

---

## 2. The inverse problem is one-to-many

Because our photonic system is highly nonlinear, different knob configurations can produce essentially the same output spectrum.

For example,

$$
S(\theta_1)\approx y^\star,
$$

$$
S(\theta_2)\approx y^\star,
$$

$$
S(\theta_3)\approx y^\star,
$$

even though

$$
\theta_1\neq\theta_2\neq\theta_3.
$$

Therefore, the inverse mapping is not generally

$$
y^\star\rightarrow\theta^\star.
$$

Instead, it is more accurately represented as

$$
\boxed{
y^\star
\longrightarrow
\{\theta_1,\theta_2,\theta_3,\ldots\}.
}
$$

For a particular target, we can define its approximate solution set as

$$
\boxed{
\mathcal S(y^\star)
=

\{\theta\in\mathbb R^{42}:S(\theta)\approx y^\star\}.
}
$$

Our goal is therefore not simply:

> Find one point in $\mathcal S(y^\star)$.

Our goal is:

> **Understand and model the distribution of valid knob configurations associated with a target spectrum, so that we can generate many different valid designs.**

This changes the nature of the problem.

---

## 3. From deterministic inverse design to conditional generation

Instead of asking our model to produce one deterministic answer,

$$
\theta=f(y^\star),
$$

we want it to represent a conditional distribution:

$$
\boxed{
\theta\sim p(\theta\mid y^\star).
}
$$

Here,

$$
p(\theta\mid y^\star)
$$

means:

> the probability distribution over knob configurations that are considered solutions, given the desired spectrum $y^\star$.

Our ultimate objective is therefore:

$$
\boxed{
\text{Given }y^\star,\text{ generate many different }\theta
\text{ such that }S(\theta)\approx y^\star.
}
$$

This gives us two fundamental requirements.

### Requirement 1: Validity

Every generated design should reproduce the target:

$$
S(\theta_i)\approx y^\star.
$$

Equivalently,

$$
d(S(\theta_i),y^\star)\rightarrow0.
$$

### Requirement 2: Diversity

Different generated samples should represent genuinely different designs:

$$
\theta_i\neq\theta_j
$$

for different samples, while still satisfying

$$
S(\theta_i)\approx S(\theta_j)\approx y^\star.
$$

Conceptually, we therefore need something like

$$
\boxed{
\text{good solutions}
=

\text{validity}
+
\text{diversity}.
}
$$

A naive objective such as

$$
L_{\mathrm{physics}}
=

\mathbb E_z
\left[
d(S(f_\phi(z;y^\star)),y^\star)
\right]
$$

handles validity, but creates an important problem.

---

## 4. The collapse problem

Suppose we introduce a random variable $z$ and generate

$$
\theta=f_\phi(z;y^\star).
$$

We sample many different values:

$$
z_1,z_2,\ldots,z_N.
$$

This produces

$$
\theta_i=f_\phi(z_i;y^\star).
$$

We then run the simulator:

$$
y_i=S(\theta_i)
$$

and calculate

$$
L_i=d(y_i,y^\star).
$$

It is tempting to train only using

$$
\boxed{
L_{\mathrm{physics}}
=

\mathbb E_z
[
d(S(f_\phi(z;y^\star)),y^\star)
].
}
$$

But there is a serious problem.

The easiest way for the model to minimize this loss may be to learn one excellent solution $\theta^\star$ and produce it regardless of $z$:

$$
f_\phi(z_1;y^\star)
\approx
f_\phi(z_2;y^\star)
\approx
\cdots
\approx
\theta^\star.
$$

Then

$$
S(\theta^\star)\approx y^\star,
$$

so the physics loss becomes very small.

But we have lost exactly what we wanted: the ability to explore different solutions.

This is a form of **collapse**: different latent inputs fail to produce meaningful variation in the output.

Therefore, simply optimizing the simulator error is not sufficient.

---

## 5. We need to model a distribution, not just minimize an error

The central idea is therefore to distinguish between:

$$
\boxed{\text{the distribution we want}}
$$

and

$$
\boxed{\text{the distribution our model produces}.}
$$

We denote the desired distribution by

$$
p(\theta\mid y^\star)
$$

and the distribution generated by our conditional model by

$$
q_\phi(\theta\mid y^\star).
$$

Our objective becomes

$$
\boxed{
q_\phi(\theta\mid y^\star)
\approx
p(\theta\mid y^\star).
}
$$

The important point is that $p(\theta\mid y^\star)$ is not something for which we currently have a dataset.

We do **not** have a collection of known correct pairs

$$
(\theta,y^\star).
$$

In fact, generating such a dataset is one of the problems we are trying to avoid.

Instead, we have:

1. a desired target spectrum $y^\star$,
2. a simulator $S(\theta)$,
3. the ability to evaluate how well any proposed $\theta$ reproduces $y^\star$.

This leads to a simulator-in-the-loop formulation.

---

## 6. What exactly is $p(\theta\mid y^\star)$?

An important subtlety is that the simulator naturally gives us a **solution set**, but not necessarily a unique probability distribution over that set.

Suppose

$$
\theta_1,\theta_2,\theta_3
$$

all satisfy

$$
S(\theta_i)\approx y^\star.
$$

The simulator tells us that these are valid solutions.

But the simulator alone does not necessarily tell us whether

$$
p(\theta_1\mid y^\star)
=

p(\theta_2\mid y^\star)
$$

or whether

$$
p(\theta_1\mid y^\star)
=

10p(\theta_2\mid y^\star).
$$

Physics determines **which configurations are compatible with the target**. A probability distribution additionally requires us to specify how we want probability to be distributed among those configurations.

This gives us an experimental degree of freedom.

We can experiment with different choices of the prior $p(\theta)$, different mismatch functions $d$, and different values of $\beta$.

---

## 7. A probabilistic formulation of the solution distribution

A natural way to construct a distribution over solutions is

$$
\boxed{
p(\theta\mid y^\star)
\propto
p(\theta)
\exp[-\beta d(S(\theta),y^\star)].
}
$$

This equation is central to our formulation, so it is worth deriving and understanding carefully.

Here:

* $p(\theta)$ is a prior over knob configurations;
* $d(S(\theta),y^\star)$ measures how poorly $\theta$ reproduces the target;
* $\beta>0$ controls how strongly we enforce the spectral match.

For our current experiments, a simple choice could be a uniform prior over the physically allowed knob ranges.

If

$$
p(\theta)=\text{constant},
$$

then

$$
p(\theta\mid y^\star)
\propto
\exp[-\beta d(S(\theta),y^\star)].
$$

---

## 8. Why the exponential?

Suppose

$$
d(S(\theta),y^\star)
=

|S(\theta)-y^\star|_2^2.
$$

A perfect solution has

$$
d=0.
$$

Then

$$
e^{-\beta d}=1.
$$

A solution with a larger error receives a smaller weight.

For example, if $\beta=1$:

$$
e^{-0}=1,
$$

$$
e^{-1}\approx0.37,
$$

$$
e^{-5}\approx0.0067.
$$

Therefore,

$$
\exp[-\beta d(S(\theta),y^\star)]
$$

acts as a soft measure of compatibility with the target.

A small spectral error gives high probability weight; a large spectral error gives very little probability weight.

---

## 9. What does $\beta$ mean?

The parameter $\beta$ determines how strongly spectral accuracy controls the distribution.

For small $\beta$, configurations with moderate errors still retain some probability.

For large $\beta$, even relatively small errors are strongly suppressed.

In the limit

$$
\beta\rightarrow\infty,
$$

the distribution increasingly concentrates around configurations satisfying

$$
S(\theta)\approx y^\star.
$$

Conceptually,

$$
\boxed{
\beta\rightarrow\infty
\quad\Rightarrow\quad
\text{probability concentrates near the solution set}.
}
$$

For finite $\beta$, we instead obtain a "soft" solution set.

This is useful because the exact equation

$$
S(\theta)=y^\star
$$

may define a complicated lower-dimensional set, whereas a finite-$\beta$ distribution gives us a smooth distribution around that set.

---

## 10. The prior $p(\theta)$

The prior allows us to express preferences over knob configurations independently of the target.

For example, if all physically allowed configurations are considered equally desirable, we can use a uniform prior.

But we could also construct a prior that favors:

* particular ranges of knob values,
* robust designs,
* fabrication-friendly configurations,
* physically preferred configurations,
* or other properties.

Thus,

$$
p(\theta)
$$

is not necessarily "the truth."

It is part of how we define the particular distribution of solutions that we want our model to represent.

This means that experimenting with different priors is meaningful.

---

## 11. Why the conditional generator needs a latent variable

We now need a way to generate many different $\theta$'s for the same $y^\star$.

We introduce a latent random variable

$$
z.
$$

We sample

$$
\boxed{
z\sim p_Z(z)
}
$$

where a convenient choice is

$$
\boxed{
p_Z(z)=\mathcal N(0,I).
}
$$

The Gaussian is **not** an assumption that our photonic knobs are Gaussian.

It is simply a convenient source of randomness.

We then learn a transformation

$$
\boxed{
\theta=f_\phi(z;y^\star).
}
$$

The target spectrum $y^\star$ tells the model **which solution distribution to generate**, while $z$ tells the model **which random sample from that distribution to generate**.

After training,

$$
z\sim p_Z(z)
$$

should produce

$$
\theta\sim q_\phi(\theta\mid y^\star).
$$

Our hope is that

$$
\boxed{
q_\phi(\theta\mid y^\star)
\approx
p(\theta\mid y^\star).
}
$$

---

## 12. What does $z$ actually mean?

It is tempting to think that $z$ must have some physical interpretation.

It does not necessarily.

The latent variable is simply a source of variation.

If

$$
z_1\neq z_2,
$$

then the generator produces

$$
\theta_1=f_\phi(z_1;y^\star)
$$

and

$$
\theta_2=f_\phi(z_2;y^\star).
$$

A successful generative model should use the different regions of $z$-space to represent different parts of the solution distribution.

Thus, $z$ can be thought of as a coordinate for exploring the variability of the possible designs.

It is not necessarily a physical coordinate.

---

## 13. Why use a normalizing flow?

A normalizing flow gives us a particularly useful construction because it learns an invertible transformation between a simple base distribution and a complicated target distribution.

We start with

$$
z\sim\mathcal N(0,I)
$$

and learn

$$
\theta=f_\phi(z;y^\star).
$$

The transformation can bend, stretch, compress, and rearrange the simple Gaussian distribution into a much more complicated distribution over the 42 knob values.

Thus:

$$
\boxed{
\text{simple latent distribution}
\overset{f_\phi}{\longrightarrow}
\text{complex distribution of solutions}.
}
$$

The important advantage of a normalizing flow is that the resulting density

$$
q_\phi(\theta\mid y^\star)
$$

is tractable and can be evaluated using the change-of-variables formula.

A normalizing flow does **not**, however, automatically discover the "true" distribution. Training is still required. Its advantage is that it gives us a flexible and explicitly tractable representation of the distribution that we choose to learn.

---

## 14. Why is the latent dimension normally 42?

Our design variable is

$$
\theta\in\mathbb R^{42}.
$$

A standard normalizing flow uses an invertible transformation

$$
f_\phi:\mathbb R^{42}\rightarrow\mathbb R^{42}.
$$

Therefore, normally,

$$
\boxed{
z\in\mathbb R^{42}.
}
$$

The dimensions match because the transformation is invertible.

A generic generator, VAE, GAN, or other non-invertible generative model does not necessarily require this equality. For example, such a model could use a lower-dimensional latent variable.

This is worth keeping in mind because our physical solution set may have lower intrinsic dimensionality than 42. If that turns out to be important, a standard full-dimensional normalizing flow may not be the ideal representation.

For the initial experiments, however, we can use

$$
z\in\mathbb R^{42}
$$

and investigate other dimensionalities with alternative generative architectures if necessary.

---

## 15. The simulator gives us something extremely valuable: gradients

Our forward simulator is differentiable.

We have

$$
y=S(\theta).
$$

Therefore, we can calculate how the output spectrum changes as we change the knobs:

$$
J_S(\theta)
=

\frac{\partial S}{\partial\theta}.
$$

Since

$$
\theta\in\mathbb R^{42}
$$

and

$$
y\in\mathbb R^{400},
$$

the simulator Jacobian has dimensions

$$
\boxed{
J_S\in\mathbb R^{400\times42}.
}
$$

Now define our spectral loss:

$$
L(\theta)
=

d(S(\theta),y^\star).
$$

We first consider the gradient of the loss with respect to the spectrum:

$$
\nabla_y d(y,y^\star)\in\mathbb R^{400}.
$$

We want the gradient with respect to the 42 knobs:

$$
\nabla_\theta L\in\mathbb R^{42}.
$$

By the chain rule, using column-vector gradients,

$$
\boxed{
\nabla_\theta L
=

J_S(\theta)^T\nabla_y d(y,y^\star).
}
$$

The dimensions make this clear:

$$
(42\times400)(400\times1)
=

42\times1.
$$

So the transpose appears because the Jacobian maps changes in the 42-dimensional knob vector into changes in the 400-dimensional spectrum, while the backward gradient must propagate in the opposite direction.

We therefore write

$$
\boxed{
\nabla_\theta d(S(\theta),y^\star)
=

J_S(\theta)^T
\nabla_y d(y,y^\star)
}
$$

with

$$
y=S(\theta).
$$

We do **not** need to explicitly construct the full $400\times42$ Jacobian if our automatic-differentiation framework can compute the corresponding vector-Jacobian product directly.

---

## 16. This gives us simulator-in-the-loop training

We can now put the pieces together.

For a target spectrum $y^\star$:

1. Sample

$$
z\sim\mathcal N(0,I).
$$

2. Generate knobs:

$$
\theta=f_\phi(z;y^\star).
$$

3. Run the simulator:

$$
y=S(\theta).
$$

4. Compare the simulated spectrum to the target:

$$
d(y,y^\star).
$$

5. Backpropagate through

$$
z
\rightarrow
\theta
\rightarrow
S(\theta)
\rightarrow
d.
$$

Thus we do **not** need to first create a huge dataset of pairs

$$
(\theta,S(\theta)).
$$

Instead, the simulator itself provides the training signal.

This is the central **simulator-in-the-loop** idea.

---

## 17. What happens if we only use the physics loss?

If we use only

$$
\boxed{
L_{\mathrm{physics}}
=

\mathbb E_{z}
\left[
d(S(f_\phi(z;y^\star)),y^\star)
\right],
}
$$

then the model is being trained to generate valid designs.

But this alone does not force it to represent the entire solution distribution.

It can collapse:

$$
f_\phi(z_1;y^\star)
\approx
f_\phi(z_2;y^\star)
$$

for many different $z_1,z_2$.

Therefore, if our goal is distribution learning, we need a distributional objective.

---

## 18. The distribution-matching objective

We define

$$
q_\phi(\theta\mid y^\star)
$$

as the distribution generated by our flow.

Our target is

$$
p(\theta\mid y^\star).
$$

We therefore want

$$
\boxed{
q_\phi(\theta\mid y^\star)
\approx
p(\theta\mid y^\star).
}
$$

One principled way to express this is through KL divergence:

$$
\boxed{
D_{\mathrm{KL}}
\left(
q_\phi(\theta\mid y^\star)
|
p(\theta\mid y^\star)
\right).
}
$$

We minimize this with respect to $\phi$:

$$
\boxed{
\phi^\star
=

\arg\min_\phi
D_{\mathrm{KL}}
\left(
q_\phi(\theta\mid y^\star)
|
p(\theta\mid y^\star)
\right).
}
$$

This is different from the familiar VAE expression

$$
D_{\mathrm{KL}}(q_\phi(z\mid x)|p(z)).
$$

The VAE expression compares a latent posterior with a latent prior.

Here, we are comparing:

$$
\boxed{
\text{our generated distribution over knobs}
}
$$

with

$$
\boxed{
\text{our desired distribution over knobs given the target}.
}
$$

---

## 19. Deriving the training objective

We start from

$$
D_{\mathrm{KL}}(q_\phi|p)
=

\mathbb E_{q_\phi}
\left[
\log q_\phi(\theta\mid y^\star)
-

\log p(\theta\mid y^\star)
\right].
$$

Our target distribution is

$$
p(\theta\mid y^\star)
=

\frac{
p(\theta)
e^{-\beta d(S(\theta),y^\star)}
}{
Z(y^\star)
},
$$

where

$$
Z(y^\star)
=

\int
p(\theta)
e^{-\beta d(S(\theta),y^\star)}
d\theta
$$

is the normalization constant.

Taking the logarithm:

$$
\log p(\theta\mid y^\star)
=

- \log p(\theta)

- \beta d(S(\theta),y^\star)

\log Z(y^\star).
$$

Substituting into the KL divergence gives

$$
D_{\mathrm{KL}}(q_\phi|p)
=

\mathbb E_{q_\phi}
\left[
\log q_\phi(\theta\mid y^\star)
-

\log p(\theta)
+
\beta d(S(\theta),y^\star)
\right]
+
\log Z(y^\star).
$$

The term

$$
\log Z(y^\star)
$$

does not depend on $\phi$, so it does not affect the optimization.

Therefore, our optimization objective is

$$
\boxed{
L(\phi)
=

\mathbb E_{q_\phi(\theta\mid y^\star)}
\left[
\log q_\phi(\theta\mid y^\star)
-

\log p(\theta)
+
\beta d(S(\theta),y^\star)
\right].
}
$$

This is the key equation connecting the probabilistic formulation to the simulator-in-the-loop training procedure.

---

## 20. Interpreting the three terms

Our objective is

$$
L=
\mathbb E_q
\left[
\underbrace{\beta d(S(\theta),y^\star)}_{\text{physics}}
+
\underbrace{\log q_\phi(\theta\mid y^\star)}_{\text{distribution/entropy}}
-

\underbrace{\log p(\theta)}_{\text{prior}}
\right].
$$

### Physics term

$$
\boxed{
\beta d(S(\theta),y^\star)
}
$$

says:

> Generated knob configurations should produce the desired spectrum.

### Prior term

$$
\boxed{
-\log p(\theta)
}
$$

says:

> Prefer knob configurations according to our chosen prior.

### Distribution/entropy term

$$
\boxed{
\log q_\phi(\theta\mid y^\star)
}
$$

is part of the KL divergence and prevents us from reducing the problem to merely minimizing the expected physics error. Together with the target distribution, it determines how probability mass is represented.

It is therefore better not to call this term simply a "diversity loss." The more precise statement is:

> **We are matching the entire generated distribution to a specified target distribution, and diversity emerges according to the structure of that target distribution.**

---

## 21. What does "diversity" mean in our formulation?

We do **not** want arbitrary differences between knob vectors.

For example, it would be useless to generate

$$
\theta_1,\theta_2,\theta_3
$$

that are extremely different but all produce terrible spectra.

Our desired diversity is:

$$
\boxed{
\text{different designs}
+
\text{same desired behavior}.
}
$$

Therefore:

$$
\theta_i\neq\theta_j
$$

should occur while

$$
S(\theta_i)\approx y^\star
$$

and

$$
S(\theta_j)\approx y^\star.
$$

The simulator determines validity.

The distribution $p(\theta\mid y^\star)$ determines how we want the valid solutions to be represented.

---

## 22. A crucial distinction: solution set versus solution distribution

The simulator gives us a solution set:

$$
\mathcal S(y^\star)
=

\{\theta:S(\theta)\approx y^\star\}.
$$

But a solution set is not automatically a probability distribution.

For example, suppose we have three solutions:

$$
\theta_1,\theta_2,\theta_3.
$$

The set is simply

$$
\{\theta_1,\theta_2,\theta_3\}.
$$

A distribution could instead be

$$
p(\theta_1)=p(\theta_2)=p(\theta_3)=\frac13,
$$

or perhaps

$$
p(\theta_1)=0.8,\qquad
p(\theta_2)=0.1,\qquad
p(\theta_3)=0.1.
$$

The physics tells us which solutions are valid, but it does not necessarily specify these probabilities.

Therefore, part of our experimental work is to investigate different choices of

$$
p(\theta)
$$

and different values of

$$
\beta.
$$

We can ask experimentally:

> Which choice gives us a useful representation of the solution space?

---

## 23. The complete conceptual picture

The complete idea can now be summarized as

```text
                         TARGET SPECTRUM
                              y*
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Desired distribution│
                    │    p(θ | y*)        │
                    └─────────────────────┘
                               │
                               │ approximate
                               ▼
                    ┌─────────────────────┐
                    │   Conditional Flow │
                    │   qφ(θ | y*)        │
                    └─────────────────────┘
                               ▲
                               │
                       θ = fφ(z ; y*)
                               ▲
                               │
                         z ~ N(0,I)
                               │
                               │
                        random variation
                               │
                               ▼
                       θ ∈ R⁴²
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Photonics         │
                    │   Simulator S(θ)    │
                    └─────────────────────┘
                               │
                               ▼
                            y = S(θ)
                               │
                               ▼
                  d(S(θ), y*) = spectrum error
                               │
                               ▼
                         BACKPROPAGATION
                               │
                               └──────► update φ
```

The target spectrum determines **which conditional distribution** we want.

The latent variable determines **which sample from that distribution** we generate.

The simulator determines **whether that sample actually produces the desired physical behavior**.

The distributional objective determines **how we learn the whole distribution rather than only one solution**.

---

## 24. We do not train a separate model for every target

A very important question is:

> **How can one model be trained when everything is conditioned on different $y^\star$'s?**

We do **not** train one flow for every target spectrum.

Instead, we train **one conditional flow**:

$$
\boxed{
q_\phi(\theta\mid y).
}
$$

The target spectrum $y$ is an input to the model.

For example, suppose our training set contains target spectra

$$
y_1^\star,y_2^\star,\ldots,y_K^\star.
$$

For each target, we sample latent variables:

$$
z_{1},z_{2},\ldots,z_M.
$$

The model generates

$$
\theta_{k,m}
=

f_\phi(z_m;y_k^\star).
$$

We then run the simulator:

$$
\hat y_{k,m}
=

S(\theta_{k,m})
$$

and calculate

$$
d(\hat y_{k,m},y_k^\star).
$$

We can then average the objective over targets and latent samples.

Conceptually:

$$
\boxed{
L(\phi)
=

\mathbb E_{y^\star}
\mathbb E_{z}
\left[
\log q_\phi(\theta\mid y^\star)
-\log p(\theta)
+
\beta d(S(\theta),y^\star)
\right],
}
$$

where

$$
\theta=f_\phi(z;y^\star).
$$

---

## 25. What one training iteration looks like

Suppose one minibatch contains $B$ target spectra:

$$
y_1^\star,\ldots,y_B^\star.
$$

For every target, we sample perhaps $M$ latent vectors:

$$
z_{b,m}\sim N(0,I).
$$

Then:

$$
\theta_{b,m}
=

f_\phi(z_{b,m};y_b^\star).
$$

The simulator produces:

$$
\hat y_{b,m}
=

S(\theta_{b,m}).
$$

We calculate:

$$
d_{b,m}
=

d(\hat y_{b,m},y_b^\star).
$$

The loss is approximately the average:

$$
\boxed{
L
\approx
\frac1{BM}
\sum_{b=1}^{B}
\sum_{m=1}^{M}
\left[
\log q_\phi(\theta_{b,m}\mid y_b^\star)
-\log p(\theta_{b,m})
+
\beta d_{b,m}
\right].
}
$$

Then we backpropagate through the entire chain:

$$
y_b^\star
\rightarrow
f_\phi
\rightarrow
\theta_{b,m}
\rightarrow
S
\rightarrow
d
$$

and update the **same parameters $\phi$**.

The model therefore learns a general rule:

$$
\boxed{
y^\star
\longrightarrow
\text{distribution over suitable }\theta.
}
$$

It is not memorizing one target at a time.

---

## 26. An intuitive example of conditional training

Suppose our training targets contain:

* a bandpass spectrum,
* a bandstop spectrum,
* a Dirac-like spectrum,
* another bandpass with a different center wavelength,
* another bandstop with a different bandwidth.

During training, the same flow sees different conditions:

$$
q_\phi(\theta\mid y_{\text{bandpass}})
$$

then

$$
q_\phi(\theta\mid y_{\text{bandstop}})
$$

then

$$
q_\phi(\theta\mid y_{\text{Dirac}})
$$

and so on.

The flow parameters $\phi$ are shared.

The target $y^\star$ tells the network which conditional distribution it should generate.

Thus the model is learning the general conditional relationship:

$$
\boxed{
\text{spectrum requirement}
\longrightarrow
\text{distribution of physical designs}.
}
$$

At inference time, we give it a new target spectrum:

$$
y_{\mathrm{new}}^\star
$$

and sample

$$
z_1,z_2,\ldots,z_M.
$$

The flow generates

$$
\theta_1,\theta_2,\ldots,\theta_M
$$

which are intended to be different designs for that new target.

---

## 27. Where do the target spectra come from?

This is an important distinction.

Our training does not require a conventional dataset of randomly generated

$$
(\theta,S(\theta))
$$

pairs.

Instead, we can have a collection of desired target spectra:

$$
\boxed{
\mathcal D_y=
\{y_1^\star,y_2^\star,\ldots,y_K^\star\}.
}
$$

These targets can be designed according to the photonics problems we actually care about.

For every training target, the simulator is queried **during training** to evaluate the generated knob configurations.

Thus the expensive question

> "Which random knob configurations should we simulate beforehand so that the dataset contains enough examples of our rare target spectra?"

is replaced by

> "Given the target we care about, which knob configurations does our current generator propose, and what does the simulator say about them?"

This is the motivation for simulator-in-the-loop learning.

---

## 28. Why this is particularly useful for rare target spectra

Suppose random sampling of the 42 knobs almost never produces a spectrum resembling a desired bandpass.

A conventional supervised approach might do:

$$
\theta
\rightarrow
S(\theta)
$$

millions or billions of times, hoping that enough useful examples appear.

But our targets are known in advance.

We already know that a particular spectrum $y^\star$ is interesting.

So instead of waiting for random knob configurations to accidentally produce it, we actively ask the generator:

$$
\boxed{
\text{What knobs could produce this target?}
}
$$

and immediately test those proposed knobs through the simulator.

This focuses simulation effort on the region of design space relevant to the inverse problem.

---

## 29. Simulator gradients complete the loop

Because

$$
S(\theta)
$$

is differentiable, an error in the spectrum can be propagated backward.

The chain is:

$$
z
\rightarrow
\theta
\rightarrow
S(\theta)
\rightarrow
d(S(\theta),y^\star).
$$

We can calculate

$$
\nabla_\theta d
=

J_S^T\nabla_y d.
$$

Then because

$$
\theta=f_\phi(z;y^\star),
$$

we can continue the chain rule:

$$
\boxed{
\nabla_\phi L
=

\frac{\partial L}{\partial\theta}
\frac{\partial\theta}{\partial\phi}.
}
$$

Therefore, the simulator does not merely tell us whether a generated design is good or bad.

Its differentiability tells us **how to change the generated design to improve the spectrum**, and this information can propagate back into the parameters of the conditional generator.

---

## 30. Important distinction: simulator loss versus maximum likelihood

If we train only with

$$
L_{\mathrm{physics}}
=

E_z[d(S(f_\phi(z;y^\star)),y^\star)],
$$

then we are training a **stochastic inverse-design generator**.

We are not automatically performing maximum-likelihood training of a normalizing flow.

If, however, we explicitly define

$$
p(\theta\mid y^\star)
$$

and minimize

$$
D_{\mathrm{KL}}
(q_\phi(\theta\mid y^\star)
|
p(\theta\mid y^\star)),
$$

then we have a **distribution-matching / variational inference formulation**.

The normalizing flow is useful here because it provides a tractable density

$$
q_\phi(\theta\mid y^\star).
$$

This distinction is important: the simulator provides the physical information, while the probabilistic formulation specifies what distribution we want the generator to learn.

---

## 31. What we want to investigate experimentally

The formulation gives us several natural experimental questions.

### 31.1 Latent dimensionality

For a standard normalizing flow,

$$
\dim z=42
$$

is the natural choice.

However, the actual solution variability may have lower intrinsic dimensionality.

We can therefore investigate how alternative generative formulations behave when the latent dimensionality is changed.

The question is:

> **How many latent degrees of freedom are actually required to represent the useful diversity of our photonic solutions?**

---

### 31.2 Different priors $p(\theta)$

We can experiment with different choices of

$$
p(\theta).
$$

For example:

$$
p(\theta)=\text{Uniform}
$$

or priors that favor particular regions of the design space.

The question is:

> **How does our definition of desirable knob configurations affect the generated solution distribution?**

---

### 31.3 Different values of $\beta$

We can investigate:

$$
\beta_1,\beta_2,\ldots
$$

to control the tradeoff between matching the target and maintaining a broader distribution.

The question is:

> **How strongly should the generated designs be required to reproduce the target spectrum?**

---

### 31.4 Spectral resolution

Our current spectrum has

$$
D=400
$$

samples.

We can investigate representations with:

$$
D=400,\qquad D=200,\qquad D=80.
$$

The question is:

> **How much spectral information is necessary for the conditional generator to learn the inverse mapping and the diversity of solutions?**

This is not only a computational question. Reducing the spectrum dimensionality may also change the geometry of the inverse problem because fewer output constraints can potentially leave more ambiguity in the 42-dimensional knob space.

---

### 31.5 Number of latent samples per target

During training, we can generate

$$
M
$$

different $z$'s for each target.

We can investigate whether increasing $M$ improves the representation of the solution distribution.

The question is:

> **How many different generated designs do we need to expose the model to the structure of the solution space?**

---

## 32. An important alternative: best-of-$M$

There is also a simpler formulation worth testing.

For a target $y^\star$, sample

$$
z_1,\ldots,z_M.
$$

Generate

$$
\theta_i=G_\phi(y^\star,z_i).
$$

Run the simulator:

$$
y_i=S(\theta_i).
$$

Calculate

$$
L_i=d(y_i,y^\star).
$$

Then use

$$
\boxed{
L_{\mathrm{best}}
=

\min_i L_i.
}
$$

This asks a different question:

> **Can our generator produce at least one excellent design among $M$ samples?**

This could be useful if our practical objective is simply to obtain one excellent inverse design.

However, it does **not** necessarily learn the entire solution distribution.

For example, the model could generate:

$$
\theta_1,\ldots,\theta_{M-1}
$$

that are poor solutions and one excellent solution $\theta_M$.

The minimum loss would still be excellent.

Therefore:

$$
\boxed{
L_{\mathrm{best}}
}
$$

is suitable for **"at least one good design"**, whereas distribution matching is aimed at **"many different valid designs."**

We can experimentally compare these objectives because they correspond to different practical goals.

---

## 33. The central research question

Everything above can ultimately be reduced to one question:

$$
\boxed{
\textbf{Can we learn a conditional distribution of photonic designs directly through a differentiable simulator, without first constructing a massive supervised dataset of knob--spectrum pairs?}
}
$$

More explicitly, we want to investigate whether we can learn

$$
\boxed{
q_\phi(\theta\mid y^\star)
\approx
p(\theta\mid y^\star)
}
$$

where

$$
p(\theta\mid y^\star)
\propto
p(\theta)
e^{-\beta d(S(\theta),y^\star)},
$$

using the simulator itself as the source of physical supervision.

The resulting system should ideally take a desired spectrum

$$
y^\star
$$

and allow us to sample

$$
\theta_1,\theta_2,\ldots,\theta_M
\sim
q_\phi(\theta\mid y^\star),
$$

such that

$$
S(\theta_i)\approx y^\star
$$

while the generated designs capture meaningful diversity.

---

## 34. The entire idea in one picture

```text
                    OUR HIGH-LEVEL GOAL
                            │
                            ▼
             Given a desired spectrum y*,
       generate MANY different knob configurations
             that produce approximately y*
                            │
                            ▼
              ┌────────────────────────┐
              │  Desired distribution  │
              │       p(θ | y*)        │
              └────────────────────────┘
                            │
                    We cannot observe
                    this distribution
                            │
                            ▼
              ┌────────────────────────┐
              │ Conditional generator  │
              │    qφ(θ | y*)           │
              └────────────────────────┘
                            │
                    θ = fφ(z ; y*)
                            ▲
                            │
                      z ~ N(0,I)
                            │
                  provides variation
                            │
                            ▼
                       θ ∈ R⁴²
                            │
                            ▼
              ┌────────────────────────┐
              │ Differentiable forward │
              │      simulator S       │
              └────────────────────────┘
                            │
                            ▼
                       y = S(θ)
                            │
                            ▼
                  Compare with target
                            │
                            ▼
                  d(S(θ), y*) = error
                            │
                            ▼
                     Backpropagate
                            │
                            ▼
                    Update qφ / fφ
```

The conceptual progression is therefore:

$$
\boxed{
\text{inverse design}
}
$$

becomes

$$
\boxed{
\text{one-to-many inverse design}
}
$$

which becomes

$$
\boxed{
\text{conditional distribution learning}
}
$$

which becomes

$$
\boxed{
q_\phi(\theta\mid y^\star)
\approx
p(\theta\mid y^\star)
}
$$

which can be formulated through

$$
\boxed{
p(\theta\mid y^\star)
\propto
p(\theta)
e^{-\beta d(S(\theta),y^\star)}
}
$$

and learned through a conditional generative model whose samples are evaluated directly by the differentiable simulator.

---

## 35. The key equations

For reference, the core mathematical structure is:

### Forward simulator

$$
\boxed{
y=S(\theta),
\qquad
\theta\in\mathbb R^{42},
\quad
y\in\mathbb R^{400}.
}
$$

### Solution set

$$
\boxed{
\mathcal S(y^\star)
=

\{\theta:S(\theta)\approx y^\star\}.
}
$$

### Desired conditional distribution

$$
\boxed{
p(\theta\mid y^\star)
\propto
p(\theta)
e^{-\beta d(S(\theta),y^\star)}.
}
$$

### Conditional flow

$$
\boxed{
z\sim\mathcal N(0,I),
\qquad
\theta=f_\phi(z;y^\star).
}
$$

### Generated distribution

$$
\boxed{
\theta\sim q_\phi(\theta\mid y^\star).
}
$$

### Desired approximation

$$
\boxed{
q_\phi(\theta\mid y^\star)
\approx
p(\theta\mid y^\star).
}
$$

### Distribution-matching objective

$$
\boxed{
\phi^\star
=

\arg\min_\phi
D_{\mathrm{KL}}
\left(
q_\phi(\theta\mid y^\star)
|
p(\theta\mid y^\star)
\right).
}
$$

### Equivalent training objective

$$
\boxed{
L(\phi)
=

\mathbb E_{q_\phi}
\left[
\log q_\phi(\theta\mid y^\star)
-

\log p(\theta)
+
\beta d(S(\theta),y^\star)
\right].
}
$$

### Simulator gradient

$$
\boxed{
\nabla_\theta d(S(\theta),y^\star)
=

J_S(\theta)^T
\nabla_y d(y,y^\star),
}
$$

where

$$
\boxed{
J_S(\theta)
=

\frac{\partial S}{\partial\theta}
\in\mathbb R^{400\times42}.
}
$$

### Conditional training over multiple targets

$$
\boxed{
L(\phi)
=

\mathbb E_{y^\star}
\mathbb E_{z}
\left[
\log q_\phi(\theta\mid y^\star)
-\log p(\theta)
+
\beta d(S(\theta),y^\star)
\right],
}
$$

with

$$
\theta=f_\phi(z;y^\star).
$$

This last equation is what allows **one conditional model** to learn many different inverse problems simultaneously: every target spectrum supplies a different condition $y^\star$, while the same parameters $\phi$ learn the general mapping from desired spectra to distributions of physical designs.
