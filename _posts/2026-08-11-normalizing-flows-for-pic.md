---
layout: post
title: "Normalizing Flows for PIC"
date: 2026-08-11
category: pic
---

<!-- Working draft. Fill in each section as the work progresses; push updates any time. -->

## 1. Motivation

The inverse problem is many-to-one: many different knob combinations can produce the same (or nearly the same) spectrum. What we actually want is the full solution set for a given target, so the object to model is **p(knobs | spectrum)**, not a single point estimate.

VAE, MoG-VAE, and cVAE all bake in an assumption about the *shape* of that distribution a single Gaussian, or a mixture of a fixed number of Gaussians. But p(knobs | spectrum) isn't known to have that shape, or any particular shape at all: it's whatever the (highly nonlinear) forward map happens to induce for a given target, and there's no reason to expect it's unimodal or Gaussian-shaped. That mismatch between "the true solution set" and "the shape the model is allowed to represent" is the specific thing that motivates moving to normalizing flows, which make no shape assumption.

<!--
Still to fill in:
- the specific empirical trigger: did a cVAE run actually show blurry/invalid
  samples, multimodal collapse, or another concrete failure that a
  Gaussian-shaped posterior can't represent? or is this purely an a priori
  argument before running the VAE baselines?
-->

## 2. Model and Theory

**Getting the Bayesian labels right first**, since this is easy to mix up: for `p(knobs|spectrum) = p(spectrum|knobs)·p(knobs) / p(spectrum)`,

- `p(knobs | spectrum)` is the **posterior** the thing we want to learn and sample from.
- `p(spectrum | knobs)` is the **likelihood** given *exactly* by the cheap forward simulator, no approximation needed.
- `p(knobs)` is the **prior** over knob configurations (e.g. whatever distribution the training-data sampling scheme induces, see below).
- `p(spectrum)` is the **evidence** (marginal likelihood), the normalizing constant.

There's a second, unrelated use of the word "likelihood" worth flagging so it doesn't get confused with the Bayesian one above: in the normalizing-flow / ML literature, "exact likelihood" means the model can compute an exact density for whatever distribution *it* is trained to output which here is `p(knobs | spectrum)`, i.e. the Bayesian *posterior*, not the Bayesian likelihood term. Same word, two different distributions. The flow gives us an exact density for the posterior; the simulator separately gives us an exact density for the likelihood.

**The flow itself.** Let `x` be a random variable over knob configurations (the space of possible knob combinations, `x ∈ R^42`). We want an invertible transformation `f : R^42 → R^42`, `x ↦ h`, such that the components of `h = (h_1, ..., h_42)` are independent i.e. `h` follows a simple, factorized base distribution. `f` is what the model learns, and because it's invertible, `f^-1 : R^42 → R^42`, `h ↦ x`, lets us go back from an easy-to-sample `h` to an actual knob configuration `x`. (This `h` is the same latent variable referred to as `z` elsewhere in this draft's outline.)

<!--
Still to fill in:
- change of variables in full: log p_K(z_K) = log p_0(z_0) - sum_i log |det J_i|
- the specific architecture (start with conditional RealNVP? affine coupling
  layers: z_a' = z_a, z_b' = z_b * exp(s(z_a)) + t(z_a))
- why the Jacobian is cheap to compute for this architecture (triangular ->
  log det = sum(s(z_a)))
- loss: pure negative log-likelihood, no reconstruction term
-->

## 3. How It's Applied Here

**Inference / sampling.** Once trained, getting a candidate solution is a single forward pass, not an optimization: sample `h ~ (base distribution)`, then `knobs = f^-1(h, spectrum)`. Optionally refine that sample afterward with gradient descent through the differentiable simulator, nudging it to tighten the match to the exact target spectrum this is a separate polishing step, not part of sampling itself.

**Training data.** The simulator being cheap changes the usual objection to conditional flows. Normally you'd worry about needing many observed solutions per target but standard amortized training for a conditional flow doesn't need that: draw knobs from a prior, run the simulator once per draw to get its spectrum, and train on those `(knobs, spectrum)` pairs. The model amortizes across many different spectra in the training set, not across repeated solutions for one spectrum. This matches the existing data-generation plan (Latin Hypercube sampling over knobs, filtered to "interesting" spectrum shapes) described in the README.

**Open question to check empirically, not yet decided:** amortized training only generalizes well to a specific target spectrum if that target's neighborhood is actually represented in the training distribution. Since the targets we care about are specific, not just whatever the LHS-and-filter process happens to produce, we need to check whether coverage is good enough there. If a target turns out to be out-of-distribution relative to the training set, the fallback is a sequential/active scheme draw more knob samples focused near an initial estimate for that target, rerun the simulator, refit (closer to "SNPE"-style sequential simulation-based inference) rather than assuming single-round amortized training is sufficient.

<!--
Still to fill in:
- Flow input/condition: target spectrum Et(λ) (real + imag parts, shape 800);
  confirm this shape/representation
- how this connects back to the simulator-as-decoder architecture used for the
  VAE variants (any reuse of that dataset / training setup?)
-->

## 4. Demo

<!--
Small, concrete example: pick one target spectrum (or a couple of interesting
representative shapes: passband, Dirac-like, triangular), show what the flow
proposes, run it back through the simulator, compare.
Drop plots in /assets/ and reference them, e.g.:
![Target vs. reconstructed spectrum](/assets/nf-demo-spectrum.png)
-->

## 5. Results

<!--
What actually happened. Be concrete and quantitative where possible:
- reconstruction error vs. the VAE/cVAE baselines, on the same held-out spectra
- coverage/diversity of sampled knob solutions for a fixed target (does it
  find multiple valid solutions, or collapse to one?)
- anything surprising, and what it suggests for the next step
-->
