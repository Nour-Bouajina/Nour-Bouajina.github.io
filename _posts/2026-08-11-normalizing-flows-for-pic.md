---
layout: post
title: "Normalizing Flows for PIC"
date: 2026-08-11
category: pic
---

<!-- Working draft. Fill in each section as the work progresses; push updates any time. -->

## 1. Motivation

<!--
Why normalizing flows, specifically, for this problem? Recap briefly:
- the inverse problem is many-to-one (many knob combos -> same spectrum)
- VAE / MoG-VAE / cVAE all assume the solution distribution p(knobs | spectrum)
  is (a mixture of) Gaussians, which may not hold
- what specifically pushed you from "try a flow" to "we need a flow": e.g. did a
  cVAE run show blurry/invalid samples, multimodal collapse, a specific failure
  that a Gaussian-shaped posterior can't represent?
-->

## 2. Model and Theory

<!--
Explain the flow itself, from first principles:
- change of variables: log p_K(z_K) = log p_0(z_0) - sum_i log |det J_i|
- why this gives an EXACT likelihood, no ELBO approximation like the VAE family
- the specific architecture you're using (start with conditional RealNVP?
  affine coupling layers: z_a' = z_a, z_b' = z_b * exp(s(z_a)) + t(z_a))
- why the Jacobian is cheap to compute for this architecture (triangular ->
  log det = sum(s(z_a)))
- loss: pure negative log-likelihood, no reconstruction term
-->

## 3. How It's Applied Here

<!--
Concretely, for this problem:
- Flow input/condition: target spectrum Et(λ) (real + imag parts, shape 800)
- Flow output: distribution over the 42 knobs
- p(knobs | spectrum) learned as a conditional flow
- inference procedure: sample z ~ N(0,1), knobs = flow_inverse(z, spectrum),
  verify simulator(knobs) ≈ spectrum
- how this connects back to the simulator-as-decoder architecture used for
  the VAE variants (any reuse of that dataset / training setup?)
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
