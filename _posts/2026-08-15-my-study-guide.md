---
layout: post
title: "My Study Guide: How to Read Math Papers and Textbooks"
date: 2026-08-15
published: false
---

Personal notes on how to actually study mathematical papers and textbooks, rather than passively reading them.

## 1. Papers are read backwards

Mathematical papers are not written in the order in which humans understand them. They are to be read backwards from the question that one wants answered.

## 2. Identify the load-bearing concept

Not every line is equally important. For example, many optimization proofs that look sophisticated are ultimately: smoothness → upper quadratic bound → descent → convergence. Experts do not process more symbols per second, they just recognize the skeleton fast enough.

## 3. Three levels of understanding

**Level 1: Symbolic understanding.** Knowing what every symbol means.

For example: $L\text{-smooth} \iff \|\nabla f(x) - \nabla f(y)\| \le L\|x-y\|$.

**Level 2: Operational understanding.** Knowing what can be done with it.

For example: if $f$ is $L$-smooth, I can upper-bound it by a quadratic:

$$f(y) \le f(x) + \langle \nabla f(x), y-x\rangle + \frac{L}{2}\|y-x\|^2.$$

**Level 3: Geometric understanding.** Knowing why the object exists.

For example: the gradient can't change arbitrarily quickly, so locally the function cannot rise faster than a quadratic.

## 4. Mental compression and representation

A beginner sees:

$$\langle \nabla f(x), y-x\rangle + \frac{L}{2}\|y-x\|^2.$$

An expert sees: "The smoothness quadratic upper bound."

So the long-term objective is not to read more papers per week. It's to increase the number of mathematical patterns you recognize automatically.

## 5. The three-pass method for reading papers

**Pass 1: 15-30 minutes.**

You are not allowed to understand the paper. Read only:

- abstract
- introduction
- theorem statements
- figures
- conclusion
- headings

Then answer:

- What problem is this solving?
- What is the main result?
- Why is it difficult?
- What is the conceptual trick?
- What mathematical machinery is involved?

No reading the proof until those are answered and clear.

**Pass 2: reconstructing the argument.**

For every theorem, write:

- Claim:
- Why should it be true?
- Main obstacle:
- Key trick:
- Tools used:

For example:

- Claim: SGD converges under X.
- Why: each step decreases expected objective.
- Obstacle: stochastic gradient noise.
- Trick: bound noise variance.
- Tools: smoothness + conditional expectation + telescoping.

**Pass 3: technical reconstruction.**

Never spend more than 10-15 minutes on one step. Write a note about the particular point that is missing and investigate it later.

Otherwise: 4 hours → one equation → still confused → entire day destroyed.

## 6. Notes should be local, precise questions, not global confusion

## 7. When notation is horrible, translate the paper into my own language

If a paper says $F_\theta(P) = \mathbb{E}_{x \sim P}[\ell(f_\theta(x), y)]$, write: population loss.

If it defines $\Delta_t = \theta_t - \theta^\star$, write: distance from optimum.

## 8. For reading textbooks: they are not novels

What theorem am I trying to understand?

Then: what prerequisite do I lack?

Then: find that prerequisite.

Then: solve 3-5 problems involving it.

Then: return to the theorem.

## 9. Use problems for constructing the concept, not for testing myself

Never do: definition → theorem → proof → solve exercises.

Reverse that. When you encounter a concept, ask: what is the smallest example where this concept does something?

- For strong convexity, construct examples.
- For Lipschitz continuity, construct examples.
- For duality, actually derive a dual problem.
- For gradient descent, calculate iterations by hand.
- For concentration, calculate the bound for a Bernoulli variable.
- For eigenvalues, take a $2\times2$ matrix.

Make the concept manipulable. Mathematical objects stop being abstract nouns and become things to be pushed around.

## 10. Toy problems are key

Don't study the complicated neural net. Reduce the idea to $f(x) = ax+b$ or $f(x) = x^2$.

If the example $f(x) = x^2$ is not grasped, nothing else will be.

Always: forget the general case. What happens in dimension 1?

## 11. When things get complicated, change representations

Abstract: $x_{t+1} = x_t - \eta \nabla f(x_t)$.

Geometric: move downhill in the direction of steepest descent.

## 12. Fix a limited time for every learning step

$$\text{maximize} \quad \frac{\text{useful mathematical structure acquired}}{\text{hours spent confused}}$$

## 13. Layers of learning

**Layer 1: Core language.** Become extremely comfortable with:

- linear algebra
- multivariable calculus
- probability
- real analysis
- basic measure theory
- basic optimization

**Layer 2: Core ML theory.** Then:

- convex analysis
- statistical learning theory
- concentration inequalities
- empirical process theory
- optimization theory
- stochastic optimization

**Layer 3: Deeper theory.** Depending on the direction:

- functional analysis
- information theory
- variational analysis
- operator theory
- advanced optimization
- ...

The mistake is trying to jump into Layer 3 because the interesting paper uses Layer 3 mathematics. You end up learning ten subjects simultaneously.

## 14. But don't wait until ready

Not: calculus → linear algebra → probability → analysis → optimization → ML.

It is: research paper → discover missing concept → learn concept → return to paper.

## 15. Build my personal "theorem library"

Not a notebook of definitions. A notebook of reusable proof patterns.

Example: pattern: smoothness.

$$f(y) \le f(x) + \langle \nabla f(x), y-x\rangle + \frac{L}{2}\|y-x\|^2.$$

Use when:

- analyzing gradient methods
- bounding function decrease
- deriving convergence rates

## 16. Never ever measure progress by pages

## 17. Answer these questions for every important result

1. What does it say?
2. Why should it be true intuitively?
3. What is the minimal mathematical mechanism that makes it true?
4. Could I reproduce the proof without looking?
