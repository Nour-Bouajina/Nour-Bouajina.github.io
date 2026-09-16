---
layout: post
title: "Mamba, Mamba-2, and Mamba-3: State Space Models From Scratch"
date: 2026-09-16
---

<style>
.exercise { border-left: 4px solid #395C68; background: rgba(57,92,104,0.06); padding: 0.9em 1.3em; margin: 1.6em 0; border-radius: 4px; }
.exercise h4 { margin-top: 0; margin-bottom: 0.5em; }
details { margin: 0.5em 0 1em 0; }
details summary { cursor: pointer; font-weight: 600; color: #395C68; }
details[open] summary { margin-bottom: 0.6em; }
.diagram { margin: 2em 0; text-align: center; }
.diagram svg { max-width: 100%; height: auto; }
.diagram figcaption { font-size: 0.85em; color: #555; margin-top: 0.6em; }
.flow { display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 0.4em; margin: 1.6em 0; font-size: 0.95em; }
.flow .box { border: 1.5px solid #395C68; border-radius: 6px; padding: 0.5em 0.8em; background: rgba(57,92,104,0.05); text-align: center; }
.flow .arrow { font-size: 1.3em; color: #395C68; }
.flow-col { display: flex; flex-direction: column; align-items: center; gap: 0.3em; }
.small { font-size: 0.85em; color: #555; }
table.clean { border-collapse: collapse; width: 100%; margin: 1.2em 0; }
table.clean th, table.clean td { border: 1px solid #ccc; padding: 0.4em 0.7em; text-align: left; font-size: 0.9em; }
table.clean th { background: rgba(57,92,104,0.08); }
</style>

*A from-scratch, self-contained walk from "what is a differential equation" to the Mamba-3 paper (2026), built around one running analogy. Every worked exercise has a hint and a full answer hidden behind a click, try each one on paper before opening it.*

## Why any of this matters

Transformers have a problem hiding in plain sight. Attention lets every token look at every other token, which is why transformers are so good, but it means the cost of processing a sequence of length $L$ grows like $L^2$, and at generation time you have to keep every past token's key and value sitting in memory, growing forever as the conversation gets longer. Old-fashioned RNNs have the opposite problem: they carry a small fixed-size summary forward one step at a time, which makes generation cheap and memory constant, but training them is serial, you cannot compute step 500 before step 499 is done, so they cannot use a GPU's parallelism during training the way transformers can.

The dream that this whole post is chasing: a model with an RNN's cheap, constant-size memory at generation time, that somehow *also* trains as parallelizable as a transformer. That dream has a name: **structured state space models**. Mamba (2023) is the version that made it actually work well for language. Mamba-2 (2024) found a startling mathematical link between that RNN-like machine and attention itself. Mamba-3, published in 2026, fixes three concrete things that were quietly broken in both of its predecessors. We are going to build all three, from the ground up, assuming nothing beyond basic algebra going in.

The running analogy for the whole post: **a bank account that earns continuous interest and receives continuous deposits.** Keep that picture in your head, everything else is a variation on it.

## Act 1: what a "state" even is

Imagine reading a 400 page novel and being asked on page 200 "what has happened so far?" You don't re-read the first 200 pages, you keep a compressed mental summary and update it as you read. That compressed summary is a **state**. Every sequence model needs one, the entire design question is: how do you update it, and how expensive is that update?

A transformer's answer is "don't summarize, just look back at everything" (attention). An RNN's answer is "keep a small summary and update it one token at a time." A state space model is going to give an RNN-flavored answer, but built out of continuous-time mathematics borrowed from control theory and physics, which turns out to unlock a much richer set of computational tricks than a standard RNN's update rule allows.

## Act 2: the continuous-time picture

Here is the running analogy stated as math. You have a bank balance $x(t)$ at time $t$. It earns continuous interest at rate $a$ (if $a>0$ it grows, if $a<0$ it decays, think of it as a leak), and money is deposited into it continuously at rate $u(t)$, scaled by some factor $b$. The balance's rate of change is:

$$\dot{x}(t) = a\,x(t) + b\,u(t)$$

where $\dot{x}(t)$ just means $\frac{d}{dt}x(t)$, the rate of change of $x$ with respect to time, I'll write it as $\dot x(t)$ from here since that's what you'll see in every paper, but it means nothing more than "how fast is $x$ changing right now."

This is a **differential equation**: an equation relating a function to its own rate of change. It's the smallest possible nontrivial one, everything in this post is built out of bigger versions of exactly this line.

The general, vector-valued version, which is what actual state space models use, replaces the single balance $x(t)$ with a whole vector of "memory cells" $x(t) \in \mathbb{R}^N$, replaces $a$ with a matrix $A$ (an $N \times N$ matrix describing how the memory cells drive each other), replaces $b$ with a matrix $B$ (how the single input feeds into each memory cell), and adds a readout equation that turns the internal memory into an actual output number using a matrix $C$:

$$\dot{x}(t) = Ax(t) + Bu(t), \qquad y(t) = Cx(t)$$

That's the entire object called a **state space model (SSM)**. $x(t)$ is not the input or output, it's the hidden memory. $A$ is the internal dynamics (what the memory does to itself, left alone). $B$ is how new information enters. $C$ is how you read an answer back out.

<div class="exercise">
<h4>Exercise 1: solve the smallest case, by hand</h4>

Solve the scalar equation $\dot x(t) = a\,x(t) + b\,u(t)$ for $x(t)$, given $x(0)=x_0$ and a general (not necessarily constant) deposit rate $u(t)$. Leave $u(t)$ inside an integral, don't assume it's constant. Method: multiply both sides by an "integrating factor" $\mu(t)$ chosen so the entire left side collapses into a single derivative $\frac{d}{dt}[\mu(t)x(t)]$, then integrate both sides.

<details>
<summary>Hint</summary>
Recall the product rule: $\frac{d}{dt}[\mu(t)x(t)] = \mu(t)\dot x(t) + \dot\mu(t) x(t)$. If you multiply the original equation through by $\mu(t)$, you get $\mu(t)\dot x(t) - a\mu(t)x(t) = \mu(t) b u(t)$ (after moving the $a x(t)$ term to the left). Compare the left side to the product rule above: for them to match, what does $\dot\mu(t)$ need to equal, in terms of $\mu(t)$? That's a tiny equation with no $x$ or $u$ in it, its solution is a function you already know from basic calculus: whatever grows or decays at a rate proportional to itself.
</details>

<details>
<summary>Answer</summary>

The integrating factor is $\mu(t) = e^{-at}$ (since $e^{kt}$ is exactly the function whose own derivative is $k$ times itself, here $k=-a$). Multiplying through:

$$e^{-at}\dot x(t) - a e^{-at} x(t) = e^{-at} b\, u(t)$$

The left side is exactly $\frac{d}{dt}\left[e^{-at}x(t)\right]$ by the product rule. Integrating both sides from $0$ to $t$:

$$e^{-at}x(t) - x(0) = \int_0^t e^{-as}\, b\, u(s)\, ds$$

$$x(t) = e^{at}x_0 + \int_0^t e^{a(t-s)}\, b\, u(s)\, ds$$

**Why this is the important sentence of the whole post:** the current balance is the decayed starting balance, plus a weighted sum over every past deposit, where older deposits get multiplied by a bigger power of the decay. This is memory: a single scalar $a$ controls how fast old information fades. Every state space model, all the way to Mamba-3, is a fancier version of "control how fast things fade, and control what gets let in."
</details>
</div>

## Act 3: from continuous to discrete

Real data comes as tokens, not a continuous signal, so we need to turn that continuous update into a step-by-step recurrence. Chop time into steps of size $\Delta$ (this step size becomes a genuinely important, even learned, quantity later, so don't file it away as a boring implementation detail). Over one short interval, assume the input is held constant.

<div class="exercise">
<h4>Exercise 2: derive the discrete recurrence</h4>

Using the exact solution from Exercise 1, evaluate the balance one step of length $\Delta$ later, holding $u(s) = u_{k-1}$ constant over that interval. You should get a recurrence of the form $x_k = \bar a\, x_{k-1} + \bar b\, u_{k-1}$. Find $\bar a$ and $\bar b$ exactly, then find what $\bar b$ simplifies to when $\Delta$ is small (first-order approximation, i.e. keep only the linear term of $e^{a\Delta} \approx 1 + a\Delta$).

<details>
<summary>Hint</summary>
Plug $u(s) = u_{k-1}$ (constant) into the integral from Exercise 1, over an interval of length $\Delta$. You'll need $\int_0^\Delta e^{a\tau}\,d\tau$, which has a clean closed form. Do the exact version first, then Taylor-expand $e^{a\Delta}$ to get the small-step approximation.
</details>

<details>
<summary>Answer</summary>

Exact (this is called **zero-order hold**, ZOH):

$$\bar a = e^{a\Delta}, \qquad \bar b = \frac{e^{a\Delta}-1}{a}\,b$$

Small-$\Delta$ approximation ($e^{a\Delta}\approx 1+a\Delta$):

$$\bar b \approx \frac{(1+a\Delta)-1}{a}b = \Delta b$$

So the crude version is just $x_k \approx e^{a\Delta}x_{k-1} + \Delta b\, u_{k-1}$. Hang on to both versions, exact ZOH and this cruder linear approximation, they matter a lot in Act 9: it turns out Mamba-1 and Mamba-2 both quietly use the *crude* approximation for the input term while advertising ZOH, and Mamba-3's first fix is exactly about that gap.
</details>
</div>

From here on, switch notation to match what every paper actually writes: the hidden memory is called $h_t$ (not $x$), and the input token at position $t$ is called $x_t$ (yes, the letter swaps meaning, that's genuinely confusing the first time, it's just a naming collision between the control-theory tradition and the deep learning tradition). The discrete recurrence, in paper notation:

$$h_t = \bar A\, h_{t-1} + \bar B\, x_t, \qquad y_t = C\, h_t$$

## Act 4: two views of the same computation

If $A$, $B$, $C$ never change across time steps (this property is called **linear time-invariant**, LTI), something remarkable happens: you can unroll the recurrence and write the output directly as a weighted sum over all past inputs:

$$y_t = \sum_{k=0}^{t} C\bar A^{k}\bar B\; x_{t-k}$$

That is a **convolution**: sliding a fixed filter $K_j = C\bar A^{j}\bar B$ over the input sequence. So the exact same function can be computed two completely different ways:

<div class="flow">
  <div class="flow-col">
    <div class="box"><b>Recurrent view</b><br><span class="small">serial, one step at a time</span></div>
    <div class="small">h<sub>0</sub> → h<sub>1</sub> → h<sub>2</sub> → h<sub>3</sub> → ...</div>
    <div class="small">O(1) memory, O(1) per generated token</div>
  </div>
  <div class="flow-col"><span class="arrow">=</span><div class="small">same y<sub>t</sub></div></div>
  <div class="flow-col">
    <div class="box"><b>Convolutional view</b><br><span class="small">one filter K slid over the whole sequence</span></div>
    <div class="small">y = K * x, computed via FFT</div>
    <div class="small">O(L log L), fully parallel over the sequence</div>
  </div>
</div>

This is the whole reason the S4 model (the direct ancestor of Mamba) was exciting before Mamba even existed: train with the fast parallel convolution, deploy with the cheap recurrence, same math, pick whichever algorithm suits the job. It is the same idea as choosing between two provably-equal ways to compute a sum, just applied to a sequence model.

One more thing S4 needed, briefly, because you'll see the name: a plain random matrix $A$ makes the memory decay into noise almost immediately. S4 uses a specially constructed matrix (from a scheme called **HiPPO**) that makes the hidden state behave like the coefficients of a running polynomial approximation of the entire input history, so old information is compressed rather than destroyed. This is a genuinely deep rabbit hole (it lives in the same territory as orthogonal polynomials and functional analysis) and it is not load-bearing for understanding Mamba's actual contribution, so we take it on faith and move on, exactly the kind of layer-3 detour the study guide says to skip unless the current goal needs it.

<div class="exercise">
<h4>Exercise 3: verify the two views agree, with real numbers</h4>

Let $\bar A = 0.5$, $\bar B = 1$, $C=1$, $h_0=0$, and inputs $x_1=1,\ x_2=0,\ x_3=1$. Compute $h_1,h_2,h_3$ and $y_3$ using the recurrence. Then compute $y_3$ again using the convolution sum $y_3 = C\bar B x_3 + C\bar A\bar B x_2 + C\bar A^2 \bar B x_1$, and confirm they match.

<details>
<summary>Hint</summary>
The recurrence is just $h_t = 0.5\,h_{t-1} + x_t$. Grind through $h_1, h_2, h_3$ one at a time, then plug the same three numbers into the convolution formula (which is just three multiply-adds, no recursion needed).
</details>

<details>
<summary>Answer</summary>

Recurrence: $h_1 = 0.5(0)+1 = 1$, $h_2 = 0.5(1)+0=0.5$, $h_3=0.5(0.5)+1=1.25$, so $y_3 = 1.25$.

Convolution: $y_3 = (1)(1)(1) + (1)(0.5)(1)(0) + (1)(0.25)(1)(1) = 1 + 0 + 0.25 = 1.25$. Same answer, two completely different computational paths. That's the duality in miniature, you'll see a much bigger version of it in Act 8.
</details>
</div>

## Act 5: the problem with being time-invariant, and Mamba's fix

Here's the catch. Because $A,B,C$ never change, the filter treats every position identically no matter what token is actually sitting there, exactly like a fixed audio EQ filter applies the same boost to every occurrence of a given frequency, it cannot decide "boost this instance but not that one" based on content. That's a real limitation: tasks like "copy the value that followed the last occurrence of this exact token" need content-based decisions, and a fixed linear filter mathematically cannot express that.

Mamba's core idea (this is the single most important sentence in the original 2023 paper): make $\Delta$, $B$, and $C$ **functions of the current input token**, computed via small learned linear projections of $x_t$. Concretely:

- $\Delta_t = \text{softplus}(\text{linear}(x_t))$, always positive. A large $\Delta_t$ means "take a big step, let this token strongly overwrite the memory." A tiny $\Delta_t$ means "barely move, keep what was already there."
- $B_t = \text{linear}(x_t)$, controls how much of *this token's content* is allowed into the memory.
- $C_t = \text{linear}(x_t)$, controls how the memory gets read out at this position.

This is called **selection**, and it's the reason for the model's name: Mamba is a *selective* state space model. The model now gets to choose, token by token, whether to remember, overwrite, or ignore.

<div class="exercise">
<h4>Exercise 4: feel what selection buys you, with real numbers</h4>

Sequence: $x = [3, 0, 5]$, where the middle $0$ is a meaningless filler token you'd like the model to skip over so it doesn't dilute the memory of the $3$ before the $5$ arrives. Use $\bar A_t = e^{-\Delta_t}$ and $\bar B_t = \Delta_t$ (the small-step approximation from Exercise 2, with $b=1$), $C=1$, $h_0=0$.

**Non-selective baseline:** $\Delta_t = 1$ for every token, always (so $\bar A \approx 0.37$, $\bar B=1$, constant). Compute $h_1, h_2, h_3$.

**Selective version:** $\Delta_t = 1$ for the meaningful tokens ($x_1=3$ and $x_3=5$), but $\Delta_t = 0.01$ for the filler zero ($\bar A \approx 0.99$, $\bar B = 0.01$). Compute $h_1, h_2, h_3$ again.

Compare the final $h_3$ in both cases. Which one preserves more memory of the $3$ by the time the model reaches the $5$?

<details>
<summary>Hint</summary>
It's the same recurrence, $h_t = \bar A_t h_{t-1} + \bar B_t x_t$, run twice with different $\bar A_t,\bar B_t$ on the middle step only. The first and third steps use identical numbers in both versions, only step 2 differs. Use $\bar A \approx 0.37$ when $\Delta=1$ and $\bar A \approx 0.99$ when $\Delta = 0.01$.
</details>

<details>
<summary>Answer</summary>

**Non-selective** ($\bar A=0.37,\bar B=1$ throughout): $h_1 = 0.37(0)+1(3)=3$. $h_2=0.37(3)+1(0)=1.11$. $h_3=0.37(1.11)+1(5)=0.41+5=5.41$. The memory of the $3$ has decayed to a contribution of about $0.41$ by the end, because the filler token forced a full decay step regardless of the fact that it carried no information.

**Selective** ($\bar A=0.99,\bar B=0.01$ only on the filler step): $h_1=0.37(0)+1(3)=3$ (same, real token). $h_2=0.99(3)+0.01(0)=2.97$ (barely touched). $h_3=0.37(2.97)+1(5)=1.10+5=6.10$. The memory of the $3$ contributes about $1.10$, roughly $2.7\times$ more than the non-selective case.

**Why this matters:** selection lets the model decide, per token, how much the past should be allowed to fade. A fixed-rate filter cannot protect memory from irrelevant tokens; a selective one can. This is the entire quality jump of Mamba over S4, in one hand-computed example.
</details>
</div>

## Act 6: getting the speed back, the hardware-aware parallel scan

There's an immediate problem with selection: since $\bar A_t, \bar B_t$ now change every step, the system is no longer time-invariant, so the neat "just FFT a fixed filter" trick from Act 4 is gone. Naively, that means back to a serial, one-step-at-a-time computation, which would make training painfully slow on a GPU built for doing thousands of things at once.

The fix: notice that the update $h_t = \bar A_t h_{t-1} + \bar B_t x_t$ is an **affine map**, and affine maps compose. If step 1 is $f_1(h) = a_1 h + c_1$ and step 2 is $f_2(h) = a_2 h + c_2$, then applying step 1 then step 2 is itself a single affine map:

$$f_2(f_1(h)) = a_2(a_1 h + c_1) + c_2 = (a_2 a_1)\,h + (a_2 c_1 + c_2)$$

Combining two adjacent steps into one is a well-defined, order-respecting operation, which means you can combine steps 1&2 and steps 3&4 *at the same time*, then combine those two results together, and so on, doubling the span at each round. This is the classic **parallel scan** (the same idea behind parallel prefix-sum), and it computes the exact same sequence of states as the serial recurrence, but in $\log_2(L)$ rounds instead of $L$ serial steps.

<div class="flow">
  <div class="flow-col"><div class="box">step 1</div></div>
  <div class="flow-col"><div class="box">step 2</div></div>
  <div class="flow-col"><div class="box">step 3</div></div>
  <div class="flow-col"><div class="box">step 4</div></div>
</div>
<div class="flow">
  <div class="flow-col"><div class="box">(1⊕2)</div></div>
  <div class="flow-col"><div class="box">(3⊕4)</div></div>
</div>
<div class="flow">
  <div class="flow-col"><div class="box">(1⊕2⊕3⊕4)</div></div>
</div>

There's a second, less glamorous but equally important piece called **hardware-aware**: even with a parallel algorithm, if you write every intermediate hidden state out to the GPU's slow main memory for every one of thousands of tokens, you become bottlenecked on data movement, not arithmetic, exactly the problem FlashAttention solved for attention. Mamba fuses the whole scan into one GPU kernel that keeps the expanding state in fast on-chip memory and recomputes intermediate values during the backward pass instead of storing all of them. Two separate ideas, often conflated: a parallel *algorithm* (the scan) and an IO-conscious *implementation* (keep it in fast memory). Mamba needed both.

<div class="exercise">
<h4>Exercise 5: run the parallel scan by hand and check it against the serial answer</h4>

Four steps, each an affine pair $(a_t, c_t)$ with combination rule $(a_1,c_1)\oplus(a_2,c_2) = (a_1 a_2,\ a_2 c_1 + c_2)$ (meaning "step 1 then step 2"). Steps: $(0.5, 1), (0.5, 2), (0.5, 0), (0.5, 4)$, starting from $h_0=0$.

(a) Compute $h_1, h_2, h_3, h_4$ the ordinary serial way.

(b) Compute the same $h_4$ using two rounds of pairwise combination: first combine (step1⊕step2) and (step3⊕step4) separately, then combine those two results together. Confirm you get the same $h_4$.

<details>
<summary>Hint</summary>
For (a), $h_t = 0.5\,h_{t-1}+c_t$. For (b), apply the combination formula to get $(a_{12},c_{12})$ from steps 1&2, then $(a_{34},c_{34})$ from steps 3&4, then combine those two pairs into one $(a_{14},c_{14})$. Applying that final pair to $h_0=0$ should reproduce $h_4$.
</details>

<details>
<summary>Answer</summary>

(a) $h_1 = 0.5(0)+1=1$. $h_2=0.5(1)+2=2.5$. $h_3=0.5(2.5)+0=1.25$. $h_4=0.5(1.25)+4=4.625$.

(b) Combine steps 1&2: $(0.5\cdot0.5,\ 0.5\cdot1+2) = (0.25, 2.5)$. Combine steps 3&4: $(0.5\cdot0.5,\ 0.5\cdot0+4)=(0.25,4)$. Combine those two: $(0.25\cdot0.25,\ 0.25\cdot2.5+4) = (0.0625, 4.625)$. Applied to $h_0=0$: $0.0625(0)+4.625 = 4.625$. Matches exactly, in $\log_2(4)=2$ combination rounds instead of 4 serial steps.
</details>
</div>

## Act 7: the Mamba block

Putting the selective scan into an actual neural network layer. The input projects into two parallel branches: one runs through a short causal 1D convolution (a few neighboring tokens smoothed together) then a SiLU activation then the selective scan we just built; the other is a plain gating branch (linear projection, SiLU). The two branches multiply elementwise, then project back down, wrapped in a residual connection and a normalization layer.

<div class="flow">
  <div class="box">input</div><span class="arrow">→</span>
  <div class="flow-col">
    <div class="box">linear projection, split in two</div>
  </div>
</div>
<div class="flow">
  <div class="flow-col">
    <div class="box">short causal conv1d</div><span class="small">↓</span>
    <div class="box">SiLU</div><span class="small">↓</span>
    <div class="box">selective SSM scan</div>
  </div>
  <div class="flow-col"><div class="box">gate branch: linear → SiLU</div></div>
</div>
<div class="flow">
  <div class="box">elementwise multiply (gating)</div><span class="arrow">→</span>
  <div class="box">linear projection down</div><span class="arrow">→</span>
  <div class="box">+ residual, norm</div>
</div>

Stack this one block type repeatedly, there's no separate attention layer to alternate with, unlike a transformer's attention-then-MLP pattern. At generation time the cost per new token is constant (update one fixed-size state), independent of how long the context has grown, versus a transformer's ever-growing key/value cache.

**Pause and check yourself, no notes:** (1) what does selection say, in your own words? (2) why should it be true that it helps, intuitively? (3) what's the smallest mechanism that makes it work (hint: it's one word from Act 5)? (4) could you reproduce Exercise 4's numbers from memory? If the answer to (4) is no, that's exactly what belongs in a revisit queue, not a vague "review this again", tell me the specific step that didn't come back and we'll log it precisely.

## Act 8: Mamba-2, and the surprising duality with attention

Mamba-2's core discovery: restrict $\bar A_t$ to be a single scalar (times identity) rather than a full diagonal matrix, meaning every channel inside one "head" decays at the same shared rate, only $B_t, C_t$, and the input still vary per channel. Under that restriction, the entire sequence's output can be written as **one matrix multiplication**:

$$y_t = \sum_{s \le t} \underbrace{C_t\, a^{t-s}\, B_s}_{L_{t,s}}\; x_s$$

$L$ is a lower-triangular matrix (only $s\le t$ contributes, tokens can't see the future) whose entries are a decay factor times a content-based weight. That is *exactly the shape of attention*: a masked matrix product between something query-like ($C$), something key-like ($B$), and the values ($x$), except the mask entries are structured decay products instead of softmax scores. This is the **SSD duality** (structured state space duality): the recurrent form and this quadratic masked-matmul form compute the identical function, the same relationship as the recurrent/convolutional views in Act 4, but now covering the *selective* case too.

Why bother with the quadratic form if the recurrence is cheaper? Because GPUs have tensor cores built for large matrix multiplies, and a short local chunk of tokens computed as one matmul uses that hardware far better than a long serial scan. Mamba-2's actual algorithm chunks the sequence: inside a chunk, use the fast matmul form; between chunks, pass along only a small state using the cheap recurrent form. Best of both, and because the matmul form is efficient, Mamba-2 can afford a much bigger state size than Mamba-1, organized into multiple heads that share $B_t,C_t$ across groups of channels, similar to grouped-query attention.

<div class="diagram">
<svg viewBox="0 0 340 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Lower triangular matrix L with decay-weighted entries">
<style>
text{font-family:inherit;font-size:13px;fill:#333}
.cell{stroke:#395C68;stroke-width:1}
</style>
<text x="170" y="20" text-anchor="middle" font-weight="600">the L matrix from Exercise 6 (a = 0.5)</text>
<g transform="translate(60,40)">
<rect class="cell" x="0" y="0" width="70" height="70" fill="rgba(57,92,104,0.85)"/>
<text x="35" y="40" text-anchor="middle" fill="#fff">1.0</text>
<rect class="cell" x="70" y="0" width="70" height="70" fill="#fff"/>
<rect class="cell" x="140" y="0" width="70" height="70" fill="#fff"/>
<rect class="cell" x="0" y="70" width="70" height="70" fill="rgba(57,92,104,0.35)"/>
<text x="35" y="110" text-anchor="middle">0.5</text>
<rect class="cell" x="70" y="70" width="70" height="70" fill="rgba(57,92,104,0.85)"/>
<text x="105" y="110" text-anchor="middle" fill="#fff">2.0</text>
<rect class="cell" x="140" y="70" width="70" height="70" fill="#fff"/>
<rect class="cell" x="0" y="140" width="70" height="70" fill="rgba(57,92,104,0.35)"/>
<text x="35" y="180" text-anchor="middle">0.5</text>
<rect class="cell" x="70" y="140" width="70" height="70" fill="rgba(57,92,104,0.6)"/>
<text x="105" y="180" text-anchor="middle" fill="#fff">2.0</text>
<rect class="cell" x="140" y="140" width="70" height="70" fill="rgba(57,92,104,0.6)"/>
<text x="175" y="180" text-anchor="middle" fill="#fff">2.0</text>
</g>
<text x="95" y="235" text-anchor="middle" class="small">s=1</text>
<text x="165" y="235" text-anchor="middle" class="small">s=2</text>
<text x="235" y="235" text-anchor="middle" class="small">s=3</text>
<text x="30" y="80" text-anchor="middle" class="small">t=1</text>
<text x="30" y="150" text-anchor="middle" class="small">t=2</text>
<text x="30" y="220" text-anchor="middle" class="small">t=3</text>
<text x="170" y="270" text-anchor="middle" class="small">white = masked out (future), darker = larger weight</text>
</svg>
<figcaption>Same shape as a causal attention mask, but the weights come from decay × content, not softmax.</figcaption>
</div>

<div class="exercise">
<h4>Exercise 6: build the L matrix yourself and confirm it reproduces the recurrence</h4>

$a=0.5$ (shared scalar decay), $B_1=1, B_2=2, B_3=1$, $C_1=1, C_2=1, C_3=2$, inputs $x_1=x_2=x_3=1$, $h_0=0$.

(a) Run the ordinary recurrence $h_t = a h_{t-1} + B_t x_t$, $y_t = C_t h_t$, to get $y_1,y_2,y_3$.

(b) Build the $3\times 3$ matrix with entries $L_{t,s} = C_t\, a^{t-s}\, B_s$ for $s \le t$ (zero above the diagonal), and compute $y = Lx$. Confirm it matches (a).

<details>
<summary>Hint</summary>
For (a) it's the same recurrence you've now done three times in this post. For (b), fill in the 3x3 grid entry by entry using the formula, remembering $a^0=1$, and that entries with $s>t$ are zero. Then do ordinary matrix-vector multiplication with $x=(1,1,1)$.
</details>

<details>
<summary>Answer</summary>

(a) $h_1=0.5(0)+1(1)=1 \Rightarrow y_1=1(1)=1$. $h_2=0.5(1)+2(1)=2.5 \Rightarrow y_2=1(2.5)=2.5$. $h_3=0.5(2.5)+1(1)=2.25 \Rightarrow y_3=2(2.25)=4.5$.

(b) $L_{11}=C_1 a^0 B_1=1$. $L_{21}=C_2 a^1 B_1=0.5$, $L_{22}=C_2 a^0 B_2=2$. $L_{31}=C_3 a^2 B_1=2(0.25)(1)=0.5$, $L_{32}=C_3 a^1 B_2=2(0.5)(2)=2$, $L_{33}=C_3 a^0 B_3=2(1)(1)=2$.

$$L = \begin{pmatrix}1 & 0 & 0\\ 0.5 & 2 & 0\\ 0.5 & 2 & 2\end{pmatrix}, \qquad Lx = \begin{pmatrix}1\\2.5\\4.5\end{pmatrix}$$

Matches (a) exactly. This is the duality made concrete: a recurrence and a masked matrix multiply computing the identical numbers.
</details>
</div>

## Act 9: Mamba-3 (2026), three fixes at once

Mamba-3 doesn't change the philosophy, it repairs three specific, concrete gaps in Mamba-1/2. All three connect directly to things already built above.

### 9a. The discretization was quietly first-order (callback to Act 3)

Remember Exercise 2's two versions: exact ZOH, $\bar B = \frac{e^{a\Delta}-1}{a}b$, versus the crude small-step approximation, $\bar B \approx \Delta b$. The Mamba-3 authors formally show that Mamba-1 and Mamba-2, despite describing themselves as using ZOH, actually implement the crude version for the input term. In numerical-methods language, that's **Euler's method**, and Euler's method approximates the area under a curve between two points using only the height at the *left* endpoint, a rectangle. It's simple, but the error shrinks slowly as the step gets smaller.

The classic fix from basic calculus is the **trapezoidal rule**: approximate that same area using the *average* of the heights at both endpoints, a trapezoid instead of a rectangle, which is a strictly better approximation for the same step size. Mamba-3 does exactly this, but makes the mixing weight between "old endpoint" and "new endpoint" a learned, data-dependent number $\lambda_t \in [0,1]$ per token:

$$h_t = \underbrace{e^{\Delta_t A_t}}_{\alpha_t} h_{t-1} + \underbrace{(1-\lambda_t)\Delta_t e^{\Delta_t A_t}}_{\beta_t}\,B_{t-1}x_{t-1} + \underbrace{\lambda_t \Delta_t}_{\gamma_t}\, B_t x_t$$

In plain words: instead of updating the state using only the newest token's contribution, the state also keeps a fading echo of the *previous* token's contribution, blended in with a learned weight. It's literally a small, width-2 convolution baked directly into the recurrence itself, rather than bolted on beforehand. Setting $\lambda_t=1$ recovers plain Euler (Mamba-1/2's behavior); $\lambda_t=1/2$ recovers the textbook trapezoidal rule exactly.

### 9b. Real numbers can't rotate (state tracking)

Try to track something like parity: "is the number of 1s seen so far even or odd?" Every time you see a 1, the answer must flip. A state governed only by real decay/growth (our $a$ from Act 2, always just shrinking or stretching along one direction) can never cleanly flip between two states on command, it can only ease toward zero or blow up. What *can* flip cleanly is a **rotation**: multiplying a 2D vector by $e^{i\theta}$ rotates it by angle $\theta$; rotate by $\pi$ (180°) on a 1, rotate by $0$ on a 0, and you get an exact parity flag for free, no decay required.

Mamba-3 makes the state complex-valued, with the imaginary part supplying a rotation angle $\theta_t$ that, crucially, depends on the current token's content, not just its position. The complex update turns out to be mathematically identical to applying the "rotary position embeddings" trick already used all over transformers (RoPE), except the rotation angle here is chosen by the *content* of each token rather than by its raw position:

$$h_t = e^{\Delta_t A_t} h_{t-1} + \Big(\textstyle\prod_{i=0}^{t} R_i^\top\Big)\Delta_t B_t x_t, \qquad y_t = \Big[\big(\textstyle\prod_{i=0}^{t}R_i^\top\big)C_t\Big]^\top h_t$$

where each $R_i$ is a small rotation matrix built from that token's own $\theta_i$. The payoff is dramatic and directly testable: on a formal parity task, Mamba-2 scores essentially chance, $0.9\%$; Mamba-3 scores $100.0\%$. On arithmetic-with-brackets, Mamba-2 gets $0.88\%$, Mamba-3 gets $87.75\%$. Strip out the data-dependence (use standard, content-independent RoPE instead) and performance collapses right back down to near-chance, which is the clean experimental proof that it's the *content-dependence* of the rotation doing the work, not rotation in general.

<div class="diagram">
<svg viewBox="0 0 460 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Real decay versus complex rotation trajectories">
<style>text{font-family:inherit;font-size:12px;fill:#333}</style>
<text x="115" y="20" text-anchor="middle" font-weight="600">real state (decay only)</text>
<line x1="30" y1="110" x2="200" y2="110" stroke="#ccc"/>
<circle cx="190" cy="110" r="4" fill="#395C68"/>
<circle cx="150" cy="110" r="4" fill="#395C68"/>
<circle cx="110" cy="110" r="4" fill="#395C68"/>
<circle cx="75" cy="110" r="4" fill="#395C68"/>
<circle cx="50" cy="110" r="4" fill="#395C68"/>
<text x="115" y="135" text-anchor="middle" class="small">can only ease toward zero, never flip</text>

<text x="345" y="20" text-anchor="middle" font-weight="600">complex state (rotation)</text>
<circle cx="345" cy="110" r="60" fill="none" stroke="#ccc"/>
<circle cx="345" cy="50" r="4" fill="#395C68"/>
<circle cx="405" cy="110" r="4" fill="#395C68"/>
<circle cx="345" cy="170" r="4" fill="#395C68"/>
<circle cx="285" cy="110" r="4" fill="#395C68"/>
<line x1="345" y1="110" x2="345" y2="50" stroke="#395C68"/>
<line x1="345" y1="110" x2="345" y2="170" stroke="#395C68" stroke-dasharray="3,3"/>
<text x="345" y="200" text-anchor="middle" class="small">a 180 degree rotation flips the state exactly</text>
</svg>
</div>

### 9c. Autoregressive decoding wastes almost the entire GPU (MIMO)

New idea to build from scratch: **arithmetic intensity**, the ratio of how much arithmetic a chip does to how many bytes it has to move to do it. If intensity is low, the chip sits idle waiting for data to arrive, no matter how fast its compute units are, this is called being **memory-bound**. Generating one token at a time is close to a worst case for this: you touch the entire state's worth of bytes to do a tiny amount of arithmetic. The paper measures roughly $2.5$ operations per byte during decoding, against an H100 GPU's capacity of roughly $295$ operations per byte, meaning well over $99\%$ of the chip's arithmetic capability sits unused during generation.

The fix: instead of one input stream feeding one state and one readout, run $R$ parallel input/output "lanes" through the *same shared state matrix* at once ($B_t$ becomes an $N\times R$ matrix instead of a vector, the input becomes $P \times R$). Same amount of state to move, but $R$ times the useful arithmetic done with it, multiplying arithmetic intensity by roughly $R$ almost for free. It's the same logistics trick as sending four people up the same staircase carrying their own boxes instead of one person making four trips: the expensive part (climbing the stairs, i.e. moving the state through memory) happens once, not four times.

<div class="diagram">
<svg viewBox="0 0 420 170" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Arithmetic intensity comparison bar chart">
<style>text{font-family:inherit;font-size:12px;fill:#333}</style>
<line x1="60" y1="140" x2="400" y2="140" stroke="#333"/>
<rect x="90" y="136" width="30" height="4" fill="#395C68"/>
<text x="105" y="155" text-anchor="middle" class="small">SISO decode</text>
<text x="105" y="128" text-anchor="middle" class="small">~2.5</text>
<rect x="200" y="118" width="30" height="22" fill="#395C68"/>
<text x="215" y="155" text-anchor="middle" class="small">MIMO, R=4</text>
<text x="215" y="110" text-anchor="middle" class="small">~10</text>
<rect x="320" y="20" width="30" height="120" fill="rgba(57,92,104,0.35)"/>
<text x="335" y="155" text-anchor="middle" class="small">H100 capacity</text>
<text x="335" y="14" text-anchor="middle" class="small">~295</text>
<text x="210" y="12" text-anchor="middle" font-weight="600">operations per byte moved (log-ish scale, illustrative)</text>
</svg>
</div>

Empirically, Mamba-3 with $R=4$ at state size 64 matches Mamba-2's perplexity at state size 128, at roughly half the decode latency. Same quality, half the memory traffic, because more useful work gets squeezed out of every byte moved.

### 9d. Two small but telling housekeeping wins

Mamba-3 normalizes $B_t$ and $C_t$ (called **BCNorm**, the same idea as QK-norm in modern transformers), which turns out to remove the need for the extra stabilizing normalization Mamba-2 required after gating. It also adds a learnable, per-channel bias to $B_t,C_t$ after normalization, which the authors find lets the recurrence itself reproduce a convolution-like smoothing effect. Remember the short external causal conv1d from Act 7, the one we treated as a basic, seemingly essential ingredient? With the trapezoidal update and these biases in place, it becomes optional. The very block diagram you built earlier gets one piece deleted.

<div class="flow">
  <div class="box">S4<br><span class="small">HiPPO init + convolution/FFT training</span></div><span class="arrow">→</span>
  <div class="box">Mamba<br><span class="small">selection + hardware-aware scan</span></div><span class="arrow">→</span>
  <div class="box">Mamba-2<br><span class="small">SSD duality + chunked matmul</span></div><span class="arrow">→</span>
  <div class="box">Mamba-3<br><span class="small">trapezoidal + rotation + MIMO</span></div>
</div>

<div class="exercise">
<h4>Exercise 7: the capstone, in your own words, no formulas required</h4>

Two questions, answer both before opening the answer.

1. Explain why Mamba-3's trapezoidal update is doing exactly the same *kind* of thing you did in Exercise 2, just averaging two things instead of using one.
2. Using the parity example, explain why the purely real scalar state you solved by hand in Exercise 1 could never solve parity, no matter what value of $a$ you picked.

<details>
<summary>Hint</summary>
For (1): in Exercise 2 you approximated an integral (an area) using only one endpoint's height. What did the trapezoidal rule change about which endpoints get used? For (2): think about what a real number multiplied repeatedly by a fixed factor $e^{a\Delta}$ can ever do to its own sign, versus what "flip on demand" requires.
</details>

<details>
<summary>Answer</summary>

1. Exercise 2's discretization approximated the deposit's contribution to the state using only the value at the start of the interval, a rectangle-shaped area estimate. That's first-order accurate. The trapezoidal update instead blends the contribution from the *previous* token and the *current* token (with a learned weight $\lambda_t$ deciding the mix), exactly matching the textbook trapezoidal rule's use of both endpoints of an interval to estimate the area under a curve. Same numerical-methods idea, just now applied per-token with a content-dependent mixing weight instead of a fixed 50/50 split.

2. A real scalar state updated by $h_t = e^{a\Delta}h_{t-1} + \dots$ can only ever shrink toward zero (if $e^{a\Delta}<1$) or grow (if $e^{a\Delta}>1$), continuously, in one direction. Parity needs the state to jump discretely between two distinguishable values every time a 1 appears, a "flip," and a smooth exponential decay or growth curve has no mechanism to reverse itself on command, it's monotone by construction, whatever direction it's headed, it keeps heading that way. A rotation, by contrast, can move a state a full 180 degrees around a circle and land exactly back where a "flip" needs it to land, then rotate 180 degrees again to flip back, an operation with no real-number equivalent.
</details>
</div>

## Closing: what's actually solid now, and what to bring back

Before moving on, answer these for the whole arc, without scrolling back up: what does selection say, why should it be true that a hardware-aware scan is even necessary, what's the minimal mechanism behind the SSD duality, and could you reproduce Exercise 6 from memory? Whatever doesn't come back cleanly is not a failure, it's exactly what a revisit queue is for, tell me which specific piece didn't stick and it goes on the list with a real due date rather than a vague "review this again."

Natural next steps from here, whenever you want to pick one: implement a tiny selective scan in actual code and run it on a real GPU tensor to see the timings for yourself; go one layer deeper into the HiPPO derivation we skipped in Act 4, if state-tracking theory becomes load-bearing for something you're reading; or work through the full SSD paper's proof of the duality in Act 8 rather than just the worked numeric example.
