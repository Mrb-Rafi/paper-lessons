# Batch Normalization: Interactive Web Book

Lesson 4 of [Modern AI in 49 Papers](../). It teaches Ioffe & Szegedy's Batch Normalization from zero, starting with "what is a neuron?". By the end you have worked through:

- every line of Algorithm 1 (the BN transform) and Algorithm 2 (training and inference);
- all six chain-rule equations for backpropagating through BN;
- the unbiased population variance m/(m−1);
- BN for convolutional layers;
- the scale-invariance and Jacobian arguments for higher learning rates;
- the regularization effect;
- every experiment in the paper: MNIST (Fig. 1) and the ImageNet Inception results (Fig. 2–4).

## What's inside

- 34 chapters in 5 parts, plus an equation cheat sheet, a glossary and sources:
  - **Foundations**: neurons and layers, minibatch SGD, backpropagation, learning rates, mean and variance, histograms and percentiles, standardizing, whitening, sigmoid saturation.
  - **Internal covariate shift** (§1–2): covariate shift and sub-networks, internal covariate shift live, the big idea, the §2 bias blow-up and the cost of whitening.
  - **The Batch Normalizing Transform** (§3): the two simplifications, Algorithm 1, γ and β, batch-mates, backpropagation through BN (B1–B6), and the gradient as a projection.
  - **Batch-normalized networks** (§3.1–3.3, §4.2.1): inference with population statistics, Algorithm 2 and folding, batch size, where BN goes, convolutional BN, scale invariance, Jacobian singular values, BN as a regularizer.
  - **The experiments** (§4–5): MNIST activations over time (Fig. 1), Inception on ImageNet, the seven changes of §4.2.1, the race to 72.2% (Fig. 2, 3), sigmoid networks, the ensemble record (Fig. 4), limits and later work.
- 51 interactive widgets. Highlights:
  - Fig. 1 rebuilt live: accuracy and {15, 50, 85}th percentiles of a sigmoid input, with and without BN;
  - an Algorithm 1 stepper that checks itself against the shared `NN.batchNorm`;
  - a B1–B6 gradient tracer checked against finite differences and autograd;
  - training vs inference mode on a trained network, and Algorithm 2 vs PyTorch-style moving averages;
  - per-feature-map convolutional BN, and scale invariance checked by autograd;
  - a toy learning-rate race (×1, ×5, ×30, with and without BN) and a deep sigmoid race;
  - explorers for Fig. 2 (redrawn ≈), Fig. 3 and Fig. 4.
- 98 self-check quiz questions.

## Sources

- Paper: Sergey Ioffe, Christian Szegedy. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML 2015 (PMLR 37). https://arxiv.org/abs/1502.03167. Numbers are quoted from the ICML proceedings version.
- Code: the authors released none. The book follows Algorithms 1 and 2 themselves, and cites PyTorch v2.5.0 (BSD-3-Clause) with line numbers:
  - `torch/nn/modules/batchnorm.py`;
  - `torch/nn/functional.py` (`batch_norm`);
  - `aten/src/ATen/native/Normalization.cpp` (forward, running statistics, backward).

  Every BN in the page is the collection's shared `NN.batchNorm` (`../shared/lib/nn.js`).
- Later work discussed: Layer Normalization (Ba, Kiros & Hinton, 2016), Group Normalization (Wu & He, 2018), and Santurkar et al., "How Does Batch Normalization Help Optimization?" (NeurIPS 2018).
- Figures: none reproduced. Fig. 1 is a live toy experiment, Fig. 2 is redrawn from values read by eye (marked ≈), and Fig. 3 and 4 are quoted as tables. All diagrams and charts are original and computed in the page.

## Toy vs. paper

The method is exact:
- Algorithm 1 and 2;
- B1–B6;
- the unbiased variance;
- folding;
- convolutional BN;
- scale invariance;
- the §2 bias blow-up.

The data and networks are toys:
- 8×8 binary "glyphs" (pixel digits from a 5×7 font, with noise) stand in for 28×28 MNIST;
- 3×32 sigmoid networks trained for 3,000 steps stand in for 3×100 trained for 50,000;
- an 8-layer, 32-unit network stands in for the 13.6M-parameter Inception in the learning-rate race.

ImageNet results appear only as the paper's own numbers. Toy numbers are labelled "toy". In the toy, the network without BN survives a 5× learning rate (Inception did not), and the regularization gain is small. The book says so where it happens.

This is an independent educational companion, not an official publication of the authors.

## Files

- `index.html`: the book.
- `bn.js`: toy library (glyph data, MLPs with optional BN on the shared autograd, running statistics, Algorithm 2, time-sliced trainer, charts).
- `widgets-found.js`, `widgets-problem.js`, `widgets-transform.js`, `widgets-network.js`, `widgets-evidence.js`: the widgets for each part.
- Engine, libraries, refreshers and glossary come from `../shared/`.
