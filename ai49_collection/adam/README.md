# Adam: Interactive Web Book

Lesson 3 of [Modern AI in 49 Papers](../). It teaches Kingma & Ba's Adam optimizer from zero, starting with "what is a gradient?". By the end you have worked through every line of Algorithm 1 and every equation of the paper (Eq. 1–12). That includes the step-size bounds, bias correction, the convergence theorem with its appendix proof and the gap later found in that proof, AdaMax, and every experiment in §6.

## What's inside

- 32 chapters in 5 parts, plus an equation cheat sheet, a glossary and sources:
  - **Foundations**: parameters and loss, gradients, minibatch noise, condition number, momentum, running averages, moments, AdaGrad and RMSProp.
  - **The Adam algorithm** (§1–3): the big idea, Algorithm 1 line by line, hyper-parameters, the fast form, step-size bounds, trust region and SNR, scale invariance, bias correction (Eq. 1–4).
  - **Does Adam converge?** (§4, App. 10.1): regret (Eq. 5), convexity, Theorem 4.1 term by term, sparsity and Corollary 4.2, the proof.
  - **Relatives and extensions** (§5, §7): AdaGrad/RMSProp/natural gradient, AdaMax (Eq. 6–12, Algorithm 2), temporal averaging.
  - **The experiments** (§6, Fig. 1–4) and the limits, including AMSGrad and AdamW as later work.
- 52 interactive widgets. Highlights:
  - an optimizer race on 2-D loss surfaces (SGD, momentum, AdaGrad, RMSProp, Adam, AdaMax);
  - a line-by-line Algorithm 1 stepper that checks itself against the shared `NN.Adam`;
  - step-bound and SNR explorers, and bias-correction demos with β2 up to 0.9999;
  - Theorem 4.1 evaluated on real runs, and a stage-by-stage proof walk that counts where the proof's telescoping step fails;
  - Reddi et al.'s counterexample;
  - live toy versions of Fig. 1, 2a and 4.
- 94 self-check quiz questions.

## Sources

- Paper: Diederik P. Kingma, Jimmy Ba. *Adam: A Method for Stochastic Optimization*. ICLR 2015. https://arxiv.org/abs/1412.6980. Numbers are quoted from arXiv v9 (30 Jan 2017).
- Code: the authors released none. The book follows Algorithm 1 and 2 themselves, and cites PyTorch v2.5.0 `torch/optim/adam.py` and `adamax.py` (BSD-3-Clause) with line numbers. Every Adam in the page is the collection's shared `NN.Adam` (`../shared/lib/nn.js`); the other optimizers are in `opt.js`.
- Later work discussed: Reddi, Kale & Kumar, ICLR 2018 (AMSGrad); Loshchilov & Hutter, ICLR 2019 (AdamW).
- Figures: none reproduced (arXiv licence). Values marked ≈ were read off Fig. 1–4 by eye. All diagrams and charts are original and computed in the page.

## Toy vs. paper

The algorithms are exact. The problems are toys:
- 2-D surfaces;
- sparse online games for the theory;
- 8×8 "glyphs" in place of MNIST;
- a synthetic 3,000-word bag of words in place of IMDB;
- a 64-64-64-10 network in place of the 1000-unit MLPs;
- a 4×4 softmax regression in place of the VAE of Fig. 4.

Toy numbers are labelled "toy". CNNs appear only through the paper's numbers and a two-layer stand-in.

This is an independent educational companion, not an official publication of the authors.

## Files

- `index.html`: the book.
- `opt.js`: toy library (optimizers, surfaces, data, trainers, online games).
- `widgets-found.js`, `widgets-method.js`, `widgets-theory.js`, `widgets-evidence.js`: the widgets for each part.
- Engine, libraries, refreshers and glossary come from `../shared/`.
