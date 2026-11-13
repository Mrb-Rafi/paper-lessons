# Dropout: Interactive Web Book

Lesson 2 of [Modern AI in 49 Papers](../). It teaches Srivastava, Hinton, Krizhevsky, Sutskever and Salakhutdinov (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, from absolute zero: neurons and the forward pass, backpropagation and SGD, overfitting, weight penalties, model averaging, Bernoulli and Gaussian random variables and convolutional nets; then the dropout model in the paper's order (the idea and Fig. 1, the co-adaptation argument of §2, the forward pass of §4 and Fig. 3, test-time weight scaling and Fig. 2, why weight scaling approximates averaging all thinned networks, backpropagation through thinned networks, max-norm with large decaying learning rates and high momentum, pretraining and finetuning); then every result (Tables 1–9), the analyses of §7 (Fig. 7–11), dropout RBMs (§8, Fig. 12–13), marginalisation (§9), Gaussian dropout (§10, Table 10), the practical guide (App. A) and the limits.

## What's inside

- 31 chapters in 5 parts (Foundations · The dropout model · Results · Inside a dropout network · Beyond Bernoulli noise), plus an equation cheat sheet, glossary and sources.
- 64 interactive widgets (5 shared refreshers + 59 of this book), including: animated thinned networks (hero, Fig. 1); a standard and a dropout network trained live side by side on toy digits; a knockout test of co-adaptation; Fig. 3 as a clickable unit with its gates; the shared library's `NN.dropout` masks over a minibatch; the expectation behind weight scaling; the paper convention vs inverted dropout on a trained net; all 4,096 thinned networks of a tiny net enumerated (arithmetic mean, geometric mean, weight scaling) and the exact one-layer case; gradients through masks with the shared autograd; max-norm projection, momentum and deepnet's schedules; pretraining with an autoencoder then dropout finetuning; explorers for Tables 1–10 and read-off versions of Fig. 9–10; toy re-runs of Fig. 4, 7, 8, 9, 10, 11, 12, 13 and Tables 9 and 10; a brute-force RBM and CD-1 step by step; the §9 dropout-equals-ridge identity checked by Monte-Carlo; fast dropout's Gaussian approximation; Bernoulli vs Gaussian multipliers; the code-quality score; an Appendix A calculator.
- 76 self-check quiz questions.

## Sources

- Paper: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research 15(56):1929–1958, 2014. https://jmlr.org/papers/v15/srivastava14a.html — numbers quoted from this final JMLR version.
- Code: https://github.com/nitishsrivastava/deepnet (the first author's GPU library on cudamat and cuda-convnet, the CUDA libraries named in App. B), BSD-style licence, Copyright 2012–13 Nitish Srivastava, commit f4e4ff2. Referenced: `neuralnet.py`, `parameter.py`, `relu_layer.py`, `dbm.py`, `mc_avg.py`, `fastdropoutnet.py`, `examples/ff/model_dropout.pbtxt`, `examples/ff/train.pbtxt`.
- Figures: none reproduced (JMLR, authors' copyright, no CC licence). Figures 1–3 are redrawn as original interactive SVG; Figures 4 and 7–13 are explored with toy re-runs, and Figures 9–11 with values read off the plots (marked approximate); the photos of Figures 5–6 are described in words; tables are retyped as data with citations.

This is an independent educational companion, not an official publication of the authors.

## Toy vs. paper

The toy data (`toy.js`) are "pixel digits": ten hand-drawn 5×5 digit shapes on a 7×7 grid (49 inputs, 10 classes) with missing stroke pixels, specks and grey noise, generated from seeds (500 training and 1,000 test examples by default). Networks are trained with the collection's shared autograd (`NN.Linear`, `NN.SGD` with momentum, `NN.dropout`), plus max-norm, L1/L2 and deepnet's learning-rate and momentum schedules, exactly as §5.1 describes; the toy uses inverted dropout (×1/p while training), which §10 says is equivalent to the paper's weight scaling, and one widget shows the equivalence on a trained net. Exact in the page: forward passes, masks, gradients, weight scaling, enumeration of all thinned networks of a tiny net, Monte-Carlo averaging, the §9 identity, CD-1, code quality. Stand-ins: pixel digits for MNIST; conv nets, ImageNet, TIMIT, Reuters and the splicing data appear only through the paper's numbers. Five sweeps (toy Fig. 4, 9, 10 and Tables 9, 10) ship precomputed in `toy-results.js`, produced by the same code and seeds (`Toy.EXPERIMENTS`), and can be recomputed with "Re-run live". At this scale dropout's gains are real but smaller than the paper's, consistent with the paper's own Fig. 10. Every number labelled "toy" is computed by this code; every number cited to a section, figure or table is the paper's.

## Files

```
index.html             the book (uses ../shared/ for the engine, styles, ML/NN libraries, refreshers and glossary)
toy.js                 pixel digits, MLP trainer, autoencoder, RBM and the toy experiments (window.Toy)
toy-results.js         precomputed toy sweeps (window.TOY_RESULTS), reproducible with "Re-run live"
widgets-found.js       helpers (window.DOUI), hero and Part I widgets
widgets-method.js      Part II widgets
widgets-evidence.js    Parts III–V widgets
_work/                 inventory, outline, screenshots, precompute scripts (git-ignored)
```
