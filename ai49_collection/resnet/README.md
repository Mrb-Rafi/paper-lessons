# ResNet: Interactive Web Book

Lesson 5 of [Modern AI in 49 Papers](../), and the last lesson of Part A. It teaches *Deep Residual Learning for Image Recognition* (He, Zhang, Ren & Sun, CVPR 2016) from absolute zero. It starts with neurons, SGD, backpropagation, initialization and batch normalization, then builds convolutions (kernels, stride, padding, channels, 1×1 convolutions, pooling, receptive fields, FLOPs) from scratch. From there it rebuilds the paper: the degradation problem, residual learning F(x) = H(x) − x, the building block (Eq. 1), projection shortcuts and options A/B/C (Eq. 2), the bottleneck block, every architecture of Table 1, the training recipe, and every experiment (Tables 1–8, Figs. 1–7, and the arXiv appendix's Tables 9–14).

## What's inside

- 38 chapters in 6 parts, plus an equation cheat sheet, a glossary and sources:
  - Foundations
  - Seeing with convolutions
  - The degradation problem
  - Residual learning
  - The ResNet architectures
  - The evidence
- 49 interactive widgets, among them:
  - a live race between plain and residual networks (the hero, and degradation at 6/14/30/50 layers);
  - a check that rules out overfitting and vanishing gradients;
  - the constructed identity solution, and zero versus identity learning;
  - gradient flow through shortcuts, options A/B/C, the unrolled ensemble of paths, and highway gates;
  - a Fig. 3 explorer, a block builder, and a Table 1 explorer whose parameter totals match torchvision exactly;
  - augmentation, learning-rate schedules and 10-crop testing;
  - the official model files' layer names;
  - results explorers for Tables 2–8 and the appendix tables;
  - the CIFAR 6n+2 family and its parameter formula;
  - toy-CNN versions of Fig. 4/6 and Fig. 7;
  - IoU and mAP.
- 114 self-check quiz questions.

## Sources

- Paper: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016, pp. 770–778. https://arxiv.org/abs/1512.03385
- Code:
  - The official release is [KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks) (MIT licence, commit a7026cb). It contains the Caffe deploy prototxts for ResNet-50/101/152 and a README; there is no training code.
  - The book also cites [torchvision v0.20.0 `models/resnet.py`](https://github.com/pytorch/vision/blob/v0.20.0/torchvision/models/resnet.py) (BSD-3-Clause) with line numbers. It is the most widely used faithful re-implementation, and the book notes where it differs: option B at every depth, and the bottleneck stride on the 3×3 ("V1.5").
- Figures: all diagrams are original. The paper (arXiv non-exclusive licence) is not reproduced. Figs. 1, 4, 6 and 7 are redrawn from values read by eye (marked ≈) or rebuilt as live toy experiments.

This is an independent educational companion, not an official publication of the authors.

Main-text numbers come from the CVPR open-access version; Tables 1–8 are identical in arXiv v1. Appendix numbers (Tables 9–14) come from arXiv v1. Toy-model numbers come from small networks that run in the page (marked "toy").

## What is exact and what is a toy

- **Exact:**
  - Convolution, BN for feature maps, pooling and option-A shortcuts, all with their backward passes (checked against finite differences).
  - The parameter and FLOP arithmetic of Table 1. Parameter totals equal torchvision's `num_params` for all five depths. FLOPs come out at 1.81/3.66/3.86/7.57/11.28 G against the paper's 1.8/3.6/3.8/7.6/11.3.
  - The CIFAR 6n+2 parameter counts (0.27M … 19.4M, matching Table 6).
- **Toy:**
  - Fully-connected nets on a 5-arm spiral.
  - A miniature CIFAR-style CNN family: 8/20/32 layers, 6/12/24 filters on 8×8/4×4/2×2 maps, 2,000 noisy 8×8 digits, 800 SGD steps.
  - These reproduce degradation and its removal by shortcuts, not the paper's numbers.

## Files

- `index.html`: the book, assembled from `_work/parts/` by `_work/build.py` (`_work/` is not committed).
- `rn.js`: the toy library: conv autograd on the shared `NN`, data, the CNN and MLP, the trainer, Table 1 arithmetic, and the CNN experiments.
- `toy-results.js`: stored runs of the CNN experiments, which the "Re-run live" button reproduces.
- `ui.js`: drawing helpers.
- Widgets: `widgets-found.js`, `widgets-conv.js`, `widgets-degrade.js`, `widgets-residual.js`, `widgets-arch.js`, `widgets-evidence.js`.
