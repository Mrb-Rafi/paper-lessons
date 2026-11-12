# AlphaFold Interactive Web Book

An interactive web book that teaches **AlphaFold** — highly accurate protein structure prediction — starting from absolute zero (no biology or machine-learning background assumed) and ending at every equation behind the paper.

**Read it:** serve this folder (`python -m http.server`) and open `index.html`, or visit the GitHub Pages site for this repository.

## What's inside

- 26 chapters in five parts: Proteins → Learning machines → Evolution as evidence → The AlphaFold network → Evidence, plus an equation cheat sheet, glossary and sources.
- 38 interactive widgets: amino acids, Levinthal's paradox, an HP lattice folder with exhaustive search, the sequence/structure gap, a 3-D viewer linked to a distance map, a Ramachandran plot that builds backbones atom by atom (NeRF), softmax as a distogram, attention with bias and gating, an MSA viewer, coevolution (mutual information + APC) with live contact prediction versus MSA depth, folding from predicted contacts (distance geometry, with the mirror-image problem), superposition, RMSD / r.m.s.d.95 / GDT / lDDT / TM-score, the architecture, representation sizes, row/column attention, the outer product mean, the triangle inequality and the four triangle operations, rigid frames, invariant point attention (with a live invariance check), the residue gas, FAPE (with the mirror image), recycling trajectories, the loss recipe, a masked-MSA game, self-distillation, pLDDT, PAE/pTM, calibration fits, CASP14 results, run times and ablations.
- 52 self-check quiz questions.

## Sources

- Paper: Jumper, Evans, Pritzel, et al. *Highly accurate protein structure prediction with AlphaFold.* Nature 596, 583–589 (2021). [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2). Open access, CC BY 4.0.
- Code: [github.com/deepmind/alphafold](https://github.com/deepmind/alphafold).
- Figures in `assets/img/` are cropped from the paper and reused under its CC BY 4.0 licence.

This is an independent educational companion, not an official publication of the authors or DeepMind. Numbers quoted from the paper come from the main text and figures; values marked as read off a figure are approximate. Equations follow the paper's Supplementary Information (SI) algorithms.

## The toy protein

Widgets use a 50-residue three-helix bundle built from ideal geometry, and a toy "evolution" (a Potts model with a four-letter chemistry sampled by Monte Carlo) in which residues touching across the fold co-vary. Geometry, superposition, metrics, frames and FAPE are computed exactly; no widget runs the trained AlphaFold network, and illustrative animations are labelled as such.

## Structure

```
index.html                    the book
assets/css/book.css           styles (light and dark)
assets/js/protein.js          toy protein engine: vectors, frames, NeRF, superposition, metrics, FAPE,
                              toy evolution, coevolution, distance geometry, SVG 3-D viewer
assets/js/core.js             navigation, progress, quizzes, chart helpers
assets/js/widgets-*.js        interactive widgets per part
assets/img/                   figures cropped from the paper (CC BY 4.0)
```

No build step; it is a static site. Math is rendered with KaTeX from a CDN. Built in the same format as the companion NCM Interactive Web Book.
