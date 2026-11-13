# Word2Vec: Interactive Web Book

Lesson 1 of [Modern AI in 49 Papers](../). It teaches Mikolov, Chen, Corrado and Dean (2013), *Efficient Estimation of Word Representations in Vector Space*, from absolute zero: what a vector and a cosine are, how words can be described by the company they keep, how networks learn, softmax and language models; then the paper's cost model (Eq. 1–3), the NNLM and RNNLM it compares against, the Huffman tree and hierarchical softmax; then CBOW and skip-gram (Eq. 4–5, Figure 1), trained live in the browser; and finally every result: the analogy method, the Semantic-Syntactic test set and Tables 1–8.

## What's inside

- 24 chapters in 4 parts (Words as numbers · Neural language models before word2vec · The two new models · Evidence), plus an equation cheat sheet, glossary and sources.
- 49 interactive widgets (5 shared refreshers + 44 of this book), including: a skip-gram model that trains in the hero and computes king − man + woman; a live CBOW/skip-gram trainer with a PCA map, loss curve and toy analogy accuracy; one-step CBOW and skip-gram updates number by number; a step-by-step Huffman tree builder (port of `CreateBinaryTree`); hierarchical softmax on a 4-word tree and inside the toy model (probabilities sum to 1); a tiny NNLM and a tiny RNN language model trained with the collection's autograd library; cost calculators for Eq. 1–5; the code's sigmoid table and a check of its one-line gradient against autograd; negative sampling and subsampling (labelled as the follow-up paper's); an analogy calculator, offset arrows, odd-one-out and averaged-offset experiments; toy sentence completion combining skip-gram with a bigram model; explorers for Tables 1–8.
- 50 self-check quiz questions.

## Sources

- Paper: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean. *Efficient Estimation of Word Representations in Vector Space*. ICLR 2013 (workshop). https://arxiv.org/abs/1301.3781 — numbers quoted from arXiv v3 (7 Sep 2013).
- Code: https://github.com/tmikolov/word2vec (Tomas Mikolov's copy of the original Google Code project, version 0.1c), Apache License 2.0. Referenced: `word2vec.c`, `word-analogy.c`, `compute-accuracy.c`, `questions-words.txt`.
- Follow-up paper (negative sampling, subsampling; Chapter 16 only): Mikolov, Sutskever, Chen, Corrado, Dean, *Distributed Representations of Words and Phrases and their Compositionality*, NIPS 2013, https://arxiv.org/abs/1310.4546.
- Figures: none reproduced (arXiv default licence). Figure 1 is redrawn as an original interactive SVG; tables are retyped as data with citations.

This is an independent educational companion, not an official publication of the authors.

## Toy vs. paper

The toy corpus (`toy.js`) is generated from a seeded grammar: about 2,400 short sentences, ~14,000 words, 120 distinct words, about people (gender × royal × young), five countries with capitals, languages and dishes, and four adjectives with comparative and superlative forms. On it the paper's methods run exactly as ported from `word2vec.c`: Huffman tree, hierarchical softmax, CBOW and skip-gram with the random reach R, linear learning-rate decay, analogy search with unit vectors and input exclusion. Differences: the corpus is synthetic and tiny, vectors have 2–40 dimensions, the exact sigmoid is used (with the code's MAX_EXP skipping rule), and skip-gram uses the paper's input/output direction (the code swaps them). Every number labelled "toy" is computed in the page; every number cited to a section or table is the paper's.

## Files

```
index.html             the book (uses ../shared/ for the engine, styles, ML/NN libraries, refreshers and glossary)
toy.js                 the toy corpus and the word2vec port (window.W2V)
widgets-found.js       helpers (window.W2VUI), hero and Part I widgets
widgets-method.js      Parts II–III widgets
widgets-evidence.js    Part IV widgets
_work/                 inventory, outline, screenshots (git-ignored)
```
