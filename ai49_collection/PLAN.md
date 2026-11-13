# Modern AI in 49 Papers — build plan

A collection of 49 interactive web books, one per paper, ordered from easiest to hardest.
On the home page the collection shows up as one card, next to the two existing books (AlphaFold, NCM). Clicking that card opens the collection page: the 49 lessons in learning order, each linking to its own book.

Status (2026-11-14): **Phase 1 (infrastructure) is done**: shared engine, maths/autograd library (tests pass), refresher widgets, glossary, lesson template and scaffold script, collection page and home-page card. **Phase 2 (the 49 lessons) is next**, starting with Part A; see `STANDARD.md` for how each lesson is built. (Working name and folder name: “Modern AI in 49 Papers”, `ai49_collection/`; easy to rename before Phase 1.)

---

## 0. Decisions to confirm before building

| # | Question | Default in this plan |
|---|---|---|
| D1 | What goes to GitHub? | **Decided (2026-11-13): only the PDFs stay local.** The lessons, the collection page and the home-page card are committed and pushed like the other two books, so the live site shows 3 entries. PDFs are already ignored by `*.pdf`; `_src/` (cloned repositories) is ignored as well. |
| D2 | Collection name | “Modern AI in 49 Papers” |
| D3 | Should the two existing books move onto the new shared engine? | **No.** They stay exactly as they are. |

---

## 1. The quality bar (applies to every one of the 49 books)

Your original instructions for the first book (NCM), applied paper by paper:

1. **Read the paper very carefully**, including the appendix, using the final published version in `papers/` (the arXiv version where that's all we have). **Clone the official code repository** when one exists. Clones go into a git-ignored `_src/` folder, and the code is read, not run.
2. **Assume the reader knows absolutely nothing**, not even the concepts the paper is built on. Each book stands on its own. Collection order lets a reader skip a refresher, but no book *requires* it.
3. **Explain everything with interactive content.** Use as many modules (chapters) as the paper needs, never fewer to save time and never padding.
4. Call it an **“Interactive Web Book”**, never a tutorial.
5. **Explain every piece of math under the idea**, symbol by symbol. Show every small idea visually, and interactively where that helps.
6. A reader starting from zero must be able to reach **the highest level of understanding the paper presents.**
7. **Follow the paper’s and repository’s own methods and examples.** Where they are too complex for a beginner, build simpler new ones and label them as toys.
8. **Mention the paper and its code** whenever an example, method or figure comes from them, so the book stays rooted in the actual evidence.

Format refinements we settled on for the AlphaFold book (now the house standard):

- Chapters grouped into parts; callouts of five kinds (intuition / math / paper / code / warn).
- Widgets that **compute real things** on a labelled toy, never animations that only look right.
- Quizzes in every chapter, plus an equation cheat sheet, a glossary and a sources chapter.
- **Numbers only from the paper**, with approximate values labelled as approximate.
- Figures only where the licence allows reuse, with credit. Otherwise, redraw them as our own SVG.
- A “← All lessons” link (here it goes to the collection page), progress saved in the reader's own browser, light and dark themes, readable on a phone.

### Definition of done (checklist for every book)

- [ ] Every equation in the paper's main text appears in the book, is unpacked, and is in the cheat sheet
- [ ] Every main-text figure or table is either reproduced under licence, redrawn, or explored with a widget
- [ ] Every headline number has been checked against the paper (page or table cited)
- [ ] At least one widget per chapter (except front and back matter), and each one computes something
- [ ] At least 2 quiz questions per chapter, each with an explanation for every answer
- [ ] A “Before you start” refresher section, so the book works from zero
- [ ] A “Where this leads” chapter linking to later lessons in the collection
- [ ] No console errors and no KaTeX errors; checked in light and dark themes and at 375 px width
- [ ] Each chapter screenshotted in headless Chrome and reviewed
- [ ] The book's README lists the paper, the code, the figure licences, and which numbers come from the paper versus a toy

---

## 2. Learning order: easy to advanced

Difficulty runs from ★ (needs only the refreshers) to ★★★★★ (heavy maths or systems work). “Builds on” lists earlier lessons that help but are never required. Each lesson's slug is also its folder name. The same order is in `papers.csv` (columns `learning_order`, `part`, `difficulty`, `slug`, `builds_on`).

### Part A · Learning machines

| # | Lesson | ★ | Builds on | Signature widgets (the toys that carry the book) |
|---|---|---|---|---|
| 1 | Word2Vec (2013) | ★ | — | Train skip-gram on a toy corpus live; drag words in 2-D; the analogy calculator king − man + woman |
| 2 | Dropout (2014) | ★ | 1 | A small net that overfits, then dropout switched on with a slider; thinned-network ensemble viewer |
| 3 | Adam (2014) | ★★ | 1 | Optimizer race on 2-D loss surfaces (SGD / momentum / RMSProp / Adam); bias-correction explorer |
| 4 | Batch Normalization (2015) | ★★ | 2, 3 | Live activation histograms with BN on and off; train vs. inference statistics |
| 5 | ResNet (2015) | ★★ | 4 | Gradient flow through plain vs. residual stacks of depth N; a degradation-problem demo |

### Part B · From sequences to Transformers

| # | Lesson | ★ | Builds on | Signature widgets |
|---|---|---|---|---|
| 6 | Seq2Seq (2014) | ★★ | 1, 3 | An RNN unrolled step by step; encoder-to-bottleneck-to-decoder on a toy reversal task (the paper's reversal trick) |
| 7 | Bahdanau attention (2014) | ★★ | 6 | Alignment heatmap for toy translation; the fixed-vector bottleneck vs. attention as sentence length grows |
| 8 | Transformer (2017) | ★★★ | 5, 7 | Q/K/V calculator; multi-head attention viewer; positional encodings; masks; a tiny Transformer forward pass in the browser |
| 9 | BERT (2018) | ★★★ | 8 | Masked-word filling on a toy model; the 80/10/10 masking rule; [CLS] fine-tuning heads |
| 10 | GPT-3 (2020) | ★★ | 8 | Zero/one/few-shot prompt builder; in-context learning curves from the paper's figures |
| 11 | T5 (2019) | ★★★ | 9, 10 | Text-to-text task formatter; span-corruption generator; the paper's ablation-table explorer |

### Part C · Prompting and reasoning

| # | Lesson | ★ | Builds on | Signature widgets |
|---|---|---|---|---|
| 12 | Chain-of-Thought (2022) | ★ | 10 | Direct vs. step-by-step on toy arithmetic; accuracy vs. model scale from the paper |
| 13 | Self-Consistency (2022) | ★ | 12 | Sample N reasoning paths, then a majority-vote simulator showing accuracy vs. N |
| 14 | Lost in the Middle (2023) | ★★ | 10 | Move the gold document through the context; the U-shaped accuracy curve |
| 15 | ReAct (2022) | ★★ | 12 | Step-through Thought → Action → Observation trace over a toy wiki |
| 16 | Tree of Thoughts (2023) | ★★ | 13 | Game of 24 solved by BFS/DFS over thoughts with an evaluator |
| 17 | Reflexion (2023) | ★★ | 15 | An agent loop with a self-written memory of its failures; trial-by-trial success |
| 18 | Toolformer (2023) | ★★★ | 15 | Self-labelling API calls, then the loss-reduction filter that decides which calls to keep |
| 19 | SlopShape (2026) | ★★ | 10 | Structural-feature extractor on example posts; a classifier on the paper's 203-feature design |

### Part D · Search and retrieval

| # | Lesson | ★ | Builds on | Signature widgets |
|---|---|---|---|---|
| 20 | Sentence-BERT (2019) | ★★ | 1, 9 | Siamese encoder; cosine-similarity search; cross-encoder vs. bi-encoder cost calculator |
| 21 | Dense Passage Retrieval (2020) | ★★★ | 20 | BM25 vs. dense retrieval on a toy corpus; in-batch negatives |
| 22 | RAG (2020) | ★★★ | 11, 21 | Retrieve-then-generate pipeline; RAG-Sequence vs. RAG-Token marginalisation |
| 23 | HyDE (2022) | ★★ | 21 | Write a fake answer, embed it, search with it, then compare with querying directly |
| 24 | ColBERT (2020) | ★★★ | 21 | Token-level MaxSim matrix; late-interaction scoring vs. a single vector |

### Part E · Seeing, hearing, imagining

| # | Lesson | ★ | Builds on | Signature widgets |
|---|---|---|---|---|
| 25 | Vision Transformer (2020) | ★★★ | 8 | Cut an image into 16×16 patches; patch embeddings; attention maps over the image |
| 26 | CLIP (2021) | ★★★ | 20, 25 | Contrastive image–text matrix; zero-shot classifier from prompts |
| 27 | Whisper (2022) | ★★★ | 11 | Waveform to log-Mel spectrogram; the multitask token format |
| 28 | Flamingo (2022) | ★★★★ | 10, 26 | Perceiver resampler; gated cross-attention with the tanh gate; interleaved image-text prompts |
| 29 | DDPM (2020) | ★★★★★ | 5 | Forward noising slider; a reverse denoising process on a 2-D toy distribution; the ELBO unpacked |
| 30 | Latent Diffusion (2021) | ★★★★ | 26, 29 | Pixel space vs. latent space cost; an autoencoder toy; cross-attention conditioning; classifier-free guidance |

### Part F · Scale

| # | Lesson | ★ | Builds on | Signature widgets |
|---|---|---|---|---|
| 31 | Scaling Laws (2020) | ★★★ | 10 | Fit power laws on log-log plots; L(N), L(D), L(C) explorers from the paper's fits |
| 32 | Chinchilla (2022) | ★★★ | 31 | Compute-optimal N and D calculator; why about 20 tokens per parameter |
| 33 | Switch Transformer (2021) | ★★★★ | 11, 31 | A top-1 router; capacity factor and dropped tokens; the load-balancing loss |
| 34 | Mixtral (2024) | ★★★ | 33 | Top-2 of 8 routing; active vs. total parameters calculator |

### Part G · Lighter and faster models

| # | Lesson | ★ | Builds on | Signature widgets |
|---|---|---|---|---|
| 35 | RoFormer / RoPE (2021) | ★★★ | 8 | Rotate query and key vectors in 2-D pairs; relative position falls out of the dot product |
| 36 | GQA (2023) | ★★★ | 8 | Grouping slider from multi-head (MHA) to grouped (GQA) to multi-query (MQA); KV-cache size calculator |
| 37 | LoRA (2021) | ★★★ | 8 | Low-rank B·A update with a rank slider; parameter-count calculator |
| 38 | GPTQ (2022) | ★★★★ | 8 | Round-to-nearest vs. GPTQ error compensation on a toy layer; bits vs. error |
| 39 | QLoRA (2023) | ★★★★ | 37, 38 | NF4 quantile bins; double quantization; memory budget calculator |
| 40 | Speculative Decoding (2023) | ★★★★ | 10 | Draft model and verifier; acceptance-rule simulator that provably keeps the output distribution; speed-up vs. acceptance rate α |
| 41 | PagedAttention / vLLM (2023) | ★★★★ | 36 | KV-cache memory as pages; fragmentation vs. paging; block-table viewer |
| 42 | FlashAttention (2022) | ★★★★★ | 8 | GPU memory hierarchy; tiling with online softmax, step by step; an IO-count calculator |

### Part H · Alignment and reinforcement learning

| # | Lesson | ★ | Builds on | Signature widgets |
|---|---|---|---|---|
| 43 | PPO (2017) | ★★★★★ | 3 | Policy-gradient bandit; the clipped objective plotted against the probability ratio; advantage estimation (GAE) |
| 44 | InstructGPT (2022) | ★★★★ | 10, 43 | The three-stage pipeline: SFT → reward model (Bradley–Terry) → PPO with KL penalty; the paper's human-preference results |
| 45 | Constitutional AI (2022) | ★★★ | 44 | Critique → revise loop with written principles; AI preference labels (RLAIF) |
| 46 | DPO (2023) | ★★★★ | 44 | Derivation from the RLHF objective to the DPO loss, step by step; implicit reward explorer; β slider |
| 47 | Let’s Verify Step by Step (2023) | ★★★★ | 12, 44 | Process reward vs. outcome reward on toy solutions; best-of-N reranking curves |
| 48 | DeepSeek-R1 (2025) | ★★★★★ | 43, 47 | GRPO group-relative advantage; rule-based rewards; the growing response-length curve from the paper |

### Part I · Beyond attention

| # | Lesson | ★ | Builds on | Signature widgets |
|---|---|---|---|---|
| 49 | Mamba (2023) | ★★★★★ | 8, 42 | State-space recurrence; discretization; selective (input-dependent) parameters; scan vs. attention cost |

Why this order: each lesson's main prerequisite comes earlier. Ideas that are simple once you know Transformers (prompting, retrieval) come before maths-heavy ones (diffusion, quantization, RL). The hardest maths (DDPM, FlashAttention, PPO, R1, Mamba) comes last in its part. Publication year is ignored on purpose.

Rough size: 12–26 chapters per book depending on the paper, about 850 chapters across the collection. Short papers (Self-Consistency, HyDE, GQA) get short books. They are not padded to NCM size.

---

## 3. Project structure (modular)

The two existing books each have their own copy of `core.js` and `book.css`. That doesn't scale to 49, so the collection gets **one shared engine** plus **thin per-lesson folders**.

```
paper_lessons/
  index.html                 home page: learns to render a "collection" card
  home/lessons.js            the 2 books + one entry of type 'collection'
  ai49_collection/           committed and pushed (only PDFs and _src/ are ignored)
    index.html               collection page: 9 parts, 49 cards, progress, "next up"
    collection.js            THE catalogue: order, part, difficulty, builds-on, status, stats, paper/code links
    PLAN.md                  this file
    STANDARD.md              the quality bar (§1) and lesson template, for every build session
    papers/                  the 49 PDFs (moved here from paper_lessons/papers/; local only)
    _src/                    cloned official repositories (read only; local only)
    shared/
      core.js                one engine: namespace PL, book id from <body data-book>, store prefix "ai49:<slug>:"
      book.css               one stylesheet; per-book accent via a CSS variable
      lib/                   tested maths used by many books: vectors/matrices, a tiny autograd,
                             softmax/sampling, toy tokenizer, attention, a mini Transformer, plotting
      widgets/               refresher widgets reused in every "Before you start"
                             (gradient descent, softmax, embeddings, attention, next-token sampling, ...)
      glossary.js            one glossary; each book shows the terms it uses
      template/              starter lesson folder (index.html skeleton with parts, cheat sheet, glossary, sources)
    <slug>/                  one folder per lesson, e.g. word2vec/, transformer/
      index.html             the book (chapters only; engine and styles come from ../shared)
      widgets.js             widgets specific to this paper
      img/                   cover.jpg (~800 px) + licensed or redrawn figures
      README.md              sources, licences, what is from the paper vs. toy
```

How the pieces connect:

- **Home page:** an entry in `home/lessons.js` with `type: 'collection'` renders as a collection card (“49 lessons · 9 parts”, lessons ready so far, combined progress, “Open collection →”). The root `README.md` table gets a row for the collection.
- **Collection page:** reads `collection.js`. Lessons not built yet show as “Coming soon”, so the collection works from day one and fills in as lessons are finished.
- **Each book:** its top bar links “← Modern AI in 49 Papers” (`../`), plus “Previous / Next lesson” and its “Builds on” lessons.
- **Shared code changes** happen only in infrastructure work, never inside a lesson build. That keeps parallel lesson builds from colliding.

---

## 4. Phases

### Phase 1 — Infrastructure (one session)

1. Confirm D2–D3; move `papers/` into `ai49_collection/`; add `_src/` to `.gitignore` (PDFs are already ignored).
2. Build `shared/core.js` and `shared/book.css` by generalising the NCM/AlphaFold engine (nearly identical already; only the namespace and the paper-specific helpers differ).
3. Build `shared/lib/`, with each module checked against hand-computed values: tiny autograd, attention, mini Transformer, plotting.
4. Build the refresher widgets and the glossary skeleton.
5. Build the lesson template and `STANDARD.md`.
6. Build the collection page and `collection.js` with all 49 entries (status “planned”).
7. Teach the home page to render collection cards; add the collection entry to `home/lessons.js` and the README table.
8. Verify: home page shows 3 cards; the collection page shows 49 “Coming soon” cards in order; the template book renders cleanly in light, dark and mobile. Check `git status` shows no PDFs, then commit and push.

### Phase 2 — Lessons, one session per book, in learning order

For each paper:

1. Read the PDF end to end, plus its appendix; clone and read the official code (find and verify the repository first).
2. Write a concept map: every idea, equation, figure, table and headline number, each mapped to a chapter.
3. Plan the parts and chapters; choose toys (the paper's own examples where a beginner can follow them, simpler labelled toys where not).
4. Write the chapters with paper and code callouts, citing page, figure or table.
5. Build `widgets.js`; each widget computes its result live.
6. Add quizzes, the cheat sheet, glossary terms and the sources chapter; add `cover.jpg` and figures with licences checked.
7. Run the definition-of-done checklist (§1), including headless-Chrome screenshots of every chapter.
8. Mark the lesson “ready” in `collection.js` with its real chapter, widget and quiz counts.
9. Commit and push to `main` (no PDFs, no `_src/`); GitHub Pages republishes.

Batches follow the parts (A, B, …, I). Lessons inside a part only touch their own folders, so 2–3 can be built in parallel sessions once Phase 1 is done. Anything a lesson needs from `shared/` gets noted and added in a short infrastructure pass between batches.

### Phase 3 — Collection polish (one session)

Cross-links between lessons (the “Builds on” and “Where this leads” chapters), glossary consistency, a link check across all 49 books, final stats on the home and collection pages, and a full-collection screenshot pass.

---

## 5. Risks and how the plan handles them

- **Papers with heavy appendices** (T5, Scaling Laws, Chinchilla, DDPM, R1): main-text results are covered in full; appendix material is covered where it explains a main-text claim, and noted in the README.
- **No official code** (GPT-3, Chinchilla, InstructGPT, Chain-of-Thought, Self-Consistency, Adam, Batch Normalization, and others): the book says so and builds faithful toys from the paper's equations instead.
- **Figure licences** vary by paper: the default is redrawing in SVG; we copy a figure only when its licence is confirmed.
- **Browser limits:** toys must run in under ~1 s on a laptop. Large real models are never run; their numbers come from the paper and are labelled.
