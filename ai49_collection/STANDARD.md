# Building a lesson of “Modern AI in 49 Papers”

Every lesson is a full Interactive Web Book, built to the same standard as the standalone NCM and AlphaFold books. This file covers what is specific to the collection. The full method is the **paper-to-web-book** skill ([github.com/Mrb-Rafi/paper-to-tutorial-skill](https://github.com/Mrb-Rafi/paper-to-tutorial-skill)). Read its `SKILL.md` and `references/` before starting, and use its `scripts/check_book.py`.

## The brief (unchanged from the first book)

Read the paper very carefully and clone the official code. Assume the reader knows absolutely nothing, not even the concepts the paper builds on. Explain everything with interactive content, using as many chapters as the paper needs. Explain every piece of math, and every small idea visually and interactively. A reader starting from zero must reach the highest level of understanding the paper presents. Follow the paper's and repository's own methods and examples, and invent simpler ones where they are too hard. Cite the paper and its code throughout. Call it an "Interactive Web Book".

## One lesson, step by step

1. **Scaffold** from the repository root:
   ```bash
   python ai49_collection/shared/new_lesson.py <slug> --mark XX --code-url https://github.com/… --code-name org/repo
   ```
   This creates `ai49_collection/<slug>/` from `shared/template/` and prints the path of the paper's PDF. The PDF is in `papers/`, which is local only.
2. **Read and inventory**: read the PDF end to end, appendix included. Clone the code into `ai49_collection/_src/<slug>/` (git-ignored). Fill in `_work/inventory.md`.
3. **Toy and outline**: fill in `_work/outline.md` (parts, chapters, widgets, coverage of every inventory item).
4. **Build**:
   - Write the chapters in `index.html` and this paper's widgets in `widgets.js` (split per part if it grows).
   - Reuse `shared/` wherever it fits: `ML` (math.js), `NN` (nn.js autograd with Linear, SGD and Adam), and the refresher widgets `r_vectors`, `r_neuron`, `r_descent`, `r_softmax` and `r_trainNet` (see them all in `shared/widgets/gallery.html`).
5. **Verify**:
   - Run `python <skill>/scripts/check_book.py ai49_collection/<slug>` until it reports no errors.
   - Screenshot every chapter (`<skill>/scripts/screenshot_chapters.py`) and look at each one, in both light and dark themes, and at phone width.
   - Audit every number against the inventory.
6. **Publish**:
   - Add `img/cover.jpg` if you want one.
   - In `collection.js`, set the lesson's `status: 'ready'`, its `code` link, and its `stats` (the counts that `check_book.py` prints).
   - Finish the lesson's `README.md`.
   - Commit and push. Check that `git status` shows no PDF, `_work/` or `_src/`.

## Collection rules

- **Paths**: a lesson uses `../shared/` for the engine (`core.js`, `book.css`), the libraries, the refreshers and the glossary, and `../collection.js` for navigation. Don't copy them into the lesson.
- **Storage**: the prefix is `ai49:<slug>:` (set by the template), and the theme key is shared, `ai49:theme`.
- **Never required, always optional**: the strip under the title links the lessons this one builds on, but the book's own foundations part must teach everything it needs.
- **"Limits and where this leads"**: the last chapter links forward to later lessons in the collection (`../<slug>/`), even before they are built (the pager shows "coming soon").
- **Glossary**: list the shared terms you use in `data-terms="…; …"`, and add new general terms to `shared/glossary.js` (alphabetical). Write paper-specific terms directly in the book.
- **Changing `shared/`**: lesson builds don't edit `shared/`. If a lesson needs a new shared helper, note it in the session's report and add it in a separate infrastructure change, so parallel lesson builds don't collide. Rerun `shared/lib/tests.html` (it must show ALL PASS) and check `shared/widgets/gallery.html` after any shared change.
- **Accent colour**: optional per book, via the commented `<style>` block in the template.

## Planned shared additions

- **Before Part B (lessons 6–11):** an attention module (scaled dot-product, masks, multi-head), a tiny Transformer forward pass, and refreshers `r_attention` and `r_nextToken`.
- **Later parts:** add only what two or more lessons need. Everything else stays in the lesson's own `widgets.js`.
