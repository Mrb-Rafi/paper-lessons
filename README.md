# Paper Lessons

Interactive web books that teach research papers from zero: short chapters, a widget for every idea, every equation unpacked, and self-check quizzes. The home page (`index.html`) lists every lesson; each lesson lives in its own folder.

| Lesson | Folder | Paper |
|---|---|---|
| AlphaFold | [`alphafold_lessons/`](alphafold_lessons/) | Jumper et al., *Highly accurate protein structure prediction with AlphaFold*, Nature 2021 |
| Next-Concept Modeling | [`ncm_lessons/`](ncm_lessons/) | Kumbong, Khare et al., *Next-Concept Modeling*, arXiv:2610.14162, 2026 |
| **Modern AI in 49 Papers** (collection) | [`ai49_collection/`](ai49_collection/) | 49 landmark papers, 2013–2026, ordered from easiest to hardest, one web book each; built lesson by lesson (see [`ai49_collection/STANDARD.md`](ai49_collection/STANDARD.md)) |

A **collection** is one home-page card that opens its own list of books. Its books share one engine, maths library and set of refresher widgets in `<collection>/shared/`, and its catalogue is `<collection>/collection.js`.

Everything is static (no build step) and is served by GitHub Pages from the root of `main`. To preview locally, run `python -m http.server` here and open http://localhost:8000.

## Layout

```
index.html            home page (renders the catalogue)
home/lessons.js       the catalogue: one entry per lesson
home/home.css         home page styles
<name>_lessons/       one self-contained web book per paper
  index.html          the book
  assets/css/book.css, assets/js/core.js, assets/js/widgets-*.js, assets/img/
  README.md           what the book covers, its sources and licences
```

## Adding a lesson

1. Put the paper's PDF in a new folder `<name>_lessons/` (PDFs are git-ignored).
2. Build the book in that folder in the same format as the existing ones: copy `core.js` and `book.css`, give `core.js` its own namespace and localStorage prefix (e.g. `xyzbook:`), and keep the "← All lessons" link (`href="../"`) in the top bar and the footer.
3. Add a `cover.jpg` (about 800 px wide) under `assets/img/`, using only images you may reuse (check the paper's licence and credit it).
4. Append an entry to `home/lessons.js` (fields are documented at the top of that file) and a row to the table above.
5. Commit and push to `main`; GitHub Pages republishes automatically.

Each lesson is an independent educational companion, not an official publication of the paper's authors.
