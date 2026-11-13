/* The catalogue of lessons shown on the home page.
   To add a lesson: build it in its own folder (e.g. ./newpaper_lessons/), then append one entry below.
   Fields:
     id        folder name of the lesson (its index.html is the entry point)
     title     short name of the method / paper topic
     subtitle  one line saying what the paper does
     field     subject area, used for the filter chips
     paper     { authors, title, venue, year, url }
     code      optional link to the official code
     cover     image path relative to the site root
     stats     { chapters, widgets, quizzes }
     store     localStorage prefix the book uses ("<prefix>done", "<prefix>last"), for progress on the home page
     added     date the lesson was published (YYYY-MM-DD)
   A collection (one card that opens a list of books) has instead:
     type: 'collection', id (its folder), title, subtitle, field, cover, added,
     data      name of the global its catalogue defines (index.html loads <id>/collection.js), e.g. 'AI49'
   Its paper count, ready count and reading progress are computed from that catalogue. */
window.LESSONS = [
  {
    type: 'collection',
    id: 'ai49_collection',
    title: 'Modern AI in 49 Papers',
    subtitle: 'From word vectors to reasoning models: 49 landmark papers, ordered from easiest to hardest, each with its own interactive web book.',
    field: 'Language models',
    cover: 'ai49_collection/cover.svg',
    data: 'AI49',
    added: '2026-11-14'
  },
  {
    id: 'alphafold_lessons',
    title: 'AlphaFold',
    subtitle: 'How a neural network predicts a protein’s 3-D structure from its amino-acid sequence with near-experimental accuracy.',
    field: 'Biology',
    paper: { authors: 'Jumper, Evans, Pritzel et al.', title: 'Highly accurate protein structure prediction with AlphaFold', venue: 'Nature 596', year: 2021, url: 'https://doi.org/10.1038/s41586-021-03819-2' },
    code: 'https://github.com/deepmind/alphafold',
    cover: 'alphafold_lessons/assets/img/cover.jpg',
    stats: { chapters: 26, widgets: 38, quizzes: 52 },
    store: 'afbook:',
    added: '2026-11-13'
  },
  {
    id: 'ncm_lessons',
    title: 'Next-Concept Modeling',
    subtitle: 'Fast, parallel, coarse-to-fine text generation: write the gist first, then sharpen it level by level.',
    field: 'Language models',
    paper: { authors: 'Kumbong, Khare et al.', title: 'Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation', venue: 'arXiv:2610.14162', year: 2026, url: 'https://arxiv.org/abs/2610.14162' },
    code: 'https://github.com/HazyResearch/ncm',
    cover: 'ncm_lessons/assets/img/cover.jpg',
    stats: { chapters: 26, widgets: 46, quizzes: 50 },
    store: 'ncmbook:',
    added: '2026-10-23'
  }
];
