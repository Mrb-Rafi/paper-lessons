/* Widgets for {{TITLE}} (lesson {{N}}). Shared helpers: Book (core.js), ML (lib/math.js), NN (lib/nn.js).
   Split into widgets-<part>.js files if this grows past ~500 lines, and list each in index.html. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN;

  // The hero: a 10-second demo of the paper's core idea (often an auto-playing version of a later widget).
  W.heroDemo = function (host) {
    Book.head(host, '{{The core idea, as a title}}', '{{One line: what to do and what to look for.}}');
    el('div', { class: 'note', text: 'Replace this with the paper’s core idea.' }, host);
  };
})();
