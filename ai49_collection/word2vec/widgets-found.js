/* Word2Vec Interactive Web Book — shared UI helpers (window.W2VUI), the hero demo and Part I (foundations).
   Uses Book (core.js), ML (lib/math.js), NN (lib/nn.js) and W2V (toy.js). */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, W2V = window.W2V;

  // ------------------------------------------------------------------ helpers shared by all widget files
  const U = window.W2VUI = {};
  // In this book, bar charts on a 0–1 scale get two-decimal ticks (0.25 rather than 0.3). Applies to this page only.
  const bars0 = Book.chart.bars;
  Book.chart.bars = function (host, data, o) {
    o = o || {};
    if (o.max === 1 && !o.tickFmt) o = Object.assign({}, o, { tickFmt: function (v) { return v.toFixed(2); } });
    return bars0(host, data, o);
  };
  const GROUP = {};
  W2V.people.forEach(function (p) { GROUP[p.w] = p.g === 'm' ? 'male' : 'female'; });
  W2V.places.forEach(function (p) { GROUP[p.country] = 'country'; GROUP[p.city] = 'city'; GROUP[p.lang] = 'language'; GROUP[p.food] = 'food'; });
  W2V.adjectives.forEach(function (a) { GROUP[a.base] = 'adjective'; GROUP[a.cmp] = 'comparative'; GROUP[a.sup] = 'superlative'; GROUP[a.animal] = 'animal'; });
  U.COLORS = { male: 'var(--blue)', female: 'var(--red)', country: 'var(--green)', city: 'var(--teal)', language: 'var(--purple)', food: '#b7791f',
    adjective: '#b7791f', comparative: '#d53f8c', superlative: 'var(--purple)', animal: 'var(--green)', other: 'var(--muted)' };
  U.group = function (w) { return GROUP[w] || 'other'; };
  U.color = function (w) { return U.COLORS[U.group(w)]; };
  U.SETS = {
    people: W2V.people.map(function (p) { return p.w; }),
    places: [].concat.apply([], W2V.places.map(function (p) { return [p.country, p.city, p.lang]; })),
    adjectives: [].concat.apply([], W2V.adjectives.map(function (a) { return [a.base, a.cmp, a.sup]; }))
  };
  U.SETS.all = U.SETS.people.concat(U.SETS.places, U.SETS.adjectives);
  U.corpus = function () { return W2V.corpusDefault(); };
  // the two cached reference models used by many widgets (skip-gram and CBOW, 20 dims, window 3, 6 epochs)
  U.model = function (cbow) { return W2V.trained({ cbow: !!cbow }); };
  U.select = function (parent, label, words, value, onchange) {
    return Book.select(parent, label, words.map(function (w) { return { value: w, label: w }; }), value, onchange);
  };
  // PCA to 2-D with the signs of the axes kept stable between redraws (state object remembers last axes)
  U.project = function (rows, state) {
    const p = ML.pca2(rows);
    if (state && state.comps) {
      [0, 1].forEach(function (k) {
        if (ML.dot(p.components[k], state.comps[k]) < 0) {
          p.components[k] = ML.scale(p.components[k], -1);
          p.points.forEach(function (pt) { pt[k] = -pt[k]; });
        }
      });
    }
    if (state) state.comps = p.components;
    return p;
  };
  // A labelled scatter plot. items: [{x, y, label, color, r}], arrows: [{a: index, b: index, color}]
  U.scatter = function (host, items, opts) {
    opts = opts || {};
    host.innerHTML = '';
    const Wd = opts.width || 520, Ht = opts.height || 340, pad = 34;
    const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + Ht, role: 'img', 'aria-label': opts.aria || 'Word vectors projected to two dimensions' }, host);
    S('rect', { x: 0.5, y: 0.5, width: Wd - 1, height: Ht - 1, rx: 10, fill: 'none', stroke: 'var(--line)' }, svg);
    if (!items.length) return svg;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    items.forEach(function (it) { x0 = Math.min(x0, it.x); x1 = Math.max(x1, it.x); y0 = Math.min(y0, it.y); y1 = Math.max(y1, it.y); });
    if (opts.square) { const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, h = Math.max(x1 - x0, (y1 - y0) * Wd / Ht) / 2 || 1; x0 = cx - h; x1 = cx + h; y0 = cy - h * Ht / Wd; y1 = cy + h * Ht / Wd; }
    const sx = function (x) { return pad + (x - x0) / ((x1 - x0) || 1) * (Wd - 2 * pad - 40); };
    const sy = function (y) { return Ht - pad - (y - y0) / ((y1 - y0) || 1) * (Ht - 2 * pad); };
    const defs = S('defs', null, svg);
    (opts.arrows || []).forEach(function (ar, k) {
      const id = 'ah' + Math.random().toString(36).slice(2, 8) + k;
      const mk = S('marker', { id: id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
      S('path', { d: 'M0,0 L10,5 L0,10 z', fill: ar.color || 'var(--muted)' }, mk);
      const A = items[ar.a], B = items[ar.b];
      S('line', { x1: sx(A.x), y1: sy(A.y), x2: sx(B.x), y2: sy(B.y), stroke: ar.color || 'var(--muted)', 'stroke-width': 2, 'marker-end': 'url(#' + id + ')', opacity: 0.8 }, svg);
    });
    items.forEach(function (it) {
      S('circle', { cx: sx(it.x), cy: sy(it.y), r: it.r || 4.5, fill: it.color || 'var(--accent)', stroke: 'var(--panel)', 'stroke-width': 1.2 }, svg);
      S('text', { x: sx(it.x) + 7, y: sy(it.y) + 4, 'font-size': it.bold ? 13.5 : 12, 'font-weight': it.bold ? 700 : 500, class: 'svg-text', fill: it.bold ? 'var(--ink)' : '', text: it.label }, svg);
    });
    return svg;
  };
  // Plot a list of words of a model in 2-D
  U.plotWords = function (host, model, words, state, opts) {
    const C = model.corpus, ws = words.filter(function (w) { return C.index[w] != null; });
    const rows = ws.map(function (w) { return model.vector(w); });
    const p = U.project(rows, state);
    const items = ws.map(function (w, i) { return { x: p.points[i][0], y: p.points[i][1], label: w, color: U.color(w), bold: opts && opts.bold && opts.bold.indexOf(w) >= 0 }; });
    const arrows = ((opts && opts.pairs) || []).map(function (pr) { return { a: ws.indexOf(pr[0]), b: ws.indexOf(pr[1]), color: pr[2] }; }).filter(function (a) { return a.a >= 0 && a.b >= 0; });
    U.scatter(host, items, Object.assign({ arrows: arrows }, opts || {}));
    return p;
  };
  U.legend = function (parent, groups) {
    const lg = el('div', { class: 'legend' }, parent);
    groups.forEach(function (g) { el('span', { html: '<i style="background:' + U.COLORS[g] + '"></i>' + g }, lg); });
    return lg;
  };
  U.pct = function (x) { return (100 * x).toFixed(0) + '%'; };
  // Run a function on chunks with setTimeout so long work does not freeze the page. Returns a cancel function.
  U.loop = function (stepFn, delay) {
    let alive = true;
    (function tick() { if (!alive) return; if (stepFn() !== false) setTimeout(tick, delay || 16); })();
    return function () { alive = false; };
  };

  // ------------------------------------------------------------------ HERO
  W.heroDemo = function (host) {
    Book.head(host, 'Train word vectors, then do arithmetic with them',
      'A skip-gram model reads a small toy corpus (about 2,400 sentences) and learns a 20-number vector per word. Watch the people words sort themselves out, then see what <b>king − man + woman</b> lands on.');
    const st = { seed: 1, cancel: null, model: null, view: {} };
    const ctl = el('div', { class: 'controls' }, host);
    const qs = [['man', 'king', 'woman'], ['france', 'paris', 'italy'], ['big', 'bigger', 'small'], ['boy', 'prince', 'girl']];
    let q = qs[0];
    Book.select(ctl, 'question', qs.map(function (x, i) { return { value: i, label: x[1] + ' − ' + x[0] + ' + ' + x[2] }; }), 0, function (v) { q = qs[+v]; answer(); });
    Book.button(ctl, 'Train again (new random start)', function () { st.seed++; run(); }, 'primary');
    const box = el('div', { class: 'grid2' }, host), map = el('div', null, box), right = el('div', null, box);
    const out = el('div', { class: 'out' }, right), ans = el('div', { class: 'out' }, right);
    U.legend(host, ['male', 'female']);
    function draw() {
      U.plotWords(map, st.model, U.SETS.people, st.view, { pairs: [['man', 'woman', 'var(--muted)'], ['king', 'queen', 'var(--muted)']], width: 420, height: 300 });
    }
    function answer() {
      const m = st.model, C = m.corpus, M = m.normalized();
      const r = W2V.analogy(M, C.words, C.index[q[0]], C.index[q[1]], C.index[q[2]], 3);
      ans.innerHTML = '<b>' + q[1] + ' − ' + q[0] + ' + ' + q[2] + '</b> is closest to:<br>' + r.top.map(function (t, i) {
        return (i ? '' : '→ ') + '<b>' + t.w + '</b> (cosine ' + Book.fmt(t.cos, 2) + ')'; }).join('<br>') +
        '<br><span class="note">Computed now from the toy vectors (inputs excluded, as in the paper §4). Toy result.</span>';
    }
    function run() {
      if (st.cancel) st.cancel();
      const C = U.corpus();
      st.model = new W2V.Model({ corpus: C, cbow: false, D: 20, window: 3, epochs: 6, seed: st.seed });
      st.view = {};
      st.cancel = U.loop(function () {
        const more = st.model.trainWords(1400);
        draw();
        out.innerHTML = 'words read: <b>' + st.model.wordsDone.toLocaleString() + '</b> of ' + st.model.totalWords.toLocaleString() +
          ' · learning rate α = ' + Book.fmt(st.model.alpha, 4) + '<br>average loss (this epoch): <b>' + Book.fmt(st.model.currentLoss(), 3) + '</b> (toy)';
        answer();
        return more;
      }, 30);
    }
    run();
  };

  // ------------------------------------------------------------------ ch-onehot
  W.oneHot = function (host) {
    Book.head(host, 'One-hot vectors: every word is equally far from every other',
      'Pick two words. Each is a vector of V numbers with a single 1 at the word’s position in the vocabulary. Compare them.');
    const C = U.corpus();
    const mini = ['the', 'king', 'queen', 'man', 'woman', 'paris', 'rome', 'dog', 'big', 'bigger'];
    const st = { a: 'king', b: 'queen' };
    const ctl = el('div', { class: 'controls' }, host);
    U.select(ctl, 'word A', mini, st.a, function (v) { st.a = v; draw(); });
    U.select(ctl, 'word B', mini, st.b, function (v) { st.b = v; draw(); });
    const pic = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      pic.innerHTML = '';
      const cw = 46, Wd = 90 + mini.length * cw, Ht = 128;
      const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + Ht, role: 'img', 'aria-label': 'One-hot vectors of two words' }, pic);
      mini.forEach(function (w, j) {
        S('text', { x: 90 + j * cw + cw / 2, y: 16, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: w }, svg);
      });
      [['A', st.a, 'var(--blue)'], ['B', st.b, 'var(--red)']].forEach(function (r, i) {
        const y = 28 + i * 44;
        S('text', { x: 8, y: y + 24, 'font-size': 13, 'font-weight': 700, class: 'svg-text', text: r[0] + ': ' + r[1] }, svg);
        mini.forEach(function (w, j) {
          const on = w === r[1];
          S('rect', { x: 90 + j * cw + 3, y: y, width: cw - 6, height: 36, rx: 6, fill: on ? r[2] : 'var(--bg-soft)', stroke: 'var(--line)' }, svg);
          S('text', { x: 90 + j * cw + cw / 2, y: y + 23, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700, fill: on ? '#fff' : 'var(--muted)', text: on ? '1' : '0' }, svg);
        });
      });
      const a = mini.map(function (w) { return w === st.a ? 1 : 0; }), b = mini.map(function (w) { return w === st.b ? 1 : 0; });
      const dot = ML.dot(a, b);
      out.innerHTML = 'dot product A·B = <b>' + dot + '</b> · cosine similarity = <b>' + Book.fmt(ML.cosine(a, b), 2) + '</b> · distance = <b>' + Book.fmt(Math.sqrt(ML.dist2(a, b)), 3) + '</b><br>' +
        (st.a === st.b ? 'Same word: the only way to get a non-zero dot product.' : 'Different words: dot product 0 and distance √2 ≈ 1.414, for <i>every</i> pair, whether “king/queen” or “king/bigger”.') +
        '<br><span class="note">This shows 10 words. The toy vocabulary has V = ' + C.V + ' words (“' + st.a + '” is word number ' + C.index[st.a] + ' when sorted by frequency), so each one-hot vector has ' + C.V + ' entries. The paper’s vocabulary has 1 million words (§4.2).</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-vectors
  W.featureWords = function (host) {
    Book.head(host, 'Hand-made word vectors with three features',
      'Here each word gets 3 numbers that we chose by hand: how royal, how female, how young. Pick a word to see its cosine similarity with all the others, or build your own vector with the sliders.');
    const F = { king: [1, 0, 0], queen: [1, 1, 0], prince: [1, 0, 1], princess: [1, 1, 1], man: [0, 0, 0], woman: [0, 1, 0], boy: [0, 0, 1], girl: [0, 1, 1] };
    // shift to −1/+1 so that "not royal" is a direction too
    Object.keys(F).forEach(function (k) { F[k] = F[k].map(function (v) { return v ? 1 : -1; }); });
    const words = Object.keys(F);
    const st = { mode: 'word', w: 'king', v: [0.8, 0.9, -0.7] };
    const ctl = el('div', { class: 'controls' }, host);
    U.select(ctl, 'compare', words.concat(['(my vector)']), st.w, function (v) { st.w = v; draw(); });
    const ctl2 = el('div', { class: 'controls' }, host);
    ['royal', 'female', 'young'].forEach(function (n, i) {
      Book.slider(ctl2, 'my vector: ' + n, -1, 1, 0.1, st.v[i], function (x) { st.v[i] = x; if (st.w === '(my vector)') draw(); }, function (x) { return x.toFixed(1); });
    });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const q = st.w === '(my vector)' ? st.v : F[st.w];
      const data = words.filter(function (w) { return w !== st.w; }).map(function (w) { return { label: w, value: ML.cosine(q, F[w]), color: U.color(w) }; });
      data.sort(function (a, b) { return b.value - a.value; });
      Book.chart.bars(chart, data.map(function (d) { return { label: d.label, value: (d.value + 1) / 2, note: Book.fmt(d.value, 2), color: d.color }; }),
        { max: 1, height: 220, ylabel: 'cosine (−1 … 1)', tickFmt: function (v) { return Book.fmt(2 * v - 1, 1); } });
      out.innerHTML = (st.w === '(my vector)' ? 'my vector' : st.w) + ' = (' + q.map(function (x) { return Book.fmt(x, 1); }).join(', ') + ') → nearest word: <b>' + data[0].label + '</b> (cosine ' + Book.fmt(data[0].value, 2) + ')' +
        '<br><span class="note">Hand-made vectors, not learned. Word2Vec learns vectors like these from raw text, with no one telling it which features exist.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-context
  W.contextWindow = function (host) {
    Book.head(host, 'A word and its context window',
      'Click any word of the sentence to make it the centre. The window takes up to C words on each side; each (centre, context) pair is one piece of evidence about what the centre word means.');
    const C = U.corpus();
    const sents = [C.raw[0], C.raw.find(function (s) { return s.indexOf('capital') >= 0; }), C.raw.find(function (s) { return s.indexOf('than') >= 0; }), C.raw.find(function (s) { return s.indexOf('said') >= 0; })].filter(Boolean);
    const st = { s: 3, pos: 1, C: 2 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'sentence', sents.map(function (s, i) { return { value: i, label: s.join(' ') }; }), st.s, function (v) { st.s = +v; st.pos = Math.min(st.pos, sents[st.s].length - 1); draw(); });
    Book.slider(ctl, 'window C', 1, 5, 1, st.C, function (v) { st.C = v; draw(); });
    const row = el('div', { class: 'tokrow' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const s = sents[st.s];
      row.innerHTML = '';
      s.forEach(function (w, i) {
        const d = Math.abs(i - st.pos);
        const cls = i === st.pos ? 'chip hl' : (d <= st.C ? 'chip ok' : 'chip mask');
        el('span', { class: cls, text: w, style: 'cursor:pointer', onclick: function () { st.pos = i; draw(); } }, row);
      });
      const pairs = [];
      for (let i = Math.max(0, st.pos - st.C); i <= Math.min(s.length - 1, st.pos + st.C); i++) if (i !== st.pos) pairs.push('(' + s[st.pos] + ', ' + s[i] + ')');
      out.innerHTML = 'centre word: <b>' + s[st.pos] + '</b> · ' + pairs.length + ' context pairs:<br>' + pairs.join('  ') +
        '<br><span class="note">Toy corpus sentence. Like word2vec.c, windows stop at the sentence boundary.</span>';
    }
    draw();
  };

  W.coocMatrix = function (host) {
    Book.head(host, 'Counting the company words keep',
      'Each row counts how often a word appears within 2 positions of each context word, over the whole toy corpus. Words used in similar places get similar rows. Choose two rows to compare.');
    const C = U.corpus(), V = C.V;
    const rowsW = ['king', 'queen', 'man', 'woman', 'paris', 'rome', 'france', 'italy', 'big', 'small'];
    const counts = {};
    rowsW.forEach(function (w) { counts[w] = new Array(V).fill(0); });
    C.sentences.forEach(function (s) {
      s.forEach(function (w, i) {
        const name = C.words[w];
        if (!counts[name]) return;
        for (let j = Math.max(0, i - 2); j <= Math.min(s.length - 1, i + 2); j++) if (j !== i) counts[name][s[j]]++;
      });
    });
    // show the 14 context columns with the most total counts across these rows, skipping 'the'
    const tot = new Array(V).fill(0); rowsW.forEach(function (w) { counts[w].forEach(function (c, j) { tot[j] += c; }); });
    const cols = ML.topk(tot, 15).filter(function (j) { return C.words[j] !== 'the'; }).slice(0, 14);
    const st = { a: 'king', b: 'queen' };
    const ctl = el('div', { class: 'controls' }, host);
    U.select(ctl, 'row A', rowsW, st.a, function (v) { st.a = v; draw(); });
    U.select(ctl, 'row B', rowsW, st.b, function (v) { st.b = v; draw(); });
    const pic = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      pic.innerHTML = '';
      const cw = 44, rh = 22, L = 70, Wd = L + cols.length * cw, Ht = 64 + rowsW.length * rh;
      const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + Ht, role: 'img', 'aria-label': 'Co-occurrence counts' }, pic);
      let mx = 1; rowsW.forEach(function (w) { cols.forEach(function (j) { mx = Math.max(mx, counts[w][j]); }); });
      cols.forEach(function (j, k) { S('text', { x: L + k * cw + cw / 2, y: 54, 'text-anchor': 'start', 'font-size': 11, class: 'svg-text', transform: 'rotate(-40 ' + (L + k * cw + cw / 2) + ' 54)', text: C.words[j] }, svg); });
      rowsW.forEach(function (w, i) {
        const y = 62 + i * rh, hl = w === st.a || w === st.b;
        S('text', { x: L - 6, y: y + 15, 'text-anchor': 'end', 'font-size': 12, 'font-weight': hl ? 700 : 400, class: 'svg-text', text: w }, svg);
        cols.forEach(function (j, k) {
          const c = counts[w][j];
          S('rect', { x: L + k * cw + 1, y: y + 1, width: cw - 2, height: rh - 2, rx: 3, fill: 'var(--accent)', 'fill-opacity': c ? 0.12 + 0.88 * c / mx : 0.03, stroke: hl ? 'var(--ink-soft)' : 'none' }, svg);
          if (c) S('text', { x: L + k * cw + cw / 2, y: y + 15, 'text-anchor': 'middle', 'font-size': 10.5, fill: c / mx > 0.55 ? '#fff' : 'var(--ink-soft)', text: c }, svg);
        });
      });
      const cos = ML.cosine(counts[st.a], counts[st.b]);
      out.innerHTML = 'cosine similarity of the full count rows (all ' + V + ' columns): <b>' + Book.fmt(cos, 3) + '</b>' +
        '<br><span class="note">Toy counts. Rows have one entry per vocabulary word, so with the paper’s 1-million-word vocabulary each row would have a million entries, almost all zero. Word2Vec squeezes this information into a few hundred numbers.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-softmax
  W.xentGrad = function (host) {
    Book.head(host, 'The loss of one prediction, and which way to push each score',
      'The model scores 5 candidate next words. The correct word is marked ✓. Press “Gradient step” to change every score by −η × gradient and watch the loss fall.');
    const words = ['crown', 'throne', 'farm', 'school', 'cheese'];
    const st = { s: [0.5, 1.2, 0.2, -0.3, 0.0], y: 0, eta: 1.0, hist: [] };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'correct word', words.map(function (w, i) { return { value: i, label: w }; }), st.y, function (v) { st.y = +v; st.hist = []; draw(); });
    Book.slider(ctl, 'step size η', 0.1, 3, 0.1, st.eta, function (v) { st.eta = v; }, function (v) { return v.toFixed(1); });
    Book.button(ctl, 'Gradient step', function () {
      const p = ML.softmax(st.s); st.hist.push(-Math.log(p[st.y]));
      st.s = st.s.map(function (s, i) { return s - st.eta * (p[i] - (i === st.y ? 1 : 0)); }); draw();
    }, 'primary');
    Book.button(ctl, 'Reset', function () { st.s = [0.5, 1.2, 0.2, -0.3, 0.0]; st.hist = []; draw(); });
    const box = el('div', { class: 'grid2' }, host), c1 = el('div', null, box), c2 = el('div', null, box), out = el('div', { class: 'out' }, host);
    function draw() {
      const p = ML.softmax(st.s), L = -Math.log(p[st.y]);
      Book.chart.bars(c1, words.map(function (w, i) { return { label: w + (i === st.y ? ' ✓' : ''), value: p[i], note: (100 * p[i]).toFixed(0) + '%', color: i === st.y ? 'var(--green)' : 'var(--muted)' }; }),
        { max: 1, width: 340, height: 200, ylabel: 'probability', tickFmt: function (v) { return Math.round(v * 100) + '%'; } });
      const g = p.map(function (pi, i) { return pi - (i === st.y ? 1 : 0); });
      Book.chart.bars(c2, words.map(function (w, i) { return { label: w, value: g[i] + 1, note: (g[i] >= 0 ? '+' : '') + Book.fmt(g[i], 2), color: g[i] < 0 ? 'var(--green)' : 'var(--red)' }; }),
        { max: 2, width: 340, height: 200, ylabel: 'gradient pᵢ − yᵢ', tickFmt: function (v) { return Book.fmt(v - 1, 1); } });
      out.innerHTML = 'loss = −log p(' + words[st.y] + ') = −log ' + Book.fmt(p[st.y], 3) + ' = <b>' + Book.fmt(L, 3) + '</b>' +
        (st.hist.length ? ' · after ' + st.hist.length + ' steps (was ' + Book.fmt(st.hist[0], 3) + ')' : '') +
        '<br><span class="note">Green gradient (negative): the step raises that score. Red (positive): the step lowers it. Every one of the V scores gets a gradient, which is why a full softmax over a million words is expensive.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-lm
  function bigramCounts(C) {
    const big = {}, uni = new Array(C.V + 1).fill(0), BOS = C.V;
    C.sentences.forEach(function (s) {
      let prev = BOS;
      s.forEach(function (w) { const k = prev * 1000 + w; big[k] = (big[k] || 0) + 1; uni[prev]++; prev = w; });
    });
    return { big: big, uni: uni, BOS: BOS };
  }
  U.bigram = function () { const C = U.corpus(); return U._bigram || (U._bigram = bigramCounts(C)); };

  W.bigramModel = function (host) {
    Book.head(host, 'A bigram language model, counted from the toy corpus',
      'P(next | previous) = count(previous, next) ÷ count(previous). Pick a previous word to see its next-word distribution, or type a sentence to score it.');
    const C = U.corpus(), B = U.bigram();
    const st = { prev: 'queen', text: 'the queen wore the crown' };
    const ctl = el('div', { class: 'controls' }, host);
    U.select(ctl, 'previous word', ['the', 'queen', 'is', 'capital', 'speak', 'than', 'he'], st.prev, function (v) { st.prev = v; draw(); });
    const chart = el('div', null, host);
    const ctl2 = el('div', { class: 'controls' }, host);
    const inp = el('input', { type: 'text', value: st.text, style: 'flex:1;min-width:220px', 'aria-label': 'Sentence to score' }, ctl2);
    inp.addEventListener('input', function () { st.text = inp.value; score(); });
    ['the queen wore the crown', 'the queen sells cheese', 'the king said that he was tired', 'the king said that she was tired'].forEach(function (s) {
      Book.button(ctl2, s, function () { inp.value = s; st.text = s; score(); });
    });
    const out = el('div', { class: 'out' }, host);
    function draw() {
      const p = C.index[st.prev], row = [];
      for (let w = 0; w < C.V; w++) { const c = B.big[p * 1000 + w]; if (c) row.push({ w: C.words[w], p: c / B.uni[p] }); }
      row.sort(function (a, b) { return b.p - a.p; });
      Book.chart.bars(chart, row.slice(0, 10).map(function (r) { return { label: r.w, value: r.p, note: (100 * r.p).toFixed(0) + '%', color: 'var(--accent)' }; }),
        { max: 1, height: 200, ylabel: 'P(next | ' + st.prev + ')', tickFmt: function (v) { return Math.round(v * 100) + '%'; } });
    }
    function score() {
      const ws = st.text.trim().toLowerCase().split(/\s+/).filter(Boolean);
      let prev = B.BOS, logp = 0, lines = [], zero = false;
      ws.forEach(function (w) {
        const i = C.index[w];
        const c = i == null ? 0 : (B.big[prev * 1000 + i] || 0), p = B.uni[prev] ? c / B.uni[prev] : 0;
        lines.push('P(' + w + ' | ' + (prev === B.BOS ? '⟨start⟩' : C.words[prev]) + ') = ' + Book.fmt(p, 3) + (i == null ? '  (word not in vocabulary)' : ''));
        if (p === 0) zero = true; else logp += Math.log(p);
        prev = i == null ? B.BOS : i;
      });
      out.innerHTML = lines.join('<br>') + '<br>sentence probability = <b>' + (zero ? '0 (an unseen pair kills it)' : Book.fmt(Math.exp(logp), 5)) + '</b>' +
        '<br><span class="note">Toy counts. “queen … cheese” never occurred, so its probability is exactly zero, even though it is a fine sentence. A count-based model cannot share what it knows about “king” with “queen”.</span>';
    }
    draw(); score();
  };

  W.ngramSparsity = function (host) {
    Book.head(host, 'Longer N-grams are rarely seen before',
      'Train an N-gram table on part of the toy corpus, then check new sentences: what fraction of their N-grams never appeared in training? Move the training size.');
    const st = { n: 1200 };
    const test = W2V.makeCorpus({ seed: 99, sentences: 400 }).raw;
    const all = W2V.makeCorpus({}).raw;
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'training sentences', 50, 2400, 50, st.n, function (v) { st.n = v; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function grams(sents, n) { const set = new Set(); sents.forEach(function (s) { const t = ['<s>'].concat(s); for (let i = 0; i + n <= t.length; i++) set.add(t.slice(i, i + n).join(' ')); }); return set; }
    function draw() {
      const train = all.slice(0, st.n), pts = [];
      for (let n = 1; n <= 5; n++) {
        const seen = grams(train, n); let tot = 0, miss = 0;
        test.forEach(function (s) { const t = ['<s>'].concat(s); for (let i = 0; i + n <= t.length; i++) { tot++; if (!seen.has(t.slice(i, i + n).join(' '))) miss++; } });
        pts.push([n, miss / tot]);
      }
      Book.chart.lines(chart, [{ name: 'unseen', color: 'var(--red)', points: pts }], { xmin: 1, xmax: 5, ymin: 0, ymax: Math.max(0.2, pts[4][1] * 1.15), xticks: [1, 2, 3, 4, 5], xlabel: 'N (words per N-gram)', ylabel: 'unseen in test', yfmt: function (v) { return Math.round(100 * v) + '%'; }, height: 220 });
      out.innerHTML = pts.map(function (p) { return p[0] + '-grams: <b>' + (100 * p[1]).toFixed(1) + '%</b> unseen'; }).join(' · ') +
        '<br><span class="note">Toy numbers. The toy world is tiny and repetitive; in real text the gap is far larger, which is why N-gram models need trillions of words (paper §1).</span>';
    }
    draw();
  };
})();
