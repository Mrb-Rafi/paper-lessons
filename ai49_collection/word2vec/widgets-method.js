/* Word2Vec Interactive Web Book — Part II (neural language models before word2vec) and Part III (the two new models).
   Uses Book (core.js), ML, NN, W2V (toy.js) and W2VUI (widgets-found.js). */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, W2V = window.W2V, U = window.W2VUI;
  const big = function (x) {
    if (x >= 1e15) { const e = Math.floor(Math.log10(x)); return Book.fmt(x / Math.pow(10, e), 2) + ' × 10^' + e; }
    if (x >= 1e12) return Book.fmt(x / 1e12, 2) + ' trillion'; if (x >= 1e9) return Book.fmt(x / 1e9, 2) + ' billion';
    if (x >= 1e6) return Book.fmt(x / 1e6, 2) + ' million'; if (x >= 1e4) return Math.round(x).toLocaleString(); return Book.fmt(x, x < 100 ? 1 : 0);
  };
  // horizontal log-scale bars: data [{label, value, color}]
  function logBars(host, data, opts) {
    opts = opts || {};
    host.innerHTML = '';
    const Wd = 640, rh = 30, L = opts.labelW || 150, Ht = 12 + data.length * rh;
    const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + Ht, role: 'img', 'aria-label': opts.aria || 'Bar chart on a logarithmic scale' }, host);
    const lo = opts.lo || 1, hi = Math.max.apply(null, data.map(function (d) { return d.value; })) * 3;
    const X = function (v) { return L + (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (Wd - L - 120); };
    data.forEach(function (d, i) {
      const y = 6 + i * rh;
      S('text', { x: L - 8, y: y + 18, 'text-anchor': 'end', 'font-size': 12.5, class: 'svg-text', text: d.label }, svg);
      S('rect', { x: L, y: y + 4, width: Math.max(2, X(d.value) - L), height: rh - 10, rx: 4, fill: d.color || 'var(--accent)' }, svg);
      S('text', { x: X(d.value) + 6, y: y + 18, 'font-size': 12, 'font-weight': 600, class: 'svg-text', text: d.note || big(d.value) }, svg);
    });
    return svg;
  }
  U.logBars = logBars; U.big = big;

  // ------------------------------------------------------------------ ch-nnlm: cost
  W.nnlmCost = function (host) {
    Book.head(host, 'Where the NNLM spends its time (Eq. 1 and Eq. 2)',
      'Each bar is one term of Q = N×D + N×D×H + H×V, the operations per training word. Move the sizes; switch on hierarchical softmax to replace V by log₂V. The axis is logarithmic: each grid step is ×10.');
    const st = { N: 10, D: 100, H: 500, lv: 6, hs: false, E: 3, lt: 9 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'N (previous words)', 1, 10, 1, st.N, function (v) { st.N = v; draw(); });
    Book.slider(ctl, 'D (vector size)', 20, 1000, 10, st.D, function (v) { st.D = v; draw(); });
    Book.slider(ctl, 'H (hidden units)', 100, 1000, 50, st.H, function (v) { st.H = v; draw(); });
    Book.slider(ctl, 'V (vocabulary)', 3, 6, 0.1, st.lv, function (v) { st.lv = v; draw(); }, function (v) { return big(Math.pow(10, v)); });
    const hsBox = el('label', null, ctl); const cb = el('input', { type: 'checkbox' }, hsBox); hsBox.appendChild(document.createTextNode(' hierarchical softmax'));
    cb.addEventListener('change', function () { st.hs = cb.checked; draw(); });
    const ctl2 = el('div', { class: 'controls' }, host);
    Book.slider(ctl2, 'E (epochs)', 1, 50, 1, st.E, function (v) { st.E = v; draw(); });
    Book.slider(ctl2, 'T (training words)', 6, 9, 0.1, st.lt, function (v) { st.lt = v; draw(); }, function (v) { return big(Math.pow(10, v)); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const V = Math.pow(10, st.lv), out2 = st.hs ? Math.log2(V) : V;
      const t = [st.N * st.D, st.N * st.D * st.H, st.H * out2], Q = t[0] + t[1] + t[2];
      logBars(chart, [
        { label: 'N×D (projection)', value: t[0], color: 'var(--teal)' },
        { label: 'N×D×H (hidden)', value: t[1], color: 'var(--purple)' },
        { label: st.hs ? 'H×log₂V (output)' : 'H×V (output)', value: t[2], color: 'var(--red)' },
        { label: 'Q (total)', value: Q, color: 'var(--accent)' }]);
      const T = Math.pow(10, st.lt), O = st.E * T * Q, k = ML.argmax(t);
      out.innerHTML = 'Q = ' + big(t[0]) + ' + ' + big(t[1]) + ' + ' + big(t[2]) + ' = <b>' + big(Q) + '</b> per training word' +
        '<br>dominating term: <b>' + ['N×D', 'N×D×H', st.hs ? 'H×log₂V' : 'H×V'][k] + '</b> (' + U.pct(t[k] / Q) + ' of Q)' +
        '<br>O = E × T × Q = ' + st.E + ' × ' + big(T) + ' × ' + big(Q) + ' = <b>' + big(O) + '</b> operations';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-nnlm: a tiny NNLM trained live with the shared autograd
  W.nnlmTrain = function (host) {
    Book.head(host, 'A tiny feedforward NNLM, trained live',
      'N = 2 previous words → projection (2 × D = 2 × 10) → H = 24 tanh units → softmax over every toy word. It starts training as soon as it appears (about 4 seconds); then ask it for the next word.');
    const C = U.corpus(), V = C.V, PAD = V, D = 10, H = 24;
    // all (w[t−2], w[t−1]) → w[t] examples, with a padding token before the sentence start
    const X1 = [], X2 = [], Y = [];
    C.sentences.forEach(function (s) { const t = [PAD, PAD].concat(s); for (let i = 2; i < t.length; i++) { X1.push(t[i - 2]); X2.push(t[i - 1]); Y.push(t[i]); } });
    const st = { timer: null, steps: 0, losses: [], a: 'king', b: 'said' };
    let net, opt, rng;
    function reset() {
      stop(); rng = ML.rng(5);
      net = { Cm: NN.param(V + 1, D, rng, 0.3), W1a: NN.param(D, H, rng), W1b: NN.param(D, H, rng), b1: NN.zeros(1, H, true), W2: NN.param(H, V, rng), b2: NN.zeros(1, V, true) };
      opt = new NN.Adam([net.Cm, net.W1a, net.W1b, net.b1, net.W2, net.b2], { lr: 0.01 });
      st.steps = 0; st.losses = []; draw();
    }
    // the projection layer [C(w1), C(w2)] times W1 is the same as C(w1)·W1a + C(w2)·W1b
    function forward(i1, i2) {
      const h = NN.tanh(NN.add(NN.add(NN.matmul(NN.embedding(net.Cm, i1), net.W1a), NN.matmul(NN.embedding(net.Cm, i2), net.W1b)), net.b1));
      return NN.add(NN.matmul(h, net.W2), net.b2);
    }
    function train(k) {
      for (let s = 0; s < k; s++) {
        const idx = []; for (let j = 0; j < 64; j++) idx.push(rng.int(Y.length));
        NN.zeroGrad(opt.params);
        const L = NN.softmaxCE(forward(idx.map(function (i) { return X1[i]; }), idx.map(function (i) { return X2[i]; })), idx.map(function (i) { return Y[i]; }));
        L.backward(); opt.step(); st.steps++;
        if (st.steps % 10 === 0) st.losses.push([st.steps, L.item()]);
      }
    }
    const ctl = el('div', { class: 'controls' }, host);
    const go = Book.button(ctl, 'Train ▶', function () { if (st.timer) stop(); else { go.textContent = 'Pause ❚❚'; tick(); } }, 'primary');
    Book.button(ctl, 'Reset', reset);
    const ctxWords = C.words.filter(function (w) { return w !== 'the'; }).slice(0, 60).sort();
    U.select(ctl, 'word t−2', ['the'].concat(ctxWords), st.a, function (v) { st.a = v; draw(); });
    U.select(ctl, 'word t−1', ['the'].concat(ctxWords), st.b, function (v) { st.b = v; draw(); });
    const box = el('div', { class: 'grid2' }, host), lossPlot = el('div', null, box), pred = el('div', null, box), out = el('div', { class: 'out' }, host);
    function stop() { if (st.timer) clearTimeout(st.timer); st.timer = null; go.textContent = 'Train ▶'; }
    function tick() { train(8); draw(); if (st.steps >= 800) { stop(); return; } st.timer = setTimeout(tick, 20); }
    function draw() {
      Book.chart.lines(lossPlot, [{ color: Book.palette[0], marker: 'none', points: st.losses.length ? st.losses : [[0, Math.log(V)]] }],
        { width: 320, height: 200, ymin: 0, ymax: Math.log(V) * 1.05, xlabel: 'training steps (64 examples each)', ylabel: 'cross-entropy', xticks: [] });
      const logits = forward([C.index[st.a]], [C.index[st.b]]), p = ML.softmax(Array.from(logits.data)), top = ML.topk(p, 5);
      Book.chart.bars(pred, top.map(function (i) { return { label: C.words[i], value: p[i], note: (100 * p[i]).toFixed(0) + '%', color: 'var(--accent)' }; }),
        { max: 1, width: 320, height: 200, ylabel: 'P(next | ' + st.a + ' ' + st.b + ')', tickFmt: function (v) { return Math.round(100 * v) + '%'; } });
      const M = []; for (let i = 0; i < V; i++) M.push(ML.normalize(net.Cm.row(i)));
      const nn = W2V.nearest(M, C.words, C.index.king, 3);
      out.innerHTML = 'steps: <b>' + st.steps + '</b> · loss: <b>' + (st.losses.length ? Book.fmt(st.losses[st.losses.length - 1][1], 3) : '—') + '</b> (log V = ' + Book.fmt(Math.log(V), 2) + ' means “no idea”)' +
        '<br>projection-matrix neighbours of “king”: ' + nn.map(function (n) { return n.w + ' (' + Book.fmt(n.cos, 2) + ')'; }).join(', ') +
        '<br><span class="note">Toy NNLM built with the collection’s autograd library (NN.embedding, NN.tanh, NN.softmaxCE, Adam). Parameters: ' +
        ((V + 1) * D + 2 * D * H + H + H * V + V).toLocaleString() + '. The rows of the projection matrix are word vectors learned as a by-product.</span>';
    }
    reset();
    go.textContent = 'Pause ❚❚'; tick();          // start training straight away so the default view is interesting
  };

  // ------------------------------------------------------------------ ch-huffman
  W.huffmanBuild = function (host) {
    Book.head(host, 'Building a Huffman tree, one merge at a time',
      'The words are sorted by count. Each step joins the two nodes with the smallest counts under a new parent (exactly the loop in word2vec.c’s CreateBinaryTree). Frequent words end up near the root, with short codes.');
    const C = U.corpus();
    const sets = {
      'toy counts': ['the', 'is', 'king', 'paris', 'crown', 'bigger', 'sushi', 'rules'],
      'equal counts': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    };
    const st = { set: 'toy counts', k: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'words', Object.keys(sets), st.set, function (v) { st.set = v; st.k = 0; draw(); });
    Book.button(ctl, 'Next merge', function () { st.k = Math.min(st.k + 1, 7); draw(); }, 'primary');
    Book.button(ctl, 'Build all', function () { st.k = 7; draw(); });
    Book.button(ctl, 'Reset', function () { st.k = 0; draw(); });
    const pic = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      // like word2vec.c, the tree builder expects words sorted from most to least frequent
      const words = st.set === 'equal counts' ? sets[st.set] : sets[st.set].slice().sort(function (a, b) { return C.counts[C.index[b]] - C.counts[C.index[a]]; });
      const counts = st.set === 'equal counts' ? words.map(function () { return 100; }) : words.map(function (w) { return C.counts[C.index[w]]; });
      const h = W2V.huffman(counts), V = words.length;
      // node geometry: leaves in final in-order, internal nodes above, height = 1 + max child height
      const kids = {}; h.merges.forEach(function (m) { kids[m.node] = [m.left, m.right]; });
      const order = []; (function walk(n) { if (n < V) { order.push(n); return; } walk(kids[n][0]); walk(kids[n][1]); })(2 * V - 2);
      const pos = {}, hgt = {};
      order.forEach(function (n, i) { pos[n] = i; hgt[n] = 0; });
      h.merges.forEach(function (m) { pos[m.node] = (pos[m.left] + pos[m.right]) / 2; hgt[m.node] = 1 + Math.max(hgt[m.left], hgt[m.right]); });
      const maxH = hgt[2 * V - 2];
      pic.innerHTML = '';
      const cw = 78, Wd = V * cw + 20, Ht = 70 + maxH * 52;
      const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + Ht, role: 'img', 'aria-label': 'Huffman tree' }, pic);
      const X = function (n) { return 10 + cw / 2 + pos[n] * cw; }, Yc = function (n) { return Ht - 40 - hgt[n] * 52; };
      h.merges.slice(0, st.k).forEach(function (m) {
        [[m.left, 0], [m.right, 1]].forEach(function (c) {
          S('line', { x1: X(m.node), y1: Yc(m.node), x2: X(c[0]), y2: Yc(c[0]), stroke: 'var(--muted)', 'stroke-width': 1.6 }, svg);
          S('text', { x: (X(m.node) + X(c[0])) / 2 + (c[1] ? 5 : -12), y: (Yc(m.node) + Yc(c[0])) / 2, 'font-size': 11, 'font-weight': 700, fill: c[1] ? 'var(--red)' : 'var(--blue)', text: c[1] }, svg);
        });
      });
      const nodes = order.concat(h.merges.slice(0, st.k).map(function (m) { return m.node; }));
      nodes.forEach(function (n) {
        const isLeaf = n < V;
        const isNew = st.k && h.merges[st.k - 1].node === n;
        S('circle', { cx: X(n), cy: Yc(n), r: isLeaf ? 6 : 13, fill: isNew ? 'var(--accent)' : (isLeaf ? 'var(--ink-soft)' : 'var(--panel)'), stroke: 'var(--accent)', 'stroke-width': 1.5 }, svg);
        if (!isLeaf) S('text', { x: X(n), y: Yc(n) + 4, 'text-anchor': 'middle', 'font-size': 9.5, fill: isNew ? '#fff' : 'var(--ink-soft)', text: Math.round(h.count[n]) }, svg);
        else {
          S('text', { x: X(n), y: Yc(n) + 22, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, class: 'svg-text', text: words[n] }, svg);
          S('text', { x: X(n), y: Yc(n) + 35, 'text-anchor': 'middle', 'font-size': 10.5, class: 'svg-text', text: counts[n] }, svg);
        }
      });
      if (st.k < V - 1) {
        const m = h.merges[st.k];
        out.innerHTML = 'step ' + st.k + ' of ' + (V - 1) + '. Next: the two smallest nodes have counts ' + Math.round(h.count[m.left]) + ' and ' + Math.round(h.count[m.right]) + ' → new parent with count ' + Math.round(m.count) + '.';
      } else {
        const avg = W2V.avgCodeLength(counts, h.codes);
        out.innerHTML = words.map(function (w, i) { return w + ': <b>' + h.codes[i].join('') + '</b>'; }).join(' · ') +
          '<br>average code length (weighted by count) = <b>' + Book.fmt(avg, 2) + '</b> · a balanced tree needs log₂ ' + V + ' = <b>3</b>' +
          '<br><span class="note">Blue 0 = first (smaller) child, red 1 = second, as binary[min2i] = 1 in the code. ' + (st.set === 'toy counts' ? 'Counts are from the toy corpus.' : 'With equal counts Huffman builds a balanced tree.') + '</span>';
      }
    }
    draw();
  };

  W.huffmanSavings = function (host) {
    Book.head(host, 'How much does Huffman coding save?',
      'A vocabulary whose counts follow Zipf’s law (the k-th most common word has count ∝ 1/kˢ). Compare the outputs per prediction for a balanced tree (log₂V), a Huffman tree (computed), and the unigram entropy, log₂ of the unigram perplexity.');
    const st = { lv: 6, s: 1.0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'V', 2, 6, 0.1, st.lv, function (v) { st.lv = v; draw(); }, function (v) { return big(Math.round(Math.pow(10, v))); });
    Book.slider(ctl, 'Zipf exponent s', 0.8, 1.4, 0.05, st.s, function (v) { st.s = v; draw(); }, function (v) { return v.toFixed(2); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const V = Math.round(Math.pow(10, st.lv)), c = new Float64Array(V); let Z = 0, H = 0;
      for (let i = 0; i < V; i++) { c[i] = 1 / Math.pow(i + 1, st.s); Z += c[i]; }
      for (let i = 0; i < V; i++) { const p = c[i] / Z; H -= p * Math.log2(p); }
      const hu = W2V.huffmanAvgLength(c), bal = Math.log2(V);
      Book.chart.bars(chart, [{ label: 'balanced tree\nlog₂ V', value: bal, note: Book.fmt(bal, 1), color: 'var(--muted)' },
        { label: 'Huffman tree\n(average)', value: hu, note: Book.fmt(hu, 1), color: 'var(--accent)' },
        { label: 'log₂ unigram\nperplexity', value: H, note: Book.fmt(H, 1), color: 'var(--teal)' }], { height: 220, ylabel: 'outputs per prediction' });
      const toy = U.corpus(), th = W2V.huffman(toy.counts);
      out.innerHTML = 'V = ' + V.toLocaleString() + ': Huffman evaluates <b>' + Book.fmt(hu, 2) + '</b> outputs on average instead of ' + Book.fmt(bal, 2) + ' → speedup <b>×' + Book.fmt(bal / hu, 2) + '</b>' +
        '<br>toy corpus (V = ' + toy.V + '): balanced ' + Book.fmt(Math.log2(toy.V), 2) + ' vs Huffman ' + Book.fmt(W2V.avgCodeLength(toy.counts, th.codes), 2) +
        '<br><span class="note">Zipf counts are a model of word frequencies, not the paper’s data. The paper reports “about two times speedup” for a one-million-word vocabulary (§2.1); real text is more skewed than s = 1, so try s ≈ 1.1–1.2.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-hsoftmax
  W.hsMini = function (host) {
    Book.head(host, 'Four words, three yes/no questions',
      'A tree with 4 leaves has 3 inner nodes, each with its own vector. A word’s probability is the product of the sigmoid decisions on its path. Move the input vector h and watch the 4 probabilities, which always add up to 1.');
    const nodes = { root: [1.2, 0.4], L: [-0.8, 1.1], R: [0.9, -1.3] };
    const leaves = [{ w: 'king', path: [['root', 0], ['L', 0]] }, { w: 'queen', path: [['root', 0], ['L', 1]] }, { w: 'paris', path: [['root', 1], ['R', 0]] }, { w: 'rome', path: [['root', 1], ['R', 1]] }];
    const st = { h: [0.6, 0.8] };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'h₁', -2, 2, 0.05, st.h[0], function (v) { st.h[0] = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'h₂', -2, 2, 0.05, st.h[1], function (v) { st.h[1] = v; draw(); }, function (v) { return v.toFixed(2); });
    const box = el('div', { class: 'grid2' }, host), pic = el('div', null, box), chart = el('div', null, box), out = el('div', { class: 'out' }, host);
    function draw() {
      const sgm = {}; Object.keys(nodes).forEach(function (k) { sgm[k] = ML.sigmoid(ML.dot(st.h, nodes[k])); });
      pic.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 320 200', role: 'img', 'aria-label': 'A tree with three inner nodes' }, pic);
      const P = { root: [160, 30], L: [80, 100], R: [240, 100] }, LX = [40, 120, 200, 280];
      [['root', 'L', 0], ['root', 'R', 1]].forEach(function (e) { S('line', { x1: P[e[0]][0], y1: P[e[0]][1], x2: P[e[1]][0], y2: P[e[1]][1], stroke: 'var(--muted)' }, svg); });
      leaves.forEach(function (l, i) { const par = P[l.path[1][0]]; S('line', { x1: par[0], y1: par[1], x2: LX[i], y2: 170, stroke: 'var(--muted)' }, svg); });
      Object.keys(P).forEach(function (k) {
        S('circle', { cx: P[k][0], cy: P[k][1], r: 20, fill: 'var(--panel)', stroke: 'var(--accent)', 'stroke-width': 1.5 }, svg);
        S('text', { x: P[k][0], y: P[k][1] + 4, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: Book.fmt(sgm[k], 2) }, svg);
      });
      leaves.forEach(function (l, i) { S('text', { x: LX[i], y: 188, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, class: 'svg-text', text: l.w }, svg); });
      const probs = leaves.map(function (l) { return l.path.reduce(function (p, s) { return p * (s[1] ? 1 - sgm[s[0]] : sgm[s[0]]); }, 1); });
      Book.chart.bars(chart, leaves.map(function (l, i) { return { label: l.w, value: probs[i], note: Book.fmt(probs[i], 3), color: 'var(--accent)' }; }), { max: 1, width: 320, height: 200, ylabel: 'P(word | h)', tickFmt: function (v) { return v.toFixed(2); } });
      out.innerHTML = 'node numbers show σ(h·u) = the probability of taking the left (code 0) branch.<br>' +
        leaves.map(function (l, i) { return 'P(' + l.w + ') = ' + l.path.map(function (s) { return s[1] ? '(1 − ' + Book.fmt(sgm[s[0]], 2) + ')' : Book.fmt(sgm[s[0]], 2); }).join(' × ') + ' = ' + Book.fmt(probs[i], 3); }).join('<br>') +
        '<br>sum = <b>' + Book.fmt(ML.sum(probs), 6) + '</b> — always 1, with only 2 sigmoids per word instead of 4 exponentials.';
    }
    draw();
  };

  W.hsPath = function (host) {
    Book.head(host, 'Hierarchical softmax inside the trained toy model',
      'The toy skip-gram model uses a Huffman tree over all its words. Choose an input word and an output word: the probability is the product of the decisions along the output word’s path.');
    const m = U.model(false), C = m.corpus;
    const st = { inp: 'king', out: 'crown' };
    const ctl = el('div', { class: 'controls' }, host);
    U.select(ctl, 'input word', U.SETS.all, st.inp, function (v) { st.inp = v; draw(); });
    U.select(ctl, 'output word', ['crown', 'throne', 'he', 'she', 'the', 'capital', 'farm', 'than', 'cheese', 'of'], st.out, function (v) { st.out = v; draw(); });
    const path = el('div', { class: 'tokrow' }, host), chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const v = m.vector(st.inp), o = C.index[st.out], code = m.tree.codes[o], pt = m.tree.points[o];
      path.innerHTML = '';
      let p = 1;
      code.forEach(function (b, d) {
        let f = 0; for (let c = 0; c < m.D; c++) f += v[c] * m.syn1[pt[d] * m.D + c];
        const s = ML.sigmoid(f), take = b ? 1 - s : s; p *= take;
        el('span', { class: 'chip', html: 'node ' + pt[d] + ': code ' + b + ' → ' + Book.fmt(take, 3) }, path);
      });
      const all = []; let tot = 0;
      for (let w = 0; w < C.V; w++) { const q = m.hsProb(v, w); tot += q; all.push(q); }
      const top = ML.topk(all, 10);
      Book.chart.bars(chart, top.map(function (i) { return { label: C.words[i], value: all[i], note: Book.fmt(all[i], 2), color: C.words[i] === st.out ? 'var(--accent)' : 'var(--muted)' }; }), { height: 200, ylabel: 'P(context word | ' + st.inp + ')' });
      out.innerHTML = 'P(' + st.out + ' | ' + st.inp + ') = product of ' + code.length + ' factors = <b>' + Book.fmt(p, 4) + '</b>' +
        '<br>sum of P(w | ' + st.inp + ') over all ' + C.V + ' words = <b>' + Book.fmt(tot, 6) + '</b>' +
        '<br>dot products for this prediction: <b>' + code.length + '</b> (a full softmax needs ' + C.V + '); average over the corpus: ' + Book.fmt(W2V.avgCodeLength(C.counts, m.tree.codes), 2) +
        '<br><span class="note">Toy model (skip-gram, 20 dimensions). “the” is a neighbour of almost every word, so it takes the biggest share of probability; it is also the most frequent word, so it has the shortest code.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-rnnlm
  W.rnnTrain = function (host) {
    Book.head(host, 'A tiny recurrent language model, trained live',
      'The hidden state h (16 numbers, the coloured cells) is updated after every word from the new word and the previous h. It trains as soon as it appears (about 4 seconds); then step through a sentence with “Next word” and watch what it expects.');
    const C = U.corpus(), V = C.V, EOS = V, Hd = 16, T = 8;
    const seqs = C.sentences.filter(function (s) { return s.length <= T - 1; });
    const st = { timer: null, steps: 0, losses: [], s: 0, t: 0 };
    let net, opt, rng;
    function reset() {
      stop(); rng = ML.rng(11);
      net = { Uw: NN.param(V + 1, Hd, rng, 0.3), Wh: NN.param(Hd, Hd, rng, 0.25), b: NN.zeros(1, Hd, true), Wo: NN.param(Hd, V + 1, rng), bo: NN.zeros(1, V + 1, true) };
      opt = new NN.Adam([net.Uw, net.Wh, net.b, net.Wo, net.bo], { lr: 0.02 });
      st.steps = 0; st.losses = []; draw();
    }
    function run(inputs) {                               // inputs: array over time of id arrays (batch)
      let h = NN.zeros(inputs[0].length, Hd, false); const hs = [], logits = [];
      inputs.forEach(function (x) {
        h = NN.tanh(NN.add(NN.add(NN.embedding(net.Uw, x), NN.matmul(h, net.Wh)), net.b));
        hs.push(h); logits.push(NN.add(NN.matmul(h, net.Wo), net.bo));
      });
      return { hs: hs, logits: logits };
    }
    function batchOf(ss) {                               // input: </s> w1 … ; target: w1 … </s> (padded with </s>)
      const inp = [], tgt = [];
      for (let t = 0; t < T; t++) {
        inp.push(ss.map(function (s) { return t === 0 ? EOS : (t - 1 < s.length ? s[t - 1] : EOS); }));
        tgt.push(ss.map(function (s) { return t < s.length ? s[t] : EOS; }));
      }
      return { inp: inp, tgt: tgt };
    }
    function train(k) {
      for (let i = 0; i < k; i++) {
        const ss = []; for (let j = 0; j < 24; j++) ss.push(seqs[rng.int(seqs.length)]);
        const b = batchOf(ss), r = run(b.inp);
        NN.zeroGrad(opt.params);
        let L = null; r.logits.forEach(function (lg, t) { const l = NN.softmaxCE(lg, b.tgt[t]); L = L ? NN.add(L, l) : l; });
        L = NN.scale(L, 1 / T); L.backward(); opt.step(); st.steps++;
        if (st.steps % 5 === 0) st.losses.push([st.steps, L.item()]);
      }
    }
    const ctl = el('div', { class: 'controls' }, host);
    const go = Book.button(ctl, 'Train ▶', function () { if (st.timer) stop(); else { go.textContent = 'Pause ❚❚'; tick(); } }, 'primary');
    Book.button(ctl, 'Reset', reset);
    const demo = [['the', 'king', 'said', 'that', 'he', 'was', 'tired'], ['the', 'queen', 'said', 'that', 'she', 'was', 'tired'], ['people', 'in', 'japan', 'speak', 'japanese']];
    Book.select(ctl, 'sentence', demo.map(function (s, i) { return { value: i, label: s.join(' ') }; }), 0, function (v) { st.s = +v; st.t = 0; draw(); });
    Book.button(ctl, 'Next word →', function () { st.t = (st.t + 1) % demo[st.s].length; draw(); });
    const box = el('div', { class: 'grid2' }, host), lossPlot = el('div', null, box), right = el('div', null, box), out = el('div', { class: 'out' }, host);
    function stop() { if (st.timer) clearTimeout(st.timer); st.timer = null; go.textContent = 'Train ▶'; }
    function tick() { train(4); draw(); if (st.steps >= 400) { stop(); return; } st.timer = setTimeout(tick, 15); }
    function draw() {
      Book.chart.lines(lossPlot, [{ color: Book.palette[0], marker: 'none', points: st.losses.length ? st.losses : [[0, Math.log(V + 1)]] }],
        { width: 320, height: 200, ymin: 0, ymax: Math.log(V + 1) * 1.05, xlabel: 'training steps', ylabel: 'cross-entropy', xticks: [] });
      const s = demo[st.s], ids = [EOS].concat(s.slice(0, st.t + 1).map(function (w) { return C.index[w]; }));
      const r = run(ids.map(function (i) { return [i]; }));
      right.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 320 ' + (24 + ids.length * 22), role: 'img', 'aria-label': 'Hidden states over time' }, right);
      ids.forEach(function (id, t) {
        S('text', { x: 66, y: 34 + t * 22, 'text-anchor': 'end', 'font-size': 11.5, class: 'svg-text', text: id === EOS ? '</s>' : C.words[id] }, svg);
        r.hs[t].row(0).forEach(function (v, j) {
          S('rect', { x: 72 + j * 15, y: 20 + t * 22, width: 14, height: 19, rx: 2, fill: v >= 0 ? 'var(--blue)' : 'var(--red)', 'fill-opacity': Math.min(1, Math.abs(v)) * 0.9 + 0.05 }, svg);
        });
      });
      const last = ML.softmax(Array.from(r.logits[r.logits.length - 1].data)), top = ML.topk(last, 3);
      out.innerHTML = 'after “' + s.slice(0, st.t + 1).join(' ') + '” the model expects: ' + top.map(function (i) { return '<b>' + (i === EOS ? '</s>' : C.words[i]) + '</b> ' + U.pct(last[i]); }).join(', ') +
        ' · steps: ' + st.steps +
        '<br>cost per word (Eq. 3): H×H = ' + Hd * Hd + ', H×V = ' + Hd * (V + 1) + ' (toy) — the paper’s models have H = 640 and V = 82,000 or more.' +
        '<br><span class="note">Toy RNN (16 hidden units) trained by backpropagation through time with the shared autograd library. Blue = positive, red = negative hidden values.</span>';
    }
    reset();
    go.textContent = 'Pause ❚❚'; tick();          // start training straight away so the default view is interesting
  };

  // ------------------------------------------------------------------ ch-idea
  W.bigIdeaCost = function (host) {
    Book.head(host, 'The cost per training word of the four architectures',
      'Q for each model, using the formulas of the paper (Eq. 2–5) with hierarchical softmax everywhere. Same vector size D for all; log scale.');
    const st = { D: 300, H: 500, lv: 6 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'D (= H for the RNN)', 50, 1000, 10, st.D, function (v) { st.D = v; draw(); });
    Book.slider(ctl, 'H (NNLM hidden)', 100, 1000, 50, st.H, function (v) { st.H = v; draw(); });
    Book.slider(ctl, 'V', 4, 6, 0.1, st.lv, function (v) { st.lv = v; draw(); }, function (v) { return big(Math.pow(10, v)); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const D = st.D, H = st.H, lg = Math.log2(Math.pow(10, st.lv)), N = 10;
      const q = [
        { label: 'NNLM (N = 10)', value: N * D + N * D * H + H * lg, color: 'var(--purple)' },
        { label: 'RNNLM (H = D)', value: D * D + D * lg, color: 'var(--red)' },
        { label: 'CBOW (N = 8)', value: 8 * D + D * lg, color: 'var(--teal)' },
        { label: 'Skip-gram (C = 10)', value: 10 * (D + D * lg), color: 'var(--accent)' }];
      logBars(chart, q);
      out.innerHTML = 'NNLM costs <b>×' + Book.fmt(q[0].value / q[2].value, 0) + '</b> a CBOW step and <b>×' + Book.fmt(q[0].value / q[3].value, 0) + '</b> a skip-gram step.' +
        '<br><span class="note">Operation counts from the paper’s formulas, not measured times. CBOW uses 4 history + 4 future words (N = 8) as in §3.1; skip-gram uses C = 10 as in §3.2.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-cbow: Figure 1, interactive
  W.fig1 = function (host) {
    Book.head(host, 'Figure 1, with a real sentence',
      'Click a word to make it w(t). Left: CBOW sums (averages) the context vectors to predict w(t). Right: skip-gram uses w(t) to predict each context word. Window of 2, as drawn in the paper.');
    const C = U.corpus();
    const sent = ['the', 'king', 'wore', 'the', 'crown'];
    const st = { pos: 2 };
    const row = el('div', { class: 'tokrow' }, host), pic = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    function box(svg, x, y, label, hl) {
      S('rect', { x: x, y: y, width: 74, height: 26, rx: 5, fill: hl ? 'var(--accent-soft)' : 'var(--bg-soft)', stroke: hl ? 'var(--accent)' : 'var(--line)' }, svg);
      S('text', { x: x + 37, y: y + 17, 'text-anchor': 'middle', 'font-size': 11.5, class: 'svg-text', text: label }, svg);
    }
    function arrow(svg, x1, y1, x2, y2) { S('line', { x1: x1, y1: y1, x2: x2, y2: y2, stroke: 'var(--muted)', 'stroke-width': 1.3 }, svg); S('circle', { cx: x2, cy: y2, r: 2.5, fill: 'var(--muted)' }, svg); }
    function draw() {
      row.innerHTML = '';
      sent.forEach(function (w, i) { el('span', { class: 'chip' + (i === st.pos ? ' hl' : (Math.abs(i - st.pos) <= 2 ? ' ok' : '')), text: w, style: 'cursor:pointer', onclick: function () { st.pos = i; draw(); } }, row); });
      const ctx = []; for (let d = -2; d <= 2; d++) if (d) ctx.push({ d: d, w: sent[st.pos + d] });
      pic.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 700 250', role: 'img', 'aria-label': 'CBOW and skip-gram architectures (redrawn after paper Figure 1)' }, pic);
      ['INPUT', 'PROJECTION', 'OUTPUT'].forEach(function (t, i) { S('text', { x: 40 + i * 120, y: 16, 'font-size': 10.5, class: 'svg-text', text: t }, svg); S('text', { x: 390 + i * 115, y: 16, 'font-size': 10.5, class: 'svg-text', text: t }, svg); });
      ctx.forEach(function (c, k) {
        const y = 30 + k * 48 + (k > 1 ? 20 : 0);
        box(svg, 10, y, 'w(t' + (c.d > 0 ? '+' : '') + c.d + ') ' + (c.w || '—'), false); arrow(svg, 84, y + 13, 150, 125);
        box(svg, 590, y, 'w(t' + (c.d > 0 ? '+' : '') + c.d + ') ' + (c.w || '—'), false); arrow(svg, 518, 125, 588, y + 13);
      });
      S('rect', { x: 150, y: 110, width: 40, height: 30, rx: 5, fill: 'var(--panel)', stroke: 'var(--accent)' }, svg);
      S('text', { x: 170, y: 104, 'text-anchor': 'middle', 'font-size': 10.5, class: 'svg-text', text: 'SUM' }, svg);
      arrow(svg, 190, 125, 250, 125); box(svg, 252, 112, 'w(t) ' + sent[st.pos], true);
      box(svg, 380, 112, 'w(t) ' + sent[st.pos], true); arrow(svg, 454, 125, 478, 125);
      S('rect', { x: 478, y: 110, width: 40, height: 30, rx: 5, fill: 'var(--panel)', stroke: 'var(--accent)' }, svg);
      S('text', { x: 170, y: 240, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700, class: 'svg-text', text: 'CBOW' }, svg);
      S('text', { x: 520, y: 240, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 700, class: 'svg-text', text: 'Skip-gram' }, svg);
      const present = ctx.filter(function (c) { return c.w; });
      const m = U.model(true), mS = U.model(false);
      const avg = ML.scale(present.reduce(function (a, c) { return ML.add(a, m.vector(c.w)); }, ML.zeros(m.D)), 1 / present.length);
      const pC = m.hsProb(avg, C.index[sent[st.pos]]);
      out.innerHTML = '<b>CBOW</b>: 1 training example: (' + present.map(function (c) { return c.w; }).join(', ') + ') → ' + sent[st.pos] + ' · toy CBOW gives P = ' + Book.fmt(pC, 3) +
        '<br><b>Skip-gram</b>: ' + present.length + ' training examples: ' + present.map(function (c) { return sent[st.pos] + ' → ' + c.w + ' (P = ' + Book.fmt(mS.hsProb(mS.vector(sent[st.pos]), C.index[c.w]), 3) + ')'; }).join(', ') +
        '<br><span class="note">Redrawn after paper Figure 1. Probabilities are from the trained toy models.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-cbow / ch-skipgram: one training step
  function stepWidget(cbow) {
    return function (host) {
      Book.head(host, cbow ? 'One CBOW training step, number by number' : 'One skip-gram training step, number by number',
        cbow ? 'Pick a sentence position. The context vectors are averaged into h, h is pushed along the Huffman path of the middle word, and every vector involved moves a little. Watch P(middle word | context) rise.'
             : 'Pick a sentence position. The middle word’s vector is the input; for each context word, its Huffman path gives a probability, and the vectors move so that probability rises.');
      const C = U.corpus();
      const sent = ['the', 'king', 'wore', 'the', 'crown'];
      const st = { pos: 1, alpha: cbow ? 0.05 : 0.025, R: 2, model: null };
      function fresh() { st.model = new W2V.Model({ corpus: C, cbow: cbow, D: 20, window: 3, epochs: 6, seed: 3 }); st.model.trainWords(C.T); }
      fresh();
      const ctl = el('div', { class: 'controls' }, host);
      Book.slider(ctl, 'learning rate α', 0.005, 0.5, 0.005, st.alpha, function (v) { st.alpha = v; }, function (v) { return v.toFixed(3); });
      Book.slider(ctl, 'reach R', 1, 3, 1, st.R, function (v) { st.R = v; draw(); });
      Book.button(ctl, 'Apply one update', function () { update(); }, 'primary');
      Book.button(ctl, 'Reset model', function () { fresh(); draw(); });
      const row = el('div', { class: 'tokrow' }, host), out = el('div', { class: 'out' }, host);
      let last = null;
      function ctxIdx() { const r = []; for (let i = st.pos - st.R; i <= st.pos + st.R; i++) if (i !== st.pos && i >= 0 && i < sent.length) r.push(i); return r; }
      function avgVec(m) { const ix = ctxIdx(); return ML.scale(ix.reduce(function (a, i) { return ML.add(a, m.vector(sent[i])); }, ML.zeros(m.D)), 1 / ix.length); }
      function probs() {
        const m = st.model;
        if (cbow) return [{ lab: 'P(' + sent[st.pos] + ' | ' + ctxIdx().map(function (i) { return sent[i]; }).join(', ') + ')', p: m.hsProb(avgVec(m), C.index[sent[st.pos]]) }];
        return ctxIdx().map(function (i) { return { lab: 'P(' + sent[i] + ' | ' + sent[st.pos] + ')', p: m.hsProb(m.vector(sent[st.pos]), C.index[sent[i]]) }; });
      }
      function update() {
        const m = st.model, D = m.D, before = probs(), recs = [];
        if (cbow) {
          const h = avgVec(m); m.neu1e.fill(0); const rec = []; m._hs(h, 0, C.index[sent[st.pos]], st.alpha, rec); recs.push(rec);
          ctxIdx().forEach(function (i) { const l = C.index[sent[i]] * D; for (let k = 0; k < D; k++) m.syn0[l + k] += m.neu1e[k]; });
        } else {
          const l1 = C.index[sent[st.pos]] * D;
          ctxIdx().forEach(function (i) { m.neu1e.fill(0); const rec = []; m._hs(m.syn0, l1, C.index[sent[i]], st.alpha, rec); recs.push(rec); for (let k = 0; k < D; k++) m.syn0[l1 + k] += m.neu1e[k]; });
        }
        last = { before: before, recs: recs }; draw();
      }
      function draw() {
        row.innerHTML = '';
        const ix = ctxIdx();
        sent.forEach(function (w, i) { el('span', { class: 'chip' + (i === st.pos ? ' hl' : (ix.indexOf(i) >= 0 ? ' ok' : ' mask')), text: w, style: 'cursor:pointer', onclick: function () { st.pos = i; last = null; draw(); } }, row); });
        const now = probs();
        let html = now.map(function (q, k) { return q.lab + ' = <b>' + Book.fmt(q.p, 4) + '</b>' + (last && last.before[k] ? '  (before the update: ' + Book.fmt(last.before[k].p, 4) + ')' : ''); }).join('<br>');
        if (last) {
          html += '<br>Huffman-path nodes updated (code bit, σ before, g = (1 − code − σ)·α):';
          last.recs.forEach(function (rec, k) { html += '<br>  ' + (cbow ? '' : 'pair ' + (k + 1) + ': ') + rec.map(function (r) { return r.code + ' / ' + Book.fmt(r.p, 2) + ' / ' + (r.g >= 0 ? '+' : '') + Book.fmt(r.g, 4); }).join(' · '); });
        }
        html += '<br><span class="note">' + (cbow ? '1 classification for this position' : ix.length + ' classifications for this position (one per context word)') +
          '. Toy model after one epoch, so there is still room to improve. Each update is word2vec.c’s ' + (cbow ? 'CBOW branch (lines 435–494).' : 'skip-gram branch (lines 495–543), with the roles as in the paper.') + '</span>';
        out.innerHTML = html;
      }
      draw();
    };
  }
  W.cbowStep = stepWidget(true);
  W.sgStep = stepWidget(false);

  // ------------------------------------------------------------------ ch-skipgram: the random window
  W.skipPairs = function (host) {
    Book.head(host, 'The random reach R gives near words more weight',
      'For each word, skip-gram draws R uniformly from 1 … C and uses R words on each side. Draw once to see the pairs; the chart counts, over 20,000 draws, how often each distance gets used.');
    const C0 = U.corpus();
    const sent = C0.raw.find(function (s) { return s.length >= 7; });
    const st = { C: 5, pos: 3, seed: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'C (max distance)', 1, 10, 1, st.C, function (v) { st.C = v; draw(); });
    Book.button(ctl, 'Draw R again', function () { st.seed++; draw(); }, 'primary');
    const row = el('div', { class: 'tokrow' }, host), out = el('div', { class: 'out' }, host), chart = el('div', null, host), out2 = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(st.seed), R = 1 + r.int(st.C);
      row.innerHTML = '';
      sent.forEach(function (w, i) { el('span', { class: 'chip' + (i === st.pos ? ' hl' : (Math.abs(i - st.pos) <= R ? ' ok' : ' mask')), text: w, style: 'cursor:pointer', onclick: function () { st.pos = i; draw(); } }, row); });
      const pairs = []; for (let i = st.pos - R; i <= st.pos + R; i++) if (i !== st.pos && i >= 0 && i < sent.length) pairs.push(sent[st.pos] + ' → ' + sent[i]);
      out.innerHTML = 'R = <b>' + R + '</b> → ' + pairs.length + ' (input, output) pairs: ' + pairs.join(', ');
      const r2 = ML.rng(7), hits = new Array(st.C + 1).fill(0), N = 20000;
      for (let k = 0; k < N; k++) { const R2 = 1 + r2.int(st.C); for (let d = 1; d <= R2; d++) hits[d]++; }
      const data = []; for (let d = 1; d <= st.C; d++) data.push({ label: 'd = ' + d, value: hits[d] / N, note: Book.fmt(hits[d] / N, 2), color: 'var(--accent)' });
      Book.chart.bars(chart, data, { max: 1, height: 200, ylabel: 'share of draws using distance d' });
      out2.innerHTML = 'Distance d is used with probability (C − d + 1) / C: d = 1 always, d = C only 1 time in ' + st.C + '.' +
        '<br>Average classifications per word = 2 × (C + 1) / 2 = <b>' + (st.C + 1) + '</b>; Eq. 5 uses the upper bound C, the paper’s “R × 2”.' +
        '<br><span class="note">Toy sentence; simulated draws. The paper uses C = 10 (§3.2); word2vec.c draws b = random % window and uses reach window − b.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-train: the live trainer
  W.trainLive = function (host) {
    Book.head(host, 'Train CBOW or skip-gram on the toy corpus, live',
      'It trains once as soon as it appears. Change the architecture or sizes and press Train to train again. The map shows a 2-D projection (PCA) of the chosen words; the numbers show the loss, the learning rate and, at the end, the toy analogy accuracy.');
    const C = U.corpus();
    const st = { cbow: false, D: 20, win: 3, epochs: 6, set: 'people', sgCode: false, model: null, cancel: null, view: {}, focus: 'king' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'architecture', [{ value: 'sg', label: 'skip-gram' }, { value: 'cbow', label: 'CBOW' }], 'sg', function (v) { st.cbow = v === 'cbow'; reset(); });
    Book.slider(ctl, 'D', 2, 40, 1, st.D, function (v) { st.D = v; reset(); });
    Book.slider(ctl, 'window', 1, 5, 1, st.win, function (v) { st.win = v; reset(); });
    Book.slider(ctl, 'epochs', 1, 15, 1, st.epochs, function (v) { st.epochs = v; reset(); });
    const ctl2 = el('div', { class: 'controls' }, host);
    const go = Book.button(ctl2, 'Train ▶', function () { if (st.cancel) stop(); else start(); }, 'primary');
    Book.button(ctl2, 'Reset', reset);
    Book.select(ctl2, 'show', ['people', 'places', 'adjectives', 'all'], st.set, function (v) { st.set = v; st.view = {}; draw(); });
    U.select(ctl2, 'neighbours of', U.SETS.all, st.focus, function (v) { st.focus = v; draw(); });
    const box = el('div', { class: 'grid2' }, host), map = el('div', null, box), right = el('div', null, box);
    const lossPlot = el('div', null, right), out = el('div', { class: 'out' }, host);
    U.legend(host, ['male', 'female', 'country', 'city', 'language', 'adjective', 'comparative', 'superlative']);
    function reset() {
      stop();
      st.model = new W2V.Model({ corpus: C, cbow: st.cbow, D: st.D, window: st.win, epochs: st.epochs, seed: 1 });
      st.view = {}; st.result = null; draw();
    }
    function stop() { if (st.cancel) st.cancel(); st.cancel = null; go.textContent = 'Train ▶'; }
    function start() {
      if (st.model.epoch >= st.model.epochs) reset();
      go.textContent = 'Pause ❚❚';
      st.cancel = U.loop(function () {
        const more = st.model.trainWords(1200);
        if (!more) { st.result = W2V.evaluate(st.model); stop(); }
        draw(); return more;
      }, 25);
    }
    function draw() {
      const m = st.model;
      U.plotWords(map, m, U.SETS[st.set], st.view, { width: 420, height: 320, bold: [st.focus] });
      const pts = m.history.map(function (l, i) { return [i + 1, l]; });
      Book.chart.lines(lossPlot, [{ color: Book.palette[0], points: pts.length ? pts : [[0, 0]] }], { width: 320, height: 200, xmin: 0, xmax: m.epochs, ymin: 0, ymax: pts.length ? undefined : 4, xlabel: 'epoch', ylabel: 'average loss per prediction', xticks: [0, m.epochs] });
      const M = m.normalized(), nn = W2V.nearest(M, C.words, C.index[st.focus], 5);
      out.innerHTML = 'epoch ' + Math.min(m.epoch + 1, m.epochs) + ' of ' + m.epochs + ' · progress ' + U.pct(m.progress()) + ' · α = ' + Book.fmt(m.alpha, 5) + ' · loss now ' + (isFinite(m.currentLoss()) ? Book.fmt(m.currentLoss(), 3) : '—') +
        '<br>nearest to <b>' + st.focus + '</b>: ' + nn.map(function (n) { return n.w + ' ' + Book.fmt(n.cos, 2); }).join(', ') +
        (st.result ? '<br>toy analogy test: <b>' + U.pct(st.result.acc) + '</b> (' + st.result.n + ' questions; semantic ' + U.pct(st.result.semantic) + ', syntactic ' + U.pct(st.result.syntactic) + ')' : '<br>toy analogy test: runs when training ends') +
        '<br><span class="note">Toy numbers. Hierarchical softmax with the Huffman tree, starting α = ' + m.startAlpha + ' (word2vec.c’s defaults: 0.025 for skip-gram, 0.05 for CBOW) decreasing linearly to 0.</span>';
    }
    reset();
    start();                                   // train once straight away; the controls retrain
  };

  W.lrSchedule = function (host) {
    Book.head(host, 'The learning rate falls in a straight line',
      'α = α₀ × (1 − words processed ÷ total words), never below α₀ × 0.0001 (word2vec.c line 397). Move the progress point.');
    const st = { a0: 0.025, prog: 0.5 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'starting α₀', 0.005, 0.05, 0.005, st.a0, function (v) { st.a0 = v; draw(); }, function (v) { return v.toFixed(3); });
    Book.slider(ctl, 'progress', 0, 1, 0.01, st.prog, function (v) { st.prog = v; draw(); }, function (v) { return Math.round(100 * v) + '%'; });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const f = function (p) { return Math.max(st.a0 * (1 - p), st.a0 * 1e-4); };
      const pts = ML.linspace(0, 1, 101).map(function (p) { return [p, f(p)]; });
      Book.chart.lines(chart, [{ color: 'var(--accent)', marker: 'none', points: pts }, { color: 'var(--red)', points: [[st.prog, f(st.prog)]], r: 6 }],
        { xmin: 0, xmax: 1, ymin: 0, ymax: 0.05, xticks: [0, 0.25, 0.5, 0.75, 1], xfmt: function (v) { return Math.round(100 * v) + '%'; }, xlabel: 'share of training done', ylabel: 'α', yfmt: function (v) { return v.toFixed(4); }, height: 210, ml: 58 });
      out.innerHTML = 'at ' + Math.round(100 * st.prog) + '%: α = <b>' + Book.fmt(f(st.prog), 5) + '</b><br><span class="note">The paper: “starting learning rate 0.025 and decreased it linearly, so that it approaches zero at the end of the last training epoch” (§4.2).</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-code
  W.expTable = function (host) {
    Book.head(host, 'The sigmoid as a lookup table',
      'word2vec.c never calls exp() while training. It precomputes σ at EXP_TABLE_SIZE points between −MAX_EXP and +MAX_EXP and looks the answer up. Outside that range the update is skipped.');
    const st = { size: 1000, x: 1.3 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'EXP_TABLE_SIZE', 10, 1000, 10, st.size, function (v) { st.size = v; draw(); });
    Book.slider(ctl, 'x', -8, 8, 0.01, st.x, function (v) { st.x = v; draw(); }, function (v) { return v.toFixed(2); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function table(i) { const e = Math.exp((i / st.size * 2 - 1) * 6); return e / (e + 1); }
    function look(x) { return table(Math.floor((x + 6) * (st.size / 6 / 2))); }
    function draw() {
      const xs = ML.linspace(-5.99, 5.99, 400);
      let maxErr = 0; xs.forEach(function (x) { maxErr = Math.max(maxErr, Math.abs(look(x) - ML.sigmoid(x))); });
      Book.chart.lines(chart, [
        { name: 'exact σ', color: 'var(--muted)', marker: 'none', points: ML.linspace(-8, 8, 200).map(function (x) { return [x, ML.sigmoid(x)]; }) },
        { name: 'table', color: 'var(--accent)', marker: 'none', width: 1.6, points: xs.map(function (x) { return [x, look(x)]; }) }],
        { xmin: -8, xmax: 8, ymin: 0, ymax: 1, xticks: [-8, -6, -3, 0, 3, 6, 8], xlabel: 'x = h · u', ylabel: 'σ(x)', height: 220 });
      const inRange = st.x > -6 && st.x < 6, idx = Math.floor((st.x + 6) * (st.size / 6 / 2));
      out.innerHTML = 'x = ' + Book.fmt(st.x, 2) + ': ' + (inRange ? 'index (int)((x + 6) × (' + st.size + ' / 6 / 2)) = <b>' + idx + '</b> → table value ' + Book.fmt(look(st.x), 5) + ', exact ' + Book.fmt(ML.sigmoid(st.x), 5)
        : '|x| ≥ MAX_EXP = 6 → hierarchical softmax <b>skips</b> this node (the sigmoid is already within 0.0025 of 0 or 1)') +
        '<br>largest table error over (−6, 6): <b>' + Book.fmt(maxErr, 5) + '</b>' +
        '<br><span class="note">Port of word2vec.c lines 708–712 (table) and 454–456 (lookup). The book’s toy uses the exact sigmoid but keeps the same skipping rule.</span>';
    }
    draw();
  };

  W.gradCheck = function (host) {
    Book.head(host, 'Check the code’s one-line update against automatic differentiation',
      'For one tree node: input vector h, node vector u, code bit b. The loss is −log P(the correct branch). The autograd library differentiates it; word2vec.c instead writes g = (1 − b − σ(h·u)) · α. Compare.');
    const st = { seed: 3, code: 0, alpha: 0.025, D: 5 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'code bit b', [0, 1], 0, function (v) { st.code = +v; draw(); });
    Book.button(ctl, 'New random h and u', function () { st.seed++; draw(); }, 'primary');
    const out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(st.seed), hv = [], uv = [];
      for (let i = 0; i < st.D; i++) { hv.push(r.normal(0, 0.8)); uv.push(r.normal(0, 0.8)); }
      const h = NN.tensor([hv], true), u = NN.tensor(uv.map(function (x) { return [x]; }), true);
      const label = 1 - st.code, logit = NN.matmul(h, u), L = NN.sigmoidBCE(logit, [label]);
      L.backward();
      const f = ML.sigmoid(logit.item()), g = (1 - st.code - f) * st.alpha;
      const autoH = Array.from(h.grad).map(function (x) { return -st.alpha * x; }), codeH = uv.map(function (x) { return g * x; });
      const autoU = Array.from(u.grad).map(function (x) { return -st.alpha * x; }), codeU = hv.map(function (x) { return g * x; });
      const fmtv = function (v) { return '(' + v.map(function (x) { return Book.fmt(x, 5); }).join(', ') + ')'; };
      let maxd = 0; autoH.forEach(function (x, i) { maxd = Math.max(maxd, Math.abs(x - codeH[i]), Math.abs(autoU[i] - codeU[i])); });
      out.innerHTML = 'h·u = ' + Book.fmt(logit.item(), 3) + ', σ = ' + Book.fmt(f, 4) + ', target (1 − b) = ' + label + ', loss = ' + Book.fmt(L.item(), 4) +
        '<br>g = (1 − b − σ)·α = <b>' + Book.fmt(g, 6) + '</b>' +
        '<br>change of h, autograd  −α·∂L/∂h = ' + fmtv(autoH) + '<br>change of h, code      g·u      = ' + fmtv(codeH) +
        '<br>change of u, autograd  −α·∂L/∂u = ' + fmtv(autoU) + '<br>change of u, code      g·h      = ' + fmtv(codeU) +
        '<br>largest difference: <b>' + maxd.toExponential(1) + '</b> — the same numbers.' +
        '<br><span class="note">Autograd: the collection’s NN library (NN.sigmoidBCE). Code: word2vec.c lines 458–462. In the code, h’s change is collected in neu1e and applied after all nodes.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-followup
  W.negSampling = function (host) {
    Book.head(host, 'Negative sampling: tell the real word apart from a few random ones',
      'Follow-up paper. Instead of a tree, the model answers “is this the real neighbour?” for the true word and k “noise” words drawn from the unigram distribution raised to a power (0.75 in the code). Move the power; draw noise words; compare with hierarchical softmax on the toy.');
    const C = U.corpus();
    const st = { pow: 0.75, seed: 1, cmp: null };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'power', 0, 1, 0.05, st.pow, function (v) { st.pow = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'Draw 5 noise words', function () { st.seed++; draw(); });
    Book.button(ctl, 'Compare HS and negative sampling on the toy', function () { compare(); }, 'primary');
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function compare() {
      out.innerHTML = 'training…';
      setTimeout(function () {
        const res = [{ name: 'hierarchical softmax', o: { hs: true, negative: 0 } }, { name: 'negative sampling, k = 5', o: { hs: false, negative: 5 } }].map(function (r) {
          const t0 = performance.now(), m = new W2V.Model(Object.assign({ corpus: C, cbow: false, D: 20, window: 3, epochs: 6, seed: 1 }, r.o)).trainAll();
          return r.name + ': toy accuracy <b>' + U.pct(W2V.evaluate(m).acc) + '</b> in ' + Math.round(performance.now() - t0) + ' ms';
        });
        st.cmp = res.join('<br>'); draw();
      }, 20);
    }
    function draw() {
      const w = C.counts.map(function (c) { return Math.pow(c, st.pow); }), Z = ML.sum(w), N = ML.sum(C.counts);
      const show = [0, 1, 2, 3, 10, 30, 60, 100, C.V - 1];
      Book.chart.bars(chart, [].concat.apply([], show.map(function (i) {
        return [{ label: C.words[i], value: C.counts[i] / N, note: (100 * C.counts[i] / N).toFixed(1), color: 'var(--muted)' }, { label: '', value: w[i] / Z, note: (100 * w[i] / Z).toFixed(1), color: 'var(--accent)' }];
      })), { height: 220, ylabel: '% of draws', tickFmt: function (v) { return Book.fmt(100 * v, 0); } });
      const r = ML.rng(st.seed), table = W2V.unigramTable(C.counts, st.pow, 20000), noise = [];
      for (let i = 0; i < 5; i++) noise.push(C.words[table[r.int(table.length)]]);
      out.innerHTML = 'grey = raw frequency, colour = frequency<sup>' + st.pow.toFixed(2) + '</sup> (normalised). Raising to a power below 1 gives rare words more chances.' +
        '<br>noise words drawn: <b>' + noise.join(', ') + '</b>' + (st.cmp ? '<br>' + st.cmp : '') +
        '<br><span class="note">Negative sampling and the 0.75 power come from Mikolov et al. 2013b (arXiv 1310.4546), not from this paper; word2vec.c lines 52–68 and 465–484. Toy results.</span>';
    }
    draw();
  };

  W.subsample = function (host) {
    Book.head(host, 'Subsampling: throw away some copies of very frequent words',
      'Follow-up paper. Each occurrence of a word is kept with a probability that falls as the word gets more frequent. Move the threshold “sample”; see the keep probability of toy words and one example sentence.');
    const C = U.corpus();
    const st = { ls: -2, seed: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'sample', -4, -1, 0.1, st.ls, function (v) { st.ls = v; draw(); }, function (v) { return '1e' + v.toFixed(1); });
    Book.button(ctl, 'Another draw', function () { st.seed++; draw(); });
    const chart = el('div', null, host), row = el('div', { class: 'tokrow' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const s = Math.pow(10, st.ls), ids = [0, 1, 2, 5, 10, 20, 40, 80, C.V - 1];
      Book.chart.bars(chart, ids.map(function (i) { const k = W2V.keepProb(C.counts[i], s, C.T); return { label: C.words[i] + '\n' + (100 * C.counts[i] / C.T).toFixed(1) + '%', value: k, note: U.pct(k), color: k < 1 ? 'var(--accent)' : 'var(--green)' }; }),
        { max: 1, height: 220, ylabel: 'keep probability', mb: 52 });
      const r = ML.rng(st.seed), sent = C.raw[0];
      row.innerHTML = '';
      sent.forEach(function (w) { const keep = r.next() < W2V.keepProb(C.counts[C.index[w]], s, C.T); el('span', { class: 'chip' + (keep ? '' : ' mask'), html: keep ? w : '<s>' + w + '</s>' }, row); });
      out.innerHTML = 'keep = (√(f / sample) + 1) × sample / f, where f is the word’s share of the corpus (capped at 1).' +
        '<br><span class="note">This is word2vec.c’s formula (lines 408–412). The follow-up paper (Mikolov et al. 2013b, §2.3) writes a slightly different discard rule, 1 − √(t / f). Neither is part of the 2013a paper this book is about. Toy frequencies.</span>';
    }
    draw();
  };
})();
