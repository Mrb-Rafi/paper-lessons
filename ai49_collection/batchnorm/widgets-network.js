/* Batch Normalization book, Part IV (batch-normalized networks) widgets. Uses Book, ML, NN, BNT. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, T = window.BNT;
  const f = T.f, sig = function (x) { return 1 / (1 + Math.exp(-x)); };

  // trained toy networks, shared by several widgets (trained once, in the background)
  const cache = {};
  const RECIPES = {
    bn: { net: { sizes: [64, 32, 32, 32, 10], act: 'sigmoid', bn: true, sd: 'xavier', seed: 5 }, lr: 0.5, momentum: 0, bs: 60, steps: 1500, every: 500 },
    relu: { net: { sizes: [64, 48, 48, 10], act: 'relu', bn: false, sd: 'he', seed: 6 }, lr: 0.02, momentum: 0.9, bs: 32, steps: 1000, every: 500 }
  };
  function trained(kind) {
    if (!cache[kind]) cache[kind] = new Promise(function (res) { const J = T.job(RECIPES[kind]); T.runner().start([J], function () {}, function () { res(J); }); });
    return cache[kind];
  }
  function waiting(host) { return el('div', { class: 'note', text: 'Training the toy network in your browser (a couple of seconds)…' }, host); }
  function table(rows, head) {
    let h = '<table class="mtab"><tr>' + head.map(function (c) { return '<th>' + c + '</th>'; }).join('') + '</tr>';
    rows.forEach(function (r) { h += '<tr>' + r.map(function (c) { return '<td>' + c + '</td>'; }).join('') + '</tr>'; });
    return h + '</table>';
  }
  function softmax(row) { const m = Math.max.apply(null, row), e = row.map(function (v) { return Math.exp(v - m); }), s = ML.sum(e); return e.map(function (v) { return v / s; }); }

  // ------------------------------------------------------------------ ch20: training mode vs inference mode
  W.trainVsInfer = function (host) {
    Book.head(host, 'One glyph, classified in training mode and in inference mode',
      'A toy BN network (trained for 1,500 steps). Training mode: the glyph is classified inside 40 random minibatches of m test glyphs; its probability for the right class jumps around. Inference mode: population statistics, one fixed answer.');
    const st = { k: 3, m: 8 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'test glyph #', 0, 59, 1, st.k, function (v) { st.k = v; draw(); });
    Book.slider(ctl, 'minibatch size m', 2, 64, 1, st.m, function (v) { st.m = v; draw(); });
    const row = el('div', { class: 'grid2' }, host), gl = el('div', null, row), view = el('div', null, row), out = el('div', { class: 'out' }, host), wait = waiting(host);
    let J;
    function probs(net, X, mode, i) { const o = net.forward(X, mode); return softmax(o.row(i)); }
    function draw() {
      if (!J) return;
      const D = T.data(), net = J.net, k = st.k, y = D.test.y[k], r = ML.rng(100 + k * 7 + st.m), pts = [];
      for (let t = 0; t < 40; t++) { const idx = [k]; while (idx.length < st.m) idx.push(60 + r.int(D.test.n - 60)); pts.push([t + 1, probs(net, T.rows(D.test, idx), 'batch', 0)[y]]); }
      const pe = probs(net, T.rows(D.test, [k]), 'eval', 0), pc = probs(net, T.rows(D.test, Array(st.m).fill(k)), 'batch', 0);
      gl.innerHTML = ''; el('div', { class: 'note', html: 'glyph #' + k + ', true digit <b>' + y + '</b>' }, gl); T.glyph(gl, D.test.X, k, 110);
      T.lines(view, [{ name: 'training mode (random batch-mates)', color: '#b7791f', points: pts, r: 3, noline: true }, { name: 'inference mode', color: T.colors.on, marker: 'none', points: [[1, pe[y]], [40, pe[y]]] }],
        { xmin: 1, xmax: 40, ymin: 0, ymax: 1, xticks: [1, 10, 20, 30, 40], xlabel: 'random minibatch', ylabel: 'P(correct digit)', height: 220, width: 420 });
      const mv = T.mv(pts.map(function (p) { return p[1]; }));
      out.innerHTML = 'inference mode: P(' + y + ') = <b>' + f(pe[y], 3) + '</b>, always · training mode: mean ' + f(mv.mean, 3) + ', spread <b>' + f(Math.sqrt(mv.var), 3) + '</b> across batches · a batch of ' + st.m + ' identical copies of this glyph: σ²_B = 0, every x̂ = 0, and the network answers <b>' + ML.argmax(pc) + '</b> (P = ' + f(Math.max.apply(null, pc), 2) + ') whatever the glyph.';
    }
    trained('bn').then(function (j) { J = j; wait.remove(); draw(); });
  };

  // ------------------------------------------------------------------ ch20: unbiased variance
  W.unbiasedVar = function (host) {
    Book.head(host, 'Batch variances are too small by (m − 1)/m',
      'Draw 4,000 minibatches of size m from a distribution with true variance 1, compute each batch’s variance σ²_B (dividing by m), and average.');
    const st = { m: 4 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'minibatch size m', 2, 40, 1, st.m, function (v) { st.m = v; draw(); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    function avgVar(m, n, r) { let s = 0; for (let k = 0; k < n; k++) { const b = []; for (let i = 0; i < m; i++) b.push(r.normal(0, 1)); s += ML.variance(b); } return s / n; }
    const r0 = ML.rng(44), curve = [];
    for (let m = 2; m <= 40; m += 2) curve.push([m, avgVar(m, 800, r0)]);
    function draw() {
      const a = avgVar(st.m, 4000, ML.rng(st.m)), c = a * st.m / (st.m - 1);
      T.lines(view, [{ name: 'average σ²_B (simulated)', color: '#c0392b', points: curve, r: 2.5 }, { name: '(m − 1)/m', color: 'var(--muted)', marker: 'none', dash: '5 4', points: ML.range(39).map(function (i) { const m = i + 2; return [m, (m - 1) / m]; }) },
        { name: 'true variance', color: T.colors.on, marker: 'none', points: [[2, 1], [40, 1]] }, { name: '', color: 'var(--ink)', points: [[st.m, a]], r: 5 }],
        { xmin: 2, xmax: 40, ymin: 0.4, ymax: 1.1, xticks: [2, 10, 20, 30, 40], xlabel: 'minibatch size m', ylabel: 'variance', height: 220 });
      out.innerHTML = 'm = ' + st.m + ': average batch variance <b>' + f(a, 3) + '</b> (theory (m−1)/m = ' + f((st.m - 1) / st.m, 3) + ') · × m/(m−1) = <b>' + f(c, 3) + '</b> ≈ the true variance 1.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch21: Algorithm 2, lines 8–12
  W.alg2Runner = function (host) {
    Book.head(host, 'Estimate the population statistics (Algorithm 2, line 10)',
      'The trained toy BN network. For one unit, average μ_B and σ²_B over more and more training minibatches of m = 60, and compare with the moving average PyTorch-style training kept (momentum 0.1). Then test accuracy in inference mode with each.');
    const st = { l: 1, nb: 20 }, ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'layer', [{ value: '0', label: '1' }, { value: '1', label: '2' }, { value: '2', label: '3' }], '1', function (v) { st.l = +v; draw(); });
    Book.slider(ctl, 'minibatches averaged', 1, 100, 1, st.nb, function (v) { st.nb = v; draw(); });
    const g = el('div', { class: 'grid2' }, host), L = el('div', null, g), R = el('div', null, g), out = el('div', { class: 'out' }, host), wait = waiting(host);
    let J, ema, stats = {};
    function collect(l) {
      if (stats[l]) return stats[l];
      const D = T.data(), r = ML.rng(808 + l), mus = [], vs = [];
      for (let k = 0; k < 100; k++) { const idx = []; for (let i = 0; i < 60; i++) idx.push(r.int(D.train.n)); J.net.forward(T.rows(D.train, idx), 'batch'); const s = J.net.rec.z[l].stats; mus.push(s.mean[0]); vs.push(s.var[0]); }
      return (stats[l] = { mu: mus, v: vs });
    }
    function draw() {
      if (!J) return;
      const c = collect(st.l), n = st.nb, m = 60, pm = [], pv = [];
      let sm = 0, sv = 0; for (let k = 0; k < 100; k++) { sm += c.mu[k]; sv += c.v[k]; pm.push([k + 1, sm / (k + 1)]); pv.push([k + 1, sv / (k + 1) * m / (m - 1)]); }
      const Ex = pm[n - 1][1], Vx = pv[n - 1][1], e = ema[st.l];
      T.lines(L, [{ name: 'E[x] ← E_B[μ_B]', color: T.colors.on, marker: 'none', points: pm }, { name: 'moving average', color: '#b7791f', marker: 'none', dash: '5 4', points: [[1, e.mean[0]], [100, e.mean[0]]] }, { name: '', color: 'var(--ink)', points: [[n, Ex]], r: 4 }], { xmin: 1, xmax: 100, xticks: [1, 25, 50, 75, 100], xlabel: 'minibatches', ylabel: 'E[x] (unit 1)', height: 200, width: 320, ydec: 2 });
      T.lines(R, [{ name: 'Var[x] ← m/(m−1)·E_B[σ²_B]', color: T.colors.on, marker: 'none', points: pv }, { name: 'moving average', color: '#b7791f', marker: 'none', dash: '5 4', points: [[1, e.var[0]], [100, e.var[0]]] }, { name: '', color: 'var(--ink)', points: [[n, Vx]], r: 4 }], { xmin: 1, xmax: 100, xticks: [1, 25, 50, 75, 100], xlabel: 'minibatches', ylabel: 'Var[x] (unit 1)', height: 200, width: 320, ydec: 2 });
      // accuracy with Algorithm 2 statistics from n minibatches (all units), vs the moving averages
      const net = J.net, keep = net.run.map(function (q) { return q && { mean: Float64Array.from(q.mean), var: Float64Array.from(q.var) }; });
      const accEma = net.accuracy(T.data().test, 'eval');
      net.population(T.data().train, n, 60, 4242); const accPop = net.accuracy(T.data().test, 'eval');
      net.run.forEach(function (q, i) { if (q) { q.mean.set(keep[i].mean); q.var.set(keep[i].var); } });
      out.innerHTML = 'after ' + n + ' minibatch' + (n > 1 ? 'es' : '') + ': E[x] = <b>' + f(Ex, 3) + '</b>, Var[x] = <b>' + f(Vx, 3) + '</b> (moving average: ' + f(e.mean[0], 3) + ', ' + f(e.var[0], 3) + ') · test accuracy (toy) with Algorithm 2 statistics <b>' + (100 * accPop).toFixed(1) + '%</b>, with the moving averages <b>' + (100 * accEma).toFixed(1) + '%</b>';
    }
    trained('bn').then(function (j) { J = j; ema = J.net.run.map(function (q) { return q && { mean: Array.from(q.mean), var: Array.from(q.var) }; }); wait.remove(); draw(); });
  };

  // ------------------------------------------------------------------ ch21: folding BN into the weights
  W.foldBN = function (host) {
    Book.head(host, 'BN at inference = one multiply and one add, folded into W',
      'Layer 1 of the trained toy BN network. For the chosen unit: a = γ/√(Var[x]+ε) and c = β − γ·E[x]/√(Var[x]+ε). The folded layer computes (a·w)·u + c directly from the pixels u.');
    const st = { j: 0 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'unit', 0, 31, 1, st.j, function (v) { st.j = v; draw(); });
    const tab = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host), wait = waiting(host);
    let J;
    function draw() {
      if (!J) return;
      const net = J.net, j = st.j, R = net.run[0], g = net.g[0].data[j], b = net.be[0].data[j], E = R.mean[j], V = R.var[j], eps = 1e-5;
      const a = g / Math.sqrt(V + eps), c = b - g * E / Math.sqrt(V + eps), D = T.data(), idx = [0, 1, 2, 3, 4];
      const o = net.forward(T.rows(D.test, idx), 'eval'), z = net.rec.z[0], rows = [];
      idx.forEach(function (k, i) {
        let wu = 0; for (let p = 0; p < 64; p++) wu += D.test.X[k * 64 + p] * net.W[0].data[p * 32 + j];
        let fold = c; for (let p = 0; p < 64; p++) fold += D.test.X[k * 64 + p] * (a * net.W[0].data[p * 32 + j]);
        rows.push(['glyph ' + k, f(wu, 4), f(z.data[i * 32 + j], 5), f(fold, 5), Math.abs(fold - z.data[i * 32 + j]) < 1e-9 ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>']);
      });
      void o;
      tab.innerHTML = table(rows, ['input', 'x = w·u', 'BN (inference mode)', 'folded (a·w)·u + c', 'same?']);
      out.innerHTML = 'γ = ' + f(g, 3) + ', β = ' + f(b, 3) + ', E[x] = ' + f(E, 3) + ', Var[x] = ' + f(V, 3) + ' → a = <b>' + f(a, 4) + '</b>, c = <b>' + f(c, 4) + '</b>. The BN layer has vanished into the weights.';
    }
    trained('bn').then(function (j) { J = j; wait.remove(); draw(); });
  };

  // ------------------------------------------------------------------ ch22: batch size vs noise
  W.batchSizeStats = function (host) {
    Book.head(host, 'Smaller batches, noisier statistics',
      'One unit of layer 2 of the trained toy BN network. For each minibatch size m, 300 random training minibatches: how much μ_B and σ²_B scatter around the population values. Histogram: μ_B at the chosen m.');
    const st = { m: 8 }, ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'minibatch size m', [1, 2, 4, 8, 16, 32, 64, 128].map(String), '8', function (v) { st.m = +v; draw(); });
    const g = el('div', { class: 'grid2' }, host), L = el('div', null, g), R = el('div', null, g), out = el('div', { class: 'out' }, host), wait = waiting(host);
    let J, cache2 = {};
    function sample(m) {
      if (cache2[m]) return cache2[m];
      const D = T.data(), r = ML.rng(m * 11), mus = [], vs = [];
      for (let k = 0; k < 300; k++) { const idx = []; for (let i = 0; i < m; i++) idx.push(r.int(D.train.n)); J.net.forward(T.rows(D.train, idx), 'batch'); const s = J.net.rec.z[1].stats; mus.push(s.mean[0]); vs.push(s.var[0]); }
      return (cache2[m] = { mu: mus, v: vs });
    }
    function draw() {
      if (!J) return;
      const ms = [2, 4, 8, 16, 32, 64, 128], sdm = ms.map(function (m) { return [m, Math.sqrt(ML.variance(sample(m).mu))]; }), sdv = ms.map(function (m) { return [m, Math.sqrt(ML.variance(sample(m).v))]; });
      T.lines(L, [{ name: 'spread of μ_B', color: T.colors.on, points: sdm }, { name: 'spread of σ²_B', color: '#c0392b', points: sdv }, { name: '∝ 1/√m', color: 'var(--muted)', dash: '4 4', marker: 'none', points: ms.map(function (m) { return [m, sdm[0][1] * Math.sqrt(2 / m)]; }) }],
        { logx: true, xmin: 2, xmax: 128, xticks: ms, xlabel: 'minibatch size m (log scale)', ylabel: 'spread', height: 220, width: 320, ydec: 2 });
      if (st.m === 1) { R.innerHTML = ''; out.innerHTML = 'm = 1: μ_B is the value itself, σ²_B = 0, x̂ = 0 for every input. PyTorch refuses to train like this (“Expected more than 1 value per channel when training”).'; return; }
      const s = sample(st.m), pop = J.net.run[1].mean[0];
      const w = Math.max(0.05, 4 * Math.sqrt(ML.variance(sample(2).mu)));
      T.hist(R, s.mu, { lo: pop - w, hi: pop + w, bins: 40, title: 'μ_B over 300 batches, m = ' + st.m, width: 320, marks: [{ x: pop, label: 'population', color: '#c0392b' }], ticks: [] });
      out.innerHTML = 'm = ' + st.m + ': μ_B scatters by <b>' + f(Math.sqrt(ML.variance(s.mu)), 3) + '</b>, σ²_B by <b>' + f(Math.sqrt(ML.variance(s.v)), 3) + '</b> (population: E[x] = ' + f(pop, 3) + ', Var[x] = ' + f(J.net.run[1].var[0], 3) + ').';
    }
    trained('bn').then(function (j) { J = j; wait.remove(); draw(); });
  };

  // ------------------------------------------------------------------ ch23: where to put BN
  W.whereBN = function (host) {
    Book.head(host, 'Layer inputs u are lopsided; pre-activations Wu + b look Gaussian',
      'A trained toy ReLU network without BN, 600 test glyphs. Left: u, one input of layer 2 (the output of a layer-1 ReLU). Right: x = Wu + b, one pre-activation of layer 2.');
    const st = { j: 0 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'unit', 0, 47, 1, st.j, function (v) { st.j = v; draw(); });
    const g = el('div', { class: 'grid2' }, host), L = el('div', null, g), R = el('div', null, g), out = el('div', { class: 'out' }, host), wait = waiting(host);
    let J;
    function skew(v) { const s = T.mv(v), sd = Math.sqrt(s.var) || 1; let k = 0; v.forEach(function (x) { k += Math.pow((x - s.mean) / sd, 3); }); return k / v.length; }
    function draw() {
      if (!J) return;
      J.net.forward(T.rows(T.data().test, ML.range(600)), 'eval');
      const u = T.col(J.net.rec.h[0], st.j), x = T.col(J.net.rec.z[1], st.j);
      const su = T.mv(u), sx = T.mv(x), zu = u.filter(function (v) { return v === 0; }).length;
      T.hist(L, u, { lo: Math.min.apply(null, u) - 0.1, hi: Math.max.apply(null, u) + 0.1, bins: 40, title: 'u (ReLU output)', width: 300, color: 'var(--muted)', ticks: [] });
      T.hist(R, x, { lo: sx.mean - 4 * Math.sqrt(sx.var), hi: sx.mean + 4 * Math.sqrt(sx.var), bins: 40, title: 'x = Wu + b (pre-activation)', width: 300, ticks: [] });
      out.innerHTML = 'u: <b>' + (100 * zu / u.length).toFixed(0) + '%</b> exact zeros, skewness <b>' + f(skew(u), 2) + '</b> · x: ' + (100 * x.filter(function (v) { return v === 0; }).length / x.length).toFixed(0) + '% zeros, skewness <b>' + f(skew(x), 2) + '</b> (a bell curve has 0). Mean and variance describe x well; they say little about u’s shape.';
    }
    trained('relu').then(function (j) { J = j; wait.remove(); draw(); });
  };

  // ------------------------------------------------------------------ ch23: the bias cancels
  W.biasCancel = function (host) {
    Book.head(host, 'A bias before BN does nothing',
      'Six pre-activations Wu in a minibatch, plus a bias b. Compare BN(Wu + b) with BN(Wu), and with a plain sigmoid(Wu + b) without BN.');
    const st = { b: 2 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'bias b', -5, 5, 0.1, st.b, function (v) { st.b = v; draw(); }, function (v) { return v.toFixed(1); });
    const tab = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    const wu = [-1.3, 0.4, 2.2, -0.2, 1.1, 0.6];
    function bn(v) { const t = NN.tensor(v.map(function (q) { return [q]; })); return Array.from(NN.batchNorm(t, NN.full(1, 1, 1), NN.zeros(1, 1), {}).data); }
    function draw() {
      const a = bn(wu.map(function (v) { return v + st.b; })), c = bn(wu);
      tab.innerHTML = table(wu.map(function (v, i) { return [f(v, 1), f(v + st.b, 1), f(a[i], 4), f(c[i], 4), f(sig(v + st.b), 3)]; }), ['Wu', 'Wu + b', 'BN(Wu + b)', 'BN(Wu)', 'sigmoid(Wu + b), no BN']);
      out.innerHTML = 'largest difference between BN(Wu + b) and BN(Wu): <b>' + Math.max.apply(null, a.map(function (v, i) { return Math.abs(v - c[i]); })).toExponential(1) + '</b> · without BN the bias moves every output.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch24: convolutional BN
  W.convBN = function (host) {
    Book.head(host, 'One statistic per feature map, over images and locations',
      'A minibatch of m = 2 images, each with 2 feature maps of 4×4 locations (p = q = 4). The maps have a brightness gradient across locations. Compare per-feature-map BN (the paper) with normalizing each location separately.');
    const st = { mode: 'map', seed: 1 }, ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'normalize', [{ value: 'raw', label: 'nothing (raw)' }, { value: 'map', label: 'per feature map (paper §3.2)' }, { value: 'loc', label: 'per location' }], st.mode, function (v) { st.mode = v; draw(); });
    Book.button(ctl, 'New batch', function () { st.seed++; draw(); });
    const view = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(st.seed * 9), m = 2, C = 2, P = 4, base = [3, -1], spread = [1.5, 0.4];
      const X = [];   // X[n][c][i][j]
      for (let n = 0; n < m; n++) { X.push([]); for (let c = 0; c < C; c++) { const g = []; for (let i = 0; i < P; i++) { g.push([]); for (let j = 0; j < P; j++) g[i].push(base[c] + spread[c] * (0.6 * (j - 1.5) + r.normal(0, 1))); } X[n].push(g); } }
      const Y = X.map(function (img) { return img.map(function (g) { return g.map(function (row) { return row.slice(); }); }); });
      const statsTxt = [];
      for (let c = 0; c < C; c++) {
        if (st.mode === 'map') {
          const vals = []; X.forEach(function (img) { img[c].forEach(function (row) { row.forEach(function (v) { vals.push(v); }); }); });
          const s = T.mv(vals), sd = Math.sqrt(s.var + 1e-5); statsTxt.push('map ' + (c + 1) + ': μ = ' + f(s.mean, 2) + ', σ = ' + f(sd, 2) + ' from m′ = ' + vals.length + ' values');
          Y.forEach(function (img) { img[c] = img[c].map(function (row) { return row.map(function (v) { return (v - s.mean) / sd; }); }); });
        } else if (st.mode === 'loc') {
          for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) { const vals = X.map(function (img) { return img[c][i][j]; }), s = T.mv(vals), sd = Math.sqrt(s.var + 1e-5); Y.forEach(function (img, n) { img[c][i][j] = (vals[n] - s.mean) / sd; }); }
          statsTxt.push('map ' + (c + 1) + ': 16 separate (μ, σ) pairs, each from only m = 2 values');
        }
      }
      view.innerHTML = '';
      const cell = 22, gap = 18, Wd = m * C * (P * cell + gap) + 40, svg = S('svg', { viewBox: '0 0 ' + Wd + ' 150', style: 'max-width:' + Wd + 'px' }, view);
      let ox = 10;
      for (let n = 0; n < m; n++) for (let c = 0; c < C; c++) {
        S('text', { x: ox, y: 14, 'font-size': 11.5, class: 'svg-text', text: 'image ' + (n + 1) + ', map ' + (c + 1) }, svg);
        for (let i = 0; i < P; i++) for (let j = 0; j < P; j++) {
          const v = Y[n][c][i][j], t = Book.clamp(v / 3, -1, 1);
          S('rect', { x: ox + j * cell, y: 22 + i * cell, width: cell - 1, height: cell - 1, fill: t >= 0 ? 'rgba(192,57,43,' + t + ')' : 'rgba(43,108,176,' + (-t) + ')', stroke: 'var(--line)' }, svg);
          S('text', { x: ox + j * cell + cell / 2, y: 22 + i * cell + cell / 2 + 3.5, 'text-anchor': 'middle', 'font-size': 8, class: 'svg-text', text: f(v, 1) }, svg);
        }
        ox += P * cell + gap + (c === C - 1 ? 14 : 0);
      }
      out.innerHTML = (st.mode === 'raw' ? 'raw values: map 1 around 3, map 2 around −1, both brighter to the right.' : statsTxt.join(' · ')) + (st.mode === 'map' ? ' · the left-to-right gradient (a real feature of the images) survives.' : st.mode === 'loc' ? ' · every location is forced to ±1 across the two images: the left-to-right gradient is erased, and the same pattern is treated differently at different places.' : '');
    }
    draw();
  };

  // ------------------------------------------------------------------ ch25: scale invariance, checked by autograd
  W.scaleInvariance = function (host) {
    Book.head(host, 'Scale the weights: same output, same input-gradient, 1/a weight-gradient',
      'A layer with 5 inputs and 3 outputs, a minibatch of m = 8, BN on its outputs and a fixed loss ℓ = Σ c·y. Scale W by a and compare with a = 1, all computed by the shared autograd.');
    const st = { lg: 0.5 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'scale a', -1, 1.5, 0.05, st.lg, function (v) { st.lg = v; draw(); }, function (v) { return f(Math.pow(10, v), 2); });
    const tab = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    const r = ML.rng(25), U0 = Array.from({ length: 8 }, function () { return Array.from({ length: 5 }, function () { return r.normal(0, 1); }); });
    const W0 = Array.from({ length: 5 }, function () { return Array.from({ length: 3 }, function () { return r.normal(0, 1); }); }), Cc = NN.tensor(Array.from({ length: 8 }, function () { return [r.normal(0, 1), r.normal(0, 1), r.normal(0, 1)]; }));
    function run(a) {
      const U = NN.tensor(U0, true), Wt = NN.tensor(W0.map(function (row) { return row.map(function (v) { return a * v; }); }), true);
      const Y = NN.batchNorm(NN.matmul(U, Wt), NN.full(1, 3, 1, true), NN.zeros(1, 3, true), { eps: 0 }); NN.sum(NN.mul(Y, Cc)).backward();
      const nrm = function (g) { let s = 0; for (let i = 0; i < g.length; i++) s += g[i] * g[i]; return Math.sqrt(s); };
      return { y: Y.data, gu: U.grad, gw: nrm(Wt.grad), sd: Math.sqrt(Y.stats.var[0]) };
    }
    const ref = run(1);
    function draw() {
      const a = Math.pow(10, st.lg), q = run(a), md = function (x, y) { let m = 0; for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i] - y[i])); return m; };
      tab.innerHTML = table([
        ['σ_B of output 1', f(ref.sd, 4), f(q.sd, 4), '×' + f(q.sd / ref.sd, 3)],
        ['output y = BN((aW)u)', '…', '…', 'largest difference ' + md(ref.y, q.y).toExponential(1)],
        ['∂ℓ/∂u (the Jacobian at work)', '…', '…', 'largest difference ' + md(ref.gu, q.gu).toExponential(1)],
        ['‖∂ℓ/∂(aW)‖', f(ref.gw, 4), f(q.gw, 4), '×' + f(q.gw / ref.gw, 4) + ' (1/a = ' + f(1 / a, 4) + ')']], ['quantity', 'a = 1', 'a = ' + f(a, 2), 'ratio / difference']);
      out.innerHTML = 'a = <b>' + f(a, 2) + '</b>: the batch spread grows ×' + f(a, 2) + ' and cancels; outputs and ∂ℓ/∂u are unchanged to rounding error; the weight gradient is ×' + f(q.gw / ref.gw, 3) + ' = 1/a.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch25: weight growth is self-limiting
  W.weightGrowth = function (host) {
    Book.head(host, 'Big steps grow the weights, and growing weights shrink the steps',
      'A toy 3-layer BN sigmoid network trained with plain SGD at a large learning rate (3.0). Tracked for layer 1: the size of its weights ‖W‖, of their gradient ‖∂ℓ/∂W‖, and the relative step α‖∂ℓ/∂W‖/‖W‖.');
    const ctl = el('div', { class: 'controls' }, host), go = Book.button(ctl, 'Run again', function () { start(); }, 'primary');
    const g = el('div', { class: 'grid2' }, host), L = el('div', null, g), R = el('div', null, g), out = el('div', { class: 'out' }, host);
    const Rn = T.runner(); void go;
    function start() {
      const D = T.data(), net = new T.Net({ sizes: [64, 32, 32, 32, 10], act: 'sigmoid', bn: true, sd: 'xavier', seed: 5 }), lr = 3, opt = new NN.SGD(net.params, { lr: lr }), next = T.batcher(D.train.n, 60, 3);
      const rec = { w: [], g: [], rel: [] };
      const nrm = function (a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); };
      const job = { done: false, t: 0, run: function (ms) { const t0 = performance.now(); while (!job.done && performance.now() - t0 < ms) { net.step(D.train, next(), opt); job.t++; if (job.t % 10 === 0) { const w = nrm(net.W[0].data), gg = nrm(net.W[0].grad); rec.w.push([job.t, w]); rec.g.push([job.t, gg]); rec.rel.push([job.t, lr * gg / w]); } if (job.t >= 800) job.done = true; } return job.done; } };
      function draw() {
        if (!rec.w.length) return;
        T.lines(L, [{ name: '‖W‖', color: T.colors.on, marker: 'none', points: rec.w }], { xmin: 0, xmax: 800, ymin: 0, xticks: [0, 400, 800], xlabel: 'step', ylabel: '‖W₁‖', height: 200, width: 320 });
        T.lines(R, [{ name: '', color: '#c0392b', marker: 'none', points: rec.rel }], { xmin: 0, xmax: 800, ymin: 0, xticks: [0, 400, 800], xlabel: 'step', ylabel: 'relative step α‖∂ℓ/∂W‖/‖W‖', height: 200, width: 320, ydec: 2 });
        const a = rec.w[0], b = rec.w[rec.w.length - 1];
        out.innerHTML = 'step ' + job.t + ': ‖W₁‖ ' + f(a[1], 2) + ' → <b>' + f(b[1], 2) + '</b> · ‖∂ℓ/∂W₁‖ ' + f(rec.g[0][1], 3) + ' → <b>' + f(rec.g[rec.g.length - 1][1], 3) + '</b> · test accuracy (toy) <b>' + (100 * net.accuracy(D.test, 'batch', 300)).toFixed(0) + '%</b>. The weights grew, the relative steps shrank: the network calmed itself down.';
      }
      let last = 0; Rn.start([job], function () { if (performance.now() - last > 300) { last = performance.now(); draw(); } }, draw);
    }
    start();
  };

  // ------------------------------------------------------------------ ch26: Jacobian singular values
  function sv2(J) { const a = J[0][0], b = J[0][1], c = J[1][0], d = J[1][1], p = a * a + b * b, q = a * c + b * d, s = c * c + d * d, tr = p + s, det = p * s - q * q, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det)); return [Math.sqrt(tr / 2 + disc), Math.sqrt(Math.max(0, tr / 2 - disc))]; }
  W.jacobianSV = function (host) {
    Book.head(host, 'How normalization pulls singular values towards 1',
      'Left: a random 2×2 Jacobian J maps the unit circle (inputs x̂ with unit covariance) to an ellipse; its half-axes are the singular values. Then the outputs are normalized per feature (BN) or fully whitened. Right: a gradient’s size after passing back through 30 such 16-unit layers.');
    const st = { scale: 1.6, seed: 2 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'weight scale', 0.3, 3, 0.05, st.scale, function (v) { st.scale = v; draw(); }, function (v) { return '×' + v.toFixed(2); });
    Book.button(ctl, 'New random J', function () { st.seed++; draw(); });
    const g = el('div', { class: 'grid2' }, host), L = el('div', { style: 'max-width:340px' }, g), R = el('div', null, g), out = el('div', { class: 'out' }, host);
    function rowNorm(J) { return J.map(function (row) { const n = Math.sqrt(row.reduce(function (s, v) { return s + v * v; }, 0)) || 1; return row.map(function (v) { return v / n; }); }); }
    function draw() {
      const r = ML.rng(st.seed * 31), J = [[r.normal(0, 1) * st.scale, r.normal(0, 1) * st.scale], [r.normal(0, 1) * st.scale, r.normal(0, 1) * st.scale]];
      const Jb = rowNorm(J);                                // per-feature normalization of z = Jx̂ (Cov[x̂] = I ⇒ Var[z_k] = ‖row k‖²)
      const ell = function (M) { const p = []; for (let i = 0; i <= 64; i++) { const t = i / 64 * 2 * Math.PI, x = [Math.cos(t), Math.sin(t)]; p.push([M[0][0] * x[0] + M[0][1] * x[1], M[1][0] * x[0] + M[1][1] * x[1]]); } return p; };
      const lim = Math.max(2, 1.1 * sv2(J)[0]);
      T.plot(L, [{ pts: ell([[1, 0], [0, 1]]), color: '#2f855a', line: true, r: 0, dash: '4 3' }, { pts: ell(J), color: '#c0392b', line: true, r: 0 }, { pts: ell(Jb), color: T.colors.on, line: true, r: 0, width: 2.6 }], { xmin: -lim, xmax: lim, ymin: -lim, ymax: lim, width: 320, height: 320 });
      T.legend(L, [{ name: 'raw J', color: '#c0392b' }, { name: 'after BN (per feature)', color: T.colors.on }, { name: 'whitened: circle (orthogonal)', color: '#2f855a' }]);
      // depth experiment, 16 units
      const d = 16, rr = ML.rng(st.seed * 7 + 1), layers = [];
      for (let l = 0; l < 30; l++) layers.push(Array.from({ length: d }, function () { return Array.from({ length: d }, function () { return rr.normal(0, st.scale / Math.sqrt(d)); }); }));
      function back(norm) {
        let v = Array.from({ length: d }, function () { return rr.normal(0, 1); }); const n0 = Math.sqrt(ML.dot(v, v)); v = v.map(function (x) { return x / n0; });
        const pts = [[0, 0]];
        for (let l = 29; l >= 0; l--) { const M = norm ? rowNorm(layers[l]) : layers[l]; v = Array.from({ length: d }, function (_, j) { let s = 0; for (let i = 0; i < d; i++) s += M[i][j] * v[i]; return s; }); pts.push([30 - l, Math.log10(Math.sqrt(ML.dot(v, v)) || 1e-30)]); }
        return pts;
      }
      const raw = back(false), bn = back(true);
      T.lines(R, [{ name: 'no normalization', color: '#c0392b', marker: 'none', points: raw.map(function (p) { return [p[0], Book.clamp(p[1], -8, 8)]; }) }, { name: 'per-feature normalization', color: T.colors.on, marker: 'none', points: bn }],
        { xmin: 0, xmax: 30, ymin: -8, ymax: 8, yticks: 4, xticks: [0, 10, 20, 30], xlabel: 'layers passed (backwards)', ylabel: 'gradient size (log₁₀)', height: 260, yfmt: function (v) { return '10^' + v; } });
      const s1 = sv2(J), s2 = sv2(Jb);
      out.innerHTML = 'singular values: raw J <b>' + f(s1[0], 2) + ', ' + f(s1[1], 2) + '</b> · after per-feature normalization <b>' + f(s2[0], 2) + ', ' + f(s2[1], 2) + '</b> (squares average ' + f((s2[0] * s2[0] + s2[1] * s2[1]) / 2, 3) + ' = 1 exactly) · fully whitened: 1, 1. After 30 layers the gradient size is ×10^' + f(raw[30][1], 1) + ' raw, ×10^' + f(bn[30][1], 1) + ' normalized.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch27: shuffling and BN noise
  W.shuffleDemo = function (host) {
    Book.head(host, 'BN’s noise needs changing batch-mates',
      'A toy data set of 16 values, minibatches of 4, 12 epochs. Each dot: the normalized value x̂ that example #1 (value 1.0) receives in that epoch. “Fixed groups”: the same 4 examples always form a batch; “reshuffled”: new random groups every epoch.');
    const st = { mode: 'shuffle' }, ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'minibatches', [{ value: 'fixed', label: 'fixed groups' }, { value: 'shuffle', label: 'reshuffled every epoch' }], st.mode, function (v) { st.mode = v; draw(); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    const r0 = ML.rng(61), data = [1].concat(Array.from({ length: 15 }, function () { return +(r0.normal(0, 1)).toFixed(2); }));
    function draw() {
      const r = ML.rng(5), pts = []; let order = ML.range(16);
      for (let e = 1; e <= 12; e++) {
        if (st.mode === 'shuffle') order = r.shuffle(order.slice());
        const pos = order.indexOf(0), start = Math.floor(pos / 4) * 4, b = order.slice(start, start + 4).map(function (i) { return data[i]; });
        const s = T.mv(b); pts.push([e, (1 - s.mean) / Math.sqrt(s.var + 1e-5)]);
      }
      T.lines(view, [{ name: '', color: st.mode === 'fixed' ? '#c0392b' : T.colors.on, points: pts, r: 4 }], { xmin: 1, xmax: 12, ymin: -2, ymax: 2, xticks: [1, 4, 8, 12], xlabel: 'epoch', ylabel: 'x̂ of example #1', height: 210 });
      const sd = Math.sqrt(T.mv(pts.map(function (p) { return p[1]; })).var);
      out.innerHTML = 'spread of example #1’s x̂ across epochs: <b>' + f(sd, 3) + '</b>' + (st.mode === 'fixed' ? ' — zero: the network sees the exact same normalized example every epoch, so BN adds no noise.' : ' — each epoch it looks a little different, like a mild, data-dependent dropout.');
    }
    draw();
  };

  // ------------------------------------------------------------------ ch27: regularization toy
  W.regToy = function (host) {
    Book.head(host, 'Overfitting a small training set: plain, dropout, BN',
      'Three ReLU networks (64-128-128-10) train on only 300 noisy glyphs for 2,000 steps, then are tested on 600 new ones. All three reach 100% on their training glyphs; what differs is how well they do on new ones (toy).');
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    const tr = T.glyphs(300, 31, { flip: 0.05, drop: 0.12 }), te = T.glyphs(600, 32, { flip: 0.05, drop: 0.12 });
    const specs = [['plain', false, 0, '#718096'], ['dropout 0.4 (no BN)', false, 0.4, '#b7791f'], ['BN', true, 0, T.colors.on]];
    const jobs = specs.map(function (s) { return T.job({ data: { train: tr, test: te }, net: { sizes: [64, 128, 128, 10], act: 'relu', bn: s[1], drop: s[2], sd: 'he', seed: 5 }, lr: 0.02, momentum: 0.9, bs: 30, steps: 2000, every: 100 }); });
    function draw() {
      T.lines(view, jobs.map(function (J, i) { return { name: specs[i][0], color: specs[i][3], marker: 'none', points: J.acc }; }), { xmin: 0, xmax: 2000, ymin: 0.5, ymax: 0.85, xticks: [0, 500, 1000, 1500, 2000], xlabel: 'training step', ylabel: 'test accuracy', height: 230, yfmt: function (v) { return Math.round(100 * v) + '%'; } });
      out.innerHTML = 'step ' + jobs[0].step + ' · test accuracy (toy), mean of the last 5 checks: ' + jobs.map(function (J, i) { const a = J.acc.slice(-5).map(function (p) { return p[1]; }); return specs[i][0] + ' <b>' + (100 * ML.mean(a)).toFixed(1) + '%</b>'; }).join(', ');
    }
    let last = 0; T.runner().start(jobs, function () { if (performance.now() - last > 400) { last = performance.now(); draw(); } }, draw);
  };
})();
