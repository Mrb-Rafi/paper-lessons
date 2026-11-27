/* Widgets for the hero and Part I (foundations) of the ResNet book. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, T = window.RNT, U = window.RNU;

  // ---------------------------------------------------------------- shared: the spiral task and a decision-map canvas
  T.spiralData = function () { if (!T._sp) T._sp = { train: T.spirals(1200, 5, 9), test: T.spirals(600, 5, 19) }; return T._sp; };
  const ARM = ['#2b6cb0', '#c0392b', '#2f855a', '#b7791f', '#6b46c1'];
  function decisionMap(cv, net, data) {
    const ctx = cv.getContext('2d'), N = 44, s = cv.width / N, grid = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) grid.push([-1.15 + 2.3 * (j + 0.5) / N, 1.15 - 2.3 * (i + 0.5) / N]);
    const lg = net.forward(NN.tensor(grid), null, 'eval'), K = lg.cols;
    ctx.clearRect(0, 0, cv.width, cv.height);
    for (let k = 0; k < grid.length; k++) {
      let b = 0; for (let j = 1; j < K; j++) if (lg.data[k * K + j] > lg.data[k * K + b]) b = j;
      ctx.fillStyle = U.rgba(ARM[b], 0.3); ctx.fillRect((k % N) * s, Math.floor(k / N) * s, s + 1, s + 1);
    }
    for (let i = 0; i < data.n; i += 3) {
      const px = (data.X[2 * i] + 1.15) / 2.3 * cv.width, py = (1.15 - data.X[2 * i + 1]) / 2.3 * cv.height;
      ctx.beginPath(); ctx.arc(px, py, 2.2, 0, 2 * Math.PI); ctx.fillStyle = ARM[data.y[i]]; ctx.fill();
    }
  }
  T.decisionMap = decisionMap;

  // ---------------------------------------------------------------- hero: plain vs residual, 30 layers each
  W.heroDemo = function (host) {
    Book.head(host, 'Thirty layers, with and without shortcuts',
      'Two networks with the same 30 weighted layers, the same widths and the same training learn to tell 5 interleaved spirals apart. The only difference: the right one adds its input back after every pair of layers. Watch the training error (toy, live in your browser).');
    const ctl = el('div', { class: 'controls' }, host);
    const go = Book.button(ctl, 'Race again ▶', function () { start(); }, 'primary');
    const grid = el('div', { class: 'grid2' }, host);
    const L = el('div', null, grid), R = el('div', null, grid);
    const cvs = [L, R].map(function (h, i) {
      el('div', { class: 'note', html: '<b style="color:' + (i ? U.C.res : U.C.plain) + '">' + (i ? 'residual-30' : 'plain-30') + '</b> · what it predicts' }, h);
      return el('canvas', { width: 220, height: 220, style: 'width:100%;max-width:220px;border-radius:8px;border:1px solid var(--line)' }, h);
    });
    const plot = el('div', null, host), out = U.out(host);
    let runner = null;
    function start() {
      if (runner) runner.stop();
      const d = T.spiralData();
      const jobs = [false, true].map(function (res) {
        const net = new T.MLP({ inDim: 2, width: 16, blocks: 14, residual: res, classes: 5, seed: 1 });
        return T.trainer(net, d.train, { steps: 1500, batch: 64, lr: 0.1, schedule: [[1000, 0.01]], seed: 11, every: 25 });
      });
      let last = 0;
      runner = T.run(jobs, function () {
        draw(jobs);
        if (jobs[0].step - last >= 300 || jobs.every(function (j) { return j.done(); })) { last = jobs[0].step; jobs.forEach(function (j, i) { decisionMap(cvs[i], j.net, d.train); }); }
      }, function () {
        const e = jobs.map(function (j) { return T.evaluate(j.net, d.train).err; });
        out.innerHTML = 'Final training error on all 1,200 points: plain-30 <b style="color:' + U.C.plain + '">' + (100 * e[0]).toFixed(1) + '%</b>, residual-30 <b style="color:' + U.C.res + '">' + (100 * e[1]).toFixed(1) + '%</b> (toy). The deeper plain network is worse at fitting the <em>training</em> data itself: that is the <b>degradation problem</b>, and the shortcuts remove it.';
      });
    }
    function draw(jobs) {
      U.lines(plot, jobs.map(function (j, i) { return { name: i ? 'residual-30' : 'plain-30', color: i ? U.C.res : U.C.plain, marker: 'none', points: j.hist.map(function (h) { return [h.step, 100 * h.err]; }) }; }),
        { xmin: 0, xmax: 1500, ymin: 0, ymax: 80, xticks: [0, 500, 1000, 1500], xlabel: 'training steps', ylabel: 'training error (%)', height: 200 });
      out.innerHTML = 'step ' + jobs[0].step + ' / 1500 · running training error: plain-30 <b>' + (100 * jobs[0].errAvg).toFixed(1) + '%</b>, residual-30 <b>' + (100 * jobs[1].errAvg).toFixed(1) + '%</b>';
    }
    start();
  };

  // ---------------------------------------------------------------- ch-network: depth multiplies the number of bends
  W.depthStack = function (host) {
    Book.head(host, 'Stacking layers multiplies what a network can draw',
      'Each layer here is two ReLU neurons that fold the line [0, 1] in half (a “tent”). Stack more layers and count the straight pieces of the result.');
    const ctl = el('div', { class: 'controls' }, host);
    let L = 3;
    Book.slider(ctl, 'layers', 1, 7, 1, L, function (v) { L = v; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    function layer(x) { return 2 * Math.max(0, x) - 4 * Math.max(0, x - 0.5); }   // tent: two ReLUs
    function draw() {
      const pts = []; for (let i = 0; i <= 1024; i++) { let x = i / 1024, y = x; for (let l = 0; l < L; l++) y = layer(y); pts.push([i / 1024, y]); }
      let pieces = 1; for (let i = 2; i < pts.length; i++) { const s1 = pts[i - 1][1] - pts[i - 2][1], s2 = pts[i][1] - pts[i - 1][1]; if (Math.sign(s1) !== Math.sign(s2)) pieces++; }
      U.lines(plot, [{ color: 'var(--accent)', marker: 'none', points: pts }], { xmin: 0, xmax: 1, ymin: 0, ymax: 1.05, xticks: [0, 0.5, 1], xlabel: 'input x', ylabel: 'output after ' + L + ' layer' + (L > 1 ? 's' : ''), height: 190, noLegend: true });
      out.innerHTML = 'layers: <b>' + L + '</b> · neurons used: <b>' + 2 * L + '</b> · straight pieces: <b>' + pieces + '</b> = 2<sup>' + L + '</sup>. A single layer with ' + 2 * L + ' neurons can make at most ' + (2 * L + 1) + ' pieces.';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-sgd: momentum and weight decay on a narrow valley
  W.momentumWd = function (host) {
    Book.head(host, 'Momentum and weight decay on a narrow valley',
      'The loss is a long, narrow bowl. Plain SGD zig-zags across it; momentum (the paper uses 0.9) builds up speed along it. Weight decay adds a pull towards zero, which moves the end point.');
    const st = { mom: 0.9, wd: 0, lr: 0.01 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'momentum μ', 0, 0.95, 0.05, st.mom, function (v) { st.mom = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'weight decay λ', 0, 2, 0.1, st.wd, function (v) { st.wd = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'learning rate η', 0.005, 0.06, 0.005, st.lr, function (v) { st.lr = v; draw(); }, function (v) { return v.toFixed(3); });
    const view = el('div', null, host), out = U.out(host);
    const a = 1, b = 25, c = [2, 1];                               // loss = a/2 (w1-2)^2 + b/2 (w2-1)^2
    function draw() {
      view.innerHTML = '';
      const Wd = 560, H = 260, X = function (w) { return 30 + (w + 0.5) / 3.5 * (Wd - 60); }, Y = function (w) { return H / 2 - (w - 1) * 90; };
      const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H }, view);
      for (let lv = 1; lv <= 6; lv++) { const L = lv * lv * 0.35; S('ellipse', { cx: X(c[0]), cy: Y(c[1]), rx: Math.sqrt(2 * L / a) / 3.5 * (Wd - 60), ry: Math.sqrt(2 * L / b) * 90, fill: 'none', stroke: 'var(--line)', 'stroke-width': 1.4 }, svg); }
      // run
      let w = [-0.3, 2.2], v = [0, 0]; const path = [w.slice()];
      for (let t = 0; t < 80; t++) {
        const g = [a * (w[0] - c[0]) + st.wd * w[0], b * (w[1] - c[1]) + st.wd * w[1]];
        v = [st.mom * v[0] + g[0], st.mom * v[1] + g[1]]; w = [w[0] - st.lr * v[0], w[1] - st.lr * v[1]];
        path.push(w.slice()); if (!isFinite(w[0]) || Math.abs(w[1]) > 50) break;
      }
      const ok = path.every(function (p) { return Math.abs(p[1]) < 50; });
      S('polyline', { points: path.filter(function (p) { return Math.abs(p[1]) < 4; }).map(function (p) { return X(p[0]) + ',' + Y(p[1]); }).join(' '), fill: 'none', stroke: U.C.res, 'stroke-width': 2 }, svg);
      path.forEach(function (p, i) { if (i % 4 === 0 && Math.abs(p[1]) < 4) S('circle', { cx: X(p[0]), cy: Y(p[1]), r: 2.4, fill: U.C.res }, svg); });
      S('circle', { cx: X(c[0]), cy: Y(c[1]), r: 5, fill: U.C.green }, svg);
      S('text', { x: X(c[0]) + 8, y: Y(c[1]) - 8, 'font-size': 12, class: 'svg-text', text: 'lowest loss' }, svg);
      const eq = [a * c[0] / (a + st.wd), b * c[1] / (b + st.wd)];
      if (st.wd > 0) { S('circle', { cx: X(eq[0]), cy: Y(eq[1]), r: 5, fill: 'none', stroke: U.C.gold, 'stroke-width': 2.5 }, svg); S('text', { x: X(eq[0]) - 8, y: Y(eq[1]) + 18, 'text-anchor': 'end', 'font-size': 12, class: 'svg-text', text: 'with decay' }, svg); }
      S('line', { x1: X(0), x2: X(0), y1: 6, y2: H - 6, stroke: 'var(--muted)', 'stroke-dasharray': '3 3' }, svg);
      S('text', { x: X(0) + 4, y: 16, 'font-size': 11, class: 'svg-text', text: 'w₁ = 0' }, svg);
      const last = path[path.length - 1], dist = Math.hypot(last[0] - eq[0], last[1] - eq[1]);
      out.innerHTML = ok ? 'after 80 steps: w = (' + T.f(last[0], 2) + ', ' + T.f(last[1], 2) + ') · distance to its target: <b>' + T.f(dist, 3) + '</b>' + (st.wd > 0 ? ' · weight decay moves the target from (2, 1) to (' + T.f(eq[0], 2) + ', ' + T.f(eq[1], 2) + ')' : '') : '<span class="bad">diverged</span>: the steep direction (curvature 25) needs η·25 &lt; 2(1 + μ).';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-backprop: a product of factors
  W.chainProduct = function (host) {
    Book.head(host, 'A gradient through many layers is a product',
      'Going backwards, each layer multiplies the gradient by its own local slope. Set the typical slope and the depth, and watch the gradient that reaches layer 1.');
    const st = { f: 0.8, L: 30, jitter: 0.3 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'typical factor per layer', 0.5, 1.5, 0.01, st.f, function (v) { st.f = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'layers', 2, 60, 1, st.L, function (v) { st.L = v; draw(); });
    Book.slider(ctl, 'randomness', 0, 0.6, 0.05, st.jitter, function (v) { st.jitter = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = U.out(host);
    function draw() {
      const r = ML.rng(5), pts = []; let g = 1; pts.push([st.L, 1]);
      for (let l = st.L; l >= 1; l--) { g *= st.f * Math.exp(r.normal(0, st.jitter)); pts.push([l - 1, g]); }
      U.logLines(plot, [{ color: 'var(--accent)', points: pts.reverse(), r: 2.6 }], { ymin: 1e-6, ymax: 1e6, xmin: 0, xmax: st.L, xticks: [0, Math.round(st.L / 2), st.L], xlabel: 'layer (loss is at the right)', ylabel: 'gradient size', height: 220, noLegend: true });
      const gl = pts[0][1];
      out.innerHTML = 'gradient arriving at the first layer: <b>' + (gl < 1e-3 || gl > 1e3 ? gl.toExponential(1) : T.f(gl, 3)) + '</b> × the gradient at the top. ' + (gl < 0.05 ? '<span class="bad">vanishing</span>: the lowest layers barely learn.' : gl > 20 ? '<span class="bad">exploding</span>: one step throws the lowest layers far away.' : '<span class="ok">healthy</span>.') + ' Every factor slightly below 1 is enough: 0.9<sup>50</sup> ≈ 0.005.';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-training: underfitting vs overfitting (polynomial fits)
  W.fitCurves = function (host) {
    Book.head(host, 'Training error and test error tell different stories',
      'Fit curves of increasing flexibility to 12 noisy training points. Training error always falls; error on new test points falls, then rises again (overfitting). Degradation, later, will be different: there the <em>training</em> error rises.');
    let deg = 3;
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'flexibility (polynomial degree)', 0, 11, 1, deg, function (v) { deg = v; draw(); });
    const g2 = el('div', { class: 'grid2' }, host), fit = el('div', null, g2), curve = el('div', null, g2), out = U.out(host);
    const r = ML.rng(3), f = function (x) { return Math.sin(2.5 * x); };
    const tr = [], te = [];
    for (let i = 0; i < 12; i++) { const x = -1 + 2 * (i + 0.5) / 12 + r.normal(0, 0.03); tr.push([x, f(x) + r.normal(0, 0.25)]); }
    for (let i = 0; i < 200; i++) { const x = r.uniform(-1, 1); te.push([x, f(x) + r.normal(0, 0.25)]); }
    function solve(A, bv) {                                      // Gaussian elimination with partial pivoting
      const n = bv.length, M = A.map(function (row, i) { return row.concat([bv[i]]); });
      for (let c = 0; c < n; c++) {
        let p = c; for (let i = c + 1; i < n; i++) if (Math.abs(M[i][c]) > Math.abs(M[p][c])) p = i;
        const tmp = M[c]; M[c] = M[p]; M[p] = tmp;
        for (let i = c + 1; i < n; i++) { const k = M[i][c] / M[c][c]; for (let j = c; j <= n; j++) M[i][j] -= k * M[c][j]; }
      }
      const x = new Array(n).fill(0);
      for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
      return x;
    }
    function polyFit(d) {
      const n = d + 1, A = ML.mat(n, n), bv = new Array(n).fill(0);
      tr.forEach(function (p) { for (let i = 0; i < n; i++) { bv[i] += Math.pow(p[0], i) * p[1]; for (let j = 0; j < n; j++) A[i][j] += Math.pow(p[0], i + j); } });
      for (let i = 0; i < n; i++) A[i][i] += 1e-9;
      const cf = solve(A, bv);
      return function (x) { let s = 0; for (let i = 0; i < n; i++) s += cf[i] * Math.pow(x, i); return s; };
    }
    function mse(fn, d) { return d.reduce(function (a, p) { return a + (fn(p[0]) - p[1]) * (fn(p[0]) - p[1]); }, 0) / d.length; }
    const curves = { tr: [], te: [] };
    for (let d = 0; d <= 11; d++) { const fn = polyFit(d); curves.tr.push([d, mse(fn, tr)]); curves.te.push([d, Math.min(mse(fn, te), 3)]); }
    function draw() {
      const fn = polyFit(deg), pts = [];
      for (let i = 0; i <= 200; i++) { const x = -1 + 2 * i / 200; pts.push([x, Math.max(-2, Math.min(2, fn(x)))]); }
      U.lines(fit, [{ name: 'fitted curve', color: 'var(--accent)', marker: 'none', points: pts }, { name: 'training points', color: U.C.plain, noline: true, points: tr }],
        { xmin: -1, xmax: 1, ymin: -2, ymax: 2, xticks: [-1, 0, 1], xlabel: 'x', ylabel: 'y', width: 340, height: 220 });
      U.lines(curve, [{ name: 'training error', color: U.C.res, points: curves.tr, dash: '5 4' }, { name: 'test error', color: U.C.plain, points: curves.te }, { color: U.C.gold, points: [[deg, curves.te[deg][1]]], r: 6, noline: true }],
        { xmin: 0, xmax: 11, ymin: 0, ymax: 1, xticks: [0, 3, 6, 9, 11], xlabel: 'degree', ylabel: 'mean squared error', width: 340, height: 220 });
      out.innerHTML = 'degree ' + deg + ': training error <b>' + T.f(curves.tr[deg][1], 3) + '</b>, test error <b>' + T.f(curves.te[deg][1], 3) + '</b> · ' +
        (deg <= 1 ? 'underfitting: too simple to follow the data.' : curves.te[deg][1] > 2 * Math.min.apply(null, curves.te.map(function (p) { return p[1]; })) ? 'overfitting: the curve chases the noise in the 12 points.' : 'a good balance.');
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-init: signal size through 50 ReLU layers
  W.initDepth = function (host) {
    Book.head(host, 'Starting weights decide whether a signal survives 50 layers',
      'Random weights with spread σ = scale × √(2/n) (n = 64 inputs per neuron), each layer followed by ReLU. Scale 1 is the initialization of He et al. (2015) that ResNet uses. Watch the size of the activations, layer by layer.');
    let sc = 1, init = 'he';
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'scale', 0.5, 1.5, 0.05, sc, function (v) { sc = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.select(ctl, 'rule', [{ value: 'he', label: 'He: σ² = 2/n' }, { value: 'xavier', label: 'Xavier: σ² = 1/n' }], init, function (v) { init = v; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    function draw() {
      const r = ML.rng(4), n = 64, B = 64, depth = 50;
      let h = ML.randn(B, n, r, 1); const pts = [[0, std(h)]];
      function std(M) { let s = 0, c = 0; M.forEach(function (row) { row.forEach(function (v) { s += v * v; c++; }); }); return Math.sqrt(s / c); }
      for (let l = 1; l <= depth; l++) {
        const sd = sc * Math.sqrt((init === 'he' ? 2 : 1) / n), Wm = ML.randn(n, n, r, sd);
        h = ML.matmul(h, Wm).map(function (row) { return row.map(function (v) { return Math.max(0, v); }); });
        pts.push([l, Math.max(1e-6, std(h))]);
      }
      U.logLines(plot, [{ color: 'var(--accent)', points: pts, r: 2.4 }], { ymin: 1e-6, ymax: 1e6, xmin: 0, xmax: 50, xticks: [0, 10, 20, 30, 40, 50], xlabel: 'layer', ylabel: 'activation size (rms)', height: 220, noLegend: true });
      const end = pts[50][1];
      out.innerHTML = 'size after 50 layers: <b>' + (end < 1e-2 || end > 1e2 ? end.toExponential(1) : T.f(end, 2)) + '</b> (it started at 1). Each layer multiplies the size by about ' + T.f(sc * Math.sqrt(init === 'he' ? 1 : 0.5), 3) + ', so 50 layers multiply it by ≈ ' + Math.pow(sc * Math.sqrt(init === 'he' ? 1 : 0.5), 50).toExponential(1) + '.';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-bn: BN recap with the shared NN.batchNorm
  W.bnRecap = function (host) {
    Book.head(host, 'Batch normalization re-centres and rescales every channel',
      'Eight examples in a minibatch, three channels. Shift and stretch the inputs as you like: after BN each channel has mean β and spread γ again (computed by the collection’s <code>NN.batchNorm</code>).');
    const st = { shift: 3, stretch: 4, gamma: 1, beta: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'input shift', -6, 6, 0.5, st.shift, function (v) { st.shift = v; draw(); });
    Book.slider(ctl, 'input stretch', 0.2, 8, 0.2, st.stretch, function (v) { st.stretch = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'γ', 0, 2, 0.1, st.gamma, function (v) { st.gamma = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'β', -2, 2, 0.1, st.beta, function (v) { st.beta = v; draw(); }, function (v) { return v.toFixed(1); });
    const g2 = el('div', { class: 'grid2' }, host), A = el('div', null, g2), B = el('div', null, g2), out = U.out(host);
    const base = ML.randn(8, 3, ML.rng(8), 1);
    function draw() {
      const X = NN.tensor(base.map(function (row, i) { return row.map(function (v, j) { return st.shift * (j + 1) / 3 + st.stretch * v * (1 + j) / 3; }); }));
      const y = NN.batchNorm(X, NN.full(1, 3, st.gamma), NN.full(1, 3, st.beta));
      function dots(host2, M, title) {
        const series = [0, 1, 2].map(function (j) { return { name: 'channel ' + (j + 1), color: U.stage[j + 1], noline: true, points: ML.range(8).map(function (i) { return [M.get(i, j), j + 1]; }) }; });
        U.lines(host2, series, { xmin: -12, xmax: 12, ymin: 0.5, ymax: 3.5, yticks: 3, yfmt: function () { return ''; }, xticks: [-12, -6, 0, 6, 12], xlabel: title, width: 320, height: 150 });
      }
      dots(A, X, 'values before BN'); dots(B, y, 'values after BN');
      const m = y.stats.mean.map(function (v) { return T.f(v, 2); }).join(', ');
      let s2 = [0, 0, 0], mu2 = [0, 0, 0]; for (let i = 0; i < 8; i++) for (let j = 0; j < 3; j++) mu2[j] += y.get(i, j) / 8;
      for (let i = 0; i < 8; i++) for (let j = 0; j < 3; j++) s2[j] += (y.get(i, j) - mu2[j]) * (y.get(i, j) - mu2[j]) / 8;
      out.innerHTML = 'batch means before BN: (' + m + ') · after BN: mean (' + mu2.map(function (v) { return T.f(v, 2); }).join(', ') + '), spread (' + s2.map(function (v) { return T.f(Math.sqrt(v), 2); }).join(', ') + ')';
    }
    draw();
  };
})();
