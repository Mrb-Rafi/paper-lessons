/* Toy library of the Batch Normalization book (lesson 4), exposed as window.BNT.
   - glyphs(): 8×8 binary "pixel digits" 0–9, our small stand-in for the paper's binarized 28×28 MNIST (§4.1)
   - Net: a multilayer perceptron on the shared autograd (NN), with or without Batch Normalization before every
     nonlinearity, as in the paper's §3.2 (z = g(BN(Wu)), bias dropped). BN itself is the shared NN.batchNorm
     (Algorithm 1 forward, the chain-rule equations backward); running statistics follow PyTorch v2.5.0
     (momentum 0.1, unbiased variance) or Algorithm 2 (average over many training minibatches).
   - job(): time-sliced SGD-with-momentum training that records accuracy, loss and percentiles of one sigmoid input
   - runner(), lines(), hist(), legend(), pct(): small helpers for the widgets.
   Everything is seeded, so the pictures in the book are reproducible. */
(function () {
  'use strict';
  const ML = window.ML, NN = window.NN;
  const T = window.BNT = {};

  // ------------------------------------------------------------------ formatting
  T.f = function (x, d) {                                   // number with a real minus sign
    if (!isFinite(x)) return isNaN(x) ? 'NaN' : (x > 0 ? '∞' : '−∞');
    d = d == null ? 3 : d; const s = Number(x).toFixed(d);
    return s.charAt(0) === '-' ? (/^-0\.?0*$/.test(s) ? s.slice(1) : '−' + s.slice(1)) : s;
  };
  T.pct = function (sorted, p) {                            // p-th percentile (0–100) of an ascending array
    if (!sorted.length) return NaN;
    const r = (p / 100) * (sorted.length - 1), i = Math.floor(r), f = r - i;
    return i + 1 < sorted.length ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i];
  };
  T.sorted = function (a) { return Array.from(a).sort(function (x, y) { return x - y; }); };
  T.rgb = function (hex, a) { hex = hex.replace('#', ''); const n = parseInt(hex, 16); return 'rgba(' + (n >> 16 & 255) + ',' + (n >> 8 & 255) + ',' + (n & 255) + ',' + a + ')'; };
  T.colors = { off: '#c0392b', on: '#2b6cb0', sig: '#d53f8c', base: '#718096', x5: '#2b6cb0', x30: '#1f4e8c' };

  // ------------------------------------------------------------------ data: "glyphs"
  // A 5×7 bitmap font for the digits, placed at a random offset in an 8×8 grid; each stroke pixel is lost with
  // probability `drop`, and each pixel is flipped with probability `flip`. Pixels are 0 or 1, like binarized MNIST.
  const FONT = [
    '01110100011001110101110011000101110', '00100011000010000100001000010001110', '01110100010000100010001000100011111',
    '11111000100010000010000011000101110', '00010001100101010010111110001000010', '11111100001111000001000011000101110',
    '00110010001000011110100011000101110', '11111000010001000100010000100001000', '01110100011000101110100011000101110',
    '01110100011000101111000010001001100'];
  T.FONT = FONT;
  T.glyphs = function (n, seed, o) {
    o = o || {}; const flip = o.flip == null ? 0.03 : o.flip, drop = o.drop == null ? 0.08 : o.drop;
    const r = ML.rng(seed || 7), D = 64, X = new Float64Array(n * D), y = new Int32Array(n);
    for (let k = 0; k < n; k++) {
      const c = r.int(10), ox = r.int(4), oy = r.int(2); y[k] = c;
      for (let i = 0; i < 7; i++) for (let j = 0; j < 5; j++) if (FONT[c][i * 5 + j] === '1' && r.next() >= drop) X[k * D + (i + oy) * 8 + j + ox] = 1;
      for (let p = 0; p < D; p++) if (r.next() < flip) X[k * D + p] = 1 - X[k * D + p];
    }
    return { X: X, y: y, n: n, D: D, K: 10 };
  };
  T.data = function () {                                    // the shared train/test split (cached)
    if (!T._data) T._data = { train: T.glyphs(3000, 11), test: T.glyphs(600, 12) };
    return T._data;
  };
  T.rows = function (d, idx) {                              // gather rows of d.X into a Tensor
    const D = d.D, X = new NN.Tensor(idx.length, D);
    for (let a = 0; a < idx.length; a++) { const k = idx[a]; for (let j = 0; j < D; j++) X.data[a * D + j] = d.X[k * D + j]; }
    return X;
  };

  // ------------------------------------------------------------------ the network
  // cfg: {sizes:[64,h1,…,10], act:'sigmoid'|'relu', bn:bool, seed, sd:'small'|'xavier'|'he'|number, mom:0.1 (running stats)}
  T.Net = function (cfg) {
    this.cfg = cfg; const s = cfg.sizes, r = ML.rng(cfg.seed || 5); this.L = s.length - 1;
    this.W = []; this.b = []; this.g = []; this.be = []; this.run = []; this.params = [];
    for (let l = 0; l < this.L; l++) {
      const sd = typeof cfg.sd === 'number' ? cfg.sd : cfg.sd === 'he' ? Math.sqrt(2 / s[l]) : cfg.sd === 'xavier' ? 1 / Math.sqrt(s[l]) : 0.1;
      const W = NN.param(s[l], s[l + 1], r, l === this.L - 1 && cfg.sd === 'small' ? 0.1 : sd);
      this.W.push(W); this.params.push(W);
      const hidden = l < this.L - 1;
      if (hidden && cfg.bn) {                                 // §3.2: z = g(BN(Wu)), one γ, β per unit, no bias
        const g = NN.full(1, s[l + 1], 1, true), be = NN.zeros(1, s[l + 1], true);
        this.g.push(g); this.be.push(be); this.b.push(null); this.params.push(g, be);
        this.run.push({ mean: new Float64Array(s[l + 1]), var: new Float64Array(s[l + 1]).fill(1), n: 0 });
      } else {
        const b = NN.zeros(1, s[l + 1], true); this.b.push(b); this.params.push(b); this.g.push(null); this.be.push(null); this.run.push(null);
      }
    }
  };
  // mode: 'train' (batch statistics, and update the running averages), 'batch' (batch statistics only), 'eval' (running statistics)
  T.Net.prototype.forward = function (X, mode) {
    const c = this.cfg, act = c.act === 'relu' ? NN.relu : NN.sigmoid, rec = { x: [], z: [], h: [] };
    let h = X;
    for (let l = 0; l < this.L; l++) {
      let x = NN.matmul(h, this.W[l]);
      if (l === this.L - 1) { h = NN.add(x, this.b[l]); break; }
      let z;
      if (c.bn) {
        const R = this.run[l];
        z = NN.batchNorm(x, this.g[l], this.be[l], mode === 'eval' ? { training: false, running: R } : {});
        if (mode === 'train') {
          const m = x.rows, mom = c.mom == null ? 0.1 : c.mom;
          for (let j = 0; j < R.mean.length; j++) {
            R.mean[j] = (1 - mom) * R.mean[j] + mom * z.stats.mean[j];
            R.var[j] = (1 - mom) * R.var[j] + mom * z.stats.var[j] * m / (m - 1);
          }
          R.n++;
        }
      } else { x = NN.add(x, this.b[l]); z = x; }
      rec.x.push(x); rec.z.push(z);
      h = act(z); rec.h.push(h);
      if (c.drop && mode === 'train') { if (!this.drng) this.drng = ML.rng((c.seed || 5) + 99); h = NN.dropout(h, c.drop, this.drng, true); }
    }
    this.rec = rec;
    return h;
  };
  // Algorithm 2, lines 8–10: E[x] ← E_B[μ_B], Var[x] ← m/(m−1)·E_B[σ²_B] over `nb` training minibatches of size m
  T.Net.prototype.population = function (d, nb, m, seed) {
    if (!this.cfg.bn) return;
    const r = ML.rng(seed || 99), acc = this.run.map(function (R) { return R && { mean: new Float64Array(R.mean.length), var: new Float64Array(R.mean.length) }; });
    for (let k = 0; k < nb; k++) {
      const idx = []; for (let i = 0; i < m; i++) idx.push(r.int(d.n));
      this.forward(T.rows(d, idx), 'batch');
      for (let l = 0; l < this.L - 1; l++) {
        const st = this.rec.z[l].stats;
        for (let j = 0; j < st.mean.length; j++) { acc[l].mean[j] += st.mean[j] / nb; acc[l].var[j] += st.var[j] / nb; }
      }
    }
    for (let l = 0; l < this.L - 1; l++) for (let j = 0; j < acc[l].mean.length; j++) { this.run[l].mean[j] = acc[l].mean[j]; this.run[l].var[j] = acc[l].var[j] * m / (m - 1); }
  };
  T.Net.prototype.accuracy = function (d, mode, limit) {
    const n = Math.min(d.n, limit || d.n), bs = 200; let ok = 0;
    for (let s = 0; s < n; s += bs) {
      const idx = []; for (let i = s; i < Math.min(n, s + bs); i++) idx.push(i);
      const out = this.forward(T.rows(d, idx), mode || 'eval');
      for (let a = 0; a < idx.length; a++) {
        let best = 0; for (let j = 1; j < 10; j++) if (out.data[a * 10 + j] > out.data[a * 10 + best]) best = j;
        if (best === d.y[idx[a]]) ok++;
      }
    }
    return ok / n;
  };
  T.Net.prototype.step = function (d, idx, opt) {
    NN.zeroGrad(this.params);
    const y = Array.from(idx, function (k) { return d.y[k]; });
    const L = NN.softmaxCE(this.forward(T.rows(d, idx), 'train'), y);
    L.backward(); opt.step(); return L.item();
  };

  // ------------------------------------------------------------------ training jobs
  T.batcher = function (n, bs, seed) {
    const r = ML.rng(seed || 17); let order = r.shuffle(ML.range(n)), pos = 0;
    return function () { if (pos + bs > n) { order = r.shuffle(order); pos = 0; } const b = order.slice(pos, pos + bs); pos += bs; return b; };
  };
  // cfg: {net (cfg for T.Net), lr, momentum, bs, steps, every, probe:{layer, unit}, evalN, decay:{every, factor}}
  T.job = function (cfg) {
    const D = cfg.data || T.data(), net = new T.Net(cfg.net), bs = cfg.bs || 60, next = T.batcher(D.train.n, bs, cfg.seed || 17);
    const opt = new NN.SGD(net.params, { lr: cfg.lr, momentum: cfg.momentum == null ? 0.9 : cfg.momentum });
    const J = { net: net, opt: opt, step: 0, acc: [], tacc: [], loss: [], pct: [], done: false, diverged: false, cfg: cfg, best: 0 };
    const every = cfg.every || 50, probeIdx = ML.range(300).map(function (i) { return i; });
    let lsum = 0, lcnt = 0;
    function record() {
      const a = net.accuracy(D.test, 'eval', cfg.evalN || 600); J.acc.push([J.step, a]); J.best = Math.max(J.best, a);
      if (cfg.trainAcc) J.tacc.push([J.step, net.accuracy(D.train, 'eval', cfg.evalN || 600)]);
      if (lcnt) J.loss.push([J.step, lsum / lcnt]); lsum = 0; lcnt = 0;
      if (cfg.probe) {
        net.forward(T.rows(D.test, probeIdx), 'batch');
        const z = net.rec.z[cfg.probe.layer], v = [];
        for (let i = 0; i < z.rows; i++) v.push(z.data[i * z.cols + cfg.probe.unit]);
        const s = T.sorted(v); J.pct.push([J.step, T.pct(s, 15), T.pct(s, 50), T.pct(s, 85)]);
      }
    }
    record();
    J.stepOnce = function () {
      if (cfg.decay && J.step > 0 && J.step % cfg.decay.every === 0) opt.lr *= cfg.decay.factor;
      const L = net.step(D.train, next(), opt); J.step++;
      if (!isFinite(L) || L > 1e4) { J.diverged = true; J.done = true; J.acc.push([J.step, 0.1]); return; }
      lsum += L; lcnt++;
      if (J.step % every === 0) record();
      if (J.step >= cfg.steps) J.done = true;
    };
    J.run = function (ms) { const t0 = performance.now(); while (!J.done && performance.now() - t0 < ms) J.stepOnce(); return J.done; };
    J.runAll = function () { while (!J.done) J.stepOnce(); return J; };
    return J;
  };
  // a time-sliced runner: jobs have .run(ms) → done; tick() is called between slices so widgets can redraw
  // T.active counts runs in progress (verification scripts wait for it to reach 0).
  T.active = 0;
  T.runner = function () {
    let token = 0, live = false;
    function end() { if (live) { live = false; T.active--; } }
    return {
      start: function (jobs, tick, done) {
        const my = ++token; if (!live) { live = true; T.active++; }
        (function loop() {
          if (my !== token) return;
          const t0 = performance.now();
          for (let i = 0; i < jobs.length && performance.now() - t0 < 32; i++) if (!jobs[i].done) jobs[i].run(14);
          const all = jobs.every(function (j) { return j.done; });
          if (all) end();
          tick(); if (all) { if (done) done(); return; }
          setTimeout(loop, 16);
        })();
      },
      stop: function () { token++; end(); },
      busy: function () { return live; }
    };
  };

  // ------------------------------------------------------------------ drawing helpers
  const el = function (t, a, p) { return window.Book.el(t, a, p); }, S = function (t, a, p) { return window.Book.svg(t, a, p); };
  T.legend = function (host, items) {
    const L = el('div', { class: 'legend' }, host);
    items.forEach(function (it) { const s = el('span', null, L); el('i', { style: 'background:' + it.color + (it.dash ? ';opacity:.6' : '') }, s); s.appendChild(document.createTextNode(it.name)); });
    return L;
  };
  // line chart with a legend under it when there are 2+ named series
  T.lines = function (host, series, o) {
    o = o || {};
    const named = series.filter(function (s) { return s.name; });
    const useLegend = o.legend !== false && named.length >= 2;
    const ser = useLegend ? series.map(function (s) { return Object.assign({}, s, { name: null }); }) : series;
    const res = window.Book.chart.lines(host, ser, useLegend ? Object.assign({ mr: 20 }, o) : o);
    if (useLegend) T.legend(host, named.map(function (s) { return { name: s.name, color: s.color, dash: !!s.dash }; }));
    return res;
  };
  // histogram of values; o: {lo, hi, bins, color, width, height, title, curve:(x)=>y in [0,1], curveColor, marks:[{x,label,color}], ymax}
  T.hist = function (host, values, o) {
    o = o || {}; host.innerHTML = '';
    const W = o.width || 300, H = o.height || 150, m = { l: 8, r: 8, t: o.title ? 20 : 8, b: 22 };
    const lo = o.lo == null ? -4 : o.lo, hi = o.hi == null ? 4 : o.hi, nb = o.bins || 32, cnt = new Array(nb).fill(0);
    let out = 0;
    for (let i = 0; i < values.length; i++) {
      const v = values[i]; if (!isFinite(v)) { out++; continue; }
      const k = Math.floor((v - lo) / (hi - lo) * nb);
      if (k < 0) { cnt[0]++; out++; } else if (k >= nb) { cnt[nb - 1]++; out++; } else cnt[k]++;
    }
    const mx = o.ymax || Math.max.apply(null, cnt) || 1;
    const svg = S('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' }, host);
    const X = function (v) { return m.l + (v - lo) / (hi - lo) * (W - m.l - m.r); }, Y = function (c) { return H - m.b - Math.min(1, c / mx) * (H - m.t - m.b); };
    if (o.title) S('text', { x: m.l, y: 13, 'font-size': 12, 'font-weight': 700, class: 'svg-text', text: o.title }, svg);
    if (o.band) S('rect', { x: X(o.band[0]), y: m.t, width: Math.max(0, X(o.band[1]) - X(o.band[0])), height: H - m.t - m.b, fill: o.bandColor || 'var(--accent-soft)' }, svg);
    const bw = (W - m.l - m.r) / nb;
    cnt.forEach(function (c, k) { if (c) S('rect', { x: m.l + k * bw + 0.5, y: Y(c), width: Math.max(0.5, bw - 1), height: H - m.b - Y(c), fill: o.color || 'var(--accent)', opacity: 0.85 }, svg); });
    S('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: 'var(--muted)' }, svg);
    const ticks = o.ticks || [lo, (lo + hi) / 2, hi];
    ticks.forEach(function (t) { S('text', { x: X(t), y: H - 6, 'text-anchor': 'middle', 'font-size': 10.5, class: 'svg-text', text: T.f(t, Math.abs(t) < 10 && t % 1 ? 1 : 0) }, svg); });
    if (o.curve) {
      const pts = []; for (let i = 0; i <= 80; i++) { const v = lo + (hi - lo) * i / 80; pts.push(X(v) + ',' + (H - m.b - o.curve(v) * (H - m.t - m.b))); }
      S('polyline', { points: pts.join(' '), fill: 'none', stroke: o.curveColor || 'var(--muted)', 'stroke-width': 1.6, 'stroke-dasharray': '4 3' }, svg);
    }
    (o.marks || []).forEach(function (mk) {
      if (mk.x < lo || mk.x > hi) return;
      S('line', { x1: X(mk.x), x2: X(mk.x), y1: m.t, y2: H - m.b, stroke: mk.color || 'var(--ink)', 'stroke-width': 1.4, 'stroke-dasharray': mk.dash || '' }, svg);
      if (mk.label) S('text', { x: X(mk.x) + 3, y: m.t + 10, 'font-size': 10.5, class: 'svg-text', text: mk.label }, svg);
    });
    return { svg: svg, X: X, Y: Y, out: out, counts: cnt };
  };
  // draw an 8×8 glyph as SVG squares
  T.glyph = function (host, X, k, size) {
    size = size || 56; const svg = S('svg', { viewBox: '0 0 8 8', width: size, height: size, style: 'border:1px solid var(--line);border-radius:4px;background:var(--panel)' }, host);
    for (let p = 0; p < 64; p++) if (X[k * 64 + p] > 0.5) S('rect', { x: p % 8, y: Math.floor(p / 8), width: 1, height: 1, fill: 'var(--ink)' }, svg);
    return svg;
  };
  // 2-D plot with axes through the box; sets: [{pts:[[x,y]…], color, r, line, dash, width}], o: {xmin,xmax,ymin,ymax,width,height,xlabel,ylabel,grid}
  T.plot = function (host, sets, o) {
    o = o || {}; host.innerHTML = '';
    const W = o.width || 320, H = o.height || 320, m = { l: 34, r: 10, t: o.title ? 22 : 10, b: 30 };
    const svg = S('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' }, host);
    const X = function (v) { return m.l + (v - o.xmin) / (o.xmax - o.xmin) * (W - m.l - m.r); };
    const Y = function (v) { return H - m.b - (v - o.ymin) / (o.ymax - o.ymin) * (H - m.t - m.b); };
    if (o.title) S('text', { x: m.l, y: 14, 'font-size': 12, 'font-weight': 700, class: 'svg-text', text: o.title }, svg);
    S('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'none', stroke: 'var(--line)' }, svg);
    if (o.xmin < 0 && o.xmax > 0) S('line', { x1: X(0), x2: X(0), y1: m.t, y2: H - m.b, stroke: 'var(--line)' }, svg);
    if (o.ymin < 0 && o.ymax > 0) S('line', { x1: m.l, x2: W - m.r, y1: Y(0), y2: Y(0), stroke: 'var(--line)' }, svg);
    [o.xmin, o.xmax].forEach(function (v) { S('text', { x: X(v), y: H - m.b + 13, 'text-anchor': 'middle', 'font-size': 10.5, class: 'svg-text', text: T.f(v, v % 1 ? 1 : 0) }, svg); });
    [o.ymin, o.ymax].forEach(function (v) { S('text', { x: m.l - 4, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 10.5, class: 'svg-text', text: T.f(v, v % 1 ? 1 : 0) }, svg); });
    if (o.xlabel) S('text', { x: (W + m.l) / 2, y: H - 4, 'text-anchor': 'middle', 'font-size': 11.5, class: 'svg-text', text: o.xlabel }, svg);
    if (o.ylabel) S('text', { x: 11, y: (H - m.b + m.t) / 2, transform: 'rotate(-90 11 ' + (H - m.b + m.t) / 2 + ')', 'text-anchor': 'middle', 'font-size': 11.5, class: 'svg-text', text: o.ylabel }, svg);
    const inb = function (p) { return p[0] >= o.xmin && p[0] <= o.xmax && p[1] >= o.ymin && p[1] <= o.ymax; };
    sets.forEach(function (s) {
      if (s.line && s.pts.length > 1) S('polyline', { points: s.pts.map(function (p) { return X(Book.clamp(p[0], o.xmin, o.xmax)) + ',' + Y(Book.clamp(p[1], o.ymin, o.ymax)); }).join(' '), fill: 'none', stroke: s.color, 'stroke-width': s.width || 2, 'stroke-dasharray': s.dash || '', opacity: s.opacity || 1 }, svg);
      if (s.r !== 0) s.pts.forEach(function (p) { if (inb(p)) S('circle', { cx: X(p[0]), cy: Y(p[1]), r: s.r || 2.4, fill: s.color, opacity: s.opacity || 0.75 }, svg); });
    });
    return { svg: svg, X: X, Y: Y };
  };
  const Book = window.Book;
  // mean and (biased) variance of an array
  T.mv = function (a) { let m = 0; for (let i = 0; i < a.length; i++) m += a[i]; m /= a.length || 1; let v = 0; for (let i = 0; i < a.length; i++) v += (a[i] - m) * (a[i] - m); return { mean: m, var: v / (a.length || 1) }; };
  // column j of a Tensor as an array
  T.col = function (t, j) { const v = []; for (let i = 0; i < t.rows; i++) v.push(t.data[i * t.cols + j]); return v; };
})();
