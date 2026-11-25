/* Batch Normalization book, Part I (Foundations) widgets and the hero. Uses Book (core.js), ML, NN and BNT (bn.js). */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, T = window.BNT;
  const f = T.f, sig = ML.sigmoid || function (x) { return 1 / (1 + Math.exp(-x)); };

  // ------------------------------------------------------------------ hero: per-layer histograms, with vs without BN
  W.heroDemo = function (host) {
    Book.head(host, 'Same network, with and without Batch Normalization',
      'Two copies of one 3-layer sigmoid network learn to read 8×8 pixel digits (toy). Left: test accuracy. Right: the input to one sigmoid of the last hidden layer, as its 15th, 50th and 85th percentiles over the test glyphs. Without BN it drifts; with BN it stays put.');
    const ctl = el('div', { class: 'controls' }, host);
    const go = Book.button(ctl, 'Pause', function () { toggle(); }, 'primary');
    Book.button(ctl, 'Restart', function () { reset(); toggle(true); });
    const grid = el('div', { class: 'hgrid' }, host), pa = el('div', null, grid), pb = el('div', null, grid), pc = el('div', null, grid), out = el('div', { class: 'out' }, host);
    const R = T.runner(); let jobs, running = false;
    function cfg(bn) { return { net: { sizes: [64, 32, 32, 32, 10], act: 'sigmoid', bn: bn, sd: 'xavier', seed: 5 }, lr: 0.5, momentum: 0, bs: 60, steps: 2000, every: 25, probe: { layer: 2, unit: 3 } }; }
    function reset() { R.stop(); running = false; jobs = [T.job(cfg(false)), T.job(cfg(true))]; draw(); }
    function band(host, J, color, title) {
      const ser = [1, 2, 3].map(function (k) { return { name: '', color: color, marker: 'none', width: k === 2 ? 2.4 : 1.5, points: J.pct.map(function (p) { return [p[0], Book.clamp(p[k], -4, 4)]; }) }; });
      el('div', { class: 'note', html: '<b style="color:' + color + '">' + title + '</b>' }, host);
      const h = el('div', null, host);
      Book.chart.lines(h, ser, { xmin: 0, xmax: 2000, ymin: -4, ymax: 4, yticks: 4, xticks: [0, 1000, 2000], xlabel: 'training step', height: 230, width: 330, ydec: 0 });
      host.replaceChildren(host.firstChild, h);
    }
    function draw() {
      Book.chart.lines(pa, [{ name: '', color: T.colors.off, marker: 'none', dash: '6 4', points: jobs[0].acc }, { name: '', color: T.colors.on, marker: 'none', points: jobs[1].acc }],
        { xmin: 0, xmax: 2000, ymin: 0, ymax: 1, xticks: [0, 1000, 2000], xlabel: 'training step', ylabel: 'test accuracy', height: 230, width: 330, yfmt: function (v) { return Math.round(100 * v) + '%'; } });
      pb.innerHTML = ''; pc.innerHTML = '';
      band(pb, jobs[0], T.colors.off, 'Without BN: one sigmoid input'); band(pc, jobs[1], T.colors.on, 'With BN: one sigmoid input');
      const a0 = jobs[0].acc[jobs[0].acc.length - 1][1], a1 = jobs[1].acc[jobs[1].acc.length - 1][1];
      out.innerHTML = 'step <b>' + jobs[0].step + '</b> · test accuracy (toy): <span style="color:' + T.colors.off + '">without BN (dashed) <b>' + (100 * a0).toFixed(0) + '%</b></span>, <span style="color:' + T.colors.on + '">with BN <b>' + (100 * a1).toFixed(0) + '%</b></span>';
    }
    function toggle(force) {
      running = force != null ? force : !running; go.textContent = running ? 'Pause' : 'Train ▶';
      if (!running) { R.stop(); return; }
      let last = 0;
      R.start(jobs, function () { if (performance.now() - last > 250) { last = performance.now(); draw(); } }, function () { running = false; go.textContent = 'Train ▶'; draw(); });
    }
    reset(); toggle(true);
  };

  // ------------------------------------------------------------------ ch1: small changes amplify
  W.layerStack = function (host) {
    Book.head(host, 'A nudge at the bottom moves everything above',
      'A fixed 3-layer sigmoid network (4 units per layer) sees 200 random inputs between 0 and 1. Nudge all of layer 1’s weights by the same small amount and watch the mean ± spread of one unit’s pre-activation in each layer move.');
    const ctl = el('div', { class: 'controls' }, host);
    let k = 0.3;
    Book.slider(ctl, 'nudge to layer-1 weights', -0.5, 0.5, 0.05, k, function (v) { k = v; draw(); }, function (v) { return (v >= 0 ? '+' : '') + v.toFixed(2); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    const r = ML.rng(3), Ws = [0, 1, 2].map(function () { return Array.from({ length: 4 }, function () { return Array.from({ length: 4 }, function () { return r.normal(0, 1.1); }); }); });
    const bs = [0, 1, 2].map(function () { return Array.from({ length: 4 }, function () { return r.normal(0, 0.5); }); });
    const U = Array.from({ length: 200 }, function () { return [0, 1, 2, 3].map(function () { return r.next(); }); });
    function stats(scale) {
      const res = [[], [], []];
      U.forEach(function (u) {
        let h = u;
        for (let l = 0; l < 3; l++) {
          const x = [0, 1, 2, 3].map(function (j) { let s = bs[l][j]; for (let i = 0; i < 4; i++) s += h[i] * (Ws[l][i][j] + (l === 0 ? scale : 0)); return s; });
          res[l].push(x[0]); h = x.map(sig);
        }
      });
      return res.map(T.mv);
    }
    const base = stats(0);
    function draw() {
      const now = stats(k); view.innerHTML = '';
      const Wd = 560, H = 170, svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H }, view), X = function (v) { return 90 + (v + 4) / 8 * (Wd - 110); };
      [-4, -2, 0, 2, 4].forEach(function (t) { S('line', { x1: X(t), x2: X(t), y1: 10, y2: H - 20, stroke: 'var(--line)' }, svg); S('text', { x: X(t), y: H - 6, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: f(t, 0) }, svg); });
      [0, 1, 2].forEach(function (l) {
        const y = 30 + l * 45, b = base[l], n = now[l];
        S('text', { x: 6, y: y + 4, 'font-size': 12, class: 'svg-text', text: 'layer ' + (l + 1) }, svg);
        S('line', { x1: X(b.mean - Math.sqrt(b.var)), x2: X(b.mean + Math.sqrt(b.var)), y1: y - 7, y2: y - 7, stroke: 'var(--muted)', 'stroke-width': 5, opacity: 0.5 }, svg);
        S('circle', { cx: X(b.mean), cy: y - 7, r: 4, fill: 'var(--muted)' }, svg);
        S('line', { x1: X(n.mean - Math.sqrt(n.var)), x2: X(n.mean + Math.sqrt(n.var)), y1: y + 7, y2: y + 7, stroke: 'var(--accent)', 'stroke-width': 5 }, svg);
        S('circle', { cx: X(n.mean), cy: y + 7, r: 4, fill: 'var(--accent)' }, svg);
      });
      out.innerHTML = 'Grey: before the nudge. Coloured: after. The mean of unit 1 moves by ' + [0, 1, 2].map(function (l) { return 'layer ' + (l + 1) + ': <b>' + f(now[l].mean - base[l].mean, 2) + '</b>'; }).join(', ') + '. Only layer 1’s weights changed.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch2: minibatch gradient estimates
  W.minibatchGrad = function (host) {
    Book.head(host, 'Minibatch gradients scatter around the true gradient',
      'A line y = w·x + b is fitted to 500 points. Each dot is the gradient (∂ℓ/∂w, ∂ℓ/∂b) estimated from one random minibatch of size m; the star is the exact gradient over all 500.');
    const ctl = el('div', { class: 'controls' }, host); let m = 8;
    Book.slider(ctl, 'minibatch size m', 1, 128, 1, m, function (v) { m = v; draw(); });
    Book.button(ctl, 'Re-draw minibatches', function () { seed++; draw(); });
    const view = el('div', { style: 'max-width:380px' }, host), out = el('div', { class: 'out' }, host);
    const r = ML.rng(4), N = 500, xs = [], ys = []; let seed = 1;
    for (let i = 0; i < N; i++) { const x = r.uniform(-2, 2); xs.push(x); ys.push(1.5 * x + 0.5 + r.normal(0, 1)); }
    const w = 0.3, b = -0.4;
    function grad(idx) { let gw = 0, gb = 0; idx.forEach(function (i) { const e = w * xs[i] + b - ys[i]; gw += 2 * e * xs[i]; gb += 2 * e; }); return [gw / idx.length, gb / idx.length]; }
    const full = grad(ML.range(N));
    function draw() {
      const rr = ML.rng(seed * 31 + m), pts = [];
      for (let k = 0; k < 120; k++) { const idx = []; for (let i = 0; i < m; i++) idx.push(rr.int(N)); pts.push(grad(idx)); }
      const P = T.plot(view, [{ pts: pts, color: 'var(--accent)', r: 2.6 }], { xmin: full[0] - 6, xmax: full[0] + 6, ymin: full[1] - 6, ymax: full[1] + 6, xlabel: '∂ℓ/∂w', ylabel: '∂ℓ/∂b', width: 340, height: 320 });
      const cx = P.X(full[0]), cy = P.Y(full[1]), st = [];
      for (let j = 0; j < 10; j++) { const rad = j % 2 ? 4 : 10, a = -Math.PI / 2 + j * Math.PI / 5; st.push((cx + rad * Math.cos(a)) + ',' + (cy + rad * Math.sin(a))); }
      S('polygon', { points: st.join(' '), fill: '#c0392b', stroke: 'var(--panel)' }, P.svg);
      const sd = Math.sqrt(ML.mean(pts.map(function (p) { return (p[0] - full[0]) * (p[0] - full[0]) + (p[1] - full[1]) * (p[1] - full[1]); })));
      const mean = [ML.mean(pts.map(function (p) { return p[0]; })), ML.mean(pts.map(function (p) { return p[1]; }))];
      out.innerHTML = 'true gradient (' + f(full[0], 2) + ', ' + f(full[1], 2) + ') · average of the 120 estimates (' + f(mean[0], 2) + ', ' + f(mean[1], 2) + ') · typical error <b>' + f(sd, 2) + '</b> (shrinks like 1/√m)';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch3: chain rule on a tiny graph
  W.chainRule = function (host) {
    Book.head(host, 'Gradients flow backwards and add up',
      'The loss is ℓ = (a·c + c)². Values flow forward (black), gradients flow backward (coloured). c is used twice, so its gradient is the sum of two paths. A finite-difference check confirms each one.');
    const st = { a: 1.5, c: -0.5 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'a', -2, 2, 0.1, st.a, function (v) { st.a = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'c', -2, 2, 0.1, st.c, function (v) { st.c = v; draw(); }, function (v) { return v.toFixed(1); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    function L(a, c) { const p = a * c, s = p + c; return s * s; }
    function draw() {
      const a = st.a, c = st.c, p = a * c, s = p + c, l = s * s;
      const dl_ds = 2 * s, dl_dp = dl_ds, dl_da = dl_dp * c, dl_dc1 = dl_dp * a, dl_dc2 = dl_ds, dl_dc = dl_dc1 + dl_dc2;
      view.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 600 200' }, view);
      const node = function (x, y, label, val, grad) {
        S('rect', { x: x - 48, y: y - 24, width: 96, height: 48, rx: 10, fill: 'var(--panel)', stroke: 'var(--line)' }, svg);
        S('text', { x: x, y: y - 6, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, class: 'svg-text', text: label + ' = ' + f(val, 2) }, svg);
        S('text', { x: x, y: y + 13, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--accent)', text: '∂ℓ/∂' + label.split(' ')[0] + ' = ' + f(grad, 2) }, svg);
      };
      const arrow = function (x1, y1, x2, y2) { S('line', { x1: x1, y1: y1, x2: x2, y2: y2, stroke: 'var(--muted)', 'stroke-width': 1.5, 'marker-end': 'url(#arr)' }, svg); };
      const defs = S('defs', null, svg), mk = S('marker', { id: 'arr', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto' }, defs);
      S('path', { d: 'M0,0 L10,5 L0,10 z', fill: 'var(--muted)' }, mk);
      node(60, 50, 'a', a, dl_da); node(60, 150, 'c', c, dl_dc); node(230, 50, 'p', p, dl_dp); node(390, 100, 's', s, dl_ds); node(540, 100, 'ℓ', l, 1);
      arrow(108, 50, 182, 50); arrow(108, 140, 190, 70); arrow(278, 60, 342, 90); arrow(108, 155, 342, 110); arrow(438, 100, 492, 100);
      S('text', { x: 230, y: 92, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: 'p = a·c' }, svg);
      S('text', { x: 390, y: 140, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: 's = p + c' }, svg);
      S('text', { x: 540, y: 140, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: 'ℓ = s²' }, svg);
      const h = 1e-5, fd_a = (L(a + h, c) - L(a - h, c)) / (2 * h), fd_c = (L(a, c + h) - L(a, c - h)) / (2 * h);
      out.innerHTML = '∂ℓ/∂s = 2s = <b>' + f(dl_ds, 3) + '</b> · ∂ℓ/∂a = ∂ℓ/∂p · c = <b>' + f(dl_da, 3) + '</b> (finite difference ' + f(fd_a, 3) + ') · ∂ℓ/∂c = ∂ℓ/∂p · a + ∂ℓ/∂s · 1 = ' + f(dl_dc1, 3) + ' + ' + f(dl_dc2, 3) + ' = <b>' + f(dl_dc, 3) + '</b> (finite difference ' + f(fd_c, 3) + ')';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch4: learning rate on a bowl
  W.lrBowl = function (host) {
    Book.head(host, 'When the learning rate is too large, every step makes things worse',
      'Gradient descent on ℓ(w) = ½·k·w², starting at w = 1. The chart shows |w| after each step (log scale). Push α·k past 2.');
    const st = { k: 2, a: 0.4 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'curvature k', 0.5, 8, 0.1, st.k, function (v) { st.k = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'learning rate α', 0.02, 1.5, 0.01, st.a, function (v) { st.a = v; draw(); }, function (v) { return v.toFixed(2); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const fac = 1 - st.a * st.k, pts = []; let w = 1;
      for (let t = 0; t <= 20; t++) { pts.push([t, Math.log10(Math.max(1e-6, Math.min(1e6, Math.abs(w))))]); w *= fac; }
      Book.chart.lines(view, [{ name: '', color: Math.abs(fac) > 1 ? '#c0392b' : 'var(--accent)', points: pts, r: 2.5 }], { xmin: 0, xmax: 20, ymin: -6, ymax: 6, yticks: 6, xticks: [0, 5, 10, 15, 20], xlabel: 'step t', ylabel: '|w_t| (log₁₀)', yfmt: function (v) { return v === 0 ? '1' : '10^' + v; }, height: 230 });
      const kind = Math.abs(fac) > 1 ? '<span class="bad">diverges</span>' : fac < 0 ? 'oscillates and shrinks' : 'shrinks smoothly';
      out.innerHTML = 'α·k = <b>' + f(st.a * st.k, 2) + '</b> · each step multiplies w by 1 − α·k = <b>' + f(fac, 2) + '</b> → ' + kind + ' · largest safe α for this k: 2/k = ' + f(2 / st.k, 2);
    }
    draw();
  };

  // ------------------------------------------------------------------ ch5: mean and variance
  W.meanVar = function (host) {
    Book.head(host, 'Every number of μ and σ², worked out',
      'Six values on a number line. Drag the sliders: the table shows each deviation from the mean and its square, exactly as in the first two lines of Algorithm 1.');
    const xs = [1, 2, 2.5, 4, 5.5, 3], ctl = el('div', { class: 'controls' }, host);
    xs.forEach(function (v, i) { Book.slider(ctl, 'x' + '₁₂₃₄₅₆'[i], -4, 8, 0.1, v, function (val) { xs[i] = val; draw(); }, function (val) { return val.toFixed(1); }); });
    const view = el('div', null, host), tab = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const s = T.mv(xs), sd = Math.sqrt(s.var);
      view.innerHTML = '';
      const Wd = 560, svg = S('svg', { viewBox: '0 0 ' + Wd + ' 80' }, view), X = function (v) { return 20 + (v + 4) / 12 * (Wd - 40); };
      S('rect', { x: X(s.mean - sd), y: 20, width: X(s.mean + sd) - X(s.mean - sd), height: 30, fill: 'var(--accent-soft)' }, svg);
      S('line', { x1: X(-4), x2: X(8), y1: 50, y2: 50, stroke: 'var(--muted)' }, svg);
      for (let t = -4; t <= 8; t += 2) S('text', { x: X(t), y: 68, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: f(t, 0) }, svg);
      xs.forEach(function (v) { S('circle', { cx: X(v), cy: 40, r: 6, fill: 'var(--accent)', opacity: 0.8 }, svg); });
      S('line', { x1: X(s.mean), x2: X(s.mean), y1: 12, y2: 52, stroke: '#c0392b', 'stroke-width': 2.5 }, svg);
      S('text', { x: X(s.mean), y: 10, 'text-anchor': 'middle', 'font-size': 11.5, fill: '#c0392b', text: 'μ' }, svg);
      let h = '<table class="mtab"><tr><th>i</th>' + xs.map(function (_, i) { return '<th>' + (i + 1) + '</th>'; }).join('') + '<th>sum</th><th>÷ m</th></tr>';
      h += '<tr><th>x_i</th>' + xs.map(function (v) { return '<td>' + f(v, 1) + '</td>'; }).join('') + '<td>' + f(ML.sum(xs), 2) + '</td><td class="hl">μ = ' + f(s.mean, 3) + '</td></tr>';
      h += '<tr><th>x_i − μ</th>' + xs.map(function (v) { return '<td>' + f(v - s.mean, 2) + '</td>'; }).join('') + '<td>' + f(0, 2) + '</td><td></td></tr>';
      h += '<tr><th>(x_i − μ)²</th>' + xs.map(function (v) { return '<td>' + f((v - s.mean) * (v - s.mean), 2) + '</td>'; }).join('') + '<td>' + f(s.var * 6, 2) + '</td><td class="hl">σ² = ' + f(s.var, 3) + '</td></tr></table>';
      tab.innerHTML = h;
      out.innerHTML = 'μ = <b>' + f(s.mean, 3) + '</b>, σ² = <b>' + f(s.var, 3) + '</b>, σ = √σ² = <b>' + f(sd, 3) + '</b> (shaded band: μ ± σ). The deviations always add up to 0.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch6: percentiles
  W.percentiles = function (host) {
    Book.head(host, 'Three lines summarize a distribution',
      'Draw 2,000 values from a distribution, see their histogram and the 15th, 50th and 85th percentiles (the paper’s Fig. 1 uses exactly these three).');
    const st = { shape: 'normal', mu: 0, sd: 1 }, ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'shape', [{ value: 'normal', label: 'bell (normal)' }, { value: 'skew', label: 'skewed' }, { value: 'two', label: 'two bumps' }], st.shape, function (v) { st.shape = v; draw(); });
    Book.slider(ctl, 'shift', -2, 2, 0.1, st.mu, function (v) { st.mu = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'spread', 0.3, 2, 0.05, st.sd, function (v) { st.sd = v; draw(); }, function (v) { return v.toFixed(2); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(9), v = [];
      for (let i = 0; i < 2000; i++) {
        let z = st.shape === 'normal' ? r.normal(0, 1) : st.shape === 'skew' ? (-Math.log(1 - r.next()) - 1) : (r.next() < 0.5 ? -1.2 : 1.2) + r.normal(0, 0.45);
        v.push(st.mu + st.sd * z);
      }
      const s = T.sorted(v), p = [15, 50, 85].map(function (q) { return T.pct(s, q); }), mv = T.mv(v);
      T.hist(view, v, { lo: -6, hi: 6, bins: 60, width: 600, height: 200, ticks: [-6, -4, -2, 0, 2, 4, 6], marks: [{ x: p[0], label: '15th', color: '#c0392b' }, { x: p[1], label: '50th', color: 'var(--ink)' }, { x: p[2], label: '85th', color: '#c0392b' }] });
      out.innerHTML = 'percentiles 15/50/85: <b>' + p.map(function (x) { return f(x, 2); }).join(' / ') + '</b> · mean ' + f(mv.mean, 2) + ', σ ' + f(Math.sqrt(mv.var), 2) + ' · μ − 1.04σ = ' + f(mv.mean - 1.04 * Math.sqrt(mv.var), 2) + ', μ + 1.04σ = ' + f(mv.mean + 1.04 * Math.sqrt(mv.var), 2) + (st.shape === 'normal' ? ' (match for a bell)' : ' (only approximate for this shape)');
    }
    draw();
  };

  // ------------------------------------------------------------------ ch7: z-scores
  W.zscore = function (host) {
    Book.head(host, 'Standardizing undoes any shift and stretch',
      'Left: a raw feature (500 values). Shift and stretch it however you like. Right: the same values after subtracting their mean and dividing by their standard deviation.');
    const st = { a: 3, b: 10 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'stretch', 0.2, 8, 0.1, st.a, function (v) { st.a = v; draw(); }, function (v) { return '×' + v.toFixed(1); });
    Book.slider(ctl, 'shift', -20, 20, 0.5, st.b, function (v) { st.b = v; draw(); }, function (v) { return v.toFixed(1); });
    const g = el('div', { class: 'grid2' }, host), L = el('div', null, g), R = el('div', null, g), out = el('div', { class: 'out' }, host);
    const r = ML.rng(5), base = Array.from({ length: 500 }, function () { return r.normal(0, 1) + 0.6 * (r.next() < 0.3 ? 1 : 0); });
    function draw() {
      const raw = base.map(function (z) { return st.a * z + st.b; }), s = T.mv(raw), sd = Math.sqrt(s.var), zs = raw.map(function (x) { return (x - s.mean) / sd; }), s2 = T.mv(zs);
      T.hist(L, raw, { lo: -40, hi: 40, bins: 50, title: 'raw x', ticks: [-40, -20, 0, 20, 40], color: 'var(--muted)' });
      T.hist(R, zs, { lo: -4, hi: 4, bins: 50, title: 'x̂ = (x − μ)/σ', ticks: [-4, -2, 0, 2, 4] });
      out.innerHTML = 'raw: mean <b>' + f(s.mean, 2) + '</b>, variance <b>' + f(s.var, 2) + '</b> → standardized: mean <b>' + f(s2.mean, 3) + '</b>, variance <b>' + f(s2.var, 3) + '</b>. Same shape, new units.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch7: round bowl vs narrow valley
  W.roundBowl = function (host) {
    Book.head(host, 'Standardized inputs make the loss a round bowl',
      'A linear model with two inputs, one ranging over about ±1 and one over about ±scale. The map shows the loss contours over the two weights and 40 gradient-descent steps with the best learning rate for this map.');
    const st = { scale: 6, std: false }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'scale of input 2', 1, 10, 0.5, st.scale, function (v) { st.scale = v; draw(); }, function (v) { return '×' + v.toFixed(1); });
    const lab = el('label', null, ctl), cb = el('input', { type: 'checkbox' }, lab); lab.appendChild(document.createTextNode(' standardize the inputs'));
    cb.addEventListener('change', function () { st.std = cb.checked; draw(); });
    const view = el('div', { style: 'max-width:380px' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      // loss ½(w1²·v1 + w2²·v2) around the optimum, with input variances v1 = 1, v2 = scale²; standardizing sets both to 1
      const v1 = 1, v2 = st.std ? 1 : st.scale * st.scale, alpha = 2 / (v1 + v2);       // best fixed step for a quadratic
      const opt = [1, 1], path = [[-1.5, 2.2]];
      let w = [-1.5, 2.2], steps = 40, hit = -1;
      for (let t = 0; t < steps; t++) {
        w = [w[0] - alpha * v1 * (w[0] - opt[0]), w[1] - alpha * v2 * (w[1] - opt[1])]; path.push(w.slice());
        if (hit < 0 && Math.hypot(w[0] - opt[0], w[1] - opt[1]) < 0.05) hit = t + 1;
      }
      const P = T.plot(view, [{ pts: path, color: 'var(--accent)', line: true, r: 2.5 }], { xmin: -2, xmax: 3, ymin: -1.5, ymax: 3.5, xlabel: 'weight 1', ylabel: 'weight 2', width: 340, height: 340 });
      [0.3, 1, 2.5, 5, 9].forEach(function (c) {          // ellipse ½(v1 a² + v2 b²) = c
        const pts = []; for (let i = 0; i <= 64; i++) { const t = i / 64 * 2 * Math.PI; pts.push(P.X(opt[0] + Math.sqrt(2 * c / v1) * Math.cos(t)) + ',' + P.Y(opt[1] + Math.sqrt(2 * c / v2) * Math.sin(t))); }
        S('polyline', { points: pts.join(' '), fill: 'none', stroke: 'var(--muted)', opacity: 0.5 }, P.svg);
      });
      out.innerHTML = 'curvatures ' + f(v1, 1) + ' and ' + f(v2, 1) + ' (ratio ' + f(v2 / v1, 1) + ') · best α = ' + f(alpha, 3) + ' · distance to the minimum after 40 steps: <b>' + f(Math.hypot(w[0] - opt[0], w[1] - opt[1]), 3) + '</b>' + (hit > 0 ? ' · within 0.05 after <b>' + hit + '</b> steps' : ' · not within 0.05 after 40 steps');
    }
    draw();
  };

  // ------------------------------------------------------------------ ch8: whitening in 2-D
  W.whiten2D = function (host) {
    Book.head(host, 'Standardizing rescales; whitening also un-tilts',
      'A cloud of 300 two-feature points with correlation ρ. Left: raw. Middle: each feature standardized on its own (what BN does). Right: fully whitened with Cov⁻¹ᐟ².');
    const st = { rho: 0.8 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'correlation ρ', -0.95, 0.95, 0.05, st.rho, function (v) { st.rho = v; draw(); }, function (v) { return v.toFixed(2); });
    const g = el('div', { class: 'hgrid' }, host), P = [0, 1, 2].map(function () { return el('div', null, g); }), out = el('div', { class: 'out' }, host);
    function cov(pts) {
      const n = pts.length, m = [0, 0]; pts.forEach(function (p) { m[0] += p[0] / n; m[1] += p[1] / n; });
      let a = 0, b = 0, c = 0; pts.forEach(function (p) { const x = p[0] - m[0], y = p[1] - m[1]; a += x * x / n; b += x * y / n; c += y * y / n; });
      return { m: m, a: a, b: b, c: c };
    }
    function invSqrt(C) {                                   // inverse square root of the 2×2 symmetric matrix [[a,b],[b,c]]
      const tr = C.a + C.c, det = C.a * C.c - C.b * C.b, s = Math.sqrt(det), t = Math.sqrt(tr + 2 * s);
      const R = [[(C.a + s) / t, C.b / t], [C.b / t, (C.c + s) / t]], d = R[0][0] * R[1][1] - R[0][1] * R[1][0];
      return [[R[1][1] / d, -R[0][1] / d], [-R[1][0] / d, R[0][0] / d]];
    }
    const r = ML.rng(8), base = Array.from({ length: 300 }, function () { return [r.normal(0, 1), r.normal(0, 1)]; });
    function draw() {
      const q = Math.sqrt(1 - st.rho * st.rho);
      const raw = base.map(function (z) { const a = z[0], b = st.rho * z[0] + q * z[1]; return [3 + 2.2 * a, 1 + 0.8 * b]; });
      const C = cov(raw), sx = Math.sqrt(C.a), sy = Math.sqrt(C.c);
      const stdz = raw.map(function (p) { return [(p[0] - C.m[0]) / sx, (p[1] - C.m[1]) / sy]; });
      const M = invSqrt(C), wh = raw.map(function (p) { const x = p[0] - C.m[0], y = p[1] - C.m[1]; return [M[0][0] * x + M[0][1] * y, M[1][0] * x + M[1][1] * y]; });
      const box = { xmin: -4, xmax: 8, ymin: -4, ymax: 8, width: 230, height: 230 };
      T.plot(P[0], [{ pts: raw, color: 'var(--muted)', r: 2 }], Object.assign({}, box, { title: 'raw' }));
      const b2 = { xmin: -4, xmax: 4, ymin: -4, ymax: 4, width: 230, height: 230 };
      T.plot(P[1], [{ pts: stdz, color: 'var(--accent)', r: 2 }], Object.assign({}, b2, { title: 'standardized per feature' }));
      T.plot(P[2], [{ pts: wh, color: '#2f855a', r: 2 }], Object.assign({}, b2, { title: 'whitened' }));
      const c1 = cov(stdz), c2 = cov(wh);
      out.innerHTML = 'Cov after standardizing: [[' + f(c1.a, 2) + ', <b>' + f(c1.b, 2) + '</b>], [' + f(c1.b, 2) + ', ' + f(c1.c, 2) + ']] (variances 1, correlation kept) · after whitening: [[' + f(c2.a, 2) + ', <b>' + f(c2.b, 2) + '</b>], [' + f(c2.b, 2) + ', ' + f(c2.c, 2) + ']] (the identity)';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch9: sigmoid saturation
  W.saturation = function (host) {
    Book.head(host, 'Inputs that drift into the flat ends stop passing gradient',
      'A distribution of 2,000 pre-activations (histogram) under the sigmoid (dashed). Move its centre and spread: the readout gives the average slope g′(x) that backpropagation would multiply by.');
    const st = { mu: 0, sd: 1 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'centre', -8, 8, 0.1, st.mu, function (v) { st.mu = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'spread', 0.2, 6, 0.1, st.sd, function (v) { st.sd = v; draw(); }, function (v) { return v.toFixed(1); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    const r = ML.rng(6), base = Array.from({ length: 2000 }, function () { return r.normal(0, 1); });
    function draw() {
      const v = base.map(function (z) { return st.mu + st.sd * z; });
      let gsum = 0, sat = 0; v.forEach(function (x) { const g = sig(x); gsum += g * (1 - g); if (Math.abs(x) > 4) sat++; });
      T.hist(view, v, { lo: -12, hi: 12, bins: 60, width: 600, height: 200, curve: sig, ticks: [-12, -8, -4, 0, 4, 8, 12], band: [-4, 4] });
      out.innerHTML = 'average slope g′(x) = <b>' + f(gsum / v.length, 4) + '</b> (best possible 0.25) · <b>' + (100 * sat / v.length).toFixed(0) + '%</b> of the values are saturated (|x| &gt; 4, outside the shaded band, where g′ &lt; 0.018)';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch9: gradient size through a deep sigmoid net
  W.deepGradient = function (host) {
    Book.head(host, 'Gradients shrink on their way down',
      'A real 10-layer sigmoid network (32 units per layer, random weights) and a batch of 60 glyphs. Bars: the average size of ∂ℓ/∂x at each layer’s pre-activations, computed by backpropagation. Shift the pre-activations to push the sigmoids towards saturation.');
    const st = { shift: 0, bn: false, scale: 1 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'bias shift (all layers)', -4, 4, 0.25, st.shift, function (v) { st.shift = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'weight scale', 0.5, 3, 0.1, st.scale, function (v) { st.scale = v; draw(); }, function (v) { return '×' + v.toFixed(1); });
    const lab = el('label', null, ctl), cb = el('input', { type: 'checkbox' }, lab); lab.appendChild(document.createTextNode(' with Batch Normalization'));
    cb.addEventListener('change', function () { st.bn = cb.checked; draw(); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    const D = T.data(), X = T.rows(D.train, ML.range(60)), y = Array.from(D.train.y.slice(0, 60));
    function draw() {
      const net = new T.Net({ sizes: [64, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 10], act: 'sigmoid', bn: st.bn, sd: 'xavier', seed: 2 });
      net.W.forEach(function (Wt, l) { if (l < net.L - 1) for (let i = 0; i < Wt.data.length; i++) Wt.data[i] *= st.scale; });
      if (!st.bn) net.b.forEach(function (b, l) { if (b && l < net.L - 1) b.data.fill(st.shift); });   // a BN net has no bias: BN would cancel it (Chapter 23)
      NN.zeroGrad(net.params);
      const L = NN.softmaxCE(net.forward(X, 'batch'), y); L.backward();
      const g = net.rec.z.map(function (z) { let s = 0; for (let i = 0; i < z.grad.length; i++) s += Math.abs(z.grad[i]); return s / z.grad.length; });
      const lg = g.map(function (v) { return Math.log10(Math.max(v, 1e-12)); }), lo = Math.floor(Math.min.apply(null, lg)) - 0.5;
      Book.chart.bars(view, g.map(function (v, l) { return { label: 'L' + (l + 1), value: Math.log10(Math.max(v, 1e-12)) - lo, note: v.toExponential(0).replace('e', '·10^').replace('+', ''), color: st.bn ? T.colors.on : T.colors.off }; }),
        { height: 220, tickFmt: function (v) { return '10^' + (v + lo).toFixed(1); }, ylabel: 'mean |∂ℓ/∂x| (log)' });
      out.innerHTML = 'gradient at layer 10: <b>' + g[9].toExponential(2) + '</b> · at layer 1: <b>' + g[0].toExponential(2) + '</b> · ratio <b>' + f(g[9] / g[0], 0) + '×</b> smaller at the bottom.' + (st.bn ? ' With BN the sliders change nothing: every layer’s pre-activations are re-centred and re-scaled using the batch (Chapters 15, 23, 25).' : '');
    }
    draw();
  };
})();
