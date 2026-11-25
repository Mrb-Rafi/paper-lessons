/* Batch Normalization book, Part II (internal covariate shift) widgets. Uses Book, ML, NN, BNT. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, T = window.BNT;
  const f = T.f, sig = function (x) { return 1 / (1 + Math.exp(-x)); };

  // two-colour histogram (classes) on one axis, with an optional vertical threshold
  function hist2(host, a, b, o) {
    const r = T.hist(host, a.concat(b), Object.assign({ color: 'transparent' }, o));
    const nb = o.bins || 40, lo = o.lo, hi = o.hi, bw = (hi - lo) / nb;
    const ca = new Array(nb).fill(0), cb = new Array(nb).fill(0);
    a.forEach(function (v) { const k = Math.floor((v - lo) / bw); if (k >= 0 && k < nb) ca[k]++; });
    b.forEach(function (v) { const k = Math.floor((v - lo) / bw); if (k >= 0 && k < nb) cb[k]++; });
    const mx = Math.max.apply(null, ca.concat(cb)) || 1, H = o.height || 150, base = H - 22, top = 10;
    [[ca, '#c0392b'], [cb, '#2b6cb0']].forEach(function (p) {
      p[0].forEach(function (c, k) { if (c) S('rect', { x: r.X(lo + k * bw) + 0.5, y: base - c / mx * (base - top), width: Math.max(0.5, r.X(lo + bw) - r.X(lo) - 1), height: c / mx * (base - top), fill: p[1], opacity: 0.55 }, r.svg); });
    });
    if (o.thr != null) S('line', { x1: r.X(o.thr), x2: r.X(o.thr), y1: top, y2: base, stroke: 'var(--ink)', 'stroke-width': 2, 'stroke-dasharray': '5 3' }, r.svg);
    return r;
  }

  // ------------------------------------------------------------------ ch10: covariate shift
  W.covShift = function (host) {
    Book.head(host, 'A shifted input breaks a trained model',
      'Two classes (red, blue) of a single input. The model learned the threshold (dashed) on the training inputs. Now shift and stretch the inputs it sees at test time.');
    const st = { s: 1.2, k: 1, norm: false }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'shift', -3, 3, 0.1, st.s, function (v) { st.s = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'stretch', 0.4, 2.5, 0.05, st.k, function (v) { st.k = v; draw(); }, function (v) { return '×' + v.toFixed(2); });
    const lab = el('label', null, ctl), cb = el('input', { type: 'checkbox' }, lab); lab.appendChild(document.createTextNode(' standardize inputs with their current mean and spread'));
    cb.addEventListener('change', function () { st.norm = cb.checked; draw(); });
    const g = el('div', { class: 'grid2' }, host), L = el('div', null, g), R = el('div', null, g), out = el('div', { class: 'out' }, host);
    const r = ML.rng(10), A = [], B = [];
    for (let i = 0; i < 400; i++) { A.push(r.normal(-1, 0.7)); B.push(r.normal(1, 0.7)); }
    const tr = T.mv(A.concat(B)), sd0 = Math.sqrt(tr.var);
    function prep(v) { return st.norm ? v.map(function (x) { return x; }) : v; }
    function acc(a, b, thr) { let ok = 0; a.forEach(function (x) { if (x < thr) ok++; }); b.forEach(function (x) { if (x >= thr) ok++; }); return ok / (a.length + b.length); }
    function draw() {
      // the model: threshold 0 on inputs standardized with the TRAINING statistics (equivalently threshold tr.mean on raw inputs)
      const ta = A.map(function (x) { return st.k * x + st.s; }), tb = B.map(function (x) { return st.k * x + st.s; });
      let a2 = ta, b2 = tb, thr = tr.mean, lbl = 'test inputs';
      if (st.norm) { const m = T.mv(ta.concat(tb)), sd = Math.sqrt(m.var); a2 = ta.map(function (x) { return tr.mean + sd0 * (x - m.mean) / sd; }); b2 = tb.map(function (x) { return tr.mean + sd0 * (x - m.mean) / sd; }); lbl = 'test inputs, re-standardized'; }
      hist2(L, A, B, { lo: -6, hi: 6, bins: 48, title: 'training inputs', thr: tr.mean, ticks: [-6, -3, 0, 3, 6], width: 300 });
      hist2(R, a2, b2, { lo: -6, hi: 6, bins: 48, title: lbl, thr: thr, ticks: [-6, -3, 0, 3, 6], width: 300 });
      out.innerHTML = 'accuracy on training inputs <b>' + (100 * acc(A, B, tr.mean)).toFixed(1) + '%</b> · on shifted test inputs <b>' + (100 * acc(a2, b2, thr)).toFixed(1) + '%</b>' + (st.norm ? ' (standardizing undid the shift)' : '');
    }
    prep([]); draw();
  };

  // ------------------------------------------------------------------ ch11: internal covariate shift, live
  W.icsLive = function (host) {
    Book.head(host, 'Watch the layer inputs move during training',
      'A 3-layer sigmoid network (no BN) trains on the toy glyphs. Each panel: the pre-activations of one layer over 240 test glyphs (all 32 units pooled). Outline: at step 0. Tick “freeze layers 1–2” and restart: only layer 3 and the output layer learn.');
    const st = { freeze: false }, ctl = el('div', { class: 'controls' }, host);
    const go = Book.button(ctl, 'Train ▶', function () { toggle(); }, 'primary');
    Book.button(ctl, 'Restart', function () { reset(); });
    const lab = el('label', null, ctl), cb = el('input', { type: 'checkbox' }, lab); lab.appendChild(document.createTextNode(' freeze layers 1–2'));
    cb.addEventListener('change', function () { st.freeze = cb.checked; reset(); });
    const g = el('div', { class: 'hgrid' }, host), P = [0, 1, 2].map(function () { return el('div', null, g); }), out = el('div', { class: 'out' }, host);
    const R = T.runner(); let J, init, running = false;
    const D = T.data(), probe = T.rows(D.test, ML.range(240));
    function layerVals() { J.net.forward(probe, 'batch'); return J.net.rec.z.map(function (z) { return Array.from(z.data); }); }
    function reset() {
      R.stop(); running = false; go.textContent = 'Train ▶';
      J = T.job({ net: { sizes: [64, 32, 32, 32, 10], act: 'sigmoid', bn: false, sd: 'xavier', seed: 5 }, lr: 0.5, momentum: 0, bs: 60, steps: 1500, every: 100 });
      if (st.freeze) { const n = J.net, keep = [n.W[2], n.b[2], n.W[3], n.b[3]]; Object.assign(J.opt, new NN.SGD(keep, { lr: 0.5, momentum: 0 })); }
      init = layerVals(); draw();
    }
    function draw() {
      const now = layerVals();
      now.forEach(function (v, l) {
        const h = T.hist(P[l], v, { lo: -8, hi: 8, bins: 40, title: 'layer ' + (l + 1), width: 260, height: 150, ticks: [-8, -4, 0, 4, 8], curve: sig });
        const c0 = new Array(40).fill(0); init[l].forEach(function (x) { const k = Math.floor((x + 8) / 16 * 40); if (k >= 0 && k < 40) c0[k]++; });
        const mx = Math.max.apply(null, h.counts) || 1, pts = [];
        c0.forEach(function (c, k) { const x0 = h.X(-8 + k * 0.4), x1 = h.X(-8 + (k + 1) * 0.4), y = h.Y(Math.min(c, mx)); pts.push(x0 + ',' + y, x1 + ',' + y); });
        S('polyline', { points: pts.join(' '), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.2, opacity: 0.6 }, h.svg);
        const a = T.mv(init[l]), b = T.mv(v);
        el('div', { class: 'note', html: 'mean ' + f(a.mean, 2) + ' → <b>' + f(b.mean, 2) + '</b>, sd ' + f(Math.sqrt(a.var), 2) + ' → <b>' + f(Math.sqrt(b.var), 2) + '</b>' }, P[l]);
      });
      out.innerHTML = 'step <b>' + J.step + '</b> / 1500 · test accuracy (toy) <b>' + (100 * J.acc[J.acc.length - 1][1]).toFixed(0) + '%</b>' + (st.freeze ? ' · layers 1–2 frozen' : '');
    }
    function toggle() {
      running = !running; go.textContent = running ? 'Pause' : 'Train ▶';
      if (!running) { R.stop(); return; }
      let last = 0;
      R.start([J], function () { if (performance.now() - last > 300) { last = performance.now(); draw(); } }, function () { running = false; go.textContent = 'Train ▶'; draw(); });
    }
    reset(); toggle();
  };

  // ------------------------------------------------------------------ ch12: the big idea
  W.bigIdea = function (host) {
    Book.head(host, 'Whatever the layers below do, BN hands on the same distribution',
      'Left: one unit’s pre-activations x over a minibatch of 200 examples; drift them as lower-layer learning would. Middle: x̂ after BN’s standardization. Right: y = γ·x̂ + β, which goes into the sigmoid (dashed).');
    const st = { s: 3, k: 2.5, g: 1, b: 0 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'drift: shift', -6, 6, 0.1, st.s, function (v) { st.s = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'drift: stretch', 0.2, 5, 0.1, st.k, function (v) { st.k = v; draw(); }, function (v) { return '×' + v.toFixed(1); });
    Book.slider(ctl, 'γ (learned)', 0.2, 3, 0.05, st.g, function (v) { st.g = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'β (learned)', -3, 3, 0.05, st.b, function (v) { st.b = v; draw(); }, function (v) { return v.toFixed(2); });
    const g = el('div', { class: 'hgrid' }, host), P = [0, 1, 2].map(function () { return el('div', null, g); }), out = el('div', { class: 'out' }, host);
    const r = ML.rng(12), base = Array.from({ length: 200 }, function () { return r.normal(0, 1) + 0.4 * r.normal(0, 1) * r.normal(0, 1); });
    const gam = NN.zeros(1, 1), bet = NN.zeros(1, 1);
    function draw() {
      const x = base.map(function (z) { return st.k * z + st.s; });
      gam.data[0] = st.g; bet.data[0] = st.b;
      const X = NN.tensor(x.map(function (v) { return [v]; })), Y = NN.batchNorm(X, gam, bet, {});
      const xh = x.map(function (v) { return (v - Y.stats.mean[0]) / Math.sqrt(Y.stats.var[0] + 1e-5); }), y = Array.from(Y.data);
      T.hist(P[0], x, { lo: -12, hi: 12, bins: 48, title: 'x (drifting)', width: 260, color: 'var(--muted)', ticks: [-12, 0, 12] });
      T.hist(P[1], xh, { lo: -4, hi: 4, bins: 40, title: 'x̂ = (x − μ_B)/σ_B', width: 260, ticks: [-4, 0, 4] });
      T.hist(P[2], y, { lo: -8, hi: 8, bins: 40, title: 'y = γx̂ + β → sigmoid', width: 260, curve: sig, ticks: [-8, 0, 8], color: '#2f855a' });
      const a = T.mv(xh), b = T.mv(y);
      out.innerHTML = 'batch μ_B = <b>' + f(Y.stats.mean[0], 2) + '</b>, σ²_B = <b>' + f(Y.stats.var[0], 2) + '</b> · x̂: mean ' + f(a.mean, 3) + ', variance ' + f(a.var, 3) + ' (always) · y: mean ' + f(b.mean, 2) + ' = β, sd ' + f(Math.sqrt(b.var), 2) + ' = γ. Only γ and β, which the network learns, move what the sigmoid sees.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch13: the §2 bias blow-up
  W.biasBlowup = function (host) {
    Book.head(host, 'A bias that grows forever while the loss never moves',
      'The paper’s §2 example, run exactly: x = u + b, x̂ = x − E[x] over 50 examples, loss ℓ = mean of (x̂ − t)² for targets t whose mean is 0.8. Red: the gradient ignores that E[x] depends on b. Blue: the gradient goes through the mean subtraction.');
    const st = { lr: 0.2, steps: 100 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'learning rate', 0.05, 1, 0.05, st.lr, function (v) { st.lr = v; draw(); }, function (v) { return v.toFixed(2); });
    const g = el('div', { class: 'grid2' }, host), L = el('div', null, g), R = el('div', null, g), out = el('div', { class: 'out' }, host);
    const r = ML.rng(13), N = 50, u = [], t = [];
    for (let i = 0; i < N; i++) { u.push(r.normal(1, 1)); }
    const um = ML.mean(u); for (let i = 0; i < N; i++) t.push(u[i] - um + 0.8 + r.normal(0, 0.3));
    function run(naive) {
      let b = 0; const bs = [], ls = [];
      for (let s = 0; s <= st.steps; s++) {
        const x = u.map(function (v) { return v + b; }), Ex = ML.mean(x), xh = x.map(function (v) { return v - Ex; });
        const dxh = xh.map(function (v, i) { return 2 * (v - t[i]) / N; }), loss = ML.mean(xh.map(function (v, i) { return (v - t[i]) * (v - t[i]); }));
        bs.push([s, b]); ls.push([s, loss]);
        // naive: ∂ℓ/∂b = Σ ∂ℓ/∂x̂_i · 1   ;  through the mean: Σ ∂ℓ/∂x̂_i · (1 − 1) = 0
        const gb = naive ? ML.sum(dxh) : ML.sum(dxh.map(function (d) { return d * (1 - 1); }));
        b -= st.lr * gb;
      }
      return { b: bs, l: ls };
    }
    function draw() {
      const a = run(true), c = run(false);
      T.lines(L, [{ name: 'ignore E[x] in gradient', color: '#c0392b', marker: 'none', points: a.b }, { name: 'backprop through E[x]', color: '#2b6cb0', marker: 'none', dash: '6 4', points: c.b }], { xmin: 0, xmax: st.steps, xticks: [0, 50, 100], xlabel: 'step', ylabel: 'bias b', height: 220, width: 320 });
      T.lines(R, [{ name: 'ignore E[x] in gradient', color: '#c0392b', marker: 'none', points: a.l }, { name: 'backprop through E[x]', color: '#2b6cb0', marker: 'none', dash: '6 4', points: c.l }], { xmin: 0, xmax: st.steps, ymin: 0, ymax: 1.2, xticks: [0, 50, 100], xlabel: 'step', ylabel: 'loss ℓ', height: 220, width: 320, ydec: 1 });
      out.innerHTML = 'after ' + st.steps + ' steps: naive b = <b class="bad">' + f(a.b[st.steps][1], 1) + '</b> (grows by ' + f(a.b[1][1], 3) + ' every step), loss still <b>' + f(a.l[st.steps][1], 3) + '</b> · through the mean: b = <b>' + f(c.b[st.steps][1], 1) + '</b>, loss <b>' + f(c.l[st.steps][1], 3) + '</b>. Neither can lower the loss: a constant shift is invisible after mean subtraction, and only the honest gradient knows it.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch13: the cost of whitening
  W.whitenCost = function (host) {
    Book.head(host, 'Full whitening costs grow much faster than BN’s',
      'For a layer with d activations: how many numbers each method must estimate, and roughly how many arithmetic steps it needs per update (log scale).');
    const st = { lg: 9 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'activations d', 4, 13, 1, st.lg, function (v) { st.lg = v; draw(); }, function (v) { return String(Math.pow(2, v)); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const d = Math.pow(2, st.lg), items = [
        { label: 'BN statistics\n(2 per unit)', v: 2 * d, color: T.colors.on },
        { label: 'covariance\nentries (d²)', v: d * d, color: '#c0392b' },
        { label: 'BN arithmetic\n(~ m·d)', v: 32 * d * 4, color: T.colors.on },
        { label: 'Cov^(−1/2)\n(~ d³)', v: d * d * d, color: '#c0392b' }];
      Book.chart.bars(view, items.map(function (it) { return { label: it.label, value: Math.log10(it.v), note: it.v >= 1e6 ? it.v.toExponential(1).replace('e+', '·10^') : String(it.v), color: it.color }; }),
        { max: 12, ticks: 4, height: 240, tickFmt: function (v) { return '10^' + v; }, ylabel: 'count (log₁₀)', mb: 50 });
      out.innerHTML = 'd = <b>' + d + '</b>: BN keeps <b>' + 2 * d + '</b> numbers (a mean and a variance per unit) and learns ' + 2 * d + ' more (γ, β); whitening must estimate <b>' + (d * d).toLocaleString('en-US') + '</b> covariance entries and invert a ' + d + '×' + d + ' matrix (about ' + (d * d * d).toExponential(1) + ' steps), and differentiate through it. (BN arithmetic counted for a minibatch of m = 32, ~4 steps per value.)';
    }
    draw();
  };
})();
