/* Batch Normalization book, Part III (the Batch Normalizing Transform) widgets. Uses Book, ML, NN, BNT. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, T = window.BNT;
  const f = T.f, sig = function (x) { return 1 / (1 + Math.exp(-x)); };
  const SUB = '₁₂₃₄₅₆₇₈₉';

  // BN forward of one column of numbers (Algorithm 1), returning every intermediate
  function bn1(x, g, b, eps) {
    const m = x.length, mu = ML.sum(x) / m; let v = 0; x.forEach(function (xi) { v += (xi - mu) * (xi - mu); }); v /= m;
    const sd = Math.sqrt(v + eps), xh = x.map(function (xi) { return (xi - mu) / sd; });
    return { m: m, mu: mu, v: v, sd: sd, xh: xh, y: xh.map(function (z) { return g * z + b; }) };
  }
  // signed bar chart: values[] → bars around a zero line
  function sbars(host, groups, o) {
    host.innerHTML = ''; o = o || {};
    const n = groups[0].values.length, G = groups.length, Wd = o.width || 600, H = o.height || 200, m = { l: 40, r: 10, t: 14, b: 26 };
    let mx = o.max || 0; if (!mx) groups.forEach(function (g) { g.values.forEach(function (v) { mx = Math.max(mx, Math.abs(v)); }); }); mx = mx || 1;
    const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H }, host), Y = function (v) { return m.t + (H - m.t - m.b) / 2 * (1 - v / mx); };
    S('line', { x1: m.l, x2: Wd - m.r, y1: Y(0), y2: Y(0), stroke: 'var(--muted)' }, svg);
    [mx, -mx].forEach(function (v) { S('text', { x: m.l - 4, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 10.5, class: 'svg-text', text: f(v, 2) }, svg); });
    const cw = (Wd - m.l - m.r) / n, bw = cw * 0.8 / G;
    for (let i = 0; i < n; i++) {
      groups.forEach(function (g, k) { const v = g.values[i], x = m.l + i * cw + cw * 0.1 + k * bw; S('rect', { x: x, y: Math.min(Y(0), Y(v)), width: bw - 1, height: Math.abs(Y(v) - Y(0)), fill: g.color, rx: 1.5 }, svg); });
      S('text', { x: m.l + (i + 0.5) * cw, y: H - 8, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: 'i=' + (i + 1) }, svg);
    }
    T.legend(host, groups.map(function (g) { return { name: g.name, color: g.color }; }));
  }
  function table(rows, head, hl) {
    let h = '<table class="mtab"><tr>' + head.map(function (c) { return '<th>' + c + '</th>'; }).join('') + '</tr>';
    rows.forEach(function (r) { h += '<tr>' + r.map(function (c, j) { return '<td' + (hl && hl(j) ? ' class="hl"' : '') + '>' + c + '</td>'; }).join('') + '</tr>'; });
    return h + '</table>';
  }

  // ------------------------------------------------------------------ ch14: per-feature normalization
  W.perFeature = function (host) {
    Book.head(host, 'Each column is normalized with its own mean and spread',
      'A minibatch of m = 6 examples (rows) of a layer with d = 4 activations (columns) on very different scales. BN computes one μ and one σ per column, over the rows.');
    const ctl = el('div', { class: 'controls' }, host); let seed = 1;
    Book.button(ctl, 'New minibatch', function () { seed++; draw(); });
    const g = el('div', { class: 'grid2' }, host), L = el('div', { class: 'scroll-x' }, g), R = el('div', { class: 'scroll-x' }, g), out = el('div', { class: 'out' }, host);
    const scale = [[50, 10], [0, 0.1], [-3, 2], [800, 120]];
    function draw() {
      const r = ML.rng(seed * 7), X = [];
      for (let i = 0; i < 6; i++) X.push(scale.map(function (s) { return s[0] + s[1] * r.normal(0, 1); }));
      const cols = [0, 1, 2, 3].map(function (k) { return bn1(X.map(function (row) { return row[k]; }), 1, 0, 0); });
      const head = ['', 'x⁽¹⁾', 'x⁽²⁾', 'x⁽³⁾', 'x⁽⁴⁾'];
      L.innerHTML = '<div class="note"><b>raw</b></div>' + table(X.map(function (row, i) { return ['ex ' + (i + 1)].concat(row.map(function (v) { return f(v, Math.abs(v) < 1 ? 3 : 1); })); }).concat([['μ'].concat(cols.map(function (c) { return f(c.mu, 2); })), ['σ'].concat(cols.map(function (c) { return f(c.sd, 2); }))]), head);
      R.innerHTML = '<div class="note"><b>normalized x̂</b></div>' + table(X.map(function (_, i) { return ['ex ' + (i + 1)].concat(cols.map(function (c) { return f(c.xh[i], 2); })); }).concat([['mean'].concat(cols.map(function (c) { return f(ML.mean(c.xh), 2); })), ['var'].concat(cols.map(function (c) { return f(ML.variance(c.xh), 2); }))]), head);
      out.innerHTML = 'Four columns, four separate (μ, σ) pairs: <b>' + cols.map(function (c) { return '(' + f(c.mu, 1) + ', ' + f(c.sd, 2) + ')'; }).join(' ') + '</b>. After normalization every column has mean 0 and variance 1; rows are never mixed.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch14: singular covariance when m ≤ d
  function eigSym(A) {                        // Jacobi eigenvalues of a small symmetric matrix
    const n = A.length, a = A.map(function (r) { return r.slice(); });
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0; for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-30) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      }
    }
    return a.map(function (r, i) { return r[i]; }).sort(function (x, y) { return y - x; });
  }
  W.singularCov = function (host) {
    Book.head(host, 'A small minibatch cannot estimate a big covariance',
      'd = 12 correlated activations, a minibatch of m examples. Bars: the eigenvalues of the estimated 12×12 covariance (how much spread in each independent direction). Zero bars are directions the batch cannot see; whitening would divide by them. Per-feature variances (dots) stay fine.');
    const st = { m: 6 }, ctl = el('div', { class: 'controls' }, host); let seed = 3;
    Book.slider(ctl, 'minibatch size m', 2, 40, 1, st.m, function (v) { st.m = v; draw(); });
    Book.button(ctl, 'New minibatch', function () { seed++; draw(); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    const d = 12, r0 = ML.rng(77), Mx = Array.from({ length: d }, function () { return Array.from({ length: d }, function () { return r0.normal(0, 0.4); }); });
    function draw() {
      const r = ML.rng(seed * 13 + st.m), X = [];
      for (let i = 0; i < st.m; i++) { const z = Array.from({ length: d }, function () { return r.normal(0, 1); }); X.push(z.map(function (_, j) { let s = z[j]; for (let k = 0; k < d; k++) s += Mx[j][k] * z[k]; return s; })); }
      const mu = Array.from({ length: d }, function (_, j) { return ML.mean(X.map(function (x) { return x[j]; })); });
      const C = Array.from({ length: d }, function (_, j) { return Array.from({ length: d }, function (_, k) { let s = 0; X.forEach(function (x) { s += (x[j] - mu[j]) * (x[k] - mu[k]); }); return s / st.m; }); });
      const ev = eigSym(C), zero = ev.filter(function (v) { return v < 1e-9 * (ev[0] || 1); }).length;
      Book.chart.bars(view, ev.map(function (v, i) { return { label: String(i + 1), value: Math.max(0, v), note: v < 1e-9 * (ev[0] || 1) ? '0' : f(v, 1), color: v < 1e-9 * (ev[0] || 1) ? '#c0392b' : 'var(--accent)' }; }), { height: 210, ylabel: 'eigenvalue' });
      const vars = C.map(function (row, j) { return row[j]; });
      out.innerHTML = 'm = <b>' + st.m + '</b>: rank of the estimate = min(m − 1, d) = <b>' + Math.min(st.m - 1, d) + '</b>, so <b class="' + (zero ? 'bad' : 'ok') + '">' + zero + '</b> zero eigenvalue' + (zero === 1 ? '' : 's') + (zero ? ': the covariance is singular and Cov^(−1/2) does not exist.' : ': invertible (but still noisy).') + ' Smallest per-feature variance: ' + f(Math.min.apply(null, vars), 2) + (st.m > 1 ? ' (positive: per-feature normalization works for any m > 1).' : '');
    }
    draw();
  };

  // ------------------------------------------------------------------ ch15: Algorithm 1 stepper
  W.bnStepper = function (host) {
    Book.head(host, 'Run Algorithm 1 by hand',
      'Set the four values of one activation in a minibatch (m = 4), and γ, β. Press “Next line” to execute Algorithm 1 one line at a time.');
    const st = { x: [1, 2, 3, 6], g: 2, b: 1, eps: 0, line: 0 }, ctl = el('div', { class: 'controls' }, host);
    st.x.forEach(function (v, i) { Book.slider(ctl, 'x' + SUB[i], -5, 10, 0.5, v, function (val) { st.x[i] = val; draw(); }, function (val) { return val.toFixed(1); }); });
    Book.slider(ctl, 'γ', -3, 3, 0.1, st.g, function (v) { st.g = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'β', -3, 3, 0.1, st.b, function (v) { st.b = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.select(ctl, 'ε', [{ value: '0', label: '0' }, { value: '0.00001', label: '10⁻⁵ (PyTorch)' }, { value: '0.1', label: '0.1' }], '0', function (v) { st.eps = +v; draw(); });
    const c2 = el('div', { class: 'controls' }, host);
    Book.button(c2, 'Next line ▶', function () { st.line = Math.min(4, st.line + 1); draw(); }, 'primary');
    Book.button(c2, 'Run all', function () { st.line = 4; draw(); });
    Book.button(c2, 'Reset', function () { st.line = 0; draw(); });
    const algo = el('div', { class: 'algo' }, host), tab = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const R = bn1(st.x, st.g, st.b, st.eps), L = st.line;
      const lines = ['1 · μ_B ← (1/m) Σ x_i &nbsp;// mini-batch mean', '2 · σ²_B ← (1/m) Σ (x_i − μ_B)² &nbsp;// mini-batch variance', '3 · x̂_i ← (x_i − μ_B) / √(σ²_B + ε) &nbsp;// normalize', '4 · y_i ← γ·x̂_i + β &nbsp;// scale and shift'];
      algo.innerHTML = lines.map(function (s, i) { return '<div' + (i + 1 === L ? ' class="on"' : '') + (i + 1 > L ? ' style="opacity:.45"' : '') + '>' + s + '</div>'; }).join('');
      const rows = st.x.map(function (x, i) { return ['x' + SUB[i] + ' = ' + f(x, 1), L >= 2 ? f(x - R.mu, 3) : '·', L >= 2 ? f((x - R.mu) * (x - R.mu), 3) : '·', L >= 3 ? f(R.xh[i], 4) : '·', L >= 4 ? f(R.y[i], 4) : '·']; });
      tab.innerHTML = table(rows, ['value', 'x_i − μ_B', '(x_i − μ_B)²', 'x̂_i', 'y_i'], function (j) { return j === L; });
      const msg = ['Press “Next line”.',
        'Line 1: μ_B = (' + st.x.map(function (v) { return f(v, 1); }).join(' + ') + ') / 4 = <b>' + f(R.mu, 4) + '</b>',
        'Line 2: σ²_B = (sum of squared deviations) / 4 = <b>' + f(R.v, 4) + '</b>',
        'Line 3: divide each deviation by √(σ²_B + ε) = √(' + f(R.v, 4) + ' + ' + st.eps + ') = <b>' + f(R.sd, 4) + '</b>. Check: Σx̂ = ' + f(ML.sum(R.xh), 4) + ', mean of x̂² = ' + f(ML.mean(R.xh.map(function (z) { return z * z; })), 4),
        'Line 4: y_i = ' + f(st.g, 1) + '·x̂_i + ' + f(st.b, 1) + '. '];
      let html = msg[L];
      if (L === 4) {
        const X = NN.tensor(st.x.map(function (v) { return [v]; })), G = NN.full(1, 1, st.g), B = NN.full(1, 1, st.b);
        const Y = NN.batchNorm(X, G, B, { eps: st.eps }), err = Math.max.apply(null, R.y.map(function (y, i) { return Math.abs(y - Y.data[i]); }));
        html += 'Shared <code>NN.batchNorm</code> gives (' + Array.from(Y.data).map(function (v) { return f(v, 4); }).join(', ') + '): ' + (err < 1e-9 ? '<span class="ok">✓ identical</span>' : '<span class="bad">differs by ' + err + '</span>');
        if (!isFinite(R.y[0])) html += ' All four values are equal and ε = 0, so line 3 divides 0 by 0.';
      }
      out.innerHTML = html;
    }
    draw();
  };

  // ------------------------------------------------------------------ ch15: epsilon
  W.epsDemo = function (host) {
    Book.head(host, 'What ε does when a batch has almost no spread',
      'A batch of 8 values c + s·z, where s is the spread. With ε = 0, line 3 always rescales the tiny differences up to unit size (and at s = 0 divides 0 by 0); with ε > 0, tiny spreads are left tiny.');
    const st = { lg: -2 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'spread s', -6, 1, 0.25, st.lg, function (v) { st.lg = v; draw(); }, function (v) { return '10^' + f(v, 2).replace(/\.?0+$/, ''); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    const r = ML.rng(21), z = Array.from({ length: 8 }, function () { return r.normal(0, 1); });
    const zm = ML.mean(z), zs = Math.sqrt(ML.variance(z)), zz = z.map(function (v) { return (v - zm) / zs; });
    const epss = [[0, '#c0392b', 'ε = 0'], [1e-5, '#2b6cb0', 'ε = 10⁻⁵'], [1e-3, '#2f855a', 'ε = 10⁻³']];
    function maxxh(s, eps) { const sd = Math.sqrt(s * s + eps); return sd === 0 ? NaN : s / sd * Math.max.apply(null, zz.map(Math.abs)); }
    function draw() {
      const ser = epss.map(function (e) { const pts = []; for (let k = -6; k <= 1.001; k += 0.125) pts.push([k, maxxh(Math.pow(10, k), e[0])]); return { name: e[2], color: e[1], marker: 'none', points: pts }; });
      ser.push({ name: '', color: 'var(--ink)', points: [[st.lg, maxxh(Math.pow(10, st.lg), 1e-5)]], r: 5 });
      T.lines(view, ser, { xmin: -6, xmax: 1, ymin: 0, ymax: 2, xticks: [-6, -4, -2, 0, 1], xfmt: function (v) { return '10^' + v; }, xlabel: 'spread s of the batch', ylabel: 'largest |x̂|', height: 220 });
      const s = Math.pow(10, st.lg);
      out.innerHTML = 'spread 10^' + f(st.lg, 2).replace(/\.?0+$/, '') + ' (σ²_B = ' + (s * s).toExponential(1) + '): largest |x̂| with ε = 0 → <b>' + f(maxxh(s, 0), 3) + '</b>, with ε = 10⁻⁵ → <b>' + f(maxxh(s, 1e-5), 3) + '</b>, with ε = 10⁻³ → <b>' + f(maxxh(s, 1e-3), 3) + '</b>. ε only matters once σ²_B is comparable to it.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch16: γ, β restore the identity
  W.identityRestore = function (host) {
    Book.head(host, 'γ and β can give back the original activation',
      'One activation with mean 2 and standard deviation 3 over 1,000 examples. Left: x̂ (normalized) under the sigmoid. Right: y = γx̂ + β under the sigmoid. Move γ, β, or press “identity”.');
    const st = { g: 1, b: 0 }, ctl = el('div', { class: 'controls' }, host);
    const gs = Book.slider(ctl, 'γ', 0.1, 5, 0.05, st.g, function (v) { st.g = v; draw(); }, function (v) { return v.toFixed(2); });
    const bs = Book.slider(ctl, 'β', -4, 4, 0.05, st.b, function (v) { st.b = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'Set γ = √Var[x], β = E[x] (identity)', function () { const mv = T.mv(x); set(Math.sqrt(mv.var), mv.mean); }, 'primary');
    Book.button(ctl, 'γ = 1, β = 0', function () { set(1, 0); });
    function set(g, b) { st.g = g; st.b = b; gs.value = g; gs.dispatchEvent(new Event('input')); bs.value = b; bs.dispatchEvent(new Event('input')); draw(); }
    const g = el('div', { class: 'grid2' }, host), L = el('div', null, g), R = el('div', null, g), out = el('div', { class: 'out' }, host);
    const r = ML.rng(16), x = Array.from({ length: 1000 }, function () { return 2 + 3 * r.normal(0, 1); });
    function lin(v) { let s = 0; v.forEach(function (t) { const gg = sig(t); s += Math.abs(gg - (0.5 + t / 4)) < 0.02 ? 1 : 0; }); return s / v.length; }
    function draw() {
      const mv = T.mv(x), sd = Math.sqrt(mv.var), xh = x.map(function (v) { return (v - mv.mean) / sd; }), y = xh.map(function (v) { return st.g * v + st.b; });
      T.hist(L, xh, { lo: -10, hi: 10, bins: 50, title: 'x̂ (γ = 1, β = 0)', curve: sig, width: 300, ticks: [-10, -5, 0, 5, 10] });
      T.hist(R, y, { lo: -10, hi: 10, bins: 50, title: 'y = γx̂ + β', curve: sig, width: 300, ticks: [-10, -5, 0, 5, 10], color: '#2f855a' });
      const err = Math.max.apply(null, y.map(function (v, i) { return Math.abs(v - x[i]); }));
      out.innerHTML = 'share of inputs where the sigmoid is still almost a straight line (within 0.02 of its tangent at 0): x̂ <b>' + (100 * lin(xh)).toFixed(0) + '%</b>, y <b>' + (100 * lin(y)).toFixed(0) + '%</b> · largest |y − x| = <b>' + f(err, 4) + '</b>' + (err < 1e-3 ? ' <span class="ok">✓ y = x: BN is the identity</span>' : '');
    }
    draw();
  };

  // ------------------------------------------------------------------ ch17: batch-mates
  W.batchDependence = function (host) {
    Book.head(host, 'Same example, different batch-mates, different output',
      'One example has the value x = 1. It is put in 500 random minibatches whose other m − 1 values come from a normal distribution (mean 0, sd 1). Histogram: its normalized value x̂ in each.');
    const st = { m: 8, x: 1 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'minibatch size m', 2, 128, 1, st.m, function (v) { st.m = v; draw(); });
    Book.slider(ctl, 'the example’s x', -3, 3, 0.1, st.x, function (v) { st.x = v; draw(); }, function (v) { return v.toFixed(1); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(5 + st.m), vals = []; let last;
      for (let k = 0; k < 500; k++) { const b = [st.x]; for (let i = 1; i < st.m; i++) b.push(r.normal(0, 1)); last = bn1(b, 1, 0, 1e-5); vals.push(last.xh[0]); }
      const mv = T.mv(vals);
      T.hist(view, vals, { lo: -4, hi: 4, bins: 64, width: 600, height: 190, ticks: [-4, -2, 0, 2, 4], marks: [{ x: st.x, label: 'population value ' + f(st.x, 1), color: '#c0392b', dash: '4 3' }] });
      out.innerHTML = 'x̂ of the example: mean <b>' + f(mv.mean, 3) + '</b>, spread across batches <b>' + f(Math.sqrt(mv.var), 3) + '</b> · in the last batch Σx̂ = ' + f(ML.sum(last.xh), 6) + ' and (1/m)Σx̂² = ' + f(ML.mean(last.xh.map(function (z) { return z * z; })), 5) + (st.m === 2 ? ' · with m = 2, x̂ is always ±1: the two values are pushed to opposite sides.' : '');
    }
    draw();
  };

  // ------------------------------------------------------------------ ch18: backprop tracer
  W.bnBackprop = function (host) {
    Book.head(host, 'Trace the gradient through BN, equation by equation',
      'A minibatch of m = 4 values and upstream gradients g_i = ∂ℓ/∂y_i (the loss here is ℓ = Σ g_i·y_i, so that ∂ℓ/∂y_i is exactly g_i). Press “Next equation” to evaluate B1–B6 in order.');
    const st = { eq: 0, seed: 1, g: 1.5, b: 0.2 }, ctl = el('div', { class: 'controls' }, host);
    Book.button(ctl, 'Next equation ▶', function () { st.eq = Math.min(6, st.eq + 1); draw(); }, 'primary');
    Book.button(ctl, 'All', function () { st.eq = 6; draw(); });
    Book.button(ctl, 'New batch and gradients', function () { st.seed++; st.eq = 0; draw(); });
    Book.slider(ctl, 'γ', -2, 3, 0.1, st.g, function (v) { st.g = v; draw(); }, function (v) { return v.toFixed(1); });
    const scal = el('div', { class: 'out' }, host), tab = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    const eps = 1e-5;
    function draw() {
      const r = ML.rng(st.seed * 17), x = [0, 1, 2, 3].map(function () { return +(r.normal(1, 2)).toFixed(1); }), gy = [0, 1, 2, 3].map(function () { return +(r.normal(0, 1)).toFixed(1); });
      const m = 4, R = bn1(x, st.g, st.b, eps), mu = R.mu, v = R.v, sd = R.sd;
      const dxh = gy.map(function (gi) { return gi * st.g; });                                                       // B1
      const dvar = ML.sum(dxh.map(function (d, i) { return d * (x[i] - mu) * -0.5 * Math.pow(v + eps, -1.5); }));    // B2
      const dmu = ML.sum(dxh.map(function (d) { return d * -1 / sd; }));                                             // B3
      const t1 = dxh.map(function (d) { return d / sd; }), t2 = x.map(function (xi) { return dvar * 2 * (xi - mu) / m; }), t3 = x.map(function () { return dmu / m; });
      const dx = t1.map(function (a, i) { return a + t2[i] + t3[i]; });                                             // B4
      const dg = ML.sum(gy.map(function (gi, i) { return gi * R.xh[i]; })), db = ML.sum(gy);                       // B5, B6
      // finite differences on ℓ(x) = Σ g_i·BN(x)_i
      const loss = function (xx) { const q = bn1(xx, st.g, st.b, eps); return ML.sum(q.y.map(function (y, i) { return gy[i] * y; })); };
      const fd = x.map(function (_, i) { const h = 1e-6, a = x.slice(), b = x.slice(); a[i] += h; b[i] -= h; return (loss(a) - loss(b)) / (2 * h); });
      // the shared autograd
      const X = NN.tensor(x.map(function (q) { return [q]; }), true), G = NN.full(1, 1, st.g, true), B = NN.full(1, 1, st.b, true);
      const Y = NN.batchNorm(X, G, B, { eps: eps }), Lt = NN.sum(NN.mul(Y, NN.tensor(gy.map(function (q) { return [q]; })))); Lt.backward();
      const E = st.eq, show = function (k, s) { return E >= k ? s : '·'; };
      scal.innerHTML = 'μ_B = ' + f(mu, 3) + ', σ²_B = ' + f(v, 3) + ', √(σ²_B+ε) = ' + f(sd, 3) + ' · <b>B2</b> ∂ℓ/∂σ²_B = ' + show(2, '<b>' + f(dvar, 4) + '</b>') + ' · <b>B3</b> ∂ℓ/∂μ_B = ' + show(3, '<b>' + f(dmu, 4) + '</b>') + ' · <b>B5</b> ∂ℓ/∂γ = ' + show(5, '<b>' + f(dg, 4) + '</b>') + ' · <b>B6</b> ∂ℓ/∂β = ' + show(6, '<b>' + f(db, 4) + '</b>');
      const rows = x.map(function (xi, i) {
        return ['x' + SUB[i] + ' = ' + f(xi, 1), f(R.xh[i], 3), f(gy[i], 1), show(1, f(dxh[i], 3)), show(4, f(t1[i], 3)), show(4, f(t2[i], 3)), show(4, f(t3[i], 3)), show(4, '<b>' + f(dx[i], 4) + '</b>'), show(4, f(fd[i], 4)), show(4, f(X.grad[i], 4))];
      });
      tab.innerHTML = table(rows, ['x_i', 'x̂_i', 'g_i = ∂ℓ/∂y_i', 'B1 ∂ℓ/∂x̂_i', 'B4 term 1 (via x̂_i)', 'term 2 (via σ²)', 'term 3 (via μ)', 'B4 ∂ℓ/∂x_i', 'finite diff.', 'NN autograd'], function (j) { return (E === 1 && j === 3) || (E === 4 && j === 7); });
      if (E >= 6) {
        const err = Math.max.apply(null, dx.map(function (d, i) { return Math.max(Math.abs(d - fd[i]), Math.abs(d - X.grad[i])); }));
        const eg = Math.max(Math.abs(dg - G.grad[0]), Math.abs(db - B.grad[0]));
        out.innerHTML = 'B1–B6 vs finite differences and vs the shared autograd: largest difference <b>' + err.toExponential(1) + '</b> for ∂ℓ/∂x, <b>' + eg.toExponential(1) + '</b> for ∂ℓ/∂γ, ∂ℓ/∂β <span class="ok">✓</span> · Σ ∂ℓ/∂x_i = ' + f(ML.sum(dx), 6) + '. Treating the statistics as constants would give term 1 alone, which is wrong: it ignores two of the three paths.';
      } else out.innerHTML = ['Press “Next equation”.', 'B1: multiply each upstream gradient by γ = ' + f(st.g, 1) + '.', 'B2: one number for the whole batch, a sum over i.', 'B3: also one number, a sum over i.', 'B4: the three paths of each x_i, added. Compare with finite differences and autograd.', 'B5: sum of g_i·x̂_i.', ''][E];
    }
    draw();
  };

  // ------------------------------------------------------------------ ch19: the gradient as a projection
  W.gradProjection = function (host) {
    Book.head(host, 'The BN gradient = upstream gradient minus a shift and a stretch',
      'A minibatch of m = 8. Grey: the upstream gradient g. Orange: g with its average removed. Blue: also with its projection onto x̂ removed, then scaled by γ/σ: this is ∂ℓ/∂x. Push the upstream gradient towards a pure shift or a pure stretch.');
    const st = { shift: 0.6, stretch: 0.5, noise: 1, seed: 2 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'shift part', -2, 2, 0.1, st.shift, function (v) { st.shift = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'stretch part', -2, 2, 0.1, st.stretch, function (v) { st.stretch = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'other part', 0, 2, 0.1, st.noise, function (v) { st.noise = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.button(ctl, 'New batch', function () { st.seed++; draw(); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(st.seed * 5), x = Array.from({ length: 8 }, function () { return r.normal(2, 1.5); }), gam = 1.3;
      const R = bn1(x, gam, 0, 1e-5), e = Array.from({ length: 8 }, function () { return r.normal(0, 1); });
      const g = R.xh.map(function (z, i) { return st.shift + st.stretch * z + st.noise * e[i]; });
      const gm = ML.mean(g), gc = g.map(function (v) { return v - gm; }), gx = ML.mean(g.map(function (v, i) { return v * R.xh[i]; }));
      const dx = gc.map(function (v, i) { return gam / R.sd * (v - R.xh[i] * gx); });
      const X = NN.tensor(x.map(function (q) { return [q]; }), true), Y = NN.batchNorm(X, NN.full(1, 1, gam, true), NN.zeros(1, 1, true), {});
      NN.sum(NN.mul(Y, NN.tensor(g.map(function (q) { return [q]; })))).backward();
      sbars(view, [{ name: 'g = ∂ℓ/∂y', color: 'var(--muted)', values: g }, { name: 'g − ḡ', color: '#b7791f', values: gc }, { name: '∂ℓ/∂x = (γ/σ)(g − ḡ − x̂·mean(g·x̂))', color: '#2b6cb0', values: dx }], { height: 220 });
      const err = Math.max.apply(null, dx.map(function (d, i) { return Math.abs(d - X.grad[i]); }));
      out.innerHTML = 'ḡ = <b>' + f(gm, 3) + '</b>, mean(g·x̂) = <b>' + f(gx, 3) + '</b> · checks: Σ ∂ℓ/∂x_i = <b>' + f(ML.sum(dx), 6) + '</b>, Σ x̂_i·∂ℓ/∂x_i = <b>' + f(ML.sum(dx.map(function (d, i) { return d * R.xh[i]; })), 6) + '</b> · matches the shared autograd to ' + err.toExponential(1) + (st.noise === 0 ? ' · with only shift and stretch parts, ∂ℓ/∂x is all zero: BN cancels both.' : '');
    }
    draw();
  };
})();
