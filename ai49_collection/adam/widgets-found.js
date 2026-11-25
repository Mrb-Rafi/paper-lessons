/* Adam Interactive Web Book — helpers (window.AD) and Part I widgets: foundations.
   Uses Book (core.js), ML (math.js), NN (nn.js) and Opt (opt.js). */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, Opt = window.Opt;

  // ================================================================== helpers shared by all widget files
  const AD = window.AD = {};
  AD.f = function (x, d) { if (!isFinite(x)) return '∞'; d = d == null ? 3 : d; const a = Math.abs(x); if (a !== 0 && (a < Math.pow(10, -d) || a >= 1e6)) return x.toExponential(Math.max(1, d - 1)).replace('e', '×10^').replace(/\^\+?(-?)(\d+)/, function (_, s, n) { return '<sup>' + (s ? '−' : '') + n + '</sup>'; }); return x.toFixed(d); };
  const SUP = { '-': '⁻', '−': '⁻', '.': '·', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  AD.p10 = function (k) { return '10' + String(k).split('').map(function (c) { return SUP[c] || c; }).join(''); };     // 10 to the power k, with Unicode superscripts
  AD.ft = function (x, d) { return String(AD.f(x, d)).replace(/10<sup>(.*?)<\/sup>/, function (_, k) { return AD.p10(k); }); };       // plain-text version
  AD.rgb = function (hex, a) { hex = (hex || '#1f7a5c').replace('#', ''); if (hex.length === 3) hex = hex.split('').map(function (c) { return c + c; }).join(''); const n = parseInt(hex, 16); return 'rgba(' + (n >> 16 & 255) + ',' + (n >> 8 & 255) + ',' + (n & 255) + ',' + a + ')'; };
  AD.legend = function (host, items) {
    const L = el('div', { class: 'legend' }, host);
    items.forEach(function (it) { const s = el('span', null, L); el('i', { style: 'background:' + it.color + (it.dash ? ';opacity:.6' : '') }, s); s.appendChild(document.createTextNode(it.name)); });
    return L;
  };
  AD.checks = function (parent, items, onchange) {     // items [{key, label, on}] → checkboxes
    items.forEach(function (it) {
      const lab = el('label', null, parent), c = el('input', { type: 'checkbox' }, lab); c.checked = !!it.on;
      lab.appendChild(document.createTextNode(' ' + it.label));
      c.addEventListener('change', function () { it.on = c.checked; onchange(); });
    });
  };
  // line chart whose series names go into a legend under the chart (end-of-line labels collide when curves meet)
  AD.lines = function (host, series, o) {
    o = o || {};
    const named = series.filter(function (s) { return s.name; });
    const useLegend = o.legend !== false && named.length >= 2;
    const ser = useLegend ? series.map(function (s) { return Object.assign({}, s, { name: null }); }) : series;
    const res = Book.chart.lines(host, ser, useLegend ? Object.assign({}, o, { mr: 20 }) : o);
    if (useLegend) AD.legend(host, named.map(function (s) { return { name: s.name, color: s.color, dash: !!s.dash }; }));
    return res;
  };
  // log-scale line chart: plots log10(y), with the axis on whole powers of ten
  AD.logLines = function (host, series, o) {
    o = Object.assign({}, o || {});
    const tr = series.map(function (s) { return Object.assign({}, s, { points: s.points.filter(function (p) { return p[1] > 0 && isFinite(p[1]); }).map(function (p) { return [p[0], Math.log10(p[1])]; }) }); }).filter(function (s) { return s.points.length; });
    let lo = Infinity, hi = -Infinity;
    tr.forEach(function (s) { s.points.forEach(function (p) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); }); });
    if (o.ymin != null) lo = Math.max(lo, Math.log10(o.ymin)); if (o.ymax != null) hi = Math.min(hi, Math.log10(o.ymax));
    if (o.ymin != null && !isFinite(lo)) lo = Math.log10(o.ymin); if (o.ymax != null && !isFinite(hi)) hi = Math.log10(o.ymax);
    lo = Math.floor(lo); hi = Math.ceil(hi); if (hi <= lo) hi = lo + 1;
    const step = Math.ceil((hi - lo) / 6); hi = lo + step * Math.ceil((hi - lo) / step);
    tr.forEach(function (s) { s.points = s.points.map(function (p) { return [p[0], Book.clamp(p[1], lo, hi)]; }); });
    o.ymin = lo; o.ymax = hi; o.yticks = (hi - lo) / step;
    o.yfmt = function (v) { const r = Math.round(v); return r === 0 ? '1' : r === 1 ? '10' : AD.p10(r); };
    return AD.lines(host, tr, o);
  };
  // a time-sliced runner for long computations: jobs have .run(ms) → done; calls tick() between slices
  AD.runner = function () {
    let token = 0;
    return {
      start: function (jobs, tick, done) {
        const my = ++token;
        (function loop() {
          if (my !== token) return;
          let all = true; const t0 = performance.now();
          for (let i = 0; i < jobs.length && performance.now() - t0 < 30; i++) if (!jobs[i].done) { jobs[i].run(12); if (!jobs[i].done) all = false; }
          all = jobs.every(function (j) { return j.done; });
          tick(); if (all) { if (done) done(); return; }
          setTimeout(loop, 15);
        })();
      },
      stop: function () { token++; }
    };
  };

  // Contour map of a 2-D function on a canvas, with paths drawn on top.
  // spec: {f(x,y), box:[x0,x1,y0,y1], min:[x,y]} ; returns {draw(series, extra), toXY(evt)}
  AD.contour = function (host, spec, o) {
    o = o || {};
    const Wd = o.width || 560, Ht = o.height || 330, dpr = Math.min(2, window.devicePixelRatio || 1);
    const cv = el('canvas', { width: Wd * dpr, height: Ht * dpr, style: 'width:100%;max-width:' + Wd + 'px;aspect-ratio:' + Wd + '/' + Ht + ';height:auto;border-radius:8px;border:1px solid var(--line);background:var(--panel);touch-action:none' }, host);
    const ctx = cv.getContext('2d'); ctx.scale(dpr, dpr);
    let bg = null, curSpec = spec;
    const X = function (x) { const b = curSpec.box; return (x - b[0]) / (b[1] - b[0]) * Wd; };
    const Y = function (y) { const b = curSpec.box; return Ht - (y - b[2]) / (b[3] - b[2]) * Ht; };
    function paintBg() {
      const b = curSpec.box, N = 120, M = Math.round(N * Ht / Wd), vals = new Float64Array(N * M);
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) {
        const x = b[0] + (j + 0.5) / N * (b[1] - b[0]), y = b[3] - (i + 0.5) / M * (b[3] - b[2]), v = curSpec.f(x, y);
        vals[i * N + j] = v; if (v < lo) lo = v; if (v > hi) hi = v;
      }
      const acc = Book.css('--accent') || '#1f7a5c', off = curSpec.logOffset || (hi - lo) * 1e-3 + 1e-9;
      const L0 = Math.log(off), L1 = Math.log(hi - lo + off), bands = 14;
      const c2 = document.createElement('canvas'); c2.width = Wd * dpr; c2.height = Ht * dpr; const x2 = c2.getContext('2d');
      const cw = Wd * dpr / N, ch = Ht * dpr / M;
      for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) {
        const t = (Math.log(vals[i * N + j] - lo + off) - L0) / (L1 - L0), band = Math.min(bands - 1, Math.floor(t * bands));
        x2.fillStyle = AD.rgb(acc, 0.04 + 0.05 * (bands - 1 - band));
        const px0 = Math.round(j * cw), py0 = Math.round(i * ch); x2.fillRect(px0, py0, Math.round((j + 1) * cw) - px0, Math.round((i + 1) * ch) - py0);   // integer cells: no double-painted seams
      }
      bg = c2;
    }
    function draw(series, extra) {
      if (!bg) paintBg();
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height); ctx.drawImage(bg, 0, 0); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (curSpec.min) { ctx.fillStyle = Book.css('--ink') || '#222'; ctx.font = '16px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', X(curSpec.min[0]), Y(curSpec.min[1])); }
      (series || []).forEach(function (s) {
        const p = s.path; if (!p || !p.length) return;
        ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.setLineDash(s.dash || []); ctx.globalAlpha = s.alpha || 0.95;
        ctx.beginPath(); let started = false;
        for (let k = 0; k < p.length; k++) {
          if (!isFinite(p[k][0]) || !isFinite(p[k][1])) break;
          const px = Book.clamp(X(p[k][0]), -50, Wd + 50), py = Book.clamp(Y(p[k][1]), -50, Ht + 50);
          if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
        }
        ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
        if (s.dots) { ctx.fillStyle = s.color; p.forEach(function (q) { if (isFinite(q[0])) { ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), 1.6, 0, 7); ctx.fill(); } }); }
        let last = null; for (let k = p.length - 1; k >= 0; k--) if (isFinite(p[k][0]) && isFinite(p[k][1])) { last = p[k]; break; }
        if (last) { ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(Book.clamp(X(last[0]), 4, Wd - 4), Book.clamp(Y(last[1]), 4, Ht - 4), 4.5, 0, 7); ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.stroke(); }
      });
      if (extra) extra(ctx, X, Y);
      const st = (series && series[0] && series[0].path && series[0].path[0]);
      if (st) { ctx.strokeStyle = Book.css('--ink') || '#222'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(X(st[0]), Y(st[1]), 6, 0, 7); ctx.stroke(); }
    }
    function toXY(evt) {
      const r = cv.getBoundingClientRect(), b = curSpec.box;
      return [b[0] + (evt.clientX - r.left) / r.width * (b[1] - b[0]), b[3] - (evt.clientY - r.top) / r.height * (b[3] - b[2])];
    }
    return { canvas: cv, draw: draw, toXY: toXY, X: X, Y: Y, setSpec: function (s) { curSpec = s; bg = null; } };
  };

  // ================================================================== ch-objective
  W.fitLine = function (host) {
    Book.head(host, 'Choosing parameters to make the loss small',
      'Eight data points and a straight line y = w·x + b. Move w and b (the two parameters). The loss is the average squared vertical gap. The map on the right shows the loss for every possible (w, b): your current choice is the dot; the star is the best possible choice.');
    const r = ML.rng(5), xs = ML.linspace(-2, 2, 8), ys = xs.map(function (x) { return 0.8 * x + 0.5 + r.normal(0, 0.35); });
    const loss = function (w, b) { let s = 0; xs.forEach(function (x, i) { const e = w * x + b - ys[i]; s += e * e; }); return s / xs.length; };
    // least-squares optimum
    const mx = ML.mean(xs), my = ML.mean(ys); let sxy = 0, sxx = 0; xs.forEach(function (x, i) { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) * (x - mx); });
    const wst = sxy / sxx, bst = my - wst * mx;
    const st = { w: -0.6, b: 1.6 };
    const ctl = el('div', { class: 'controls' }, host);
    const sw = Book.slider(ctl, 'w (slope)', -1.5, 2.5, 0.01, st.w, function (v) { st.w = v; draw(); }, function (v) { return v.toFixed(2); });
    const sb = Book.slider(ctl, 'b (intercept)', -1.5, 2.5, 0.01, st.b, function (v) { st.b = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'Jump to the best (w, b)', function () { st.w = wst; st.b = bst; sw.value = wst; sb.value = bst; sw.dispatchEvent(new Event('input')); sb.dispatchEvent(new Event('input')); });
    const g = el('div', { class: 'grid2' }, host), left = el('div', null, g), right = el('div', null, g), out = el('div', { class: 'out' }, host);
    const map = AD.contour(right, { f: loss, box: [-1.5, 2.5, -1.5, 2.5], min: [wst, bst] }, { width: 300, height: 300 });
    map.canvas.addEventListener('click', function (e) { const p = map.toXY(e); st.w = p[0]; st.b = p[1]; sw.value = p[0]; sb.value = p[1]; sw.dispatchEvent(new Event('input')); sb.dispatchEvent(new Event('input')); });
    function draw() {
      left.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 300 300' }, left), X = function (x) { return 20 + (x + 2.4) / 4.8 * 270; }, Y = function (y) { return 280 - (y + 1.6) / 4.2 * 270; };
      S('line', { x1: X(-2.4), x2: X(2.4), y1: Y(0), y2: Y(0), stroke: 'var(--line)' }, svg); S('line', { x1: X(0), x2: X(0), y1: Y(-1.6), y2: Y(2.6), stroke: 'var(--line)' }, svg);
      xs.forEach(function (x, i) { const p = st.w * x + st.b; S('line', { x1: X(x), x2: X(x), y1: Y(ys[i]), y2: Y(p), stroke: Book.palette[1], 'stroke-dasharray': '3 2' }, svg); });
      S('line', { x1: X(-2.4), x2: X(2.4), y1: Y(st.w * -2.4 + st.b), y2: Y(st.w * 2.4 + st.b), stroke: 'var(--accent)', 'stroke-width': 2.5 }, svg);
      xs.forEach(function (x, i) { S('circle', { cx: X(x), cy: Y(ys[i]), r: 4.5, fill: Book.palette[0], stroke: 'var(--panel)' }, svg); });
      S('text', { x: 150, y: 296, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: 'data (blue) and the line; red dashes = errors' }, svg);
      map.draw([{ path: [[st.w, st.b]], color: Book.palette[1] }]);
      out.innerHTML = 'θ = (w, b) = (' + st.w.toFixed(2) + ', ' + st.b.toFixed(2) + ') → loss f(θ) = <b>' + loss(st.w, st.b).toFixed(3) + '</b>' +
        '<br>best possible: θ* = (' + wst.toFixed(2) + ', ' + bst.toFixed(2) + ') with loss ' + loss(wst, bst).toFixed(3) + ' · click the map to jump there' +
        '<br><span class="note">Training = searching this map for its lowest point. A real network has millions of parameters instead of two, so nobody can draw the map: the search must use only local information.</span>';
    }
    draw();
  };

  // ================================================================== ch-gradient
  W.gradField = function (host) {
    Book.head(host, 'The gradient points uphill',
      'Click anywhere on the map. The arrow shows minus the gradient, −∇f: the steepest way down. Its two components are the partial derivatives, each measured by nudging one parameter by h while holding the other fixed.');
    const sk = { key: 'ravine' }, st = { p: [-3, 1.5], h: 0.01 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'surface', ['ravine', 'bowl', 'rosenbrock'].map(function (k) { return { value: k, label: Opt.surfaces[k].label }; }), sk.key, function (v) { sk.key = v; const s = Opt.surfaces[v]; map.setSpec(s); st.p = s.start.slice(); draw(); });
    Book.slider(ctl, 'nudge h', 0.001, 0.5, 0.001, st.h, function (v) { st.h = v; draw(); }, function (v) { return v.toFixed(3); });
    const map = AD.contour(host, Opt.surfaces[sk.key]), out = el('div', { class: 'out' }, host);
    map.canvas.addEventListener('click', function (e) { st.p = map.toXY(e); draw(); });
    function draw() {
      const s = Opt.surfaces[sk.key], x = st.p[0], y = st.p[1], g = s.g(x, y), h = st.h;
      const fdx = (s.f(x + h, y) - s.f(x, y)) / h, fdy = (s.f(x, y + h) - s.f(x, y)) / h, n = Math.hypot(g[0], g[1]);
      map.draw([], function (ctx, X, Y) {
        // field of small arrows
        const b = s.box; ctx.strokeStyle = AD.rgb(Book.css('--muted') || '#888', 0.55); ctx.lineWidth = 1;
        for (let i = 0; i < 11; i++) for (let j = 0; j < 7; j++) {
          const px = b[0] + (i + 0.5) / 11 * (b[1] - b[0]), py = b[2] + (j + 0.5) / 7 * (b[3] - b[2]), q = s.g(px, py), m = Math.hypot(q[0], q[1]) || 1;
          const L = 14, dx = -q[0] / m * L, dy = q[1] / m * L, x0 = X(px), y0 = Y(py);
          ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + dx, y0 + dy); ctx.stroke();
          ctx.beginPath(); ctx.arc(x0 + dx, y0 + dy, 1.5, 0, 7); ctx.fill();
        }
        const L = 55, x0 = X(x), y0 = Y(y), ex = x0 - g[0] / (n || 1) * L, ey = y0 + g[1] / (n || 1) * L;
        ctx.strokeStyle = Book.palette[1]; ctx.fillStyle = Book.palette[1]; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(ex, ey); ctx.stroke();
        const a = Math.atan2(ey - y0, ex - x0); ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex - 10 * Math.cos(a - 0.4), ey - 10 * Math.sin(a - 0.4)); ctx.lineTo(ex - 10 * Math.cos(a + 0.4), ey - 10 * Math.sin(a + 0.4)); ctx.fill();
        ctx.beginPath(); ctx.arc(x0, y0, 5, 0, 7); ctx.fill();
      });
      out.innerHTML = 'at θ = (' + x.toFixed(2) + ', ' + y.toFixed(2) + '):  f = ' + AD.f(s.f(x, y)) +
        '<br>∂f/∂x: exact ' + AD.f(g[0]) + ' · nudge estimate (f(x+h, y) − f(x, y))/h = ' + AD.f(fdx) +
        '<br>∂f/∂y: exact ' + AD.f(g[1]) + ' · nudge estimate = ' + AD.f(fdy) +
        '<br>∇f = (' + AD.f(g[0]) + ', ' + AD.f(g[1]) + '), length ' + AD.f(n) + ' · the arrow is −∇f, the direction gradient descent moves' +
        '<br><span class="note">On the ravine the two partial derivatives differ hugely: the arrow points almost straight across the valley, not along it towards the star.</span>';
    }
    draw();
  };

  // ================================================================== ch-stochastic
  W.minibatchNoise = function (host) {
    Book.head(host, 'A minibatch gives a noisy gradient',
      'The data are 200 points; the model is y = w·x with loss = mean squared error. The full gradient (black line) averages all 200 points. A minibatch gradient averages only B random points. Draw 400 minibatches and look at the spread.');
    const r0 = ML.rng(8), N = 200, xs = [], ys = [];
    for (let i = 0; i < N; i++) { const x = r0.normal(0, 1); xs.push(x); ys.push(1.5 * x + r0.normal(0, 1)); }
    const gi = function (w, i) { return 2 * (w * xs[i] - ys[i]) * xs[i]; };
    const st = { w: 0, B: 8, seed: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'weight w', -1, 3, 0.05, st.w, function (v) { st.w = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'minibatch size B', 1, 200, 1, st.B, function (v) { st.B = v; draw(); }, function (v) { return v; });
    Book.button(ctl, 'Draw again', function () { st.seed++; draw(); }, 'primary');
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(st.seed), samples = [];
      let full = 0; for (let i = 0; i < N; i++) full += gi(st.w, i) / N;
      for (let k = 0; k < 400; k++) { const idx = r.shuffle(ML.range(N)).slice(0, st.B); let s = 0; idx.forEach(function (i) { s += gi(st.w, i); }); samples.push(s / st.B); }
      const lo = full - 8, hi = full + 8, bins = 40, h = new Array(bins).fill(0);
      samples.forEach(function (v) { const b = Math.floor((v - lo) / (hi - lo) * bins); if (b >= 0 && b < bins) h[b]++; });
      plot.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 640 200' }, plot), mxh = Math.max.apply(null, h) || 1;
      h.forEach(function (c, b) { S('rect', { x: 20 + b * 15, y: 170 - c / mxh * 150, width: 13, height: c / mxh * 150, fill: 'var(--accent)', opacity: 0.75 }, svg); });
      const X = function (v) { return 20 + (v - lo) / (hi - lo) * 600; };
      S('line', { x1: X(full), x2: X(full), y1: 10, y2: 172, stroke: 'var(--ink)', 'stroke-width': 2.5 }, svg);
      S('line', { x1: 20, x2: 620, y1: 170, y2: 170, stroke: 'var(--muted)' }, svg);
      [-8, -4, 0, 4, 8].forEach(function (d) { S('text', { x: X(full + d), y: 188, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: (full + d).toFixed(1) }, svg); });
      S('text', { x: X(full) + 6, y: 22, 'font-size': 11.5, class: 'svg-text', text: 'full gradient = E[g]' }, svg);
      const m = ML.mean(samples), sd = Math.sqrt(ML.variance(samples));
      const sdTheory = Math.sqrt(ML.variance(ML.range(N).map(function (i) { return gi(st.w, i); })) / st.B);
      out.innerHTML = 'full-data gradient at w = ' + st.w.toFixed(2) + ': <b>' + full.toFixed(3) + '</b>' + (Math.abs(full) < 0.15 ? ' (near the best w ≈ 1.5, the true slope)' : '') +
        '<br>400 minibatch gradients: average <b>' + m.toFixed(3) + '</b>, spread (std) <b>' + sd.toFixed(3) + '</b> · theory: std ≈ ' + sdTheory.toFixed(3) + ' = (std of one point)/√B' +
        '<br><span class="note">Each minibatch gradient is a random draw whose average (expected value) is the full gradient. Bigger B → less noise, but more work per step. Near the best w the noise is as large as, or larger than, the true gradient.</span>';
    }
    draw();
  };

  // ================================================================== ch-stepsize
  W.ravineLR = function (host) {
    Book.head(host, 'One learning rate for two very different directions',
      'Plain gradient descent on the ravine f = 0.05x² + 2y². Along y the walls are 40× steeper than along x. Raise the learning rate until y starts to bounce, then watch how slowly x moves.');
    const st = { lr: 0.3, steps: 60 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'learning rate', 0.02, 0.6, 0.01, st.lr, function (v) { st.lr = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'steps', 10, 200, 5, st.steps, function (v) { st.steps = v; draw(); }, function (v) { return v; });
    const map = AD.contour(host, Opt.surfaces.ravine, { height: 300 }), plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const R = Opt.runSurface('ravine', 'sgd', { lr: st.lr }, st.steps, 0, 1);
      map.draw([{ path: R.path, color: Book.palette[1], dots: true }]);
      AD.lines(plot, [
        { name: 'x', color: Book.palette[0], marker: 'none', points: R.path.map(function (p, t) { return [t, Book.clamp(p[0], -6, 6)]; }) },
        { name: 'y', color: Book.palette[1], marker: 'none', points: R.path.map(function (p, t) { return [t, Book.clamp(p[1], -6, 6)]; }) }
      ], { width: 640, height: 200, ymin: -5, ymax: 3, xlabel: 'step', ylabel: 'coordinate', xticks: [0, Math.round(st.steps / 2), st.steps] });
      const fx = 1 - 0.1 * st.lr, fy = 1 - 4 * st.lr;
      out.innerHTML = 'each step multiplies x by (1 − 0.1·lr) = <b>' + fx.toFixed(3) + '</b> and y by (1 − 4·lr) = <b>' + fy.toFixed(3) + '</b>' +
        (Math.abs(fy) > 1 ? ' → |factor| > 1: <b>y blows up</b>' : Math.abs(fy) > 0.7 && fy < 0 ? ' → y flips sign every step (zig-zag)' : '') +
        '<br>steps for x to shrink to 1 % of its start: <b>' + (fx > 0 && fx < 1 ? Math.ceil(Math.log(0.01) / Math.log(fx)) : '∞') + '</b> · largest safe learning rate for y: 2/4 = 0.5' +
        '<br><span class="note">The steep direction caps the learning rate; the flat direction then crawls. One global learning rate cannot suit both. Adam gives every parameter its own step size.</span>';
    }
    draw();
  };

  W.scaleDiff = function (host) {
    Book.head(host, 'Two weights, gradients 100× apart',
      'In a real network some weights get large gradients and some tiny ones (paper §6.3 notes this for the layers of a convolutional net). Set the two gradient sizes and one learning rate: compare how far each weight moves under plain SGD and under a normalised step like Adam’s.');
    const st = { ga: 5, gb: 0.05, lr: 0.01, alpha: 0.01 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'gradient of weight A', -2, 2, 0.1, Math.log10(st.ga), function (v) { st.ga = Math.pow(10, v); draw(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    Book.slider(ctl, 'gradient of weight B', -4, 0, 0.1, Math.log10(st.gb), function (v) { st.gb = Math.pow(10, v); draw(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    Book.slider(ctl, 'SGD learning rate', -4, 0, 0.1, Math.log10(st.lr), function (v) { st.lr = Math.pow(10, v); draw(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      // Adam's first step: m̂ = g, v̂ = g² → step = α·g/|g| = α (paper §2.1)
      const sa = st.lr * st.ga, sb = st.lr * st.gb, aa = st.alpha, ab = st.alpha;
      const bars = [{ label: 'SGD\nweight A', value: Math.log10(sa) + 6, note: AD.ft(sa, 3), color: Book.palette[7] }, { label: 'SGD\nweight B', value: Math.log10(sb) + 6, note: AD.ft(sb, 3), color: Book.palette[7] },
        { label: 'Adam\nweight A', value: Math.log10(aa) + 6, note: AD.ft(aa, 3), color: Book.palette[4] }, { label: 'Adam\nweight B', value: Math.log10(ab) + 6, note: AD.ft(ab, 3), color: Book.palette[4] }];
      Book.chart.bars(chart, bars, { max: 8, ylabel: 'step size (log scale)', tickFmt: function (v) { return v === 0 ? AD.p10(-6) : AD.p10(v - 6); }, height: 220 });
      out.innerHTML = 'SGD moves weight A by lr·g = ' + AD.f(sa) + ' and weight B by ' + AD.f(sb) + ' → ratio <b>' + AD.f(sa / sb, 1) + '×</b>' +
        '<br>Adam’s first step (α = 0.01) moves each weight by α·g/√(g²) = α = 0.01, whatever the gradient size → ratio <b>1×</b>' +
        '<br><span class="note">Big gradients do not automatically mean “move far”. Adam divides each gradient by its own typical size, so every weight takes steps of a similar, controlled size.</span>';
    }
    draw();
  };

  // ================================================================== ch-momentum
  W.momentumRace = function (host) {
    Book.head(host, 'Momentum: a heavy ball rolls through the zig-zag',
      'Plain SGD, classical momentum and Nesterov momentum, same learning rate. Momentum keeps a running sum of past gradients (velocity): the zig-zags across the valley cancel, the steady pull along it adds up.');
    const st = { surf: 'ravine', lr: 0.08, mu: 0.9, steps: 120 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'surface', [{ value: 'ravine', label: 'ravine' }, { value: 'rosenbrock', label: 'banana valley' }], st.surf, function (v) { st.surf = v; st.lr = v === 'ravine' ? 0.08 : 0.004; lrS.value = st.lr; lrS.dispatchEvent(new Event('input')); map.setSpec(Opt.surfaces[v]); draw(); });
    const lrS = Book.slider(ctl, 'learning rate', 0.001, 0.4, 0.001, st.lr, function (v) { st.lr = v; draw(); }, function (v) { return v.toFixed(3); });
    Book.slider(ctl, 'momentum μ', 0, 0.99, 0.01, st.mu, function (v) { st.mu = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'steps', 20, 400, 10, st.steps, function (v) { st.steps = v; draw(); }, function (v) { return v; });
    const map = AD.contour(host, Opt.surfaces[st.surf], { height: 300 }), lg = el('div', null, host), plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    AD.legend(lg, [{ name: 'SGD', color: Opt.colors.sgd }, { name: 'momentum', color: Opt.colors.momentum }, { name: 'Nesterov', color: Opt.colors.nesterov }]);
    function draw() {
      const runs = [['sgd', { lr: st.lr }], ['momentum', { lr: st.lr, mu: st.mu }], ['nesterov', { lr: st.lr, mu: st.mu }]].map(function (a) { return { k: a[0], R: Opt.runSurface(st.surf, a[0], a[1], st.steps, 0, 1) }; });
      map.draw(runs.map(function (u) { return { path: u.R.path, color: Opt.colors[u.k] }; }));
      AD.logLines(plot, runs.map(function (u) { return { name: Opt.names[u.k], color: Opt.colors[u.k], marker: 'none', points: u.R.loss.map(function (v, t) { return [t, Math.max(v, 1e-12)]; }) }; }),
        { width: 640, height: 200, xlabel: 'step', ylabel: 'loss (log)', xticks: [0, st.steps], mr: 100 });
      out.innerHTML = runs.map(function (u) { const L = u.R.loss[u.R.loss.length - 1]; return Opt.names[u.k] + ': final loss <b>' + (isFinite(L) && !isNaN(u.R.path[u.R.path.length - 1][0]) ? AD.f(L) : 'diverged') + '</b>'; }).join(' · ') +
        '<br><span class="note">Momentum: v ← μ·v + g, θ ← θ − lr·v. Nesterov evaluates the pull a little ahead, which damps overshooting. With μ = 0 all three are the same.</span>';
    }
    draw();
  };

  // ================================================================== ch-ema
  W.emaDemo = function (host) {
    Book.head(host, 'A running average with a memory knob β',
      'Grey dots: a noisy stream of numbers (think: gradients). The coloured line is the exponential moving average m ← β·m + (1 − β)·g, started at m = 0. Move β and watch the trade-off between smoothness and lag. Notice the start.');
    const st = { beta: 0.9, jump: true };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'β', 0, 0.995, 0.005, st.beta, function (v) { st.beta = v; draw(); }, function (v) { return v.toFixed(3); });
    const jb = el('label', null, ctl), jc = el('input', { type: 'checkbox' }, jb); jc.checked = true; jb.appendChild(document.createTextNode(' the true level jumps at step 150'));
    jc.addEventListener('change', function () { st.jump = jc.checked; draw(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    const r = ML.rng(12), noise = ML.range(300).map(function () { return r.normal(0, 1); });
    function draw() {
      const lvl = function (t) { return st.jump && t >= 150 ? -1 : 2; };
      const g = noise.map(function (n, t) { return lvl(t) + n; });
      let m = 0; const ema = [], tr = [];
      g.forEach(function (x, t) { m = st.beta * m + (1 - st.beta) * x; ema.push([t + 1, m]); tr.push([t + 1, lvl(t)]); });
      AD.lines(plot, [
        { color: 'var(--muted)', noline: true, r: 1.8, points: g.map(function (x, t) { return [t + 1, x]; }) },
        { name: 'true level', color: 'var(--ink)', marker: 'none', dash: '5 4', width: 1.5, points: tr },
        { name: 'EMA', color: 'var(--accent)', marker: 'none', width: 2.6, points: ema }
      ], { width: 640, height: 240, ymin: -4, ymax: 5, xlabel: 'step t', ylabel: 'value', xticks: [1, 50, 100, 150, 200, 250, 300], mr: 80 });
      const win = 1 / (1 - st.beta);
      out.innerHTML = 'effective memory ≈ 1/(1 − β) = <b>' + (isFinite(win) ? win.toFixed(1) : '∞') + '</b> steps · EMA after 5 steps: <b>' + ema[4][1].toFixed(2) + '</b> although the level is 2' +
        '<br><span class="note">Large β: smooth but slow to follow the jump. Small β: follows quickly but stays noisy. And because the average starts at 0, it is too small for the first ~1/(1−β) steps. That start-up bias is the subject of Chapter 16.</span>';
    }
    draw();
  };

  W.emaWeights = function (host) {
    Book.head(host, 'How much each past value counts',
      'Unroll the recursion: after t steps, the value from step i has weight (1 − β)·β^(t−i). Bars show the weights; the newest value is on the right. Their sum is 1 − β^t, not 1.');
    const st = { beta: 0.8, t: 12 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'β', 0, 0.98, 0.01, st.beta, function (v) { st.beta = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'steps so far t', 1, 30, 1, st.t, function (v) { st.t = v; draw(); }, function (v) { return v; });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const w = ML.range(st.t).map(function (k) { const i = k + 1; return (1 - st.beta) * Math.pow(st.beta, st.t - i); });
      Book.chart.bars(chart, w.map(function (v, k) { return { label: st.t <= 15 || (k + 1) % 5 === 0 || k === st.t - 1 ? 'i=' + (k + 1) : '', value: v, note: st.t <= 12 ? v.toFixed(2) : '', color: 'var(--accent)' }; }),
        { max: Math.max(0.2, 1 - st.beta) * 1.05, ylabel: 'weight', tickFmt: function (v) { return v.toFixed(2); }, height: 210 });
      const sum = ML.sum(w);
      out.innerHTML = 'weights (1 − β)β<sup>t−i</sup> for i = 1…' + st.t + ' · sum = <b>' + sum.toFixed(4) + '</b> = 1 − β<sup>t</sup> = 1 − ' + st.beta.toFixed(2) + '<sup>' + st.t + '</sup>' +
        '<br>the missing weight β<sup>t</sup> = <b>' + Math.pow(st.beta, st.t).toFixed(4) + '</b> sits on the starting value m₀ = 0' +
        '<br><span class="note">This is the shape of the paper’s Eq. 1. Dividing by 1 − β<sup>t</sup> rescales the weights to sum to 1: that is Adam’s bias correction.</span>';
    }
    draw();
  };

  // ================================================================== ch-moments
  W.momentsDemo = function (host) {
    Book.head(host, 'Mean, raw second moment and variance of a gradient',
      'Pretend one weight’s gradient is a random number with true mean μ and spread σ. Draw samples and compare the sample estimates: the 1st moment E[g], the 2nd raw moment E[g²], and the variance E[g²] − E[g]².');
    const st = { mu: 0.5, sd: 1, n: 1000, seed: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'true mean μ', -2, 2, 0.05, st.mu, function (v) { st.mu = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'true spread σ', 0, 2, 0.05, st.sd, function (v) { st.sd = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'New samples', function () { st.seed++; draw(); }, 'primary');
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(st.seed), g = ML.range(st.n).map(function () { return r.normal(st.mu, st.sd); });
      const m1 = ML.mean(g), m2 = ML.mean(g.map(function (x) { return x * x; })), vr = m2 - m1 * m1;
      const lo = -5, hi = 5, bins = 50, h = new Array(bins).fill(0);
      g.forEach(function (v) { const b = Math.floor((v - lo) / (hi - lo) * bins); if (b >= 0 && b < bins) h[b]++; });
      plot.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 640 190' }, plot), mx = Math.max.apply(null, h) || 1, X = function (v) { return 20 + (v - lo) / (hi - lo) * 600; };
      h.forEach(function (c, b) { S('rect', { x: X(lo + b * (hi - lo) / bins) + 1, y: 160 - c / mx * 140, width: 600 / bins - 2, height: c / mx * 140, fill: 'var(--accent)', opacity: 0.7 }, svg); });
      S('line', { x1: 20, x2: 620, y1: 160, y2: 160, stroke: 'var(--muted)' }, svg);
      [-4, -2, 0, 2, 4].forEach(function (v) { S('text', { x: X(v), y: 176, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: v }, svg); });
      S('line', { x1: X(m1), x2: X(m1), y1: 12, y2: 162, stroke: Book.palette[1], 'stroke-width': 2.5 }, svg);
      S('text', { x: X(m1) + 5, y: 22, 'font-size': 11.5, fill: Book.palette[1], text: 'mean' }, svg);
      const rms = Math.sqrt(m2);
      [-rms, rms].forEach(function (v) { S('line', { x1: X(v), x2: X(v), y1: 30, y2: 162, stroke: Book.palette[4], 'stroke-width': 2, 'stroke-dasharray': '5 3' }, svg); });
      S('text', { x: X(rms) + 5, y: 40, 'font-size': 11.5, fill: Book.palette[4], text: '±√E[g²]' }, svg);
      const ratio = Math.abs(m1) / Math.sqrt(m2);
      out.innerHTML = '1st moment (mean) E[g] ≈ <b>' + m1.toFixed(3) + '</b> (true ' + st.mu.toFixed(2) + ')' +
        '<br>2nd raw moment E[g²] ≈ <b>' + m2.toFixed(3) + '</b> (true μ² + σ² = ' + (st.mu * st.mu + st.sd * st.sd).toFixed(3) + ')' +
        '<br>variance E[g²] − E[g]² ≈ <b>' + vr.toFixed(3) + '</b> (true σ² = ' + (st.sd * st.sd).toFixed(3) + ')' +
        '<br>|E[g]| / √E[g²] ≈ <b>' + ratio.toFixed(3) + '</b>, never above 1: it is near 1 when the gradient is consistent (σ ≪ |μ|), near 0 when it is mostly noise' +
        '<br><span class="note">Adam keeps a running estimate of E[g] (called m) and of E[g²] (called v) for every weight. The paper calls E[g²] the “uncentered variance”: it is the variance plus the squared mean.</span>';
    }
    draw();
  };

  // ================================================================== ch-adaptive
  // online least squares with a common feature A (always on) and a rare feature B
  function sparseStream(q, T, seed) {
    const r = ML.rng(seed), out = [];
    for (let t = 0; t < T; t++) { const b = r.next() < q ? 1 : 0; out.push({ xa: 1, xb: b, y: 1 * 1 + 1 * b + r.normal(0, 0.3) }); }
    return out;
  }
  W.adagradSparse = function (host) {
    Book.head(host, 'Rare features need bigger steps',
      'A model y = a·x_A + b·x_B learns online. Feature A is present in every example; feature B only in a fraction q of them. Both true weights are 1. Plain SGD updates b only when B appears, by a step no bigger than a’s. AdaGrad divides by the root of the summed squared gradients, which is small for the rare feature.');
    const st = { q: 0.05, lr: 0.05, T: 600 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'how often B appears q', 0.01, 1, 0.01, st.q, function (v) { st.q = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'SGD learning rate', 0.01, 0.4, 0.01, st.lr, function (v) { st.lr = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function run(name, opts, data) {
      const P = Opt.vec([0, 0]), o = Opt.make(name, [P], opts), tr = [];
      data.forEach(function (d, t) { const e = P.data[0] * d.xa + P.data[1] * d.xb - d.y; P.grad[0] = e * d.xa; P.grad[1] = e * d.xb; o.step(); tr.push([t + 1, P.data[0], P.data[1]]); });
      return tr;
    }
    function draw() {
      const data = sparseStream(st.q, st.T, 4);
      const R = { sgd: run('sgd', { lr: st.lr }, data), adagrad: run('adagrad', { lr: 0.3 }, data), adam: run('adam', { lr: 0.02 }, data) };
      const sers = [];
      ['sgd', 'adagrad', 'adam'].forEach(function (k) {
        sers.push({ name: Opt.names[k] + ' b', color: Opt.colors[k], marker: 'none', points: R[k].map(function (p) { return [p[0], p[2]]; }) });
        sers.push({ color: Opt.colors[k], marker: 'none', dash: '3 3', width: 1.3, points: R[k].map(function (p) { return [p[0], p[1]]; }) });
      });
      sers.push({ color: 'var(--ink)', marker: 'none', dash: '6 4', width: 1, points: [[1, 1], [st.T, 1]] });
      AD.lines(plot, sers, { width: 640, height: 240, ymin: 0, ymax: 1.6, xlabel: 'examples seen', ylabel: 'weight', xticks: [1, 200, 400, 600], mr: 110 });
      const nB = data.filter(function (d) { return d.xb; }).length, f = function (k) { return R[k][st.T - 1][2].toFixed(2); };
      out.innerHTML = 'feature B appeared in <b>' + nB + '</b> of ' + st.T + ' examples · rare weight b after training: SGD <b>' + f('sgd') + '</b> · AdaGrad (α = 0.3) <b>' + f('adagrad') + '</b> · Adam (α = 0.02) <b>' + f('adam') + '</b> (target 1)' +
        '<br><span class="note">Solid lines: the rare weight b. Dashed: the common weight a. AdaGrad: θ ← θ − α·g/√(Σ past g²), per weight (paper §5). The rare weight has a small sum, so each of its few updates is large.</span>';
    }
    draw();
  };

  W.rmspropVsAdagrad = function (host) {
    Book.head(host, 'AdaGrad never forgets; RMSProp does',
      'A single weight must track a target that keeps moving (a non-stationary objective), from noisy gradients. AdaGrad’s sum of squared gradients only grows, so its effective step keeps shrinking and it falls further and further behind. RMSProp’s running average forgets, so its step stays the same size.');
    const st = { rho: 0.99, period: 600 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'RMSProp decay ρ', 0.5, 0.999, 0.001, st.rho, function (v) { st.rho = v; draw(); }, function (v) { return v.toFixed(3); });
    Book.slider(ctl, 'target period (steps)', 200, 2000, 50, st.period, function (v) { st.period = v; draw(); }, function (v) { return v; });
    const plot = el('div', null, host), plot2 = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(6), T = 3000, tgt = function (t) { return 2 * Math.sin(2 * Math.PI * t / st.period); }, noise = ML.range(T).map(function () { return r.normal(0, 0.5); });
      const run = function (name, o) {
        const P = Opt.vec([0]), opt = Opt.make(name, [P], o), tr = [], rate = [];
        for (let t = 0; t < T; t++) {
          P.grad[0] = P.data[0] - tgt(t) + noise[t]; const before = P.data[0]; opt.step();
          tr.push([t + 1, P.data[0]]); rate.push([t + 1, Math.abs((P.data[0] - before) / (P.grad[0] || 1e-12))]);
        }
        return { tr: tr, rate: rate };
      };
      const A = run('adagrad', { lr: 0.5 }), B = run('rmsprop', { lr: 0.03, rho: st.rho });
      const pick = function (arr) { return arr.filter(function (p, i) { return i % 5 === 0; }); };
      AD.lines(plot, [
        { name: 'target', color: 'var(--ink)', marker: 'none', dash: '6 4', width: 1.3, points: pick(ML.range(T).map(function (t) { return [t + 1, tgt(t)]; })) },
        { name: 'AdaGrad', color: Opt.colors.adagrad, marker: 'none', points: pick(A.tr) },
        { name: 'RMSProp', color: Opt.colors.rmsprop, marker: 'none', points: pick(B.tr) }
      ], { width: 640, height: 220, ymin: -3, ymax: 3, xlabel: 'step', ylabel: 'weight', xticks: [1, 1000, 2000, 3000] });
      const smooth = function (arr) { const o = []; for (let i = 0; i < arr.length; i += 25) { let s2 = 0, n = 0; for (let j = i; j < Math.min(arr.length, i + 25); j++) { s2 += arr[j][1]; n++; } o.push([arr[i][0], s2 / n]); } return o; };
      AD.logLines(plot2, [{ name: 'AdaGrad: step ÷ gradient', color: Opt.colors.adagrad, marker: 'none', points: smooth(A.rate) }, { name: 'RMSProp: step ÷ gradient', color: Opt.colors.rmsprop, marker: 'none', points: smooth(B.rate) }],
        { width: 640, height: 170, xlabel: 'step', ylabel: 'effective rate (log)', xticks: [1, 1000, 2000, 3000] });
      const err = function (tr, a0, b0) { let s2 = 0; for (let t = a0; t < b0; t++) s2 += Math.abs(tr[t][1] - tgt(t)); return s2 / (b0 - a0); };
      out.innerHTML = 'average tracking error, steps 1–1,000: AdaGrad <b>' + err(A.tr, 0, 1000).toFixed(2) + '</b> · RMSProp <b>' + err(B.tr, 0, 1000).toFixed(2) + '</b>' +
        '<br>average tracking error, steps 2,001–3,000: AdaGrad <b>' + err(A.tr, 2000, 3000).toFixed(2) + '</b> · RMSProp <b>' + err(B.tr, 2000, 3000).toFixed(2) + '</b>' +
        '<br><span class="note">AdaGrad’s effective rate falls like 1/√t (lower chart), so a target that keeps moving escapes it more and more. AdaGrad suits sparse, stationary problems; RMSProp suits on-line, non-stationary ones. The paper’s stated aim (§1) is to combine both strengths.</span>';
    }
    draw();
  };
})();
