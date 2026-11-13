/* Dropout Interactive Web Book — helpers shared by all widget files (window.DOUI), the hero demo and Part I
   (foundations). Uses Book (core.js), ML (lib/math.js), NN (lib/nn.js) and Toy (toy.js). */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, Toy = window.Toy;
  const U = window.DOUI = {};

  // ------------------------------------------------------------------ small helpers
  U.f = function (x, d) { return Book.fmt(x, d == null ? 2 : d); };
  U.pct = function (x, d) { return Book.fmt(x, d == null ? 1 : d) + '%'; };
  U.col = function (name) { return Book.css('--' + name) || '#888'; };
  // a 7×7 pixel digit (values 0…1) as an SVG grid. Signed values (weights) use blue for + and red for −.
  U.digit = function (parent, x, o) {
    o = o || {};
    const n = Math.round(Math.sqrt(x.length)), cell = o.cell || 8, gap = o.gap == null ? 1 : o.gap, size = n * cell;
    const svg = S('svg', { viewBox: '0 0 ' + size + ' ' + size, width: o.px || size, height: o.px || size, role: 'img', 'aria-label': o.aria || 'pixel image', style: 'display:inline-block;vertical-align:middle;' + (o.style || '') }, parent);
    S('rect', { x: 0, y: 0, width: size, height: size, fill: 'var(--bg-soft)', rx: 2 }, svg);
    let m = 1;
    if (o.signed) { m = 0; x.forEach(function (v) { m = Math.max(m, Math.abs(v)); }); m = m || 1; }
    for (let i = 0; i < x.length; i++) {
      const v = x[i], a = o.signed ? Math.abs(v) / m : Book.clamp(v, 0, 1);
      if (a < 0.02) continue;
      S('rect', { x: (i % n) * cell + gap / 2, y: Math.floor(i / n) * cell + gap / 2, width: cell - gap, height: cell - gap, rx: 1,
        fill: o.signed ? (v > 0 ? 'var(--blue)' : 'var(--red)') : 'var(--ink)', opacity: a.toFixed(3) }, svg);
    }
    if (o.title) S('title', { text: o.title }, svg);
    return svg;
  };
  // horizontal bar chart for result tables with long labels. rows: [{label, value, color, note, dim}]
  U.hbars = function (host, rows, o) {
    o = o || {};
    host.innerHTML = '';
    // narrow screens: put each label on its own line above its bar, so the text stays readable
    const narrow = (host.clientWidth || (host.parentNode && host.parentNode.clientWidth) || 700) < 560;
    if (narrow) return U.hbarsStacked(host, rows, o);
    const Wd = o.width || 680, lw = o.labelWidth || 330, rh = o.rowH || 24, top = 6, H = top + rows.length * rh + 26;
    const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H, role: 'img', 'aria-label': o.aria || 'bar chart' }, host);
    let max = o.max; if (!max) { max = 0; rows.forEach(function (r) { max = Math.max(max, r.value); }); max = max * 1.1 || 1; }
    const x = function (v) { return lw + (v / max) * (Wd - lw - 60); };
    const ticks = o.ticks || 4;
    for (let i = 0; i <= ticks; i++) {
      const v = max * i / ticks;
      S('line', { x1: x(v), x2: x(v), y1: top, y2: H - 22, stroke: 'var(--line)' }, svg);
      S('text', { x: x(v), y: H - 8, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: o.tickFmt ? o.tickFmt(v) : +v.toFixed(2) }, svg);
    }
    rows.forEach(function (r, i) {
      const y = top + i * rh;
      S('text', { x: lw - 8, y: y + rh * 0.66, 'text-anchor': 'end', 'font-size': 12, class: 'svg-text', 'font-weight': r.bold ? 700 : 400, opacity: r.dim ? 0.55 : 1, text: r.label }, svg);
      if (r.value == null) { S('text', { x: lw + 4, y: y + rh * 0.66, 'font-size': 12, class: 'svg-text', opacity: 0.6, text: '—' }, svg); return; }
      S('rect', { x: lw, y: y + 4, width: Math.max(1, x(r.value) - lw), height: rh - 8, rx: 3, fill: r.color || 'var(--accent)', opacity: r.dim ? 0.4 : 0.9 }, svg);
      S('text', { x: x(r.value) + 6, y: y + rh * 0.66, 'font-size': 12, 'font-weight': 600, class: 'svg-text', text: r.note != null ? r.note : (o.fmt ? o.fmt(r.value) : r.value) }, svg);
    });
    if (o.xlabel) S('text', { x: (lw + Wd) / 2, y: H + 6, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: '' }, svg);
    return svg;
  };
  U.hbarsStacked = function (host, rows, o) {
    const Wd = 340, rh = 40, top = 4, H = top + rows.length * rh + 22;
    const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H, role: 'img', 'aria-label': o.aria || 'bar chart' }, host);
    let max = o.max; if (!max) { max = 0; rows.forEach(function (r) { max = Math.max(max, r.value || 0); }); max = max * 1.1 || 1; }
    const x = function (v) { return (v / max) * (Wd - 70); };
    rows.forEach(function (r, i) {
      const y = top + i * rh;
      S('text', { x: 0, y: y + 13, 'font-size': 12, class: 'svg-text', 'font-weight': r.bold ? 700 : 400, opacity: r.dim ? 0.55 : 1, text: r.label }, svg);
      if (r.value == null) { S('text', { x: 0, y: y + 31, 'font-size': 12, class: 'svg-text', opacity: 0.6, text: '—' }, svg); return; }
      S('rect', { x: 0, y: y + 19, width: Math.max(1, x(r.value)), height: 14, rx: 3, fill: r.color || 'var(--accent)', opacity: r.dim ? 0.4 : 0.9 }, svg);
      S('text', { x: x(r.value) + 6, y: y + 31, 'font-size': 12, 'font-weight': 600, class: 'svg-text', text: r.note != null ? r.note : (o.fmt ? o.fmt(r.value) : r.value) }, svg);
    });
    return svg;
  };
  // a small table from rows of cells (first row = header)
  U.table = function (host, rows, o) {
    o = o || {};
    const wrap = el('div', { class: 'table-wrap' }, host), t = el('table', { class: 'data' }, wrap);
    rows.forEach(function (r, i) {
      const tr = el('tr', null, t);
      r.forEach(function (c) { el(i === 0 ? 'th' : 'td', { html: String(c) }, tr); });
    });
    return t;
  };
  // seeded 2-D "two moons" data (class 0 upper arc, class 1 lower arc) with Gaussian noise sd
  U.moons = function (n, seed, sd) {
    const r = ML.rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) {
      const c = i % 2, t = r.uniform(0, Math.PI);
      const px = c ? 1 - Math.cos(t) : Math.cos(t), py = c ? 0.5 - Math.sin(t) : Math.sin(t);
      X.push([(px - 0.5) * 0.9 + r.normal(0, sd), (py - 0.25) * 0.9 + r.normal(0, sd)]); y.push(c);
    }
    return { X: X, y: y };
  };
  // paint the class-1 probability of a 2-class net over [-1.4, 1.4]² on a canvas, then the points
  U.boundary = function (cv, probFn, pts, o) {
    o = o || {};
    const ctx = cv.getContext('2d'), N = o.grid || 56, s = cv.width / N, R = 1.4, grid = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) grid.push([-R + 2 * R * (j + 0.5) / N, R - 2 * R * (i + 0.5) / N]);
    const P = probFn(grid);
    const c0 = [43, 108, 176], c1 = [192, 57, 43];
    ctx.clearRect(0, 0, cv.width, cv.height);
    for (let k = 0; k < grid.length; k++) {
      const p = P[k];
      ctx.fillStyle = 'rgba(' + Math.round(c0[0] + (c1[0] - c0[0]) * p) + ',' + Math.round(c0[1] + (c1[1] - c0[1]) * p) + ',' + Math.round(c0[2] + (c1[2] - c0[2]) * p) + ',' + (0.12 + 0.3 * Math.abs(p - 0.5) * 2).toFixed(3) + ')';
      ctx.fillRect((k % N) * s, Math.floor(k / N) * s, s + 1, s + 1);
    }
    // the 0.5 contour, as dots where neighbouring cells change side
    ctx.fillStyle = 'rgba(40,40,40,0.75)';
    for (let i = 0; i < N; i++) for (let j = 0; j + 1 < N; j++) { const a = P[i * N + j] > 0.5, b = P[i * N + j + 1] > 0.5, c = i + 1 < N ? P[(i + 1) * N + j] > 0.5 : a; if (a !== b || a !== c) ctx.fillRect((j + 0.5) * s, (i + 0.5) * s, 2, 2); }
    (pts || []).forEach(function (set) {
      set.X.forEach(function (p, i) {
        const px = (p[0] + R) / (2 * R) * cv.width, py = (R - p[1]) / (2 * R) * cv.height;
        ctx.beginPath(); ctx.arc(px, py, set.r || 3.4, 0, 2 * Math.PI);
        ctx.fillStyle = set.y[i] ? '#c0392b' : '#2b6cb0'; ctx.globalAlpha = set.alpha || 1; ctx.fill(); ctx.globalAlpha = 1;
        ctx.lineWidth = 1.2; ctx.strokeStyle = set.ring || '#fff'; ctx.stroke();
      });
    });
  };
  // a layered network diagram. layers: unit counts. o.dropped: array of Sets (one per layer) of dropped unit indices,
  // o.scale: array (per layer) of factors that thin the outgoing edges (test-time weight scaling), o.labels
  U.net = function (host, layers, o) {
    o = o || {};
    host.innerHTML = '';
    const Wd = o.width || 420, Ht = o.height || 230, mx = 36, my = 22;
    const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + Ht, role: 'img', 'aria-label': o.aria || 'network diagram' }, host);
    const pos = layers.map(function (n, l) {
      const x = mx + l * (Wd - 2 * mx) / (layers.length - 1);
      return ML.range(n).map(function (i) { return [x, my + (i + 0.5) * (Ht - 2 * my) / n]; });
    });
    const dropped = o.dropped || [];
    const isDropped = function (l, i) { return dropped[l] && dropped[l].has(i); };
    for (let l = 0; l + 1 < layers.length; l++) {
      for (let i = 0; i < layers[l]; i++) for (let j = 0; j < layers[l + 1]; j++) {
        const off = isDropped(l, i) || isDropped(l + 1, j);
        const sc = o.scale ? o.scale[l] : 1;
        S('line', { x1: pos[l][i][0], y1: pos[l][i][1], x2: pos[l + 1][j][0], y2: pos[l + 1][j][1], stroke: off ? 'var(--line)' : 'var(--ink-soft)',
          'stroke-width': off ? 0.6 : (1.3 * sc).toFixed(2), opacity: off ? 0.35 : (0.35 + 0.45 * sc).toFixed(2), 'stroke-dasharray': off ? '2 3' : '' }, svg);
      }
    }
    pos.forEach(function (col, l) {
      col.forEach(function (p, i) {
        const off = isDropped(l, i), r = o.r || 11;
        S('circle', { cx: p[0], cy: p[1], r: r, fill: off ? 'var(--bg-soft)' : (l === 0 ? 'var(--blue-soft)' : l === layers.length - 1 ? 'var(--green-soft)' : 'var(--accent-soft)'),
          stroke: off ? 'var(--muted)' : 'var(--ink-soft)', 'stroke-width': 1.4 }, svg);
        if (off) {
          S('line', { x1: p[0] - r * 0.7, y1: p[1] - r * 0.7, x2: p[0] + r * 0.7, y2: p[1] + r * 0.7, stroke: 'var(--red)', 'stroke-width': 2.2 }, svg);
          S('line', { x1: p[0] - r * 0.7, y1: p[1] + r * 0.7, x2: p[0] + r * 0.7, y2: p[1] - r * 0.7, stroke: 'var(--red)', 'stroke-width': 2.2 }, svg);
        }
      });
    });
    if (o.labels) o.labels.forEach(function (t, l) { S('text', { x: pos[l][0][0], y: Ht - 4, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: t }, svg); });
    return svg;
  };
  // sample a dropout pattern: Set of dropped indices per layer (input keep pIn, hidden keep pH, outputs never dropped)
  U.sampleDrop = function (layers, pIn, pH, rng) {
    return layers.map(function (n, l) {
      const s = new Set(), keep = l === 0 ? pIn : (l === layers.length - 1 ? 1 : pH);
      for (let i = 0; i < n; i++) if (rng.next() >= keep) s.add(i);
      return s;
    });
  };
  // a "progress" line inside widgets that train
  U.status = function (host) { return el('div', { class: 'note', text: '' }, host); };

  // ------------------------------------------------------------------ the two reference networks (toy)
  // A standard net and a dropout net (p = 0.8 input, 0.5 hidden), 49-96-96-10, trained on 500 pixel digits with
  // the book's base settings. Trained once per page, in the background; many widgets reuse them.
  const pairWaiters = []; let pairState = null;
  U.pair = function (cb) {
    if (pairState && pairState.done) { cb(pairState); return; }
    pairWaiters.push(cb);
    if (pairState) return;
    pairState = { done: false, progress: 0 };
    const cfgs = [Object.assign({}, Toy.BASE, { label: 'standard' }), Object.assign({}, Toy.BASE, { label: 'dropout', keepIn: 0.8, keepHid: 0.5 })];
    pairState.handle = Toy.runJobs(cfgs, function (i, T) { pairState.progress = (i * cfgs[0].steps + T.t) / (2 * cfgs[0].steps); pairWaiters.forEach(function (w) { if (w.progress) w.progress(pairState.progress, pairState.handle.jobs); }); },
      function (jobs) {
        pairState.done = true; pairState.std = jobs[0]; pairState.drop = jobs[1]; pairState.data = jobs[0].data;
        const ws = pairWaiters.splice(0); ws.forEach(function (w) { w(pairState); });
      });
  };
  // show "training…" in a host until the pair is ready, then call ready(pair)
  U.withPair = function (host, ready, onProgress) {
    const msg = el('div', { class: 'out', text: 'Training the two reference networks on the toy digits (one time, a few seconds)…' }, host);
    const cb = function (P) { msg.remove(); ready(P); };
    cb.progress = function (f, jobs) { msg.textContent = 'Training the two reference networks on the toy digits (one time, in your browser): ' + Math.round(100 * f) + '%'; if (onProgress) onProgress(jobs); };
    U.pair(cb);
  };
  // a widget that shows precomputed toy results (window.TOY_RESULTS[name], made by this page's own code) and can
  // recompute them live. render(results) draws; results are Toy.summary objects.
  U.cached = function (host, name, render, secs) {
    const ctl = el('div', { class: 'controls' }, host), view = el('div', null, host), st = el('div', { class: 'note' }, host);
    let job = null;
    const btn = Book.button(ctl, 'Re-run live (≈ ' + secs + ' s)', function () {
      if (job) { job.stop(); job = null; btn.textContent = 'Re-run live (≈ ' + secs + ' s)'; st.textContent = 'Stopped. Showing the precomputed results.'; render(R0, view); return; }
      const cfgs = Toy.EXPERIMENTS[name]();
      btn.textContent = 'Stop';
      const sums = [];
      job = Toy.runJobs(cfgs, function (i, T, fin) { if (fin) sums[i] = Toy.summary(T); st.textContent = 'Training run ' + (i + 1 + (fin ? 0 : 0)) + ' of ' + cfgs.length + ' (' + T.cfg.label + '), step ' + T.t + ' of ' + T.cfg.steps + '…'; },
        function (jobs) { job = null; btn.textContent = 'Re-run live (≈ ' + secs + ' s)'; const res = jobs.map(Toy.summary); render(res, view); st.textContent = 'Recomputed live in your browser (toy). Same seeds, so the numbers should match the precomputed ones.'; });
    });
    const R0 = (window.TOY_RESULTS || {})[name];
    if (R0) { render(R0, view); st.textContent = 'Toy results precomputed with this page’s own code (toy.js, same seeds). Press Re-run to recompute them in your browser.'; }
    else st.textContent = 'Press Re-run to compute the toy results.';
    return view;
  };

  // ================================================================== HERO
  // Every training step samples a new thinned network; at test time the full net is used with weights × p.
  W.heroDemo = function (host) {
    Book.head(host, 'Every training case gets its own thinned network',
      'Each tick samples a new random network: every input unit survives with probability 0.8 and every hidden unit with probability 0.5 (crossed units are dropped). Press “Test time” to see the single full network used for predictions, with its weights scaled down by p.');
    const layers = [5, 7, 7, 3], st = { k: 0, test: false, timer: null, rng: ML.rng(3) };
    const ctl = el('div', { class: 'controls' }, host);
    const play = Book.button(ctl, 'Pause ❚❚', function () { if (st.timer) stop(); else start(); }, 'primary');
    Book.button(ctl, 'Test time', function () { stop(); st.test = !st.test; draw(); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    let drop = null;
    function sample() { drop = U.sampleDrop(layers, 0.8, 0.5, st.rng); st.k++; }
    function draw() {
      if (st.test) {
        U.net(view, layers, { scale: [0.8, 0.5, 0.5], labels: ['input', 'hidden 1', 'hidden 2', 'output'], aria: 'the full network at test time' });
        out.innerHTML = '<b>Test time:</b> all ' + (5 + 7 + 7 + 3) + ' units are present; every outgoing weight of a unit is multiplied by its p (0.8 for inputs, 0.5 for hidden units). Thinner lines = scaled-down weights.';
        return;
      }
      U.net(view, layers, { dropped: drop, labels: ['input (p = 0.8)', 'hidden 1 (p = 0.5)', 'hidden 2 (p = 0.5)', 'output'], aria: 'a thinned network' });
      const kept = drop.slice(0, 3).map(function (s, l) { return layers[l] - s.size; });
      out.innerHTML = 'thinned network #' + st.k + ': kept ' + kept[0] + '/5 inputs, ' + kept[1] + '/7 and ' + kept[2] + '/7 hidden units · possible thinned networks: 2<sup>19</sup> = <b>524,288</b> (all sharing one set of weights)';
    }
    function tick() { sample(); draw(); st.timer = setTimeout(tick, 900); }
    function start() { st.test = false; play.textContent = 'Pause ❚❚'; tick(); }
    function stop() { if (st.timer) clearTimeout(st.timer); st.timer = null; play.textContent = 'Play ▶'; }
    sample(); draw(); st.timer = setTimeout(tick, 900);
  };

  // ================================================================== PART I · FOUNDATIONS
  // ---------- ch-neuron: a whole layer computed at once
  W.layerForward = function (host) {
    Book.head(host, 'A layer is many neurons reading the same inputs',
      'Three input values y⁽⁰⁾ feed a layer of four units. Each unit i has its own weight vector wᵢ and bias bᵢ. Drag the inputs; the table shows zᵢ = wᵢ·y + bᵢ and yᵢ = f(zᵢ) for every unit.');
    const y = [1.0, 0.5, -0.5];
    const Wt = [[0.9, -0.4, 0.3], [-0.6, 0.8, 0.5], [0.2, 0.7, -0.9], [0.5, 0.5, 0.5]], b = [0.1, -0.2, 0.0, -0.4];
    const st = { act: 'logistic' };
    const ctl = el('div', { class: 'controls' }, host);
    y.forEach(function (v, i) { Book.slider(ctl, 'y' + (i + 1), -2, 2, 0.1, v, function (u) { y[i] = u; draw(); }, function (u) { return u.toFixed(1); }); });
    Book.select(ctl, 'f', [{ value: 'logistic', label: 'logistic 1/(1+e⁻ˣ)' }, { value: 'relu', label: 'ReLU max(0, x)' }], st.act, function (v) { st.act = v; draw(); });
    const box = el('div', { class: 'grid2' }, host), pic = el('div', null, box), tab = el('div', null, box);
    function f(z) { return st.act === 'relu' ? Math.max(0, z) : ML.sigmoid(z); }
    function draw() {
      U.net(pic, [3, 4], { width: 260, height: 190, labels: ['inputs y', 'layer'], r: 12 });
      tab.innerHTML = '';
      const rows = [['unit i', 'zᵢ = wᵢ·y + bᵢ', 'yᵢ = f(zᵢ)']];
      Wt.forEach(function (w, i) {
        const z = ML.dot(w, y) + b[i];
        rows.push([i + 1, w.map(function (v, j) { return v.toFixed(1) + '×' + y[j].toFixed(1); }).join(' + ') + ' + ' + b[i].toFixed(1) + ' = <b>' + z.toFixed(2) + '</b>', '<b>' + f(z).toFixed(3) + '</b>']);
      });
      U.table(tab, rows);
      el('div', { class: 'note', text: 'Written for the whole layer at once: z = W y + b, then f applied to every entry. The outputs of this layer are the inputs of the next.' }, tab);
    }
    draw();
  };

  // ---------- ch-learn: backpropagation on a tiny network, numbers visible
  W.backpropTiny = function (host) {
    Book.head(host, 'Backpropagation, one number at a time',
      'A network with one input x, one hidden logistic unit h and one logistic output ŷ, trained to output the target t. Press “Step” to run forward, compute the loss, send the error backwards with the chain rule, and update both weights.');
    const st = { x: 1.5, t: 0.9, w1: 0.4, w2: -0.6, lr: 1.0, n: 0, hist: [] };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'target t', 0, 1, 0.05, st.t, function (v) { st.t = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'learning rate', 0.1, 5, 0.1, st.lr, function (v) { st.lr = v; }, function (v) { return v.toFixed(1); });
    Book.button(ctl, 'Step', function () { step(); draw(); }, 'primary');
    Book.button(ctl, '20 steps', function () { for (let i = 0; i < 20; i++) step(); draw(); });
    Book.button(ctl, 'Reset', function () { st.w1 = 0.4; st.w2 = -0.6; st.n = 0; st.hist = []; draw(); });
    const box = el('div', { class: 'grid2' }, host), out = el('div', { class: 'out' }, box), plot = el('div', null, box);
    function fwd() { const h = ML.sigmoid(st.w1 * st.x), yh = ML.sigmoid(st.w2 * h); return { h: h, yh: yh, L: 0.5 * (yh - st.t) * (yh - st.t) }; }
    function grads() {
      const r = fwd(), dL_dy = r.yh - st.t, dy_dz2 = r.yh * (1 - r.yh), g2 = dL_dy * dy_dz2 * r.h;
      const dL_dh = dL_dy * dy_dz2 * st.w2, g1 = dL_dh * r.h * (1 - r.h) * st.x;
      return { r: r, dL_dy: dL_dy, dy_dz2: dy_dz2, g2: g2, dL_dh: dL_dh, g1: g1 };
    }
    function step() { const g = grads(); st.hist.push([st.n, g.r.L]); st.w1 -= st.lr * g.g1; st.w2 -= st.lr * g.g2; st.n++; }
    function draw() {
      const g = grads(), r = g.r;
      out.innerHTML = '<b>forward</b>  h = σ(w₁x) = σ(' + U.f(st.w1, 3) + '×' + st.x + ') = ' + U.f(r.h, 3) +
        '\n          ŷ = σ(w₂h) = ' + U.f(r.yh, 3) + '    loss L = ½(ŷ − t)² = <b>' + U.f(r.L, 4) + '</b>' +
        '\n<b>backward</b> ∂L/∂ŷ = ŷ − t = ' + U.f(g.dL_dy, 3) + ',  ∂ŷ/∂z₂ = ŷ(1−ŷ) = ' + U.f(g.dy_dz2, 3) +
        '\n          ∂L/∂w₂ = ∂L/∂ŷ · ∂ŷ/∂z₂ · h = <b>' + U.f(g.g2, 4) + '</b>' +
        '\n          ∂L/∂h = ∂L/∂ŷ · ∂ŷ/∂z₂ · w₂ = ' + U.f(g.dL_dh, 4) +
        '\n          ∂L/∂w₁ = ∂L/∂h · h(1−h) · x = <b>' + U.f(g.g1, 4) + '</b>' +
        '\n<b>update</b>   w ← w − (learning rate) × ∂L/∂w   (step ' + st.n + ')';
      const pts = st.hist.length ? st.hist.concat([[st.n, r.L]]) : [[0, r.L]];
      Book.chart.lines(plot, [{ color: Book.palette[0], points: pts, marker: pts.length > 30 ? 'none' : null }], { width: 320, height: 200, ymin: 0, xlabel: 'step', ylabel: 'loss', xticks: [], ydec: 3 });
    }
    draw();
  };

  // ---------- ch-overfit: a big network on few noisy points
  W.overfit2d = function (host) {
    Book.head(host, 'A big network memorises noise',
      'Thirty noisy training points (solid) come from two curved classes. Train a network with two hidden layers and watch the shaded region bend around individual points. Faint dots are 600 fresh test points from the same source: they show how well the boundary generalises.');
    const st = { h: 64, timer: null };
    const data = { train: U.moons(30, 21, 0.3), test: U.moons(600, 22, 0.3) };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'hidden units per layer', [4, 16, 64], st.h, function (v) { st.h = +v; reset(); });
    const go = Book.button(ctl, 'Train ▶', function () { if (st.timer) stop(); else run(); }, 'primary');
    Book.button(ctl, 'Reset', function () { reset(); });
    const box = el('div', { class: 'grid2' }, host);
    const cv = el('canvas', { width: 280, height: 280, style: 'width:100%;max-width:280px;border:1px solid var(--line)' }, box);
    const right = el('div', null, box), plot = el('div', null, right), out = el('div', { class: 'out' }, right);
    let T;
    function reset() { stop(); T = new Toy.Trainer({ sizes: [2, st.h, st.h, 2], data: data, steps: 4000, batch: 10, lr: 0.03, m1: 0.9, evalEvery: 50, seed: 4 }); T.evaluate(); draw(); }
    function run() { go.textContent = 'Pause ❚❚'; (function tick() { T.run(50); draw(); if (T.done()) { stop(); return; } st.timer = setTimeout(tick, 16); })(); }
    function stop() { if (st.timer) clearTimeout(st.timer); st.timer = null; go.textContent = 'Train ▶'; }
    function draw() {
      U.boundary(cv, function (g) { return T.net.probs(g).map(function (p) { return p[1]; }); }, [{ X: data.test.X, y: data.test.y, r: 1.8, alpha: 0.35, ring: 'transparent' }, data.train]);
      const c = T.curve;
      Book.chart.lines(plot, [
        { name: 'training', color: Book.palette[0], marker: 'none', points: c.map(function (r) { return [r[0], r[1]]; }) },
        { name: 'test', color: Book.palette[1], marker: 'none', points: c.map(function (r) { return [r[0], r[2]]; }) }
      ], { width: 330, height: 200, ymin: 0, ymax: 50, xmin: 0, xmax: 4000, xticks: [0, 2000, 4000], xlabel: 'training steps', ylabel: 'error %', mr: 64 });
      const L = T.last();
      out.innerHTML = 'step ' + L.step + ' · training error <b>' + U.pct(L.train) + '</b> · test error <b>' + U.pct(L.test) + '</b> (toy)' +
        '<br><span class="note">The gap between the two curves is overfitting. With 4 units the boundary stays simple; with 64 it can wrap single points.</span>';
    }
    reset();
  };

  // ---------- ch-overfit: the toy digits
  W.digitGallery = function (host) {
    Book.head(host, 'The toy data set: pixel digits',
      'The book’s stand-in for MNIST. Ten digit shapes are drawn on a 7×7 grid (49 pixels instead of MNIST’s 784) and every example is spoilt at random: missing stroke pixels, stray specks and grey noise. Press “New samples”.');
    const st = { seed: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.button(ctl, 'New samples', function () { st.seed++; draw(); }, 'primary');
    const grid = el('div', { style: 'display:grid;grid-template-columns:auto repeat(8, auto);gap:4px 6px;align-items:center;justify-content:start;overflow-x:auto' }, host);
    const out = el('div', { class: 'out' }, host);
    function draw() {
      grid.innerHTML = '';
      const rng = ML.rng(st.seed * 97);
      for (let k = 0; k < 10; k++) {
        U.digit(grid, Toy.clean(k), { px: 30, cell: 6, title: 'clean shape of ' + k, style: 'outline:2px solid var(--accent-soft)' });
        for (let j = 0; j < 8; j++) U.digit(grid, Toy.sample(k, rng), { px: 30, cell: 6, title: 'a noisy ' + k });
      }
      out.innerHTML = 'first column: the clean shapes · each example: 49 numbers between 0 and 1, one per pixel, and a label 0–9' +
        '<br>standard toy split: <b>500</b> training examples, <b>1,000</b> test examples · error of simply picking the nearest clean shape: <b>' + U.pct(Toy.templateError(Toy.data(500).test)) + '</b> on the test set (a yardstick: some examples are spoilt beyond recognition)';
    }
    draw();
  };

  // ---------- ch-overfit: learning curves of a standard net on the toy digits, with early stopping
  W.trainTest = function (host) {
    Book.head(host, 'Training error falls to zero; test error does not',
      'A 49-96-96-10 network trained on the 500 toy training digits with plain minibatch SGD. The training error falls to almost 0 %; the test error stops improving and drifts. A separate validation set (300 more examples, not used for training) marks where early stopping would stop.');
    const st = { timer: null };
    const valid = Toy.digits(300, 4242);
    const ctl = el('div', { class: 'controls' }, host);
    const go = Book.button(ctl, 'Train ▶', function () { if (st.timer) stop(); else run(); }, 'primary');
    Book.button(ctl, 'Reset', function () { reset(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    let T, vc;
    function reset() { stop(); T = new Toy.Trainer(Object.assign({}, Toy.BASE, { steps: 3000 })); vc = []; T.evaluate(); vc.push([0, T.net.error(valid)]); draw(); }
    function run() { go.textContent = 'Pause ❚❚'; (function tick() { const n0 = T.curve.length; T.run(50); if (T.curve.length > n0) vc.push([T.t, T.net.error(valid)]); draw(); if (T.done()) { stop(); return; } st.timer = setTimeout(tick, 0); })(); }
    function stop() { if (st.timer) clearTimeout(st.timer); st.timer = null; go.textContent = 'Train ▶'; }
    function draw() {
      const c = T.curve;
      let bi = 0; vc.forEach(function (r, i) { if (r[1] < vc[bi][1]) bi = i; });
      const r = Book.chart.lines(plot, [
        { name: 'training', color: Book.palette[0], marker: 'none', points: c.map(function (r) { return [r[0], r[1]]; }) },
        { name: 'validation', color: Book.palette[2], marker: 'none', dash: '5 3', points: vc, ldy: 16 },
        { name: 'test', color: Book.palette[1], marker: 'none', points: c.map(function (r) { return [r[0], r[2]]; }), ldy: -6 }
      ], { width: 640, height: 240, ymin: 0, ymax: 40, xmin: 0, xmax: 3000, xticks: [0, 1000, 2000, 3000], xlabel: 'training steps (minibatches of 20)', ylabel: 'error %', mr: 80 });
      if (vc.length > 1) { const x = r.X(vc[bi][0]); S('line', { x1: x, x2: x, y1: 16, y2: 198, stroke: 'var(--green)', 'stroke-dasharray': '2 3' }, r.svg); S('text', { x: x + 4, y: 28, 'font-size': 11, class: 'svg-text', text: 'early stop' }, r.svg); }
      const L = T.last();
      out.innerHTML = 'step ' + L.step + ' · training error <b>' + U.pct(L.train) + '</b> · test error <b>' + U.pct(L.test) + '</b>' +
        (vc.length > 1 ? ' · lowest validation error at step ' + vc[bi][0] + ' (test error there: ' + U.pct(c[Math.min(bi, c.length - 1)][2]) + ')' : '') + ' (toy)';
    }
    reset();
  };

  // ---------- ch-penalty: polynomial regression with an L2 penalty (ridge regression)
  // solve A x = b (small dense system) by Gaussian elimination with partial pivoting
  U.solve = function (A, b) {
    const n = b.length, M = A.map(function (r, i) { return r.slice().concat([b[i]]); });
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      const t = M[c]; M[c] = M[p]; M[p] = t;
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  };
  W.ridgeFit = function (host) {
    Book.head(host, 'A weight penalty tames a wiggly fit',
      'Ten noisy points from a smooth curve. A degree-9 polynomial has 10 weights, enough to pass through every point. The L2 penalty λ‖w‖² makes large weights expensive. Slide λ and compare the error on the 10 training points with the error on fresh test points.');
    const r = ML.rng(8), f = function (x) { return Math.sin(2.5 * x); };
    const tr = ML.range(10).map(function (i) { const x = -1 + 2 * (i + 0.5) / 10 + r.normal(0, 0.03); return [x, f(x) + r.normal(0, 0.25)]; });
    const te = ML.range(200).map(function () { const x = r.uniform(-1, 1); return [x, f(x) + r.normal(0, 0.25)]; });
    const deg = 9, feats = function (x) { return ML.range(deg + 1).map(function (k) { return Math.pow(x, k); }); };
    const st = { lg: -8 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'log₁₀ λ', -8, 1, 0.25, st.lg, function (v) { st.lg = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function fit(lam) {
      const X = tr.map(function (p) { return feats(p[0]); }), A = ML.mat(deg + 1, deg + 1), b = ML.zeros(deg + 1);
      X.forEach(function (row, i) { for (let a = 0; a <= deg; a++) { b[a] += row[a] * tr[i][1]; for (let c = 0; c <= deg; c++) A[a][c] += row[a] * row[c]; } });
      for (let a = 0; a <= deg; a++) A[a][a] += lam;
      return U.solve(A, b);
    }
    function mse(w, pts) { let s = 0; pts.forEach(function (p) { const d = ML.dot(w, feats(p[0])) - p[1]; s += d * d; }); return s / pts.length; }
    function draw() {
      const lam = Math.pow(10, st.lg), w = fit(lam), xs = ML.linspace(-1, 1, 160);
      Book.chart.lines(plot, [
        { color: 'var(--muted)', marker: 'none', dash: '4 3', points: xs.map(function (x) { return [x, f(x)]; }) },
        { name: 'fit', color: Book.palette[0], marker: 'none', points: xs.map(function (x) { return [x, Book.clamp(ML.dot(w, feats(x)), -2.4, 2.4)]; }) },
        { color: Book.palette[1], noline: true, points: tr, r: 4.5 }
      ], { width: 640, height: 240, xmin: -1, xmax: 1, ymin: -2.4, ymax: 2.4, xticks: [-1, -0.5, 0, 0.5, 1], xlabel: 'x', ylabel: 'y', yticks: 4, mr: 40 });
      out.innerHTML = 'λ = ' + lam.toExponential(1) + ' · training error (mean squared) <b>' + U.f(mse(w, tr), 3) + '</b> · test error <b>' + U.f(mse(w, te), 3) + '</b> · size of the weights ‖w‖ = <b>' + U.f(ML.norm(w), 1) + '</b>' +
        '<br><span class="note">Dashed: the true curve. Tiny λ: zero training error, wild curve, large test error. Moderate λ: both errors small. Huge λ: everything shrinks to a flat line (underfitting).</span>';
    }
    draw();
  };
  // ---------- ch-penalty: L2 vs L1 on a two-weight problem
  W.penaltyPaths = function (host) {
    Book.head(host, 'L2 shrinks every weight, L1 sets some to exactly zero',
      'A loss with two weights has its best point at w* (the star); the ellipses are equal-loss contours. Adding a penalty pulls the solution towards 0. Slide λ and switch the penalty: L2 (λ‖w‖²) moves along a curve, L1 (λ(|w₁| + |w₂|)) hits the axis and stays there.');
    const H = [[2.0, 0.6], [0.6, 0.8]], ws = [1.6, 0.6];      // loss = ½ (w − w*)ᵀ H (w − w*)
    const st = { lam: 0.5, kind: 'L2' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'λ', 0, 3, 0.05, st.lam, function (v) { st.lam = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.select(ctl, 'penalty', ['L2', 'L1'], st.kind, function (v) { st.kind = v; draw(); });
    const box = el('div', { class: 'grid2' }, host), pic = el('div', null, box), out = el('div', { class: 'out' }, box);
    function solve(lam, kind) {
      if (kind === 'L2') {       // minimise ½(w−w*)ᵀH(w−w*) + λ‖w‖²  →  (H + 2λI) w = H w*
        const A = [[H[0][0] + 2 * lam, H[0][1]], [H[1][0], H[1][1] + 2 * lam]], b = ML.matvec(H, ws);
        return U.solve(A, b);
      }
      let w = ws.slice();       // coordinate descent with soft-thresholding for the L1 penalty
      for (let it = 0; it < 200; it++) for (let j = 0; j < 2; j++) {
        const o = 1 - j, rj = H[j][j] * ws[j] + H[j][o] * (ws[o] - w[o]);
        w[j] = Math.sign(rj) * Math.max(0, Math.abs(rj) - lam) / H[j][j];
      }
      return w;
    }
    function draw() {
      pic.innerHTML = '';
      const Wd = 300, sc = 90, ox = 60, oy = 200;
      const svg = S('svg', { viewBox: '0 0 ' + Wd + ' 260', role: 'img', 'aria-label': 'two-weight penalty picture' }, pic);
      const P = function (w) { return [ox + w[0] * sc, oy - w[1] * sc]; };
      S('line', { x1: 10, x2: Wd - 10, y1: oy, y2: oy, stroke: 'var(--muted)' }, svg); S('line', { x1: ox, x2: ox, y1: 10, y2: 250, stroke: 'var(--muted)' }, svg);
      S('text', { x: Wd - 20, y: oy - 6, 'font-size': 12, class: 'svg-text', text: 'w₁' }, svg); S('text', { x: ox + 6, y: 20, 'font-size': 12, class: 'svg-text', text: 'w₂' }, svg);
      [0.05, 0.25, 0.6, 1.1].forEach(function (lv) {           // contours of the loss
        const pts = [];
        for (let a = 0; a <= 64; a++) {
          const t = 2 * Math.PI * a / 64, d = [Math.cos(t), Math.sin(t)], q = 0.5 * ML.dot(d, ML.matvec(H, d)), s = Math.sqrt(lv / q);
          pts.push(P([ws[0] + s * d[0], ws[1] + s * d[1]]).join(','));
        }
        S('polyline', { points: pts.join(' '), fill: 'none', stroke: 'var(--line)', 'stroke-width': 1.4 }, svg);
      });
      const path = ML.linspace(0, 3, 61).map(function (l) { return P(solve(l, st.kind)).join(','); });
      S('polyline', { points: path.join(' '), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2, 'stroke-dasharray': '4 3' }, svg);
      const s0 = P(ws); S('text', { x: s0[0] - 7, y: s0[1] + 6, 'font-size': 18, fill: 'var(--red)', text: '★' }, svg);
      const w = solve(st.lam, st.kind), pw = P(w);
      S('circle', { cx: pw[0], cy: pw[1], r: 6, fill: 'var(--accent)', stroke: 'var(--panel)', 'stroke-width': 1.5 }, svg);
      const loss = 0.5 * ML.dot(ML.sub(w, ws), ML.matvec(H, ML.sub(w, ws)));
      out.innerHTML = 'best point without penalty w* = (' + ws.join(', ') + ')\nwith ' + st.kind + ', λ = ' + st.lam.toFixed(2) + ':\nw = (<b>' + U.f(w[0], 3) + '</b>, <b>' + U.f(w[1], 3) + '</b>)\nloss ' + U.f(loss, 3) + ' + penalty ' +
        U.f(st.kind === 'L2' ? st.lam * ML.dot(w, w) : st.lam * (Math.abs(w[0]) + Math.abs(w[1])), 3) + '\n\n<span class="note">Dashed: where the solution goes as λ grows from 0 to 3. With L1, w₂ becomes exactly 0 at a moderate λ.</span>';
    }
    draw();
  };

  // ---------- ch-ensemble: averaging several trained networks
  W.ensembleAvg = function (host) {
    Book.head(host, 'Averaging networks beats each one',
      'Eight small networks (49-32-10) are trained on the same 500 toy digits, each from a different random start and a different order of examples. Press Train, then slide how many of them to average: the averaged prediction usually makes fewer mistakes than a typical single network.');
    const st = { k: 8, nets: null, job: null };
    const ctl = el('div', { class: 'controls' }, host);
    const go = Book.button(ctl, 'Train 8 networks', function () { train(); }, 'primary');
    const kS = Book.slider(ctl, 'networks averaged', 1, 8, 1, st.k, function (v) { st.k = v; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out', text: 'Press “Train 8 networks” (a few seconds, toy).' }, host);
    function train() {
      if (st.job) return;
      go.disabled = true;
      const cfgs = ML.range(8).map(function (i) { return Object.assign({}, Toy.BASE, { sizes: [49, 32, 10], steps: 1500, seed: 100 + i, label: 'net ' + (i + 1), evalEvery: 1500 }); });
      st.job = Toy.runJobs(cfgs, function (i, T) { out.textContent = 'training network ' + (i + 1) + ' of 8: step ' + T.t + '…'; }, function (jobs) { st.nets = jobs.map(function (T) { return T.net; }); st.job = null; go.disabled = false; draw(); });
    }
    function draw() {
      if (!st.nets) return;
      const te = Toy.data(500).test, P = st.nets.map(function (n) { return n.probs(te.X); });
      const single = P.map(function (p) { let w = 0; p.forEach(function (q, i) { if (ML.argmax(q) !== te.y[i]) w++; }); return 100 * w / te.y.length; });
      let wrong = 0;
      for (let i = 0; i < te.y.length; i++) { const avg = ML.zeros(10); for (let k = 0; k < st.k; k++) for (let c = 0; c < 10; c++) avg[c] += P[k][i][c] / st.k; if (ML.argmax(avg) !== te.y[i]) wrong++; }
      const ens = 100 * wrong / te.y.length;
      const data = single.map(function (e, i) { return { label: 'net ' + (i + 1), value: e, note: e.toFixed(1), color: i < st.k ? Book.palette[0] : 'var(--line)' }; });
      data.push({ label: 'average of\n' + st.k, value: ens, note: ens.toFixed(1), color: Book.palette[2] });
      Book.chart.bars(chart, data, { ylabel: 'test error % (toy)', mb: 50 });
      out.innerHTML = 'mean error of the ' + st.k + ' single networks used: <b>' + U.pct(ML.mean(single.slice(0, st.k))) + '</b> · error of their averaged prediction: <b>' + U.pct(ens) + '</b> (toy, 1,000 test digits)' +
        '<br><span class="note">Averaging helps most when the members make different mistakes. Here they differ only by their random start, so the gain is modest; members with different architectures or data would differ more.</span>';
    }
  };

  // ---------- ch-ensemble: arithmetic vs geometric mean of predictions
  W.meanTypes = function (host) {
    Book.head(host, 'Two ways to average predictions',
      'Three models each give probabilities for three classes. The arithmetic mean adds them up; the geometric mean multiplies them and takes the cube root, then renormalises so the result adds to 1. Drag model C’s belief in class 1 towards 0 and watch the geometric mean veto it.');
    const M = [[0.6, 0.3, 0.1], [0.5, 0.2, 0.3], [0.4, 0.4, 0.2]];
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'model C: P(class 1)', 0.001, 0.98, 0.001, M[2][0], function (v) { const rest = 1 - v, o = M[2][1] + M[2][2] || 1; M[2] = [v, rest * M[2][1] / o, rest * M[2][2] / o]; draw(); }, function (v) { return v.toFixed(3); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const ar = [0, 1, 2].map(function (c) { return (M[0][c] + M[1][c] + M[2][c]) / 3; });
      const g0 = [0, 1, 2].map(function (c) { return Math.pow(M[0][c] * M[1][c] * M[2][c], 1 / 3); }), Z = ML.sum(g0), ge = g0.map(function (v) { return v / Z; });
      const data = [];
      ['A', 'B', 'C'].forEach(function (n, k) { data.push({ label: n + '\nclass 1', value: M[k][0], note: M[k][0].toFixed(2), color: 'var(--muted)' }); });
      data.push({ label: 'arithmetic\nclass 1', value: ar[0], note: ar[0].toFixed(2), color: Book.palette[0] });
      data.push({ label: 'geometric\nclass 1', value: ge[0], note: ge[0].toFixed(2), color: Book.palette[4] });
      Book.chart.bars(chart, data, { max: 1, ylabel: 'P(class 1)', mb: 50, tickFmt: function (v) { return v.toFixed(2); } });
      out.innerHTML = 'arithmetic mean: (' + ar.map(function (v) { return v.toFixed(3); }).join(', ') + ')\n' +
        'geometric mean before renormalising: (' + g0.map(function (v) { return v.toFixed(3); }).join(', ') + '), sum ' + Z.toFixed(3) + '\n' +
        'normalised geometric mean:          (<b>' + ge.map(function (v) { return v.toFixed(3); }).join(', ') + '</b>)\n<span class="note">If any one model gives a class a probability near 0, the geometric mean gives it nearly 0 too. The paper describes dropout as approximating an equally weighted geometric mean (§1).</span>';
    }
    draw();
  };

  // ---------- ch-random: Bernoulli and Gaussian random variables
  W.bernoulliLab = function (host) {
    Book.head(host, 'Coin flips with a bias p',
      'A Bernoulli(p) variable r is 1 with probability p and 0 otherwise, like a biased coin. Draw many and compare the running average with p, and the spread with p(1 − p). Switch to the Gaussian N(1, σ²) used in §10 of the paper.');
    const st = { p: 0.5, kind: 'bern', sig: 1, draws: [], rng: ML.rng(17) };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'variable', [{ value: 'bern', label: 'Bernoulli(p)' }, { value: 'gauss', label: 'Gaussian N(1, σ²)' }], st.kind, function (v) { st.kind = v; st.draws = []; draw(); });
    Book.slider(ctl, 'p', 0.05, 0.95, 0.05, st.p, function (v) { st.p = v; st.draws = []; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'σ', 0.2, 2, 0.1, st.sig, function (v) { st.sig = v; st.draws = []; draw(); }, function (v) { return v.toFixed(1); });
    Book.button(ctl, 'Draw 1', function () { add(1); }, 'primary');
    Book.button(ctl, 'Draw 100', function () { add(100); });
    Book.button(ctl, 'Draw 10,000', function () { add(10000); });
    const box = el('div', { class: 'grid2' }, host), hist = el('div', null, box), out = el('div', { class: 'out' }, box);
    function add(n) { for (let i = 0; i < n; i++) st.draws.push(st.kind === 'bern' ? (st.rng.next() < st.p ? 1 : 0) : 1 + st.sig * st.rng.normal(0, 1)); draw(); }
    function draw() {
      const d = st.draws, n = d.length, m = n ? ML.mean(d) : 0, v = n ? ML.variance(d) : 0;
      let data;
      if (st.kind === 'bern') { const ones = d.filter(function (x) { return x === 1; }).length; data = [{ label: 'r = 0', value: n ? (n - ones) / n : 0, color: 'var(--muted)' }, { label: 'r = 1', value: n ? ones / n : 0, color: Book.palette[0] }]; data.forEach(function (b) { b.note = b.value.toFixed(3); }); }
      else { const edges = ML.linspace(-3, 5, 17), cnt = new Array(16).fill(0); d.forEach(function (x) { const k = Math.floor((x + 3) / 0.5); if (k >= 0 && k < 16) cnt[k]++; }); data = cnt.map(function (c, i) { return { label: i % 2 ? '' : String(edges[i]), value: n ? c / n : 0, note: '', color: Book.palette[4] }; }); }
      Book.chart.bars(hist, data, { width: 320, height: 200, ylabel: 'fraction of draws', tickFmt: function (x) { return x.toFixed(2); } });
      const EM = st.kind === 'bern' ? st.p : 1, EV = st.kind === 'bern' ? st.p * (1 - st.p) : st.sig * st.sig;
      out.innerHTML = 'draws: <b>' + n + '</b>\naverage: <b>' + U.f(m, 3) + '</b>   (expected value E[r] = ' + U.f(EM, 3) + ')\nvariance: <b>' + U.f(v, 3) + '</b>   (Var[r] = ' + (st.kind === 'bern' ? 'p(1 − p) = ' : 'σ² = ') + U.f(EV, 3) + ')' +
        (st.kind === 'bern' ? '\n\n<span class="note">With 1/p instead of 1 for a kept unit (§10), E = 1 and Var = (1 − p)/p = ' + U.f((1 - st.p) / st.p, 3) + '.</span>' : '');
    }
    draw();
  };

  // ---------- ch-random: the sum of randomly kept inputs
  W.sumOfCoins = function (host) {
    Book.head(host, 'A sum of randomly kept inputs',
      'A unit adds eight weighted inputs, z = Σⱼ wⱼ rⱼ yⱼ, where each rⱼ is a fresh Bernoulli(p) coin: input j is kept or dropped. Sample many coin patterns and look at how z is spread. Its average is p times the full sum, exactly.');
    const r0 = ML.rng(5), wy = ML.range(8).map(function () { return Math.round(r0.uniform(-1, 2) * 10) / 10; });
    const st = { p: 0.5, zs: [], rng: ML.rng(9) };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'p', 0.1, 1, 0.05, st.p, function (v) { st.p = v; st.zs = []; add(2000); }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'Sample 2,000 patterns', function () { add(2000); }, 'primary');
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function add(n) { for (let i = 0; i < n; i++) { let z = 0; for (let j = 0; j < 8; j++) if (st.rng.next() < st.p) z += wy[j]; st.zs.push(z); } draw(); }
    function draw() {
      const full = ML.sum(wy), n = st.zs.length, lo = -4, hi = 9, nb = 26, cnt = new Array(nb).fill(0);
      st.zs.forEach(function (z) { const k = Math.floor((z - lo) / (hi - lo) * nb); if (k >= 0 && k < nb) cnt[k]++; });
      const pts = cnt.map(function (c, i) { return [lo + (i + 0.5) * (hi - lo) / nb, n ? c / n : 0]; });
      const r = Book.chart.lines(plot, [{ color: Book.palette[0], points: pts, marker: 'none' }], { width: 640, height: 220, xmin: lo, xmax: hi, ymin: 0, ymax: 0.3, xticks: [-4, -2, 0, 2, 4, 6, 8], xlabel: 'z', ylabel: 'fraction', ydec: 2 });
      [[st.p * full, 'p × full sum', 'var(--red)'], [full, 'full sum', 'var(--muted)']].forEach(function (m) { const x = r.X(m[0]); S('line', { x1: x, x2: x, y1: 16, y2: 178, stroke: m[2], 'stroke-width': 2, 'stroke-dasharray': '5 3' }, r.svg); S('text', { x: x + 4, y: 30, 'font-size': 11, fill: m[2], text: m[1] }, r.svg); });
      const m = n ? ML.mean(st.zs) : 0, v = n ? ML.variance(st.zs) : 0, sq = ML.sum(wy.map(function (a) { return a * a; }));
      out.innerHTML = 'the eight products wⱼyⱼ: ' + wy.join(', ') + ' (full sum ' + U.f(full, 1) + ')\nsamples: ' + n + ' · average z = <b>' + U.f(m, 3) + '</b> vs p × full sum = <b>' + U.f(st.p * full, 3) + '</b>\nvariance of z = ' + U.f(v, 3) + ' vs p(1 − p) Σ(wⱼyⱼ)² = ' + U.f(st.p * (1 - st.p) * sq, 3);
    }
    add(2000);
  };

  // ---------- ch-conv: a filter sliding over a toy digit, then max-pooling
  W.convDemo = function (host) {
    Book.head(host, 'A filter slides over the image',
      'A convolutional layer applies the same small 3×3 filter at every position of the image: each output number is the weighted sum of the 9 pixels under the filter. Then ReLU, then max-pooling over 3×3 windows with stride 2 (the pooling used for SVHN in the paper).');
    const FIL = { 'vertical edge': [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], 'horizontal edge': [[-1, -1, -1], [0, 0, 0], [1, 1, 1]], 'blob': [[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]], 'diagonal': [[1, 0, -1], [0, 1, 0], [-1, 0, 1]] };
    const st = { k: 4, f: 'vertical edge', seed: 3 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'digit', ML.range(10).map(String), String(st.k), function (v) { st.k = +v; draw(); });
    Book.select(ctl, 'filter', Object.keys(FIL), st.f, function (v) { st.f = v; draw(); });
    Book.button(ctl, 'Another example', function () { st.seed++; draw(); });
    const row = el('div', { style: 'display:flex;flex-wrap:wrap;gap:18px;align-items:center' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      row.innerHTML = '';
      const x = Toy.sample(st.k, ML.rng(st.seed * 13 + st.k)), F = FIL[st.f], map = [];
      for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) { let s = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += F[a][b] * x[(i + a) * 7 + j + b]; map.push(Math.max(0, s)); }
      const pool = [];
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { let m = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) m = Math.max(m, map[(2 * i + a) * 5 + 2 * j + b]); pool.push(m); }
      const mx = Math.max.apply(null, map.concat([1e-9]));
      const cap = function (t, node) { const d = el('div', { style: 'text-align:center;font-size:12px;color:var(--muted)' }, row); d.appendChild(node); el('div', { text: t }, d); };
      cap('image 7×7', U.digit(el('div'), x, { px: 112, cell: 16 }));
      const fdiv = el('div'); U.digit(fdiv, [].concat.apply([], F), { px: 60, cell: 20, signed: true }); cap('filter 3×3', fdiv);
      const mdiv = el('div'); U.digit(mdiv, map.map(function (v) { return v / mx; }), { px: 90, cell: 18 }); cap('ReLU(feature map) 5×5', mdiv);
      const pdiv = el('div'); U.digit(pdiv, pool.map(function (v) { return v / mx; }), { px: 50, cell: 25 }); cap('max-pool → 2×2', pdiv);
      out.innerHTML = 'feature map (ReLU): ' + map.map(function (v) { return v.toFixed(1); }).join(' ') + '\nafter 3×3 max-pooling with stride 2: ' + pool.map(function (v) { return v.toFixed(2); }).join(', ') +
        '\n<span class="note">The filter has only 9 weights but is used at all 25 positions: that is weight sharing. Pooling keeps the strongest response in each window, so a one-pixel shift barely changes the result.</span>';
    }
    draw();
  };
  // ---------- ch-conv: parameter counts of the paper's SVHN network (App. B.2)
  W.convParams = function (host) {
    Book.head(host, 'Where the parameters of a conv net live',
      'The paper’s best SVHN network (App. B.2): three convolutional layers with 96, 128 and 256 filters of size 5×5, each followed by max-pooling, then two fully connected layers of 2048 units and 10 outputs. The paper does not state the size of the last feature map; choose it to see the first fully connected layer’s size change.');
    const st = { s: 3 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'last map side (assumed)', 1, 6, 1, st.s, function (v) { st.s = v; draw(); }, function (v) { return v + '×' + v; });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const L = [['conv1 5×5×3 → 96', 5 * 5 * 3 * 96 + 96], ['conv2 5×5×96 → 128', 5 * 5 * 96 * 128 + 128], ['conv3 5×5×128 → 256', 5 * 5 * 128 * 256 + 256],
        ['fc1 ' + (256 * st.s * st.s) + ' → 2048', 256 * st.s * st.s * 2048 + 2048], ['fc2 2048 → 2048', 2048 * 2048 + 2048], ['output 2048 → 10', 2048 * 10 + 10]];
      const tot = L.reduce(function (a, b) { return a + b[1]; }, 0);
      U.hbars(chart, L.map(function (l, i) { return { label: l[0], value: l[1] / 1e6, note: (l[1] / 1e6).toFixed(2) + ' M', color: i < 3 ? Book.palette[5] : Book.palette[0] }; }), { labelWidth: 200, tickFmt: function (v) { return v.toFixed(1) + 'M'; } });
      const conv = L[0][1] + L[1][1] + L[2][1];
      out.innerHTML = 'total ≈ <b>' + (tot / 1e6).toFixed(2) + ' million</b> weights · convolutional layers: ' + (conv / 1e6).toFixed(2) + ' M (' + (100 * conv / tot).toFixed(0) + ' %) · fully connected: ' + ((tot - conv) / 1e6).toFixed(2) + ' M' +
        '\n<span class="note">Counts are weights + biases computed from App. B.2 (5×5 filters, 3 colour channels); the fully connected input size depends on the assumed map side. Most parameters sit in the fully connected layers, which is why one might expect dropout to matter only there (§6.1.2).</span>';
    }
    draw();
  };
})();
