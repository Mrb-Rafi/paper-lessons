/* Batch Normalization book, Part V (the experiments) widgets. Uses Book, ML, NN, BNT.
   Paper numbers (Fig. 3, Fig. 4) are exact; curves marked ≈ were read off Paper Fig. 2 by eye. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, T = window.BNT;
  const f = T.f, pc = function (v) { return Math.round(100 * v) + '%'; };
  function table(rows, head) {
    let h = '<table class="mtab"><tr>' + head.map(function (c) { return '<th>' + c + '</th>'; }).join('') + '</tr>';
    rows.forEach(function (r) { h += '<tr>' + r.map(function (c) { return '<td>' + c + '</td>'; }).join('') + '</tr>'; });
    return h + '</table>';
  }
  function softmax(row) { const m = Math.max.apply(null, row), e = row.map(function (v) { return Math.exp(v - m); }), s = ML.sum(e); return e.map(function (v) { return v / s; }); }
  // horizontal bars with long labels: items [{label, value, color, note}]
  function hbars(host, items, o) {
    host.innerHTML = ''; o = o || {};
    const Wd = o.width || 640, row = 24, H = items.length * row + 26, lw = o.lw || 200, max = o.max;
    const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H }, host), X = function (v) { return lw + v / max * (Wd - lw - 70); };
    (o.ticks || []).forEach(function (t) { S('line', { x1: X(t), x2: X(t), y1: 4, y2: H - 18, stroke: 'var(--line)' }, svg); S('text', { x: X(t), y: H - 4, 'text-anchor': 'middle', 'font-size': 10.5, class: 'svg-text', text: o.tfmt ? o.tfmt(t) : t }, svg); });
    items.forEach(function (it, i) {
      const y = 6 + i * row;
      S('text', { x: lw - 8, y: y + 14, 'text-anchor': 'end', 'font-size': 12, class: 'svg-text', 'font-weight': it.bold ? 700 : 400, text: it.label }, svg);
      if (it.value != null) { S('rect', { x: lw, y: y + 2, width: Math.max(1, X(it.value) - lw), height: row - 8, rx: 3, fill: it.color || 'var(--accent)' }, svg); S('text', { x: X(it.value) + 5, y: y + 14, 'font-size': 11.5, 'font-weight': 600, class: 'svg-text', text: it.note }, svg); }
      else S('text', { x: lw + 4, y: y + 14, 'font-size': 11, class: 'svg-text', text: it.note || '–' }, svg);
    });
  }

  // ------------------------------------------------------------------ ch28: Fig. 1 at toy scale
  W.fig1Toy = function (host) {
    Book.head(host, 'Fig. 1, rebuilt: accuracy and one sigmoid’s input over training',
      'Two 3×32 sigmoid networks train live on the glyphs (toy). (a) test accuracy; (b, c) the 15th, 50th and 85th percentiles of one last-hidden-layer sigmoid input over 300 test glyphs, without and with BN. Pick another unit to see that the pattern is typical.');
    const st = { unit: 3 }, ctl = el('div', { class: 'controls' }, host);
    const go = Book.button(ctl, 'Pause', function () { toggle(); }, 'primary');
    Book.button(ctl, 'Restart', function () { reset(); toggle(true); });
    Book.slider(ctl, 'unit (restarts)', 0, 31, 1, st.unit, function (v) { st.unit = v; reset(); toggle(true); });
    const g = el('div', { class: 'hgrid' }, host), pa = el('div', null, g), pb = el('div', null, g), pcc = el('div', null, g), out = el('div', { class: 'out' }, host);
    const R = T.runner(); let jobs, running = false;
    function cfg(bn) { return { net: { sizes: [64, 32, 32, 32, 10], act: 'sigmoid', bn: bn, sd: 'xavier', seed: 5 }, lr: 0.5, momentum: 0, bs: 60, steps: 3000, every: 50, probe: { layer: 2, unit: st.unit } }; }
    function reset() { R.stop(); running = false; jobs = [T.job(cfg(false)), T.job(cfg(true))]; draw(); }
    function band(h, J, color, title) {
      h.innerHTML = ''; el('div', { class: 'note', html: '<b style="color:' + color + '">' + title + '</b>' }, h); const c = el('div', null, h);
      Book.chart.lines(c, [1, 2, 3].map(function (k) { return { name: '', color: color, marker: 'none', width: k === 2 ? 2.4 : 1.5, points: J.pct.map(function (p) { return [p[0], Book.clamp(p[k], -5, 5)]; }) }; }), { xmin: 0, xmax: 3000, ymin: -5, ymax: 5, yticks: 5, xticks: [0, 1500, 3000], xlabel: 'step', height: 230, width: 330, ydec: 0 });
    }
    function draw() {
      pa.innerHTML = ''; el('div', { class: 'note', html: '<b>(a) test accuracy</b>' }, pa); const c = el('div', null, pa);
      Book.chart.lines(c, [{ name: '', color: T.colors.off, marker: 'none', dash: '6 4', points: jobs[0].acc }, { name: '', color: T.colors.on, marker: 'none', points: jobs[1].acc }], { xmin: 0, xmax: 3000, ymin: 0, ymax: 1, xticks: [0, 1500, 3000], xlabel: 'step', height: 230, width: 330, yfmt: pc });
      band(pb, jobs[0], T.colors.off, '(b) without BN'); band(pcc, jobs[1], T.colors.on, '(c) with BN');
      const drift = function (J) { const p = J.pct.filter(function (q) { return q[0] >= 300; }); if (p.length < 2) return 0; const med = p.map(function (q) { return q[2]; }); return Math.max.apply(null, med) - Math.min.apply(null, med); };
      const a0 = jobs[0].acc[jobs[0].acc.length - 1][1], a1 = jobs[1].acc[jobs[1].acc.length - 1][1];
      out.innerHTML = 'step <b>' + jobs[0].step + '</b>/3000 · test accuracy (toy): without BN <b>' + (100 * a0).toFixed(1) + '%</b>, with BN <b>' + (100 * a1).toFixed(1) + '%</b> · range of the median after step 300: without BN <b>' + f(drift(jobs[0]), 2) + '</b>, with BN <b>' + f(drift(jobs[1]), 2) + '</b>';
    }
    function toggle(force) {
      running = force != null ? force : !running; go.textContent = running ? 'Pause' : 'Train ▶';
      if (!running) { R.stop(); return; }
      let last = 0; R.start(jobs, function () { if (performance.now() - last > 300) { last = performance.now(); draw(); } }, function () { running = false; go.textContent = 'Train ▶'; draw(); });
    }
    reset(); toggle(true);
  };

  // ------------------------------------------------------------------ ch29: top-1 and top-5
  W.topK = function (host) {
    Book.head(host, 'Top-1 and top-k: is the right answer among the best k guesses?',
      'Five photos (toy) and a classifier’s probabilities for 10 of its classes. Raise k and count how many photos have the correct class among the k highest.');
    const st = { k: 1 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'k', 1, 5, 1, st.k, function (v) { st.k = v; draw(); });
    const view = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    const cls = ['tabby cat', 'tiger cat', 'lynx', 'fox', 'dog', 'wolf', 'teapot', 'mug', 'car', 'truck'];
    const ex = [['tabby cat', [30, 45, 10, 3, 4, 1, 1, 1, 3, 2]], ['fox', [1, 1, 6, 52, 20, 15, 1, 1, 2, 1]], ['mug', [1, 1, 1, 1, 1, 1, 48, 40, 3, 3]], ['wolf', [2, 1, 8, 25, 38, 20, 1, 1, 2, 2]], ['truck', [1, 1, 1, 1, 2, 1, 2, 1, 55, 35]]];
    function draw() {
      let ok = 0; const rows = ex.map(function (e) {
        const p = e[1].map(function (v) { return v / ML.sum(e[1]); }), order = ML.range(10).sort(function (a, b) { return p[b] - p[a]; }), top = order.slice(0, st.k).map(function (i) { return cls[i]; });
        const hit = top.indexOf(e[0]) >= 0; if (hit) ok++;
        return [e[0], top.map(function (c, i) { return (c === e[0] ? '<b>' + c + '</b>' : c) + ' (' + Math.round(100 * p[order[i]]) + '%)'; }).join(', '), hit ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>'];
      });
      view.innerHTML = table(rows, ['true class', 'top-' + st.k + ' guesses', 'correct?']);
      out.innerHTML = 'top-' + st.k + ' accuracy <b>' + ok + '/5</b>, top-' + st.k + ' error <b>' + (100 * (5 - ok) / 5) + '%</b>. Top-5 is kinder: confusing two kinds of cat is forgiven.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch29: 5×5 vs two 3×3
  W.conv3x3 = function (host) {
    Book.head(host, 'Two 3×3 convolutions see as far as one 5×5, with fewer weights',
      'Left: an output location of a 5×5 filter looks at a 5×5 patch; a 3×3 filter on top of a 3×3 filter also reaches 5×5. Right: weights for C input and C output channels.');
    const st = { C: 128 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'channels C', 16, 256, 16, st.C, function (v) { st.C = v; draw(); });
    const g = el('div', { class: 'grid2' }, host), L = el('div', null, g), R = el('div', null, g), out = el('div', { class: 'out' }, host);
    function grid(svg, ox, title, fill) {
      S('text', { x: ox, y: 14, 'font-size': 11.5, class: 'svg-text', text: title }, svg);
      for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) S('rect', { x: ox + j * 18, y: 22 + i * 18, width: 17, height: 17, fill: fill(i, j), stroke: 'var(--line)' }, svg);
    }
    function draw() {
      L.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 300 160' }, L), in5 = function (i, j) { return Math.abs(i - 3) <= 2 && Math.abs(j - 3) <= 2; };
      grid(svg, 4, 'one 5×5', function (i, j) { return i === 3 && j === 3 ? '#c0392b' : in5(i, j) ? 'var(--accent-soft)' : 'var(--panel)'; });
      grid(svg, 160, 'two 3×3 (stacked)', function (i, j) { return i === 3 && j === 3 ? '#c0392b' : (Math.abs(i - 3) <= 1 && Math.abs(j - 3) <= 1) ? 'var(--accent)' : in5(i, j) ? 'var(--accent-soft)' : 'var(--panel)'; });
      const a = 25 * st.C * st.C, b = 18 * st.C * st.C;
      Book.chart.bars(R, [{ label: 'one 5×5', value: a, note: a.toLocaleString('en-US'), color: '#c0392b' }, { label: 'two 3×3', value: b, note: b.toLocaleString('en-US'), color: 'var(--accent)' }], { height: 200, tickFmt: function (v) { return v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : (v / 1e3).toFixed(0) + 'k'; } });
      out.innerHTML = 'C = ' + st.C + ': 25·C² = <b>' + a.toLocaleString('en-US') + '</b> weights vs 2·9·C² = <b>' + b.toLocaleString('en-US') + '</b> (28% fewer), with an extra nonlinearity in between. Dark square: the middle 3×3 that the second filter reads; light: the 5×5 reach of the stack.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch30: learning-rate decay 6× faster
  W.lrDecay = function (host) {
    Book.head(host, 'Exponential learning-rate decay, 6 times faster',
      'Learning rate over training for Inception (initial 0.0015) and for a BN variant with its own initial rate, decayed 6 times faster. The decay rate itself is illustrative: the paper does not give it.');
    const st = { lr0: 0.0075, r: 0.9 }, ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'BN variant', [{ value: '0.0015', label: 'BN-Baseline (0.0015)' }, { value: '0.0075', label: 'BN-x5 (0.0075)' }, { value: '0.045', label: 'BN-x30 (0.045)' }], '0.0075', function (v) { st.lr0 = +v; draw(); });
    Book.slider(ctl, 'Inception’s decay per million steps', 0.8, 0.99, 0.01, st.r, function (v) { st.r = v; draw(); }, function (v) { return '×' + v.toFixed(2); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const A = [], B = []; for (let t = 0; t <= 31; t += 0.25) { A.push([t, Math.log10(0.0015 * Math.pow(st.r, t))]); B.push([t, Math.max(-6, Math.log10(st.lr0 * Math.pow(st.r, 6 * t)))]); }
      T.lines(view, [{ name: 'Inception', color: 'var(--muted)', marker: 'none', dash: '6 4', points: A }, { name: 'BN, decayed 6× faster', color: T.colors.on, marker: 'none', points: B }],
        { xmin: 0, xmax: 31, ymin: -6, ymax: -1, yticks: 5, xticks: [0, 5, 10, 15, 20, 25, 31], xlabel: 'training steps (millions)', ylabel: 'learning rate (log₁₀)', height: 230, yfmt: function (v) { return '10^' + v; } });
      const half = Math.log(0.5) / Math.log(st.r);
      out.innerHTML = 'Inception’s rate halves every <b>' + f(half, 1) + 'M</b> steps; the BN schedule every <b>' + f(half / 6, 2) + 'M</b>. At 2.1M steps (BN-x5 reaches 72.2%) the BN rate is ' + (st.lr0 * Math.pow(st.r, 6 * 2.1)).toExponential(1) + ', Inception’s ' + (0.0015 * Math.pow(st.r, 2.1)).toExponential(1) + '.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch31: Fig. 2 explorer (≈) and Fig. 3 (exact)
  const FIG2 = [  // [name, colour, points (millions of steps, accuracy) read off Fig. 2 by eye, steps to 72.2% (Fig. 3), max accuracy (Fig. 3)]
    ['Inception', '#4a5568', [[0.4, 0.40], [1, 0.50], [2, 0.57], [3, 0.60], [4, 0.62], [5, 0.65], [7, 0.67], [10, 0.69], [13, 0.70], [15, 0.705], [20, 0.716], [25, 0.72], [31, 0.722]], 31.0, 0.722],
    ['BN-Baseline', '#c0392b', [[0.2, 0.45], [0.5, 0.58], [1, 0.63], [2, 0.665], [3, 0.68], [5, 0.70], [8, 0.705], [10, 0.715], [13.3, 0.722], [16, 0.727]], 13.3, 0.727],
    ['BN-x5', '#2b6cb0', [[0.1, 0.45], [0.3, 0.60], [0.6, 0.66], [1, 0.69], [1.5, 0.71], [2.1, 0.722], [2.5, 0.73]], 2.1, 0.730],
    ['BN-x30', '#1f4e8c', [[0.1, 0.45], [0.3, 0.58], [0.6, 0.64], [1, 0.67], [1.5, 0.695], [2, 0.712], [2.7, 0.722], [3.5, 0.735], [4.5, 0.744], [6, 0.748], [6.7, 0.748]], 2.7, 0.748],
    ['BN-x5-Sigmoid', '#d53f8c', [[0.2, 0.42], [0.5, 0.55], [1, 0.61], [1.5, 0.645], [2, 0.665], [3, 0.685], [4, 0.693], [5, 0.697], [7.5, 0.698]], null, 0.698]];
  W.fig2Explorer = function (host) {
    Book.head(host, 'Validation accuracy vs training steps (Fig. 2, redrawn ≈)',
      'Curves redrawn by eye from Paper Fig. 2 (approximate); diamonds and the table are the exact Fig. 3 numbers. Pick a target accuracy: the readout estimates the steps each network needs, read off the curves.');
    const st = { target: 0.70, on: FIG2.map(function () { return true; }) }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'target accuracy', 0.60, 0.745, 0.005, st.target, function (v) { st.target = v; draw(); }, function (v) { return (100 * v).toFixed(1) + '%'; });
    FIG2.forEach(function (m, i) { const lab = el('label', null, ctl), c = el('input', { type: 'checkbox' }, lab); c.checked = true; lab.appendChild(document.createTextNode(' ' + m[0])); c.addEventListener('change', function () { st.on[i] = c.checked; draw(); }); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    function stepsTo(pts, a) { for (let i = 1; i < pts.length; i++) if (pts[i][1] >= a) { const p = pts[i - 1], q = pts[i]; return p[0] + (a - p[1]) / (q[1] - p[1]) * (q[0] - p[0]); } return null; }
    function draw() {
      const ser = [];
      FIG2.forEach(function (m, i) { if (!st.on[i]) return; ser.push({ name: m[0], color: m[1], marker: 'none', points: m[2], dash: i === 0 ? '6 4' : i === 4 ? '2 3' : '' }); if (m[3]) ser.push({ name: '', color: m[1], points: [[m[3], 0.722]], r: 5 }); });
      ser.push({ name: '', color: 'var(--muted)', marker: 'none', dash: '3 3', width: 1, points: [[0, st.target], [31, st.target]] });
      T.lines(view, ser, { xmin: 0, xmax: 31, ymin: 0.4, ymax: 0.8, xticks: [0, 5, 10, 15, 20, 25, 30], xlabel: 'training steps (millions)', ylabel: 'validation accuracy @1', height: 260, yfmt: pc });
      const inc = stepsTo(FIG2[0][2], st.target);
      out.innerHTML = 'steps to ' + (100 * st.target).toFixed(1) + '% (≈, from the redrawn curves): ' + FIG2.map(function (m) { const s = stepsTo(m[2], st.target); return m[0] + ' <b>' + (s == null ? 'never' : '≈' + f(s, 1) + 'M') + '</b>' + (s && inc && m[0] !== 'Inception' ? ' (' + f(inc / s, 1) + '× faster)' : ''); }).join(' · ') + '. At exactly 72.2% the paper’s Fig. 3 gives 31.0M, 13.3M, 2.1M, 2.7M and never.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch31: the toy learning-rate race
  W.lrRace = function (host) {
    Book.head(host, 'The learning-rate race, at toy scale',
      'Six runs of an 8-layer, 32-unit ReLU network on the glyphs, 1,500 steps each (SGD + momentum 0.9, minibatch 32). Base learning rate 0.004. Training runs live in your browser (about 15 s).');
    const ctl = el('div', { class: 'controls' }, host);
    Book.button(ctl, 'Run again', function () { start(); }, 'primary');
    const view = el('div', null, host), tab = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    const H = [64, 32, 32, 32, 32, 32, 32, 32, 32, 10];
    const specs = [['no BN, ×1 (like Inception)', false, 1, '#4a5568', '6 4'], ['BN, ×1 (like BN-Baseline)', true, 1, '#c0392b', ''], ['BN, ×5 (like BN-x5)', true, 5, '#2b6cb0', ''], ['BN, ×30 (like BN-x30)', true, 30, '#1f4e8c', '2 3'], ['no BN, ×5', false, 5, '#b7791f', '6 4'], ['no BN, ×30', false, 30, '#d53f8c', '6 4']];
    const R = T.runner(); let jobs;
    function start() {
      jobs = specs.map(function (s) { return T.job({ net: { sizes: H, act: 'relu', bn: s[1], sd: 'he', seed: 5 }, lr: 0.004 * s[2], momentum: 0.9, bs: 32, steps: 1500, every: 50, evalN: 600 }); });
      let last = 0; R.start(jobs, function () { if (performance.now() - last > 400) { last = performance.now(); draw(); } }, draw);
    }
    function draw() {
      T.lines(view, jobs.map(function (J, i) { return { name: specs[i][0], color: specs[i][3], dash: specs[i][4], marker: 'none', points: J.acc }; }), { xmin: 0, xmax: 1500, ymin: 0, ymax: 1, xticks: [0, 500, 1000, 1500], xlabel: 'training step', ylabel: 'test accuracy (toy)', height: 250, yfmt: pc });
      const base = jobs[0].acc.find(function (a) { return a[1] >= 0.9; });
      tab.innerHTML = table(jobs.map(function (J, i) { const h = J.acc.find(function (a) { return a[1] >= 0.9; }); return [specs[i][0], f(0.004 * specs[i][2], 3), h ? String(h[0]) : (J.done ? 'never' : '…'), h && base ? f(base[0] / h[0], 1) + '×' : '–', (100 * J.best).toFixed(1) + '%' + (J.diverged ? ' (diverged)' : '')]; }), ['run', 'learning rate', 'steps to 90%', 'speed-up vs no BN ×1', 'best accuracy']);
      out.innerHTML = 'step ' + jobs[0].step + '/1500 (toy numbers; the paper’s are in the table above).';
    }
    start();
  };

  // ------------------------------------------------------------------ ch32: deep sigmoid nets
  W.sigmoidRace = function (host) {
    Book.head(host, 'A deep sigmoid network: stuck without BN, trainable with it',
      '8 hidden layers of 32 sigmoids, SGD + momentum 0.9, minibatch 32, 1,500 steps on the glyphs (toy). Chance is 10% (10 classes).');
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    const H = [64, 32, 32, 32, 32, 32, 32, 32, 32, 10];
    const specs = [['no BN, rate 0.004', false, 0.004, '#4a5568'], ['no BN, rate 0.02', false, 0.02, '#b7791f'], ['BN, rate 0.02 (×5)', true, 0.02, '#d53f8c']];
    const jobs = specs.map(function (s) { return T.job({ net: { sizes: H, act: 'sigmoid', bn: s[1], sd: 'xavier', seed: 5 }, lr: s[2], momentum: 0.9, bs: 32, steps: 1500, every: 50 }); });
    function draw() {
      T.lines(view, jobs.map(function (J, i) { return { name: specs[i][0], color: specs[i][3], marker: 'none', points: J.acc }; }).concat([{ name: '', color: 'var(--muted)', marker: 'none', dash: '3 3', width: 1, points: [[0, 0.1], [1500, 0.1]] }]), { xmin: 0, xmax: 1500, ymin: 0, ymax: 1, xticks: [0, 500, 1000, 1500], xlabel: 'training step', ylabel: 'test accuracy (toy)', height: 240, yfmt: pc });
      out.innerHTML = 'step ' + jobs[0].step + '/1500 · best test accuracy: ' + jobs.map(function (J, i) { return specs[i][0] + ' <b>' + (100 * J.best).toFixed(1) + '%</b>'; }).join(', ') + '. Paper: BN-x5-Sigmoid 69.8% on ImageNet; Inception with sigmoid, chance (0.1%).';
    }
    let last = 0; T.runner().start(jobs, function () { if (performance.now() - last > 400) { last = performance.now(); draw(); } }, draw);
  };

  // ------------------------------------------------------------------ ch33: an ensemble of toy BN networks
  W.ensembleToy = function (host) {
    Book.head(host, 'Averaging six networks’ probabilities',
      'Six toy BN networks (3×32 sigmoid) differ only in their random initial weights and minibatch order; each trains for 400 steps. The ensemble averages their class probabilities, as in the paper. Slide to include more members.');
    const st = { k: 6 }, ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'members averaged', 1, 6, 1, st.k, function (v) { st.k = v; draw(); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    const jobs = [0, 1, 2, 3, 4, 5].map(function (i) { return T.job({ net: { sizes: [64, 32, 32, 32, 10], act: 'sigmoid', bn: true, sd: 'xavier', seed: 40 + i }, lr: 0.5, momentum: 0, bs: 60, steps: 400, every: 400, seed: 70 + i }); });
    let P = null;
    function finish() {
      const D = T.data().test, X = T.rows(D, ML.range(D.n));
      P = jobs.map(function (J) { const o = J.net.forward(X, 'eval'); return ML.range(D.n).map(function (i) { return softmax(o.row(i)); }); });
      draw();
    }
    function acc(list) { const D = T.data().test; let ok = 0; for (let i = 0; i < D.n; i++) { const avg = new Array(10).fill(0); list.forEach(function (p) { for (let c = 0; c < 10; c++) avg[c] += p[i][c] / list.length; }); if (ML.argmax(avg) === D.y[i]) ok++; } return ok / D.n; }
    function draw() {
      if (!P) { out.textContent = 'training the six members (toy)… ' + jobs.reduce(function (s, J) { return s + J.step; }, 0) + '/2400 steps'; return; }
      const singles = P.map(function (p) { return acc([p]); }), ens = acc(P.slice(0, st.k));
      Book.chart.bars(view, singles.map(function (a, i) { return { label: 'net ' + (i + 1), value: a, note: (100 * a).toFixed(1) + '%', color: i < st.k ? 'var(--accent)' : 'var(--line)' }; }).concat([{ label: 'ensemble\n(' + st.k + ')', value: ens, note: (100 * ens).toFixed(1) + '%', color: '#2f855a' }]), { max: 1, height: 220, tickFmt: pc, mb: 50 });
      out.innerHTML = 'members: mean <b>' + (100 * ML.mean(singles.slice(0, st.k))).toFixed(1) + '%</b>, best <b>' + (100 * Math.max.apply(null, singles.slice(0, st.k))).toFixed(1) + '%</b> · ensemble of ' + st.k + ': <b>' + (100 * ens).toFixed(1) + '%</b> test accuracy (toy).';
    }
    let last = 0; T.runner().start(jobs, function () { if (performance.now() - last > 400) { last = performance.now(); draw(); } }, finish);
  };

  // ------------------------------------------------------------------ ch33: Fig. 4 table
  const FIG4 = [['GoogLeNet ensemble', '224', '144', '7', null, 6.67, false], ['Deep Image low-res', '256', '–', '1', null, 7.96, false], ['Deep Image high-res', '512', '–', '1', 24.88, 7.42, false], ['Deep Image ensemble', 'up to 512', '–', '–', null, 5.98, false],
    ['MSRA multicrop', 'up to 480', '–', '–', null, 5.71, false], ['MSRA ensemble', 'up to 480', '–', '–', null, 4.94, true], ['BN-Inception single crop', '224', '1', '1', 25.2, 7.82, false], ['BN-Inception multicrop', '224', '144', '1', 21.99, 5.82, false], ['BN-Inception ensemble', '224', '144', '6', 20.1, 4.82, true]];
  W.fig4Explorer = function (host) {
    Book.head(host, 'BN-Inception against the state of the art (Fig. 4)',
      'Top-5 error (lower is better) for every row of Paper Fig. 4. * = test-server result on the test set; all others are on the 50,000-image validation set. Switch to top-1 to see the rows that report it.');
    const st = { metric: 5 }, ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'metric', [{ value: '5', label: 'top-5 error' }, { value: '1', label: 'top-1 error' }], '5', function (v) { st.metric = +v; draw(); });
    const view = el('div', { class: 'scroll-x' }, host), tab = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    tab.innerHTML = table(FIG4.map(function (r) { return [r[0], r[1], r[2], r[3], r[4] == null ? '–' : (r[0] === 'Deep Image high-res' ? '24.88' : r[4] + '%'), r[5] + '%' + (r[6] ? '*' : '')]; }), ['Model', 'Resolution', 'Crops', 'Models', 'Top-1 error', 'Top-5 error']);
    function draw() {
      const items = FIG4.map(function (r) { const v = st.metric === 5 ? r[5] : r[4]; return { label: r[0] + (r[6] ? ' *' : ''), value: v, note: v == null ? 'not reported' : v + '%', color: /BN-Inception/.test(r[0]) ? T.colors.on : 'var(--muted)', bold: /BN-Inception/.test(r[0]) }; });
      hbars(view, items, { max: st.metric === 5 ? 9 : 27, lw: 190, ticks: st.metric === 5 ? [0, 2, 4, 6, 8] : [0, 5, 10, 15, 20, 25], tfmt: function (t) { return t + '%'; } });
      out.innerHTML = st.metric === 5 ? 'Best previous test-server result 4.94% (MSRA ensemble) → BN-Inception ensemble <b>4.82%</b>* (4.9% on the validation set). A single BN-Inception with one crop: 7.82%; with 144 crops: 5.82%.' : 'Top-1 error is reported for only four rows: Deep Image high-res 24.88, BN-Inception single crop 25.2%, multicrop 21.99%, ensemble 20.1%.';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch34: small minibatches
  W.smallBatchLimit = function (host) {
    Book.head(host, 'Small minibatches: training mode and inference mode drift apart',
      'Five toy BN networks (3×32 sigmoid) train for 1,000 steps with minibatch sizes 2, 4, 8, 16 and 60 (learning rate scaled with √m). Bars: test accuracy in inference mode (population statistics) and in training mode (test glyphs normalized in batches of the same size m).');
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    const ms = [2, 4, 8, 16, 60];
    const jobs = ms.map(function (m) { return T.job({ net: { sizes: [64, 32, 32, 32, 10], act: 'sigmoid', bn: true, sd: 'xavier', seed: 5 }, lr: 0.5 * Math.sqrt(m / 60), momentum: 0, bs: m, steps: 1000, every: 1000, seed: 9 }); });
    function batchAcc(J, m) { const D = T.data().test; let ok = 0, n = 0; for (let s = 0; s + m <= 600; s += m) { const idx = ML.range(m).map(function (i) { return s + i; }), o = J.net.forward(T.rows(D, idx), 'batch'); idx.forEach(function (k, a) { if (ML.argmax(o.row(a)) === D.y[k]) ok++; n++; }); } return ok / n; }
    function draw(final) {
      if (!final) { out.textContent = 'training (toy)… ' + jobs.map(function (J) { return J.step; }).join(' / ') + ' steps'; return; }
      const items = []; jobs.forEach(function (J, i) { const inf = J.acc[J.acc.length - 1][1], tr = batchAcc(J, ms[i]); items.push({ label: 'm=' + ms[i] + '\ninference', value: inf, note: (100 * inf).toFixed(0) + '%', color: T.colors.on }, { label: 'm=' + ms[i] + '\ntraining', value: tr, note: (100 * tr).toFixed(0) + '%', color: '#b7791f' }); });
      Book.chart.bars(view, items, { max: 1, height: 230, tickFmt: pc, mb: 50 });
      out.innerHTML = 'inference-mode accuracy (toy): ' + ms.map(function (m, i) { return 'm=' + m + ' <b>' + (100 * items[2 * i].value).toFixed(0) + '%</b>'; }).join(', ') + ' · training mode trails inference mode by ' + ms.map(function (m, i) { return f(100 * (items[2 * i].value - items[2 * i + 1].value), 0); }).join(', ') + ' points. With m = 2 every x̂ is ±1 and the network barely learns; small batches give noisy statistics, and the network trained on them disagrees with itself between the two modes.';
    }
    T.runner().start(jobs, function () { draw(false); }, function () { draw(true); });
  };
})();
