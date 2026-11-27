/* Widgets for Part VI (the evidence) and the closing chapter of the ResNet book. Paper numbers are typed in from the
   paper's tables; values read off its plots are marked ≈; toy numbers come from rn.js (toy-results.js holds the stored runs). */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, T = window.RNT, U = window.RNU;

  // ---------------------------------------------------------------- top-1 / top-5
  W.topK = function (host) {
    Book.head(host, 'Top-1 and top-5 error',
      'A classifier gives a score to each of 10 classes (a toy stand-in for ImageNet’s 1000). It is right “in the top k” if the true class is among its k highest scores. Pick an example and k; the error over 40 toy examples is computed below.');
    const st = { i: 0, k: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'example', 0, 39, 1, st.i, function (v) { st.i = v; draw(); });
    Book.slider(ctl, 'k', 1, 5, 1, st.k, function (v) { st.k = v; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    const r = ML.rng(31), ex = [];
    for (let i = 0; i < 40; i++) { const y = r.int(10), s = ML.range(10).map(function () { return r.normal(0, 1); }); s[y] += r.uniform(0.3, 3.6); ex.push({ y: y, p: ML.softmax(s) }); }
    function rank(e) { const order = ML.topk(e.p, 10); return order.indexOf(e.y) + 1; }
    function draw() {
      const e = ex[st.i], top = ML.topk(e.p, st.k);
      Book.chart.bars(plot, e.p.map(function (p, c) { return { label: (c === e.y ? '★' : '') + c, value: 100 * p, color: top.indexOf(c) >= 0 ? 'var(--accent)' : U.C.grey, note: (100 * p).toFixed(0) }; }), { ylabel: 'probability (%)', height: 190 });
      const err = function (k) { return ex.filter(function (q) { return rank(q) > k; }).length / ex.length; };
      out.innerHTML = 'true class ★' + e.y + ' is ranked <b>' + rank(e) + '</b> → ' + (rank(e) <= st.k ? '<span class="ok">correct</span>' : '<span class="bad">wrong</span>') + ' in the top ' + st.k + '. Over all 40 toy examples: top-1 error <b>' + (100 * err(1)).toFixed(1) + '%</b>, top-5 error <b>' + (100 * err(5)).toFixed(1) + '%</b>. Top-5 error is always ≤ top-1 error.';
    }
    draw();
  };

  // ---------------------------------------------------------------- Table 2 and Fig. 4
  const FIG4 = {   // validation (centre-crop) error, read by eye off Fig. 4 (approximate), x in 10^4 iterations
    p18: [[2, 58], [5, 52], [10, 50.5], [14.5, 50], [15.2, 40], [20, 37], [29.5, 36.5], [30.2, 33.5], [35, 32.2], [40, 31.8], [50, 31.5]],
    p34: [[2, 63], [5, 57], [10, 55], [14.5, 55], [15.2, 43], [20, 39.5], [29.5, 38.7], [30.2, 34.5], [35, 33.5], [40, 33], [50, 32.5]],
    r18: [[2, 55], [5, 49], [10, 47.5], [14.5, 47], [15.2, 38], [20, 36], [29.5, 35.5], [30.2, 32.5], [35, 31.5], [40, 31.2], [50, 31]],
    r34: [[2, 52], [5, 46.5], [10, 45], [14.5, 44.5], [15.2, 35.5], [20, 32.5], [29.5, 32.3], [30.2, 29.5], [35, 28.5], [40, 28.3], [50, 28.2]]
  };
  W.fig4 = function (host) {
    Book.head(host, 'Figure 4 and Table 2: the situation reverses',
      'ImageNet. Curves: validation error of the centre crop during training, read by eye off Fig. 4 (≈). Bars: the exact top-1 errors of Table 2 (10-crop testing). Switch between the plain networks and the ResNets.');
    let which = 'plain';
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'networks', [{ value: 'plain', label: 'plain-18 vs plain-34' }, { value: 'res', label: 'ResNet-18 vs ResNet-34' }, { value: 'all', label: 'all four' }], which, function (v) { which = v; draw(); });
    const g2 = el('div', { class: 'grid2' }, host), A = el('div', null, g2), B = el('div', null, g2), out = U.out(host);
    function draw() {
      const s = [];
      if (which !== 'res') s.push({ name: 'plain-18', color: U.C.plain2, marker: 'none', points: FIG4.p18 }, { name: 'plain-34', color: U.C.plain, marker: 'none', points: FIG4.p34 });
      if (which !== 'plain') s.push({ name: 'ResNet-18', color: U.C.res2, marker: 'none', points: FIG4.r18 }, { name: 'ResNet-34', color: U.C.res, marker: 'none', points: FIG4.r34 });
      U.lines(A, s, { xmin: 0, xmax: 50, ymin: 20, ymax: 60, xticks: [0, 10, 20, 30, 40, 50], xlabel: 'iterations (×10⁴)', ylabel: 'val. error (%) ≈', width: 340, height: 230 });
      Book.chart.bars(B, [{ label: 'plain-18', value: 27.94, color: U.C.plain2 }, { label: 'plain-34', value: 28.54, color: U.C.plain }, { label: 'ResNet-18', value: 27.88, color: U.C.res2 }, { label: 'ResNet-34', value: 25.03, color: U.C.res }],
        { ylabel: 'top-1 error (%), Table 2', max: 30, width: 340, height: 230, valFmt: function (v) { return v.toFixed(2); } });
      out.innerHTML = 'Plain: 34 layers are worse than 18 (28.54% vs 27.94%), and the 34-layer net’s <em>training</em> error is higher throughout (thin curves in Fig. 4). ResNet: 34 layers beat 18 by 2.8 points (25.03% vs 27.88%), and beat plain-34 by 3.5 points. At 18 layers both are about equal, but the ResNet gets there faster.';
    }
    draw();
  };

  // ---------------------------------------------------------------- toy CNN race (stored, re-runnable)
  function results() { return window.RNT_RESULTS || {}; }
  W.toyRace = function (host) {
    Book.head(host, 'The same experiment on this book’s toy CNN',
      'Plain and residual versions of the miniature CIFAR family (8, 20 and 32 layers; 6, 12 and 24 filters on 8×8, 4×4 and 2×2 maps; option A shortcuts; BN, He init, SGD with momentum 0.9 and weight decay 0.0001, learning rate 0.1 divided by 10 twice), trained for 800 steps on noisy 8×8 digits. Stored runs; “Re-run live” recomputes them in your browser (several minutes).');
    let show = 'train';
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'error on', [{ value: 'train', label: 'training images' }, { value: 'test', label: 'test images' }], show, function (v) { show = v; draw(); });
    const live = Book.button(ctl, 'Re-run live', function () { rerun(); });
    const g2 = el('div', { class: 'grid2' }, host), A = el('div', null, g2), B = el('div', null, g2), out = U.out(host);
    let R = results();
    const ids = ['plain-8', 'plain-20', 'plain-32', 'res-8', 'res-20', 'res-32'];
    const col = { 'plain-8': U.C.gold, 'plain-20': U.C.plain2, 'plain-32': U.C.plain, 'res-8': U.C.green, 'res-20': U.C.res2, 'res-32': U.C.res };
    function draw() {
      const k = show === 'train' ? 1 : 2;
      [['plain', A], ['res', B]].forEach(function (p) {
        U.lines(p[1], ids.filter(function (id) { return id.indexOf(p[0]) === 0 && R[id]; }).map(function (id) { return { name: id.replace('res', 'ResNet'), color: col[id], points: R[id].curve.map(function (c) { return [c[0], c[k]]; }), r: 2.6 }; }),
          { xmin: 0, xmax: 800, ymin: 0, ymax: 50, xticks: [0, 200, 400, 600, 800], xlabel: 'steps', ylabel: (show === 'train' ? 'training' : 'test') + ' error (%)', width: 340, height: 220 });
      });
      const f = function (id) { return R[id] && R[id].final ? R[id].final[show].toFixed(1) + '%' : '…'; };
      out.innerHTML = 'final ' + show + ' error (toy): plain 8/20/32 = <b>' + f('plain-8') + ' / ' + f('plain-20') + ' / ' + f('plain-32') + '</b> · ResNet 8/20/32 = <b>' + f('res-8') + ' / ' + f('res-20') + ' / ' + f('res-32') + '</b>. On the training images the pattern is the paper’s: deeper plain nets fit worse and worse, ResNets fit at every depth. Toy test errors are noisier (one run each, 500 test digits), and our ResNets do not keep improving past 20 layers the way the paper’s do on CIFAR-10.';
    }
    function rerun() {
      live.disabled = true; R = {};
      const runs = T.CNN_RUNS.filter(function (r) { return ids.indexOf(r.id) >= 0; });
      let i = 0;
      (function next() {
        if (i >= runs.length) { live.disabled = false; return; }
        const run = runs[i++], job = T.cnnJob(run);
        T.run([job], function () { out.textContent = 'running ' + run.id + ': step ' + job.step + ' / ' + T.CNN_STEPS; }, function () { R[run.id] = job.result(); draw(); next(); });
      })();
    }
    draw();
  };

  // ---------------------------------------------------------------- Table 3: options A, B, C
  W.abcBars = function (host) {
    Book.head(host, 'Table 3: identity or projection shortcuts?',
      'ImageNet, 10-crop testing. All three options beat the plain network by a wide margin; the differences between them are small. Parameter counts are computed by this book from Table 1 (with BN); the toy row compares options A and B on the 20-layer toy CNN.');
    let metric = 'top1';
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'metric', [{ value: 'top1', label: 'top-1 error' }, { value: 'top5', label: 'top-5 error' }], metric, function (v) { metric = v; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    const rows = [['plain-34', 28.54, 10.02, U.C.plain], ['ResNet-34 A', 25.03, 7.76, U.C.res2], ['ResNet-34 B', 24.52, 7.46, U.C.res], ['ResNet-34 C', 24.19, 7.40, U.C.purple]];
    function draw() {
      const k = metric === 'top1' ? 1 : 2;
      U.hbars(plot, rows.map(function (r) { return { label: r[0], value: r[k], color: r[3], note: r[k].toFixed(2) + '%' }; }), { min: metric === 'top1' ? 20 : 5, max: metric === 'top1' ? 30 : 11, ticks: metric === 'top1' ? [20, 22, 24, 26, 28, 30] : [5, 7, 9, 11], labelW: 120 });
      const p = function (o) { return T.archTotals(T.archLayers(34, { shortcut: o })).params; }, R = results();
      const toy = R['res-20'] && R['res-20-B'] ? ' Toy (20-layer CNN, one run each): option A ' + R['res-20'].final.test.toFixed(1) + '%, option B ' + R['res-20-B'].final.test.toFixed(1) + '% test error; on a toy this small, differences of this size are within run-to-run noise, so the toy cannot rank A and B.' : '';
      out.innerHTML = 'Parameters: A ' + T.big(p('A')) + ', B ' + T.big(p('B')) + ', C ' + T.big(p('C')) + '. B beats A slightly (the zero-padded channels of A do no residual learning); C beats B marginally, with 13 extra projections. The paper keeps B for the deeper nets, and uses no C.' + toy;
    }
    draw();
  };

  // ---------------------------------------------------------------- Tables 3, 4, 5 explorer
  const TAB = {
    t3: { title: 'Table 3 · single models, 10-crop, validation', rows: [['VGG-16', 28.07, 9.33], ['GoogLeNet', null, 9.15], ['PReLU-net', 24.27, 7.38], ['plain-34', 28.54, 10.02], ['ResNet-34 A', 25.03, 7.76], ['ResNet-34 B', 24.52, 7.46], ['ResNet-34 C', 24.19, 7.40], ['ResNet-50', 22.85, 6.71], ['ResNet-101', 21.75, 6.05], ['ResNet-152', 21.43, 5.71]] },
    t4: { title: 'Table 4 · best single-model results, validation († test)', rows: [['VGG (ILSVRC’14) †', null, 8.43], ['GoogLeNet (ILSVRC’14)', null, 7.89], ['VGG (v5)', 24.4, 7.1], ['PReLU-net', 21.59, 5.71], ['BN-inception', 21.99, 5.81], ['ResNet-34 B', 21.84, 5.71], ['ResNet-34 C', 21.53, 5.60], ['ResNet-50', 20.74, 5.25], ['ResNet-101', 19.87, 4.60], ['ResNet-152', 19.38, 4.49]] },
    t5: { title: 'Table 5 · ensembles, top-5 error on the test set', rows: [['VGG (ILSVRC’14)', null, 7.32], ['GoogLeNet (ILSVRC’14)', null, 6.66], ['VGG (v5)', null, 6.8], ['PReLU-net', null, 4.94], ['BN-inception', null, 4.82], ['ResNet (ILSVRC’15)', null, 3.57]] }
  };
  W.resultsExplorer = function (host) {
    Book.head(host, 'Tables 3, 4 and 5: depth pays',
      'Pick a table and a metric. ResNets are blue; earlier networks grey.');
    const st = { t: 't3', m: 2 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'table', [{ value: 't3', label: 'Table 3 (10-crop)' }, { value: 't4', label: 'Table 4 (best single models)' }, { value: 't5', label: 'Table 5 (ensembles)' }], st.t, function (v) { st.t = v; draw(); });
    Book.select(ctl, 'metric', [{ value: 2, label: 'top-5 error' }, { value: 1, label: 'top-1 error' }], st.m, function (v) { st.m = +v; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    function draw() {
      const t = TAB[st.t];
      U.hbars(plot, t.rows.map(function (r) { return { label: r[0], value: r[st.m], color: /ResNet/.test(r[0]) ? U.C.res : U.C.grey, note: r[st.m] == null ? '' : r[st.m] + '%', bold: /152|ILSVRC’15/.test(r[0]) }; }), { min: 0, max: st.m === 1 ? 30 : 11, ticks: st.m === 1 ? [0, 10, 20, 30] : [0, 2, 4, 6, 8, 10], labelW: 170 });
      out.innerHTML = t.title + (st.m === 1 && st.t === 't5' ? ' (no top-1 numbers in this table).' : '.') + ' ResNet-152’s single-model top-5 error, 4.49%, is lower than every earlier <em>ensemble</em> in Table 5. Six models of different depths (two of them 152-layer) gave 3.57% on the test set: 1st place in ILSVRC 2015.';
    }
    draw();
  };

  // ---------------------------------------------------------------- CIFAR family and Table 6
  T.cifarParams = function (n) {           // 6n+2 net, filters {16, 32, 64}, option A, BN (γ, β per channel), fc 10 with bias
    let p = 9 * 3 * 16 + 2 * 16, cin = 16;
    [16, 32, 64].forEach(function (c) { for (let i = 0; i < 2 * n; i++) { p += 9 * cin * c + 2 * c; cin = c; } });
    return p + 64 * 10 + 10;
  };
  const T6 = { 3: [0.27, 8.75], 5: [0.46, 7.51], 7: [0.66, 7.17], 9: [0.85, 6.97], 18: [1.7, 6.43], 200: [19.4, 7.93] };
  W.cifarFamily = function (host) {
    Book.head(host, 'The CIFAR-10 family: 6n + 2 layers',
      'A 3×3 conv, then 2n layers on 32×32 maps (16 filters), 2n on 16×16 (32), 2n on 8×8 (64), global average pooling and a 10-way layer. Pick n: the depth and parameter count are computed; for the paper’s choices, Table 6’s count and error are shown next to them.');
    let n = 3;
    const ctl = el('div', { class: 'controls' }, host);
    const sl = Book.slider(ctl, 'n', 1, 200, 1, n, function (v) { n = v; draw(); });
    [3, 5, 7, 9, 18, 200].forEach(function (v) { Book.button(ctl, 'n = ' + v, function () { sl.value = v; sl.dispatchEvent(new Event('input')); }); });
    const plot = el('div', null, host), out = U.out(host);
    function draw() {
      const p = T.cifarParams(n), t = T6[n];
      U.hbars(plot, [['ResNet-20', 8.75], ['ResNet-32', 7.51], ['ResNet-44', 7.17], ['ResNet-56', 6.97], ['ResNet-110', 6.43], ['ResNet-1202', 7.93], ['Highway-19', 7.54], ['FitNet-19', 8.39], ['DSN', 8.22], ['NIN', 8.81], ['Maxout', 9.38]].map(function (r) { return { label: r[0], value: r[1], color: /ResNet/.test(r[0]) ? (r[0] === 'ResNet-' + (6 * n + 2) ? 'var(--accent)' : U.C.res) : U.C.grey, note: r[1] + '%', bold: r[0] === 'ResNet-' + (6 * n + 2) }; }), { min: 5, max: 10, ticks: [5, 6, 7, 8, 9, 10], labelW: 110 });
      out.innerHTML = 'n = ' + n + ': <b>' + (6 * n + 2) + '</b> weighted layers, <b>' + 3 * n + '</b> shortcuts, <b>' + T.comma(p) + '</b> parameters (' + (p / 1e6).toFixed(2) + ' M)' + (t ? ' · Table 6: ' + t[0] + ' M, test error <b>' + t[1] + '%</b>' + (n === 18 ? ' (best of 5 runs; mean 6.61 ± 0.16)' : '') : '') + '. Bars: Table 6 test errors, all with data augmentation.';
    }
    draw();
  };

  // ---------------------------------------------------------------- Fig. 6 end points (≈) for plain and residual
  W.fig6 = function (host) {
    Book.head(host, 'Figure 6 at a glance: plain nets get worse with depth, ResNets better',
      'Final errors on CIFAR-10 for each depth. ResNet test errors are Table 6’s exact numbers; all plain-net numbers and the training errors are read by eye off Fig. 6 (≈). Plain-110 is above 60% and not drawn in the paper.');
    let what = 'test';
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'error', [{ value: 'test', label: 'test' }, { value: 'train', label: 'training ≈' }], what, function (v) { what = v; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    const D = { plain: { test: [[20, 9.7], [32, 10.2], [44, 11.2], [56, 13]], train: [[20, 1.2], [32, 0.8], [44, 3.2], [56, 5.8]] }, res: { test: [[20, 8.75], [32, 7.51], [44, 7.17], [56, 6.97], [110, 6.43]], train: [[20, 0.8], [32, 0.3], [44, 0.2], [56, 0.1], [110, 0.1]] } };
    function draw() {
      U.lines(plot, [{ name: 'plain ≈', color: U.C.plain, points: D.plain[what] }, { name: 'ResNet', color: U.C.res, points: D.res[what] }],
        { xmin: 20, xmax: 110, ymin: 0, ymax: what === 'test' ? 14 : 7, xticks: [20, 32, 44, 56, 110], xlabel: 'depth (layers)', ylabel: what + ' error (%)', height: 220 });
      out.innerHTML = what === 'test' ? 'Plain: test error rises from ≈ 9.7% (20 layers) to ≈ 13% (56). ResNet: falls from 8.75% to 6.97%, and to 6.43% at 110 layers.' : 'Plain: the deeper nets also have higher <em>training</em> error (≈ 6% for 56 layers), the signature of degradation; ResNets’ training error goes to nearly 0.';
    }
    draw();
  };

  // ---------------------------------------------------------------- 110 and 1202 layers
  W.paramsVsError = function (host) {
    Book.head(host, 'More layers, more parameters: where it stops paying',
      'Each dot is a ResNet from Table 6: parameters (log scale, computed by this book, and matching Table 6) against test error. The 1202-layer net fits its training data (training error below 0.1%) but tests worse than the 110-layer one.');
    const plot = el('div', null, host), out = U.out(host);
    const ns = [3, 5, 7, 9, 18, 200];
    const pts = ns.map(function (n) { return [Math.log10(T.cifarParams(n)), T6[n][1]]; });
    U.lines(plot, [{ color: U.C.res, points: pts, labels: ns.map(function (n) { return (6 * n + 2) + ' layers'; }), dash: '3 3' }], { xmin: 5.3, xmax: 7.4, ymin: 6, ymax: 9.2, xticks: [5.5, 6, 6.5, 7], xfmt: function (v) { return T.big(Math.pow(10, v)); }, xlabel: 'parameters', ylabel: 'test error (%)', height: 230, noLegend: true });
    out.innerHTML = 'computed parameters: ' + ns.map(function (n) { return (6 * n + 2) + ': ' + (T.cifarParams(n) / 1e6).toFixed(2) + ' M'; }).join(' · ') + '. The paper argues the 1202-layer net (19.4 M parameters for 50k training images) overfits; it uses no dropout or maxout, and leaves stronger regularization for future work.';
  };

  // ---------------------------------------------------------------- Fig. 7: layer responses
  W.responses = function (host) {
    Book.head(host, 'Figure 7 on the toy: how strongly does each layer respond?',
      'For every 3×3 layer of the trained toy CNNs: the standard deviation of its output after BN, before ReLU or addition (the paper’s Fig. 7 measurement), over 500 training images. Sort to compare the distributions.');
    let sorted = false;
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'order', [{ value: 0, label: 'original layer order' }, { value: 1, label: 'sorted by size' }], 0, function (v) { sorted = v === '1'; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    const col = { 'plain-20': U.C.plain2, 'plain-32': U.C.plain, 'res-20': U.C.res2, 'res-32': U.C.res };
    function draw() {
      const R = results(), ids = Object.keys(col).filter(function (k) { return R[k]; });
      if (!ids.length) { out.textContent = 'stored toy results are missing.'; return; }
      U.lines(plot, ids.map(function (id) { let s = R[id].std.slice(); if (sorted) s.sort(function (a, b) { return b - a; }); return { name: id.replace('res', 'ResNet'), color: col[id], dash: id.indexOf('plain') === 0 ? '5 3' : '', r: 2.2, points: s.map(function (v, i) { return [i + 1, v]; }) }; }),
        { xmin: 1, xmax: 32, ymin: 0, xticks: [1, 8, 16, 24, 32], xlabel: 'layer index' + (sorted ? ' (sorted by magnitude)' : ''), ylabel: 'std of response', height: 230 });
      const mean = function (id) { return ML.mean(R[id].std); };
      out.innerHTML = 'mean response std (toy): ' + ids.map(function (id) { return id + ' <b>' + mean(id).toFixed(2) + '</b>'; }).join(' · ') + '. In the paper, ResNets respond more weakly than plain nets, and deeper ResNets more weakly still: each residual function makes a smaller change.';
    }
    draw();
  };

  // ---------------------------------------------------------------- detection: IoU and mAP
  W.iouDemo = function (host) {
    Book.head(host, 'Is a predicted box “correct”? Intersection over union',
      'A detector outputs boxes. A predicted box (orange) counts as a hit if its overlap with the true box (green), measured as intersection ÷ union, passes a threshold. COCO’s main metric averages over thresholds 0.50, 0.55, …, 0.95.');
    const st = { x: 70, y: 50, w: 140, h: 110 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'x', 0, 200, 2, st.x, function (v) { st.x = v; draw(); });
    Book.slider(ctl, 'y', 0, 120, 2, st.y, function (v) { st.y = v; draw(); });
    Book.slider(ctl, 'width', 30, 240, 2, st.w, function (v) { st.w = v; draw(); });
    Book.slider(ctl, 'height', 30, 180, 2, st.h, function (v) { st.h = v; draw(); });
    const view = el('div', null, host), out = U.out(host);
    const gt = { x: 90, y: 60, w: 150, h: 120 };
    function draw() {
      view.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 400 260', style: 'max-width:420px' }, view);
      S('rect', { x: 0, y: 0, width: 400, height: 260, fill: 'var(--code-bg)' }, svg);
      const ix = Math.max(0, Math.min(st.x + st.w, gt.x + gt.w) - Math.max(st.x, gt.x)), iy = Math.max(0, Math.min(st.y + st.h, gt.y + gt.h) - Math.max(st.y, gt.y)), I = ix * iy, Un = st.w * st.h + gt.w * gt.h - I, iou = I / Un;
      if (I > 0) S('rect', { x: Math.max(st.x, gt.x), y: Math.max(st.y, gt.y), width: ix, height: iy, fill: U.rgba('#d69e2e', 0.35) }, svg);
      S('rect', { x: gt.x, y: gt.y, width: gt.w, height: gt.h, fill: 'none', stroke: U.C.green, 'stroke-width': 3 }, svg);
      S('rect', { x: st.x, y: st.y, width: st.w, height: st.h, fill: 'none', stroke: '#dd6b20', 'stroke-width': 3, 'stroke-dasharray': '6 3' }, svg);
      const th = ML.range(10).map(function (i) { return 0.5 + 0.05 * i; }), passed = th.filter(function (t) { return iou >= t; }).length;
      out.innerHTML = 'IoU = ' + T.comma(I) + ' / ' + T.comma(Un) + ' = <b>' + iou.toFixed(3) + '</b> · hit at threshold 0.5: <b>' + (iou >= 0.5 ? 'yes' : 'no') + '</b> · thresholds passed among 0.50…0.95: <b>' + passed + ' / 10</b>. For a single box, that fraction is what “mAP@[.5, .95]” averages; a full mAP also ranks many detections per class and averages precision over recall levels and classes.';
    }
    draw();
  };

  const DET = {
    t7: { t: 'Table 7 · PASCAL VOC mAP (%), Faster R-CNN baseline', rows: [['VGG-16, VOC 2007 test (07+12)', 73.2, 0], ['ResNet-101, VOC 2007 test (07+12)', 76.4, 1], ['VGG-16, VOC 2012 test (07++12)', 70.4, 0], ['ResNet-101, VOC 2012 test (07++12)', 73.8, 1]], max: 100 },
    t8: { t: 'Table 8 · COCO validation mAP (%), Faster R-CNN baseline', rows: [['VGG-16, mAP@.5', 41.5, 0], ['ResNet-101, mAP@.5', 48.4, 1], ['VGG-16, mAP@[.5,.95]', 21.2, 0], ['ResNet-101, mAP@[.5,.95]', 27.2, 1]], max: 60 },
    t9: { t: 'Appendix Table 9 · COCO, ResNet-101 improvements, mAP@[.5,.95] (%)', rows: [['baseline (val)', 27.2, 1], ['+ box refinement (val)', 29.9, 1], ['+ context (val)', 30.0, 1], ['+ multi-scale testing (val)', 32.5, 1], ['+ context (test-dev)', 32.2, 1], ['+ multi-scale testing (test-dev)', 34.9, 1], ['ensemble (test-dev)', 37.4, 1]], max: 40 },
    t10: { t: 'Appendix Tables 10–12 · mAP (%) of the improved system', rows: [['VOC 2007 test, VGG-16 baseline', 73.2, 0], ['VOC 2007 test, ResNet-101 “baseline+++”', 85.6, 1], ['VOC 2012 test, VGG-16 baseline', 70.4, 0], ['VOC 2012 test, ResNet-101 “baseline+++”', 83.8, 1], ['ImageNet DET test, GoogLeNet (ILSVRC’14)', 43.9, 0], ['ImageNet DET test, single ResNet-101', 58.8, 1], ['ImageNet DET test, ensemble', 62.1, 1]], max: 100 },
    t14: { t: 'Appendix Tables 13–14 · top-5 localization error (%)', rows: [['OverFeat (ILSVRC’13), test', 29.9, 0], ['GoogLeNet (ILSVRC’14), test', 26.7, 0], ['VGG (ILSVRC’14), test', 25.3, 0], ['ResNet-101 RPN, dense (val)', 14.4, 1], ['ResNet-101 RPN + R-CNN (val)', 10.6, 1], ['ResNet ensemble (val)', 8.9, 1], ['ResNet ensemble (ILSVRC’15), test', 9.0, 1]], max: 35 }
  };
  W.detectResults = function (host) {
    Book.head(host, 'Detection and localization results',
      'Pick a table. The main text (Tables 7–8) swaps only the backbone (VGG-16 → ResNet-101) inside the same Faster R-CNN. The appendix (arXiv version only) adds improvements, ensembles and the ILSVRC/COCO 2015 entries.');
    let t = 't8';
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'table', [{ value: 't7', label: 'Table 7 (VOC)' }, { value: 't8', label: 'Table 8 (COCO)' }, { value: 't9', label: 'Table 9 (COCO improvements)' }, { value: 't10', label: 'Tables 10–12 (VOC, ImageNet DET)' }, { value: 't14', label: 'Tables 13–14 (localization)' }], t, function (v) { t = v; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    function draw() {
      const d = DET[t];
      U.hbars(plot, d.rows.map(function (r) { return { label: r[0], value: r[1], color: r[2] ? U.C.res : U.C.grey, note: r[1] }; }), { min: 0, max: d.max, labelW: 270, ticks: [0, d.max / 2, d.max] });
      out.innerHTML = d.t + '. ' + (t === 't8' ? 'COCO mAP@[.5,.95] rises from 21.2 to 27.2: +6.0 points, a 28% relative gain (6.0 / 21.2), “solely due to the learned representations”.' : t === 't14' ? 'Lower is better here: error, not precision.' : 'Higher is better.');
    }
    draw();
  };

  // ---------------------------------------------------------------- limits: residual connections in later models
  W.residualEverywhere = function (host) {
    Book.head(host, 'Residual connections in later models',
      'Many later architectures in this collection wrap every sub-layer in a shortcut: output = x + sublayer(x). Pick one to count its residual additions, computed from its published layer counts.');
    const M = [
      { n: 'ResNet-152 (this paper)', blocks: 50, per: 1, f: 'y = σ(F(x) + W_s x)', link: '#ch-table1' },
      { n: 'Transformer base, 2017 (lesson 8)', blocks: 6, per: 5, f: 'x + Sublayer(x), then LayerNorm: 6 encoder layers × 2 + 6 decoder layers × 3', link: '../transformer/' },
      { n: 'BERT-base, 2018', blocks: 12, per: 2, f: 'x + Sublayer(x): 12 layers × (attention + feed-forward)', link: '../bert/' },
      { n: 'GPT-3 175B, 2020', blocks: 96, per: 2, f: 'x + Sublayer(LayerNorm(x)): 96 layers × 2', link: '../gpt3/' },
      { n: 'ViT-Base, 2020', blocks: 12, per: 2, f: 'x + MSA(LN(x)), x + MLP(LN(x)): 12 layers × 2', link: '../vit/' }
    ];
    let i = 1;
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'model', M.map(function (m, k) { return { value: k, label: m.n }; }), i, function (v) { i = +v; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    function count(m) { return m.per === 5 ? 6 * 2 + 6 * 3 : m.blocks * m.per; }
    function draw() {
      U.hbars(plot, M.map(function (m, k) { return { label: m.n, value: count(m), color: k === i ? 'var(--accent)' : U.C.grey, note: count(m), bold: k === i }; }), { min: 0, max: 200, ticks: [0, 50, 100, 150, 200], labelW: 220 });
      out.innerHTML = M[i].n + ': <b>' + count(M[i]) + '</b> residual additions · ' + M[i].f + '. <a href="' + M[i].link + '">Read more</a>.';
    }
    draw();
  };
})();
