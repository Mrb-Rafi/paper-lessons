/* Dropout Interactive Web Book — Part II widgets: the dropout model (paper §1–§5).
   Uses Book, ML, NN, Toy and the helpers in window.DOUI (widgets-found.js). */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, Toy = window.Toy, U = window.DOUI;

  // ---------- ch-idea: Figure 1, sample thinned networks
  W.thinnedNet = function (host) {
    Book.head(host, 'Figure 1, live: one network, many thinned networks',
      'A standard network with two hidden layers (after Paper Fig. 1a). Press “Drop units” to sample a thinned network (Fig. 1b): each unit is kept with its own probability p. Change the network size to see how quickly the number of possible thinned networks grows.');
    const st = { n: 5, pIn: 0.8, pH: 0.5, rng: ML.rng(11), seen: new Set(), k: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'units per hidden layer', 2, 8, 1, st.n, function (v) { st.n = v; st.seen = new Set(); st.k = 0; sample(); }, function (v) { return v; });
    Book.slider(ctl, 'p hidden', 0.1, 1, 0.05, st.pH, function (v) { st.pH = v; sample(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'p input', 0.1, 1, 0.05, st.pIn, function (v) { st.pIn = v; sample(); }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'Drop units', function () { sample(); }, 'primary');
    Book.button(ctl, 'Sample 1,000', function () { for (let i = 0; i < 999; i++) sample(true); sample(); });
    const view = el('div', null, host), out = el('div', { class: 'out' }, host);
    let drop;
    function layers() { return [4, st.n, st.n, 2]; }
    function sample(quiet) {
      drop = U.sampleDrop(layers(), st.pIn, st.pH, st.rng); st.k++;
      st.seen.add(drop.slice(0, 3).map(function (s) { return Array.from(s).sort().join('.'); }).join('|'));
      if (!quiet) draw();
    }
    function draw() {
      const L = layers(); U.net(view, L, { dropped: drop, labels: ['input', 'hidden 1', 'hidden 2', 'output'] });
      const n = 4 + 2 * st.n, kept = L.slice(0, 3).reduce(function (a, c, l) { return a + c - drop[l].size; }, 0);
      const params = 4 * st.n + st.n * st.n + st.n * 2 + 2 * st.n + 2;
      out.innerHTML = 'droppable units n = ' + n + ' → possible thinned networks 2<sup>' + n + '</sup> = <b>' + Math.pow(2, n).toLocaleString('en-US') + '</b>' +
        '\nthis one keeps ' + kept + ' of ' + n + ' units · expected kept = ' + U.f(4 * st.pIn + 2 * st.n * st.pH, 1) +
        '\nsamples drawn: ' + st.k + ' · different thinned networks seen so far: ' + st.seen.size +
        '\nweights and biases shared by all of them: ' + params + '  (O(n²), not 2ⁿ copies)';
    }
    sample();
  };

  // ---------- ch-idea: standard vs dropout training on the toy digits
  W.dropVsPlain = function (host) {
    Book.head(host, 'The same network, trained with and without dropout',
      'Two 49-96-96-10 networks learn the 500 toy training digits with identical settings (minibatch SGD with momentum, 3,000 steps). One uses dropout with p = 0.8 for the inputs and p = 0.5 for the hidden units. Both are trained in your browser.');
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw(jobs) {
      if (!jobs) return;
      const sers = [];
      [[jobs[0], 'standard', Book.palette[1]], [jobs[1], 'dropout', Book.palette[0]]].forEach(function (j) {
        const c = j[0].curve; if (!c.length) return;
        sers.push({ name: j[1] + ' test', color: j[2], marker: 'none', ldy: j[1] === 'standard' ? -6 : 14, points: c.map(function (r) { return [r[0], r[2]]; }) });
        sers.push({ color: j[2], marker: 'none', dash: '4 3', width: 1.5, points: c.map(function (r) { return [r[0], r[1]]; }) });
      });
      if (!sers.length) return;
      Book.chart.lines(plot, sers, { width: 640, height: 250, xmin: 0, xmax: 3000, ymin: 0, ymax: 30, xticks: [0, 1000, 2000, 3000], xlabel: 'training steps', ylabel: 'error % (toy)', mr: 110 });
    }
    U.withPair(host, function (P) {
      draw([P.std, P.drop]);
      const a = Toy.summary(P.std), b = Toy.summary(P.drop);
      out.innerHTML = 'final test error (mean of the last 3 checks): standard <b>' + U.pct(a.test) + '</b> · dropout <b>' + U.pct(b.test) + '</b>' +
        '\nfinal training error: standard ' + U.pct(a.train) + ' · dropout ' + U.pct(b.train) + '   (solid: test, dashed: training)' +
        '\n<span class="note">Toy numbers. The standard net fits its training set perfectly and its test error creeps up; the dropout net never fits the training set perfectly yet does better on new digits. The paper’s gaps are of a similar relative size on MNIST (1.60 % → 1.35 %, Table 2).</span>';
    }, draw);
  };

  // ---------- ch-motivation: conspiracies
  W.conspiracy = function (host) {
    Book.head(host, 'Ten conspiracies of five, or one of fifty?',
      'The paper’s analogy (§2). A plan succeeds only if every member plays their part. During rehearsal (training) everybody is reliable; on the day (test) conditions change and each member succeeds with probability q. Compare one big plan with ten small independent ones.');
    const st = { q: 0.95 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'chance each member still succeeds, q', 0.5, 1, 0.01, st.q, function (v) { st.q = v; draw(); }, function (v) { return v.toFixed(2); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const big = Math.pow(st.q, 50), small = Math.pow(st.q, 5), any = 1 - Math.pow(1 - small, 10), exp = 10 * small;
      Book.chart.bars(chart, [{ label: 'one plan of 50\nsucceeds', value: big, note: U.pct(100 * big), color: Book.palette[1] },
        { label: 'a plan of 5\nsucceeds', value: small, note: U.pct(100 * small), color: Book.palette[0] },
        { label: 'at least one of\nten plans of 5', value: any, note: U.pct(100 * any), color: Book.palette[2] }], { max: 1, ylabel: 'probability', mb: 50, tickFmt: function (v) { return v.toFixed(2); } });
      out.innerHTML = 'P(plan of 50 works) = q⁵⁰ = <b>' + U.f(big, 4) + '</b>\nP(plan of 5 works) = q⁵ = ' + U.f(small, 4) + ' → on average <b>' + U.f(exp, 2) + '</b> of the ten small plans work\nP(at least one small plan works) = 1 − (1 − q⁵)¹⁰ = ' + U.f(any, 4) +
        '\n<span class="note">A big co-adapted group is fragile when anything changes. Dropout forces each hidden unit to be useful with a random subset of partners, so the network behaves like many small groups.</span>';
    }
    draw();
  };

  // ---------- ch-motivation: knock out hidden units at test time
  // forward pass of a Toy.MLP with some hidden units removed (zeroed) in every hidden layer
  U.knockError = function (net, data, frac, rng) {
    const X = NN.tensor(data.X);
    let h = X;
    for (let i = 0; i < net.layers.length; i++) {
      h = net.layers[i].forward(h);
      if (i + 1 < net.layers.length) {
        h = NN.relu(h);
        const n = h.cols, k = Math.round(frac * n), off = rng.shuffle(ML.range(n)).slice(0, k);
        off.forEach(function (j) { for (let r = 0; r < h.rows; r++) h.data[r * n + j] = 0; });
      }
    }
    let wrong = 0; for (let r = 0; r < h.rows; r++) if (ML.argmax(h.row(r)) !== data.y[r]) wrong++;
    return 100 * wrong / h.rows;
  };
  W.knockout = function (host) {
    Book.head(host, 'Remove hidden units after training',
      'Take the two trained toy networks and delete a random fraction of the units in every hidden layer (no retraining, no rescaling). A network whose units co-adapted, each relying on specific partners, breaks down quickly. Each point averages 4 random deletions on 400 test digits.');
    const box = el('div', null, host), out = el('div', { class: 'out' }, host);
    U.withPair(host, function (P) {
      const sub = { X: P.data.test.X.slice(0, 400), y: P.data.test.y.slice(0, 400) };
      const fr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], rng = ML.rng(31);
      const curve = function (net) { return fr.map(function (f) { let s = 0; for (let t = 0; t < 4; t++) s += U.knockError(net, sub, f, rng); return [f * 100, s / 4]; }); };
      const a = curve(P.std.net), b = curve(P.drop.net);
      Book.chart.lines(box, [{ name: 'standard', color: Book.palette[1], points: a }, { name: 'dropout', color: Book.palette[0], points: b }],
        { width: 640, height: 240, xmin: 0, xmax: 70, ymin: 0, xticks: [0, 10, 20, 30, 40, 50, 60, 70], xlabel: '% of hidden units removed at test time', ylabel: 'test error % (toy)', mr: 80 });
      out.innerHTML = 'with 30 % of hidden units removed: standard <b>' + U.pct(a[3][1]) + '</b> · dropout <b>' + U.pct(b[3][1]) + '</b>\nwith 50 % removed: standard <b>' + U.pct(a[5][1]) + '</b> · dropout <b>' + U.pct(b[5][1]) + '</b>' +
        '\n<span class="note">The dropout net was trained to work with half of its hidden units missing, so losing units hurts it far less. This is the book’s own test of the §2 argument, not an experiment from the paper.</span>';
    });
  };

  // ---------- ch-forward: Figure 3, one unit with dropout gates
  W.fig3Unit = function (host) {
    Book.head(host, 'Figure 3, live: one unit with its dropout gates',
      'Unit i in layer l+1 reads three outputs y₁, y₂, y₃ of layer l (after Paper Fig. 3b). Each input passes a gate rⱼ that is 1 (kept) or 0 (dropped). Click a gate to flip it, or press “Resample” to draw all gates from Bernoulli(p).');
    const st = { y: [0.9, 0.2, 0.7], w: [1.2, -0.8, 0.6], b: -0.3, r: [1, 0, 1], p: 0.5, std: false, rng: ML.rng(2) };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'p', 0.1, 1, 0.05, st.p, function (v) { st.p = v; }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'Resample r', function () { st.r = st.r.map(function () { return st.rng.next() < st.p ? 1 : 0; }); draw(); }, 'primary');
    Book.button(ctl, 'Standard network (Fig. 3a)', function () { st.std = !st.std; draw(); });
    const box = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      box.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 600 240', role: 'img', 'aria-label': 'one unit with dropout gates', style: 'min-width:520px' }, box);
      const ys = [50, 120, 190], zx = 420, zy = 120;
      const yt = st.y.map(function (v, j) { return st.std ? v : st.r[j] * v; });
      const z = ML.dot(st.w, yt) + st.b, f = ML.sigmoid(z);
      S('circle', { cx: 330, cy: 26, r: 14, fill: 'var(--bg-soft)', stroke: 'var(--ink-soft)' }, svg); S('text', { x: 330, y: 30, 'text-anchor': 'middle', 'font-size': 12, class: 'svg-text', text: '+1' }, svg);
      S('line', { x1: 340, y1: 36, x2: zx - 16, y2: zy - 12, stroke: 'var(--ink-soft)', 'stroke-width': 1.5 }, svg);
      S('text', { x: 382, y: 62, 'font-size': 12, class: 'svg-text', text: 'b = ' + st.b }, svg);
      ys.forEach(function (yy, j) {
        S('circle', { cx: 50, cy: yy, r: 18, fill: 'var(--blue-soft)', stroke: 'var(--ink-soft)' }, svg);
        S('text', { x: 50, y: yy + 4, 'text-anchor': 'middle', 'font-size': 12, class: 'svg-text', text: st.y[j].toFixed(1) }, svg);
        S('text', { x: 50, y: yy - 24, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: 'y' + (j + 1) }, svg);
        if (!st.std) {
          const g = S('g', { style: 'cursor:pointer' }, svg);
          S('rect', { x: 110, y: yy - 14, width: 64, height: 28, rx: 6, fill: st.r[j] ? 'var(--green-soft)' : 'var(--red-soft)', stroke: st.r[j] ? 'var(--green)' : 'var(--red)' }, g);
          S('text', { x: 142, y: yy + 5, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, class: 'svg-text', text: 'r' + (j + 1) + ' = ' + st.r[j] }, g);
          g.addEventListener('click', function () { st.r[j] = 1 - st.r[j]; draw(); });
          S('line', { x1: 68, y1: yy, x2: 110, y2: yy, stroke: 'var(--ink-soft)', 'stroke-width': 1.5 }, svg);
          S('line', { x1: 174, y1: yy, x2: 214, y2: yy, stroke: st.r[j] ? 'var(--ink-soft)' : 'var(--line)', 'stroke-width': 1.5, 'stroke-dasharray': st.r[j] ? '' : '3 3' }, svg);
        } else S('line', { x1: 68, y1: yy, x2: 214, y2: yy, stroke: 'var(--ink-soft)', 'stroke-width': 1.5 }, svg);
        S('circle', { cx: 232, cy: yy, r: 18, fill: yt[j] === 0 && !st.std ? 'var(--bg-soft)' : 'var(--accent-soft)', stroke: 'var(--ink-soft)' }, svg);
        S('text', { x: 232, y: yy + 4, 'text-anchor': 'middle', 'font-size': 12, class: 'svg-text', text: yt[j].toFixed(1) }, svg);
        S('text', { x: 232, y: yy - 24, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: st.std ? 'y' + (j + 1) : 'ỹ' + (j + 1) }, svg);
        S('line', { x1: 250, y1: yy, x2: zx - 18, y2: zy, stroke: yt[j] === 0 && !st.std ? 'var(--line)' : 'var(--ink-soft)', 'stroke-width': 1.5 }, svg);
        S('text', { x: 256, y: yy + (j === 0 ? 24 : j === 1 ? -8 : -14), 'font-size': 11, class: 'svg-text', text: 'w' + (j + 1) + ' = ' + st.w[j] }, svg);
      });
      S('circle', { cx: zx, cy: zy, r: 20, fill: 'var(--panel)', stroke: 'var(--ink-soft)', 'stroke-width': 1.6 }, svg);
      S('text', { x: zx, y: zy + 4, 'text-anchor': 'middle', 'font-size': 12, class: 'svg-text', text: z.toFixed(2) }, svg);
      S('text', { x: zx, y: zy + 36, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: 'zᵢ' }, svg);
      S('line', { x1: zx + 20, y1: zy, x2: 520, y2: zy, stroke: 'var(--ink-soft)', 'stroke-width': 1.5 }, svg);
      S('text', { x: 480, y: zy - 8, 'font-size': 12, class: 'svg-text', text: 'f' }, svg);
      S('circle', { cx: 545, cy: zy, r: 22, fill: 'var(--green-soft)', stroke: 'var(--ink-soft)', 'stroke-width': 1.6 }, svg);
      S('text', { x: 545, y: zy + 4, 'text-anchor': 'middle', 'font-size': 12, class: 'svg-text', text: f.toFixed(3) }, svg);
      S('text', { x: 545, y: zy + 38, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: 'yᵢ' }, svg);
      out.innerHTML = (st.std ? 'standard: ' : 'r = (' + st.r.join(', ') + ') · ỹ = r ∗ y = (' + yt.map(function (v) { return v.toFixed(1); }).join(', ') + ')\n') +
        'zᵢ = w·' + (st.std ? 'y' : 'ỹ') + ' + b = ' + st.w.map(function (w, j) { return w + '×' + yt[j].toFixed(1); }).join(' + ') + ' + (' + st.b + ') = <b>' + z.toFixed(3) + '</b>\nyᵢ = f(zᵢ) = 1/(1 + e^(−zᵢ)) = <b>' + f.toFixed(3) + '</b>';
    }
    draw();
  };

  // ---------- ch-forward: masks over a minibatch, using NN.dropout
  W.maskBatch = function (host) {
    Book.head(host, 'A fresh mask for every training case',
      'A minibatch of 8 training cases (rows) passes through a layer of 12 hidden units (columns). NN.dropout, the shared library’s dropout, draws an independent keep/drop decision for every entry; its .mask holds the factors it multiplied by. Press “New masks”.');
    const st = { p: 0.5, seed: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'p (keep)', 0.1, 1, 0.05, st.p, function (v) { st.p = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'New masks', function () { st.seed++; draw(); }, 'primary');
    const row = el('div', { style: 'display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start' }, host), out = el('div', { class: 'out' }, host);
    const r0 = ML.rng(4), Y = NN.tensor(ML.mat(8, 12, function () { return Math.max(0, r0.normal(0.6, 0.6)); }));
    function heat(t, data, mx, labelFn, dropped) {
      const d = el('div', { style: 'text-align:center;font-size:12px;color:var(--muted)' }, row);
      const svg = S('svg', { viewBox: '0 0 ' + (12 * 18) + ' ' + (8 * 18), width: 216, height: 144, role: 'img', 'aria-label': t }, d);
      for (let i = 0; i < 8; i++) for (let j = 0; j < 12; j++) {
        const v = data[i * 12 + j];
        S('rect', { x: j * 18 + 1, y: i * 18 + 1, width: 16, height: 16, rx: 2, fill: dropped && dropped[i * 12 + j] === 0 ? 'var(--red-soft)' : v === 0 ? 'var(--bg-soft)' : 'var(--accent)', opacity: v === 0 ? 1 : (0.15 + 0.85 * Math.min(1, v / mx)).toFixed(2), stroke: dropped && dropped[i * 12 + j] === 0 ? 'var(--red)' : 'none', 'stroke-width': 0.8 }, svg);
        if (labelFn) S('text', { x: j * 18 + 9, y: i * 18 + 13, 'text-anchor': 'middle', 'font-size': 8.5, class: 'svg-text', text: labelFn(v) }, svg);
      }
      el('div', { text: t }, d);
    }
    function draw() {
      row.innerHTML = '';
      const out1 = NN.dropout(Y, 1 - st.p, ML.rng(st.seed), true), mask = out1.mask || new Float64Array(96).fill(1);
      heat('outputs y of the layer', Y.data, 2);
      heat('NN.dropout mask (0 or 1/p)', mask, 1 / st.p, function (v) { return v === 0 ? '0' : (1 / st.p).toFixed(1); }, mask);
      heat('thinned outputs passed on', out1.data, 2 / st.p, null, mask);
      const kept = ML.range(8).map(function (i) { let k = 0; for (let j = 0; j < 12; j++) if (mask[i * 12 + j] !== 0) k++; return k; });
      out.innerHTML = 'units kept per training case: ' + kept.join(', ') + '   (expected 12 × p = ' + U.f(12 * st.p, 1) + ')' +
        '\n<span class="note">Red = dropped by the mask; pale grey = an output that was already 0 (a ReLU unit that was off). NN.dropout multiplies kept units by 1/p = ' + U.f(1 / st.p, 2) + ' during training, the “inverted” variant of §10, so nothing needs rescaling at test time. The paper’s own description keeps the 1 and multiplies the weights by p at test time instead; Chapter 11 shows the two are the same.</span>';
    }
    draw();
  };

  // ---------- ch-testtime: the expected input vs the test-time input
  W.expectation = function (host) {
    Book.head(host, 'Why the weights are multiplied by p',
      'One unit with six inputs. During training each input is kept with probability p, so the unit’s input z varies from case to case (histogram). At test time all inputs are present and the weights are multiplied by p: the red line. It sits exactly at the average of z. For the output f(z) the match is only approximate.');
    const r0 = ML.rng(12), y = ML.range(6).map(function () { return +r0.uniform(0, 1.5).toFixed(2); }), w = ML.range(6).map(function () { return +r0.uniform(-1, 1.6).toFixed(2); }), b = -0.5;
    const st = { p: 0.5, rng: ML.rng(3) };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'p', 0.1, 1, 0.05, st.p, function (v) { st.p = v; draw(); }, function (v) { return v.toFixed(2); });
    const box = el('div', { class: 'grid2' }, host), pz = el('div', null, box), pf = el('div', null, box), out = el('div', { class: 'out' }, host);
    function hist(host2, vals, lo, hi, lines, xl) {
      const nb = 24, cnt = new Array(nb).fill(0); vals.forEach(function (v) { const k = Math.floor((v - lo) / (hi - lo) * nb); if (k >= 0 && k < nb) cnt[k]++; });
      const r = Book.chart.lines(host2, [{ color: Book.palette[0], marker: 'none', points: cnt.map(function (c, i) { return [lo + (i + 0.5) * (hi - lo) / nb, c / vals.length]; }) }], { width: 320, height: 200, xmin: lo, xmax: hi, ymin: 0, xlabel: xl, ylabel: 'fraction', xticks: [lo, (lo + hi) / 2, hi], ydec: 2 });
      lines.forEach(function (l) { const x = r.X(l[0]); S('line', { x1: x, x2: x, y1: 16, y2: 158, stroke: l[1], 'stroke-width': 2, 'stroke-dasharray': l[2] || '' }, r.svg); });
    }
    function draw() {
      const zs = [], fs = [];
      for (let t = 0; t < 4000; t++) { let z = b; for (let j = 0; j < 6; j++) if (st.rng.next() < st.p) z += w[j] * y[j]; zs.push(z); fs.push(ML.sigmoid(z)); }
      const ztest = st.p * ML.dot(w, y) + b, Ez = ML.mean(zs), Ef = ML.mean(fs);
      hist(pz, zs, -3, 4, [[ztest, 'var(--red)'], [Ez, 'var(--ink)', '3 3']], 'z during training');
      hist(pf, fs, 0, 1, [[ML.sigmoid(ztest), 'var(--red)'], [Ef, 'var(--ink)', '3 3']], 'f(z) during training');
      out.innerHTML = 'inputs y = (' + y.join(', ') + '), weights w = (' + w.join(', ') + '), b = ' + b +
        '\naverage z over 4,000 masks = <b>' + U.f(Ez, 3) + '</b> · test-time z = (p w)·y + b = <b>' + U.f(ztest, 3) + '</b>   (equal up to sampling noise)' +
        '\naverage f(z) = <b>' + U.f(Ef, 3) + '</b> · test-time f(z) = <b>' + U.f(ML.sigmoid(ztest), 3) + '</b>   (close, not equal: f is curved)' +
        '\n<span class="note">Red: test time. Dotted black: the average over training-time masks.</span>';
    }
    draw();
  };

  // ---------- ch-testtime: paper convention vs inverted dropout, and forgetting to scale
  W.invertedEquiv = function (host) {
    Book.head(host, 'Two bookkeeping conventions, one network',
      'The paper keeps units at 1 while training and multiplies weights by p at test time. Most code (including this book’s toy, via NN.dropout) divides kept units by p while training and leaves test time alone (§10). Below, the trained toy dropout net is evaluated both ways, and once with the scaling forgotten.');
    const out = el('div', { class: 'out' }, host), chart = el('div', null, host);
    U.withPair(host, function (P) {
      const net = P.drop.net, te = P.data.test, pw = net.paperWeights();
      const keeps = net.layers.map(function (L, i) { return i === 0 ? net.keepIn : net.keepHid; });
      function forward(x, which) {          // which: 'inverted' | 'paper' | 'forgot'
        let h = x;
        net.layers.forEach(function (L, i) {
          const Wm = which === 'inverted' ? pw[i].test : pw[i].train, k = keeps[i], b = L.b.data, z = ML.zeros(L.W.cols);
          for (let a = 0; a < h.length; a++) { const v = h[a]; if (!v) continue; for (let c = 0; c < z.length; c++) z[c] += v * Wm[a][c] * (which === 'paper' ? k : 1); }
          for (let c = 0; c < z.length; c++) z[c] += b[c];
          h = i + 1 < net.layers.length ? z.map(function (v) { return Math.max(0, v); }) : z;
        });
        return h;
      }
      const err = {}, maxd = { v: 0 };
      ['inverted', 'paper', 'forgot'].forEach(function (m) { let w = 0; te.X.forEach(function (x, i) { const o = forward(x, m); if (m === 'paper') { const o2 = forward(x, 'inverted'); o.forEach(function (v, c) { maxd.v = Math.max(maxd.v, Math.abs(v - o2[c])); }); } if (ML.argmax(o) !== te.y[i]) w++; }); err[m] = 100 * w / te.y.length; });
      U.hbars(chart, [{ label: 'inverted: ×1/p in training, nothing at test', value: err.inverted, note: U.pct(err.inverted), color: Book.palette[0] },
        { label: 'paper: nothing in training, weights × p at test', value: err.paper, note: U.pct(err.paper), color: Book.palette[2] },
        { label: 'paper convention, but forgot the × p', value: err.forgot, note: U.pct(err.forgot), color: Book.palette[1] }], { labelWidth: 320, fmt: function (v) { return v.toFixed(1); } });
      out.innerHTML = 'The paper-convention training weights are W/p (they never saw the 1/p factor). At test time they are multiplied by p, giving back W.\nlargest difference between the two conventions’ outputs on 1,000 test digits: <b>' + maxd.v.toExponential(1) + '</b> (rounding only)\ntest error (toy): see the bars. Forgetting the scaling makes every unit’s input about 1/p times too large.';
    });
  };

  // ---------- ch-averaging: enumerate every thinned network of a tiny net
  W.enumerate = function (host) {
    Book.head(host, 'Averaging all 4,096 thinned networks',
      'A tiny dropout network (2 inputs → 6 → 6 → 2, p = 0.5 in both hidden layers) learns the two-moons data. With 12 droppable units there are 2¹² = 4,096 thinned networks, few enough to run every one. For 300 test points we compare three predictions: the arithmetic mean of all 4,096, their normalised geometric mean, and the single weight-scaled network.');
    const out = el('div', { class: 'out', text: 'Training the tiny network…' }, host), box = el('div', { class: 'grid2' }, host), sc = el('div', null, box), tb = el('div', null, box);
    setTimeout(function () {
      const data = { train: U.moons(80, 41, 0.25), test: U.moons(300, 42, 0.25) };
      const T = Toy.train({ sizes: [2, 6, 6, 2], data: data, steps: 3000, batch: 16, lr: 0.05, m1: 0.9, keepHid: 0.5, seed: 9, evalEvery: 3000 });
      const L = T.net.layers.map(function (l) { return { W: l.W.toArray(), b: Array.from(l.b.data) }; });
      function lin(h, l) { const z = l.b.slice(); for (let a = 0; a < h.length; a++) for (let c = 0; c < z.length; c++) z[c] += h[a] * l.W[a][c]; return z; }
      const res = { ar: [], ge: [], ws: [] };
      data.test.X.forEach(function (x) {
        const h1 = lin(x, L[0]).map(function (v) { return Math.max(0, v); });
        const ws = ML.softmax(lin(lin(h1, L[1]).map(function (v) { return Math.max(0, v); }), L[2]));
        let ar = [0, 0], lg = [0, 0];
        for (let m = 0; m < 4096; m++) {
          const a = h1.map(function (v, j) { return (m >> j) & 1 ? v / 0.5 : 0; });
          const h2 = lin(a, L[1]).map(function (v, j) { return (m >> (6 + j)) & 1 ? Math.max(0, v) / 0.5 : 0; });
          const q = ML.softmax(lin(h2, L[2]));
          ar[0] += q[0] / 4096; ar[1] += q[1] / 4096; lg[0] += Math.log(Math.max(q[0], 1e-300)) / 4096; lg[1] += Math.log(Math.max(q[1], 1e-300)) / 4096;
        }
        const ge = ML.softmax(lg);
        res.ar.push(ar[1]); res.ge.push(ge[1]); res.ws.push(ws[1]);
      });
      const acc = function (v) { let ok = 0; v.forEach(function (p, i) { if ((p > 0.5 ? 1 : 0) === data.test.y[i]) ok++; }); return 100 - 100 * ok / v.length; };
      const mad = function (a, b) { let s = 0; a.forEach(function (v, i) { s += Math.abs(v - b[i]); }); return s / a.length; };
      Book.chart.lines(sc, [{ color: 'var(--muted)', marker: 'none', dash: '3 3', points: [[0, 0], [1, 1]] },
        { name: '', color: Book.palette[4], noline: true, r: 2.4, points: res.ws.map(function (v, i) { return [v, res.ge[i]]; }) }],
        { width: 320, height: 260, xmin: 0, xmax: 1, ymin: 0, ymax: 1, xticks: [0, 0.5, 1], xlabel: 'weight-scaled net: P(class 1)', ylabel: 'geometric mean of 4,096 nets', ydec: 1 });
      U.table(tb, [['prediction', 'test error (toy)'], ['arithmetic mean of 4,096 nets', U.pct(acc(res.ar))], ['normalised geometric mean', U.pct(acc(res.ge))], ['one weight-scaled net', U.pct(acc(res.ws))]]);
      out.innerHTML = 'average |difference| in P(class 1): weight scaling vs geometric mean <b>' + U.f(mad(res.ws, res.ge), 4) + '</b> · vs arithmetic mean <b>' + U.f(mad(res.ws, res.ar), 4) + '</b>' +
        '\n<span class="note">Each dot is a test point: near the diagonal means one cheap network reproduces the average of 4,096. Running all of them costs 4,096 forward passes; weight scaling costs one.</span>';
    }, 30);
  };

  // ---------- ch-averaging: exactness for one softmax layer
  W.oneLayerExact = function (host) {
    Book.head(host, 'For a single softmax layer the match is exact',
      'A softmax classifier with 4 inputs and 3 classes, with dropout (p = 0.5) on its inputs: 2⁴ = 16 thinned models. The table lists every one. The normalised geometric mean of their 16 predictions equals the prediction of the one model with weights × p, to the last digit. Drag the inputs to check.');
    const x = [1.0, 0.4, -0.6, 0.8], Wt = [[1.2, -0.4, 0.2], [-0.7, 0.9, 0.3], [0.5, 0.6, -1.1], [-0.2, -0.8, 1.0]], b = [0.1, 0, -0.1], p = 0.5;
    const ctl = el('div', { class: 'controls' }, host);
    x.forEach(function (v, i) { Book.slider(ctl, 'x' + (i + 1), -2, 2, 0.1, v, function (u) { x[i] = u; draw(); }, function (u) { return u.toFixed(1); }); });
    const tb = el('div', { style: 'max-height:300px;overflow:auto' }, host), out = el('div', { class: 'out' }, host);
    function logits(xx, scale) { return [0, 1, 2].map(function (c) { let s = b[c]; for (let i = 0; i < 4; i++) s += xx[i] * Wt[i][c] * scale; return s; }); }
    function draw() {
      tb.innerHTML = '';
      const rows = [['mask r', 'P(class 1, 2, 3) of this thinned model']]; const lg = [0, 0, 0], ar = [0, 0, 0];
      for (let m = 0; m < 16; m++) {
        const r = [0, 1, 2, 3].map(function (i) { return (m >> i) & 1; }), q = ML.softmax(logits(x.map(function (v, i) { return v * r[i]; }), 1));
        q.forEach(function (v, c) { lg[c] += Math.log(v) / 16; ar[c] += v / 16; });
        rows.push([r.join(''), q.map(function (v) { return v.toFixed(4); }).join(', ')]);
      }
      U.table(tb, rows);
      const ge = ML.softmax(lg), ws = ML.softmax(logits(x, p));
      out.innerHTML = 'normalised geometric mean of the 16: (<b>' + ge.map(function (v) { return v.toFixed(6); }).join(', ') + '</b>)\none model with weights × p:        (<b>' + ws.map(function (v) { return v.toFixed(6); }).join(', ') + '</b>)\narithmetic mean of the 16:          (' + ar.map(function (v) { return v.toFixed(6); }).join(', ') + ')';
    }
    draw();
  };

  // ---------- ch-backprop: gradients through a thinned network
  W.backpropMask = function (host) {
    Book.head(host, 'Dropped units get no gradient',
      'A 4-5-3 network, two training cases in a minibatch, dropout (p = 0.5) on the 5 hidden units with a different mask for each case. Gradients are computed by the shared autograd (NN). Each column of the heat maps belongs to one hidden unit: grey crosses mark units dropped for that case, and their gradients are exactly 0.');
    const st = { seed: 1, view: 'both' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.button(ctl, 'New masks', function () { st.seed++; draw(); }, 'primary');
    Book.select(ctl, 'show', [{ value: 'c1', label: 'case 1 only' }, { value: 'c2', label: 'case 2 only' }, { value: 'both', label: 'minibatch (average of both)' }], st.view, function (v) { st.view = v; draw(); });
    const row = el('div', { style: 'display:flex;flex-wrap:wrap;gap:18px' }, host), out = el('div', { class: 'out' }, host);
    const r0 = ML.rng(6), L1 = new NN.Linear(4, 5, r0), L2 = new NN.Linear(5, 3, r0);
    const X = [[1, 0.5, -0.3, 0.8], [-0.4, 1.2, 0.6, -0.9]], Y = [0, 2];
    function grads(rows, masks) {
      NN.zeroGrad([L1.W, L1.b, L2.W, L2.b]);
      const h = NN.relu(L1.forward(NN.tensor(rows.map(function (i) { return X[i]; }))));
      const m = new NN.Tensor(rows.length, 5); rows.forEach(function (i, k) { for (let j = 0; j < 5; j++) m.data[k * 5 + j] = masks[i][j] / 0.5; });
      const L = NN.softmaxCE(L2.forward(NN.mul(h, m)), rows.map(function (i) { return Y[i]; }));
      L.backward();
      return { W1: Array.from(L1.W.grad), W2: Array.from(L2.W.grad), loss: L.item() };
    }
    function heat(t, g, rowsN, colsN, byCol, dropped) {
      const d = el('div', { style: 'font-size:12px;color:var(--muted);text-align:center' }, row);
      const cs = 26, svg = S('svg', { viewBox: '0 0 ' + (colsN * cs) + ' ' + (rowsN * cs), width: colsN * cs, height: rowsN * cs, role: 'img', 'aria-label': t }, d);
      let mx = 1e-9; g.forEach(function (v) { mx = Math.max(mx, Math.abs(v)); });
      for (let i = 0; i < rowsN; i++) for (let j = 0; j < colsN; j++) {
        const v = g[i * colsN + j], unit = byCol ? j : i;
        S('rect', { x: j * cs + 1, y: i * cs + 1, width: cs - 2, height: cs - 2, rx: 3, fill: v === 0 ? 'var(--bg-soft)' : (v > 0 ? 'var(--blue)' : 'var(--red)'), opacity: v === 0 ? 1 : (0.2 + 0.8 * Math.abs(v) / mx).toFixed(2), stroke: dropped.has(unit) ? 'var(--muted)' : 'none' }, svg);
        if (dropped.has(unit)) S('text', { x: j * cs + cs / 2, y: i * cs + cs / 2 + 4, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)', text: '×' }, svg);
      }
      el('div', { html: t }, d);
    }
    function draw() {
      row.innerHTML = '';
      const r = ML.rng(st.seed * 3 + 1), masks = [0, 1].map(function () { return ML.range(5).map(function () { return r.next() < 0.5 ? 1 : 0; }); });
      const rows = st.view === 'c1' ? [0] : st.view === 'c2' ? [1] : [0, 1];
      const g = grads(rows, masks);
      const dropped = new Set(ML.range(5).filter(function (j) { return rows.every(function (i) { return !masks[i][j]; }); }));
      heat('∂L/∂W⁽¹⁾ (4 inputs × 5 hidden)', g.W1, 4, 5, true, dropped);
      heat('∂L/∂W⁽²⁾ (5 hidden × 3 outputs)', g.W2, 5, 3, false, dropped);
      out.innerHTML = 'masks: case 1 r = (' + masks[0].join(', ') + '), case 2 r = (' + masks[1].join(', ') + ')\nloss (' + (rows.length === 2 ? 'mean of both cases' : 'case ' + (rows[0] + 1)) + ') = ' + U.f(g.loss, 4) +
        '\nhidden units with all-zero gradient columns/rows: ' + (dropped.size ? Array.from(dropped).map(function (j) { return j + 1; }).join(', ') : 'none') +
        '\n<span class="note">In the minibatch view a unit’s weights get a gradient if at least one case kept it; the average then counts zero from the cases that dropped it (§5.1).</span>';
    }
    draw();
  };

  // ---------- ch-maxnorm: projection onto a ball
  W.maxNormBall = function (host) {
    Book.head(host, 'Max-norm: a fence around each unit’s weights',
      'The two incoming weights of one unit, w = (w₁, w₂), descend a noisy loss whose minimum (★) lies far out. With a big learning rate the steps are wild. With max-norm, whenever w leaves the circle of radius c it is pulled straight back to the circle (§5.1).');
    const st = { lr: 0.9, c: 1.5, on: true, noise: 1.0, seed: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'learning rate', 0.05, 1.9, 0.05, st.lr, function (v) { st.lr = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'c', 0.5, 3, 0.1, st.c, function (v) { st.c = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'gradient noise', 0, 2, 0.1, st.noise, function (v) { st.noise = v; draw(); }, function (v) { return v.toFixed(1); });
    const tg = Book.button(ctl, 'Max-norm: on', function () { st.on = !st.on; tg.textContent = 'Max-norm: ' + (st.on ? 'on' : 'off'); draw(); }, 'primary');
    Book.button(ctl, 'New noise', function () { st.seed++; draw(); });
    const box = el('div', { class: 'grid2' }, host), pic = el('div', null, box), out = el('div', { class: 'out' }, box);
    const star = [2.6, 1.6];
    function draw() {
      const r = ML.rng(st.seed), path = [[-0.3, -0.2]]; let w = path[0].slice(), proj = 0, maxN = 0;
      for (let t = 0; t < 60; t++) {
        const g = [w[0] - star[0] + st.noise * r.normal(0, 1), (w[1] - star[1]) * 1.5 + st.noise * r.normal(0, 1)];
        w = [w[0] - st.lr * g[0], w[1] - st.lr * g[1]];
        const n = ML.norm(w); maxN = Math.max(maxN, n);
        if (st.on && n > st.c) { w = ML.scale(w, st.c / n); proj++; }
        path.push(w.slice());
      }
      pic.innerHTML = '';
      const sc = 40, o = [150, 140], svg = S('svg', { viewBox: '0 0 300 280', role: 'img', 'aria-label': 'max-norm projection' }, pic);
      const P = function (v) { return [o[0] + Book.clamp(v[0], -3.6, 3.6) * sc, o[1] - Book.clamp(v[1], -3.3, 3.3) * sc]; };
      S('line', { x1: 0, x2: 300, y1: o[1], y2: o[1], stroke: 'var(--line)' }, svg); S('line', { x1: o[0], x2: o[0], y1: 0, y2: 280, stroke: 'var(--line)' }, svg);
      S('circle', { cx: o[0], cy: o[1], r: st.c * sc, fill: st.on ? 'var(--accent-soft)' : 'none', stroke: 'var(--accent)', 'stroke-width': 1.5, 'stroke-dasharray': st.on ? '' : '4 3' }, svg);
      S('polyline', { points: path.map(function (v) { return P(v).join(','); }).join(' '), fill: 'none', stroke: 'var(--red)', 'stroke-width': 1.3, opacity: 0.8 }, svg);
      const e = P(path[path.length - 1]); S('circle', { cx: e[0], cy: e[1], r: 5, fill: 'var(--red)' }, svg);
      const s0 = P(star); S('text', { x: s0[0] - 7, y: s0[1] + 6, 'font-size': 18, fill: 'var(--green)', text: '★' }, svg);
      out.innerHTML = '60 steps · final w = (' + U.f(w[0]) + ', ' + U.f(w[1]) + '), ‖w‖ = <b>' + U.f(ML.norm(w)) + '</b>\nlargest ‖w‖ reached before projection: ' + U.f(maxN) + (st.on ? '\nprojections applied: ' + proj + ' of 60 steps' : '') +
        '\n\n<span class="note">Projection: if ‖w‖ > c, w ← c · w / ‖w‖. It only rescales; the direction is kept. The paper used c = 2 (MNIST, App. B.1) and c = 4 (SVHN, TIMIT); App. A.3 says typical values are 3 to 4.</span>';
    }
    draw();
  };

  // ---------- ch-maxnorm: momentum
  W.momentumBall = function (host) {
    Book.head(host, 'Momentum: keep rolling in the same direction',
      'Gradient descent on a long narrow valley (minimum at the centre) for 40 steps. Plain gradient descent zig-zags across the steep direction and crawls along the shallow one. Momentum adds up past steps: v ← μ v + g, w ← w − ε v (the update used in deepnet’s parameter.py).');
    const st = { lr: 0.08, noise: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'learning rate ε', 0.01, 0.1, 0.005, st.lr, function (v) { st.lr = v; draw(); }, function (v) { return v.toFixed(3); });
    Book.slider(ctl, 'gradient noise', 0, 3, 0.25, st.noise, function (v) { st.noise = v; draw(); }, function (v) { return v.toFixed(2); });
    const box = el('div', { class: 'grid2' }, host), pic = el('div', null, box), out = el('div', { class: 'out' }, box);
    const a = 1, bb = 18, loss = function (w) { return 0.5 * (a * w[0] * w[0] + bb * w[1] * w[1]); };
    function run(mu) {
      const r = ML.rng(5); let w = [-4, 0.9], v = [0, 0]; const path = [w.slice()];
      for (let t = 0; t < 40; t++) {
        const g = [a * w[0] + st.noise * r.normal(0, 1), bb * w[1] + st.noise * r.normal(0, 1)];
        v = [mu * v[0] + g[0], mu * v[1] + g[1]]; w = [w[0] - st.lr * v[0], w[1] - st.lr * v[1]];
        w = [Book.clamp(w[0], -6, 6), Book.clamp(w[1], -3, 3)]; path.push(w.slice());
      }
      return path;
    }
    function draw() {
      pic.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 320 180', role: 'img', 'aria-label': 'momentum paths' }, pic), P = function (w) { return [160 + w[0] * 35, 90 - w[1] * 60]; };
      [0.5, 2, 5, 9].forEach(function (l) { S('ellipse', { cx: 160, cy: 90, rx: Math.sqrt(2 * l / a) * 35, ry: Math.sqrt(2 * l / bb) * 60, fill: 'none', stroke: 'var(--line)' }, svg); });
      const res = [];
      [[0, Book.palette[1], 'μ = 0'], [0.9, Book.palette[0], 'μ = 0.9'], [0.99, Book.palette[4], 'μ = 0.99']].forEach(function (m) {
        const p = run(m[0]); res.push([m[2], loss(p[p.length - 1])]);
        S('polyline', { points: p.map(function (w) { return P(w).join(','); }).join(' '), fill: 'none', stroke: m[1], 'stroke-width': 1.6, opacity: 0.85 }, svg);
        const e = P(p[p.length - 1]); S('circle', { cx: e[0], cy: e[1], r: 4, fill: m[1] }, svg);
      });
      out.innerHTML = 'loss after 40 steps:\n' + res.map(function (r) { return r[0] + ': <b>' + U.f(r[1], 4) + '</b>'; }).join('\n') +
        '\n\n<span class="note">Red μ = 0, blue μ = 0.9, purple μ = 0.99. Momentum averages noisy gradients over about 1/(1 − μ) steps. A very high μ overshoots and needs many steps to settle; the paper pairs it with a learning rate that decays.</span>';
    }
    draw();
  };

  // ---------- ch-maxnorm: deepnet's schedules
  W.schedules = function (host) {
    Book.head(host, 'Decaying learning rate, rising momentum',
      'The two schedules of deepnet (parameter.py, GetMomentumAndEpsilon), with the defaults of its dropout MNIST example (examples/ff/model_dropout.pbtxt): ε₀ = 1.0 halving-style decay ε₀/(1 + t/T) with T = 10,000; momentum rising from 0.5 towards 0.99 with time scale 100,000 steps, over the example’s 1,000,000 steps.');
    const st = { e0: 1, T: 10000, m0: 0.5, m1: 0.99, Tm: 100000 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'ε₀', 0.01, 2, 0.01, st.e0, function (v) { st.e0 = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'T (lr)', 1000, 100000, 1000, st.T, function (v) { st.T = v; draw(); }, function (v) { return v.toLocaleString('en-US'); });
    Book.slider(ctl, 'final momentum', 0.5, 0.999, 0.001, st.m1, function (v) { st.m1 = v; draw(); }, function (v) { return v.toFixed(3); });
    Book.slider(ctl, 'momentum time scale', 1000, 300000, 1000, st.Tm, function (v) { st.Tm = v; draw(); }, function (v) { return v.toLocaleString('en-US'); });
    const box = el('div', { class: 'grid2' }, host), a = el('div', null, box), b = el('div', null, box), out = el('div', { class: 'out' }, host);
    function draw() {
      const ts = ML.linspace(0, 1e6, 201);
      Book.chart.lines(a, [{ color: Book.palette[1], marker: 'none', points: ts.map(function (t) { return [t, Toy.lrAt(t, st.e0, st.T)]; }) }], { width: 320, height: 200, xmin: 0, xmax: 1e6, ymin: 0, xticks: [0, 5e5, 1e6], xfmt: function (v) { return v / 1e6 + 'M'; }, xlabel: 'step t', ylabel: 'learning rate ε', ydec: 2 });
      Book.chart.lines(b, [{ color: Book.palette[0], marker: 'none', points: ts.map(function (t) { return [t, Toy.momentumAt(t, st.m0, st.m1, st.Tm)]; }) }], { width: 320, height: 200, xmin: 0, xmax: 1e6, ymin: 0.4, ymax: 1, xticks: [0, 5e5, 1e6], xfmt: function (v) { return v / 1e6 + 'M'; }, xlabel: 'step t', ylabel: 'momentum μ', ydec: 2 });
      out.innerHTML = 'ε(t) = ε₀ / (1 + t/T): at t = 0: ' + U.f(st.e0, 3) + ', t = T: ' + U.f(st.e0 / 2, 3) + ', t = 1M: <b>' + U.f(Toy.lrAt(1e6, st.e0, st.T), 4) + '</b>\nμ(t) = μ_final − (μ_final − μ_initial)·e^(−t/τ): at t = 0: ' + U.f(st.m0, 3) + ', t = τ: ' + U.f(Toy.momentumAt(st.Tm, st.m0, st.m1, st.Tm), 3) + ', t = 1M: <b>' + U.f(Toy.momentumAt(1e6, st.m0, st.m1, st.Tm), 4) + '</b>' +
        '\n<span class="note">App. B.4 writes the same decay as ε₀(1 + t/T)⁻¹ and used ε₀ = 0.1 with momentum 0.95 on TIMIT.</span>';
    }
    draw();
  };

  // ---------- ch-pretrain: why the 1/p scale-up
  W.scaleUp = function (host) {
    Book.head(host, 'Scaling pretrained weights up by 1/p',
      'A pretrained unit computes z = w·y from 10 inputs without any dropout. Finetuning with dropout keeps each input with probability p, which would shrink z on average to p·(w·y). Multiplying the pretrained weights by 1/p puts the average back where pretraining left it (§5.2).');
    const r0 = ML.rng(21), y = ML.range(10).map(function () { return r0.uniform(0, 1); }), w = ML.range(10).map(function () { return r0.normal(0.2, 0.5); });
    const st = { p: 0.5, scale: true, rng: ML.rng(2) };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'p', 0.1, 1, 0.05, st.p, function (v) { st.p = v; draw(); }, function (v) { return v.toFixed(2); });
    const tg = Book.button(ctl, 'Scale ×1/p: on', function () { st.scale = !st.scale; tg.textContent = 'Scale ×1/p: ' + (st.scale ? 'on' : 'off'); draw(); }, 'primary');
    const out = el('div', { class: 'out' }, host);
    function draw() {
      const k = st.scale ? 1 / st.p : 1, pre = ML.dot(w, y); let s = 0, n = 20000;
      for (let t = 0; t < n; t++) for (let j = 0; j < 10; j++) if (st.rng.next() < st.p) s += k * w[j] * y[j];
      out.innerHTML = 'output during pretraining: w·y = <b>' + U.f(pre, 3) + '</b>\naverage over ' + n.toLocaleString('en-US') + ' dropout masks with weights × ' + U.f(k, 2) + ': <b>' + U.f(s / n, 3) + '</b>' +
        '\n<span class="note">Toy check of the expectation argument. The toy networks use inverted dropout (÷p inside the training forward pass), which builds this 1/p factor in, so pretrained weights can be copied as they are.</span>';
    }
    draw();
  };

  // ---------- ch-pretrain: finetuning from pretrained weights with dropout
  W.finetune = function (host) {
    Book.head(host, 'Finetuning with dropout: the learning rate matters',
      'An autoencoder (49 → 96 ReLU → 49) is first trained without labels on the 500 toy digits plus 2,000 unlabeled ones. Its encoder weights then start a 49-96-10 classifier that is finetuned with dropout. Compare a random start with the pretrained start at a small and at a large learning rate. Press Run (toy, ≈ 10 s).');
    const ctl = el('div', { class: 'controls' }, host);
    const go = Book.button(ctl, 'Run', function () { run(); }, 'primary');
    const plot = el('div', null, host), out = el('div', { class: 'out', text: 'Press Run.' }, host);
    let busy = false;
    function run() {
      if (busy) return; busy = true; go.disabled = true;
      out.textContent = 'pretraining the autoencoder…';
      setTimeout(function () {
        const unl = Toy.digits(2000, 9191).X.concat(Toy.data(500).train.X), ae = new Toy.AE(96, 1, 7), r = ML.rng(3);
        for (let t = 0; t < 1500; t++) { const b = []; for (let i = 0; i < 20; i++) b.push(unl[r.int(unl.length)]); ae.step(b); }
        const copy = function (net) { net.layers[0].W.data.set(ae.enc.W.data); net.layers[0].b.data.set(ae.enc.b.data); };
        const base = Object.assign({}, Toy.BASE, { sizes: [49, 96, 10], keepIn: 0.8, keepHid: 0.5, steps: 1500, evalEvery: 100, seed: 5 });
        const cfgs = [Object.assign({}, base, { label: 'random start, lr 0.005' }), Object.assign({}, base, { label: 'pretrained, lr 0.005', init: copy }), Object.assign({}, base, { label: 'pretrained, lr 0.1', init: copy, lr: 0.1 })];
        Toy.runJobs(cfgs, function (i, T) { out.textContent = 'finetuning ' + (i + 1) + ' of 3 (' + T.cfg.label + '): step ' + T.t; }, function (jobs) {
          busy = false; go.disabled = false;
          const cols = [Book.palette[7], Book.palette[0], Book.palette[1]];
          Book.chart.lines(plot, jobs.map(function (T, i) { return { name: ['random', 'pre, small', 'pre, large'][i], color: cols[i], marker: 'none', points: T.curve.map(function (r) { return [r[0], r[2]]; }) }; }),
            { width: 640, height: 240, xmin: 0, xmax: 1500, ymin: 0, ymax: 40, xticks: [0, 500, 1000, 1500], xlabel: 'finetuning steps', ylabel: 'test error % (toy)', mr: 90 });
          out.innerHTML = jobs.map(function (T) { const s = Toy.summary(T); return T.cfg.label + ': first check after 100 steps ' + U.pct(T.curve[1][2]) + ' · final ' + '<b>' + U.pct(s.test) + '</b>'; }).join('\n') +
            '\n<span class="note">Toy numbers; your run is the experiment. The paper reports that learning rates as large as those best for random starts wiped out the pretrained information, while smaller ones kept it (§5.2).</span>';
        });
      }, 30);
    }
  };
})();
