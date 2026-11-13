/* Refresher widgets shared by every book in the collection. They power each book's "Before you start" chapter,
   so every book can start from zero without repeating code. Names start with r_ to avoid clashes with a book's own widgets.
     r_vectors        two vectors: length, dot product, cosine similarity, angle
     r_neuron         one artificial neuron: weighted sum + activation
     r_descent        gradient descent on a 1-D loss curve (learning-rate slider, step / run)
     r_softmax        scores → probabilities with temperature
     r_trainNet       train a small neural network live on a 2-D toy dataset (decision regions + loss curve)
   Depends on: core.js (window.Book), lib/math.js (window.ML), lib/nn.js (window.NN, for r_trainNet). */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML;

  // ------------------------------------------------------------------ vectors
  W.r_vectors = function (host) {
    Book.head(host, 'Two vectors, three measurements',
      'Each arrow is a vector: a list of two numbers. Move them and watch the length, the dot product and the cosine similarity.');
    const st = { a: [3, 1], b: [1, 2.5] };
    const ctl = el('div', { class: 'controls' }, host);
    [['a', 0, 'a₁'], ['a', 1, 'a₂'], ['b', 0, 'b₁'], ['b', 1, 'b₂']].forEach(function (s) {
      Book.slider(ctl, s[2], -4, 4, 0.1, st[s[0]][s[1]], function (v) { st[s[0]][s[1]] = v; draw(); }, function (v) { return v.toFixed(1); });
    });
    const box = el('div', { class: 'grid2' }, host);
    const pic = el('div', null, box), out = el('div', { class: 'out' }, box);
    function draw() {
      pic.innerHTML = '';
      const Wd = 260, U = 26, c = Wd / 2;
      const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + Wd, role: 'img', 'aria-label': 'Two vectors on a grid' }, pic);
      for (let k = -4; k <= 4; k++) {
        S('line', { x1: c + k * U, x2: c + k * U, y1: 4, y2: Wd - 4, stroke: 'var(--line)' }, svg);
        S('line', { y1: c + k * U, y2: c + k * U, x1: 4, x2: Wd - 4, stroke: 'var(--line)' }, svg);
      }
      [['a', 'var(--blue)'], ['b', 'var(--red)']].forEach(function (p) {
        const v = st[p[0]], x = c + v[0] * U, y = c - v[1] * U;
        S('line', { x1: c, y1: c, x2: x, y2: y, stroke: p[1], 'stroke-width': 3, 'stroke-linecap': 'round' }, svg);
        S('circle', { cx: x, cy: y, r: 5, fill: p[1] }, svg);
        S('text', { x: x + 7, y: y - 7, class: 'svg-text', 'font-size': 13, 'font-weight': 700, text: p[0] }, svg);
      });
      const a = st.a, b = st.b, dot = ML.dot(a, b), cos = ML.cosine(a, b);
      const ang = Math.acos(Book.clamp(cos, -1, 1)) * 180 / Math.PI;
      out.innerHTML = 'length of a = √(' + a[0].toFixed(1) + '² + ' + a[1].toFixed(1) + '²) = <b>' + Book.fmt(ML.norm(a)) + '</b><br>' +
        'length of b = <b>' + Book.fmt(ML.norm(b)) + '</b><br>' +
        'dot product a·b = ' + a[0].toFixed(1) + '×' + b[0].toFixed(1) + ' + ' + a[1].toFixed(1) + '×' + b[1].toFixed(1) + ' = <b>' + Book.fmt(dot) + '</b><br>' +
        'cosine similarity = a·b ÷ (|a| |b|) = <b>' + Book.fmt(cos) + '</b><br>' +
        'angle between them = <b>' + Book.fmt(ang, 0) + '°</b><br><span class="note">Cosine 1 = same direction, 0 = at right angles, −1 = opposite.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ one neuron
  W.r_neuron = function (host) {
    Book.head(host, 'One artificial neuron',
      'A neuron multiplies each input by a weight, adds them up with a bias, then passes the sum through an activation function.');
    const st = { x: [1, 0.5], w: [0.8, -1.2], b: 0.3, act: 'relu' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'input x₁', -2, 2, 0.1, st.x[0], function (v) { st.x[0] = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'input x₂', -2, 2, 0.1, st.x[1], function (v) { st.x[1] = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'weight w₁', -2, 2, 0.1, st.w[0], function (v) { st.w[0] = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'weight w₂', -2, 2, 0.1, st.w[1], function (v) { st.w[1] = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'bias b', -2, 2, 0.1, st.b, function (v) { st.b = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.select(ctl, 'activation', [{ value: 'relu', label: 'ReLU' }, { value: 'sigmoid', label: 'sigmoid' }, { value: 'tanh', label: 'tanh' }, { value: 'none', label: 'none' }], st.act, function (v) { st.act = v; draw(); });
    const box = el('div', { class: 'grid2' }, host), plot = el('div', null, box), out = el('div', { class: 'out' }, box);
    const F = { relu: function (z) { return Math.max(0, z); }, sigmoid: ML.sigmoid, tanh: Math.tanh, none: function (z) { return z; } };
    function draw() {
      const z = st.w[0] * st.x[0] + st.w[1] * st.x[1] + st.b, y = F[st.act](z);
      const xs = ML.linspace(-4, 4, 81);
      const r = Book.chart.lines(plot, [{ name: '', color: Book.palette[0], marker: 'none', points: xs.map(function (t) { return [t, F[st.act](t)]; }) },
        { color: Book.palette[1], noline: true, points: [[Book.clamp(z, -4, 4), F[st.act](Book.clamp(z, -4, 4))]], r: 6 }],
        { width: 320, height: 220, xmin: -4, xmax: 4, ymin: st.act === 'relu' ? 0 : -1.2, ymax: st.act === 'relu' ? 4 : (st.act === 'none' ? 4 : 1.2), xticks: [-4, -2, 0, 2, 4], xlabel: 'weighted sum z', ylabel: 'output' });
      if (st.act === 'none') r.svg.querySelectorAll('polyline')[0].setAttribute('opacity', 0.9);
      out.innerHTML = 'z = w₁x₁ + w₂x₂ + b<br>= ' + st.w[0].toFixed(1) + '×' + st.x[0].toFixed(1) + ' + ' + st.w[1].toFixed(1) + '×' + st.x[1].toFixed(1) + ' + ' + st.b.toFixed(1) +
        ' = <b>' + Book.fmt(z) + '</b><br>output = ' + (st.act === 'none' ? 'z' : st.act + '(z)') + ' = <b>' + Book.fmt(y, 3) + '</b>' +
        '<br><span class="note">Learning means adjusting w₁, w₂ and b until the outputs are useful.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ gradient descent
  W.r_descent = function (host) {
    Book.head(host, 'Rolling downhill: gradient descent',
      'The curve is the loss: how wrong the model is for each value of one weight w. Each step moves w against the slope, by (learning rate × slope).');
    const loss = function (w) { return 0.25 * (w - 1) * (w - 1) * (0.3 * (w + 2) * (w + 2) + 1) / 2 + 0.4; };
    const slope = function (w) { const h = 1e-5; return (loss(w + h) - loss(w - h)) / (2 * h); };
    const st = { w: -3.2, lr: 0.3, path: [] };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'learning rate', 0.02, 1.6, 0.02, st.lr, function (v) { st.lr = v; }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'Take one step', function () { step(); draw(); }, 'primary');
    Book.button(ctl, 'Run 20 steps', function () { for (let i = 0; i < 20; i++) step(); draw(); });
    Book.button(ctl, 'Reset', function () { st.w = -3.2; st.path = []; draw(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function step() { st.path.push(st.w); st.w = Book.clamp(st.w - st.lr * slope(st.w), -4.5, 4.5); }
    function draw() {
      const xs = ML.linspace(-4.5, 4.5, 120);
      const pts = st.path.concat([st.w]).map(function (w) { return [w, loss(w)]; });
      Book.chart.lines(plot, [
        { color: 'var(--muted)', marker: 'none', points: xs.map(function (w) { return [w, loss(w)]; }) },
        { color: Book.palette[1], points: pts, width: 1.5, dash: '4 3' }
      ], { width: 640, height: 240, xmin: -4.5, xmax: 4.5, ymin: 0, ymax: 6, xticks: [-4, -2, 0, 1, 2, 4], xlabel: 'weight w', ylabel: 'loss' });
      out.innerHTML = 'step ' + st.path.length + ': w = <b>' + Book.fmt(st.w, 3) + '</b>, loss = <b>' + Book.fmt(loss(st.w), 3) + '</b>, slope = ' + Book.fmt(slope(st.w), 3) +
        ' → next step moves w by ' + Book.fmt(-st.lr * slope(st.w), 3) +
        '<br><span class="note">Too small a learning rate crawls; too large overshoots and bounces. The lowest point is at w = 1.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ softmax
  W.r_softmax = function (host) {
    Book.head(host, 'Scores become probabilities (softmax)',
      'A model outputs a score for each option. Softmax turns scores into probabilities that are positive and add up to 1. Temperature τ sharpens or flattens them.');
    const names = ['cat', 'dog', 'car', 'tree'], sc = [2.0, 1.4, -0.5, 0.3];
    let tau = 1;
    const ctl = el('div', { class: 'controls' }, host);
    names.forEach(function (n, i) { Book.slider(ctl, 'score(' + n + ')', -3, 3, 0.1, sc[i], function (v) { sc[i] = v; draw(); }, function (v) { return v.toFixed(1); }); });
    Book.slider(ctl, 'temperature τ', 0.1, 3, 0.05, tau, function (v) { tau = v; draw(); }, function (v) { return v.toFixed(2); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const p = ML.softmax(sc, tau);
      Book.chart.bars(chart, names.map(function (n, i) { return { label: n, value: p[i], note: (100 * p[i]).toFixed(1) + '%', color: Book.palette[i] }; }),
        { max: 1, ylabel: 'probability', tickFmt: function (v) { return Math.round(100 * v) + '%'; } });
      out.innerHTML = 'softmax: p<sub>i</sub> = e<sup>s<sub>i</sub>/τ</sup> ÷ Σ<sub>j</sub> e<sup>s<sub>j</sub>/τ</sup> · sum = <b>' + Book.fmt(ML.sum(p), 3) +
        '</b> · entropy = <b>' + Book.fmt(ML.entropy(p, 2), 2) + ' bits</b> (0 = certain, 2 = all four equally likely)';
    }
    draw();
  };

  // ------------------------------------------------------------------ train a network
  // Two interleaved spirals (or rings) in 2-D; an MLP learns to separate them; regions are painted live.
  W.r_trainNet = function (host) {
    Book.head(host, 'Watch a network learn',
      'Each dot is a training example with two numbers (its position) and a colour (its class). Press Train: the network adjusts its weights to lower the loss, and the shaded regions show what it currently predicts.');
    const NN = window.NN;
    const st = { data: 'rings', hidden: 8, lr: 0.03, steps: 0, model: null, opt: null, losses: [], timer: null };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'data', [{ value: 'rings', label: 'rings' }, { value: 'spiral', label: 'spirals' }, { value: 'blobs', label: 'two blobs' }], st.data, function (v) { st.data = v; reset(); });
    Book.select(ctl, 'hidden neurons', [2, 4, 8, 16], st.hidden, function (v) { st.hidden = +v; reset(); });
    Book.slider(ctl, 'learning rate', 0.002, 0.2, 0.002, st.lr, function (v) { st.lr = v; if (st.opt) st.opt.lr = v; }, function (v) { return v.toFixed(3); });
    const go = Book.button(ctl, 'Train ▶', function () { toggle(); }, 'primary');
    Book.button(ctl, 'Reset', function () { reset(); });
    const box = el('div', { class: 'grid2' }, host);
    const cv = el('canvas', { width: 260, height: 260, style: 'width:100%;max-width:260px;border-radius:8px;border:1px solid var(--line)' }, box);
    const right = el('div', null, box), lossPlot = el('div', null, right), out = el('div', { class: 'out' }, right);
    let X, y;
    function makeData() {
      const r = ML.rng(4), pts = [], lab = [];
      for (let i = 0; i < 160; i++) {
        const c = i % 2; let px, py;
        if (st.data === 'rings') { const rad = (c ? 0.75 : 0.3) + r.normal(0, 0.08), a = r.uniform(0, 2 * Math.PI); px = rad * Math.cos(a); py = rad * Math.sin(a); }
        else if (st.data === 'spiral') { const t = r.uniform(0.15, 1), a = 3.6 * t + c * Math.PI; px = t * Math.cos(a) * 0.95 + r.normal(0, 0.03); py = t * Math.sin(a) * 0.95 + r.normal(0, 0.03); }
        else { px = r.normal(c ? 0.4 : -0.4, 0.25); py = r.normal(c ? 0.3 : -0.3, 0.25); }
        pts.push([px, py]); lab.push(c);
      }
      X = NN.tensor(pts); y = lab;
    }
    function reset() {
      stop(); makeData();
      const r = ML.rng(7);
      st.model = { l1: new NN.Linear(2, st.hidden, r, { sd: 1.2 }), l2: new NN.Linear(st.hidden, 2, r) };
      st.opt = new NN.Adam(st.model.l1.params.concat(st.model.l2.params), { lr: st.lr });
      st.steps = 0; st.losses = []; draw();
    }
    function forward(inp) { return st.model.l2.forward(NN.tanh(st.model.l1.forward(inp))); }
    function trainSteps(k) {
      for (let i = 0; i < k; i++) {
        NN.zeroGrad(st.opt.params);
        const L = NN.softmaxCE(forward(X), y); L.backward(); st.opt.step(); st.steps++;
        if (st.steps % 5 === 0) st.losses.push([st.steps, L.item()]);
      }
    }
    function toggle() { if (st.timer) stop(); else { go.textContent = 'Pause ❚❚'; tick(); } }
    function stop() { if (st.timer) clearTimeout(st.timer); st.timer = null; go.textContent = 'Train ▶'; }
    function tick() { trainSteps(10); draw(); if (st.steps >= 3000) { stop(); return; } st.timer = setTimeout(tick, 30); }
    function draw() {
      const ctx = cv.getContext('2d'), N = 52, s = cv.width / N, grid = [];
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) grid.push([-1.1 + 2.2 * (j + 0.5) / N, 1.1 - 2.2 * (i + 0.5) / N]);
      const logits = forward(NN.tensor(grid));
      for (let k = 0; k < grid.length; k++) {
        const p = ML.softmax([logits.data[2 * k], logits.data[2 * k + 1]])[1];
        ctx.fillStyle = 'rgba(' + Math.round(43 + (192 - 43) * p) + ',' + Math.round(108 + (57 - 108) * p) + ',' + Math.round(176 + (43 - 176) * p) + ',0.28)';
        ctx.fillRect((k % N) * s, Math.floor(k / N) * s, s + 1, s + 1);
      }
      for (let i = 0; i < X.rows; i++) {
        const px = (X.get(i, 0) + 1.1) / 2.2 * cv.width, py = (1.1 - X.get(i, 1)) / 2.2 * cv.height;
        ctx.beginPath(); ctx.arc(px, py, 3.2, 0, 2 * Math.PI); ctx.fillStyle = y[i] ? '#c0392b' : '#2b6cb0'; ctx.fill();
        ctx.lineWidth = 1; ctx.strokeStyle = '#fff'; ctx.stroke();
      }
      const L = NN.softmaxCE(forward(X), y), probs = L.probs;
      let correct = 0; for (let i = 0; i < y.length; i++) if ((probs[2 * i + 1] > 0.5 ? 1 : 0) === y[i]) correct++;
      Book.chart.lines(lossPlot, [{ color: Book.palette[0], marker: 'none', points: st.losses.length ? st.losses : [[0, L.item()]] }],
        { width: 320, height: 170, ymin: 0, ymax: Math.max(0.8, st.losses.length ? st.losses[0][1] * 1.05 : 0.8), xlabel: 'training steps', ylabel: 'loss', xticks: [] });
      out.innerHTML = 'steps: <b>' + st.steps + '</b> · loss: <b>' + Book.fmt(L.item(), 3) + '</b> · accuracy: <b>' + Math.round(100 * correct / y.length) + '%</b>' +
        '<br><span class="note">Toy network: 2 inputs → ' + st.hidden + ' tanh neurons → 2 class scores, trained with Adam on 160 points. With 2 hidden neurons the rings cannot be separated; with 8 they can.</span>';
    }
    reset();
  };
})();
