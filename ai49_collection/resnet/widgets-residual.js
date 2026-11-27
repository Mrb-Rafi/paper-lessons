/* Widgets for Part IV (residual learning) of the ResNet book. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, T = window.RNT, U = window.RNU;
  function vec(v, d) { return '(' + Array.from(v).map(function (x) { return T.f(x, d == null ? 2 : d); }).join(', ') + ')'; }

  // ---------------------------------------------------------------- ch-bigidea
  W.residualSplit = function (host) {
    Book.head(host, 'Split the target into “input” plus “change”',
      'H(x) is the function some layers should compute. Write it as x + F(x): the input itself plus a residual F. Make H closer to or further from doing nothing, and compare the size of H with the size of F.');
    let a = 0.25, kind = 'wiggle';
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'how much H changes its input', 0, 1.5, 0.05, a, function (v) { a = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.select(ctl, 'shape of the change', [{ value: 'wiggle', label: 'wiggle' }, { value: 'shrink', label: 'shrink' }, { value: 'bend', label: 'bend (ReLU-like)' }], kind, function (v) { kind = v; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    function F(x) { return kind === 'wiggle' ? a * Math.sin(3 * x) : kind === 'shrink' ? -a * 0.5 * x : a * (Math.max(0, x) - 0.5 * x); }
    function draw() {
      const H = [], I = [], R = []; let sH = 0, sF = 0;
      for (let i = 0; i <= 120; i++) { const x = -2 + 4 * i / 120, f = F(x); H.push([x, x + f]); I.push([x, x]); R.push([x, f]); sH += (x + f) * (x + f); sF += f * f; }
      U.lines(plot, [{ name: 'x (do nothing)', color: U.C.grey, dash: '5 4', marker: 'none', points: I }, { name: 'H(x), the target', color: U.C.plain, marker: 'none', points: H }, { name: 'F(x) = H(x) − x, the residual', color: U.C.res, marker: 'none', points: R }],
        { xmin: -2, xmax: 2, ymin: -3, ymax: 3, xticks: [-2, -1, 0, 1, 2], xlabel: 'x', ylabel: 'value', height: 240 });
      out.innerHTML = 'size of H (rms): <b>' + T.f(Math.sqrt(sH / 121), 2) + '</b> · size of F (rms): <b>' + T.f(Math.sqrt(sF / 121), 2) + '</b>. When H is close to “do nothing”, the residual F is a small correction, and pushing F to zero gives the identity exactly.';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-identity: learning (near-)identity, plain vs residual stacks
  W.identityFit = function (host) {
    Book.head(host, 'Learning to do (almost) nothing',
      'A stack of 2-layer ReLU blocks must learn H(x) = x + a small change, for 4-number inputs. In the plain stack each block must output H itself; in the residual stack each block outputs x + F(x). Same layers, same small random starting weights, same SGD. Watch the loss (log scale).');
    const st = { k: 6, tweak: 0.1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'blocks in the stack', 1, 12, 1, st.k, function (v) { st.k = v; }, function (v) { return v + ' (' + 2 * v + ' layers)'; });
    Book.slider(ctl, 'size of the change', 0, 0.5, 0.05, st.tweak, function (v) { st.tweak = v; }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'Train ▶', function () { start(); }, 'primary');
    const plot = el('div', null, host), out = U.out(host);
    let run = null;
    function start() {
      if (run) run.stop();
      const r = ML.rng(4), n = 256, D = 4, X = NN.zeros(n, D), Y = NN.zeros(n, D), A = ML.randn(D, D, r, 1);
      for (let i = 0; i < n * D; i++) X.data[i] = r.normal(0, 1);
      for (let i = 0; i < n; i++) for (let j = 0; j < D; j++) { let s = 0; for (let q = 0; q < D; q++) s += A[j][q] * X.data[i * D + q]; Y.data[i * D + j] = X.data[i * D + j] + st.tweak * Math.tanh(s); }
      const nets = [false, true].map(function (res) {
        const rr = ML.rng(9), blocks = [];
        for (let b = 0; b < st.k; b++) blocks.push({ l1: new NN.Linear(D, 16, rr, { sd: Math.sqrt(2 / D) }), l2: new NN.Linear(16, D, rr, { sd: 0.1 * Math.sqrt(2 / 16) }) });
        const params = []; blocks.forEach(function (b) { params.push.apply(params, b.l1.params.concat(b.l2.params)); });
        const opt = new NN.SGD(params, { lr: 0.01, momentum: 0.9 });
        const job = { res: res, step: 0, hist: [], last: NaN };
        job.one = function () {
          NN.zeroGrad(params); let h = X;
          blocks.forEach(function (b) { const f = b.l2.forward(NN.relu(b.l1.forward(h))); h = res ? NN.add(h, f) : f; });
          const L = NN.mse(h, Y); L.backward();
          params.forEach(function (p) { for (let i = 0; i < p.grad.length; i++) p.grad[i] = Math.max(-5, Math.min(5, p.grad[i])); });
          opt.step(); job.step++; job.last = L.item(); if (job.step % 5 === 0 || job.step === 1) job.hist.push([job.step, Math.max(1e-5, L.item())]);
          return isFinite(L.item());
        };
        job.done = function () { return job.step >= 400; };
        return job;
      });
      run = T.run(nets, function () {
        U.logLines(plot, nets.map(function (j) { return { name: j.res ? 'residual stack' : 'plain stack', color: j.res ? U.C.res : U.C.plain, marker: 'none', points: j.hist.length ? j.hist : [[0, 1]] }; }),
          { ymin: 1e-5, ymax: 1e2, xmin: 0, xmax: 400, xticks: [0, 100, 200, 300, 400], xlabel: 'SGD steps', ylabel: 'mean squared error', height: 220 });
        out.innerHTML = 'after ' + nets[0].step + ' steps: plain stack error <b>' + (nets[0].dead ? 'diverged' : nets[0].last.toExponential(1)) + '</b>, residual stack error <b>' + nets[1].last.toExponential(1) + '</b>. ' +
          'The residual stack starts close to the identity and only has to learn the small change; the plain stack must build the identity out of ReLU layers first (toy).';
      });
    }
    start();
  };

  // ---------------------------------------------------------------- ch-block: numbers through one block
  W.blockStep = function (host) {
    Book.head(host, 'One building block, number by number',
      'A two-layer block on a 2-number input, as in Fig. 2 and Eq. (1): weight layer, ReLU, weight layer, add the input, ReLU. Change the input and the scale of the block’s weights; switch the shortcut off to get the plain version.');
    const st = { x1: 1, x2: -0.5, s: 0.5, sc: true, single: false };
    const W1 = [[1.2, -0.8], [0.6, 1.0]], W2 = [[0.9, -1.1], [-0.4, 0.7]];
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'x₁', -2, 2, 0.1, st.x1, function (v) { st.x1 = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'x₂', -2, 2, 0.1, st.x2, function (v) { st.x2 = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'weight scale', 0, 1.5, 0.05, st.s, function (v) { st.s = v; draw(); }, function (v) { return v.toFixed(2); });
    const c1 = el('label', null, ctl), cb = el('input', { type: 'checkbox', checked: 'checked' }, c1); c1.appendChild(document.createTextNode(' shortcut')); cb.addEventListener('change', function () { st.sc = cb.checked; draw(); });
    const c2 = el('label', null, ctl), cb2 = el('input', { type: 'checkbox' }, c2); c2.appendChild(document.createTextNode(' one-layer F')); cb2.addEventListener('change', function () { st.single = cb2.checked; draw(); });
    const view = el('div', { class: 'scroll-x' }, host), out = U.out(host);
    function mv(M, v) { return [st.s * (M[0][0] * v[0] + M[0][1] * v[1]), st.s * (M[1][0] * v[0] + M[1][1] * v[1])]; }
    function draw() {
      const x = [st.x1, st.x2], a = mv(W1, x), r = a.map(function (v) { return Math.max(0, v); }), F = st.single ? a : mv(W2, r), sum = st.sc ? [F[0] + x[0], F[1] + x[1]] : F, y = st.single ? sum : sum.map(function (v) { return Math.max(0, v); });
      const steps = st.single
        ? [['x', x], ['W₁x', a], [st.sc ? 'F + x = W₁x + x' : 'F = W₁x', sum]]
        : [['x', x], ['W₁x', a], ['σ(W₁x)', r], ['F = W₂σ(W₁x)', F], [st.sc ? 'F + x' : '(no shortcut)', sum], ['y = σ(·)', y]];
      view.innerHTML = '';
      const w = 116, svg = S('svg', { viewBox: '0 0 ' + (steps.length * w + 10) + ' 132', style: 'min-width:' + (steps.length * 96) + 'px' }, view);
      steps.forEach(function (s, i) {
        const cx = 8 + i * w;
        S('rect', { x: cx, y: 20, width: w - 22, height: 74, rx: 9, fill: i === 0 ? 'var(--accent-soft)' : 'var(--panel)', stroke: i === steps.length - 1 ? 'var(--accent)' : 'var(--line)', 'stroke-width': 1.5 }, svg);
        S('text', { x: cx + (w - 22) / 2, y: 14, 'text-anchor': 'middle', 'font-size': 11.5, class: 'svg-text', text: s[0] }, svg);
        s[1].forEach(function (v, j) { S('text', { x: cx + (w - 22) / 2, y: 50 + j * 26, 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 600, class: 'svg-text', style: 'font-family:var(--font-mono)', text: T.f(v, 2) }, svg); });
        if (i < steps.length - 1) S('text', { x: cx + w - 18, y: 62, 'font-size': 16, class: 'svg-text', text: '→' }, svg);
      });
      if (st.sc) { S('path', { d: 'M ' + (8 + (w - 22) / 2) + ' 96 C ' + (8 + (w - 22) / 2) + ' 126, ' + (8 + (steps.length - (st.single ? 1 : 2)) * w + (w - 22) / 2) + ' 126, ' + (8 + (steps.length - (st.single ? 1 : 2)) * w + (w - 22) / 2) + ' 96', fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2, 'stroke-dasharray': '5 3' }, svg); S('text', { x: 8 + w * (steps.length - 1) / 2, y: 128, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: 'identity shortcut: x is added unchanged' }, svg); }
      out.innerHTML = st.single
        ? 'With a single layer, y = W₁x + x = (W₁ + I)x is just another linear layer: the paper “has not observed advantages” for this case (§3.2).'
        : 'Output y = ' + vec(y) + '. ' + (st.sc ? 'Turn the weight scale to 0: F becomes (0, 0) and the block outputs σ(x) = ' + vec(x.map(function (v) { return Math.max(0, v); })) + ', the input itself wherever it is not negative.' : 'Without the shortcut, a weight scale of 0 outputs (0, 0): the block destroys its input.');
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-gradflow: gradient size per layer at initialization (no BN)
  W.gradFlow = function (host) {
    Book.head(host, 'Shortcuts give the gradient a direct road',
      'Real networks (16 units wide, no batch normalization), freshly initialized with weights of spread scale × √(2/n). One backward pass measures the gradient arriving at each block’s output. Plain: the gradient is a product of one factor per layer. Residual: every block adds an identity path.');
    const st = { blocks: 30, scale: 0.6 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'blocks', 5, 50, 1, st.blocks, function (v) { st.blocks = v; draw(); }, function (v) { return v + ' (' + (2 * v + 2) + ' layers)'; });
    Book.slider(ctl, 'weight scale', 0.2, 1.4, 0.05, st.scale, function (v) { st.scale = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = U.out(host);
    function measure(res) {
      const r = ML.rng(6), d = T.spiralData().train, idx = ML.range(128);
      const net = new T.MLP({ inDim: 2, width: 16, blocks: st.blocks, residual: res, bn: false, classes: 5, seed: 2 });
      [net.inW].concat(net.blocks.reduce(function (a, b) { return a.concat([b.W1, b.W2]); }, [])).forEach(function (Wt) { for (let i = 0; i < Wt.data.length; i++) Wt.data[i] *= st.scale; });
      const rec = [], X = T.rows(d, idx), L = NN.softmaxCE(net.forward(X, rec, 'batch'), idx.map(function (k) { return d.y[k]; }));
      L.backward();
      return rec.map(function (h, i) { let s = 0; for (let k = 0; k < h.grad.length; k++) s += h.grad[k] * h.grad[k]; return [i, Math.max(1e-12, Math.sqrt(s))]; });
    }
    function draw() {
      const p = measure(false), q = measure(true);
      U.logLines(plot, [{ name: 'plain', color: U.C.plain, points: p, r: 2.2 }, { name: 'residual', color: U.C.res, points: q, r: 2.2 }],
        { ymin: 1e-10, ymax: 1e4, xmin: 0, xmax: st.blocks, xticks: [0, Math.round(st.blocks / 2), st.blocks], xlabel: 'block (0 = bottom, loss at the right)', ylabel: '‖gradient at block output‖', height: 230 });
      const ratio = function (a) { return a[0][1] / a[a.length - 1][1]; };
      out.innerHTML = 'bottom ÷ top gradient: plain <b>' + ratio(p).toExponential(1) + '</b>, residual <b>' + ratio(q).toExponential(1) + '</b>. ' +
        'With small weights the plain product vanishes while the residual network stays near 1: each block’s slope is I + ∂F/∂x, and the I never shrinks. (With very large weights the residual sum grows instead.)';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-proj: options A, B, C
  W.shortcutOptions = function (host) {
    Book.head(host, 'When the shapes do not match: zero-padding or projection',
      'A block that halves the map (4×4 → 2×2) and doubles the channels (4 → 8) cannot add x directly. Option A takes every second position and pads the new channels with zeros. Option B uses a 1×1 convolution with stride 2 (Eq. 2). Compare the shortcut outputs and their cost.');
    let opt = 'A';
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'shortcut', [{ value: 'A', label: 'A: identity + zero padding' }, { value: 'B', label: 'B: 1×1 projection, stride 2' }], opt, function (v) { opt = v; draw(); });
    const view = el('div', null, host), tbl = el('div', { class: 'table-wrap' }, host), out = U.out(host);
    const r = ML.rng(12), X = new NN.Tensor(1, 64), s = { C: 4, H: 4, W: 4 };
    for (let i = 0; i < 64; i++) X.data[i] = Math.round(r.uniform(0, 9));
    const Ws = new NN.Tensor(8, 4); for (let i = 0; i < 32; i++) Ws.data[i] = Math.round(r.uniform(-2, 2) * 2) / 2;
    function draw() {
      const y = opt === 'A' ? T.padShortcut(X, s, 8, 2) : T.conv(X, s, Ws, 1, 2, 0);
      view.innerHTML = '';
      const row = el('div', { style: 'display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start' }, view);
      const a = el('div', null, row); el('div', { class: 'note', text: 'input x: 4 channels × 4×4' }, a);
      const ga = el('div', { style: 'display:grid;grid-template-columns:repeat(2,auto);gap:4px' }, a);
      for (let c = 0; c < 4; c++) { const b = el('div', null, ga); U.grid(b, X.data.subarray(c * 16, c * 16 + 16), 4, 4, { cell: 17, max: 9, numbers: true, digits: 0, hue: U.stage[c], hl: [0, 2, 8, 10] }); }
      const bb = el('div', null, row); el('div', { class: 'note', text: 'shortcut output: 8 channels × 2×2' }, bb);
      const gb = el('div', { style: 'display:grid;grid-template-columns:repeat(4,auto);gap:4px' }, bb);
      for (let c = 0; c < 8; c++) { const b = el('div', null, gb); U.grid(b, y.t.data.subarray(c * 4, c * 4 + 4), 2, 2, { cell: 22, max: opt === 'A' ? 9 : 20, numbers: true, digits: 1, hue: c < 4 || opt === 'B' ? U.stage[c % 7] : '#718096', signed: opt === 'B' }); }
      const t = function (o) { return T.archTotals(T.archLayers(34, { shortcut: o })); };
      const A = t('A'), B = t('B'), C = t('C');
      tbl.innerHTML = '<table class="data"><tr><th>ResNet-34</th><th>projection shortcuts</th><th>parameters</th><th>extra vs A</th></tr>' +
        [['A', A], ['B', B], ['C', C]].map(function (p) { return '<tr><td>option ' + p[0] + '</td><td>' + p[1].proj + '</td><td>' + T.comma(p[1].params) + '</td><td>' + (p[0] === 'A' ? '—' : '+' + T.comma(p[1].params - A.params)) + '</td></tr>'; }).join('') + '</table>';
      out.innerHTML = opt === 'A'
        ? 'Option A: the highlighted positions are kept, channels 5–8 are all zeros. 0 parameters. Those zero channels get no shortcut, so no residual learning happens in them (the paper’s explanation for why B is slightly better).'
        : 'Option B: each output channel is a weighted sum of the 4 input channels at every second position: 4 × 8 = 32 weights here (plus BN). In ResNet-34, only the 3 stage transitions need it. Option C uses projections in all 16 blocks: 13 more than B. (Parameter counts computed from Table 1, including BN.)';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-paths: the unrolled view
  W.pathEnsemble = function (host) {
    Book.head(host, 'Unrolled, a residual network is many paths',
      'Each block either transforms the signal (go through F) or skips it (take the shortcut). With n blocks there are 2ⁿ ways through. Below: how many paths have each length, and an experiment: delete one block from a trained toy network.');
    let n = 10;
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'blocks n', 1, 54, 1, n, function (v) { n = v; draw(); });
    const plot = el('div', null, host), out = U.out(host), exp = el('div', null, host), out2 = U.out(host);
    function binom(a, b) { let r = 1; for (let i = 1; i <= b; i++) r = r * (a - b + i) / i; return r; }
    function draw() {
      const bars = []; for (let k = 0; k <= n; k++) bars.push({ label: n <= 20 || k % Math.ceil(n / 20) === 0 ? String(k) : '', value: binom(n, k), color: 'var(--accent)', note: '' });
      Book.chart.bars(plot, bars, { ylabel: 'number of paths', height: 190, tickFmt: function (v) { return v >= 1e6 ? v.toExponential(0) : Math.round(v); } });
      let mean = n / 2;
      out.innerHTML = 'n = ' + n + ': <b>' + (n <= 40 ? T.comma(Math.pow(2, n)) : Math.pow(2, n).toExponential(2)) + '</b> paths (2<sup>' + n + '</sup>). Path length = number of blocks it goes through (x axis); most paths go through about n/2 = ' + mean + ' blocks. A 110-layer CIFAR ResNet has n = 54 blocks.';
    }
    draw();
    // experiment
    out2.textContent = 'training a residual and a plain network of 10 blocks (22 layers) on the spirals…';
    const d = T.spiralData();
    const jobs = [true, false].map(function (res) { return T.trainer(new T.MLP({ inDim: 2, width: 16, blocks: 10, residual: res, classes: 5, seed: 3 }), d.train, { steps: 1500, batch: 64, lr: 0.1, schedule: [[1000, 0.01]], seed: 11, every: 100 }); });
    T.run(jobs, function () { out2.textContent = 'training… step ' + jobs[0].step + ' / 1500'; }, function () {
      const res = [], pl = [];
      [0, 1].forEach(function (which) {
        const net = jobs[which].net, base = T.evaluate(net, d.train).err, arr = which ? pl : res; arr.push(base);
        for (let b = 0; b < 10; b++) {
          const saved = net.blocks; net.blocks = saved.filter(function (_, i) { return i !== b; });
          arr.push(T.evaluate(net, d.train).err); net.blocks = saved;
        }
      });
      const series = function (arr, col, nm) { return { name: nm, color: col, points: arr.slice(1).map(function (e, i) { return [i + 1, 100 * e]; }) }; };
      U.lines(exp, [series(res, U.C.res, 'residual: error after deleting block i'), series(pl, U.C.plain, 'plain: error after deleting block i')], { xmin: 1, xmax: 10, ymin: 0, ymax: 100, xticks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], xlabel: 'deleted block', ylabel: 'training error (%)', height: 210 });
      out2.innerHTML = 'Intact: residual ' + (100 * res[0]).toFixed(1) + '%, plain ' + (100 * pl[0]).toFixed(1) + '% training error. Deleting one block of the plain network breaks its only path; the residual network loses only the paths through that block (toy). This view comes from later work (Veit, Wilber &amp; Belongie, 2016), not from the paper.';
    });
  };

  // ---------------------------------------------------------------- ch-related: highway gates vs identity shortcuts
  W.highwayGate = function (host) {
    Book.head(host, 'Gated shortcuts (highway networks) against identity shortcuts',
      'A highway layer mixes a transformation H(x) and the input x with a learned gate T(x) between 0 and 1: y = T·H(x) + (1 − T)·x. Move the gate’s bias. When the gate closes the shortcut (T → 1), no part of x passes through unchanged. A residual block always adds x.');
    let bias = 0;
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'gate bias b_T', -6, 6, 0.25, bias, function (v) { bias = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = U.out(host);
    const Hf = function (x) { return Math.tanh(2 * x) * 1.2; };
    function draw() {
      const hw = [], rs = [], id = [], g = [];
      for (let i = 0; i <= 100; i++) { const x = -2 + 4 * i / 100, Tg = ML.sigmoid(1.5 * x + bias); hw.push([x, Tg * Hf(x) + (1 - Tg) * x]); rs.push([x, x + 0.3 * Hf(x)]); id.push([x, x]); g.push([x, Tg * 2 - 2]); }
      U.lines(plot, [{ name: 'x', color: U.C.grey, dash: '4 4', marker: 'none', points: id }, { name: 'highway output', color: U.C.purple, marker: 'none', points: hw }, { name: 'residual output x + F(x)', color: U.C.res, marker: 'none', points: rs }, { name: 'gate T(x) (drawn 0…1 as −2…0)', color: U.C.gold, marker: 'none', points: g, dash: '2 3' }],
        { xmin: -2, xmax: 2, ymin: -2.2, ymax: 2.2, xticks: [-2, -1, 0, 1, 2], xlabel: 'x', ylabel: 'output', height: 240 });
      const t0 = ML.sigmoid(bias);
      out.innerHTML = 'gate at x = 0: T = <b>' + T.f(t0, 3) + '</b> → share of x carried unchanged: <b>' + T.f(1 - t0, 3) + '</b>. ' +
        'A highway gate has its own weights (as many as the transformation it gates): data-dependent and learned. The identity shortcut has none and is never closed (§2).';
    }
    draw();
  };
})();
