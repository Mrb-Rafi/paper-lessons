/* Adam Interactive Web Book — Part V widgets: the experiments (paper §6) and the limits chapter.
   Uses Book, ML, NN, Opt and window.AD (widgets-found.js). Training runs are time-sliced with AD.runner so the
   page stays responsive; every number they print is a toy number. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, Opt = window.Opt, AD = window.AD;

  // shared toy data sets, built once
  const data = {};
  const dense = function () { return data.dense || (data.dense = Opt.glyphs(1000, 3, 8, 10, 1.2)); };
  const sparse = function () { return data.sparse || (data.sparse = Opt.reviews(2000, 5, 3000, 150, { ps: 0.08, off: 100, flip: 0.1 })); };
  const mlpData = function () { return data.mlp || (data.mlp = Opt.glyphs(600, 4, 8, 10, 1.2)); };
  const vaeData = function () { return data.small || (data.small = Opt.glyphs(400, 7, 4, 4, 0.5)); };
  AD.toyData = { dense: dense, sparse: sparse };

  // a small "start / running / done" status helper for widgets that train
  function trainer(host, label, build, draw, autostart) {
    const ctl = el('div', { class: 'controls' }, host), run = AD.runner();
    let jobs = null;
    const btn = Book.button(ctl, label, function () { start(); }, 'primary');
    function start() { btn.disabled = true; jobs = build(); run.start(jobs, function () { draw(jobs, false); }, function () { btn.disabled = false; btn.textContent = 'Run again'; draw(jobs, true); }); }
    if (autostart !== false) setTimeout(start, 50);
    return { controls: ctl, start: start, stop: run.stop };
  }

  // ================================================================== ch-setup
  W.gridSearch = function (host) {
    Book.head(host, 'A dense grid search, then report the best',
      'The paper searches hyper-parameters “over a dense grid” and reports each optimizer’s best setting (§6). Here: four optimizers × five learning rates on the toy glyph data (softmax regression), 10 epochs each. Cells show the final training cost; the best per row is boxed.');
    const OPTS = [['nesterov', [0.01, 0.03, 0.1, 0.3, 1]], ['adagrad', [0.03, 0.1, 0.3, 1, 3]], ['rmsprop', [0.0003, 0.001, 0.003, 0.01, 0.03]], ['adam', [0.01, 0.03, 0.1, 0.3, 1]]];
    const tab = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    const T = trainer(host, 'Run the grid (20 trainings)', function () {
      const D = dense(), jobs = [];
      OPTS.forEach(function (o) { o[1].forEach(function (lr) { const J = Opt.job(new Opt.SoftmaxReg(D, 9, 1e-3), o[0], { lr: lr, decay: o[0] === 'adam' || undefined }, { bs: 50, epochs: 10 }); J.key = o[0]; J.lr = lr; jobs.push(J); }); });
      return jobs;
    }, function (jobs, done) {
      let html = '<table><tr><th>optimizer</th>' + [0, 1, 2, 3, 4].map(function (i) { return '<th>rate ' + (i + 1) + '</th>'; }).join('') + '</tr>', k = 0;
      OPTS.forEach(function (o) {
        const row = jobs.slice(k, k + 5); k += 5;
        const fin = row.map(function (J) { return J.done ? J.curve[J.curve.length - 1][1] : null; });
        const best = fin.every(function (x) { return x != null; }) ? ML.argmin(fin) : -1;
        html += '<tr><td><b style="color:' + Opt.colors[o[0]] + '">' + Opt.names[o[0]] + '</b></td>' + row.map(function (J, i) {
          const v = fin[i]; return '<td style="' + (i === best ? 'outline:2px solid var(--accent);font-weight:700' : '') + '">' + J.lr + (o[0] === 'adam' ? '/√t' : '') + '<br>' + (v == null ? '… ' + J.epoch + '/10' : (J.diverged ? 'blew up' : v.toFixed(3))) + '</td>';
        }).join('') + '</tr>';
      });
      tab.innerHTML = html + '</table>';
      out.innerHTML = done ? 'Every optimizer gets its own best learning rate; only those best runs would be plotted in a figure like the paper’s. Toy numbers.' : 'training… (all in your browser)';
    }, false);
    tab.innerHTML = '<div class="note">Press the button to train 20 small models (about 3 seconds).</div>';
    void T;
  };

  W.modelZoo = function (host) {
    Book.head(host, 'The paper’s models, and what Adam must store for them',
      'Pick one of the models of §6. The book counts its parameters layer by layer (where the paper leaves a detail out, the assumption is marked), and how much extra memory Adam needs: one m and one v per parameter.');
    const Z = {
      lrm: { name: 'Logistic regression, MNIST (§6.1)', layers: [['784 pixels → 10 classes', 784 * 10 + 10]], note: '' },
      lri: { name: 'Logistic regression, IMDB bag of words (§6.1)', layers: [['10,000 word counts → 1 sentiment score', 10000 + 1]], note: 'Assumes one output for the two sentiment classes.' },
      mlp: { name: 'Multilayer net, MNIST (§6.2)', layers: [['784 → 1000 ReLU', 784 * 1000 + 1000], ['1000 → 1000 ReLU', 1000 * 1000 + 1000], ['1000 → 10 classes', 1000 * 10 + 10]], note: '' },
      cnn: { name: 'Convolutional net c64-c64-c128-1000, CIFAR-10 (§6.3)', layers: [['conv 5×5, 3 → 64', 5 * 5 * 3 * 64 + 64], ['conv 5×5, 64 → 64', 5 * 5 * 64 * 64 + 64], ['conv 5×5, 64 → 128', 5 * 5 * 64 * 128 + 128], ['4×4×128 = 2,048 → 1000 ReLU', 2048 * 1000 + 1000], ['1000 → 10 classes', 1000 * 10 + 10]], note: 'Assumes 32×32 inputs, “same” padding and three stride-2 poolings (32 → 16 → 8 → 4); the paper does not state padding.' },
      vae: { name: 'Variational auto-encoder (§6.4)', layers: [['encoder 784 → 500 softplus', 784 * 500 + 500], ['500 → 50 means + 50 log-variances', 500 * 100 + 100], ['decoder 50 → 500 softplus', 50 * 500 + 500], ['500 → 784 pixel probabilities', 500 * 784 + 784]], note: 'Assumes 784-pixel images as in Kingma & Welling (2013); §6.4 gives the hidden (500) and latent (50) sizes but not the data set.' }
    };
    const st = { k: 'mlp' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'model', Object.keys(Z).map(function (k) { return { value: k, label: Z[k].name }; }), st.k, function (v) { st.k = v; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const z = Z[st.k], tot = ML.sum(z.layers.map(function (l) { return l[1]; }));
      Book.chart.bars(chart, z.layers.map(function (l, i) { return { label: l[0].length > 22 ? l[0].replace(/ → /, '\n→ ') : l[0], value: l[1], note: l[1].toLocaleString('en-US'), color: Book.palette[i % 8] }; }), { height: 230, ylabel: 'parameters', mb: 56, tickFmt: function (v) { return v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? Math.round(v / 1e3) + 'k' : v; } });
      out.innerHTML = 'total parameters: <b>' + tot.toLocaleString('en-US') + '</b> · Adam stores m and v: <b>' + (2 * tot).toLocaleString('en-US') + '</b> extra numbers (' + (8 * tot / 1048576).toFixed(1) + ' MB at 4 bytes each)' +
        (z.note ? '<br><span class="note">' + z.note + '</span>' : '') + '<br><span class="note">Minibatch size 128 in every experiment of the paper. Parameter counts are the book’s arithmetic, not numbers from the paper.</span>';
    }
    draw();
  };

  // ================================================================== ch-logreg
  W.logregRace = function (host) {
    Book.head(host, 'Fig. 1, toy version: dense pixels vs sparse word counts',
      'Logistic regression trained live in your browser. “Dense” is the toy glyph data (64 pixels, 10 classes), standing in for MNIST. “Sparse” is the toy review data (3,000-word bag of words, 2 classes, 50 % dropout on the features), standing in for IMDB. Learning rates are each optimizer’s best from a small grid.');
    const st = { set: 'sparse' };
    const CFG = {
      dense: { epochs: 30, list: [['adagrad', { lr: 0.3 }], ['nesterov', { lr: 0.1 }], ['adam', { lr: 0.3, decay: true }]] },
      sparse: { epochs: 40, list: [['adagrad', { lr: 0.3 }], ['rmsprop', { lr: 0.01 }], ['nesterov', { lr: 0.03 }], ['adam', { lr: 0.3, decay: true }]] }
    };
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    const T = trainer(host, 'Train', function () {
      const c = CFG[st.set];
      return c.list.map(function (o) {
        const M = st.set === 'dense' ? new Opt.SoftmaxReg(dense(), 9, 1e-3) : new Opt.SparseLogReg(sparse(), 9);
        const J = Opt.job(M, o[0], o[1], st.set === 'dense' ? { bs: 50, epochs: c.epochs } : { bs: 32, epochs: c.epochs, pDrop: 0.5 }); J.key = o[0]; J.lr = o[1].lr; return J;
      });
    }, function (jobs, done) {
      const c = CFG[st.set];
      AD.lines(plot, jobs.map(function (J) { return { name: Opt.names[J.key], color: Opt.colors[J.key], marker: 'none', points: J.curve }; }),
        { width: 640, height: 250, xmin: 0, xmax: c.epochs, ymin: 0, ymax: st.set === 'dense' ? 1.2 : 0.7, xlabel: 'epochs (passes over the data)', ylabel: 'training cost (toy)', xticks: [0, c.epochs / 2, c.epochs], mr: 100, ydec: 2 });
      const fin = jobs.map(function (J) { return Opt.names[J.key] + ' (rate ' + J.lr + (J.key === 'adam' ? '/√t' : '') + '): <b>' + J.curve[J.curve.length - 1][1].toFixed(3) + '</b>'; }).join(' · ');
      let rare = '';
      if (st.set === 'sparse') { const S0 = sparse(), cnt = new Int32Array(S0.V); S0.rows.forEach(function (r) { r.forEach(function (w) { cnt[w]++; }); }); let k = 0; cnt.forEach(function (x) { if (x > 0 && x < S0.n / 100) k++; }); rare = '<br>' + k.toLocaleString('en-US') + ' of the 3,000 words occur in fewer than 1 % of the reviews; each review uses about 120 different words'; }
      out.innerHTML = (done ? 'final' : 'epoch ' + Math.max.apply(null, jobs.map(function (J) { return J.epoch; })) + ':') + ' ' + fin + rare +
        '<br><span class="note">Toy numbers. On the sparse data SGD’s single learning rate is held down by the common words, so the rare, informative words learn slowly; AdaGrad and Adam give each word its own step. On the dense data the gaps are small.</span>';
    });
    Book.select(T.controls, 'data', [{ value: 'sparse', label: 'sparse (IMDB-like)' }, { value: 'dense', label: 'dense (MNIST-like)' }], st.set, function (v) { st.set = v; T.stop(); T.start(); });
  };

  W.fig1Redraw = function (host) {
    Book.head(host, 'Fig. 1 in numbers (approximate, read off the plots)',
      'The paper’s final training costs, read by eye off Figure 1 (so ± about 0.01). Pick a panel.');
    const P = {
      mnist: { t: 'MNIST, after 45 epochs', d: [['AdaGrad', 0.46, 'adagrad'], ['SGD Nesterov', 0.27, 'nesterov'], ['Adam', 0.26, 'adam']] },
      imdb: { t: 'IMDB bag of words + dropout, after ~170 epochs', d: [['AdaGrad', 0.25, 'adagrad'], ['RMSProp', 0.26, 'rmsprop'], ['SGD Nesterov', 0.35, 'nesterov'], ['Adam', 0.24, 'adam']] }
    };
    const st = { k: 'imdb' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'panel', [{ value: 'mnist', label: 'left: MNIST' }, { value: 'imdb', label: 'right: IMDB' }], st.k, function (v) { st.k = v; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const p = P[st.k];
      Book.chart.bars(chart, p.d.map(function (d) { return { label: d[0], value: d[1], note: '≈ ' + d[1].toFixed(2), color: Opt.colors[d[2]] }; }), { max: 0.5, height: 220, ylabel: 'training cost', tickFmt: function (v) { return v.toFixed(2); } });
      out.innerHTML = p.t + '. ' + (st.k === 'mnist' ? 'Adam and SGD with Nesterov momentum end close together; AdaGrad is far behind.' : 'AdaGrad, RMSProp and Adam end close together; SGD with Nesterov momentum is far behind and much noisier.') +
        '<br><span class="note">Values approximate, read off Paper Fig. 1. The paper plots full curves; this book does not reproduce the figure (arXiv licence).</span>';
    }
    draw();
  };

  // ================================================================== ch-mlp
  W.mlpRace = function (host) {
    Book.head(host, 'Fig. 2a, toy version: a multilayer net with dropout',
      'A 64-64-64-10 ReLU network with dropout (20 % of inputs, 50 % of hidden units dropped) learns the toy glyphs, trained live with the five optimizers of Fig. 2a for 40 epochs (about 10 seconds). The chart shows the training cost on a log scale.');
    const LIST = [['adagrad', { lr: 0.1 }], ['rmsprop', { lr: 0.006 }], ['nesterov', { lr: 0.03 }], ['adadelta', { lr: 2 }], ['adam', { lr: 0.006 }]];
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    trainer(host, 'Train', function () {
      return LIST.map(function (o) { const J = Opt.job(new Opt.MLP(mlpData(), [64, 64, 64, 10], 21, [0.2, 0.5]), o[0], o[1], { bs: 32, epochs: 40 }); J.key = o[0]; J.lr = o[1].lr; return J; });
    }, function (jobs, done) {
      AD.logLines(plot, jobs.map(function (J) { return { name: Opt.names[J.key], color: Opt.colors[J.key], marker: 'none', points: J.curve.slice(1) }; }),
        { width: 640, height: 250, xmin: 1, xmax: 40, ymin: 0.3, ymax: 3, xlabel: 'epochs', ylabel: 'training cost (toy, log)', xticks: [1, 10, 20, 30, 40], mr: 100 });
      out.innerHTML = (done ? 'after 40 epochs: ' : 'epoch ' + Math.min.apply(null, jobs.map(function (J) { return J.epoch; })) + '/40 · ') + jobs.map(function (J) { const c = J.curve[J.curve.length - 1]; return Opt.names[J.key] + ' <b>' + c[1].toFixed(3) + '</b>'; }).join(' · ') +
        '<br><span class="note">Toy numbers; the cost is measured with dropout switched off, on 400 training examples. Learning rates: each optimizer’s best from a small grid (AdaDelta’s “rate” multiplies its update; 1 is the original method). In the paper (Fig. 2a, 200 epochs) Adam ends lowest at ≈ 0.007, against ≈ 0.011–0.018 for the others (approximate, read off the plot).</span>';
    });
  };

  W.sfoCost = function (host) {
    Book.head(host, 'Adam vs SFO: iterations, wall-clock time and memory',
      'Fig. 2b compares Adam with the quasi-Newton method SFO on the deterministic objective. Adam reached a training cost of 10⁻⁴ after about 10 epochs, SFO after about 22 (read off the plot). The paper says SFO is 5–10× slower per iteration. Set that factor and the number of minibatch partitions.');
    const st = { k: 7, lN: 2.7 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'SFO slowdown per iteration', 5, 10, 0.5, st.k, function (v) { st.k = v; draw(); }, function (v) { return v + '×'; });
    Book.slider(ctl, 'minibatch partitions N', 0, 4, 0.1, st.lN, function (v) { st.lN = v; draw(); }, function (v) { return Math.round(Math.pow(10, v)).toLocaleString('en-US'); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const N = Math.round(Math.pow(10, st.lN)), tA = 10, tS = 22 * st.k;
      Book.chart.bars(chart, [{ label: 'Adam\nepochs to 10⁻⁴', value: 10, note: '≈ 10', color: Opt.colors.adam }, { label: 'SFO\nepochs to 10⁻⁴', value: 22, note: '≈ 22', color: Book.palette[0] },
        { label: 'Adam\ntime (Adam-epochs)', value: tA, note: '≈ ' + tA, color: Opt.colors.adam }, { label: 'SFO\ntime (Adam-epochs)', value: tS, note: '≈ ' + tS.toFixed(0), color: Book.palette[0] }], { height: 220, ylabel: 'cost', tickFmt: function (v) { return Math.round(v); } });
      out.innerHTML = 'wall-clock ratio ≈ (22/10) × ' + st.k + ' = <b>' + (2.2 * st.k).toFixed(1) + '×</b> in Adam’s favour' +
        '<br>memory beyond the parameters: Adam = 2 vectors (m and v) whatever N is · SFO keeps information for each of the N = ' + N.toLocaleString('en-US') + ' partitions, so its memory grows in proportion to N (paper §5, §6.2)' +
        '<br><span class="note">Epoch counts are approximate, read off Paper Fig. 2b (upper panel). The time estimate is the book’s arithmetic from those counts and the stated 5–10× slowdown; the paper’s own lower panel plots normalised wall time directly. With dropout noise SFO “failed to converge”, so Fig. 2a leaves it out.</span>';
    }
    draw();
  };

  // ================================================================== ch-cnn
  W.layerScales = function (host) {
    Book.head(host, 'Layers whose gradients differ by orders of magnitude',
      'Two “layers” of 10 weights each. Layer B’s gradients are s times layer A’s (weight sharing in convolutional layers can do this, paper §6.3). Compare SGD with one learning rate, SGD with a hand-picked rate per layer, and Adam with one α.');
    const st = { ls: -2 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'gradient ratio s (layer B / layer A)', -3, 0, 0.1, st.ls, function (v) { st.ls = v; draw(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const s = Math.pow(10, st.ls), r = ML.rng(3), h = ML.range(20).map(function () { return r.uniform(0.5, 2); });
      const curv = function (i) { return i < 10 ? h[i] : s * s * h[i]; };      // loss = Σ ½·curv_i·w_i², grads of B scaled by s (for equal w)
      const run = function (mk) {
        const A = Opt.vec(new Array(10).fill(1)), B = Opt.vec(new Array(10).fill(1)), o = mk([A, B]), L = [];
        const loss = function () { let x = 0; for (let i = 0; i < 10; i++) { x += 0.5 * curv(i) * A.data[i] * A.data[i] + 0.5 * curv(10 + i) / (s * s) * B.data[i] * B.data[i]; } return x; };
        for (let t = 0; t <= 300; t++) { L.push([t, loss()]); for (let i = 0; i < 10; i++) { A.grad[i] = curv(i) * A.data[i]; B.grad[i] = curv(10 + i) * B.data[i]; } o.step(); }
        return L;
      };
      // loss of layer B measured in its own units (divided by s²) so that both layers count equally
      const sgd = run(function (P) { return Opt.make('sgd', P, { lr: 0.9 }); });
      const per = run(function (P) { const a = Opt.make('sgd', [P[0]], { lr: 0.9 }), b = Opt.make('sgd', [P[1]], { lr: 0.9 / (s * s) }); return { step: function () { a.step(); b.step(); } }; });
      const adam = run(function (P) { return Opt.make('adam', P, { lr: 0.05 }); });
      AD.logLines(plot, [{ name: 'SGD, one rate', color: Opt.colors.sgd, marker: 'none', points: sgd }, { name: 'SGD, per-layer rates', color: Opt.colors.nesterov, marker: 'none', points: per, ldy: 14 }, { name: 'Adam, one α', color: Opt.colors.adam, marker: 'none', points: adam }],
        { width: 640, height: 240, xmin: 0, xmax: 300, ymin: 1e-6, ymax: 30, xlabel: 'step', ylabel: 'loss (log)', xticks: [0, 100, 200, 300], mr: 150 });
      out.innerHTML = 'after 300 steps: SGD one rate <b>' + AD.f(sgd[300][1]) + '</b> · SGD per-layer <b>' + AD.f(per[300][1]) + '</b> · Adam <b>' + AD.f(adam[300][1]) + '</b>' +
        '<br><span class="note">SGD’s single rate must suit layer A, so layer B crawls when s is small. Choosing a rate per layer by hand fixes it (common practice, the paper says). Adam rescales each weight by its own gradient size, so one α serves both layers. Toy problem.</span>';
    }
    draw();
  };

  W.epsFloor = function (host) {
    Book.head(host, 'When √v̂ falls below ε, Adam turns into SGD',
      'For a steady gradient g, Adam’s step is α·|g|/(|g| + ε). The paper noticed that in its convolutional nets v̂ “vanishes to zeros after a few epochs and is dominated by ε” (§6.3). Move ε and the gradient size.');
    const st = { lg: -6, le: -8 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'gradient size |g|', -12, 0, 0.25, st.lg, function (v) { st.lg = v; draw(); }, function (v) { return AD.p10(v); });
    Book.slider(ctl, 'ε', -12, -2, 0.5, st.le, function (v) { st.le = v; draw(); }, function (v) { return AD.p10(v); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const e = Math.pow(10, st.le), g = Math.pow(10, st.lg), xs = ML.linspace(-12, 0, 97);
      AD.lines(plot, [{ name: '|Δ|/α', color: 'var(--accent)', marker: 'none', points: xs.map(function (x) { const q = Math.pow(10, x); return [x, q / (q + e)]; }) }, { color: Book.palette[1], points: [[st.lg, g / (g + e)]], r: 6 }],
        { width: 640, height: 210, xmin: -12, xmax: 0, ymin: 0, ymax: 1.1, xticks: [-12, -10, -8, -6, -4, -2, 0], xfmt: function (v) { return AD.p10(v); }, xlabel: 'gradient size |g| (log)', ylabel: 'step ÷ α', mr: 60, ydec: 1 });
      const ratio = g / (g + e);
      out.innerHTML = '|Δ|/α = |g|/(|g| + ε) = <b>' + AD.f(ratio, 4) + '</b> · ' + (g < e / 10 ? 'ε dominates: the step is ≈ (α/ε)·g, plain SGD with learning rate α/ε = ' + AD.f(0.001 / e, 1) + ' for α = 0.001' : g > 10 * e ? 'ε is negligible: the step is ≈ α, Adam’s normal behaviour' : 'transition zone') +
        '<br><span class="note">So ε is not only “for numerical stability”: it sets the gradient size below which Adam stops normalising. The paper’s conclusion for CNNs: there, the first moment (averaging out minibatch noise) does more of the work than the second.</span>';
    }
    draw();
  };

  W.fig3Redraw = function (host) {
    Book.head(host, 'Fig. 3 in numbers (approximate, read off the plots)',
      'CIFAR-10, network c64-c64-c128-1000, six runs: three optimizers, each with and without dropout. Values read by eye off Paper Fig. 3.');
    const P = {
      e3: { t: 'after 3 epochs (left panel, linear axis)', log: false, d: [['AdaGrad', 1.22, 'adagrad'], ['AdaGrad\n+dropout', 1.49, 'adagrad'], ['SGDNesterov', 0.76, 'nesterov'], ['SGDNesterov\n+dropout', 0.92, 'nesterov'], ['Adam', 0.63, 'adam'], ['Adam\n+dropout', 0.70, 'adam']] },
      e45: { t: 'after 45 epochs (right panel, log axis)', log: true, d: [['AdaGrad', 0.6, 'adagrad'], ['AdaGrad\n+dropout', 0.75, 'adagrad'], ['SGDNesterov', 5.5e-4, 'nesterov'], ['SGDNesterov\n+dropout', 0.035, 'nesterov'], ['Adam', 4e-4, 'adam'], ['Adam\n+dropout', 0.025, 'adam']] }
    };
    const st = { k: 'e45' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'panel', [{ value: 'e3', label: 'first 3 epochs' }, { value: 'e45', label: '45 epochs' }], st.k, function (v) { st.k = v; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const p = P[st.k];
      Book.chart.bars(chart, p.d.map(function (d, i) { return { label: d[0], value: p.log ? Math.log10(d[1]) + 4 : d[1], note: '≈ ' + (d[1] < 0.01 ? AD.ft(d[1], 1) : d[1].toFixed(2)), color: AD.rgb(Opt.colors[d[2]], i % 2 ? 0.55 : 1) }; }),
        { max: p.log ? 4.5 : 1.6, height: 230, mb: 56, ylabel: 'training cost' + (p.log ? ' (log)' : ''), tickFmt: function (v) { return p.log ? AD.p10(v - 4) : v.toFixed(1); } });
      out.innerHTML = p.t + '. ' + (st.k === 'e3' ? 'Early on Adam leads, and AdaGrad starts quickly too.' : 'By the end Adam and SGD with Nesterov momentum are far below AdaGrad (about three orders of magnitude without dropout); Adam is slightly below SGD.') +
        '<br><span class="note">Approximate read-offs (± roughly 10–20 % on the log axis). Paler bars: with dropout.</span>';
    }
    draw();
  };

  // ================================================================== ch-vae: Fig. 4
  W.fig4Grid = function (host) {
    Book.head(host, 'Fig. 4, toy version: with (red) and without (green) bias correction',
      'The paper’s grid: β1 ∈ {0, 0.9} × β2 ∈ {0.99, 0.999, 0.9999} × log₁₀α from −5 to −1, each run with and without bias correction. Here the model is a small softmax regression on a 4×4-pixel, 4-class toy (not a VAE), trained live: 108 runs of 100 epochs (about 7 seconds).');
    const st = { ep: 10 };
    const LA = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1], B1 = [0, 0.9], B2 = [0.99, 0.999, 0.9999];
    const grid = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:4px' }, host), out = el('div', { class: 'out' }, host);
    const cells = []; B1.forEach(function () { B2.forEach(function () { cells.push(el('div', null, grid)); }); });
    let lastJobs = null;
    const T = trainer(host, 'Run the 108 trainings', function () {
      const D = vaeData(), jobs = [];
      B1.forEach(function (b1) { B2.forEach(function (b2) { LA.forEach(function (la) { [true, false].forEach(function (bias) {
        const J = Opt.job(new Opt.SoftmaxReg(D, 9, 1e-3), 'adam', { lr: Math.pow(10, la), b1: b1, b2: b2, bias: bias, lambda: 1 }, { bs: 25, epochs: 100 });
        J.b1 = b1; J.b2 = b2; J.la = la; J.bias = bias; jobs.push(J);
      }); }); }); });
      return jobs;
    }, function (jobs, done) { lastJobs = jobs; draw(done); });
    Book.select(T.controls, 'show', [{ value: '10', label: 'after 10 epochs (Fig. 4a)' }, { value: '100', label: 'after 100 epochs (Fig. 4b)' }], '10', function (v) { st.ep = +v; draw(true); });
    function draw(done) {
      if (!lastJobs) return;
      const CAP = 1.6; let k = 0, worstNo = 0, worstYes = 0;
      B1.forEach(function (b1) { B2.forEach(function (b2) {
        const mine = lastJobs.filter(function (J) { return J.b1 === b1 && J.b2 === b2; });
        const ser = [true, false].map(function (bias) {
          return { color: bias ? '#c0392b' : '#2f855a', r: 2.6, width: 1.6, points: mine.filter(function (J) { return J.bias === bias && J.curve.length > st.ep; }).map(function (J) { const v = J.curve[st.ep][1]; if (bias) worstYes = Math.max(worstYes, v); else worstNo = Math.max(worstNo, v); return [J.la, Math.min(v, CAP)]; }) };
        });
        const host2 = cells[k++]; host2.innerHTML = '';
        el('div', { class: 'note', style: 'text-align:center', text: 'β1 = ' + b1 + ', β2 = ' + b2 }, host2);
        const c2 = el('div', null, host2);
        if (ser[0].points.length || ser[1].points.length) AD.lines(c2, ser.filter(function (s) { return s.points.length; }), { width: 230, height: 170, xmin: -5, xmax: -1, ymin: 0.2, ymax: CAP, xticks: [-5, -3, -1], ml: 34, mr: 8, ydec: 1 });
      }); });
      out.innerHTML = (done ? '' : 'training… ') + 'loss after ' + st.ep + ' epochs (capped at ' + CAP + ') against log₁₀α · red: with bias correction · green: without' +
        (done ? '<br>worst loss over the grid: with correction <b>' + worstYes.toFixed(2) + '</b> · without <b>' + worstNo.toFixed(2) + '</b>' : '') +
        '<br><span class="note">Toy numbers. Look at β2 = 0.9999 and large α: without correction the first steps are up to 100× too big (Chapter 18) and the loss jumps; with correction it does not. At small α, where the inflated early steps happen to help, the uncorrected runs can even look slightly better on this easy convex toy. The paper’s VAE shows the damage far more strongly.</span>';
    }
  };

  // ================================================================== ch-next: Reddi et al.'s counterexample
  W.reddiCounter = function (host) {
    Book.head(host, 'A convex problem where Adam goes the wrong way (later work)',
      'Reddi, Kale & Kumar (2018): on θ ∈ [−1, 1], the cost is C·θ once every three steps and −θ on the other two. Summed over three steps that is (C − 2)·θ, so the best point is θ = −1. With β1 = 0 and β2 = 1/(1 + C²), Adam drifts to +1. AMSGrad, which never lets v̂ shrink, does not.');
    const st = { C: 3, b2: 'reddi', a: 0.5 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'C', 2.2, 10, 0.1, st.C, function (v) { st.C = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.select(ctl, 'β2', [{ value: 'reddi', label: '1/(1 + C²) (their choice)' }, { value: '0.999', label: '0.999 (default)' }], st.b2, function (v) { st.b2 = v; draw(); });
    const g2 = el('div', { class: 'grid2' }, host), p1 = el('div', null, g2), p2 = el('div', null, g2), out = el('div', { class: 'out' }, host);
    function draw() {
      const T = 6000, b2 = st.b2 === 'reddi' ? 1 / (1 + st.C * st.C) : 0.999;
      const A = Opt.reddi(st.C, T, 'adam', { lr: st.a, b1: 0, b2: b2, decay: true }), M = Opt.reddi(st.C, T, 'adam', { lr: st.a, b1: 0, b2: b2, decay: true, amsgrad: true });
      const pick = function (arr, f) { const o = []; for (let t = 0; t < T; t += 30) o.push([t + 1, f(arr[t], t)]); return o; };
      AD.lines(p1, [{ name: 'Adam', color: Opt.colors.adam, marker: 'none', points: pick(A.x, function (x) { return x; }) }, { name: 'AMSGrad', color: Book.palette[2], marker: 'none', points: pick(M.x, function (x) { return x; }), ldy: 14 }],
        { width: 320, height: 210, ymin: -1.05, ymax: 1.05, xlabel: 'step', ylabel: 'θ', xticks: [1, 3000, 6000], mr: 70, ydec: 1 });
      AD.lines(p2, [{ name: 'Adam', color: Opt.colors.adam, marker: 'none', points: pick(A.R, function (r, t) { return r / (t + 1); }) }, { name: 'AMSGrad', color: Book.palette[2], marker: 'none', points: pick(M.R, function (r, t) { return r / (t + 1); }), ldy: 14 }],
        { width: 320, height: 210, ymin: 0, xlabel: 'step', ylabel: 'R(T)/T', xticks: [1, 3000, 6000], mr: 70, ydec: 2 });
      out.innerHTML = 'β2 = ' + b2.toPrecision(3) + ' · after 6,000 steps: Adam θ = <b>' + A.x[T - 1].toFixed(3) + '</b>, R/T = <b>' + (A.R[T - 1] / T).toFixed(3) + '</b> · AMSGrad θ = <b>' + M.x[T - 1].toFixed(3) + '</b>, R/T = <b>' + (M.R[T - 1] / T).toFixed(3) + '</b>' +
        '<br><span class="note">The big gradient C arrives rarely, and with a short memory (small β2) v̂ forgets it before it comes back; the two small −1 gradients then get large normalised steps towards +1. Average regret stays above 0, contradicting Corollary 4.2: the proof’s gap (Chapter 23, stage 5). With the default β2 = 0.999 Adam remembers C and finds −1 on this instance, but Reddi et al. show that, for any constant β1 < √β2, some (stochastic) convex problem defeats it.</span>';
    }
    draw();
  };
})();
