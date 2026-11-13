/* The toy world of the Dropout Interactive Web Book: "pixel digits" and small networks that train in the browser.
   Uses the shared libraries ML (math.js: seeded random numbers) and NN (nn.js: autograd, Linear, SGD, NN.dropout).

   • Pixel digits: ten hand-drawn 5×5 digit shapes placed on a 7×7 grid, shifted by up to one pixel, with broken
     strokes, speckles and grey-level noise. 49 inputs, 10 classes. A stand-in for MNIST (28×28 = 784 inputs).
   • Toy.MLP: a fully connected network trained exactly as §5.1 of the paper describes: minibatch SGD with
     momentum, optional learning-rate decay ε0(1+t/T)^−1, L2 / L1 penalties, max-norm projection, and dropout
     (Bernoulli, via NN.dropout, or multiplicative Gaussian, §10) on the input and hidden layers.
     NN.dropout is the "inverted" form (scale kept units by 1/p while training, change nothing at test). §10 of
     the paper explains that it is equivalent to the paper's form (no scaling while training, weights × p at test);
     Toy.MLP.prototype.paperWeights converts one into the other so widgets can show both.
   • Toy.AE: a one-hidden-layer ReLU autoencoder (Fig 7, Fig 8).
   • Toy.RBM: a binary restricted Boltzmann machine trained with CD-1, with optional dropout of hidden units (§8).
   Everything is seeded, so the same settings always give the same numbers. Exposed as window.Toy. */
(function () {
  'use strict';
  const ML = window.ML, NN = window.NN;
  const Toy = window.Toy = {};

  // ------------------------------------------------------------------ pixel digits
  const GLYPHS = [
    ['.###.', '#...#', '#...#', '#...#', '.###.'],
    ['..#..', '.##..', '..#..', '..#..', '.###.'],
    ['####.', '....#', '.###.', '#....', '#####'],
    ['####.', '....#', '.###.', '....#', '####.'],
    ['#..#.', '#..#.', '#####', '...#.', '...#.'],
    ['#####', '#....', '####.', '....#', '####.'],
    ['.###.', '#....', '####.', '#...#', '.###.'],
    ['#####', '....#', '...#.', '..#..', '..#..'],
    ['.###.', '#...#', '.###.', '#...#', '.###.'],
    ['.###.', '#...#', '.####', '....#', '.###.']
  ];
  Toy.SIDE = 7; Toy.D = 49; Toy.K = 10;
  Toy.GLYPHS = GLYPHS;
  // one clean digit, centred, as 49 numbers
  Toy.clean = function (k) {
    const x = ML.zeros(49);
    for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) if (GLYPHS[k][r][c] === '#') x[(r + 1) * 7 + c + 1] = 1;
    return x;
  };
  // one noisy example of digit k
  // noise settings: sx/sy allow a one-pixel shift, brk = chance a stroke pixel is missing, spk = chance of a stray
  // speckle, g = standard deviation of the grey-level noise added to every pixel
  Toy.NOISE = { sx: 0, sy: 0, brk: 0.15, spk: 0.1, g: 0.15 };
  Toy.sample = function (k, rng, o) {
    o = o || Toy.NOISE;
    const dx = o.sx ? rng.int(3) - 1 : 0, dy = o.sy ? rng.int(3) - 1 : 0, x = ML.zeros(49);
    for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
      if (GLYPHS[k][r][c] !== '#') continue;
      if (rng.next() < o.brk) continue;                             // broken stroke
      x[(r + 1 + dy) * 7 + c + 1 + dx] = rng.uniform(0.55, 1);
    }
    for (let i = 0; i < 49; i++) {
      if (rng.next() < o.spk) x[i] = Math.max(x[i], rng.uniform(0.3, 0.9));   // speckle
      x[i] = clamp(x[i] + rng.normal(0, o.g), 0, 1);
    }
    return x;
  };
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  // a data set: n examples, classes in turn (so every class is equally common), from seed
  Toy.digits = function (n, seed, noise) {
    const rng = ML.rng(seed), X = [], y = [];
    for (let i = 0; i < n; i++) { const k = i % 10; X.push(Toy.sample(k, rng, noise)); y.push(k); }
    // shuffle together
    const idx = rng.shuffle(ML.range(n));
    return { X: idx.map(function (i) { return X[i]; }), y: idx.map(function (i) { return y[i]; }) };
  };
  // the standard toy split used across the book (cached): a small training set and a large test set
  Toy.data = function (nTrain) {
    nTrain = nTrain || 500;
    Toy._cache = Toy._cache || {};
    if (!Toy._cache['tr' + nTrain]) Toy._cache['tr' + nTrain] = Toy.digits(nTrain, 1000 + nTrain);
    if (!Toy._cache.te) Toy._cache.te = Toy.digits(1000, 77);
    return { train: Toy._cache['tr' + nTrain], test: Toy._cache.te };
  };
  // error of the best possible "template" rule (nearest clean digit): a yardstick for how hard the toy is
  Toy.templateError = function (data) {
    let wrong = 0;
    data.X.forEach(function (x, i) { const d = ML.range(10).map(function (k) { return ML.dist2(x, Toy.clean(k)); }); if (ML.argmin(d) !== data.y[i]) wrong++; });
    return 100 * wrong / data.X.length;
  };

  // ------------------------------------------------------------------ the book's toy experiments
  // Shared by the widgets (Re-run live) and by the script that produced toy-results.js, so both give the same numbers.
  const BASE = { sizes: [49, 96, 96, 10], nTrain: 500, steps: 3000, batch: 20, lr: 0.005, m0: 0.5, m1: 0.9, mT: 200, evalEvery: 250 };
  function cfg(o) { return Object.assign({}, BASE, o); }
  Toy.BASE = BASE;
  Toy.EXPERIMENTS = {
    // Fig 4: several architectures, the same hyperparameters, with and without dropout
    fig4: function () {
      const out = [];
      [[96], [96, 96], [160, 160], [96, 96, 96]].forEach(function (h, a) {
        [false, true].forEach(function (d) {
          out.push(cfg({ label: h.join('-') + (d ? ' dropout' : ' standard'), sizes: [49].concat(h, [10]), steps: 4000, seed: 11 + a,
            keepIn: d ? 0.8 : 1, keepHid: d ? 0.5 : 1, nTrain: 1000, evalEvery: 200 }));
        });
      });
      return out;
    },
    // Fig 9: retention probability p, (a) n fixed, (b) p·n fixed
    fig9: function () {
      const out = [], ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
      ps.forEach(function (p) { out.push(cfg({ label: 'n fixed p=' + p, group: 'a', p: p, sizes: [49, 96, 96, 10], keepHid: p })); });
      ps.forEach(function (p) { const n = Math.round(24 / p); out.push(cfg({ label: 'pn fixed p=' + p, group: 'b', p: p, sizes: [49, n, n, 10], keepHid: p })); });
      return out;
    },
    // Fig 10: training-set size
    fig10: function () {
      const out = [];
      [50, 100, 200, 500, 1000, 2000].forEach(function (n) {
        [false, true].forEach(function (d) { out.push(cfg({ label: n + (d ? ' dropout' : ' standard'), n: n, dropout: d, nTrain: n, keepIn: d ? 0.8 : 1, keepHid: d ? 0.5 : 1 })); });
      });
      return out;
    },
    // Table 9: the same network with different regularizers (strengths set by hand for the toy)
    table9: function () {
      return [
        cfg({ label: 'L2', l2: 1e-3 }),
        cfg({ label: 'L2 + L1', l2: 1e-3, l1: 1e-4 }),
        cfg({ label: 'Max-norm', maxnorm: 2 }),
        cfg({ label: 'Dropout + L2', l2: 1e-3, keepIn: 0.8, keepHid: 0.5 }),
        cfg({ label: 'Dropout + Max-norm', maxnorm: 2, keepIn: 0.8, keepHid: 0.5 }),
        cfg({ label: 'None', })
      ];
    },
    // Table 10: Bernoulli vs Gaussian multiplicative noise with matched variance, 3 seeds each
    table10: function () {
      const out = [];
      [1, 2, 3].forEach(function (s) {
        out.push(cfg({ label: 'Bernoulli seed ' + s, kind: 'bernoulli', seed: s, keepIn: 0.8, keepHid: 0.5 }));
        out.push(cfg({ label: 'Gaussian seed ' + s, kind: 'gauss', seed: s, keepIn: 0.8, keepHid: 0.5, noise: 'gauss' }));
      });
      return out;
    }
  };
  // summary of a finished trainer: final train/test error and the mean test error of the last 3 evaluations
  Toy.summary = function (T) {
    const c = T.curve, tail = c.slice(-3).map(function (r) { return r[2]; });
    return { label: T.cfg.label, train: +c[c.length - 1][1].toFixed(2), test: +(tail.reduce(function (a, b) { return a + b; }, 0) / tail.length).toFixed(2),
      curve: c.map(function (r) { return [r[0], +r[1].toFixed(1), +r[2].toFixed(1)]; }) };
  };

  // ------------------------------------------------------------------ multi-layer perceptron
  // sizes: [49, h1, h2, …, 10]. opts: act ('relu' | 'logistic'), keepIn, keepHid (the paper's p: probability of
  // RETAINING a unit; 1 = no dropout), noise ('bernoulli' | 'gauss'), seed, sd (init scale, default 1/sqrt(fan-in)).
  Toy.MLP = function (sizes, opts) {
    opts = opts || {};
    this.sizes = sizes.slice();
    this.act = opts.act || 'relu';
    this.keepIn = opts.keepIn == null ? 1 : opts.keepIn;
    this.keepHid = opts.keepHid == null ? 1 : opts.keepHid;
    this.noise = opts.noise || 'bernoulli';
    const rng = ML.rng(opts.seed || 1);
    this.layers = [];
    for (let i = 0; i + 1 < sizes.length; i++) this.layers.push(new NN.Linear(sizes[i], sizes[i + 1], rng, { sd: opts.sd != null ? opts.sd / Math.sqrt(sizes[i]) : null }));
    this.params = [];
    const P = this.params; this.layers.forEach(function (L) { L.params.forEach(function (p) { P.push(p); }); });
    this.rng = ML.rng((opts.seed || 1) * 7919 + 13);
    this.steps = 0;
  };
  function actf(name, t) { return name === 'logistic' ? NN.sigmoid(t) : name === 'tanh' ? NN.tanh(t) : NN.relu(t); }
  // drop or perturb the units of tensor h (keep = the paper's p). Returns h unchanged when not training.
  Toy.noisy = function (h, keep, rng, training, kind) {
    if (!training || keep >= 1) return h;
    if (kind === 'gauss') {                                   // §10: multiply by r ~ N(1, σ²), σ² = (1 − p)/p
      const sd = Math.sqrt((1 - keep) / keep), m = new NN.Tensor(h.rows, h.cols);
      for (let i = 0; i < m.data.length; i++) m.data[i] = 1 + sd * rng.normal(0, 1);
      const out = NN.mul(h, m); out.mask = m.data; return out;
    }
    return NN.dropout(h, 1 - keep, rng, true);                // NN.dropout takes the DROP probability 1 − p
  };
  // forward pass. X: Tensor (batch × 49). o.train: apply dropout; o.rng: random source; o.keep: [keepIn, keepHid]
  // override; o.record: array that receives every hidden activation tensor (for widgets).
  Toy.MLP.prototype.forward = function (X, o) {
    o = o || {};
    const train = !!o.train, rng = o.rng || this.rng;
    const kIn = o.keepIn != null ? o.keepIn : this.keepIn, kH = o.keepHid != null ? o.keepHid : this.keepHid;
    let h = Toy.noisy(X, kIn, rng, train, this.noise);
    for (let i = 0; i < this.layers.length; i++) {
      h = this.layers[i].forward(h);
      if (i + 1 < this.layers.length) {
        h = actf(this.act, h);
        if (o.record) o.record.push(h);
        h = Toy.noisy(h, kH, rng, train, this.noise);
      }
    }
    return h;   // logits
  };
  // class probabilities (no dropout unless o.train), as number[][]
  Toy.MLP.prototype.probs = function (Xarr, o) {
    const logits = this.forward(NN.tensor(Xarr), o), out = [];
    for (let i = 0; i < logits.rows; i++) out.push(ML.softmax(logits.row(i)));
    return out;
  };
  Toy.MLP.prototype.error = function (data, o) {
    const P = this.probs(data.X, o);
    let wrong = 0; for (let i = 0; i < P.length; i++) if (ML.argmax(P[i]) !== data.y[i]) wrong++;
    return 100 * wrong / P.length;
  };
  // the weights in the PAPER's convention (no 1/p while training, so weights leaving a layer are multiplied by p at
  // test time). Our training is inverted dropout, so the paper's training-time weights are W/p_in(layer) and its
  // test-time weights are p · that = W. Returns {train: [...], test: [...]} of number[][] (fan-in × fan-out).
  Toy.MLP.prototype.paperWeights = function () {
    const self = this;
    return this.layers.map(function (L, i) {
      const p = i === 0 ? self.keepIn : self.keepHid, W = L.W.toArray();
      return { train: W.map(function (r) { return r.map(function (v) { return v / p; }); }), test: W };
    });
  };
  // one training step on a minibatch (arrays). opt fields: lr, momentum, l2, l1, maxnorm (c, 0 = off).
  Toy.MLP.prototype.step = function (Xb, yb, opt) {
    if (!this.sgd) this.sgd = new NN.SGD(this.params, { lr: opt.lr, momentum: opt.momentum || 0 });
    this.sgd.lr = opt.lr; this.sgd.momentum = opt.momentum || 0;
    NN.zeroGrad(this.params);
    const L = NN.softmaxCE(this.forward(NN.tensor(Xb), { train: true }), yb);
    L.backward();
    const self = this;
    if (opt.l2 || opt.l1) this.layers.forEach(function (Ly) {
      const W = Ly.W;
      for (let i = 0; i < W.data.length; i++) W.grad[i] += (opt.l2 || 0) * W.data[i] + (opt.l1 || 0) * Math.sign(W.data[i]);
    });
    this.sgd.step();
    if (opt.maxnorm) this.layers.forEach(function (Ly) { Toy.maxNorm(Ly.W, opt.maxnorm); });
    self.steps++;
    return L.item();
  };
  // max-norm (§5.1): project each unit's vector of INCOMING weights back onto the ball of radius c.
  // In NN.Linear, W is fan-in × fan-out, so unit j's incoming weights are column j (deepnet: norm_limit(…, axis=0)).
  Toy.maxNorm = function (W, c) {
    const r = W.rows, k = W.cols;
    for (let j = 0; j < k; j++) {
      let s = 0; for (let i = 0; i < r; i++) s += W.data[i * k + j] * W.data[i * k + j];
      const n = Math.sqrt(s);
      if (n > c) { const f = c / n; for (let i = 0; i < r; i++) W.data[i * k + j] *= f; }
    }
  };
  // learning rate and momentum schedules, exactly as deepnet's parameter.py GetMomentumAndEpsilon
  Toy.lrAt = function (t, base, T) { return T ? base / (1 + t / T) : base; };
  Toy.momentumAt = function (t, m0, m1, Tm) { return Tm ? m1 - (m1 - m0) * Math.exp(-t / Tm) : m1; };

  // A resumable trainer: call .run(k) to do k more steps; .curve holds [step, train error, test error].
  // cfg: {sizes, act, keepIn, keepHid, noise, seed, nTrain, steps, batch, lr, lrT, m0, m1, mT, l2, l1, maxnorm, evalEvery}
  Toy.Trainer = function (cfg) {
    this.cfg = Object.assign({ sizes: [49, 64, 64, 10], act: 'relu', keepIn: 1, keepHid: 1, seed: 1, nTrain: 200, steps: 1500, batch: 20,
      lr: 0.05, lrT: 0, m0: 0.5, m1: 0.9, mT: 200, l2: 0, l1: 0, maxnorm: 0, evalEvery: 50 }, cfg || {});
    const c = this.cfg;
    this.data = c.data || Toy.data(c.nTrain);
    this.net = new Toy.MLP(c.sizes, { act: c.act, keepIn: c.keepIn, keepHid: c.keepHid, noise: c.noise, seed: c.seed, sd: c.sd });
    if (c.init) c.init(this.net);
    this.order = ML.rng(c.seed * 31 + 7);
    this.t = 0; this.curve = []; this.losses = [];
    this.perm = []; this.pos = 0;
  };
  Toy.Trainer.prototype.nextBatch = function () {
    const n = this.data.train.X.length, b = Math.min(this.cfg.batch, n), Xb = [], yb = [];
    for (let i = 0; i < b; i++) {
      if (this.pos >= this.perm.length) { this.perm = this.order.shuffle(ML.range(n)); this.pos = 0; }
      const j = this.perm[this.pos++]; Xb.push(this.data.train.X[j]); yb.push(this.data.train.y[j]);
    }
    return { X: Xb, y: yb };
  };
  Toy.Trainer.prototype.evaluate = function () {
    const tr = this.net.error(this.data.train), te = this.net.error(this.data.test);
    this.curve.push([this.t, tr, te]);
    return { train: tr, test: te };
  };
  Toy.Trainer.prototype.done = function () { return this.t >= this.cfg.steps; };
  Toy.Trainer.prototype.run = function (k) {
    const c = this.cfg;
    if (this.t === 0 && !this.curve.length) this.evaluate();
    for (let i = 0; i < k && this.t < c.steps; i++) {
      const b = this.nextBatch();
      const loss = this.net.step(b.X, b.y, { lr: Toy.lrAt(this.t, c.lr, c.lrT), momentum: Toy.momentumAt(this.t, c.m0, c.m1, c.mT), l2: c.l2, l1: c.l1, maxnorm: c.maxnorm });
      this.t++;
      if (this.t % 10 === 0) this.losses.push([this.t, loss]);
      if (this.t % c.evalEvery === 0 || this.t === c.steps) this.evaluate();
    }
    return this;
  };
  Toy.Trainer.prototype.last = function () { const r = this.curve[this.curve.length - 1]; return { step: r[0], train: r[1], test: r[2] }; };
  // train to the end synchronously (for small jobs)
  Toy.train = function (cfg) { const T = new Toy.Trainer(cfg); while (!T.done()) T.run(200); return T; };

  // A queue of trainers run in small time slices so the page never freezes. onProgress(i, trainer) after each slice;
  // onDone() at the end. Returns {stop()}.
  Toy.runJobs = function (cfgs, onProgress, onDone, sliceMs) {
    sliceMs = sliceMs || 30;
    const jobs = cfgs.map(function (c) { return new Toy.Trainer(c); });
    let i = 0, stopped = false, timer = null;
    function tick() {
      if (stopped) return;
      const t0 = performance.now();
      while (i < jobs.length && performance.now() - t0 < sliceMs) {
        jobs[i].run(5);
        if (jobs[i].done()) { onProgress && onProgress(i, jobs[i], true); i++; }
      }
      if (i < jobs.length) { onProgress && onProgress(i, jobs[i], false); timer = setTimeout(tick, 0); }
      else onDone && onDone(jobs);
    }
    timer = setTimeout(tick, 0);
    return { jobs: jobs, stop: function () { stopped = true; if (timer) clearTimeout(timer); } };
  };

  // ------------------------------------------------------------------ autoencoder (Fig 7, Fig 8)
  // 49 → h ReLU units → 49 linear outputs, squared error; optional dropout (keep p) on the hidden layer.
  Toy.AE = function (h, keep, seed) {
    const rng = ML.rng(seed || 3);
    this.enc = new NN.Linear(49, h, rng, { sd: 0.05 }); this.dec = new NN.Linear(h, 49, rng, { sd: 0.05 });
    this.enc.b.data.fill(0.2);                                  // small positive bias: no ReLU starts dead
    this.params = this.enc.params.concat(this.dec.params);
    this.keep = keep == null ? 1 : keep; this.h = h;
    this.rng = ML.rng((seed || 3) * 101 + 1);
    this.opt = new NN.SGD(this.params, { lr: 0.1, momentum: 0.9 });
  };
  Toy.AE.prototype.forward = function (X, train) {
    const H = NN.relu(this.enc.forward(X));
    const Hd = Toy.noisy(H, this.keep, this.rng, train, 'bernoulli');
    return { H: H, out: this.dec.forward(Hd) };
  };
  Toy.AE.prototype.step = function (Xb) {
    NN.zeroGrad(this.params);
    const X = NN.tensor(Xb), L = NN.mse(this.forward(X, true).out, X);
    L.backward(); this.opt.step();
    Toy.maxNorm(this.enc.W, 4); Toy.maxNorm(this.dec.W, 4);
    return L.item();
  };
  Toy.AE.prototype.recError = function (Xarr) { const X = NN.tensor(Xarr); return NN.mse(this.forward(X, false).out, X).item() * 49; };
  // hidden activations at test time (no dropout) for a batch: number[][] (examples × units)
  Toy.AE.prototype.hidden = function (Xarr) { return this.forward(NN.tensor(Xarr), false).H.toArray(); };
  // unit j's feature: its 49 incoming weights
  Toy.AE.prototype.feature = function (j) { const W = this.enc.W, out = []; for (let i = 0; i < 49; i++) out.push(W.data[i * this.h + j]); return out; };

  // ------------------------------------------------------------------ restricted Boltzmann machine (§8)
  // Binary visible v ∈ {0,1}^49, binary hidden h ∈ {0,1}^F. Energy −(vᵀWh + aᵀh + bᵀv) with a = hidden biases,
  // b = visible biases (the paper's §8.1 energy). CD-1 learning; with keep < 1 a fresh dropout vector r is drawn
  // for every training case and the dropped hidden units are clamped to 0 in both phases (§8.2).
  Toy.RBM = function (F, keep, seed) {
    const rng = ML.rng(seed || 5);
    this.F = F; this.keep = keep == null ? 1 : keep; this.rng = ML.rng((seed || 5) * 17 + 3);
    this.W = ML.mat(49, F, function () { return rng.normal(0, 0.05); });
    this.a = ML.zeros(F); this.b = ML.zeros(49);
  };
  Toy.RBM.prototype.hidProb = function (v, r) {   // P(h_j = 1 | r_j, v) = 1(r_j = 1) σ(a_j + Σ_i W_ij v_i)
    const F = this.F, out = new Array(F);
    for (let j = 0; j < F; j++) {
      if (r && !r[j]) { out[j] = 0; continue; }
      let s = this.a[j]; for (let i = 0; i < 49; i++) s += v[i] * this.W[i][j];
      out[j] = ML.sigmoid(s);
    }
    return out;
  };
  Toy.RBM.prototype.visProb = function (h) {      // P(v_i = 1 | h) = σ(b_i + Σ_j W_ij h_j)
    const out = new Array(49);
    for (let i = 0; i < 49; i++) { let s = this.b[i]; const Wi = this.W[i]; for (let j = 0; j < this.F; j++) s += Wi[j] * h[j]; out[i] = ML.sigmoid(s); }
    return out;
  };
  // one CD-1 update on a minibatch of binary vectors; returns mean reconstruction error (squared, per image)
  Toy.RBM.prototype.cd1 = function (batch, lr) {
    const F = this.F, n = batch.length, rng = this.rng, self = this;
    const dW = ML.mat(49, F), da = ML.zeros(F), db = ML.zeros(49);
    let err = 0;
    batch.forEach(function (v) {
      let r = null;
      if (self.keep < 1) { r = new Array(F); for (let j = 0; j < F; j++) r[j] = rng.next() < self.keep ? 1 : 0; }
      const ph = self.hidProb(v, r), h = ph.map(function (q) { return rng.next() < q ? 1 : 0; });
      const pv = self.visProb(h), ph2 = self.hidProb(pv, r);
      for (let i = 0; i < 49; i++) {
        err += (v[i] - pv[i]) * (v[i] - pv[i]);
        for (let j = 0; j < F; j++) dW[i][j] += v[i] * ph[j] - pv[i] * ph2[j];
        db[i] += v[i] - pv[i];
      }
      for (let j = 0; j < F; j++) da[j] += ph[j] - ph2[j];
    });
    for (let i = 0; i < 49; i++) { for (let j = 0; j < F; j++) this.W[i][j] += lr * dW[i][j] / n; this.b[i] += lr * db[i] / n; }
    for (let j = 0; j < F; j++) this.a[j] += lr * da[j] / n;
    return err / n;
  };
  // at test time every hidden unit is present and its weights are scaled by p (the RBM version of Fig 2)
  Toy.RBM.prototype.testHid = function (v) {
    const F = this.F, out = new Array(F);
    for (let j = 0; j < F; j++) { let s = this.a[j]; for (let i = 0; i < 49; i++) s += v[i] * this.W[i][j]; out[j] = ML.sigmoid(s); }
    return out;
  };
  Toy.RBM.prototype.feature = function (j) { return this.W.map(function (row) { return row[j]; }); };
  Toy.binarize = function (X, seed) { const r = ML.rng(seed || 9); return X.map(function (x) { return x.map(function (v) { return r.next() < v ? 1 : 0; }); }); };
})();
