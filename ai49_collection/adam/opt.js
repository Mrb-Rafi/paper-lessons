/* Toy library of the Adam Interactive Web Book (lesson 3 of Modern AI in 49 Papers): window.Opt.
   Every optimizer of the book, the 2-D test surfaces, two small data sets (dense "glyphs" standing in for MNIST,
   sparse "reviews" standing in for IMDB bag-of-words), fast trainers for (sparse) logistic regression, a small
   multilayer network built on the shared autograd (NN), and the online convex games of the convergence chapters.
   Optimizers work on NN-style parameters {data: Float64Array, grad: Float64Array}, like NN.SGD and NN.Adam.
   "adam" with default options IS the shared NN.Adam (Algorithm 1 of the paper); the variants (no bias correction,
   decaying α_t = α/√t, decaying β1,t, AMSGrad) are written out below and checked against it in _work/test.html. */
(function () {
  'use strict';
  const ML = window.ML, NN = window.NN;
  const Opt = window.Opt = {};

  // ------------------------------------------------------------------ parameters
  Opt.vec = function (arr) { const d = Float64Array.from(arr); return { data: d, grad: new Float64Array(d.length), rows: 1, cols: d.length }; };
  const each = function (params, f) { params.forEach(function (p, k) { if (p.grad) f(p, k); }); };
  const zerosLike = function (params) { return params.map(function (p) { return new Float64Array(p.data.length); }); };

  // ------------------------------------------------------------------ optimizers
  // Each returns an object with .step(), .t, .lr (settable) and its state arrays (for widgets to show).
  const makers = {
    // plain SGD and classical momentum: the shared NN.SGD (v ← μv + g; θ ← θ − lr·v)
    sgd: function (P, o) { return new NN.SGD(P, { lr: o.lr, momentum: 0 }); },
    momentum: function (P, o) { return new NN.SGD(P, { lr: o.lr, momentum: o.mu == null ? 0.9 : o.mu }); },
    // Nesterov momentum in the "look-ahead folded in" form used by PyTorch's SGD(nesterov=True)
    nesterov: function (P, o) {
      const s = { lr: o.lr, mu: o.mu == null ? 0.9 : o.mu, t: 0, buf: zerosLike(P), params: P };
      s.step = function () { s.t++; each(P, function (p, k) { const b = s.buf[k]; for (let i = 0; i < b.length; i++) { b[i] = s.mu * b[i] + p.grad[i]; p.data[i] -= s.lr * (p.grad[i] + s.mu * b[i]); } }); };
      return s;
    },
    // AdaGrad (Duchi et al., 2011; paper §5): θ ← θ − α·g/√(Σ g²)
    adagrad: function (P, o) {
      const s = { lr: o.lr, eps: o.eps == null ? 1e-8 : o.eps, t: 0, G: zerosLike(P), params: P };
      s.step = function () { s.t++; each(P, function (p, k) { const G = s.G[k]; for (let i = 0; i < G.length; i++) { const g = p.grad[i]; G[i] += g * g; p.data[i] -= s.lr * g / (Math.sqrt(G[i]) + s.eps); } }); };
      return s;
    },
    // RMSProp (Tieleman & Hinton, 2012); with mu > 0, momentum on the rescaled gradient (Graves, 2013), as in paper §5
    rmsprop: function (P, o) {
      const s = { lr: o.lr, rho: o.rho == null ? 0.9 : o.rho, mu: o.mu || 0, eps: o.eps == null ? 1e-8 : o.eps, t: 0, v: zerosLike(P), buf: zerosLike(P), params: P };
      s.step = function () {
        s.t++;
        each(P, function (p, k) {
          const v = s.v[k], b = s.buf[k];
          for (let i = 0; i < v.length; i++) {
            const g = p.grad[i]; v[i] = s.rho * v[i] + (1 - s.rho) * g * g;
            const r = g / (Math.sqrt(v[i]) + s.eps);
            if (s.mu) { b[i] = s.mu * b[i] + r; p.data[i] -= s.lr * b[i]; } else p.data[i] -= s.lr * r;
          }
        });
      };
      return s;
    },
    // AdaDelta (Zeiler, 2012): no learning rate in the original; lr multiplies the update (1 = original)
    adadelta: function (P, o) {
      const s = { lr: o.lr == null ? 1 : o.lr, rho: o.rho == null ? 0.95 : o.rho, eps: o.eps == null ? 1e-6 : o.eps, t: 0, Eg: zerosLike(P), Ex: zerosLike(P), params: P };
      s.step = function () {
        s.t++;
        each(P, function (p, k) {
          const Eg = s.Eg[k], Ex = s.Ex[k];
          for (let i = 0; i < Eg.length; i++) {
            const g = p.grad[i]; Eg[i] = s.rho * Eg[i] + (1 - s.rho) * g * g;
            const dx = -Math.sqrt(Ex[i] + s.eps) / Math.sqrt(Eg[i] + s.eps) * g;
            Ex[i] = s.rho * Ex[i] + (1 - s.rho) * dx * dx; p.data[i] += s.lr * dx;
          }
        });
      };
      return s;
    },
    // Adam, Algorithm 1. Default options → the shared NN.Adam. Options: bias:false (no correction = RMSProp-style),
    // decay:true (α_t = α/√t, §4 and §6.1), lambda (β1,t = β1·λ^(t−1), §4), amsgrad:true (Reddi et al., 2018; later work)
    adam: function (P, o) {
      const b1 = o.b1 == null ? 0.9 : o.b1, b2 = o.b2 == null ? 0.999 : o.b2, eps = o.eps == null ? 1e-8 : o.eps;
      if (o.bias !== false && !o.decay && o.lambda == null && !o.amsgrad) return new NN.Adam(P, { lr: o.lr, b1: b1, b2: b2, eps: eps });
      const s = { lr: o.lr, b1: b1, b2: b2, eps: eps, t: 0, m: zerosLike(P), v: zerosLike(P), vmax: zerosLike(P), params: P, bias: o.bias !== false };
      s.step = function () {
        s.t++;
        const t = s.t, b1t = o.lambda == null ? s.b1 : s.b1 * Math.pow(o.lambda, t - 1);
        const c1 = s.bias ? 1 - Math.pow(s.b1, t) : 1, c2 = s.bias ? 1 - Math.pow(s.b2, t) : 1;
        const a = o.decay ? s.lr / Math.sqrt(t) : s.lr;
        each(P, function (p, k) {
          const m = s.m[k], v = s.v[k], vm = s.vmax[k];
          for (let i = 0; i < m.length; i++) {
            const g = p.grad[i];
            m[i] = b1t * m[i] + (1 - b1t) * g;
            v[i] = s.b2 * v[i] + (1 - s.b2) * g * g;
            let vv = v[i];
            if (o.amsgrad) { vm[i] = Math.max(vm[i], v[i]); vv = vm[i]; }
            p.data[i] -= a * (m[i] / c1) / (Math.sqrt(vv / c2) + s.eps);
          }
        });
      };
      return s;
    },
    // AdaMax, Algorithm 2 (paper §7.1): u ← max(β2·u, |g|), θ ← θ − (α/(1 − β1^t))·m/u
    adamax: function (P, o) {
      const s = { lr: o.lr, b1: o.b1 == null ? 0.9 : o.b1, b2: o.b2 == null ? 0.999 : o.b2, t: 0, m: zerosLike(P), u: zerosLike(P), params: P };
      s.step = function () {
        s.t++;
        const c = s.lr / (1 - Math.pow(s.b1, s.t));
        each(P, function (p, k) {
          const m = s.m[k], u = s.u[k];
          for (let i = 0; i < m.length; i++) {
            const g = p.grad[i];
            m[i] = s.b1 * m[i] + (1 - s.b1) * g;
            u[i] = Math.max(s.b2 * u[i], Math.abs(g));
            if (u[i] > 0) p.data[i] -= c * m[i] / u[i];      // u = 0 only if every gradient so far was exactly 0
          }
        });
      };
      return s;
    }
  };
  Opt.make = function (name, params, opts) { return makers[name](params, opts || {}); };
  Opt.names = { sgd: 'SGD', momentum: 'Momentum', nesterov: 'SGD Nesterov', adagrad: 'AdaGrad', rmsprop: 'RMSProp', adadelta: 'AdaDelta', adam: 'Adam', adamax: 'AdaMax' };
  Opt.colors = { sgd: '#718096', momentum: '#b7791f', nesterov: '#c0392b', adagrad: '#2b6cb0', rmsprop: '#2f855a', adadelta: '#d53f8c', adam: '#6b46c1', adamax: '#1f7a7a' };

  // ------------------------------------------------------------------ 2-D surfaces (the book's own toys)
  Opt.surfaces = {
    ravine: { label: 'ravine (steep one way, flat the other)', box: [-5, 5, -3, 3], start: [-4.5, 2], min: [0, 0],
      f: function (x, y) { return 0.05 * x * x + 2 * y * y; }, g: function (x, y) { return [0.1 * x, 4 * y]; } },
    rosenbrock: { label: 'banana valley (Rosenbrock)', box: [-2, 2, -1, 3], start: [-1.6, 2.6], min: [1, 1],
      f: function (x, y) { return (1 - x) * (1 - x) + 10 * (y - x * x) * (y - x * x); },
      g: function (x, y) { return [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)]; } },
    plateau: { label: 'plateau (tiny gradients far away)', box: [-4, 4, -3, 3], start: [-3.5, -2.5], min: [2, 1],
      f: function (x, y) { const r = (x - 2) * (x - 2) + (y - 1) * (y - 1); return 1 - Math.exp(-r / 4); },
      g: function (x, y) { const r = (x - 2) * (x - 2) + (y - 1) * (y - 1), e = Math.exp(-r / 4) / 2; return [(x - 2) * e, (y - 1) * e]; } },
    bowl: { label: 'round bowl', box: [-3, 3, -3, 3], start: [-2.5, 2.2], min: [0, 0],
      f: function (x, y) { return 0.5 * (x * x + y * y); }, g: function (x, y) { return [x, y]; } },
    scaled: { label: 'badly scaled bowl (x-gradients 100× smaller)', box: [-3, 3, -3, 3], start: [-2.5, 2.5], min: [0, 0],
      f: function (x, y) { return 0.005 * x * x + 0.5 * y * y; }, g: function (x, y) { return [0.01 * x, y]; } }
  };
  // run one optimizer on a surface; noise = std of Gaussian noise added to each gradient coordinate (a stochastic objective)
  Opt.runSurface = function (surfKey, name, opts, steps, noise, seed, start) {
    const S = Opt.surfaces[surfKey], P = Opt.vec(start || S.start), o = Opt.make(name, [P], opts), r = ML.rng(seed || 1);
    const path = [[P.data[0], P.data[1]]], loss = [S.f(P.data[0], P.data[1])];
    for (let t = 0; t < steps; t++) {
      const g = S.g(P.data[0], P.data[1]);
      P.grad[0] = g[0] + (noise ? r.normal(0, noise) : 0); P.grad[1] = g[1] + (noise ? r.normal(0, noise) : 0);
      o.step();
      if (!isFinite(P.data[0]) || !isFinite(P.data[1]) || Math.abs(P.data[0]) > 1e6 || Math.abs(P.data[1]) > 1e6) { P.data[0] = NaN; break; }
      path.push([P.data[0], P.data[1]]); loss.push(S.f(P.data[0], P.data[1]));
    }
    return { path: path, loss: loss, opt: o };
  };
  // the race defaults: a learning rate per optimizer that works on all surfaces (book's choice, not the paper's)
  Opt.raceDefaults = {
    sgd: { lr: 0.2 }, momentum: { lr: 0.03, mu: 0.9 }, adagrad: { lr: 0.5 }, rmsprop: { lr: 0.05 }, adam: { lr: 0.1 }, adamax: { lr: 0.1 }
  };

  // ------------------------------------------------------------------ dense data: "glyphs" (MNIST stand-in)
  // side×side images of `classes` classes; each class is a prototype made of 3 soft blobs; an example is its
  // prototype, randomly brightened, plus pixel noise. Returns {X: Float64Array(n·D), y: Int32Array, n, D, K}.
  Opt.glyphs = function (n, seed, side, classes, noise) {
    side = side || 8; classes = classes || 10; noise = noise == null ? 0.6 : noise;
    const D = side * side, r = ML.rng(seed || 3), proto = [];
    const pr = ML.rng(101);                                    // prototypes are the same for every seed
    for (let c = 0; c < classes; c++) {
      const p = new Float64Array(D);
      for (let b = 0; b < 3; b++) {
        const cx = pr.uniform(0, side - 1), cy = pr.uniform(0, side - 1), s = pr.uniform(0.8, 1.8);
        for (let i = 0; i < side; i++) for (let j = 0; j < side; j++) p[i * side + j] += Math.exp(-((i - cy) * (i - cy) + (j - cx) * (j - cx)) / (2 * s * s));
      }
      proto.push(p);
    }
    const X = new Float64Array(n * D), y = new Int32Array(n);
    for (let k = 0; k < n; k++) {
      const c = k % classes, a = r.uniform(0.6, 1.4); y[k] = c;
      for (let j = 0; j < D; j++) X[k * D + j] = a * proto[c][j] + r.normal(0, noise);
    }
    return { X: X, y: y, n: n, D: D, K: classes, proto: proto, side: side };
  };

  // ------------------------------------------------------------------ sparse data: "reviews" (IMDB bag-of-words stand-in)
  // V words with Zipf frequencies. Each review has L word slots; most are neutral common words, some are drawn from
  // the sentiment words of its class, which are rarer. Features are word counts (bag of words), stored sparsely as (index, count) pairs.
  Opt.reviews = function (n, seed, V, L, o) {
    V = V || 2000; L = L || 30; o = o || {};
    const ps = o.ps == null ? 0.2 : o.ps, off = o.off == null ? 8 : o.off, flip = o.flip || 0;
    const r = ML.rng(seed || 5), pr = ML.rng(202);
    const zipf = function (k0, k1, of) { const w = []; let z = 0; for (let k = k0; k < k1; k++) { const q = 1 / (k - k0 + (of || 8)); w.push(q); z += q; } return w.map(function (q) { return q / z; }); };
    const H = Math.floor(V / 2), neutral = zipf(0, H);                             // first half of the vocabulary: neutral, frequent
    // second half: sentiment words (rarer), shuffled into the two classes
    const senti = ML.range(V - H).map(function (i) { return H + i; }); pr.shuffle(senti);
    const half = Math.floor((V - H) / 2), pos = senti.slice(0, half), neg = senti.slice(half, 2 * half), sp = zipf(0, half, off);
    const rows = [], counts = [], y = new Int32Array(n);
    for (let k = 0; k < n; k++) {
      const c = r.next() < 0.5 ? 1 : 0, cnt = new Map(); y[k] = c;
      for (let s = 0; s < L; s++) {
        const w = r.next() < ps ? (c ? pos : neg)[ML.sampleIndex(sp, r)] : ML.sampleIndex(neutral, r);
        cnt.set(w, (cnt.get(w) || 0) + 1);
      }
      if (flip && r.next() < flip) y[k] = 1 - c;            // label noise: some reviews disagree with their words
      rows.push(Int32Array.from(cnt.keys())); counts.push(Float64Array.from(cnt.values()));
    }
    return { rows: rows, counts: counts, y: y, n: n, V: V };
  };

  // ------------------------------------------------------------------ trainers (fast, hand-written gradients)
  // Multi-class (softmax) logistic regression with L2 penalty l2·|W|²/2 on the dense data. Params: W (D·K), b (K).
  Opt.SoftmaxReg = function (data, seed, l2) {
    this.d = data; this.l2 = l2 == null ? 1e-4 : l2;
    const r = ML.rng(seed || 9), D = data.D, K = data.K;
    this.W = Opt.vec(Array.from({ length: D * K }, function () { return r.normal(0, 0.01); })); this.b = Opt.vec(new Array(K).fill(0));
    this.params = [this.W, this.b]; this.p = new Float64Array(K);
  };
  // gradient on the examples idx (sets .grad), returns the mean cross-entropy (with penalty)
  Opt.SoftmaxReg.prototype.grad = function (idx) {
    const d = this.d, D = d.D, K = d.K, W = this.W.data, b = this.b.data, gW = this.W.grad, gb = this.b.grad, p = this.p, n = idx.length;
    gW.fill(0); gb.fill(0); let loss = 0;
    for (let a = 0; a < n; a++) {
      const k = idx[a], off = k * D;
      let mx = -Infinity;
      for (let c = 0; c < K; c++) { let z = b[c]; for (let j = 0; j < D; j++) z += d.X[off + j] * W[j * K + c]; p[c] = z; if (z > mx) mx = z; }
      let Z = 0; for (let c = 0; c < K; c++) { p[c] = Math.exp(p[c] - mx); Z += p[c]; }
      for (let c = 0; c < K; c++) p[c] /= Z;
      loss -= Math.log(Math.max(p[d.y[k]], 1e-300));
      for (let c = 0; c < K; c++) {
        const e = (p[c] - (c === d.y[k] ? 1 : 0)) / n; if (!e) continue;
        gb[c] += e; for (let j = 0; j < D; j++) gW[j * K + c] += e * d.X[off + j];
      }
    }
    let pen = 0; for (let i = 0; i < W.length; i++) { pen += W[i] * W[i]; gW[i] += this.l2 * W[i]; }
    return loss / n + 0.5 * this.l2 * pen;
  };
  Opt.SoftmaxReg.prototype.fullLoss = function () { return this.grad(ML.range(this.d.n)); };

  // Binary logistic regression on sparse bag-of-words rows, with optional dropout on the input features (p_drop),
  // inverted scaling as in the shared NN.dropout. Params: w (V), b (1).
  Opt.SparseLogReg = function (data, seed, l2) {
    this.d = data; this.l2 = l2 == null ? 1e-5 : l2;
    this.w = Opt.vec(new Array(data.V).fill(0)); this.b = Opt.vec([0]); this.params = [this.w, this.b];
    this.rng = ML.rng(seed || 13);
  };
  Opt.SparseLogReg.prototype.grad = function (idx, pDrop) {
    const d = this.d, w = this.w.data, gw = this.w.grad, n = idx.length, keep = 1 - (pDrop || 0);
    gw.fill(0); this.b.grad[0] = 0; let loss = 0;
    for (let a = 0; a < n; a++) {
      const row = d.rows[idx[a]], cn = d.counts[idx[a]], yv = d.y[idx[a]];
      const xs = []; let z = this.b.data[0];
      for (let j = 0; j < row.length; j++) { const x = cn[j] * (pDrop ? (this.rng.next() < keep ? 1 / keep : 0) : 1); xs.push(x); z += w[row[j]] * x; }
      const pr = ML.sigmoid(z);
      loss += Math.max(z, 0) - z * yv + Math.log(1 + Math.exp(-Math.abs(z)));
      const e = (pr - yv) / n;
      this.b.grad[0] += e; for (let j = 0; j < row.length; j++) if (xs[j]) gw[row[j]] += e * xs[j];
    }
    let pen = 0;
    for (let i = 0; i < w.length; i++) { if (w[i]) { pen += w[i] * w[i]; gw[i] += this.l2 * w[i]; } }
    return loss / n + 0.5 * this.l2 * pen;
  };
  Opt.SparseLogReg.prototype.fullLoss = function () { return this.grad(ML.range(this.d.n), 0); };

  // minibatch index generator: shuffles each epoch
  Opt.batcher = function (n, bs, seed) {
    const r = ML.rng(seed || 17); let order = r.shuffle(ML.range(n)), pos = 0;
    return function () { if (pos + bs > n) { order = r.shuffle(order); pos = 0; } const b = order.slice(pos, pos + bs); pos += bs; return b; };
  };

  // Train a model (SoftmaxReg / SparseLogReg / MLP) with one optimizer. Returns a "job" whose .run(ms) advances
  // training for about ms milliseconds (so widgets can stay responsive) and which records the full training cost
  // once per epoch: curve = [[epoch, cost]…].
  Opt.job = function (model, optName, optOpts, cfg) {
    const bs = cfg.bs || 32, n = model.d.n, perEpoch = Math.max(1, Math.floor(n / bs)), next = Opt.batcher(n, bs, cfg.seed || 17);
    const opt = Opt.make(optName, model.params, optOpts);
    const J = { model: model, opt: opt, epoch: 0, step: 0, curve: [[0, model.fullLoss()]], done: false, epochs: cfg.epochs, diverged: false };
    J.stepOnce = function () {
      model.grad(next(), cfg.pDrop || 0); opt.step(); J.step++;
      if (J.step % perEpoch === 0) {
        J.epoch++; let L = model.fullLoss();
        if (!isFinite(L) || L > 1e3) { L = Math.min(isFinite(L) ? L : 1e3, 1e3); J.diverged = true; }
        J.curve.push([J.epoch, L]);
        if (J.epoch >= cfg.epochs || J.diverged && cfg.stopOnDiverge) J.done = true;
      }
    };
    J.run = function (ms) { const t0 = performance.now(); while (!J.done && performance.now() - t0 < ms) J.stepOnce(); return J.done; };
    J.runAll = function () { while (!J.done) J.stepOnce(); return J; };
    return J;
  };

  // Small multilayer ReLU network on the shared autograd NN, with dropout (p = probability of DROPPING, as in NN.dropout)
  Opt.MLP = function (data, sizes, seed, drop) {
    this.d = data; this.drop = drop || [0, 0]; const r = ML.rng(seed || 21); this.rng = ML.rng((seed || 21) + 1);
    this.layers = []; for (let l = 0; l + 1 < sizes.length; l++) this.layers.push(new NN.Linear(sizes[l], sizes[l + 1], r, { sd: Math.sqrt(2 / sizes[l]) }));
    this.params = []; const self = this; this.layers.forEach(function (L) { self.params = self.params.concat(L.params); });
  };
  Opt.MLP.prototype.batch = function (idx) {
    const D = this.d.D, X = new NN.Tensor(idx.length, D);
    idx.forEach(function (k, a) { for (let j = 0; j < D; j++) X.data[a * D + j] = this.d.X[k * D + j]; }, this);
    return { X: X, y: Array.from(idx, function (k) { return this.d.y[k]; }, this) };
  };
  Opt.MLP.prototype.forward = function (X, training) {
    let h = NN.dropout(X, this.drop[0], this.rng, training);
    for (let l = 0; l < this.layers.length; l++) {
      h = this.layers[l].forward(h);
      if (l + 1 < this.layers.length) h = NN.dropout(NN.relu(h), this.drop[1], this.rng, training);
    }
    return h;
  };
  Opt.MLP.prototype.grad = function (idx) {
    NN.zeroGrad(this.params); const B = this.batch(idx);
    const L = NN.softmaxCE(this.forward(B.X, true), B.y); L.backward(); return L.item();
  };
  // the paper plots the training cost of the dropout objective; we evaluate the training cost with dropout off (the
  // deterministic network) on a fixed subset, which is smoother. Said so in the book.
  Opt.MLP.prototype.fullLoss = function () {
    if (!this._evalIdx) this._evalIdx = ML.range(Math.min(this.d.n, 400));
    const B = this.batch(this._evalIdx); return NN.softmaxCE(this.forward(B.X, false), B.y).item();
  };

  // ------------------------------------------------------------------ online convex games (§4)
  // A sparse online problem in d dimensions: at step t, coordinate i is "active" with probability q_i and then the
  // cost is |θ_i − c_t,i|; inactive coordinates cost nothing. Feasible set: the box [−1, 1]^d (iterates are projected).
  Opt.onlineProblem = function (d, T, seed, qmin, qmax) {
    const r = ML.rng(seed || 31), q = [], target = [], act = [], c = [];
    qmin = qmin == null ? 0.02 : qmin; qmax = qmax == null ? 0.9 : qmax;
    for (let i = 0; i < d; i++) { q.push(qmin + (qmax - qmin) * Math.pow(i / Math.max(1, d - 1), 2)); target.push(r.uniform(-0.6, 0.6)); }
    for (let t = 0; t < T; t++) {
      const a = new Uint8Array(d), cc = new Float64Array(d);
      for (let i = 0; i < d; i++) { a[i] = r.next() < q[i] ? 1 : 0; cc[i] = Book_clamp(target[i] + r.normal(0, 0.35), -1, 1); }
      act.push(a); c.push(cc);
    }
    // θ* minimises Σ_t f_t: per coordinate, the median of its active targets (any point if never active)
    const star = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      const vals = []; for (let t = 0; t < T; t++) if (act[t][i]) vals.push(c[t][i]);
      vals.sort(function (a, b) { return a - b; }); star[i] = vals.length ? vals[Math.floor((vals.length - 1) / 2)] : 0;
    }
    return { d: d, T: T, act: act, c: c, q: q, star: star,
      f: function (t, th) { let s = 0; for (let i = 0; i < d; i++) if (act[t][i]) s += Math.abs(th[i] - c[t][i]); return s; },
      g: function (t, th, out) { for (let i = 0; i < d; i++) out[i] = act[t][i] ? (th[i] > c[t][i] ? 1 : th[i] < c[t][i] ? -1 : 0) : 0; } };
  };
  function Book_clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  // Play the online game with one optimizer; returns cumulative regret R(t) for t = 1..T and the gradient history.
  Opt.playOnline = function (prob, name, opts) {
    const d = prob.d, P = Opt.vec(new Array(d).fill(0)), o = Opt.make(name, [P], opts);
    const R = [], gsq = new Float64Array(d); let reg = 0, gmax2 = 0;
    for (let t = 0; t < prob.T; t++) {
      reg += prob.f(t, P.data) - prob.f(t, prob.star); R.push(reg);
      prob.g(t, P.data, P.grad);
      if (opts && opts.lrDecay) o.lr = opts.lr / Math.sqrt(t + 1);           // α_t = α/√t for optimizers without a built-in schedule
      let n2 = 0; for (let i = 0; i < d; i++) { gsq[i] += P.grad[i] * P.grad[i]; n2 += P.grad[i] * P.grad[i]; }
      gmax2 = Math.max(gmax2, n2);
      o.step();
      for (let i = 0; i < d; i++) P.data[i] = Book_clamp(P.data[i], -1, 1);     // projection onto the feasible box
    }
    return { R: R, gsq: gsq, G: Math.sqrt(gmax2), opt: o, theta: Array.from(P.data) };
  };
  // The three terms of Theorem 4.1 for a finished Adam run (box [−1,1]^d: D = 2√d, D∞ = 2; |g| ≤ 1: G∞ = 1)
  Opt.theoremTerms = function (run, prob, a, b1, b2, lam) {
    const d = prob.d, T = prob.T, D = 2 * Math.sqrt(d), Dinf = 2, Ginf = 1, gam = b1 * b1 / Math.sqrt(b2);
    let sumV = 0, sumG = 0; const v = run.opt.v[0], c2 = 1 - Math.pow(b2, T);
    for (let i = 0; i < d; i++) { sumV += Math.sqrt(T * v[i] / c2); sumG += Math.sqrt(run.gsq[i]); }
    const t1 = D * D / (2 * a * (1 - b1)) * sumV;
    const t2 = a * (1 + b1) * Ginf / ((1 - b1) * Math.sqrt(1 - b2) * (1 - gam) * (1 - gam)) * sumG;
    const t3 = d * Dinf * Dinf * Ginf * Math.sqrt(1 - b2) / (2 * a * (1 - b1) * (1 - lam) * (1 - lam));
    return { t1: t1, t2: t2, t3: t3, total: t1 + t2 + t3, gamma: gam, sumV: sumV, sumG: sumG, worst: d * Ginf * Math.sqrt(T), D: D, Dinf: Dinf, G: run.G, Ginf: Ginf };
  };

  // Reddi, Kale & Kumar (2018) counterexample: f_t(x) = C·x if t mod 3 = 1, else −x, on [−1, 1]. Best point x = −1.
  Opt.reddi = function (C, T, name, opts) {
    const P = Opt.vec([0]), o = Opt.make(name, [P], opts), xs = [], R = []; let reg = 0;
    for (let t = 1; t <= T; t++) {
      const coef = t % 3 === 1 ? C : -1;
      reg += coef * P.data[0] - coef * -1; R.push(reg); xs.push(P.data[0]);
      P.grad[0] = coef; o.step(); P.data[0] = Book_clamp(P.data[0], -1, 1);
    }
    return { x: xs, R: R };
  };
})();
