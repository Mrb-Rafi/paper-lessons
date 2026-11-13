/* Tiny matrix autograd for the "Modern AI in 49 Papers" books — enough to train small networks live in the browser.
   A Tensor is a 2-D matrix (rows × cols, Float64Array, row-major) that remembers how it was computed, so
   loss.backward() fills .grad on every tensor that needs it (reverse-mode automatic differentiation).
   Ops: matmul, add (incl. row-broadcast bias), sub, mul, scale, relu, tanh, sigmoid, sum, mean, rowSum,
        embedding (row gather), dropout (inverted), batchNorm (training/eval), softmaxCE, sigmoidBCE, mse.
   Layers: Linear.  Optimizers: SGD (with momentum), Adam (state exposed for visualisation).
   Exposed as window.NN. Depends on window.ML (math.js) for random numbers. Tested by shared/lib/tests.html. */
(function () {
  'use strict';
  const NN = window.NN = window.NN || {};

  function Tensor(rows, cols, data, requiresGrad) {
    this.rows = rows; this.cols = cols;
    this.data = data || new Float64Array(rows * cols);
    this.grad = null;
    this.requiresGrad = !!requiresGrad;
    this._parents = []; this._back = null;
  }
  NN.Tensor = Tensor;
  Tensor.prototype.get = function (i, j) { return this.data[i * this.cols + j]; };
  Tensor.prototype.set = function (i, j, v) { this.data[i * this.cols + j] = v; };
  Tensor.prototype.item = function () { return this.data[0]; };
  Tensor.prototype.toArray = function () { const out = []; for (let i = 0; i < this.rows; i++) out.push(Array.from(this.data.subarray(i * this.cols, (i + 1) * this.cols))); return out; };
  Tensor.prototype.row = function (i) { return Array.from(this.data.subarray(i * this.cols, (i + 1) * this.cols)); };
  Tensor.prototype.clone = function () { return new Tensor(this.rows, this.cols, Float64Array.from(this.data), false); };

  // ---------- construction ----------
  NN.tensor = function (arr2d, requiresGrad) {
    const r = arr2d.length, c = arr2d[0].length, t = new Tensor(r, c, null, requiresGrad);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) t.data[i * c + j] = arr2d[i][j];
    return t;
  };
  NN.zeros = function (r, c, requiresGrad) { return new Tensor(r, c, null, requiresGrad); };
  NN.full = function (r, c, v, requiresGrad) { const t = new Tensor(r, c, null, requiresGrad); t.data.fill(v); return t; };
  // a trainable parameter with N(0, sd²) entries (sd defaults to Xavier/Glorot-style 1/sqrt(rows))
  NN.param = function (r, c, rng, sd) {
    sd = sd == null ? 1 / Math.sqrt(r) : sd;
    const t = new Tensor(r, c, null, true);
    for (let i = 0; i < t.data.length; i++) t.data[i] = rng.normal(0, sd);
    return t;
  };

  // internal: make a result tensor that depends on parents
  function result(rows, cols, parents, back) {
    const needs = parents.some(function (p) { return p.requiresGrad; });
    const t = new Tensor(rows, cols, null, needs);
    if (needs) { t._parents = parents; t._back = back; }
    return t;
  }
  function g(t) { if (!t.grad) t.grad = new Float64Array(t.data.length); return t.grad; }

  // ---------- backward ----------
  Tensor.prototype.backward = function () {
    const order = [], seen = new Set();
    (function visit(t) {
      if (seen.has(t)) return; seen.add(t);
      t._parents.forEach(visit); order.push(t);
    })(this);
    g(this).fill(0); this.grad[0] = 1;            // d(loss)/d(loss) = 1 (loss must be 1×1)
    for (let k = order.length - 1; k >= 0; k--) { const t = order[k]; if (t._back) t._back(t); }
  };
  NN.zeroGrad = function (params) { params.forEach(function (p) { if (p.grad) p.grad.fill(0); }); };

  // ---------- ops ----------
  NN.matmul = function (A, B) {
    if (A.cols !== B.rows) throw new Error('matmul shape mismatch ' + A.rows + 'x' + A.cols + ' · ' + B.rows + 'x' + B.cols);
    const n = A.rows, k = A.cols, m = B.cols;
    const out = result(n, m, [A, B], function (o) {
      if (A.requiresGrad) { const gA = g(A); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const go = o.grad[i * m + j]; if (!go) continue; for (let p = 0; p < k; p++) gA[i * k + p] += go * B.data[p * m + j]; } }
      if (B.requiresGrad) { const gB = g(B); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const go = o.grad[i * m + j]; if (!go) continue; for (let p = 0; p < k; p++) gB[p * m + j] += go * A.data[i * k + p]; } }
    });
    for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) { const a = A.data[i * k + p]; if (!a) continue; for (let j = 0; j < m; j++) out.data[i * m + j] += a * B.data[p * m + j]; }
    return out;
  };
  // A + B, where B has A's shape or is 1×cols (bias, broadcast over rows)
  NN.add = function (A, B) {
    const bcast = B.rows === 1 && A.rows !== 1;
    if (B.cols !== A.cols || (!bcast && B.rows !== A.rows)) throw new Error('add shape mismatch');
    const c = A.cols;
    const out = result(A.rows, c, [A, B], function (o) {
      if (A.requiresGrad) { const gA = g(A); for (let i = 0; i < o.grad.length; i++) gA[i] += o.grad[i]; }
      if (B.requiresGrad) { const gB = g(B); for (let i = 0; i < o.grad.length; i++) gB[bcast ? i % c : i] += o.grad[i]; }
    });
    for (let i = 0; i < out.data.length; i++) out.data[i] = A.data[i] + B.data[bcast ? i % c : i];
    return out;
  };
  NN.sub = function (A, B) { return NN.add(A, NN.scale(B, -1)); };
  NN.scale = function (A, k) {
    const out = result(A.rows, A.cols, [A], function (o) { const gA = g(A); for (let i = 0; i < o.grad.length; i++) gA[i] += k * o.grad[i]; });
    for (let i = 0; i < out.data.length; i++) out.data[i] = k * A.data[i];
    return out;
  };
  NN.mul = function (A, B) {   // elementwise, same shape
    const out = result(A.rows, A.cols, [A, B], function (o) {
      if (A.requiresGrad) { const gA = g(A); for (let i = 0; i < o.grad.length; i++) gA[i] += o.grad[i] * B.data[i]; }
      if (B.requiresGrad) { const gB = g(B); for (let i = 0; i < o.grad.length; i++) gB[i] += o.grad[i] * A.data[i]; }
    });
    for (let i = 0; i < out.data.length; i++) out.data[i] = A.data[i] * B.data[i];
    return out;
  };
  function unary(A, f, df) {   // df(x, y) = dy/dx given input x and output y
    const out = result(A.rows, A.cols, [A], function (o) { const gA = g(A); for (let i = 0; i < o.grad.length; i++) gA[i] += o.grad[i] * df(A.data[i], o.data[i]); });
    for (let i = 0; i < out.data.length; i++) out.data[i] = f(A.data[i]);
    return out;
  }
  NN.relu = function (A) { return unary(A, function (x) { return x > 0 ? x : 0; }, function (x) { return x > 0 ? 1 : 0; }); };
  NN.tanh = function (A) { return unary(A, Math.tanh, function (x, y) { return 1 - y * y; }); };
  NN.sigmoid = function (A) { return unary(A, window.ML.sigmoid, function (x, y) { return y * (1 - y); }); };
  NN.sum = function (A) {
    const out = result(1, 1, [A], function (o) { const gA = g(A); for (let i = 0; i < gA.length; i++) gA[i] += o.grad[0]; });
    let s = 0; for (let i = 0; i < A.data.length; i++) s += A.data[i]; out.data[0] = s;
    return out;
  };
  NN.mean = function (A) { return NN.scale(NN.sum(A), 1 / A.data.length); };
  NN.rowSum = function (A) {   // rows × 1
    const c = A.cols;
    const out = result(A.rows, 1, [A], function (o) { const gA = g(A); for (let i = 0; i < gA.length; i++) gA[i] += o.grad[Math.floor(i / c)]; });
    for (let i = 0; i < A.data.length; i++) out.data[Math.floor(i / c)] += A.data[i];
    return out;
  };
  // pick rows of W (vocab × dim) by index: the embedding lookup. Gradients are scatter-added back.
  NN.embedding = function (W, idx) {
    const d = W.cols;
    const out = result(idx.length, d, [W], function (o) { const gW = g(W); idx.forEach(function (r, i) { for (let j = 0; j < d; j++) gW[r * d + j] += o.grad[i * d + j]; }); });
    idx.forEach(function (r, i) { for (let j = 0; j < d; j++) out.data[i * d + j] = W.data[r * d + j]; });
    return out;
  };
  // inverted dropout: in training, zero each entry with probability p and scale survivors by 1/(1−p); identity at test time.
  NN.dropout = function (A, p, rng, training) {
    if (!training || p <= 0) return A;
    const mask = new Float64Array(A.data.length), keep = 1 - p;
    for (let i = 0; i < mask.length; i++) mask[i] = rng.next() < keep ? 1 / keep : 0;
    const out = result(A.rows, A.cols, [A], function (o) { const gA = g(A); for (let i = 0; i < gA.length; i++) gA[i] += o.grad[i] * mask[i]; });
    for (let i = 0; i < mask.length; i++) out.data[i] = A.data[i] * mask[i];
    out.mask = mask;
    return out;
  };
  // batch normalization over the rows (the batch) of A (batch × features).
  // training: normalise with the batch mean/variance and return them in out.stats; eval: use opts.running {mean, var}.
  NN.batchNorm = function (A, gamma, beta, opts) {
    opts = opts || {};
    const n = A.rows, c = A.cols, eps = opts.eps == null ? 1e-5 : opts.eps;
    const mean = new Float64Array(c), vr = new Float64Array(c);
    if (opts.training === false && opts.running) { for (let j = 0; j < c; j++) { mean[j] = opts.running.mean[j]; vr[j] = opts.running.var[j]; } }
    else {
      for (let i = 0; i < n; i++) for (let j = 0; j < c; j++) mean[j] += A.data[i * c + j] / n;
      for (let i = 0; i < n; i++) for (let j = 0; j < c; j++) { const d = A.data[i * c + j] - mean[j]; vr[j] += d * d / n; }
    }
    const inv = vr.map(function (v) { return 1 / Math.sqrt(v + eps); });
    const xhat = new Float64Array(n * c);
    for (let i = 0; i < n; i++) for (let j = 0; j < c; j++) xhat[i * c + j] = (A.data[i * c + j] - mean[j]) * inv[j];
    const training = !(opts.training === false && opts.running);
    const out = result(n, c, [A, gamma, beta], function (o) {
      const go = o.grad;
      if (gamma.requiresGrad) { const gg = g(gamma); for (let i = 0; i < n; i++) for (let j = 0; j < c; j++) gg[j] += go[i * c + j] * xhat[i * c + j]; }
      if (beta.requiresGrad) { const gb = g(beta); for (let i = 0; i < n; i++) for (let j = 0; j < c; j++) gb[j] += go[i * c + j]; }
      if (A.requiresGrad) {
        const gA = g(A);
        for (let j = 0; j < c; j++) {
          let sdx = 0, sdxx = 0;
          for (let i = 0; i < n; i++) { const dx = go[i * c + j] * gamma.data[j]; sdx += dx; sdxx += dx * xhat[i * c + j]; }
          for (let i = 0; i < n; i++) {
            const dx = go[i * c + j] * gamma.data[j];
            gA[i * c + j] += training ? inv[j] / n * (n * dx - sdx - xhat[i * c + j] * sdxx) : dx * inv[j];
          }
        }
      }
    });
    for (let i = 0; i < n; i++) for (let j = 0; j < c; j++) out.data[i * c + j] = gamma.data[j] * xhat[i * c + j] + beta.data[j];
    out.stats = { mean: Array.from(mean), var: Array.from(vr) };
    return out;
  };

  // ---------- losses (all return 1×1, averaged over rows) ----------
  // logits: batch × classes, targets: int[] → mean cross-entropy of softmax(logits). out.probs holds the softmax.
  NN.softmaxCE = function (logits, targets) {
    const n = logits.rows, c = logits.cols, P = new Float64Array(n * c);
    let loss = 0;
    for (let i = 0; i < n; i++) {
      let m = -Infinity; for (let j = 0; j < c; j++) m = Math.max(m, logits.data[i * c + j]);
      let z = 0; for (let j = 0; j < c; j++) { const e = Math.exp(logits.data[i * c + j] - m); P[i * c + j] = e; z += e; }
      for (let j = 0; j < c; j++) P[i * c + j] /= z;
      loss -= Math.log(Math.max(P[i * c + targets[i]], 1e-300));
    }
    const out = result(1, 1, [logits], function (o) {
      const gL = g(logits), s = o.grad[0] / n;
      for (let i = 0; i < n; i++) for (let j = 0; j < c; j++) gL[i * c + j] += s * (P[i * c + j] - (j === targets[i] ? 1 : 0));
    });
    out.data[0] = loss / n; out.probs = P;
    return out;
  };
  // logits: rows × 1 (or any shape), labels: 0/1 array in the same order → mean binary cross-entropy of sigmoid(logits)
  NN.sigmoidBCE = function (logits, labels) {
    const N = logits.data.length, S = window.ML.sigmoid;
    let loss = 0;
    for (let i = 0; i < N; i++) { const x = logits.data[i]; loss += Math.max(x, 0) - x * labels[i] + Math.log(1 + Math.exp(-Math.abs(x))); }
    const out = result(1, 1, [logits], function (o) { const gL = g(logits); for (let i = 0; i < N; i++) gL[i] += o.grad[0] / N * (S(logits.data[i]) - labels[i]); });
    out.data[0] = loss / N;
    return out;
  };
  NN.mse = function (pred, target) {   // target: Tensor of the same shape (not differentiated)
    const N = pred.data.length;
    const out = result(1, 1, [pred], function (o) { const gP = g(pred); for (let i = 0; i < N; i++) gP[i] += o.grad[0] * 2 * (pred.data[i] - target.data[i]) / N; });
    let s = 0; for (let i = 0; i < N; i++) { const d = pred.data[i] - target.data[i]; s += d * d; } out.data[0] = s / N;
    return out;
  };

  // ---------- layers ----------
  NN.Linear = function (inDim, outDim, rng, opts) {
    opts = opts || {};
    this.W = NN.param(inDim, outDim, rng, opts.sd);
    this.b = NN.zeros(1, outDim, true);
    this.params = opts.bias === false ? [this.W] : [this.W, this.b];
    this.bias = opts.bias !== false;
  };
  NN.Linear.prototype.forward = function (x) { const y = NN.matmul(x, this.W); return this.bias ? NN.add(y, this.b) : y; };

  // ---------- optimizers ----------
  NN.SGD = function (params, opts) {
    opts = opts || {};
    this.params = params; this.lr = opts.lr == null ? 0.1 : opts.lr; this.momentum = opts.momentum || 0;
    this.vel = params.map(function (p) { return new Float64Array(p.data.length); });
  };
  NN.SGD.prototype.step = function () {
    const self = this;
    this.params.forEach(function (p, k) {
      if (!p.grad) return;
      const v = self.vel[k];
      for (let i = 0; i < p.data.length; i++) { v[i] = self.momentum * v[i] + p.grad[i]; p.data[i] -= self.lr * v[i]; }
    });
  };
  // Adam (Kingma & Ba, 2015, Algorithm 1). State m, v and step count t are public for widgets.
  NN.Adam = function (params, opts) {
    opts = opts || {};
    this.params = params; this.lr = opts.lr == null ? 0.001 : opts.lr;
    this.b1 = opts.b1 == null ? 0.9 : opts.b1; this.b2 = opts.b2 == null ? 0.999 : opts.b2; this.eps = opts.eps == null ? 1e-8 : opts.eps;
    this.t = 0;
    this.m = params.map(function (p) { return new Float64Array(p.data.length); });
    this.v = params.map(function (p) { return new Float64Array(p.data.length); });
  };
  NN.Adam.prototype.step = function () {
    const s = this; s.t++;
    const c1 = 1 - Math.pow(s.b1, s.t), c2 = 1 - Math.pow(s.b2, s.t);
    this.params.forEach(function (p, k) {
      if (!p.grad) return;
      const m = s.m[k], v = s.v[k];
      for (let i = 0; i < p.data.length; i++) {
        const gi = p.grad[i];
        m[i] = s.b1 * m[i] + (1 - s.b1) * gi;
        v[i] = s.b2 * v[i] + (1 - s.b2) * gi * gi;
        p.data[i] -= s.lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + s.eps);
      }
    });
  };
})();
