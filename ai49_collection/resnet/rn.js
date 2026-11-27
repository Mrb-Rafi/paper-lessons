/* Toy library of the ResNet book (lesson 5), exposed as window.RNT.
   - Convolutional operations with automatic differentiation, built on the shared autograd (NN, shared/lib/nn.js):
     a feature-map batch is an NN.Tensor with one row per example and C·H·W columns (channel-major), plus a shape {C, H, W}.
     conv (any kernel, stride, padding), BN over batch and positions (§3.4, "BN right after each convolution"),
     max/average pooling, global average pooling, zero-padding shortcuts (option A) and projection shortcuts (option B).
   - glyphs(): 8×8 grey "pixel digits" with noise, our small stand-in for CIFAR-10's 32×32 photos.
   - CNN: the paper's CIFAR family (§4.2) in miniature: a 3×3 conv, then 3 stages of 2n 3×3 convs on maps {8, 4, 2} with
     {w, 2w, 4w} filters, global average pooling and a 10-way fully-connected layer: 6n+2 weighted layers, plain or residual.
   - MLP: the same plain/residual comparison with fully-connected layers (fast enough to race live in the page).
   - trainer(): SGD with momentum 0.9 and weight decay 1e-4 (§3.4, §4.2), time-sliced so the page stays responsive.
   Everything is seeded, so the pictures in the book are reproducible. */
(function () {
  'use strict';
  const ML = window.ML, NN = window.NN, T = window.RNT = {};
  T.active = 0;                                             // running background jobs (verification waits for 0)

  // ------------------------------------------------------------------ formatting helpers
  T.f = function (x, d) {
    if (!isFinite(x)) return isNaN(x) ? 'NaN' : (x > 0 ? '∞' : '−∞');
    d = d == null ? 3 : d; const s = Number(x).toFixed(d);
    return s.charAt(0) === '-' ? (/^-0\.?0*$/.test(s) ? s.slice(1) : '−' + s.slice(1)) : s;
  };
  T.big = function (x) {                                     // 11689512 → "11.69 M", 3.6e9 → "3.60 G"
    const a = Math.abs(x);
    if (a >= 1e9) return (x / 1e9).toFixed(2) + ' G';
    if (a >= 1e6) return (x / 1e6).toFixed(2) + ' M';
    if (a >= 1e3) return (x / 1e3).toFixed(1) + ' k';
    return String(Math.round(x));
  };
  T.comma = function (x) { return Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); };

  // ------------------------------------------------------------------ autograd plumbing (same contract as nn.js)
  function result(rows, cols, parents, back) {
    const needs = parents.some(function (p) { return p.requiresGrad; });
    const t = new NN.Tensor(rows, cols, null, needs);
    if (needs) { t._parents = parents; t._back = back; }
    return t;
  }
  function G(t) { if (!t.grad) t.grad = new Float64Array(t.data.length); return t.grad; }
  T._result = result;

  // ------------------------------------------------------------------ convolution
  // index table: for output position p and filter tap q = (ci, ky, kx), which input entry (−1 = padding)
  const tabs = {};
  function table(s, k, stride, pad) {
    const Ho = Math.floor((s.H + 2 * pad - k) / stride) + 1, Wo = Math.floor((s.W + 2 * pad - k) / stride) + 1;
    const key = [s.C, s.H, s.W, k, stride, pad].join(',');
    if (!tabs[key]) {
      const K = s.C * k * k, P = Ho * Wo, tab = new Int32Array(P * K);
      for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++) {
        const p = oy * Wo + ox;
        for (let ci = 0; ci < s.C; ci++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
          const iy = oy * stride - pad + ky, ix = ox * stride - pad + kx, q = (ci * k + ky) * k + kx;
          tab[p * K + q] = (iy < 0 || ix < 0 || iy >= s.H || ix >= s.W) ? -1 : (ci * s.H + iy) * s.W + ix;
        }
      }
      tabs[key] = { tab: tab, K: K, P: P, Ho: Ho, Wo: Wo };
    }
    return tabs[key];
  }
  T.outSize = function (n, k, stride, pad) { return Math.floor((n + 2 * pad - k) / stride) + 1; };
  // X: N × (C·H·W) with shape s; Wt: Cout × (C·k·k); optional bias b: 1 × Cout. Returns {t, s}.
  T.conv = function (X, s, Wt, k, stride, pad, b) {
    const tb = table(s, k, stride, pad), K = tb.K, P = tb.P, tab = tb.tab, N = X.rows, Cout = Wt.rows, inS = X.cols, outS = Cout * P;
    const parents = b ? [X, Wt, b] : [X, Wt];
    const out = result(N, outS, parents, function (o) {
      const go = o.grad, gx = X.requiresGrad ? G(X) : null, gw = Wt.requiresGrad ? G(Wt) : null, w = Wt.data, x = X.data;
      for (let n = 0; n < N; n++) {
        const xb = n * inS, ob = n * outS;
        for (let co = 0; co < Cout; co++) {
          const wb = co * K;
          for (let p = 0; p < P; p++) {
            const gv = go[ob + co * P + p]; if (gv === 0) continue;
            const tp = p * K;
            for (let q = 0; q < K; q++) {
              const idx = tab[tp + q]; if (idx < 0) continue;
              if (gx) gx[xb + idx] += gv * w[wb + q];
              if (gw) gw[wb + q] += gv * x[xb + idx];
            }
          }
        }
      }
      if (b && b.requiresGrad) { const gb = G(b); for (let n = 0; n < N; n++) for (let co = 0; co < Cout; co++) for (let p = 0; p < P; p++) gb[co] += go[n * outS + co * P + p]; }
    });
    const w = Wt.data, x = X.data, y = out.data;
    for (let n = 0; n < N; n++) {
      const xb = n * inS, ob = n * outS;
      for (let co = 0; co < Cout; co++) {
        const wb = co * K, bv = b ? b.data[co] : 0;
        for (let p = 0; p < P; p++) {
          let sum = bv; const tp = p * K;
          for (let q = 0; q < K; q++) { const idx = tab[tp + q]; if (idx >= 0) sum += w[wb + q] * x[xb + idx]; }
          y[ob + co * P + p] = sum;
        }
      }
    }
    return { t: out, s: { C: Cout, H: tb.Ho, W: tb.Wo } };
  };

  // ------------------------------------------------------------------ batch normalization for feature maps
  // statistics per channel over the batch and all positions (BN paper §3.2, as in PyTorch BatchNorm2d).
  // bn: {g: 1×C, b: 1×C, run: {mean, var}} ; mode: 'train' (batch stats) | 'eval' (bn.run)
  T.bn2d = function (X, s, bn, mode) {
    const N = X.rows, C = s.C, P = s.H * s.W, m = N * P, eps = 1e-5, D = X.cols;
    const mean = new Float64Array(C), vr = new Float64Array(C), x = X.data;
    const training = mode !== 'eval';
    if (!training) { for (let c = 0; c < C; c++) { mean[c] = bn.run.mean[c]; vr[c] = bn.run.var[c]; } }
    else {
      for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { let a = 0; const o = n * D + c * P; for (let p = 0; p < P; p++) a += x[o + p]; mean[c] += a; }
      for (let c = 0; c < C; c++) mean[c] /= m;
      for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { let a = 0; const o = n * D + c * P, mu = mean[c]; for (let p = 0; p < P; p++) { const d = x[o + p] - mu; a += d * d; } vr[c] += a; }
      for (let c = 0; c < C; c++) vr[c] /= m;
    }
    const inv = new Float64Array(C); for (let c = 0; c < C; c++) inv[c] = 1 / Math.sqrt(vr[c] + eps);
    const xh = new Float64Array(x.length);
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { const o = n * D + c * P; for (let p = 0; p < P; p++) xh[o + p] = (x[o + p] - mean[c]) * inv[c]; }
    const gam = bn.g, bet = bn.b;
    const out = result(N, D, [X, gam, bet], function (o) {
      const go = o.grad;
      const sg = new Float64Array(C), sgx = new Float64Array(C);
      for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { const off = n * D + c * P; for (let p = 0; p < P; p++) { sg[c] += go[off + p]; sgx[c] += go[off + p] * xh[off + p]; } }
      if (gam.requiresGrad) { const gg = G(gam); for (let c = 0; c < C; c++) gg[c] += sgx[c]; }
      if (bet.requiresGrad) { const gb = G(bet); for (let c = 0; c < C; c++) gb[c] += sg[c]; }
      if (X.requiresGrad) {
        const gx = G(X);
        for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) {
          const off = n * D + c * P, k = gam.data[c] * inv[c];
          for (let p = 0; p < P; p++) gx[off + p] += training ? k * (go[off + p] - sg[c] / m - xh[off + p] * sgx[c] / m) : k * go[off + p];
        }
      }
    });
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { const o = n * D + c * P; for (let p = 0; p < P; p++) out.data[o + p] = gam.data[c] * xh[o + p] + bet.data[c]; }
    out.stats = { mean: mean, var: vr };
    return out;
  };
  T.newBN = function (C, gamma0) { return { g: NN.full(1, C, gamma0 == null ? 1 : gamma0, true), b: NN.zeros(1, C, true), run: { mean: new Float64Array(C), var: new Float64Array(C).fill(1) } }; };

  // ------------------------------------------------------------------ pooling and shortcuts
  T.pool = function (X, s, k, stride, pad, kind) {         // kind 'max' | 'avg'; padding cells are ignored
    const Ho = T.outSize(s.H, k, stride, pad), Wo = T.outSize(s.W, k, stride, pad), N = X.rows, C = s.C, D = X.cols, OD = C * Ho * Wo;
    const arg = new Int32Array(N * OD), cnt = new Float64Array(Ho * Wo);
    const out = result(N, OD, [X], function (o) {
      const gx = G(X), go = o.grad;
      if (kind === 'max') { for (let i = 0; i < go.length; i++) gx[arg[i]] += go[i]; return; }
      for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++) {
        const gv = go[n * OD + (c * Ho + oy) * Wo + ox] / cnt[oy * Wo + ox];
        for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) { const iy = oy * stride - pad + ky, ix = ox * stride - pad + kx; if (iy >= 0 && ix >= 0 && iy < s.H && ix < s.W) gx[n * D + (c * s.H + iy) * s.W + ix] += gv; }
      }
    });
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++) {
      let best = -Infinity, bi = -1, sum = 0, ct = 0;
      for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
        const iy = oy * stride - pad + ky, ix = ox * stride - pad + kx;
        if (iy < 0 || ix < 0 || iy >= s.H || ix >= s.W) continue;
        const idx = n * D + (c * s.H + iy) * s.W + ix, v = X.data[idx]; sum += v; ct++;
        if (v > best) { best = v; bi = idx; }
      }
      const oi = n * OD + (c * Ho + oy) * Wo + ox;
      out.data[oi] = kind === 'max' ? best : sum / ct; arg[oi] = bi; cnt[oy * Wo + ox] = ct;
    }
    return { t: out, s: { C: C, H: Ho, W: Wo } };
  };
  T.gap = function (X, s) {                                 // global average pooling: N × C
    const N = X.rows, C = s.C, P = s.H * s.W, D = X.cols;
    const out = result(N, C, [X], function (o) { const gx = G(X); for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { const gv = o.grad[n * C + c] / P; for (let p = 0; p < P; p++) gx[n * D + c * P + p] += gv; } });
    for (let n = 0; n < N; n++) for (let c = 0; c < C; c++) { let a = 0; for (let p = 0; p < P; p++) a += X.data[n * D + c * P + p]; out.data[n * C + c] = a / P; }
    return out;
  };
  // option A (§3.3): identity with stride 2 (take every second position) and extra zero channels
  T.padShortcut = function (X, s, Cout, stride) {
    const Ho = Math.ceil(s.H / stride), Wo = Math.ceil(s.W / stride), N = X.rows, D = X.cols, OD = Cout * Ho * Wo;
    const out = result(N, OD, [X], function (o) {
      const gx = G(X);
      for (let n = 0; n < N; n++) for (let c = 0; c < s.C; c++) for (let y = 0; y < Ho; y++) for (let x = 0; x < Wo; x++)
        gx[n * D + (c * s.H + y * stride) * s.W + x * stride] += o.grad[n * OD + (c * Ho + y) * Wo + x];
    });
    for (let n = 0; n < N; n++) for (let c = 0; c < s.C; c++) for (let y = 0; y < Ho; y++) for (let x = 0; x < Wo; x++)
      out.data[n * OD + (c * Ho + y) * Wo + x] = X.data[n * D + (c * s.H + y * stride) * s.W + x * stride];
    return { t: out, s: { C: Cout, H: Ho, W: Wo } };
  };
  // He et al. 2015 ("[12]" in the paper) initialization: N(0, 2/fan_in)
  T.heParam = function (rows, fanIn, rng, scale) { return NN.param(rows, fanIn, rng, (scale == null ? 1 : scale) * Math.sqrt(2 / fanIn)); };

  // ------------------------------------------------------------------ data: 8×8 grey glyphs
  const FONT = [
    '01110100011001110101110011000101110', '00100011000010000100001000010001110', '01110100010000100010001000100011111',
    '11111000100010000010000011000101110', '00010001100101010010111110001000010', '11111100001111000001000011000101110',
    '00110010001000011110100011000101110', '11111000010001000100010000100001000', '01110100011000101110100011000101110',
    '01110100011000101111000010001001100'];
  T.FONT = FONT;
  // each glyph: a digit's 5×7 strokes at a random offset, random ink brightness, a few lost stroke pixels, and Gaussian noise
  T.glyphs = function (n, seed, o) {
    o = o || {}; const noise = o.noise == null ? 0.35 : o.noise, drop = o.drop == null ? 0.12 : o.drop;
    const r = ML.rng(seed || 7), D = 64, X = new Float64Array(n * D), y = new Int32Array(n);
    for (let k = 0; k < n; k++) {
      const c = r.int(10), ox = r.int(4), oy = r.int(2), ink = r.uniform(0.6, 1.2); y[k] = c;
      for (let i = 0; i < 7; i++) for (let j = 0; j < 5; j++) if (FONT[c][i * 5 + j] === '1' && r.next() >= drop) X[k * D + (i + oy) * 8 + j + ox] = ink;
      for (let p = 0; p < D; p++) X[k * D + p] += r.normal(0, noise);
    }
    let mu = 0; for (let i = 0; i < X.length; i++) mu += X[i]; mu /= X.length;
    for (let i = 0; i < X.length; i++) X[i] -= mu;          // mean subtracted, as the paper's inputs (§3.4, §4.2)
    return { X: X, y: y, n: n, D: D, K: 10, s: { C: 1, H: 8, W: 8 } };
  };
  T.data = function () { if (!T._data) T._data = { train: T.glyphs(2000, 11), test: T.glyphs(500, 12) }; return T._data; };
  T.rows = function (d, idx) {
    const D = d.D, X = new NN.Tensor(idx.length, D);
    for (let a = 0; a < idx.length; a++) { const k = idx[a]; X.data.set(d.X.subarray(k * D, (k + 1) * D), a * D); }
    return X;
  };

  // ------------------------------------------------------------------ the CIFAR-style family (§4.2), shrunk
  // cfg: {n, w (first-stage filters, default 4), residual, option: 'A'|'B', seed, k0 (scale of the last conv init in each block)}
  T.CNN = function (cfg) {
    this.cfg = cfg; const r = ML.rng(cfg.seed || 3), w = cfg.w || 4, n = cfg.n;
    this.depth = 6 * n + 2; this.params = []; this.decay = [];
    const self = this;
    function conv(cin, cout, k) { const W = T.heParam(cout, cin * k * k, r); self.params.push(W); self.decay.push(W); return W; }
    function bn(C) { const b = T.newBN(C); self.params.push(b.g, b.b); return b; }
    this.stem = { W: conv(1, w, 3), bn: bn(w) };
    this.blocks = []; let cin = w;
    [w, 2 * w, 4 * w].forEach(function (cout, stage) {
      for (let i = 0; i < n; i++) {
        const stride = stage > 0 && i === 0 ? 2 : 1;
        const blk = { stride: stride, cin: cin, cout: cout, W1: conv(cin, cout, 3), bn1: bn(cout), W2: conv(cout, cout, 3), bn2: bn(cout) };
        if (cfg.residual && cfg.option === 'B' && (stride !== 1 || cin !== cout)) { blk.Ws = conv(cin, cout, 1); blk.bns = bn(cout); }
        self.blocks.push(blk); cin = cout;
      }
    });
    this.fc = new NN.Linear(4 * w, 10, r, { sd: Math.sqrt(1 / (4 * w)) });
    this.params.push(this.fc.W, this.fc.b); this.decay.push(this.fc.W);
  };
  T.CNN.prototype.bns = function () {
    const out = [this.stem.bn]; this.blocks.forEach(function (b) { out.push(b.bn1, b.bn2); if (b.bns) out.push(b.bns); }); return out;
  };
  // mode 'train' | 'eval'; rec (optional array) receives the std of every 3×3 layer's response after BN, before ReLU/addition (Fig. 7)
  T.CNN.prototype.forward = function (X, mode, rec, maps) {
    const res = this.cfg.residual, opt = this.cfg.option || 'A';
    function std(t) { const v = t.data; let m = 0; for (let i = 0; i < v.length; i++) m += v[i]; m /= v.length; let s = 0; for (let i = 0; i < v.length; i++) s += (v[i] - m) * (v[i] - m); return Math.sqrt(s / v.length); }
    let s = { C: 1, H: 8, W: 8 };
    let c = T.conv(X, s, this.stem.W, 3, 1, 1); s = c.s;
    let h = T.bn2d(c.t, s, this.stem.bn, mode); if (rec) rec.push(std(h));
    h = NN.relu(h); if (maps) maps.push({ name: 'after the first 3×3 conv', t: h, s: s });
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      const c1 = T.conv(h, s, b.W1, 3, b.stride, 1); const s1 = c1.s;
      let u = T.bn2d(c1.t, s1, b.bn1, mode); if (rec) rec.push(std(u)); u = NN.relu(u);
      const c2 = T.conv(u, s1, b.W2, 3, 1, 1);
      let v = T.bn2d(c2.t, s1, b.bn2, mode); if (rec) rec.push(std(v));
      if (res) {
        let sc = h;
        if (b.stride !== 1 || b.cin !== b.cout) {
          if (opt === 'B') { const p = T.conv(h, s, b.Ws, 1, b.stride, 0); sc = T.bn2d(p.t, p.s, b.bns, mode); }
          else sc = T.padShortcut(h, s, b.cout, b.stride).t;
        }
        v = NN.add(v, sc);
      }
      h = NN.relu(v); s = s1; if (maps) maps.push({ name: 'after block ' + (i + 1), t: h, s: s });
    }
    const pooled = T.gap(h, s); if (maps) maps.push({ name: 'global average pool', t: pooled, s: { C: s.C, H: 1, W: 1 } });
    return this.fc.forward(pooled);
  };
  // population statistics as in the official README: a plain average over a large training batch after training
  T.CNN.prototype.freeze = function (X) {
    const self = this, orig = T.bn2d;          // run once in train mode and keep each BN's batch statistics
    T.bn2d = function (A, s, bn, mode) { const o = orig(A, s, bn, mode); if (mode !== 'eval') { bn.run.mean = Float64Array.from(o.stats.mean); bn.run.var = Float64Array.from(o.stats.var); } return o; };
    try { self.forward(X, 'train'); } finally { T.bn2d = orig; }
  };
  T.CNN.prototype.nParams = function () { return this.params.reduce(function (a, p) { return a + p.data.length; }, 0); };

  // ------------------------------------------------------------------ fully-connected plain / residual nets
  // cfg: {inDim, width, blocks (2 layers each), residual, bn (default true), classes, seed}. Weighted layers = 2·blocks + 2.
  T.MLP = function (cfg) {
    this.cfg = cfg; const r = ML.rng(cfg.seed || 5), w = cfg.width, self = this;
    this.depth = 2 * cfg.blocks + 2; this.params = []; this.decay = [];
    function lin(a, b, scale) { const W = NN.param(a, b, r, (scale == null ? 1 : scale) * Math.sqrt(2 / a)); self.params.push(W); self.decay.push(W); return W; }
    function bn(C) { const g = NN.full(1, C, 1, true), b = NN.zeros(1, C, true); self.params.push(g, b); return { g: g, b: b, run: { mean: new Float64Array(C), var: new Float64Array(C).fill(1) } }; }
    this.inW = lin(cfg.inDim, w); this.inBN = bn(w);
    this.blocks = [];
    for (let i = 0; i < cfg.blocks; i++) this.blocks.push({ W1: lin(w, w), bn1: bn(w), W2: lin(w, w), bn2: bn(w) });
    this.out = new NN.Linear(w, cfg.classes || 2, r, { sd: Math.sqrt(1 / w) }); this.params.push(this.out.W, this.out.b); this.decay.push(this.out.W);
  };
  // mode 'train' (batch statistics; running averages updated with momentum 0.1 as in PyTorch) | 'eval' (running statistics)
  T.MLP.prototype.forward = function (X, rec, mode) {
    const useBN = this.cfg.bn !== false, ev = mode === 'eval';
    function norm(A, b) {
      if (!useBN) return A;
      const o = NN.batchNorm(A, b.g, b.b, ev ? { training: false, running: b.run } : {});
      if (!ev && mode === 'train') for (let j = 0; j < A.cols; j++) { b.run.mean[j] = 0.9 * b.run.mean[j] + 0.1 * o.stats.mean[j]; b.run.var[j] = 0.9 * b.run.var[j] + 0.1 * o.stats.var[j] * X.rows / Math.max(1, X.rows - 1); }
      return o;
    }
    let h = NN.relu(norm(NN.matmul(X, this.inW), this.inBN));
    if (rec) rec.push(h);
    for (let i = 0; i < this.blocks.length; i++) {
      const b = this.blocks[i];
      const u = NN.relu(norm(NN.matmul(h, b.W1), b.bn1));
      let v = norm(NN.matmul(u, b.W2), b.bn2);
      if (this.cfg.residual) v = NN.add(v, h);
      h = NN.relu(v); if (rec) rec.push(h);
    }
    return this.out.forward(h);
  };

  // ------------------------------------------------------------------ 2-D toy data for the MLP
  // k interleaved spiral arms; a hard-enough task that a shallow net cannot fit it perfectly in a few hundred steps
  T.spirals = function (n, arms, seed, noise) {
    const r = ML.rng(seed || 9), X = new Float64Array(n * 2), y = new Int32Array(n);
    noise = noise == null ? 0.04 : noise;
    for (let i = 0; i < n; i++) {
      const c = i % arms, t = r.uniform(0.1, 1), a = 4.2 * t + (2 * Math.PI * c) / arms;
      X[2 * i] = t * Math.cos(a) + r.normal(0, noise); X[2 * i + 1] = t * Math.sin(a) + r.normal(0, noise); y[i] = c;
    }
    return { X: X, y: y, n: n, D: 2, K: arms };
  };

  // ------------------------------------------------------------------ training
  // SGD with momentum (v ← μv + g + λw; w ← w − ηv, the Caffe form the official README describes), minibatches drawn with a seeded RNG.
  // job = T.trainer(net, data, {lr, steps, batch, momentum, wd, schedule:[[step, lr],…], every, seed, eval:fn})
  T.trainer = function (net, data, o) {
    o = o || {};
    const opt = new NN.SGD(net.params, { lr: o.lr == null ? 0.1 : o.lr, momentum: o.momentum == null ? 0.9 : o.momentum });
    const wd = o.wd == null ? 1e-4 : o.wd, B = o.batch || 32, r = ML.rng(o.seed || 21), every = o.every || 10;
    const job = { net: net, opt: opt, step: 0, steps: o.steps || 300, hist: [], lossAvg: null, errAvg: null };
    const isCNN = net instanceof T.CNN;
    job.one = function () {
      if (o.schedule) o.schedule.forEach(function (sc) { if (job.step === sc[0]) opt.lr = sc[1]; });
      const idx = []; for (let i = 0; i < B; i++) idx.push(r.int(data.n));
      const X = T.rows(data, idx), y = idx.map(function (k) { return data.y[k]; });
      NN.zeroGrad(net.params);
      const logits = isCNN ? net.forward(X, 'train') : net.forward(X, null, 'train');
      const L = NN.softmaxCE(logits, y); L.backward();
      net.decay.forEach(function (p) { if (!p.grad) return; for (let i = 0; i < p.data.length; i++) p.grad[i] += wd * p.data[i]; });
      opt.step();
      let wrong = 0; const K = logits.cols;
      for (let i = 0; i < B; i++) { let best = 0; for (let j = 1; j < K; j++) if (L.probs[i * K + j] > L.probs[i * K + best]) best = j; if (best !== y[i]) wrong++; }
      const l = L.item(), e = wrong / B;
      job.lossAvg = job.lossAvg == null ? l : 0.9 * job.lossAvg + 0.1 * l;
      job.errAvg = job.errAvg == null ? e : 0.9 * job.errAvg + 0.1 * e;
      job.step++;
      if (job.step % every === 0 || job.step === job.steps) {
        const rec = { step: job.step, loss: job.lossAvg, err: job.errAvg };
        if (o.eval) Object.assign(rec, o.eval(net));
        job.hist.push(rec);
      }
      return isFinite(l);
    };
    job.done = function () { return job.step >= job.steps; };
    return job;
  };
  // full-data error / loss (MLP: training mode statistics on the whole set; CNN: frozen statistics)
  T.evaluate = function (net, data, limit) {
    const n = Math.min(data.n, limit || data.n), idx = ML.range(n), X = T.rows(data, idx), y = idx.map(function (k) { return data.y[k]; });
    const isCNN = net instanceof T.CNN;
    const logits = isCNN ? net.forward(X, 'eval') : net.forward(X, null, 'eval');
    const L = NN.softmaxCE(logits, y), K = logits.cols; let wrong = 0;
    for (let i = 0; i < n; i++) { let best = 0; for (let j = 1; j < K; j++) if (L.probs[i * K + j] > L.probs[i * K + best]) best = j; if (best !== y[i]) wrong++; }
    return { loss: L.item(), err: wrong / n };
  };

  // ------------------------------------------------------------------ time-sliced runner
  // runs several jobs round-robin in ~25 ms slices; onTick(jobs) after each slice; returns {stop()}
  T.run = function (jobs, onTick, onDone, budget) {
    let stopped = false, timer = null; T.active++;
    budget = budget || 25;
    function finish() { if (!stopped) { stopped = true; T.active--; if (onDone) onDone(jobs); } }
    function slice() {
      if (stopped) return;
      const t0 = performance.now();
      do {
        let any = false;
        for (let j = 0; j < jobs.length; j++) if (!jobs[j].done() && !jobs[j].dead) { any = true; if (!jobs[j].one()) jobs[j].dead = true; }
        if (!any) { if (onTick) onTick(jobs); finish(); return; }
      } while (performance.now() - t0 < budget);
      if (onTick) onTick(jobs);
      timer = setTimeout(slice, 0);
    }
    timer = setTimeout(slice, 0);
    return { stop: function () { if (!stopped) { clearTimeout(timer); stopped = true; T.active--; } } };
  };

  // ------------------------------------------------------------------ architecture arithmetic (Table 1)
  // the paper's ImageNet family. Returns a list of layers with shapes, parameters and multiply-adds.
  T.ARCH = {
    18: { block: 'basic', n: [2, 2, 2, 2] }, 34: { block: 'basic', n: [3, 4, 6, 3] },
    50: { block: 'bottleneck', n: [3, 4, 6, 3] }, 101: { block: 'bottleneck', n: [3, 4, 23, 3] }, 152: { block: 'bottleneck', n: [3, 8, 36, 3] }
  };
  // o: {shortcut: 'A'|'B'|'C', bn: true, strideOn: '1x1' (paper/Caffe) | '3x3' (torchvision v1.5), plain: false, size: 224}
  T.archLayers = function (depth, o) {
    o = o || {}; const A = T.ARCH[depth], size = o.size || 224, sc = o.shortcut || (A.block === 'basic' ? 'A' : 'B');
    const L = []; let H = size, C = 3;
    function conv(name, k, cout, stride, pad, group) {
      const Ho = Math.floor((H + 2 * pad - k) / stride) + 1;
      const p = k * k * C * cout, f = Ho * Ho * p;
      L.push({ name: name, kind: 'conv', k: k, cin: C, cout: cout, stride: stride, inH: H, outH: Ho, params: p, flops: f, bn: o.bn === false ? 0 : 2 * cout, group: group });
      return Ho;
    }
    H = conv('conv1', 7, 64, 2, 3, 'conv1'); C = 64;
    const Hp = Math.floor((H + 2 - 3) / 2) + 1; L.push({ name: 'max pool', kind: 'pool', k: 3, cin: C, cout: C, stride: 2, inH: H, outH: Hp, params: 0, flops: 0, bn: 0, group: 'conv2_x' }); H = Hp;
    const widths = [64, 128, 256, 512];
    A.n.forEach(function (nb, st) {
      const g = 'conv' + (st + 2) + '_x', w = widths[st], out = A.block === 'basic' ? w : 4 * w;
      for (let i = 0; i < nb; i++) {
        const stride = st > 0 && i === 0 ? 2 : 1, cin = C, Hin = H, nm = g.replace('_x', '_' + (i + 1));
        if (A.block === 'basic') {
          H = conv(nm + ' a', 3, w, stride, 1, g); C = w; H = conv(nm + ' b', 3, w, 1, 1, g); C = w;
        } else {
          const s1 = o.strideOn === '3x3' ? 1 : stride, s2 = o.strideOn === '3x3' ? stride : 1;
          H = conv(nm + ' a', 1, w, s1, 0, g); C = w; H = conv(nm + ' b', 3, w, s2, 1, g); C = w; H = conv(nm + ' c', 1, out, 1, 0, g); C = out;
        }
        if (!o.plain) {
          const needs = stride !== 1 || cin !== out, saveC = C, saveH = H;
          if (sc === 'C' || (needs && sc === 'B')) {
            C = cin; H = Hin; conv(nm + ' proj', 1, out, stride, 0, g); L[L.length - 1].kind = 'proj'; C = saveC; H = saveH;
          }
          L.push({ name: nm + ' add', kind: 'add', cin: out, cout: out, inH: H, outH: H, params: 0, flops: H * H * out, bn: 0, group: g, counted: false });
        }
      }
    });
    L.push({ name: 'avg pool', kind: 'pool', cin: C, cout: C, inH: H, outH: 1, params: 0, flops: 0, bn: 0, group: 'fc' });
    L.push({ name: 'fc 1000', kind: 'fc', cin: C, cout: 1000, inH: 1, outH: 1, params: C * 1000 + 1000, flops: C * 1000, bn: 0, group: 'fc' });
    return L;
  };
  T.archTotals = function (L) {
    let params = 0, flops = 0, bn = 0, weighted = 0, proj = 0;
    L.forEach(function (l) {
      params += l.params; bn += l.bn;
      if (l.kind === 'conv' || l.kind === 'fc' || l.kind === 'proj') flops += l.flops;
      if (l.kind === 'conv' || l.kind === 'fc') weighted++;
      if (l.kind === 'proj') proj++;
    });
    return { params: params + bn, convParams: params, bnParams: bn, flops: flops, weighted: weighted, proj: proj };
  };
})();

/* The CNN experiments of the evidence part: one function, used both by _work/precompute.html (which writes toy-results.js)
   and by the widgets' "Re-run live" buttons, so the stored numbers are exactly what the page computes. */
(function () {
  'use strict';
  const T = window.RNT, ML = window.ML;
  T.CNN_RUNS = [
    { id: 'plain-8', n: 1, residual: false }, { id: 'plain-20', n: 3, residual: false }, { id: 'plain-32', n: 5, residual: false },
    { id: 'res-8', n: 1, residual: true }, { id: 'res-20', n: 3, residual: true }, { id: 'res-32', n: 5, residual: true },
    { id: 'res-20-B', n: 3, residual: true, option: 'B' }
  ];
  T.CNN_STEPS = 800;
  // returns a job; job.result() gives {id, depth, params, curve:[[step, trainErr, testErr]], final:{train, test}, std:[…]}
  T.cnnJob = function (run) {
    const d = { train: T.glyphs(2000, 11, { noise: 0.25 }), test: T.glyphs(500, 12, { noise: 0.25 }) };
    const net = new T.CNN({ n: run.n, w: 6, residual: run.residual, option: run.option || 'A', seed: 3 });
    const S = T.CNN_STEPS, freezeX = T.rows(d.train, ML.range(500));
    const curve = [];
    function measure(step) {
      net.freeze(freezeX);
      const a = T.evaluate(net, d.train, 1000), b = T.evaluate(net, d.test);
      curve.push([step, +(100 * a.err).toFixed(2), +(100 * b.err).toFixed(2)]);
      return { train: a.err, test: b.err };
    }
    const job = T.trainer(net, d.train, { steps: S, batch: 32, lr: 0.1, schedule: [[Math.round(S * 0.6), 0.01], [Math.round(S * 0.85), 0.001]], seed: 5, every: 1e9 });
    const inner = job.one;
    job.one = function () { const ok = inner(); if (job.step % 100 === 0) measure(job.step); return ok; };
    job.result = function () {
      const fin = curve.length ? { train: curve[curve.length - 1][1], test: curve[curve.length - 1][2] } : null;
      net.freeze(freezeX); const rec = []; net.forward(T.rows(d.train, ML.range(500)), 'eval', rec);
      return { id: run.id, depth: net.depth, params: net.nParams(), curve: curve, final: fin, std: rec.map(function (v) { return +v.toFixed(4); }) };
    };
    return job;
  };
})();
