/* Shared maths for the "Modern AI in 49 Papers" books: seeded randomness, vectors, matrices, softmax and friends,
   and a 2-D projection for plotting embeddings. Plain arrays everywhere (number[] for vectors, number[][] for
   matrices, row-major) so widgets can read results directly. Everything is deterministic given a seed.
   Exposed as window.ML. Tested by shared/lib/tests.html. */
(function () {
  'use strict';
  const ML = window.ML = window.ML || {};

  // ---------- seeded random numbers (mulberry32) ----------
  ML.rng = function (seed) {
    let a = (seed >>> 0) || 1;
    const next = function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    const r = {
      next: next,                                   // uniform in [0, 1)
      uniform: function (lo, hi) { return lo + (hi - lo) * next(); },
      int: function (n) { return Math.floor(next() * n); },   // 0 … n−1
      normal: function (mu, sd) {                   // Box–Muller
        mu = mu || 0; sd = sd == null ? 1 : sd;
        if (spare != null) { const s = spare; spare = null; return mu + sd * s; }
        let u = 0, v = 0;
        while (u === 0) u = next();
        v = next();
        const m = Math.sqrt(-2 * Math.log(u));
        spare = m * Math.sin(2 * Math.PI * v);
        return mu + sd * m * Math.cos(2 * Math.PI * v);
      },
      shuffle: function (arr) {                     // in place, returns arr
        for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(next() * (i + 1)); const t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
        return arr;
      },
      choice: function (probs) { return ML.sampleIndex(probs, r); }
    };
    return r;
  };

  // ---------- vectors ----------
  ML.zeros = function (n) { return new Array(n).fill(0); };
  ML.range = function (n) { return Array.from({ length: n }, function (_, i) { return i; }); };
  ML.linspace = function (a, b, n) { return Array.from({ length: n }, function (_, i) { return n === 1 ? a : a + (b - a) * i / (n - 1); }); };
  ML.sum = function (v) { let s = 0; for (let i = 0; i < v.length; i++) s += v[i]; return s; };
  ML.mean = function (v) { return v.length ? ML.sum(v) / v.length : 0; };
  ML.variance = function (v) { const m = ML.mean(v); let s = 0; for (let i = 0; i < v.length; i++) s += (v[i] - m) * (v[i] - m); return v.length ? s / v.length : 0; };
  ML.dot = function (a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  ML.add = function (a, b) { return a.map(function (x, i) { return x + b[i]; }); };
  ML.sub = function (a, b) { return a.map(function (x, i) { return x - b[i]; }); };
  ML.scale = function (a, k) { return a.map(function (x) { return x * k; }); };
  ML.norm = function (a) { return Math.sqrt(ML.dot(a, a)); };
  ML.normalize = function (a) { const n = ML.norm(a) || 1; return a.map(function (x) { return x / n; }); };
  ML.cosine = function (a, b) { const d = ML.norm(a) * ML.norm(b); return d ? ML.dot(a, b) / d : 0; };
  ML.dist2 = function (a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return s; };
  ML.argmax = function (v) { let k = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[k]) k = i; return k; };
  ML.argmin = function (v) { let k = 0; for (let i = 1; i < v.length; i++) if (v[i] < v[k]) k = i; return k; };
  // indices of the k largest values, largest first
  ML.topk = function (v, k) { return ML.range(v.length).sort(function (i, j) { return v[j] - v[i]; }).slice(0, k); };

  // ---------- matrices (number[][], row-major) ----------
  ML.mat = function (r, c, f) { return Array.from({ length: r }, function (_, i) { return Array.from({ length: c }, function (_, j) { return f ? f(i, j) : 0; }); }); };
  ML.randn = function (r, c, rng, sd) { sd = sd == null ? 1 : sd; return ML.mat(r, c, function () { return rng.normal(0, sd); }); };
  ML.transpose = function (A) { return A.length ? A[0].map(function (_, j) { return A.map(function (row) { return row[j]; }); }) : []; };
  ML.matmul = function (A, B) {
    const n = A.length, m = B[0].length, k = B.length, C = ML.mat(n, m);
    for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) { const a = A[i][p]; if (a === 0) continue; const Bp = B[p], Ci = C[i]; for (let j = 0; j < m; j++) Ci[j] += a * Bp[j]; }
    return C;
  };
  ML.matvec = function (A, v) { return A.map(function (row) { return ML.dot(row, v); }); };

  // ---------- probabilities ----------
  // softmax with temperature tau (default 1); subtracts the max for numerical stability
  ML.softmax = function (s, tau) {
    tau = tau || 1;
    const m = Math.max.apply(null, s), e = s.map(function (x) { return Math.exp((x - m) / tau); }), z = ML.sum(e);
    return e.map(function (x) { return x / z; });
  };
  ML.logsumexp = function (s) { const m = Math.max.apply(null, s); return m + Math.log(ML.sum(s.map(function (x) { return Math.exp(x - m); }))); };
  ML.sigmoid = function (x) { return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); };
  ML.entropy = function (p, base) { let h = 0; for (let i = 0; i < p.length; i++) if (p[i] > 0) h -= p[i] * Math.log(p[i]); return base ? h / Math.log(base) : h; };
  ML.crossEntropy = function (p, target) { return -Math.log(Math.max(p[target], 1e-300)); };
  ML.sampleIndex = function (probs, rng) {
    let u = (rng ? rng.next() : Math.random()) * ML.sum(probs);
    for (let i = 0; i < probs.length; i++) { u -= probs[i]; if (u <= 0) return i; }
    return probs.length - 1;
  };

  // ---------- 2-D projection for plotting high-dimensional points (PCA by power iteration) ----------
  // rows: number[][] (n points × d dims). Returns { points: [[x, y]…], components: [pc1, pc2], explained: [f1, f2] }.
  ML.pca2 = function (rows, iters) {
    iters = iters || 200;
    const n = rows.length, d = rows[0].length;
    const mu = ML.zeros(d);
    rows.forEach(function (r) { for (let j = 0; j < d; j++) mu[j] += r[j] / n; });
    const X = rows.map(function (r) { return ML.sub(r, mu); });
    const C = ML.mat(d, d);                       // covariance
    X.forEach(function (r) { for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += r[i] * r[j] / n; });
    let total = 0; for (let i = 0; i < d; i++) total += C[i][i];
    const comps = [], vals = [];
    for (let c = 0; c < Math.min(2, d); c++) {
      let v = ML.normalize(ML.range(d).map(function (j) { return Math.sin(j * 1.7 + c + 1); }));   // deterministic start
      for (let t = 0; t < iters; t++) {
        let w = ML.matvec(C, v);
        comps.forEach(function (u) { w = ML.sub(w, ML.scale(u, ML.dot(w, u))); });            // deflate
        const nw = ML.norm(w); if (nw < 1e-12) break; v = ML.scale(w, 1 / nw);
      }
      comps.push(v); vals.push(ML.dot(v, ML.matvec(C, v)));
    }
    while (comps.length < 2) { comps.push(ML.zeros(d)); vals.push(0); }
    return { points: X.map(function (r) { return [ML.dot(r, comps[0]), ML.dot(r, comps[1])]; }), components: comps,
             explained: vals.map(function (v) { return total ? v / total : 0; }), mean: mu };
  };
})();
