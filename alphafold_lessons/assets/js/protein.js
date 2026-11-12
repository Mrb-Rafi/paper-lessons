/* Toy protein engine for the AlphaFold Interactive Web Book.
   Everything here is real geometry (vectors, rotations, superposition, the paper's metrics, frames, FAPE),
   applied to a small hand-built toy protein. Nothing here is a trained neural network. */
(function () {
  'use strict';
  const P = window.Prot = {};

  // ---------------------------------------------------------------- random numbers
  P.rng = function (seed) {
    let a = seed >>> 0;
    return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  };
  P.gauss = function (r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

  // ---------------------------------------------------------------- vectors (3-D arrays)
  const add = P.add = function (a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; };
  const sub = P.sub = function (a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; };
  const scale = P.scale = function (a, s) { return [a[0] * s, a[1] * s, a[2] * s]; };
  const dot = P.dot = function (a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  const cross = P.cross = function (a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; };
  const norm = P.norm = function (a) { return Math.sqrt(dot(a, a)); };
  const unit = P.unit = function (a) { const n = norm(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
  const dist = P.dist = function (a, b) { return norm(sub(a, b)); };
  P.centroid = function (X) { const c = [0, 0, 0]; X.forEach(function (x) { c[0] += x[0]; c[1] += x[1]; c[2] += x[2]; }); return scale(c, 1 / X.length); };
  P.lerp = function (a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; };

  // ---------------------------------------------------------------- 3x3 matrices (row-major arrays of rows)
  const mv = P.mv = function (R, v) { return [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]]; };
  const mm = P.mm = function (A, B) { const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j]; return C; };
  const tr = P.transpose = function (R) { return [[R[0][0], R[1][0], R[2][0]], [R[0][1], R[1][1], R[2][1]], [R[0][2], R[1][2], R[2][2]]]; };
  P.I3 = function () { return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; };
  P.det3 = function (R) { return R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]); };
  P.rotAxis = function (axis, ang) {
    const u = unit(axis), c = Math.cos(ang), s = Math.sin(ang), t = 1 - c, x = u[0], y = u[1], z = u[2];
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x], [t * x * z - s * y, t * y * z + s * x, t * z * z + c]];
  };
  P.rotEuler = function (a, b, c) { return mm(P.rotAxis([0, 0, 1], a), mm(P.rotAxis([0, 1, 0], b), P.rotAxis([1, 0, 0], c))); };
  // quaternion (a,b,c,d) -> rotation matrix (SI Alg. 23 uses (1,b,c,d)/norm)
  P.quatToRot = function (q) {
    const n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]), a = q[0] / n, b = q[1] / n, c = q[2] / n, d = q[3] / n;
    return [[a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c], [2 * b * c + 2 * a * d, a * a - b * b + c * c - d * d, 2 * c * d - 2 * a * b], [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a - b * b - c * c + d * d]];
  };

  // ---------------------------------------------------------------- rigid transforms T = (R, t):  T∘x = R x + t
  P.apply = function (T, x) { return add(mv(T.R, x), T.t); };
  P.invApply = function (T, x) { return mv(tr(T.R), sub(x, T.t)); };
  P.compose = function (T1, T2) { return { R: mm(T1.R, T2.R), t: add(mv(T1.R, T2.t), T1.t) }; };
  P.identityFrame = function () { return { R: P.I3(), t: [0, 0, 0] }; };
  // SI Alg. 21: rigid frame from three backbone atoms by Gram–Schmidt (x1 = N, x2 = CA, x3 = C)
  P.rigidFrom3 = function (x1, x2, x3) {
    const v1 = sub(x3, x2), v2 = sub(x1, x2);
    const e1 = unit(v1);
    const u2 = sub(v2, scale(e1, dot(e1, v2)));
    const e2 = unit(u2), e3 = cross(e1, e2);
    return { R: [[e1[0], e2[0], e3[0]], [e1[1], e2[1], e3[1]], [e1[2], e2[2], e3[2]]], t: x2.slice() };
  };

  // ---------------------------------------------------------------- symmetric eigen-decomposition (Jacobi)
  P.jacobi = function (A0) {
    const n = A0.length, A = A0.map(function (r) { return r.slice(); }), V = [];
    for (let i = 0; i < n; i++) { V.push([]); for (let j = 0; j < n; j++) V[i].push(i === j ? 1 : 0); }
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-18) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-20) continue;
        const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
      }
    }
    return { values: A.map(function (r, i) { return r[i]; }), vectors: V };
  };

  // ---------------------------------------------------------------- optimal superposition (Horn's quaternion method)
  // returns {R, t} such that R*X_i + t best matches Y_i (least squares, proper rotation only), plus rmsd
  P.superpose = function (X, Y, w) {
    const n = X.length; w = w || X.map(function () { return 1; });
    let W = 0; const cx = [0, 0, 0], cy = [0, 0, 0];
    for (let i = 0; i < n; i++) { W += w[i]; for (let k = 0; k < 3; k++) { cx[k] += w[i] * X[i][k]; cy[k] += w[i] * Y[i][k]; } }
    if (W <= 0) return { R: P.I3(), t: [0, 0, 0], rmsd: Infinity };
    for (let k = 0; k < 3; k++) { cx[k] /= W; cy[k] /= W; }
    const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < n; i++) { const a = sub(X[i], cx), b = sub(Y[i], cy); for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[r][c] += w[i] * a[r] * b[c]; }
    const Sxx = S[0][0], Sxy = S[0][1], Sxz = S[0][2], Syx = S[1][0], Syy = S[1][1], Syz = S[1][2], Szx = S[2][0], Szy = S[2][1], Szz = S[2][2];
    const N = [[Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx], [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz], [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy], [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]];
    const e = P.jacobi(N); let bi = 0; for (let i = 1; i < 4; i++) if (e.values[i] > e.values[bi]) bi = i;
    const q = [e.vectors[0][bi], e.vectors[1][bi], e.vectors[2][bi], e.vectors[3][bi]];
    const R = P.quatToRot(q), t = sub(cy, mv(R, cx));
    let s = 0; for (let i = 0; i < n; i++) { const d = sub(add(mv(R, X[i]), t), Y[i]); s += w[i] * dot(d, d); }
    return { R: R, t: t, rmsd: Math.sqrt(s / W) };
  };
  P.transformAll = function (T, X) { return X.map(function (x) { return add(mv(T.R, x), T.t); }); };
  P.rmsd = function (X, Y) { return P.superpose(X, Y).rmsd; };
  P.rmsdNoAlign = function (X, Y) { let s = 0; for (let i = 0; i < X.length; i++) { const d = sub(X[i], Y[i]); s += dot(d, d); } return Math.sqrt(s / X.length); };

  // r.m.s.d.95 exactly as in the Methods: 5 iterations of (align on chosen atoms; keep 95% lowest error)
  P.rmsd95 = function (X, Y) {
    const n = X.length; let sel = X.map(function () { return 1; }); let T = null;
    for (let it = 0; it < 5; it++) {
      T = P.superpose(X, Y, sel);
      const err = X.map(function (x, i) { return { i: i, d: dist(P.apply(T, x), Y[i]) }; }).sort(function (a, b) { return a.d - b.d; });
      sel = X.map(function () { return 0; }); err.slice(0, Math.max(3, Math.round(0.95 * n))).forEach(function (e) { sel[e.i] = 1; });
    }
    T = P.superpose(X, Y, sel);
    let s = 0, m = 0; for (let i = 0; i < n; i++) if (sel[i]) { const d = dist(P.apply(T, X[i]), Y[i]); s += d * d; m++; }
    return Math.sqrt(s / m);
  };

  // fraction of residues within cutoff after a superposition that tries to maximise that fraction (simplified LGA)
  function bestFractionWithin(X, Y, cutoff) {
    const n = X.length; let best = 0;
    const seeds = [X.map(function () { return 1; })];
    for (const len of [Math.max(4, Math.floor(n / 2)), Math.max(4, Math.floor(n / 4))]) for (let s = 0; s + len <= n; s += Math.max(1, Math.floor(len / 2))) { const w = X.map(function (_, i) { return i >= s && i < s + len ? 1 : 0; }); seeds.push(w); }
    seeds.forEach(function (w) {
      for (let it = 0; it < 8; it++) {
        const T = P.superpose(X, Y, w);
        const d = X.map(function (x, i) { return dist(P.apply(T, x), Y[i]); });
        const f = d.filter(function (v) { return v <= cutoff; }).length / n;
        if (f > best) best = f;
        const nw = d.map(function (v) { return v <= Math.max(cutoff, 1) * 1.0 ? 1 : 0; });
        if (nw.reduce(function (a, b) { return a + b; }, 0) < 3) break;
        w = nw;
      }
    });
    return best;
  }
  // GDT_TS: mean of fractions within 1, 2, 4, 8 Å (x100)
  P.gdt = function (X, Y) { return 25 * [1, 2, 4, 8].reduce(function (s, c) { return s + bestFractionWithin(X, Y, c); }, 0); };
  P.gdtParts = function (X, Y) { return [1, 2, 4, 8].map(function (c) { return bestFractionWithin(X, Y, c); }); };

  // TM-score (Zhang & Skolnick 2004), normalised by target length, with a superposition search
  P.d0 = function (L) { return Math.max(0.5, 1.24 * Math.cbrt(Math.max(L, 16) - 15) - 1.8); };
  P.tmscore = function (X, Y) {
    const n = X.length, d0 = P.d0(n); let best = 0;
    const tm = function (T) { let s = 0; for (let i = 0; i < n; i++) { const d = dist(P.apply(T, X[i]), Y[i]); s += 1 / (1 + (d / d0) * (d / d0)); } return s / n; };
    const seeds = [X.map(function () { return 1; })];
    for (const len of [Math.max(4, Math.floor(n / 2)), Math.max(4, Math.floor(n / 4))]) for (let s = 0; s + len <= n; s += Math.max(1, Math.floor(len / 2))) seeds.push(X.map(function (_, i) { return i >= s && i < s + len ? 1 : 0; }));
    seeds.forEach(function (w) {
      for (let it = 0; it < 10; it++) {
        const T = P.superpose(X, Y, w); const v = tm(T); if (v > best) best = v;
        const d = X.map(function (x, i) { return dist(P.apply(T, x), Y[i]); });
        let cut = d0 + 1.5, nw;
        do { nw = d.map(function (v) { return v < cut ? 1 : 0; }); cut += 0.5; } while (nw.reduce(function (a, b) { return a + b; }, 0) < 3 && cut < 50);
        w = nw;
      }
    });
    return best;
  };

  // lDDT (Mariani et al. 2013), Cα-only version: superposition-free.
  // Returns {global, perResidue}. Pairs: true distance < R0 (15 Å), i != j. Thresholds 0.5, 1, 2, 4 Å.
  P.lddt = function (X, Y, R0) {
    R0 = R0 || 15; const n = X.length, per = [];
    let tot = 0, cnt = 0;
    for (let i = 0; i < n; i++) {
      let s = 0, c = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dt = dist(Y[i], Y[j]); if (dt >= R0) continue;
        const dp = dist(X[i], X[j]), e = Math.abs(dp - dt);
        s += ((e < 0.5) + (e < 1) + (e < 2) + (e < 4)) / 4; c++;
      }
      per.push(c ? 100 * s / c : 100); tot += s; cnt += c;
    }
    return { global: cnt ? 100 * tot / cnt : 100, perResidue: per };
  };
  // distance-matrix RMSD (invariant to rotation, translation AND reflection)
  P.drmsd = function (X, Y) { let s = 0, c = 0; for (let i = 0; i < X.length; i++) for (let j = i + 1; j < X.length; j++) { const e = dist(X[i], X[j]) - dist(Y[i], Y[j]); s += e * e; c++; } return Math.sqrt(s / c); };
  P.distMatrix = function (X) { return X.map(function (a) { return X.map(function (b) { return dist(a, b); }); }); };

  // ---------------------------------------------------------------- FAPE (SI Alg. 28)
  // frames: [{R,t}], x: [[x,y,z]]. Returns {loss (unitless, divided by Z), mean distance (Å), matrix}.
  P.fape = function (predF, predX, trueF, trueX, opt) {
    opt = opt || {}; const clamp = opt.clamp == null ? 10 : opt.clamp, Z = opt.Z || 10, eps = opt.eps || 1e-4;
    let s = 0, sRaw = 0, c = 0; const M = [];
    for (let i = 0; i < predF.length; i++) {
      const row = [];
      for (let j = 0; j < predX.length; j++) {
        const a = P.invApply(predF[i], predX[j]), b = P.invApply(trueF[i], trueX[j]);
        const d = Math.sqrt(dot(sub(a, b), sub(a, b)) + eps);
        row.push(d); s += Math.min(clamp, d); sRaw += d; c++;
      }
      M.push(row);
    }
    return { loss: s / c / Z, meanClamped: s / c, meanRaw: sRaw / c, matrix: M };
  };

  // ---------------------------------------------------------------- NeRF: build a backbone from torsion angles
  const DEG = Math.PI / 180; P.DEG = DEG;
  P.place = function (a, b, c, bond, angle, torsion) {
    const bc = unit(sub(c, b)), n = unit(cross(sub(b, a), bc)), m = cross(n, bc);
    const d2 = [-bond * Math.cos(angle), bond * Math.sin(angle) * Math.cos(torsion), bond * Math.sin(angle) * Math.sin(torsion)];
    return add(c, add(add(scale(bc, d2[0]), scale(m, d2[1])), scale(n, d2[2])));
  };
  // ideal geometry (Engh & Huber); phi, psi, omega in degrees; returns [{N, CA, C, O}]
  P.buildBackbone = function (phi, psi, omega) {
    const n = phi.length, res = [];
    let N = [0, 0, 0], CA = [1.458, 0, 0];
    let C = add(CA, [Math.cos(Math.PI - 111.2 * DEG) * 1.525, Math.sin(Math.PI - 111.2 * DEG) * 1.525, 0]);
    for (let i = 0; i < n; i++) {
      if (i > 0) {
        const prev = res[i - 1];
        N = P.place(prev.N, prev.CA, prev.C, 1.329, 116.2 * DEG, psi[i - 1] * DEG);
        CA = P.place(prev.CA, prev.C, N, 1.458, 121.7 * DEG, (omega ? omega[i - 1] : 180) * DEG);
        C = P.place(prev.C, N, CA, 1.525, 111.2 * DEG, phi[i] * DEG);
      }
      res.push({ N: N, CA: CA, C: C });
    }
    for (let i = 0; i < n; i++) {
      const r = res[i];
      r.O = i < n - 1 ? P.place(res[i + 1].N, r.CA, r.C, 1.231, 120.5 * DEG, 180 * DEG) : P.place(r.N, r.CA, r.C, 1.231, 120.5 * DEG, (psi[i] + 180) * DEG);
      r.CB = P.place(r.C, r.N, r.CA, 1.53, 110.5 * DEG, -122.6 * DEG);
    }
    return res;
  };
  // torsion (dihedral) angle a-b-c-d in degrees
  P.dihedral = function (a, b, c, d) {
    const b0 = sub(a, b), b1 = unit(sub(c, b)), b2 = sub(d, c);
    const v = sub(b0, scale(b1, dot(b0, b1))), w = sub(b2, scale(b1, dot(b2, b1)));
    return Math.atan2(dot(cross(b1, v), w), dot(v, w)) / DEG;
  };

  // ---------------------------------------------------------------- amino acids
  P.AA = 'ARNDCQEGHILKMFPSTWYV';
  P.AA_INFO = {
    A: ['Ala', 'Alanine', 'hydrophobic'], R: ['Arg', 'Arginine', 'positive'], N: ['Asn', 'Asparagine', 'polar'], D: ['Asp', 'Aspartate', 'negative'],
    C: ['Cys', 'Cysteine', 'special'], Q: ['Gln', 'Glutamine', 'polar'], E: ['Glu', 'Glutamate', 'negative'], G: ['Gly', 'Glycine', 'special'],
    H: ['His', 'Histidine', 'positive'], I: ['Ile', 'Isoleucine', 'hydrophobic'], L: ['Leu', 'Leucine', 'hydrophobic'], K: ['Lys', 'Lysine', 'positive'],
    M: ['Met', 'Methionine', 'hydrophobic'], F: ['Phe', 'Phenylalanine', 'hydrophobic'], P: ['Pro', 'Proline', 'special'], S: ['Ser', 'Serine', 'polar'],
    T: ['Thr', 'Threonine', 'polar'], W: ['Trp', 'Tryptophan', 'hydrophobic'], Y: ['Tyr', 'Tyrosine', 'polar'], V: ['Val', 'Valine', 'hydrophobic']
  };
  // number of side-chain chi angles per residue type (SI Table: up to 4)
  P.CHI = { A: 0, G: 0, S: 1, C: 1, T: 1, V: 1, P: 2, I: 2, L: 2, D: 2, N: 2, H: 2, F: 2, W: 2, Y: 2, M: 3, E: 3, Q: 3, K: 4, R: 4 };
  P.CLASS_COLOR = { hydrophobic: '#d69e2e', positive: '#3182ce', negative: '#e53e3e', polar: '#38a169', special: '#805ad5' };
  P.aaColor = function (a) { const i = P.AA_INFO[a]; return i ? P.CLASS_COLOR[i[2]] : '#999'; };
  P.aaClass = function (a) { const i = P.AA_INFO[a]; return i ? i[2] : 'gap'; };

  // ---------------------------------------------------------------- the toy protein: a 3-helix bundle
  // Built geometrically (ideal α-helices: radius 2.3 Å, rise 1.5 Å, 100° per residue), loops relaxed to 3.8 Å Cα–Cα spacing.
  function relaxChain(X, fixed, iters) {
    for (let it = 0; it < iters; it++) {
      for (let i = 0; i < X.length - 1; i++) {
        const d = sub(X[i + 1], X[i]), L = norm(d) || 1e-6, corr = (L - 3.8) / L * 0.5;
        if (!fixed[i]) X[i] = add(X[i], scale(d, corr * (fixed[i + 1] ? 2 : 1)));
        if (!fixed[i + 1]) X[i + 1] = sub(X[i + 1], scale(d, corr * (fixed[i] ? 2 : 1)));
      }
      for (let i = 0; i < X.length; i++) for (let j = i + 2; j < X.length; j++) {
        const d = sub(X[j], X[i]), L = norm(d), m = j === i + 2 ? 5.3 : 4.4;
        if (L < m && L > 1e-6) { const c = (m - L) / L * 0.25; if (!fixed[i]) X[i] = sub(X[i], scale(d, c)); if (!fixed[j]) X[j] = add(X[j], scale(d, c)); }
      }
    }
    return X;
  }
  P.makeBundle = function (opt) {
    opt = opt || {};
    const hl = opt.helixLen || 14, ll = opt.loopLen || 4, sep = opt.sep || 10.0, tail = opt.tail || 0;
    const axes = [0, 1, 2].map(function (k) { const a = (k * 120 + 90) * DEG, r = sep / Math.sqrt(3); return [r * Math.cos(a), r * Math.sin(a)]; });
    const X = [], ss = [], fixed = [];
    const centre = [0, 0];
    let phase = 0;
    for (let h = 0; h < 3; h++) {
      const up = h % 2 === 0, ax = axes[h];
      // choose the phase so that a residue faces the bundle centre at the helix start
      const toC = Math.atan2(centre[1] - ax[1], centre[0] - ax[0]);
      phase = toC;
      const helix = [];
      for (let i = 0; i < hl; i++) {
        const ang = phase + (up ? 1 : -1) * i * 100 * DEG, z = (up ? -1 : 1) * (hl - 1) * 0.75 + (up ? 1 : -1) * i * 1.5;
        helix.push([ax[0] + 2.3 * Math.cos(ang), ax[1] + 2.3 * Math.sin(ang), z]);
      }
      if (h > 0) {
        const a = X[X.length - 1], b = helix[0], mid = scale(add(a, b), 0.5);
        const outward = unit([mid[0], mid[1], 0]), bulge = add(add(mid, scale(outward, 3.0)), [0, 0, (h % 2 === 1 ? 1 : -1) * 4.0]);
        for (let i = 1; i <= ll; i++) {
          const t = i / (ll + 1), p1 = P.lerp(a, bulge, t), p2 = P.lerp(bulge, b, t);
          X.push(P.lerp(p1, p2, t)); ss.push('L'); fixed.push(false);
        }
      }
      helix.forEach(function (p) { X.push(p); ss.push('H'); fixed.push(true); });
    }
    if (tail > 0) {
      const r = P.rng(opt.seed || 7); let p = X[X.length - 1], dir = unit(sub(X[X.length - 1], X[X.length - 2]));
      for (let i = 0; i < tail; i++) {
        dir = unit(add(dir, [0.6 * P.gauss(r), 0.6 * P.gauss(r), 0.6 * P.gauss(r)])); p = add(p, scale(dir, 3.8)); X.push(p); ss.push('C'); fixed.push(false);
      }
    }
    relaxChain(X, fixed, 200);
    const c = P.centroid(X);
    return { ca: X.map(function (x) { return sub(x, c); }), ss: ss.join('') };
  };
  // approximate N and C atoms from a Cα trace (for frames); real AlphaFold uses the true N, Cα, C atoms
  P.backboneFromCA = function (X) {
    const n = X.length, out = [];
    for (let i = 0; i < n; i++) {
      const prev = i > 0 ? X[i - 1] : add(X[i], sub(X[i + 1], X[i + 2])), next = i < n - 1 ? X[i + 1] : add(X[i], sub(X[i - 1], X[i - 2]));
      const a = unit(sub(prev, X[i])), b = unit(sub(next, X[i])), bis = unit(add(a, b)), perp = unit(cross(a, b));
      // rotate the N / C directions slightly out of the Cα plane so the frames are well defined
      const Nd = unit(add(add(scale(a, 0.85), scale(bis, -0.25)), scale(perp, 0.35)));
      const Cd = unit(add(add(scale(b, 0.85), scale(bis, -0.25)), scale(perp, -0.35)));
      out.push({ N: add(X[i], scale(Nd, 1.458)), CA: X[i].slice(), C: add(X[i], scale(Cd, 1.525)) });
    }
    return out;
  };
  P.framesOf = function (bb) { return bb.map(function (r) { return P.rigidFrom3(r.N, r.CA, r.C); }); };

  P.contacts = function (ca, cut, minSep) {
    cut = cut || 8; minSep = minSep || 3; const L = ca.length, C = [];
    for (let i = 0; i < L; i++) for (let j = i + minSep; j < L; j++) if (dist(ca[i], ca[j]) < cut) C.push([i, j]);
    return C;
  };

  // ---------------------------------------------------------------- toy evolution: a Potts model sampled by MCMC
  // A deliberately simple "toy chemistry" with four kinds of residue:
  //   B = big hydrophobic, S = small (hydrophobic or tiny), K = positive, E = negative.
  // Residues that touch across the fold (contacts with |i − j| ≥ 6) prefer compatible partners:
  // big packs against small (two bigs clash), + pairs with −.  Buried positions prefer B/S, exposed ones K/E.
  P.TOY_CLASSES = 'BSKE';
  P.TOY_LETTERS = { B: 'FLIWMLF', S: 'AVSGTAV', K: 'KRKRKHK', E: 'EDEDEQE' };
  P.TOY_CLASS_NAME = { B: 'big hydrophobic', S: 'small', K: 'positive', E: 'negative' };
  const JT = [[2.0, -1.5, 0.8, 0.8], [-1.5, -0.3, 0.6, 0.6], [0.8, 0.6, 1.5, -1.8], [0.8, 0.6, -1.8, 1.5]];
  const ENVT = { b: [-1.0, -1.0, 1.5, 1.5], e: [1.5, 1.0, -0.8, -0.8] };
  P.toyPairEnergy = function (ca, cb) { return JT['BSKE'.indexOf(ca)]['BSKE'.indexOf(cb)]; };
  P.burial = function (ca, ss) {
    const c = P.centroid(ca);
    return ca.map(function (x, i) {
      if (ss[i] !== 'H') return false;
      let ax = [0, 0, 0], m = 0;
      for (let k = i - 3; k <= i + 3; k++) if (k >= 0 && k < ca.length && ss[k] === 'H') { ax = add(ax, ca[k]); m++; }
      ax = scale(ax, 1 / m);
      const radial = unit([x[0] - ax[0], x[1] - ax[1], 0]), toC = unit([c[0] - ax[0], c[1] - ax[1], 0]);
      return dot(radial, toC) > 0.2;
    });
  };
  // returns { cls: [strings over BSKE], seqs: [amino-acid strings] }
  P.simulateMSA = function (L, couplings, buried, opt) {
    opt = opt || {};
    const N = opt.n || 200, steps = opt.steps || 6, T = opt.T || 1.8, r = P.rng(opt.seed || 11);
    const nb = []; for (let i = 0; i < L; i++) nb.push([]);
    couplings.forEach(function (c) { nb[c[0]].push(c[1]); nb[c[1]].push(c[0]); });
    const env = buried.map(function (b) { return b ? ENVT.b : ENVT.e; });
    const loc = function (s, i, a) { let e = env[i][a]; for (const j of nb[i]) e += JT[a][s[j]]; return e; };
    let s = []; for (let i = 0; i < L; i++) s.push(Math.floor(r() * 4));
    const mc = function (k) { for (let q = 0; q < k * L; q++) { const i = Math.floor(r() * L), a = Math.floor(r() * 4), d = loc(s, i, a) - loc(s, i, s[i]); if (d <= 0 || r() < Math.exp(-d / T)) s[i] = a; } };
    mc(50);
    const cls = [], seqs = [];
    while (cls.length < N) {
      mc(steps);
      const c = s.map(function (a) { return 'BSKE'[a]; }).join('');
      cls.push(c);
      seqs.push(c.split('').map(function (k) { const L2 = P.TOY_LETTERS[k]; return L2[Math.floor(r() * L2.length)]; }).join(''));
    }
    return { cls: cls, seqs: seqs };
  };
  // mutual information with average-product correction (APC), the classic coevolution statistic
  P.coevolution = function (msa, opt) {
    opt = opt || {};
    const L = msa[0].length, N = msa.length, AA = opt.alphabet || (P.AA + '-'), q = AA.length, idx = {};
    for (let k = 0; k < q; k++) idx[AA[k]] = k;
    const lam = opt.pseudo == null ? 0.5 : opt.pseudo;
    const X = msa.map(function (s) { return s.split('').map(function (c) { return idx[c] == null ? q - 1 : idx[c]; }); });
    const fi = []; for (let i = 0; i < L; i++) { const f = new Float64Array(q).fill(lam / q); for (let n = 0; n < N; n++) f[X[n][i]] += 1; const z = N + lam; for (let a = 0; a < q; a++) f[a] /= z; fi.push(f); }
    const MI = []; for (let i = 0; i < L; i++) MI.push(new Float64Array(L));
    for (let i = 0; i < L; i++) for (let j = i + 1; j < L; j++) {
      const f = new Float64Array(q * q).fill(lam / (q * q));
      for (let n = 0; n < N; n++) f[X[n][i] * q + X[n][j]] += 1;
      const z = N + lam; let mi = 0;
      for (let a = 0; a < q; a++) for (let b = 0; b < q; b++) { const p = f[a * q + b] / z; if (p > 0) mi += p * Math.log(p / (fi[i][a] * fi[j][b])); }
      MI[i][j] = MI[j][i] = mi;
    }
    let tot = 0; const row = new Float64Array(L);
    for (let i = 0; i < L; i++) { for (let j = 0; j < L; j++) if (i !== j) row[i] += MI[i][j]; row[i] /= (L - 1); tot += row[i]; }
    const mean = tot / L, APC = [];
    for (let i = 0; i < L; i++) { APC.push(new Float64Array(L)); for (let j = 0; j < L; j++) APC[i][j] = i === j ? 0 : MI[i][j] - row[i] * row[j] / (mean || 1); }
    return { MI: MI, APC: APC, freqs: fi };
  };
  P.topPairs = function (M, minSep, k) {
    const L = M.length, list = [];
    for (let i = 0; i < L; i++) for (let j = i + minSep; j < L; j++) list.push([i, j, M[i][j]]);
    list.sort(function (a, b) { return b[2] - a[2]; });
    return list.slice(0, k);
  };
  P.conservation = function (msa, i, opt_q) {
    const c = {}; msa.forEach(function (s) { c[s[i]] = (c[s[i]] || 0) + 1; });
    let H = 0; Object.values(c).forEach(function (v) { const p = v / msa.length; H -= p * Math.log(p); });
    return 1 - H / Math.log(opt_q || 20);
  };

  // ---------------------------------------------------------------- distance geometry: coordinates from (partial) distances
  // minimise  Σ w_ij (|x_i − x_j| − D_ij)²  by gradient descent; returns a stepper
  P.distanceGeometry = function (L, restraints, seed) {
    const r = P.rng(seed || 1); let X = [];
    for (let i = 0; i < L; i++) X.push([8 * P.gauss(r), 8 * P.gauss(r), 8 * P.gauss(r)]);
    function step(lr, n) {
      for (let it = 0; it < (n || 1); it++) {
        const G = X.map(function () { return [0, 0, 0]; }); let E = 0;
        restraints.forEach(function (c) {
          const i = c[0], j = c[1], D = c[2], w = c[3] == null ? 1 : c[3], mode = c[4] || 'eq';
          const d = sub(X[i], X[j]), l = norm(d) || 1e-6; let e = l - D;
          if (mode === 'min' && e > 0) return; if (mode === 'max' && e < 0) return;
          E += w * e * e; const g = scale(d, 2 * w * e / l);
          G[i] = add(G[i], g); G[j] = sub(G[j], g);
        });
        for (let i = 0; i < L; i++) X[i] = sub(X[i], scale(G[i], lr));
        step.energy = E;
      }
      return X;
    }
    step.coords = function () { return X; };
    step.set = function (Y) { X = Y.map(function (y) { return y.slice(); }); };
    return step;
  };

  // classical multidimensional scaling: coordinates whose distances best match D
  P.mds = function (D) {
    const n = D.length, B = [], r = new Array(n).fill(0); let a = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { r[i] += D[i][j] * D[i][j] / n; a += D[i][j] * D[i][j] / (n * n); }
    for (let i = 0; i < n; i++) { B.push([]); for (let j = 0; j < n; j++) B[i].push(-0.5 * (D[i][j] * D[i][j] - r[i] - r[j] + a)); }
    const e = P.jacobi(B), idx = e.values.map(function (v, i) { return [v, i]; }).sort(function (x, y) { return y[0] - x[0]; }).slice(0, 3);
    return D.map(function (_, i) { return idx.map(function (vk) { return e.vectors[i][vk[1]] * Math.sqrt(Math.max(vk[0], 0)); }); });
  };
  // a pre-AlphaFold-style pipeline: local (secondary-structure) distances + predicted contacts
  //  -> shortest-path distance bounds -> MDS -> gradient-descent refinement.  Returns {start, steps(fn), coords}
  P.foldFromContacts = function (toy, pairs, opt) {
    opt = opt || {}; const L = toy.L, D = [], base = [];
    for (let i = 0; i < L; i++) { D.push(new Array(L).fill(1e9)); D[i][i] = 0; }
    for (let i = 0; i < L; i++) for (let k = 1; k <= 4; k++) {
      const j = i + k; if (j >= L) break;
      if (k === 1 || (toy.ss[i] === 'H' && toy.ss[j] === 'H')) { const d = dist(toy.ca[i], toy.ca[j]); base.push([i, j, d, k === 1 ? 10 : 3]); D[i][j] = D[j][i] = d; }
    }
    pairs.forEach(function (p) { if (D[p[0]][p[1]] > 6) D[p[0]][p[1]] = D[p[1]][p[0]] = 6; });
    for (let k = 0; k < L; k++) for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) if (D[i][k] + D[k][j] < D[i][j]) D[i][j] = D[i][k] + D[k][j];
    const X0 = P.mds(D);
    const rs = base.concat(pairs.map(function (p) { return [p[0], p[1], 7, 1, 'max']; }));
    for (let i = 0; i < L; i++) for (let j = i + 3; j < L; j++) rs.push([i, j, 4.5, 1, 'min']);
    const st = P.distanceGeometry(L, rs, 1); st.set(X0);
    return { start: X0, step: st, bounds: D };
  };
  P.mirror = function (X) { return X.map(function (x) { return [-x[0], x[1], x[2]]; }); };

  // ---------------------------------------------------------------- 3-D viewer (SVG, orthographic, drag to rotate)
  P.Viewer = function (host, opt) {
    opt = opt || {};
    const W = opt.width || 640, H = opt.height || 380;
    const wrap = document.createElement('div'); wrap.className = 'viewer'; host.appendChild(wrap);
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H); wrap.appendChild(svg);
    if (opt.hint !== false) { const h = document.createElement('div'); h.className = 'hint'; h.textContent = 'drag to rotate'; wrap.appendChild(h); }
    let R = opt.rot || P.rotEuler(0.4, -0.5, 0.2), layers = [], zoom = opt.zoom || null, centre = null, spin = !!opt.spin, raf = null, drag = null;
    const view = { svg: svg, wrap: wrap };
    function S(tag, a) { const e = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; }
    view.set = function (ls, keepCentre) {
      layers = ls;
      if (!keepCentre || !centre) {
        const ref = opt.centreOn || (ls[0] && ls[0].pts) || [];
        const all = ref.length ? ref : [];
        centre = all.length ? P.centroid(all) : [0, 0, 0];
        if (!opt.zoom) { let m = 1; all.forEach(function (p) { m = Math.max(m, norm(sub(p, centre))); }); zoom = Math.min(W, H) * 0.44 / m; }
      }
      view.draw();
    };
    view.setZoomFrom = function (pts) { const c = P.centroid(pts); centre = c; let m = 1; pts.forEach(function (p) { m = Math.max(m, norm(sub(p, c))); }); zoom = Math.min(W, H) * 0.44 / m; };
    view.project = function (p) { const q = mv(R, sub(p, centre)); return [W / 2 + q[0] * zoom, H / 2 - q[1] * zoom, q[2]]; };
    view.draw = function () {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const items = [];
      layers.forEach(function (L) {
        if (!L.pts) return;
        const pr = L.pts.map(view.project), col = function (i) { return typeof L.color === 'function' ? L.color(i) : (L.color || 'var(--accent)'); };
        const op = L.opacity == null ? 1 : L.opacity, wd = L.width || 5;
        if (L.mode === 'points' || L.mode === 'both' || L.mode === 'tube+points') pr.forEach(function (p, i) { items.push({ z: p[2], k: 'c', p: p, c: col(i), r: (L.r || 3.2) * (1 + p[2] * 0.004 * zoom / 10), op: op }); });
        if (!L.mode || L.mode === 'tube' || L.mode === 'both' || L.mode === 'tube+points') for (let i = 0; i < pr.length - 1; i++) {
          if (L.breaks && L.breaks[i]) continue;
          items.push({ z: (pr[i][2] + pr[i + 1][2]) / 2, k: 'l', a: pr[i], b: pr[i + 1], c: col(i), w: wd, op: op, dash: L.dash });
        }
        (L.segments || []).forEach(function (s) { const a = view.project(s[0]), b = view.project(s[1]); items.push({ z: (a[2] + b[2]) / 2 + 0.01, k: 'l', a: a, b: b, c: s[2] || col(0), w: s[3] || 1.5, op: s[4] == null ? op : s[4], dash: s[5] }); });
        (L.tris || []).forEach(function (t) { const a = view.project(t[0]), b = view.project(t[1]), c = view.project(t[2]); items.push({ z: (a[2] + b[2] + c[2]) / 3, k: 't', pts: [a, b, c], c: t[3] || col(0), op: t[4] == null ? 0.85 : t[4] }); });
        (L.labels || []).forEach(function (lb) { const a = view.project(lb[0]); items.push({ z: a[2] + 100, k: 'x', p: a, s: lb[1], c: lb[2] }); });
      });
      items.sort(function (a, b) { return a.z - b.z; });
      items.forEach(function (it) {
        const shade = 0.75 + 0.25 * Math.max(-1, Math.min(1, it.z * zoom / 120));
        if (it.k === 'l') S('line', { x1: it.a[0], y1: it.a[1], x2: it.b[0], y2: it.b[1], stroke: it.c, 'stroke-width': it.w * (0.85 + 0.3 * (shade - 0.75) * 4 / 4), 'stroke-linecap': 'round', opacity: it.op * (0.55 + 0.45 * shade), 'stroke-dasharray': it.dash || '' });
        else if (it.k === 'c') S('circle', { cx: it.p[0], cy: it.p[1], r: Math.max(1, it.r), fill: it.c, stroke: 'var(--panel)', 'stroke-width': 0.8, opacity: it.op * (0.6 + 0.4 * shade) });
        else if (it.k === 't') S('polygon', { points: it.pts.map(function (p) { return p[0] + ',' + p[1]; }).join(' '), fill: it.c, stroke: 'var(--ink-soft)', 'stroke-width': 0.6, opacity: it.op });
        else if (it.k === 'x') { const t = S('text', { x: it.p[0] + 6, y: it.p[1] - 6, 'font-size': 12, class: 'svg-text', 'font-weight': 600 }); t.textContent = it.s; if (it.c) t.setAttribute('style', 'fill:' + it.c); }
      });
      if (view.onDraw) view.onDraw();
    };
    view.rotate = function (dx, dy) { R = mm(P.rotAxis([1, 0, 0], dy), mm(P.rotAxis([0, 1, 0], dx), R)); view.draw(); };
    view.getRot = function () { return R; };
    view.setRot = function (r) { R = r; view.draw(); };
    wrap.addEventListener('pointerdown', function (e) { drag = [e.clientX, e.clientY]; wrap.setPointerCapture(e.pointerId); spinPause = true; });
    wrap.addEventListener('pointermove', function (e) { if (!drag) return; const dx = e.clientX - drag[0], dy = e.clientY - drag[1]; drag = [e.clientX, e.clientY]; view.rotate(dx * 0.01, dy * 0.01); });
    wrap.addEventListener('pointerup', function () { drag = null; spinPause = false; });
    wrap.addEventListener('pointercancel', function () { drag = null; spinPause = false; });
    let spinPause = false;
    view.spin = function (on) {
      spin = on; if (raf) cancelAnimationFrame(raf); raf = null;
      if (!on) return;
      let last = performance.now();
      const tick = function (t) {
        if (!spin) return;
        if (!document.body.contains(wrap)) return;
        const dt = Math.min(50, t - last); last = t;
        if (!spinPause && isVisible()) view.rotate(dt * 0.00035, 0);
        raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    };
    function isVisible() { const r = wrap.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight; }
    if (spin) view.spin(true);
    return view;
  };

  // ---------------------------------------------------------------- the shared toy instance
  P.toy = function () {
    if (P._toy) return P._toy;
    const b = P.makeBundle({ helixLen: 14, loopLen: 4 });
    const L = b.ca.length, buried = P.burial(b.ca, b.ss);
    const contacts = P.contacts(b.ca, 8, 3), longRange = contacts.filter(function (c) { return c[1] - c[0] >= 6; });
    // the "native" (query) sequence is one sample of the toy evolution, with loop residues set to G/N/D/P/S
    const nat = P.simulateMSA(L, longRange, buried, { n: 1, seed: 2024, steps: 30 });
    const r = P.rng(9);
    const seq = nat.seqs[0].split('').map(function (a, i) { return b.ss[i] === 'L' ? 'GNDPS'[Math.floor(r() * 5)] : a; }).join('');
    const bb = P.backboneFromCA(b.ca);
    P._toy = { ca: b.ca, ss: b.ss, seq: seq, cls: nat.cls[0], bb: bb, frames: P.framesOf(bb), contacts: contacts, longRange: longRange, buried: buried, L: L };
    P._toy.msa = function (n, seed) { return P.simulateMSA(L, longRange, buried, { n: n, seed: seed || 11 }); };
    return P._toy;
  };
})();
