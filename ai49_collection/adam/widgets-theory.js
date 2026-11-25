/* Adam Interactive Web Book — Part III widgets (convergence, paper §4 and Appendix 10.1) and Part IV widgets
   (relatives and extensions, §5 and §7). Uses Book, ML, Opt and window.AD (widgets-found.js). */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, Opt = window.Opt, AD = window.AD;

  // ================================================================== ch-online
  W.regretGame = function (host) {
    Book.head(host, 'Play the online game, then let Adam play',
      'Each round you choose a number θ_t first. Only then is the round’s cost revealed: f_t(θ) = |θ − c_t|, where c_t is a hidden, noisy target. Regret compares your total cost with the best single θ chosen in hindsight. 40 rounds.');
    const T = 40, r = ML.rng(77), c = ML.range(T).map(function () { return Book.clamp(0.35 + r.normal(0, 0.3), -1, 1); });
    const st = { th: 0, you: [], t: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    const sl = Book.slider(ctl, 'your θ_t', -1, 1, 0.01, st.th, function (v) { st.th = v; }, function (v) { return v.toFixed(2); });
    const play = Book.button(ctl, 'Commit θ and reveal c_t', function () { if (st.t >= T) return; st.you.push(st.th); st.t++; draw(); }, 'primary');
    Book.button(ctl, 'Auto-play the rest with “last target”', function () { while (st.t < T) { st.you.push(st.t ? c[st.t - 1] : 0); st.t++; } draw(); });
    Book.button(ctl, 'Restart', function () { st.you = []; st.t = 0; draw(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function regretCurve(thetas, n) {
      const R = []; let cost = 0;
      for (let t = 0; t < n; t++) {
        cost += Math.abs(thetas[t] - c[t]);
        const seen = c.slice(0, t + 1).slice().sort(function (a, b) { return a - b; }), med = seen[Math.floor(t / 2)];
        let best = 0; for (let k = 0; k <= t; k++) best += Math.abs(med - c[k]);
        R.push([t + 1, cost - best]);
      }
      return R;
    }
    // Adam plays the same game: α_t = α/√t, β1 = 0.9 (Theorem 4.1 setting with λ close to 1)
    const P = Opt.vec([0]), o = Opt.make('adam', [P], { lr: 0.3, decay: true, lambda: 1 - 1e-8 }), adamTh = [];
    for (let t = 0; t < T; t++) { adamTh.push(P.data[0]); P.grad[0] = P.data[0] > c[t] ? 1 : -1; o.step(); P.data[0] = Book.clamp(P.data[0], -1, 1); }
    function draw() {
      const n = st.t, Ra = regretCurve(adamTh, T), Ry = regretCurve(st.you, n);
      const sers = [{ name: 'Adam', color: Opt.colors.adam, marker: 'none', points: Ra }];
      if (n) sers.push({ name: 'you', color: Book.palette[1], points: Ry, r: 3 });
      AD.lines(plot, sers, { width: 640, height: 220, xmin: 1, xmax: T, xlabel: 'round t', ylabel: 'regret R(t)', xticks: [1, 10, 20, 30, 40], mr: 60, ydec: 1 });
      play.disabled = n >= T;
      const last = n ? 'round ' + n + ': you chose ' + st.you[n - 1].toFixed(2) + ', target was c = <b>' + c[n - 1].toFixed(2) + '</b>, cost ' + Math.abs(st.you[n - 1] - c[n - 1]).toFixed(2) + '<br>' : 'Choose θ₁ and commit. Nothing about c₁ is known yet.<br>';
      out.innerHTML = last + (n ? 'your regret after ' + n + ' rounds: <b>' + Ry[n - 1][1].toFixed(2) + '</b> (average per round ' + (Ry[n - 1][1] / n).toFixed(3) + ') · ' : '') +
        'Adam after 40 rounds: <b>' + Ra[T - 1][1].toFixed(2) + '</b> (average ' + (Ra[T - 1][1] / T).toFixed(3) + ')' +
        '<br><span class="note">Regret can be negative when you get lucky. A good online method keeps R(T) growing slower than T, so the average regret R(T)/T goes to 0: on average you do as well as the best fixed choice.</span>';
    }
    draw();
  };

  // ================================================================== ch-convex
  const FUNCS = {
    sq: { label: 'x² (convex)', f: function (x) { return x * x; }, d: function (x) { return 2 * x; } },
    abs: { label: '|x| (convex, kink at 0)', f: Math.abs, d: function (x) { return x > 0 ? 1 : x < 0 ? -1 : 0; } },
    exp: { label: 'eˣ (convex)', f: Math.exp, d: Math.exp },
    soft: { label: 'log(1 + eˣ) (convex: logistic loss)', f: function (x) { return Math.log(1 + Math.exp(x)); }, d: function (x) { return ML.sigmoid(x); } },
    sin: { label: 'sin(2x) (not convex)', f: function (x) { return Math.sin(2 * x); }, d: function (x) { return 2 * Math.cos(2 * x); } },
    w: { label: 'x⁴ − 2x² (not convex)', f: function (x) { return x * x * x * x - 2 * x * x; }, d: function (x) { return 4 * x * x * x - 4 * x; } }
  };
  W.convexChord = function (host) {
    Book.head(host, 'Convex: the chord lies above, the tangent lies below',
      'Pick two points x and y and a mixing weight λ. Definition 10.1: the chord value λf(x) + (1−λ)f(y) must be at least f(λx + (1−λ)y). Lemma 10.2: the tangent line at x never rises above the curve.');
    const st = { fn: 'sq', x: -1.5, y: 1.2, lam: 0.4 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'function', Object.keys(FUNCS).map(function (k) { return { value: k, label: FUNCS[k].label }; }), st.fn, function (v) { st.fn = v; draw(); });
    Book.slider(ctl, 'x', -2, 2, 0.05, st.x, function (v) { st.x = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'y', -2, 2, 0.05, st.y, function (v) { st.y = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'λ', 0, 1, 0.01, st.lam, function (v) { st.lam = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const F = FUNCS[st.fn], xs = ML.linspace(-2.2, 2.2, 200), ys = xs.map(F.f);
      const lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys), pad = (hi - lo) * 0.15 + 0.2;
      const z = st.lam * st.x + (1 - st.lam) * st.y, chord = st.lam * F.f(st.x) + (1 - st.lam) * F.f(st.y), fz = F.f(z);
      const tan = function (u) { return F.f(st.x) + F.d(st.x) * (u - st.x); };
      let worst = Infinity; xs.forEach(function (u) { worst = Math.min(worst, F.f(u) - tan(u)); });
      const res = AD.lines(plot, [
        { name: 'f', color: 'var(--accent)', marker: 'none', width: 2.6, points: xs.map(function (u, i) { return [u, ys[i]]; }) },
        { name: 'tangent at x', color: Book.palette[4], marker: 'none', dash: '5 4', width: 1.5, points: [[-2.2, tan(-2.2)], [2.2, tan(2.2)]] },
        { name: 'chord', color: Book.palette[1], width: 1.8, points: [[st.x, F.f(st.x)], [st.y, F.f(st.y)]] }
      ], { width: 640, height: 260, xmin: -2.2, xmax: 2.2, ymin: lo - pad, ymax: hi + pad, xticks: [-2, -1, 0, 1, 2], xlabel: 'input', ylabel: 'f', mr: 100, ydec: 1 });
      S('circle', { cx: res.X(z), cy: res.Y(chord), r: 5, fill: Book.palette[1] }, res.svg);
      S('circle', { cx: res.X(z), cy: res.Y(fz), r: 5, fill: 'var(--accent)', stroke: 'var(--panel)' }, res.svg);
      S('line', { x1: res.X(z), x2: res.X(z), y1: res.Y(chord), y2: res.Y(fz), stroke: 'var(--muted)', 'stroke-dasharray': '2 2' }, res.svg);
      const ok1 = chord >= fz - 1e-12, ok2 = worst >= -1e-9;
      out.innerHTML = 'λx + (1−λ)y = ' + z.toFixed(2) + ': chord value <b>' + chord.toFixed(3) + '</b> ' + (ok1 ? '≥' : '<') + ' f = <b>' + fz.toFixed(3) + '</b> ' + (ok1 ? '✓' : '✗ (the definition fails here)') +
        '<br>tangent at x: lowest value of f − tangent over the plot = <b>' + worst.toFixed(3) + '</b> ' + (ok2 ? '✓ never below' : '✗ the curve dips below its tangent') +
        '<br><span class="note">For a convex f both checks pass for every choice. Lemma 10.2 is what the proof uses: f(θ_t) − f(θ*) ≤ ∇f(θ_t)·(θ_t − θ*), a bound that needs only the gradient Adam already has.</span>';
    }
    draw();
  };

  // ================================================================== ch-theorem
  W.theoremTerms = function (host) {
    Book.head(host, 'Theorem 4.1 on a real run',
      'A sparse online problem in d = 20 dimensions (this book’s toy): at each step only some coordinates have a cost |θ_i − c_t,i|, so most gradient entries are 0. Adam runs with α_t = α/√t and β1,t = β1·λ^(t−1). The bars compare the actual regret with the three terms of the bound.');
    const st = { T: 2000, a: 0.3, b1: 0.9, b2: 0.999, lam: 0.99 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'T', ['500', '1000', '2000', '4000', '8000'], '2000', function (v) { st.T = +v; draw(); });
    Book.slider(ctl, 'α', 0.02, 1, 0.01, st.a, function (v) { st.a = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'β1', 0, 0.97, 0.01, st.b1, function (v) { st.b1 = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.select(ctl, 'β2', ['0.9', '0.99', '0.999', '0.9999'], '0.999', function (v) { st.b2 = +v; draw(); });
    Book.slider(ctl, 'λ', 0.5, 0.999, 0.001, st.lam, function (v) { st.lam = v; draw(); }, function (v) { return v.toFixed(3); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const prob = Opt.onlineProblem(20, st.T, 31), run = Opt.playOnline(prob, 'adam', { lr: st.a, b1: st.b1, b2: st.b2, decay: true, lambda: st.lam });
      const TT = Opt.theoremTerms(run, prob, st.a, st.b1, st.b2, st.lam), R = run.R[run.R.length - 1];
      const lg = function (x) { return Math.log10(Math.max(x, 1e-3)) + 3; };
      Book.chart.bars(chart, [
        { label: 'actual\nregret R(T)', value: lg(R), note: AD.ft(R, 1), color: 'var(--accent)' },
        { label: 'term 1\n(D², √v̂)', value: lg(TT.t1), note: AD.ft(TT.t1, 0), color: Book.palette[0] },
        { label: 'term 2\n(‖g₁:T,i‖)', value: lg(TT.t2), note: AD.ft(TT.t2, 0), color: Book.palette[4] },
        { label: 'term 3\n(λ, D∞²)', value: lg(TT.t3), note: AD.ft(TT.t3, 0), color: Book.palette[1] },
        { label: 'T (no learning\nat all ~ cost T)', value: lg(st.T), note: String(st.T), color: Book.palette[7] }
      ], { height: 240, max: Math.max(9, Math.ceil(lg(Math.max(TT.t1, TT.t2, TT.t3)))), ylabel: 'size (log scale)', tickFmt: function (v) { return AD.p10(Math.round(v - 3)); } });
      const gamOk = TT.gamma < 1;
      out.innerHTML = 'γ = β1²/√β2 = <b>' + TT.gamma.toFixed(3) + '</b> ' + (gamOk ? '< 1 ✓' : '≥ 1 ✗: the theorem does not apply') +
        ' · D = 2√d = ' + TT.D.toFixed(2) + ', D∞ = 2, G∞ = 1, G = ' + TT.G.toFixed(2) +
        '<br>Σ_i ‖g₁:T,i‖₂ = <b>' + TT.sumG.toFixed(1) + '</b> vs its worst case d·G∞·√T = ' + TT.worst.toFixed(1) + ' (sparse gradients make it smaller)' +
        '<br>actual regret <b>' + R.toFixed(1) + '</b> ≤ bound <b>' + AD.f(TT.total, 0) + '</b> ✓ · average regret R(T)/T = <b>' + (R / st.T).toFixed(4) + '</b>' +
        '<br><span class="note">The bound holds but is very loose: it is a worst-case guarantee for any convex costs. Term 3 explodes as λ → 1 (the paper’s example λ = 1 − 10⁻⁸ makes 1/(1−λ)² = 10¹⁶), yet it does not grow with T: only terms 1 and 2 do, like √T.</span>';
    }
    draw();
  };

  // ================================================================== ch-sparse
  W.sparseBound = function (host) {
    Book.head(host, 'Sparse gradients shrink Σ‖g₁:T,i‖₂',
      'Gradient entries are ±1 (so G∞ = 1) when coordinate i is “on”, else 0. Choose how often coordinates are on. The sum Σ_i ‖g₁:T,i‖₂ appears in the bound; its worst case is d·√T.');
    const st = { kind: 'power', ld: 2, q: 0.05, T: 10000, pw: 2 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'how often', [{ value: 'equal', label: 'every coordinate: probability q' }, { value: 'power', label: 'coordinate i: probability i^(−a) (Duchi et al.’s example)' }], st.kind, function (v) { st.kind = v; draw(); });
    Book.slider(ctl, 'dimensions d', 1, 5, 0.1, st.ld, function (v) { st.ld = v; draw(); }, function (v) { return Math.round(Math.pow(10, v)).toLocaleString('en-US'); });
    Book.slider(ctl, 'q', 0.001, 1, 0.001, st.q, function (v) { st.q = v; draw(); }, function (v) { return v.toFixed(3); });
    Book.slider(ctl, 'a', 1, 3, 0.1, st.pw, function (v) { st.pw = v; draw(); }, function (v) { return v.toFixed(1); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    // ‖g₁:T,i‖₂ = √(number of steps where coordinate i is on); its expected value ≈ √(T·p_i)
    function sum(d, T) { let s = 0; for (let i = 1; i <= d; i++) { const p = st.kind === 'equal' ? st.q : Math.min(1, Math.pow(i, -st.pw)); s += Math.sqrt(T * p); } return s; }
    function draw() {
      const d = Math.round(Math.pow(10, st.ld)), Ts = [100, 300, 1000, 3000, 10000, 30000, 100000];
      const S1 = Ts.map(function (T) { return [T, sum(d, T)]; }), S2 = Ts.map(function (T) { return [T, d * Math.sqrt(T)]; });
      AD.lines(plot, [
        { name: 'worst case d√T', color: Book.palette[7], points: S2.map(function (p) { return [p[0], Math.log10(p[1])]; }) },
        { name: 'Σ‖g₁:T,i‖₂', color: 'var(--accent)', points: S1.map(function (p) { return [p[0], Math.log10(p[1])]; }) }
      ], { width: 640, height: 220, logx: true, xmin: 100, xmax: 100000, xticks: [100, 1000, 10000, 100000], xfmt: function (v) { return v.toLocaleString('en-US'); }, xlabel: 'T (log)', ylabel: 'size (log₁₀)', mr: 120, ydec: 1 });
      const s = sum(d, 10000), w = d * 100;
      out.innerHTML = 'at T = 10,000 with d = ' + d.toLocaleString('en-US') + ': Σ‖g₁:T,i‖₂ ≈ <b>' + s.toFixed(0) + '</b> vs worst case <b>' + w.toLocaleString('en-US') + '</b> → ratio <b>' + (s / w).toFixed(4) + '</b>' +
        (st.kind === 'power' && st.pw >= 2 ? '<br>with a = 2 the sum is about √T·(1 + 1/2 + … + 1/d) ≈ √T·(ln d + 0.58) = ' + (100 * (Math.log(d) + 0.5772)).toFixed(0) + ' at T = 10,000: the “log d” in O(log d·√T)' : '') +
        '<br><span class="note">Both lines grow like √T (same slope on a log-log plot); sparsity lowers the constant in front, and with power-law sparsity that constant grows only like log d instead of d. A non-adaptive method’s bound has √d·√T instead.</span>';
    }
    draw();
  };

  W.avgRegret = function (host) {
    Book.head(host, 'Average regret goes to zero (Corollary 4.2)',
      'Four methods play the same sparse online problem (d = 20) for 20,000 rounds. The chart shows the average regret R(T)/T on log-log axes, next to a reference line of slope −1/2 (a constant ÷ √T).');
    const st = { a: 0.3 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'α (all methods)', 0.05, 1, 0.05, st.a, function (v) { st.a = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const T = 20000, prob = Opt.onlineProblem(20, T, 31);
      const runs = [
        ['Adam (α/√t)', Opt.colors.adam, Opt.playOnline(prob, 'adam', { lr: st.a, decay: true, lambda: 1 - 1e-8 })],
        ['AdaGrad', Opt.colors.adagrad, Opt.playOnline(prob, 'adagrad', { lr: st.a })],
        ['SGD (α/√t)', Opt.colors.nesterov, Opt.playOnline(prob, 'sgd', { lr: st.a, lrDecay: true })],
        ['SGD (constant α)', Opt.colors.sgd, Opt.playOnline(prob, 'sgd', { lr: st.a })]
      ];
      const Ts = [10, 30, 100, 300, 1000, 3000, 10000, 20000];
      const sers = runs.map(function (u) { return { name: u[0], color: u[1], points: Ts.map(function (t) { return [t, Math.log10(Math.max(u[2].R[t - 1] / t, 1e-4))]; }) }; });
      const c0 = runs[0][2].R[99] / 100 * 10;
      sers.push({ name: '∝ 1/√T', color: 'var(--ink)', dash: '5 4', marker: 'none', width: 1.2, points: Ts.map(function (t) { return [t, Math.log10(c0 / Math.sqrt(t))]; }) });
      AD.lines(plot, sers, { width: 640, height: 240, logx: true, xmin: 10, xmax: 20000, xticks: [10, 100, 1000, 10000], xfmt: function (v) { return v.toLocaleString('en-US'); }, xlabel: 'T (log)', ylabel: 'log₁₀ R(T)/T', mr: 130, ydec: 1 });
      out.innerHTML = runs.map(function (u) { return u[0] + ': R/T = <b>' + (u[2].R[T - 1] / T).toFixed(4) + '</b>'; }).join(' · ') +
        '<br><span class="note">Toy problem with sparse gradients. The adaptive methods (Adam, AdaGrad) reach lower average regret because they give rarely-active coordinates larger steps. Constant-step SGD keeps jittering and stalls; decaying SGD is slow on the rare coordinates.</span>';
    }
    draw();
  };

  // ================================================================== ch-proof
  W.lemma103 = function (host) {
    Book.head(host, 'Lemma 10.3, checked on sequences you choose',
      'For one coordinate i, the lemma says Σ_t √(g_t²/t) ≤ 2·G∞·‖g₁:T‖₂ for every T. The chart shows both sides growing with T.');
    const st = { kind: 'random', T: 300, seed: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'sequence', [{ value: 'random', label: 'random in [−1, 1]' }, { value: 'sparse', label: 'sparse: ±1 with prob. 0.05' }, { value: 'early', label: 'large early, then tiny' }, { value: 'late', label: 'tiny early, then large' }, { value: 'const', label: 'constant 1' }], st.kind, function (v) { st.kind = v; draw(); });
    Book.slider(ctl, 'T', 10, 2000, 10, st.T, function (v) { st.T = v; draw(); }, function (v) { return v; });
    Book.button(ctl, 'New sequence', function () { st.seed++; draw(); }, 'primary');
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(st.seed), g = [];
      for (let t = 1; t <= st.T; t++) {
        let x = r.uniform(-1, 1);
        if (st.kind === 'sparse') x = r.next() < 0.05 ? (r.next() < 0.5 ? -1 : 1) : 0; else if (st.kind === 'early') x = t <= st.T / 10 ? 1 : 0.01; else if (st.kind === 'late') x = t > st.T * 0.9 ? 1 : 0.01; else if (st.kind === 'const') x = 1;
        g.push(x);
      }
      const Ginf = Math.max.apply(null, g.map(Math.abs)) || 1; let L = 0, n2 = 0; const lhs = [], rhs = [];
      g.forEach(function (x, k) { L += Math.sqrt(x * x / (k + 1)); n2 += x * x; lhs.push([k + 1, L]); rhs.push([k + 1, 2 * Ginf * Math.sqrt(n2)]); });
      AD.lines(plot, [{ name: '2·G∞·‖g₁:T‖₂', color: Book.palette[1], marker: 'none', points: rhs }, { name: 'Σ √(g_t²/t)', color: 'var(--accent)', marker: 'none', points: lhs, ldy: 14 }],
        { width: 640, height: 220, ymin: 0, xlabel: 'T', ylabel: 'value', xticks: [1, Math.round(st.T / 2), st.T], mr: 120, ydec: 1 });
      let viol = 0; lhs.forEach(function (p, k) { if (p[1] > rhs[k][1] + 1e-9) viol++; });
      const T = st.T, gT = g[T - 1], prevL = lhs[T - 2] ? lhs[T - 2][1] : 0;
      out.innerHTML = 'at T = ' + T + ': left side <b>' + L.toFixed(3) + '</b> ≤ right side <b>' + (2 * Ginf * Math.sqrt(n2)).toFixed(3) + '</b> · violations for any T ≤ ' + T + ': <b>' + viol + '</b>' +
        '<br>induction step at T: Σ_{t<T} = ' + prevL.toFixed(3) + ' ≤ 2G∞√(‖g₁:T‖² − g_T²) = ' + (2 * Ginf * Math.sqrt(Math.max(0, n2 - gT * gT))).toFixed(3) + ', then add √(g_T²/T) = ' + Math.abs(gT / Math.sqrt(T)).toFixed(3) +
        '<br><span class="note">G∞ here is the largest |g_t| in the sequence. The 1/√t inside makes late gradients count less, which is what lets the decaying step size α/√t pay off.</span>';
    }
    draw();
  };

  W.proofSteps = function (host) {
    Book.head(host, 'Walk through the proof of Theorem 4.1 on a concrete run',
      'Adam (ε = 0, α_t = α/√t, β1,t = β1λ^(t−1)) plays a one-dimensional online game with costs |θ − c_t| for T = 500 rounds. Each stage of the proof states an inequality; the numbers below are that inequality evaluated on this run.');
    const T = 500, a = 0.3, b1 = 0.9, b2 = 0.999, lam = 0.99, r = ML.rng(42);
    const c = ML.range(T).map(function () { return Book.clamp(0.3 + r.normal(0, 0.4), -1, 1); });
    // run Adam exactly as the proof writes it
    let th = 0, m = 0, v = 0; const H = [];
    for (let t = 1; t <= T; t++) {
      const g = th > c[t - 1] ? 1 : th < c[t - 1] ? -1 : 0, b1t = b1 * Math.pow(lam, t - 1);
      m = b1t * m + (1 - b1t) * g; v = b2 * v + (1 - b2) * g * g;
      const mh = m / (1 - Math.pow(b1, t)), vh = v / (1 - Math.pow(b2, t)), at = a / Math.sqrt(t);
      H.push({ t: t, th: th, g: g, mh: mh, vh: vh, at: at, b1t: b1t });
      th = th - at * mh / Math.sqrt(vh);
    }
    const sorted = c.slice().sort(function (x, y) { return x - y; }), star = sorted[Math.floor((T - 1) / 2)];
    let regret = 0, lin = 0; H.forEach(function (h, k) { regret += Math.abs(h.th - c[k]) - Math.abs(star - c[k]); lin += h.g * (h.th - star); });
    let drops = 0; for (let k = 1; k < T; k++) if (Math.sqrt(H[k].vh) / H[k].at < Math.sqrt(H[k - 1].vh) / H[k - 1].at - 1e-12) drops++;
    let L104 = 0, n2 = 0; H.forEach(function (h) { L104 += h.mh * h.mh / Math.sqrt(h.t * h.vh); n2 += h.g * h.g; });
    const gam = b1 * b1 / Math.sqrt(b2), Ginf = 1;
    const R104stmt = 2 / (1 - gam) / Math.sqrt(1 - b2) * Math.sqrt(n2), R104proof = 2 * Ginf / ((1 - gam) * (1 - gam) * Math.sqrt(1 - b2)) * Math.sqrt(n2);
    let ag = 0; H.forEach(function (h) { ag += h.b1t / (1 - h.b1t) * Math.sqrt(h.t); });
    const k5 = 100, h5 = H[k5], A5 = (h5.th - star), step = h5.at * h5.mh / Math.sqrt(h5.vh);
    const STAGES = [
      ['1 · Convexity turns regret into a sum of gradients (Lemma 10.2)',
        'f_t(θ_t) − f_t(θ*) ≤ g_t·(θ_t − θ*), summed over t.',
        'regret R(T) = <b>' + regret.toFixed(2) + '</b> ≤ Σ g_t(θ_t − θ*) = <b>' + lin.toFixed(2) + '</b> ' + (regret <= lin + 1e-9 ? '✓' : '✗') + ' (θ* = ' + star.toFixed(3) + ')'],
      ['2 · Write the update with the decaying β1,t',
        'θ_{t+1} = θ_t − α_t·m̂_t/√v̂_t, with m_t = β1,t·m_{t−1} + (1 − β1,t)·g_t.',
        'at t = ' + h5.t + ': α_t = ' + h5.at.toFixed(4) + ', β1,t = ' + h5.b1t.toFixed(4) + ', m̂/√v̂ = ' + (h5.mh / Math.sqrt(h5.vh)).toFixed(4) + ' → step = ' + step.toFixed(5)],
      ['3 · Square the distance to θ*',
        '(θ_{t+1} − θ*)² = (θ_t − θ*)² − 2α_t(m̂_t/√v̂_t)(θ_t − θ*) + α_t²(m̂_t/√v̂_t)².',
        'at t = ' + h5.t + ': left ' + Math.pow(H[k5 + 1].th - star, 2).toFixed(6) + ' = right ' + (A5 * A5 - 2 * step * A5 + step * step).toFixed(6) + ' ✓ (pure algebra)'],
      ['4 · Solve for g_t(θ_t − θ*) and use Young’s inequality ab ≤ a²/2 + b²/2',
        'This splits the cross term with m_{t−1} into a squared-distance part and a squared-momentum part.',
        'example: a = ' + A5.toFixed(3) + ', b = ' + (h5.mh).toFixed(3) + ': ab = ' + (A5 * h5.mh).toFixed(4) + ' ≤ a²/2 + b²/2 = ' + ((A5 * A5 + h5.mh * h5.mh) / 2).toFixed(4) + ' ✓'],
      ['5 · Sum over t: the distance terms telescope',
        'Σ_t (√v̂_t/α_t)·[(θ_t − θ*)² − (θ_{t+1} − θ*)²] is bounded by D∞²·√v̂_T/α_T only if √v̂_t/α_t never decreases.',
        '√v̂_t/α_t decreased at <b>' + drops + '</b> of ' + (T - 1) + ' steps in this run. ' + (drops ? 'So this step of the proof is not justified here: the gap found by Reddi, Kale &amp; Kumar (2018), see Chapter 32.' : 'Here it never decreased, so the step is fine for this run.')],
      ['6 · Lemma 10.4 bounds the momentum terms',
        'Σ_t m̂_t²/√(t·v̂_t) ≤ C·‖g₁:T‖₂ with γ = β1²/√β2 < 1.',
        'left side <b>' + L104.toFixed(2) + '</b> · constant as stated, 2/((1−γ)√(1−β2)): ' + R104stmt.toFixed(1) + ' · constant at the end of its proof, 2G∞/((1−γ)²√(1−β2)): ' + R104proof.toFixed(1) + ' (γ = ' + gam.toFixed(3) + ')'],
      ['7 · An arithmetic-geometric series bounds the β1,t terms',
        'Σ_t β1,t/(1 − β1,t)·√t ≤ 1/((1 − β1)(1 − λ)²).',
        'sum over this run <b>' + ag.toFixed(2) + '</b> ≤ ' + (1 / ((1 - b1) * (1 - lam) * (1 - lam))).toFixed(0) + ' ✓ (a constant: it does not grow with T)']
    ];
    const st = { k: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    const prev = Book.button(ctl, '← Previous', function () { st.k = Math.max(0, st.k - 1); draw(); });
    const next = Book.button(ctl, 'Next stage →', function () { st.k = Math.min(STAGES.length - 1, st.k + 1); draw(); }, 'primary');
    const box = el('div', { class: 'out' }, host);
    function draw() {
      const s = STAGES[st.k];
      box.innerHTML = '<b>' + s[0] + '</b><br>' + s[1] + '<br><br>On this run: ' + s[2] + '<br><span class="note">Stage ' + (st.k + 1) + ' of ' + STAGES.length + '. Toy run; the inequalities are the paper’s (Appendix 10.1).</span>';
      prev.disabled = st.k === 0; next.disabled = st.k === STAGES.length - 1;
    }
    draw();
  };

  // ================================================================== ch-related
  W.familyReduce = function (host) {
    Book.head(host, 'AdaGrad is a special case of Adam (with bias correction)',
      'The same sparse online gradients are fed to AdaGrad and to Adam with β1 = 0, β2 = 1 − 10^(−k) and α_t = α/√t. As k grows, Adam’s path converges onto AdaGrad’s. Switch off the bias correction to see the §5 warning come true.');
    const st = { k: 1, bias: true };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'k in β2 = 1 − 10^(−k)', 1, 7, 1, st.k, function (v) { st.k = v; draw(); }, function (v) { return v + ' (β2 = ' + (1 - Math.pow(10, -v)).toPrecision(8) + ')'; });
    const bl = el('label', null, ctl), bc = el('input', { type: 'checkbox' }, bl); bc.checked = true; bl.appendChild(document.createTextNode(' bias correction'));
    bc.addEventListener('change', function () { st.bias = bc.checked; draw(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function run(name, o, prob, T) {
      const P = Opt.vec(new Array(prob.d).fill(0)), opt = Opt.make(name, [P], o), tr = [];
      for (let t = 0; t < T; t++) { prob.g(t, P.data, P.grad); opt.step(); tr.push([t + 1, P.data[0]]); }
      return tr;
    }
    function draw() {
      const T = 300, prob = Opt.onlineProblem(3, T, 8, 0.3, 0.3), a = 0.2;
      const A = run('adagrad', { lr: a, eps: 1e-12 }, prob, T), B = run('adam', { lr: a, b1: 0, b2: 1 - Math.pow(10, -st.k), decay: true, bias: st.bias, eps: 1e-12 }, prob, T);
      let gap = 0; A.forEach(function (p, i) { gap = Math.max(gap, Math.abs(p[1] - B[i][1])); });
      let lim = 1; A.concat(B).forEach(function (p) { if (isFinite(p[1])) lim = Math.max(lim, Math.abs(p[1]) * 1.15); }); lim = Math.min(lim, 50);
      AD.lines(plot, [{ name: 'AdaGrad', color: Opt.colors.adagrad, marker: 'none', width: 3, points: A }, { name: 'Adam', color: Opt.colors.adam, marker: 'none', dash: '5 3', points: B.map(function (p) { return [p[0], Book.clamp(p[1], -lim, lim)]; }), ldy: 14 }],
        { width: 640, height: 230, ymin: -lim, ymax: lim, xlabel: 'step', ylabel: 'θ (coordinate 1)', xticks: [1, 100, 200, 300], mr: 80, ydec: 1 });
      out.innerHTML = 'largest gap between the two paths: <b>' + AD.f(gap, 4) + '</b>' + (st.bias ? '' : ' · first Adam step without correction: |Δ₁| = α·|g|/√((1−β2)g²) = α/√(1−β2) = <b>' + AD.f(a / Math.sqrt(Math.pow(10, -st.k)), 1) + '</b>') +
        '<br><span class="note">With correction, lim β2→1 of v̂_t = (1/t)·Σg², and α/√t ÷ √((1/t)Σg²) = α/√(Σg²): AdaGrad’s step exactly (paper §5). Without correction v_t → 0 as β2 → 1 and the steps become “infinitely large”, as the paper warns.</span>';
    }
    draw();
  };

  W.rmspropMomentum = function (host) {
    Book.head(host, 'RMSProp with momentum vs Adam',
      'RMSProp with momentum (Graves, 2013) rescales each gradient by its running RMS, then applies momentum to the rescaled gradients; it has no bias correction. Adam averages the raw gradients and corrects both averages. Matched settings: μ = β1 = 0.9, ρ = β2, learning rate α(1 − β1).');
    const st = { b2: 0.999, la: Math.log10(0.05) };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'β2 = ρ', ['0.9', '0.99', '0.999', '0.9999'], '0.999', function (v) { st.b2 = +v; draw(); });
    Book.slider(ctl, 'α', -3, -0.5, 0.05, st.la, function (v) { st.la = v; draw(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    const map = AD.contour(host, Opt.surfaces.ravine, { height: 300 }), lg = el('div', null, host), out = el('div', { class: 'out' }, host);
    AD.legend(lg, [{ name: 'RMSProp + momentum', color: Opt.colors.rmsprop }, { name: 'Adam', color: Opt.colors.adam }]);
    function draw() {
      const a = Math.pow(10, st.la);
      const A = Opt.runSurface('ravine', 'adam', { lr: a, b2: st.b2 }, 200, 0, 1), R = Opt.runSurface('ravine', 'rmsprop', { lr: a * 0.1, rho: st.b2, mu: 0.9 }, 200, 0, 1);
      map.draw([{ path: R.path, color: Opt.colors.rmsprop, width: 2.2 }, { path: A.path, color: Opt.colors.adam, width: 2.2 }]);
      const s1 = function (p) { return Math.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1]); };
      out.innerHTML = 'length of the first step: Adam <b>' + AD.f(s1(A.path)) + '</b> · RMSProp + momentum <b>' + AD.f(s1(R.path)) + '</b> · final loss: Adam ' + AD.f(A.loss[A.loss.length - 1]) + ', RMSProp ' + (isNaN(R.path[R.path.length - 1][0]) ? 'blew up' : AD.f(R.loss[R.loss.length - 1])) +
        '<br><span class="note">RMSProp’s v starts at 0 and is uncorrected, so its first rescaled gradient is 1/√(1 − ρ) = ' + (1 / Math.sqrt(1 - st.b2)).toFixed(0) + ' times too big, and its first step is (1 − β1)/√(1 − ρ) = ' + (0.1 / Math.sqrt(1 - st.b2)).toFixed(1) + ' times α per coordinate (Adam’s is exactly α). The closer ρ is to 1, the wilder the start. Toy surface.</span>';
    }
    draw();
  };

  W.precondCompare = function (host) {
    Book.head(host, 'Three ways to divide the gradient',
      'One weight whose gradient has mean μ·c and spread c, where c is an overall scale (think: units of the loss). Compare the expected update size for SGD (no division), Adam (divide by √E[g²], the square root of a diagonal Fisher estimate) and a diagonal natural-gradient step (divide by E[g²]).');
    const st = { mu: 0.5 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'consistency μ', 0, 3, 0.05, st.mu, function (v) { st.mu = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const cs = [-3, -2, -1, 0, 1, 2, 3], lr = 0.01, sers = [['SGD: lr·E[g]', Opt.colors.sgd, function (c) { return lr * st.mu * c; }], ['Adam: α·E[g]/√E[g²]', Opt.colors.adam, function (c) { return lr * st.mu * c / Math.sqrt((st.mu * st.mu + 1) * c * c); }], ['diag. NGD: lr·E[g]/E[g²]', Book.palette[1], function (c) { return lr * st.mu * c / ((st.mu * st.mu + 1) * c * c); }]];
      AD.lines(plot, sers.map(function (s) { return { name: s[0], color: s[1], points: cs.map(function (lc) { return [lc, Math.log10(Math.max(s[2](Math.pow(10, lc)), 1e-12))]; }) }; }),
        { width: 640, height: 230, xmin: -3, xmax: 3, ymin: -6, ymax: 1, yticks: 7, yfmt: function (v) { return AD.p10(Math.round(v)); }, xticks: cs, xfmt: function (v) { return AD.p10(v); }, xlabel: 'gradient scale c', ylabel: 'update size (log)', mr: 160 });
      out.innerHTML = 'Adam’s update size = lr·μ/√(μ² + 1) = <b>' + AD.f(lr * st.mu / Math.sqrt(st.mu * st.mu + 1), 4) + '</b> for every scale c' +
        '<br><span class="note">SGD grows with c, the natural-gradient-like step shrinks like 1/c, and Adam’s is flat. Using the square root of the inverse (rather than the full inverse) is what the paper calls “more conservative” than natural gradient descent (§5).</span>';
    }
    draw();
  };

  // ================================================================== ch-adamax
  W.lpNorm = function (host) {
    Book.head(host, 'From the L² norm to the L^∞ norm',
      'Bars: the last 12 gradient sizes, each discounted by β2^(t−i) (newest on the right). The L^p combination (Σ (β2^(t−i)|g_i|)^p)^(1/p) is computed for the p you choose. As p grows it approaches the largest bar: Eq. 11.');
    const st = { lp: Math.log2(2), seed: 3, b2: 0.9 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'p', 0, 7, 0.1, st.lp, function (v) { st.lp = v; draw(); }, function (v) { return Math.pow(2, v).toFixed(1); });
    Book.slider(ctl, 'β2', 0.5, 0.99, 0.01, st.b2, function (v) { st.b2 = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'New gradients', function () { st.seed++; draw(); }, 'primary');
    const g2 = el('div', { class: 'grid2' }, host), bars = el('div', null, g2), plot = el('div', null, g2), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(st.seed), t = 12, g = ML.range(t).map(function () { return Math.abs(r.normal(0, 1)); });
      const w = g.map(function (x, k) { return Math.pow(st.b2, t - (k + 1)) * x; }), mx = Math.max.apply(null, w);
      const Lp = function (p) { let s = 0; w.forEach(function (x) { s += Math.pow(x / mx, p); }); return mx * Math.pow(s, 1 / p); };
      const p = Math.pow(2, st.lp);
      Book.chart.bars(bars, w.map(function (x, k) { return { label: String(k + 1), value: x, note: '', color: x === mx ? Book.palette[1] : 'var(--accent)' }; }), { width: 320, height: 210, ylabel: 'β2^(t−i)·|g_i|', tickFmt: function (v) { return v.toFixed(1); } });
      const ps = ML.linspace(0, 7, 36);
      AD.lines(plot, [{ name: 'L^p', color: 'var(--accent)', marker: 'none', points: ps.map(function (q) { return [q, Lp(Math.pow(2, q))]; }) }, { name: 'max', color: Book.palette[1], marker: 'none', dash: '5 4', points: [[0, mx], [7, mx]] }, { color: Book.palette[4], points: [[st.lp, Lp(p)]], r: 5 }],
        { width: 320, height: 210, xmin: 0, xmax: 7, xticks: [0, 2, 4, 6], xfmt: function (v) { return 'p=' + Math.pow(2, v); }, xlabel: 'p (log scale)', ylabel: 'value', mr: 40, ydec: 1 });
      // Eq. 12 recursion gives the same max
      let u = 0; g.forEach(function (x) { u = Math.max(st.b2 * u, x); });
      const pre = Math.pow(1 - Math.pow(st.b2, p), 1 / p);
      out.innerHTML = 'p = ' + p.toFixed(1) + ': L^p = <b>' + Lp(p).toFixed(4) + '</b> · with the factor (1 − β2^p)^(1/p) = ' + pre.toFixed(4) + ' of Eq. 9: ' + (pre * Lp(p)).toFixed(4) +
        '<br>limit p → ∞ (Eq. 11): largest discounted bar = <b>' + mx.toFixed(4) + '</b> · recursion u_t = max(β2·u_{t−1}, |g_t|) from u₀ = 0 (Eq. 12): <b>' + u.toFixed(4) + '</b> ✓' +
        '<br><span class="note">The factor (1 − β2^p)^(1/p) tends to 1 as p grows, which is why AdaMax needs no bias correction for u. p = 2 is Adam’s choice (√ of an average of squares).</span>';
    }
    draw();
  };

  W.adamaxStepper = function (host) {
    Book.head(host, 'AdaMax (Algorithm 2) vs Adam on the same gradients',
      'One weight, 300 steps. Pick a gradient pattern; one of them has rare huge outliers. The chart shows each method’s |Δ_t|/α. AdaMax uses u_t = max(β2·u_{t−1}, |g_t|) instead of √v̂_t.');
    const st = { kind: 'outlier' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'pattern', [{ value: 'noisy', label: 'noisy, mean 0.3' }, { value: 'outlier', label: 'noisy + rare ×50 outliers' }, { value: 'sparse', label: 'sparse: 1 every 100 steps' }, { value: 'const', label: 'constant 1' }], st.kind, function (v) { st.kind = v; draw(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(11), T = 300, gs = [];
      for (let t = 1; t <= T; t++) { let g = r.normal(0.3, 1); if (st.kind === 'outlier' && t % 97 === 0) g = 50; if (st.kind === 'sparse') g = t % 100 === 0 ? 1 : 0; if (st.kind === 'const') g = 1; gs.push(g); }
      const go = function (name, a) { const P = Opt.vec([0]), o = Opt.make(name, [P], { lr: a }), pts = []; let mx = 0; gs.forEach(function (g, k) { const p0 = P.data[0]; P.grad[0] = g; o.step(); const d = Math.abs(P.data[0] - p0) / a; mx = Math.max(mx, d); pts.push([k + 1, d]); }); return { pts: pts, mx: mx, o: o }; };
      const A = go('adamax', 0.002), B = go('adam', 0.001);
      AD.lines(plot, [{ name: 'AdaMax', color: Opt.colors.adamax, marker: 'none', points: A.pts }, { name: 'Adam', color: Opt.colors.adam, marker: 'none', points: B.pts, ldy: 14 }, { name: 'α', color: 'var(--ink)', marker: 'none', dash: '5 4', width: 1.2, points: [[1, 1], [T, 1]] }],
        { width: 640, height: 230, ymin: 0, ymax: Math.max(1.3, B.mx * 1.1), xlabel: 'step', ylabel: '|Δt| / α', xticks: [1, 100, 200, 300], mr: 70, ydec: 1 });
      out.innerHTML = 'largest step: AdaMax <b>' + A.mx.toFixed(3) + '·α</b> (paper: |Δ_t| ≤ α) · Adam <b>' + B.mx.toFixed(3) + '·α</b> · AdaMax’s final u = ' + AD.f(A.o.u[0][0], 3) +
        '<br><span class="note">Paper defaults: AdaMax α = 0.002, Adam α = 0.001, β1 = 0.9, β2 = 0.999. After an outlier, u jumps to its size and decays by β2 per step, so AdaMax’s steps shrink for a while; Adam’s √v̂ reacts more mildly to one outlier (it averages squares).</span>';
    }
    draw();
  };

  // ================================================================== ch-averaging
  W.polyakDemo = function (host) {
    Book.head(host, 'Average the iterates to cancel the jitter',
      'Adam with a constant α on a bowl whose gradients are very noisy. The raw iterate θ_t keeps jittering around the optimum. The Polyak average (1/t)·Σθ_k and the bias-corrected exponential average (§7.2) are much closer.');
    const st = { sd: 2, b: 0.999 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'gradient noise', 0, 4, 0.1, st.sd, function (v) { st.sd = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.select(ctl, 'EMA decay β2', ['0.9', '0.99', '0.999'], '0.999', function (v) { st.b = +v; draw(); });
    const map = AD.contour(host, Opt.surfaces.bowl, { height: 280 }), plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const R = Opt.runSurface('bowl', 'adam', { lr: 0.05 }, 2000, st.sd, 4);
      const pol = [], ema = []; let s0 = 0, s1 = 0, e0 = 0, e1 = 0;
      R.path.forEach(function (p, k) { const t = k + 1; s0 += p[0]; s1 += p[1]; pol.push([s0 / t, s1 / t]); e0 = st.b * e0 + (1 - st.b) * p[0]; e1 = st.b * e1 + (1 - st.b) * p[1]; const c = 1 - Math.pow(st.b, t); ema.push([e0 / c, e1 / c]); });
      map.draw([{ path: R.path.slice(1000), color: Opt.colors.sgd, width: 1, alpha: 0.6 }, { path: pol, color: Book.palette[1], width: 2.4 }, { path: ema, color: Opt.colors.adam, width: 2.4 }]);
      const dist = function (p) { return Math.hypot(p[0], p[1]); }, pick = [10, 30, 100, 300, 1000, 2000];
      AD.lines(plot, [
        { name: 'raw θ_t', color: Opt.colors.sgd, points: pick.map(function (t) { return [t, Math.log10(dist(R.path[t]) + 1e-6)]; }) },
        { name: 'Polyak', color: Book.palette[1], points: pick.map(function (t) { return [t, Math.log10(dist(pol[t]) + 1e-6)]; }) },
        { name: 'EMA (corrected)', color: Opt.colors.adam, points: pick.map(function (t) { return [t, Math.log10(dist(ema[t]) + 1e-6)]; }) }
      ], { width: 640, height: 200, logx: true, xmin: 10, xmax: 2000, xticks: [10, 100, 1000], xlabel: 'step (log)', ylabel: 'log₁₀ distance to optimum', mr: 120, ydec: 1 });
      out.innerHTML = 'distance to the optimum after 2,000 steps: raw θ <b>' + dist(R.path[2000]).toFixed(3) + '</b> · Polyak average <b>' + dist(pol[2000]).toFixed(3) + '</b> · EMA <b>' + dist(ema[2000]).toFixed(3) + '</b>' +
        '<br><span class="note">Grey: the raw iterates of the last 1,000 steps. The Polyak average remembers the long walk in from the start; the exponential average forgets it after about 1/(1 − β2) steps, and its bias correction removes the pull towards θ̄₀ = 0. Toy bowl.</span>';
    }
    draw();
  };
})();
