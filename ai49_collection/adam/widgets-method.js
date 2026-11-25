/* Adam Interactive Web Book — Part II widgets: the Adam algorithm (paper §1–§3, Algorithm 1).
   Uses Book, ML, NN, Opt and the helpers in window.AD (widgets-found.js). */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, Opt = window.Opt, AD = window.AD;

  // learning rates per surface for the race (the book's own choice: each works on that surface without blowing up)
  const RACE = {
    rosenbrock: { sgd: { lr: 0.002 }, momentum: { lr: 0.0005, mu: 0.9 }, adagrad: { lr: 0.4 }, rmsprop: { lr: 0.01 }, adam: { lr: 0.05 }, adamax: { lr: 0.05 } },
    ravine: { sgd: { lr: 0.3 }, momentum: { lr: 0.05, mu: 0.9 }, adagrad: { lr: 0.5 }, rmsprop: { lr: 0.03 }, adam: { lr: 0.1 }, adamax: { lr: 0.1 } },
    plateau: { sgd: { lr: 1 }, momentum: { lr: 0.2, mu: 0.9 }, adagrad: { lr: 0.4 }, rmsprop: { lr: 0.02 }, adam: { lr: 0.1 }, adamax: { lr: 0.1 } },
    scaled: { sgd: { lr: 1 }, momentum: { lr: 0.15, mu: 0.9 }, adagrad: { lr: 0.4 }, rmsprop: { lr: 0.03 }, adam: { lr: 0.1 }, adamax: { lr: 0.1 } },
    bowl: { sgd: { lr: 0.1 }, momentum: { lr: 0.03, mu: 0.9 }, adagrad: { lr: 0.4 }, rmsprop: { lr: 0.03 }, adam: { lr: 0.1 }, adamax: { lr: 0.1 } }
  };
  AD.RACE = RACE;
  const ORDER = ['sgd', 'momentum', 'adagrad', 'rmsprop', 'adam', 'adamax'];

  function makeRace(host, cfg) {
    const st = { surf: cfg.surf || 'rosenbrock', noise: cfg.noise || 0, max: cfg.max || 400, on: {}, start: null, timer: null, lrs: {} };
    (cfg.show || ORDER).forEach(function (k) { st.on[k] = true; });
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'surface', Object.keys(RACE).map(function (k) { return { value: k, label: Opt.surfaces[k].label }; }), st.surf, function (v) { st.surf = v; st.start = null; map.setSpec(Opt.surfaces[v]); st.lrs = {}; syncLr(); restart(); });
    if (!cfg.compact) Book.slider(ctl, 'gradient noise', 0, 2, 0.05, st.noise, function (v) { st.noise = v; restart(); }, function (v) { return v.toFixed(2); });
    const go = Book.button(ctl, 'Pause ❚❚', function () { if (st.timer) stop(); else play(); }, 'primary');
    Book.button(ctl, 'Restart', function () { restart(); });
    let lrSel = 'adam', lrS = null;
    if (!cfg.compact) {
      const c2 = el('div', { class: 'controls' }, host);
      AD.checks(c2, ORDER.map(function (k) { return { key: k, label: Opt.names[k], get on() { return st.on[k]; }, set on(v) { st.on[k] = v; } }; }), function () { draw(); });
      const c3 = el('div', { class: 'controls' }, host);
      Book.select(c3, 'learning rate of', ORDER.map(function (k) { return { value: k, label: Opt.names[k] }; }), lrSel, function (v) { lrSel = v; syncLr(); });
      lrS = Book.slider(c3, '', -4, 0.5, 0.05, -1, function (v) { st.lrs[lrSel] = Math.pow(10, v); restart(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    }
    function lrOf(k) { return st.lrs[k] != null ? st.lrs[k] : RACE[st.surf][k].lr; }
    function syncLr() { if (lrS) { lrS.value = Math.log10(lrOf(lrSel)); lrS.dispatchEvent(new Event('input')); } }
    const map = AD.contour(host, Opt.surfaces[st.surf], { height: cfg.height || 330 });
    map.canvas.style.cursor = 'crosshair';
    map.canvas.addEventListener('click', function (e) { st.start = map.toXY(e); restart(); });
    const lg = el('div', null, host); AD.legend(lg, ORDER.filter(function (k) { return st.on[k] != null; }).map(function (k) { return { name: Opt.names[k], color: Opt.colors[k] }; }));
    const plot = cfg.compact ? null : el('div', null, host), out = el('div', { class: 'out' }, host);
    let runs = [];
    function restart() {
      stop();
      const S0 = Opt.surfaces[st.surf], p0 = st.start || S0.start;
      runs = ORDER.filter(function (k) { return st.on[k] != null; }).map(function (k, i) {
        const P = Opt.vec(p0), o = Opt.make(k, [P], Object.assign({}, RACE[st.surf][k], { lr: lrOf(k) }));
        return { k: k, P: P, o: o, path: [[p0[0], p0[1]]], loss: [S0.f(p0[0], p0[1])], r: ML.rng(7 + i), dead: false };
      });
      draw(); play();
    }
    function advance(n) {
      const S0 = Opt.surfaces[st.surf];
      runs.forEach(function (u) {
        for (let s = 0; s < n && !u.dead && u.path.length <= st.max; s++) {
          const g = S0.g(u.P.data[0], u.P.data[1]);
          u.P.grad[0] = g[0] + (st.noise ? u.r.normal(0, st.noise) : 0); u.P.grad[1] = g[1] + (st.noise ? u.r.normal(0, st.noise) : 0);
          u.o.step();
          const x = u.P.data[0], y = u.P.data[1];
          if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 1e4 || Math.abs(y) > 1e4) { u.dead = true; break; }
          u.path.push([x, y]); u.loss.push(S0.f(x, y));
        }
      });
    }
    function play() { go.textContent = 'Pause ❚❚'; (function tick() { advance(cfg.speed || 3); draw(); if (runs.every(function (u) { return u.dead || u.path.length > st.max; })) { stop(); return; } st.timer = setTimeout(tick, 30); })(); }
    function stop() { if (st.timer) clearTimeout(st.timer); st.timer = null; go.textContent = 'Play ▶'; }
    function draw() {
      const vis = runs.filter(function (u) { return st.on[u.k]; });
      map.draw(vis.map(function (u) { return { path: u.path, color: Opt.colors[u.k], width: u.k === 'adam' ? 2.8 : 1.8 }; }));
      if (plot) AD.logLines(plot, vis.map(function (u) { return { color: Opt.colors[u.k], marker: 'none', points: u.loss.map(function (v, t) { return [t, Math.max(v, 1e-10)]; }) }; }),
        { width: 640, height: 200, xmin: 0, xmax: st.max, ymin: 1e-10, xlabel: 'step', ylabel: 'loss (log)', xticks: [0, st.max / 2, st.max], mr: 100 });
      const t = runs.length ? Math.max.apply(null, runs.map(function (u) { return u.path.length - 1; })) : 0;
      out.innerHTML = 'step ' + t + ' of ' + st.max + ' · ' + vis.map(function (u) { return '<span style="color:' + Opt.colors[u.k] + ';font-weight:700">' + Opt.names[u.k] + '</span> ' + (u.dead ? 'blew up' : AD.f(u.loss[u.loss.length - 1], 3)); }).join(' · ') +
        '<br><span class="note">' + (cfg.note || 'Click the map to start somewhere else. Learning rates are the book’s choice per surface (any rate can be changed). Toy surfaces, not from the paper.') + '</span>';
    }
    restart();
    return { restart: restart };
  }

  // ================================================================== hero + ch-idea
  W.heroDemo = function (host) {
    Book.head(host, 'Six optimizers race to the bottom of a banana-shaped valley',
      'Each line is one optimizer walking downhill from the same start, using only gradients. Adam (purple, thick) keeps steady steps of about the same size; plain SGD barely moves in the flat valley floor. Click the map to start elsewhere.');
    makeRace(host, { surf: 'rosenbrock', compact: true, height: 300, max: 400, speed: 4, note: 'Toy surface. Full controls in Chapter 9.' });
  };
  W.optimizerRace = function (host) {
    Book.head(host, 'The optimizer race',
      'Pick a surface, add noise to make the gradients stochastic, switch optimizers on and off, change any learning rate, click the map to move the start. The chart shows each optimizer’s loss on a log scale.');
    makeRace(host, { surf: 'rosenbrock', max: 400 });
  };

  // ================================================================== ch-alg1: Algorithm 1 line by line
  W.adamStepper = function (host) {
    Book.head(host, 'Algorithm 1, one line at a time',
      'Two weights θ = (θ₁, θ₂). Each step draws a noisy gradient; θ₂’s gradient is 100× smaller than θ₁’s. Press “Next line” to run Algorithm 1 line by line, or “Next step” to run a whole loop. All operations are per weight (elementwise).');
    const LINES = [
      't ← t + 1',
      'g_t ← ∇θ f_t(θ_{t−1})                 (get the gradient)',
      'm_t ← β1·m_{t−1} + (1 − β1)·g_t      (biased 1st moment)',
      'v_t ← β2·v_{t−1} + (1 − β2)·g_t²     (biased 2nd raw moment)',
      'm̂_t ← m_t / (1 − β1^t)               (bias-corrected 1st moment)',
      'v̂_t ← v_t / (1 − β2^t)               (bias-corrected 2nd moment)',
      'θ_t ← θ_{t−1} − α·m̂_t / (√v̂_t + ε)   (update)'
    ];
    const st = { alpha: 0.1, b1: 0.9, b2: 0.999, eps: 1e-8 };
    let S0, ref, refOpt, rng;
    const ctl = el('div', { class: 'controls' }, host);
    Book.button(ctl, 'Next line', function () { line(); }, 'primary');
    Book.button(ctl, 'Next step', function () { do line(); while (S0.pc !== 0); });
    Book.button(ctl, '10 steps', function () { for (let k = 0; k < 10; k++) do line(); while (S0.pc !== 0); });
    Book.button(ctl, 'Reset', function () { reset(); });
    const c2 = el('div', { class: 'controls' }, host);
    Book.slider(c2, 'α', -3, 0, 0.1, Math.log10(st.alpha), function (v) { st.alpha = Math.pow(10, v); reset(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    Book.slider(c2, 'β1', 0, 0.99, 0.01, st.b1, function (v) { st.b1 = v; reset(); }, function (v) { return v.toFixed(2); });
    Book.select(c2, 'β2', ['0.9', '0.99', '0.999', '0.9999'], '0.999', function (v) { st.b2 = +v; reset(); });
    const code = el('pre', { style: 'font-size:12.5px;line-height:1.7;margin:0 0 8px;white-space:pre;overflow-x:auto' }, host), tab = el('div', { class: 'scroll-x' }, host);
    const out = el('div', { class: 'out' }, host);
    function reset() {
      S0 = { t: 0, pc: 0, theta: [2, 2], m: [0, 0], v: [0, 0], mh: [0, 0], vh: [0, 0], g: [0, 0] };
      rng = ML.rng(3); ref = Opt.vec([2, 2]); refOpt = new NN.Adam([ref], { lr: st.alpha, b1: st.b1, b2: st.b2, eps: st.eps });
      draw();
    }
    function line() {
      const s = S0, b1 = st.b1, b2 = st.b2;
      switch (s.pc) {
        case 0: s.t++; break;
        case 1: s.g = [s.theta[0] - 0.5 + rng.normal(0, 0.5), 0.01 * (s.theta[1] + 1 + rng.normal(0, 0.5))]; break;   // f_t(θ) = ½(θ₁−0.5−noise)² + 0.005(θ₂+1−noise)²
        case 2: s.m = s.m.map(function (m, i) { return b1 * m + (1 - b1) * s.g[i]; }); break;
        case 3: s.v = s.v.map(function (v, i) { return b2 * v + (1 - b2) * s.g[i] * s.g[i]; }); break;
        case 4: s.mh = s.m.map(function (m) { return m / (1 - Math.pow(b1, s.t)); }); break;
        case 5: s.vh = s.v.map(function (v) { return v / (1 - Math.pow(b2, s.t)); }); break;
        case 6:
          s.theta = s.theta.map(function (th, i) { return th - st.alpha * s.mh[i] / (Math.sqrt(s.vh[i]) + st.eps); });
          ref.grad[0] = s.g[0]; ref.grad[1] = s.g[1]; refOpt.step();       // the shared NN.Adam on the same gradient
          break;
      }
      s.pc = (s.pc + 1) % 7; draw();
    }
    function draw() {
      code.innerHTML = '';
      el('div', { class: 'note', text: 'while θ not converged do' }, code);
      LINES.forEach(function (L, i) { const d = el('div', { text: '  ' + L }, code); if (i === S0.pc) d.style.cssText = 'background:var(--accent-soft);color:var(--accent-ink);font-weight:700;border-radius:4px'; });
      el('div', { class: 'note', text: 'end while   ▸ highlighted = runs next' }, code);
      const f = function (x) { return AD.ft(x, 4); }, s = S0;
      tab.innerHTML = '<table><tr><td></td><td><b>weight θ₁</b></td><td><b>weight θ₂</b></td></tr>' +
        [['t', s.t, ''], ['g', f(s.g[0]), f(s.g[1])], ['m', f(s.m[0]), f(s.m[1])], ['v', f(s.v[0]), f(s.v[1])], ['m̂', f(s.mh[0]), f(s.mh[1])], ['v̂', f(s.vh[0]), f(s.vh[1])],
          ['m̂/√v̂', s.t ? f(s.mh[0] / Math.sqrt(s.vh[0])) : '', s.t ? f(s.mh[1] / Math.sqrt(s.vh[1])) : ''], ['θ', f(s.theta[0]), f(s.theta[1])]]
          .map(function (r) { return '<tr><td><b>' + r[0] + '</b></td><td>' + r[1] + '</td><td>' + r[2] + '</td></tr>'; }).join('') + '</table>';
      const agree = Math.abs(ref.data[0] - s.theta[0]) < 1e-12 && Math.abs(ref.data[1] - s.theta[1]) < 1e-12;
      out.innerHTML = 'shared library NN.Adam on the same gradients: θ = (' + f(ref.data[0]) + ', ' + f(ref.data[1]) + ')' + (S0.pc === 0 || S0.pc === 1 ? (agree ? ' ✓ identical' : '') : ' (catches up when the update line runs)') +
        '<br><span class="note">θ₂’s gradient is 100× smaller, yet m̂/√v̂ has a similar size for both weights, so both move by roughly α per step at first. α = 0.1 here so you can see movement; the paper’s default is 0.001.</span>';
    }
    reset();
  };

  // ================================================================== ch-hyper
  W.hyperExplorer = function (host) {
    Book.head(host, 'The four knobs: α, β1, β2, ε',
      'Adam on the noisy banana valley, 600 steps. Change one knob at a time and watch the path and the final loss. “Paper defaults” sets α = 0.001, β1 = 0.9, β2 = 0.999, ε = 10⁻⁸.');
    const st = { la: Math.log10(0.05), b1: 0.9, lb2: -3, le: -8, noise: 0.5 };
    const ctl = el('div', { class: 'controls' }, host), sl = {};
    sl.a = Book.slider(ctl, 'α', -4, 0, 0.05, st.la, function (v) { st.la = v; draw(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    sl.b1 = Book.slider(ctl, 'β1', 0, 0.99, 0.01, st.b1, function (v) { st.b1 = v; draw(); }, function (v) { return v.toFixed(2); });
    sl.b2 = Book.slider(ctl, 'β2', -4, -0.5, 0.1, st.lb2, function (v) { st.lb2 = v; draw(); }, function (v) { return (1 - Math.pow(10, v)).toPrecision(5); });
    sl.e = Book.slider(ctl, 'ε', -12, 0, 0.5, st.le, function (v) { st.le = v; draw(); }, function (v) { return AD.p10(v); });
    const c2 = el('div', { class: 'controls' }, host);
    Book.slider(c2, 'gradient noise', 0, 2, 0.1, st.noise, function (v) { st.noise = v; draw(); }, function (v) { return v.toFixed(1); });
    const setAll = function (a, b1, lb2, le) { [[sl.a, a], [sl.b1, b1], [sl.b2, lb2], [sl.e, le]].forEach(function (p) { p[0].value = p[1]; p[0].dispatchEvent(new Event('input')); }); };
    Book.button(c2, 'Paper defaults', function () { setAll(-3, 0.9, -3, -8); }, 'primary');
    Book.button(c2, 'Toy-friendly α = 0.05', function () { setAll(Math.log10(0.05), 0.9, -3, -8); });
    const map = AD.contour(host, Opt.surfaces.rosenbrock, { height: 300 }), plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const o = { lr: Math.pow(10, st.la), b1: st.b1, b2: 1 - Math.pow(10, st.lb2), eps: Math.pow(10, st.le) };
      const R = Opt.runSurface('rosenbrock', 'adam', o, 600, st.noise, 2);
      map.draw([{ path: R.path, color: Opt.colors.adam, width: 2.2 }]);
      AD.logLines(plot, [{ color: Opt.colors.adam, marker: 'none', points: R.loss.map(function (v, t) { return [t, Math.max(v, 1e-8)]; }) }], { width: 640, height: 180, xmin: 0, xmax: 600, ymin: 1e-6, xlabel: 'step', ylabel: 'loss (log)', xticks: [0, 200, 400, 600] });
      let travel = 0; for (let k = 1; k < R.path.length; k++) travel += Math.hypot(R.path[k][0] - R.path[k - 1][0], R.path[k][1] - R.path[k - 1][1]);
      const L = R.loss[R.loss.length - 1];
      out.innerHTML = 'α = ' + AD.f(o.lr, 4) + ', β1 = ' + o.b1.toFixed(2) + ', β2 = ' + o.b2.toPrecision(5) + ', ε = 10<sup>' + st.le + '</sup> → final loss <b>' + (isFinite(L) ? AD.f(L) : 'blew up') + '</b> · path length <b>' + travel.toFixed(2) + '</b> over 600 steps (600 steps of length exactly α would be ' + (600 * o.lr).toFixed(2) + ')' +
        '<br><span class="note">α sets the step length: with the paper’s 0.001 Adam walks only about 0.6 units in 600 steps, far too little for this toy, whose start is about 3 units from the star (Chapter 14 explains how to pick α from distances). β1 smooths the direction, β2 the step size; ε matters only when it is comparable to √v̂ (try 10⁰).</span>';
    }
    draw();
  };

  // ================================================================== ch-efficient
  W.efficientForm = function (host) {
    Book.head(host, 'Two ways to compute the same update',
      'Algorithm 1 corrects m and v, then divides. The paper’s faster order folds both corrections into one number α_t and divides the raw m by √v. Run both on the same 1,000 random gradients and compare.');
    const st = { le: -8, lg: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'ε', -12, -1, 0.5, st.le, function (v) { st.le = v; draw(); }, function (v) { return AD.p10(v); });
    Book.slider(ctl, 'gradient size', -8, 2, 0.5, st.lg, function (v) { st.lg = v; draw(); }, function (v) { return AD.p10(v); });
    const out = el('div', { class: 'out' }, host);
    function draw() {
      const a = 0.001, b1 = 0.9, b2 = 0.999, eps = Math.pow(10, st.le), sc = Math.pow(10, st.lg), r = ML.rng(4);
      let m = 0, v = 0, A = 0, B = 0, C = 0; const rows = [];
      for (let t = 1; t <= 1000; t++) {
        const g = sc * r.normal(0.3, 1);
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g;
        A -= a * (m / (1 - Math.pow(b1, t))) / (Math.sqrt(v / (1 - Math.pow(b2, t))) + eps);              // Algorithm 1
        const at = a * Math.sqrt(1 - Math.pow(b2, t)) / (1 - Math.pow(b1, t));
        B -= at * m / (Math.sqrt(v) + eps);                                                               // fast form, same ε
        C -= at * m / (Math.sqrt(v) + eps * Math.sqrt(1 - Math.pow(b2, t)));                              // fast form, ε̂ = ε·√(1−β2^t)
        if (t === 1 || t === 10 || t === 100 || t === 1000) rows.push([t, at, A, B, C]);
      }
      const f = function (x) { return AD.ft(x, 6); };
      out.innerHTML = '<table><tr><td><b>\\(t\\)</b></td><td><b>\\(\\alpha_t\\)</b></td><td><b>θ, Algorithm 1</b></td><td><b>θ, fast form with \\(\\epsilon\\)</b></td><td><b>θ, fast form with \\(\\hat\\epsilon\\)</b></td></tr>' +
        rows.map(function (q) { return '<tr><td>' + q[0] + '</td><td>' + f(q[1]) + '</td><td>' + f(q[2]) + '</td><td>' + f(q[3]) + '</td><td>' + f(q[4]) + '</td></tr>'; }).join('') + '</table>' +
        'after 1,000 steps: |Algorithm 1 − fast with \\(\\epsilon\\)| = <b>' + AD.f(Math.abs(rows[3][2] - rows[3][3]), 3) + '</b> · |Algorithm 1 − fast with \\(\\hat\\epsilon\\)| = <b>' + AD.f(Math.abs(rows[3][2] - rows[3][4]), 3) + '</b>' +
        '<br>work per weight per step: Algorithm 1 = 2 corrections + √ + add + divide + multiply; fast form = √ + add + divide + multiply (α_t is one number per step, shared by all weights)' +
        '<br><span class="note">The two forms differ only in where ε sits. With \\(\\hat\\epsilon = \\epsilon\\sqrt{1-\\beta_2^t}\\) they are exactly equal. The difference matters only when the gradients are as small as ε (try gradient size 10⁻⁸).</span>';
      Book.tex(out);
    }
    draw();
  };

  // ================================================================== ch-bound
  W.stepBound = function (host) {
    Book.head(host, 'How big can one step be?',
      'Pick a gradient pattern for one weight and run Adam (ε = 0) for 2,000 steps. The chart shows the effective step |Δ_t| divided by α. The dashed lines are the paper’s two bounds: 1 and (1 − β1)/√(1 − β2).');
    const st = { kind: 'spike', b1: 0.9, lb2: -3, mu: 0.3, sd: 1, k: 1000 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'gradient pattern', [{ value: 'const', label: 'constant g = 1' }, { value: 'noisy', label: 'noisy: mean μ, spread σ' }, { value: 'sparse', label: 'sparse: g = 1 every k steps, else 0' }, { value: 'spike', label: 'one spike: 0 for 1,999 steps, then 1' }, { value: 'flip', label: 'sign flips: +1, −1, +1, …' }],
      st.kind, function (v) { st.kind = v; draw(); });
    Book.slider(ctl, 'β1', 0, 0.99, 0.01, st.b1, function (v) { st.b1 = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'β2', -4, -1, 0.1, st.lb2, function (v) { st.lb2 = v; draw(); }, function (v) { return (1 - Math.pow(10, v)).toPrecision(5); });
    const c2 = el('div', { class: 'controls' }, host);
    Book.slider(c2, 'μ', -1, 1, 0.05, st.mu, function (v) { st.mu = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(c2, 'σ', 0, 3, 0.05, st.sd, function (v) { st.sd = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(c2, 'k (sparse)', 2, 1000, 1, st.k, function (v) { st.k = v; draw(); }, function (v) { return v; });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const b1 = st.b1, b2 = 1 - Math.pow(10, st.lb2), r = ML.rng(9), T = 2000; let m = 0, v = 0, mx = 0, mxT = 0; const pts = [];
      for (let t = 1; t <= T; t++) {
        let g = 1;
        if (st.kind === 'noisy') g = r.normal(st.mu, st.sd); else if (st.kind === 'sparse') g = t % st.k === 0 ? 1 : 0; else if (st.kind === 'spike') g = t === T ? 1 : 0; else if (st.kind === 'flip') g = t % 2 ? 1 : -1;
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g;
        const mh = m / (1 - Math.pow(b1, t)), vh = v / (1 - Math.pow(b2, t)), d = vh > 0 ? Math.abs(mh) / Math.sqrt(vh) : 0;
        pts.push([t, d]); if (d > mx) { mx = d; mxT = t; }
      }
      const B = (1 - b1) / Math.sqrt(1 - b2), bound = (1 - b1) > Math.sqrt(1 - b2) ? B : 1;
      const top = Math.max(1.2, B * 1.15, mx * 1.1);
      AD.lines(plot, [
        { color: 'var(--accent)', marker: 'none', width: 1.6, points: pts },
        { name: 'α', color: 'var(--ink)', marker: 'none', dash: '6 4', width: 1.2, points: [[1, 1], [T, 1]] },
        { name: '(1−β1)/√(1−β2)', color: Book.palette[1], marker: 'none', dash: '6 4', width: 1.2, points: [[1, B], [T, B]] }
      ], { width: 640, height: 230, ymin: 0, ymax: top, xlabel: 'step t', ylabel: '|Δt| / α', xticks: [1, 500, 1000, 1500, 2000], mr: 120, ydec: 2 });
      out.innerHTML = 'largest step: <b>' + mx.toFixed(3) + '·α</b> at t = ' + mxT + ' · (1 − β1) = ' + (1 - b1).toFixed(3) + ' vs √(1 − β2) = ' + Math.sqrt(1 - b2).toFixed(4) +
        ' → the paper’s bound here is <b>' + bound.toFixed(3) + '·α</b>' + ((1 - b1) > Math.sqrt(1 - b2) ? ' (first case: (1−β1) > √(1−β2))' : ' (second case)') +
        '<br><span class="note">Only the sparse patterns get near the big bound: a gradient that was zero for a long time and suddenly appears (“one spike” is the paper’s most severe case: zero at all timesteps except the current one). Every other pattern stays at or below α. With the defaults the big bound is 0.1/√0.001 ≈ 3.16.</span>';
    }
    draw();
  };

  // ================================================================== ch-trust
  W.snrAnneal = function (host) {
    Book.head(host, 'Steps shrink by themselves near the optimum',
      'One weight, loss ½θ², gradient θ plus noise of size σ. Far from 0 the gradient is consistent (high signal-to-noise ratio); near 0 it is mostly noise. Adam’s step follows the SNR = m̂/√v̂ down: automatic annealing.');
    const st = { sd: 1, la: Math.log10(0.05) };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'noise σ', 0, 3, 0.1, st.sd, function (v) { st.sd = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'α', -3, 0, 0.05, st.la, function (v) { st.la = v; draw(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    const p1 = el('div', null, host), p2 = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const P = Opt.vec([5]), o = Opt.make('adam', [P], { lr: Math.pow(10, st.la) }), r = ML.rng(5), th = [], snr = [], step = [];
      for (let t = 1; t <= 1000; t++) {
        const prev = P.data[0]; P.grad[0] = P.data[0] + r.normal(0, st.sd); o.step();
        const mh = o.m[0][0] / (1 - Math.pow(0.9, t)), vh = o.v[0][0] / (1 - Math.pow(0.999, t));
        th.push([t, P.data[0]]); snr.push([t, Math.abs(mh / Math.sqrt(vh))]); step.push([t, Math.abs(P.data[0] - prev) / o.lr]);
      }
      AD.lines(p1, [{ name: 'θ', color: 'var(--accent)', marker: 'none', points: th }], { width: 640, height: 160, ymin: -1, ymax: 5.5, xlabel: 'step', ylabel: 'θ', xticks: [1, 250, 500, 750, 1000], mr: 40 });
      AD.lines(p2, [{ name: '|SNR|', color: Book.palette[4], marker: 'none', points: snr }, { name: '|Δ|/α', color: Book.palette[1], marker: 'none', dash: '4 3', width: 1.4, points: step, ldy: 14 }],
        { width: 640, height: 180, ymin: 0, ymax: 1.2, xlabel: 'step', ylabel: 'size', xticks: [1, 250, 500, 750, 1000], mr: 70, ydec: 1 });
      const late = ML.mean(snr.slice(800).map(function (p) { return p[1]; })), early = ML.mean(snr.slice(5, 50).map(function (p) { return p[1]; }));
      out.innerHTML = 'average |SNR| in steps 6–50: <b>' + early.toFixed(2) + '</b> · in steps 801–1000: <b>' + late.toFixed(2) + '</b> · final θ = ' + th[999][1].toFixed(3) +
        '<br><span class="note">With ε = 0 the step is exactly |Δ| = α·|SNR| (the two curves coincide). Far away the SNR is near 1 and steps are about α; near the optimum the gradient’s sign is a coin flip, the SNR falls, and the steps shrink without any schedule. With σ = 0 there is no noise and the SNR stays near 1 until θ reaches 0.</span>';
    }
    draw();
  };

  W.alphaPlanner = function (host) {
    Book.head(host, 'Choosing α from what you know about distances',
      'Suppose you believe the best weight is within distance R of where you start. Adam moves at most about α per step, so it needs at least about R/α steps. Here one weight starts at 0 and the best value is R; the loss is k·(θ − R)²/2. Change k: Adam does not care, SGD does.');
    const st = { lR: 0, la: -2, lk: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'distance R', -2, 2, 0.1, st.lR, function (v) { st.lR = v; draw(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    Book.slider(ctl, 'α (and SGD’s learning rate)', -4, -0.5, 0.1, st.la, function (v) { st.la = v; draw(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    Book.slider(ctl, 'curvature k', -3, 3, 0.1, st.lk, function (v) { st.lk = v; draw(); }, function (v) { return Math.pow(10, v).toPrecision(2); });
    const out = el('div', { class: 'out' }, host);
    function steps(name, R, a, k) {
      const P = Opt.vec([0]), o = Opt.make(name, [P], { lr: a });
      for (let t = 1; t <= 100000; t++) { P.grad[0] = k * (P.data[0] - R); o.step(); if (!isFinite(P.data[0]) || Math.abs(P.data[0]) > 1e6 * R) return 'blew up'; if (Math.abs(P.data[0] - R) < 0.05 * R) return t; }
      return '> 100,000';
    }
    function draw() {
      const R = Math.pow(10, st.lR), a = Math.pow(10, st.la), k = Math.pow(10, st.lk);
      out.innerHTML = 'prediction R/α = <b>' + Math.ceil(R / a).toLocaleString('en-US') + '</b> steps (to get within 5 %: about 0.95·R/α)' +
        '<br>Adam actually took <b>' + steps('adam', R, a, k).toLocaleString('en-US') + '</b> steps · plain SGD with the same rate took <b>' + steps('sgd', R, a, k).toLocaleString('en-US') + '</b> steps' +
        '<br><span class="note">Adam’s count does not move when you change k (the gradient’s scale) and stays within a small factor of R/α (the last few percent are slower: momentum overshoots and the steps shrink). SGD’s count depends on k·lr, a number you rarely know in advance. That is the practical meaning of the paper’s “trust region” remark (§2.1).</span>';
    }
    draw();
  };

  // ================================================================== ch-invariance
  W.scaleInvariance = function (host) {
    Book.head(host, 'Multiply the loss by c: SGD changes, Adam does not',
      'The ravine loss is multiplied by a constant c, which multiplies every gradient by c. Dashed paths: c = 1. Solid paths: your c. SGD’s path depends on c; Adam’s is the same until the gradients become as small as ε.');
    const st = { lc: -1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'scale c', -10, 3, 0.25, st.lc, function (v) { st.lc = v; draw(); }, function (v) { return AD.p10(v); });
    const map = AD.contour(host, Opt.surfaces.ravine, { height: 300 }), out = el('div', { class: 'out' }, host);
    function run(name, o, c) {
      const S0 = Opt.surfaces.ravine, P = Opt.vec(S0.start), opt = Opt.make(name, [P], o), path = [[P.data[0], P.data[1]]];
      for (let t = 0; t < 150; t++) { const g = S0.g(P.data[0], P.data[1]); P.grad[0] = c * g[0]; P.grad[1] = c * g[1]; opt.step(); if (!isFinite(P.data[0]) || Math.abs(P.data[1]) > 1e4) { path.push([NaN, NaN]); break; } path.push([P.data[0], P.data[1]]); }
      return path;
    }
    function draw() {
      const c = Math.pow(10, st.lc);
      const s1 = run('sgd', { lr: 0.2 }, 1), s2 = run('sgd', { lr: 0.2 }, c), a1 = run('adam', { lr: 0.1 }, 1), a2 = run('adam', { lr: 0.1 }, c);
      map.draw([{ path: s1, color: Opt.colors.sgd, dash: [5, 4], width: 1.4 }, { path: a1, color: Opt.colors.adam, dash: [5, 4], width: 1.4 }, { path: s2, color: Opt.colors.sgd, width: 2.4 }, { path: a2, color: Opt.colors.adam, width: 2.4 }]);
      let dA = 0; for (let k = 0; k < Math.min(a1.length, a2.length); k++) dA = Math.max(dA, Math.hypot(a1[k][0] - a2[k][0], a1[k][1] - a2[k][1]) || 0);
      const sEnd = s2[s2.length - 1];
      out.innerHTML = 'c = ' + AD.f(c, 3) + ' · Adam: largest gap between the two paths = <b>' + AD.f(dA, 3) + '</b> · SGD with c: ' + (isNaN(sEnd[0]) ? '<b>blew up</b>' : 'ends at (' + sEnd[0].toFixed(2) + ', ' + sEnd[1].toFixed(2) + ')') +
        '<br><span class="note">(c·m̂)/√(c²·v̂) = m̂/√v̂: the c cancels (paper §2.1). Only ε = 10⁻⁸ breaks the tie: at c ≈ 10⁻⁸ and below, √v̂ is no longer much bigger than ε and Adam slows down too.</span>';
    }
    draw();
  };

  // ================================================================== ch-bias
  W.biasDemo = function (host) {
    Book.head(host, 'Zero start makes the averages too small',
      'A weight’s gradients are random with mean 1 and spread 0.5, so E[g] = 1 and E[g²] = 1.25. Watch the raw averages m and v start at 0 and creep up, and the corrected m̂ and v̂ sit on the truth from the first step. Try β2 = 0.9999.');
    const st = { b1: 0.9, b2: 0.999, T: 200 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'β1', 0, 0.99, 0.01, st.b1, function (v) { st.b1 = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.select(ctl, 'β2', ['0.9', '0.99', '0.999', '0.9999'], '0.999', function (v) { st.b2 = +v; draw(); });
    Book.slider(ctl, 'steps shown', 20, 2000, 10, st.T, function (v) { st.T = v; draw(); }, function (v) { return v; });
    const g2 = el('div', { class: 'grid2' }, host), p1 = el('div', null, g2), p2 = el('div', null, g2), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(2); let m = 0, v = 0; const M = [], Mh = [], V = [], Vh = [];
      for (let t = 1; t <= st.T; t++) {
        const g = r.normal(1, 0.5); m = st.b1 * m + (1 - st.b1) * g; v = st.b2 * v + (1 - st.b2) * g * g;
        M.push([t, m]); Mh.push([t, m / (1 - Math.pow(st.b1, t))]); V.push([t, v]); Vh.push([t, v / (1 - Math.pow(st.b2, t))]);
      }
      const o = { width: 320, height: 210, ymin: 0, xlabel: 'step t', xticks: [1, st.T], mr: 30, ydec: 1 };
      AD.lines(p1, [{ color: 'var(--ink)', marker: 'none', dash: '5 4', width: 1, points: [[1, 1], [st.T, 1]] }, { name: 'm', color: Book.palette[7], marker: 'none', points: M }, { name: 'm̂', color: 'var(--accent)', marker: 'none', points: Mh, ldy: -8 }], Object.assign({ ymax: 2, ylabel: '1st moment' }, o));
      AD.lines(p2, [{ color: 'var(--ink)', marker: 'none', dash: '5 4', width: 1, points: [[1, 1.25], [st.T, 1.25]] }, { name: 'v', color: Book.palette[7], marker: 'none', points: V }, { name: 'v̂', color: 'var(--accent)', marker: 'none', points: Vh, ldy: -8 }], Object.assign({ ymax: 3, ylabel: '2nd raw moment' }, o));
      const t10 = Math.min(10, st.T) - 1;
      out.innerHTML = 'at t = ' + (t10 + 1) + ': m = ' + M[t10][1].toFixed(3) + ', m̂ = <b>' + Mh[t10][1].toFixed(3) + '</b> (truth 1) · v = ' + AD.f(V[t10][1], 4) + ', v̂ = <b>' + Vh[t10][1].toFixed(3) + '</b> (truth 1.25)' +
        '<br>the raw v needs about 1/(1 − β2) = <b>' + Math.round(1 / (1 - st.b2)).toLocaleString('en-US') + '</b> steps to forget its zero start' +
        '<br><span class="note">Dashed lines: the true moments. Without correction, the update m/√v would be inflated at the start, because v is much more shrunk than m when β2 is closer to 1 than β1.</span>';
    }
    draw();
  };

  W.eq1Weights = function (host) {
    Book.head(host, 'Eq. 1: v_t is a weighted sum of all past squared gradients',
      'Random gradients g₁…g_t. Each bar is the contribution (1 − β2)·β2^(t−i)·g_i² of one past gradient to v_t. Their sum equals the value from the recursion, exactly.');
    const st = { t: 15, b2: 0.8, seed: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 't', 1, 40, 1, st.t, function (v) { st.t = v; draw(); }, function (v) { return v; });
    Book.slider(ctl, 'β2', 0.5, 0.99, 0.01, st.b2, function (v) { st.b2 = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'New gradients', function () { st.seed++; draw(); }, 'primary');
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = ML.rng(st.seed), g = ML.range(st.t).map(function () { return r.normal(0, 1); });
      let v = 0; g.forEach(function (x) { v = st.b2 * v + (1 - st.b2) * x * x; });
      const c = g.map(function (x, k) { return (1 - st.b2) * Math.pow(st.b2, st.t - (k + 1)) * x * x; });
      Book.chart.bars(chart, c.map(function (x, k) { return { label: st.t <= 20 || (k + 1) % 5 === 0 ? String(k + 1) : '', value: x, note: '', color: 'var(--accent)' }; }), { ylabel: 'contribution to v_t', height: 200, tickFmt: function (x) { return x.toFixed(2); } });
      const sum = ML.sum(c);
      out.innerHTML = 'recursion v_t = β2·v_{t−1} + (1−β2)·g_t², started at v₀ = 0, gives <b>' + v.toFixed(6) + '</b>' +
        '<br>Eq. 1, (1 − β2)·Σ β2^(t−i)·g_i² (the sum of the bars), gives <b>' + sum.toFixed(6) + '</b>' + (Math.abs(v - sum) < 1e-12 ? ' → identical' : '') +
        '<br><span class="note">Recent gradients count most; a gradient k steps old is discounted by β2^k. The weights add up to 1 − β2^t, which is why E[v_t] falls short of E[g²] by exactly that factor.</span>';
    }
    draw();
  };

  // ================================================================== ch-biasderive
  W.biasExpect = function (host) {
    Book.head(host, 'Average over many runs: E[v_t] = E[g²]·(1 − β2^t)',
      'Run 300 independent gradient streams (all from the same distribution), average their v_t at each step, and compare with the paper’s Eq. 4 with ζ = 0.');
    const st = { b2: 0.95, mu: 0.5, sd: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'β2', 0.8, 0.995, 0.005, st.b2, function (v) { st.b2 = v; draw(); }, function (v) { return v.toFixed(3); });
    Book.slider(ctl, 'mean μ', -1, 1, 0.1, st.mu, function (v) { st.mu = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'spread σ', 0.1, 2, 0.1, st.sd, function (v) { st.sd = v; draw(); }, function (v) { return v.toFixed(1); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const T = 150, N = 300, r = ML.rng(7), avg = new Float64Array(T), Eg2 = st.mu * st.mu + st.sd * st.sd;
      for (let n = 0; n < N; n++) { let v = 0; for (let t = 0; t < T; t++) { const g = r.normal(st.mu, st.sd); v = st.b2 * v + (1 - st.b2) * g * g; avg[t] += v / N; } }
      AD.lines(plot, [
        { name: 'E[g²]', color: 'var(--ink)', marker: 'none', dash: '5 4', width: 1.2, points: [[1, Eg2], [T, Eg2]] },
        { name: 'average v_t', color: 'var(--accent)', marker: 'none', width: 2.6, points: Array.from(avg, function (x, t) { return [t + 1, x]; }) },
        { name: 'E[g²]·(1−β2^t)', color: Book.palette[1], marker: 'none', dash: '3 3', width: 2, points: ML.range(T).map(function (t) { return [t + 1, Eg2 * (1 - Math.pow(st.b2, t + 1))]; }), ldy: 16 }
      ], { width: 640, height: 230, ymin: 0, ymax: Eg2 * 1.3, xlabel: 'step t', ylabel: 'value', xticks: [1, 50, 100, 150], mr: 130, ydec: 2 });
      out.innerHTML = 'at t = 10: average v = <b>' + avg[9].toFixed(4) + '</b> · Eq. 4 predicts E[g²](1 − β2¹⁰) = <b>' + (Eg2 * (1 - Math.pow(st.b2, 10))).toFixed(4) + '</b> · dividing by (1 − β2¹⁰) gives ' + (avg[9] / (1 - Math.pow(st.b2, 10))).toFixed(4) + ' ≈ E[g²] = ' + Eg2.toFixed(4) +
        '<br><span class="note">The shortfall is the same factor (1 − β2^t) whatever μ and σ are, so one division fixes it for every weight. That division is line 6 of Algorithm 1.</span>';
    }
    draw();
  };

  W.zetaDemo = function (host) {
    Book.head(host, 'When the gradients drift: the leftover ζ',
      'Now E[g_t²] changes over time (it ramps up). E[v_t] is computed exactly from the recursion on expectations. ζ_t is what is left after Eq. 4: ζ_t = E[v_t] − E[g_t²]·(1 − β2^t). Shorter memory (smaller β2) keeps ζ small.');
    const st = { b2: 0.99, drift: 0.01 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'β2', 0.5, 0.999, 0.001, st.b2, function (v) { st.b2 = v; draw(); }, function (v) { return v.toFixed(3); });
    Book.slider(ctl, 'drift per step', 0, 0.05, 0.001, st.drift, function (v) { st.drift = v; draw(); }, function (v) { return v.toFixed(3); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const T = 300; let Ev = 0; const s = [], ev = [], z = [];
      for (let t = 1; t <= T; t++) { const Eg2 = 1 + st.drift * t; Ev = st.b2 * Ev + (1 - st.b2) * Eg2; const corr = Eg2 * (1 - Math.pow(st.b2, t)); s.push([t, Eg2]); ev.push([t, Ev / (1 - Math.pow(st.b2, t))]); z.push([t, Ev - corr]); }
      AD.lines(plot, [
        { name: 'E[g_t²]', color: 'var(--ink)', marker: 'none', dash: '5 4', width: 1.4, points: s },
        { name: 'E[v̂_t]', color: 'var(--accent)', marker: 'none', points: ev, ldy: 14 },
        { name: 'ζ_t', color: Book.palette[1], marker: 'none', points: z }
      ], { width: 640, height: 230, xlabel: 'step t', ylabel: 'value', xticks: [1, 100, 200, 300], mr: 80, ydec: 1 });
      const zr = z[T - 1][1] / s[T - 1][1];
      out.innerHTML = 'at t = 300: ζ = <b>' + z[T - 1][1].toFixed(3) + '</b>, which is <b>' + (100 * zr).toFixed(1) + ' %</b> of E[g²] · memory ≈ 1/(1 − β2) = ' + Math.round(1 / (1 - st.b2)) + ' steps' +
        '<br><span class="note">ζ = 0 when nothing drifts (try drift 0). When the moment drifts, v lags behind by roughly (drift per step)×(memory). The paper’s answer: keep the memory short enough that old gradients get little weight.</span>';
    }
    draw();
  };

  // ================================================================== ch-biaswhy
  W.firstSteps = function (host) {
    Book.head(host, 'The first steps, with and without the correction',
      'A constant gradient (any size: Adam’s step does not depend on it). Each line is |Δ_t|/α for the first 300 steps. Solid: with bias correction (always exactly 1 here). Dashed: without, as in RMSProp with momentum.');
    const st = { b1: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'β1', ['0', '0.9'], '0', function (v) { st.b1 = +v; draw(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const sers = [], firsts = [];
      [0.99, 0.999, 0.9999].forEach(function (b2, i) {
        const pts = [], ptc = [];
        for (let t = 1; t <= 300; t++) { const m = 1 - Math.pow(st.b1, t), v = 1 - Math.pow(b2, t); pts.push([t, m / Math.sqrt(v)]); ptc.push([t, 1]); }
        sers.push({ name: 'β2 = ' + b2 + ' (no corr.)', color: Book.palette[i + 1], marker: 'none', dash: '6 3', points: pts, labelAt: 0, ldx: 10, ldy: i * 2 });
        firsts.push('β2 = ' + b2 + ': <b>' + (pts[0][1]).toFixed(2) + '·α</b> at t = 1, ' + pts[99][1].toFixed(2) + '·α at t = 100');
      });
      sers.push({ name: 'with correction', color: 'var(--accent)', marker: 'none', width: 3, points: [[1, 1], [300, 1]] });
      AD.logLines(plot, sers, { width: 640, height: 240, xmin: 1, xmax: 300, ymin: 0.5, ymax: 200, xlabel: 'step t', ylabel: '|Δt| / α (log)', xticks: [1, 100, 200, 300], mr: 10 });
      out.innerHTML = 'without bias correction, β1 = ' + st.b1 + ':<br>' + firsts.join('<br>') +
        '<br>formula at t = 1: (1 − β1)/√(1 − β2) · with correction: always 1 for a constant gradient' +
        '<br><span class="note">With β2 = 0.9999 and β1 = 0 the very first step is 100× larger than α intends. If α was chosen as a safe step, the uncorrected method jumps far outside the region where the gradient means anything. Chapter 31 shows the paper’s experiment on this.</span>';
    }
    draw();
  };
})();
