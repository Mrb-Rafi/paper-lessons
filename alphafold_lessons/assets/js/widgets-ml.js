/* Part II widgets: learning, softmax/distograms, cross-entropy, attention, gating. */
(function () {
  'use strict';
  const AF = window.AF, W = AF.widgets, el = AF.el, S = AF.svg, P = window.Prot;

  function softmax(l, T) { T = T || 1; const m = Math.max.apply(null, l); const e = l.map(function (x) { return Math.exp((x - m) / T); }); const s = e.reduce(function (a, b) { return a + b; }); return e.map(function (x) { return x / s; }); }
  AF.softmax = softmax;

  // ---------------------------------------------------------------- softmax as a distogram
  W.softmax = function (host) {
    AF.head(host, 'Softmax: a network’s guess about one distance', 'A network cannot say “I don’t know” with a single number, so AlphaFold predicts distances as a <b>probability over bins</b> (a <i>distogram</i>). Each bin gets a raw score (a <i>logit</i>); softmax turns scores into probabilities. Drag the sliders, then change the <b>temperature</b>.');
    const bins = ['2–4', '4–6', '6–8', '8–10', '10–12', '12–14', '14–16', '16–22'];
    let logits = [-1, 0.5, 2.2, 1.4, 0.2, -0.5, -1, -1.2], T = 1;
    const g = el('div', { class: 'grid2' }, host), left = el('div', null, g), right = el('div', null, g);
    logits.forEach(function (v, i) { const c = el('div', { class: 'controls', style: 'margin:2px 0' }, left); AF.slider(c, 'logit ' + bins[i] + ' Å', -4, 4, 0.1, v, function (x) { logits[i] = x; draw(); }, function (x) { return x.toFixed(1); }); });
    const c2 = el('div', { class: 'controls' }, left);
    AF.slider(c2, 'temperature T', 0.1, 3, 0.05, 1, function (x) { T = x; draw(); }, function (x) { return x.toFixed(2); });
    const plot = el('div', null, right), out = el('div', { class: 'out' }, right);
    function draw() {
      const p = softmax(logits, T);
      AF.chart.bars(plot, p.map(function (x, i) { return { label: bins[i], value: x, note: (x * 100).toFixed(0) + '%' }; }), { max: 1, height: 220, tickFmt: function (v) { return (v * 100).toFixed(0) + '%'; }, ylabel: 'probability' });
      const mids = [3, 5, 7, 9, 11, 13, 15, 19]; const mean = p.reduce(function (s, x, i) { return s + x * mids[i]; }, 0);
      out.textContent = 'p_b = exp(ℓ_b / T) / Σ_b′ exp(ℓ_b′ / T)\nexpected distance ≈ ' + mean.toFixed(1) + ' Å;  P(d < 8 Å, “in contact”) = ' + ((p[0] + p[1] + p[2]) * 100).toFixed(0) + '%\nAlphaFold’s real distogram head uses 64 bins from 2 to 22 Å for every residue pair.';
    }
    draw();
  };

  // ---------------------------------------------------------------- cross-entropy
  W.crossEntropy = function (host) {
    AF.head(host, 'Cross-entropy: the price of being surprised', 'In training we know the true distance bin. The loss is −log(probability the model gave to it). Confident and right ⇒ loss near 0; confident and wrong ⇒ a huge loss.');
    const ctr = el('div', { class: 'controls' }, host);
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    let p = 0.3;
    AF.slider(ctr, 'p(true bin)', 0.01, 1, 0.01, p, function (x) { p = x; draw(); }, function (x) { return x.toFixed(2); });
    function draw() {
      const pts = []; for (let x = 0.01; x <= 1.0001; x += 0.01) pts.push([x, -Math.log(x)]);
      AF.chart.lines(plot, [{ color: 'var(--purple)', points: pts, marker: 'none' }, { color: 'var(--accent)', points: [[p, -Math.log(p)]], r: 7, name: 'you are here' }], { xmin: 0, xmax: 1, ymin: 0, ymax: 4.8, xticks: [0, 0.25, 0.5, 0.75, 1], xlabel: 'probability given to the true answer', ylabel: 'loss = −log p', height: 230 });
      out.textContent = 'loss = −log(' + p.toFixed(2) + ') = ' + (-Math.log(p)).toFixed(3) + '\nAlphaFold uses this loss for its distogram head, its masked-MSA head and its confidence (pLDDT) head (Chapter 22).';
    }
    draw();
  };

  // ---------------------------------------------------------------- gradient descent
  W.gradDescent = function (host) {
    AF.head(host, 'Learning = rolling downhill on the loss', 'One parameter w, one loss curve. Each <b>step</b> computes the slope (gradient) and moves w a little against it: w ← w − η · slope. Try a tiny, a good, and a too-large learning rate η.');
    const f = function (w) { return 0.15 * (w - 3) * (w - 3) + 0.6 * Math.sin(1.4 * w) + 1.2; };
    const df = function (w) { return 0.3 * (w - 3) + 0.84 * Math.cos(1.4 * w); };
    let w = -2.5, lr = 0.6, hist = [];
    const ctr = el('div', { class: 'controls' }, host);
    AF.slider(ctr, 'learning rate η', 0.05, 5, 0.05, lr, function (x) { lr = x; }, function (x) { return x.toFixed(2); });
    AF.button(ctr, 'Step', function () { step(); draw(); }, 'primary');
    AF.button(ctr, 'Run 20 steps', function () { for (let i = 0; i < 20; i++) step(); draw(); });
    AF.button(ctr, 'Reset', function () { w = -2.5; hist = []; draw(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function step() { hist.push(w); w = AF.clamp(w - lr * df(w), -4, 9); }
    function draw() {
      const pts = []; for (let x = -4; x <= 9; x += 0.05) pts.push([x, f(x)]);
      AF.chart.lines(plot, [{ color: 'var(--purple)', points: pts, marker: 'none' }, { color: 'var(--muted)', points: hist.map(function (h) { return [h, f(h)]; }), r: 3, dash: '3 3', width: 1 }, { color: 'var(--accent)', points: [[w, f(w)]], r: 8 }], { xmin: -4, xmax: 9, ymin: 0, ymax: 6.5, xticks: [-4, -2, 0, 2, 4, 6, 8], xlabel: 'parameter w', ylabel: 'loss', height: 240 });
      out.textContent = 'step ' + hist.length + ':  w = ' + w.toFixed(3) + '   loss = ' + f(w).toFixed(3) + '   slope = ' + df(w).toFixed(3) + '\nAlphaFold does exactly this with millions of parameters at once, on 128 TPU v3 cores for about a week, plus ~4 days of fine-tuning (Methods).';
    }
    draw();
  };

  // ---------------------------------------------------------------- attention over residues (+ gating, + bias)
  W.attention = function (host) {
    AF.head(host, 'Attention: every residue looks at the others', 'Click a residue (the <i>query</i>). Its attention weights over all residues are softmax(q·k/√c + bias). Here q and k come from a small fixed projection of each residue’s chemistry. Turn on a <b>pair bias</b> to see how AlphaFold injects extra pairwise information straight into the attention logits — and a <b>gate</b> that scales the output.');
    const toy = AF.toy(), seq = toy.seq.slice(0, 14).split('');
    const emb = { hydrophobic: [1.2, 0.2], positive: [-0.6, 1.1], negative: [-0.6, -1.1], polar: [-0.3, 0.2], special: [0.1, -0.2] };
    const E = seq.map(function (a) { return emb[P.aaClass(a)]; });
    const Wq = [[0.9, 0.5], [-0.4, 1.4]], Wk = [[0.8, -0.3], [0.6, -1.3]];
    const mul = function (M, v) { return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]; };
    let qi = 3, bias = false, gate = 1;
    const ctr = el('div', { class: 'controls' }, host);
    const cb = el('label', null, ctr); const cbi = el('input', { type: 'checkbox' }, cb); cb.appendChild(document.createTextNode(' add pair bias b_ij = −0.4·|i − j| (prefer near neighbours)'));
    cbi.addEventListener('change', function () { bias = cbi.checked; draw(); });
    AF.slider(ctr, 'gate g = sigmoid(·)', 0, 1, 0.05, gate, function (x) { gate = x; draw(); }, function (x) { return x.toFixed(2); });
    const row = el('div', { class: 'tokrow' }, host), svgHost = el('div', null, host), out = el('div', { class: 'out' }, host);
    function weights(i) {
      const q = mul(Wq, E[i]);
      const s = E.map(function (e, j) { const k = mul(Wk, e); return (q[0] * k[0] + q[1] * k[1]) / Math.SQRT2 + (bias ? -0.4 * Math.abs(i - j) : 0); });
      return softmax(s);
    }
    function draw() {
      row.innerHTML = '';
      seq.forEach(function (a, i) { const c = el('span', { class: 'chip' + (i === qi ? ' hl' : ''), text: (i + 1) + ' ' + a, style: 'cursor:pointer;border-left:4px solid ' + P.aaColor(a) }, row); c.addEventListener('click', function () { qi = i; draw(); }); });
      const a = weights(qi);
      svgHost.innerHTML = '';
      const Wd = 640, H = 130, svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + H }, svgHost);
      const x = function (i) { return 30 + i * (Wd - 60) / (seq.length - 1); };
      seq.forEach(function (w, j) {
        if (a[j] > 0.002) S('path', { d: 'M' + x(qi) + ',100 Q' + (x(qi) + x(j)) / 2 + ',' + (100 - 30 - Math.abs(qi - j) * 5) + ' ' + x(j) + ',100', fill: 'none', stroke: 'var(--accent)', 'stroke-width': 0.5 + 9 * a[j] * gate, opacity: 0.25 + 0.75 * a[j] }, svg);
        S('text', { x: x(j), y: 120, 'text-anchor': 'middle', 'font-size': 13, class: 'svg-text', 'font-weight': j === qi ? 700 : 400, text: w }, svg);
        S('text', { x: x(j), y: 18, 'text-anchor': 'middle', 'font-size': 10.5, class: 'svg-text', text: (a[j] * 100).toFixed(0) + '%' }, svg);
      });
      out.textContent = 'output_i = g ⊙ Σ_j a_ij v_j      with   a_ij = softmax_j( q_i·k_j / √c + b_ij )' +
        '\nWeights for residue ' + (qi + 1) + ' sum to 100%. The gate g (here ' + gate.toFixed(2) + ') lets the network switch an update off or on — every attention layer in the Evoformer is gated this way.';
    }
    draw();
  };
})();
