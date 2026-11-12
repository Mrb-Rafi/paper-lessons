/* Part IV–V widgets: recycling, training, confidence, results, ablations. */
(function () {
  'use strict';
  const AF = window.AF, W = AF.widgets, el = AF.el, S = AF.svg, P = window.Prot;

  function foldStructure(n) { const toy = AF.toy(), f = P.foldFromContacts(toy, AF.predictContacts(n).top); f.step(0.003, 1500); let X = f.step.coords(); if (P.tmscore(P.mirror(X), toy.ca) > P.tmscore(X, toy.ca)) X = P.mirror(X); return P.transformAll(P.superpose(X, toy.ca), X); }
  AF.foldCache = {};
  AF.foldedModel = function (n) { if (!AF.foldCache[n]) AF.foldCache[n] = foldStructure(n); return AF.foldCache[n]; };
  function noisy(X, s, seed) { const r = P.rng(seed); return X.map(function (x) { return P.add(x, [s * P.gauss(r), s * P.gauss(r), s * P.gauss(r)]); }); }
  // predictions used by the confidence widgets
  AF.predictionChoices = [
    { value: 'good', label: 'a good prediction (≈0.3 Å error)' },
    { value: 'hinge', label: 'good domains, wrong hinge (helix 3 swung 50°)' },
    { value: 'f100', label: 'folded from 100-sequence coevolution' },
    { value: 'f30', label: 'folded from 30-sequence coevolution' }
  ];
  AF.makePrediction = function (kind) {
    const toy = AF.toy();
    if (kind === 'good') return noisy(toy.ca, 0.15, 4);
    if (kind === 'hinge') return AF.hinge(noisy(toy.ca, 0.15, 5), 50);
    if (kind === 'f100') return AF.foldedModel(100);
    return AF.foldedModel(30);
  };

  // ---------------------------------------------------------------- recycling trajectory (Fig. 4b, toy version)
  W.trajectory = function (host) {
    const toy = AF.toy();
    AF.head(host, 'Watching the network think: accuracy block by block', 'The paper attached a separate structure module to each of the 48 Evoformer blocks and ran 4 recycling iterations, giving 192 intermediate structures (Fig. 4b). This toy version shows the two behaviours they describe: an “easy” target settled within a few blocks, and a “hard” one that rearranges for most of the network. <i>The paths are hand-made; the GDT values are computed on them.</i>');
    const ctr = el('div', { class: 'controls' }, host);
    let b = 0;
    const sl = AF.slider(ctr, 'Evoformer block (incl. recycling)', 0, 192, 4, 0, function (x) { b = x; show(); });
    const plot = el('div', null, host);
    const g = el('div', { class: 'two-col' }, host), l = el('div', null, g), r = el('div', null, g);
    el('div', { class: 'small', text: 'easy target' }, l); el('div', { class: 'small', text: 'hard target' }, r);
    const v1 = P.Viewer(l, { height: 240, width: 320, hint: false }), v2 = P.Viewer(r, { height: 240, width: 320, hint: false });
    v1.setZoomFrom(toy.ca); v2.setZoomFrom(toy.ca);
    const out = el('div', { class: 'note' }, host);
    const start = AF.foldedModel(10), wrong = P.transformAll(P.superpose(P.mirror(toy.ca), toy.ca), noisy(P.mirror(toy.ca), 1.2, 6));
    const easyKeys = [[0, start], [10, noisy(toy.ca, 1.6, 7)], [40, noisy(toy.ca, 0.7, 8)], [192, noisy(toy.ca, 0.5, 9)]];
    const hardKeys = [[0, start], [30, wrong], [100, noisy(wrong, 0.6, 10)], [120, AF.hinge(noisy(toy.ca, 1.5, 11), 60)], [160, noisy(toy.ca, 1.0, 12)], [192, noisy(toy.ca, 0.6, 13)]];
    function at(keys, t) {
      for (let k = 0; k < keys.length - 1; k++) if (t <= keys[k + 1][0]) { const a = keys[k], c = keys[k + 1], f = (t - a[0]) / (c[0] - a[0]); return a[1].map(function (x, i) { return P.lerp(x, c[1][i], f); }); }
      return keys[keys.length - 1][1];
    }
    const blocks = []; for (let t = 0; t <= 192; t += 4) blocks.push(t);
    const gE = [], gH = [];
    let idx = 0;
    (function compute() {
      const stop = Math.min(blocks.length, idx + 6);
      for (; idx < stop; idx++) { const t = blocks[idx]; gE.push([t, P.gdt(at(easyKeys, t), toy.ca)]); gH.push([t, P.gdt(at(hardKeys, t), toy.ca)]); }
      chart();
      if (idx < blocks.length) setTimeout(compute, 0);
    })();
    function chart() {
      const c = AF.chart.lines(plot, [{ color: '#4c78a8', points: gE, marker: 'none', name: 'easy', labelAt: Math.min(8, gE.length - 1), ldx: 6, ldy: -10 }, { color: '#54a24b', points: gH, marker: 'none', name: 'hard', labelAt: Math.min(12, gH.length - 1), ldx: 6, ldy: 18 }], { xmin: 0, xmax: 192, ymin: 0, ymax: 100, xticks: [0, 48, 96, 144, 192], xlabel: 'Evoformer block (48 per recycling iteration)', ylabel: 'GDT', height: 220, mr: 50 });
      S('line', { x1: c.X(b), x2: c.X(b), y1: c.Y(0), y2: c.Y(100), stroke: '#e53e3e', 'stroke-dasharray': '4 3' }, c.svg);
    }
    function show() {
      v1.set([{ pts: toy.ca, color: 'var(--muted)', width: 6, opacity: 0.35 }, { pts: at(easyKeys, b), color: function (i) { return AF.rainbow(i / 49); }, width: 4 }], true);
      v2.set([{ pts: toy.ca, color: 'var(--muted)', width: 6, opacity: 0.35 }, { pts: at(hardKeys, b), color: function (i) { return AF.rainbow(i / 49); }, width: 4 }], true);
      if (idx >= blocks.length) chart();
      out.textContent = 'Block ' + b + ' (recycling iteration ' + Math.min(4, Math.floor(b / 48) + 1) + ' of 4). Grey = true structure. In the paper, LmrP (T1024) behaved like the easy curve and SARS-CoV-2 ORF8 (T1064) like the hard one.';
    }
    show();
  };

  // ---------------------------------------------------------------- the loss recipe
  W.lossMix = function (host) {
    AF.head(host, 'The training objective: one main loss and several helpers', 'AlphaFold minimises a weighted sum of losses (SI §1.9). Drag each loss’s current value and see its weighted share of the total. Switch to fine-tuning to add the extra terms.');
    const terms = [
      { k: 'FAPE', w: 0.5, v: 1.0, d: 'frame-aligned point error of the final structure (Chapter 20)' },
      { k: 'aux', w: 0.5, v: 1.2, d: 'FAPE + torsion-angle loss on every structure-module block' },
      { k: 'distogram', w: 0.3, v: 2.5, d: 'cross-entropy of 64 distance bins from the pair representation' },
      { k: 'masked MSA', w: 2.0, v: 1.4, d: 'BERT-style recovery of hidden MSA entries' },
      { k: 'pLDDT', w: 0.01, v: 2.0, d: 'cross-entropy of 50 lDDT bins (confidence head)' },
      { k: 'exp. resolved', w: 0.01, v: 0.3, d: 'is each atom experimentally resolved? (fine-tuning)', ft: true },
      { k: 'violation', w: 1.0, v: 0.2, d: 'bond lengths/angles and clashes (fine-tuning)', ft: true }
    ];
    let ft = false;
    const ctr = el('div', { class: 'controls' }, host);
    const cb = el('label', null, ctr), ci = el('input', { type: 'checkbox' }, cb); cb.appendChild(document.createTextNode(' fine-tuning stage'));
    ci.addEventListener('change', function () { ft = ci.checked; draw(); });
    const sl = el('div', null, host);
    terms.forEach(function (t) { const c = el('div', { class: 'controls', style: 'margin:2px 0' }, sl); t.inp = AF.slider(c, t.k + ' (weight ' + t.w + ')', 0, 4, 0.05, t.v, function (x) { t.v = x; draw(); }, function (x) { return x.toFixed(2); }); t.row = c; });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const act = terms.filter(function (t) { return ft || !t.ft; });
      terms.forEach(function (t) { t.row.style.opacity = ft || !t.ft ? 1 : 0.35; });
      const tot = act.reduce(function (s, t) { return s + t.w * t.v; }, 0);
      AF.chart.bars(plot, act.map(function (t) { return { label: t.k, value: t.w * t.v, note: (100 * t.w * t.v / tot).toFixed(0) + '%' }; }), { height: 210, ylabel: 'weight × loss' });
      out.textContent = 'L = ' + act.map(function (t) { return t.w + '·L_' + t.k.replace(/\s|\./g, ''); }).join(' + ') + ' = ' + tot.toFixed(3) + '\n' + act.map(function (t) { return '• ' + t.k + ': ' + t.d; }).join('\n');
    }
    draw();
  };

  // ---------------------------------------------------------------- masked MSA puzzle
  W.maskedMSA = function (host) {
    const toy = AF.toy();
    AF.head(host, 'The masked-MSA game (AlphaFold’s BERT objective)', 'One entry of the alignment is hidden (?). Which kind of residue was there? Hint: its column is coupled to the highlighted column, which is still visible. AlphaFold plays this game on random 15% of MSA entries during training — and must learn coevolution to win it.');
    const ctr = el('div', { class: 'controls' }, host);
    AF.button(ctr, 'New puzzle', function () { make(); }, 'primary');
    const box = el('div', { class: 'msa' }, host), choices = el('div', { class: 'controls' }, host), out = el('div', { class: 'out' }, host);
    const CLS = { B: '#d69e2e', S: '#a0aec0', K: '#3182ce', E: '#e53e3e' };
    let pz = null;
    const big = AF.toyMSA(1000, 77).cls;
    const pairs = AF.predictContacts(1000).top.filter(function (p, k) { return AF.predictContacts(1000).hit[k]; });
    function make() {
      const m = AF.toyMSA(30, 5), pr = pairs[Math.floor(Math.random() * pairs.length)], flip = Math.random() < 0.5;
      const i = flip ? pr[1] : pr[0], j = flip ? pr[0] : pr[1], row = 1 + Math.floor(Math.random() * 9);
      pz = { m: m, i: i, j: j, row: row, ans: m.cls[row][i], done: false };
      draw();
    }
    function draw() {
      box.innerHTML = '';
      for (let r = 0; r < 10; r++) {
        const row = el('div', { class: 'r' }, box); el('span', { class: 'lab', text: 'species ' + (r + 1) }, row);
        const s = pz.m.seqs[r], c = pz.m.cls[r];
        for (let q = 0; q < toy.L; q++) {
          const hidden = r === pz.row && q === pz.i && !pz.done;
          const sp = el('span', { text: hidden ? '?' : s[q] }, row);
          sp.style.background = hidden ? 'var(--ink)' : CLS[c[q]]; sp.style.color = '#fff';
          if (q === pz.j || q === pz.i) sp.style.outline = '2px solid ' + (q === pz.i ? 'var(--ink)' : '#2f855a');
          if (q !== pz.j && q !== pz.i) sp.style.opacity = 0.35;
        }
      }
      choices.innerHTML = '';
      'BSKE'.split('').forEach(function (k) { const b = AF.button(choices, P.TOY_CLASS_NAME[k], function () { guess(k); }); b.style.borderLeft = '6px solid ' + CLS[k]; });
      out.textContent = 'Hidden: column ' + (pz.i + 1) + ', species ' + (pz.row + 1) + '. Coupled column ' + (pz.j + 1) + ' (green outline) shows “' + P.TOY_CLASS_NAME[pz.m.cls[pz.row][pz.j]] + '” in that row.';
    }
    function guess(k) {
      const partner = pz.m.cls[pz.row][pz.j];
      const cnt = { B: 0, S: 0, K: 0, E: 0 }, cntAll = { B: 0, S: 0, K: 0, E: 0 }; let n = 0;
      big.forEach(function (s) { cntAll[s[pz.i]]++; if (s[pz.j] === partner) { cnt[s[pz.i]]++; n++; } });
      pz.done = true; draw();
      out.textContent = (k === pz.ans ? '✅ Correct: ' : '❌ It was: ') + P.TOY_CLASS_NAME[pz.ans] + '.' +
        '\n\nWhat the statistics of a deep alignment say:' +
        '\n  p(column ' + (pz.i + 1) + ')                         ' + 'BSKE'.split('').map(function (c) { return c + ' ' + (100 * cntAll[c] / big.length).toFixed(0) + '%'; }).join('  ') +
        '\n  p(column ' + (pz.i + 1) + ' | column ' + (pz.j + 1) + ' = ' + partner + ')     ' + 'BSKE'.split('').map(function (c) { return c + ' ' + (100 * cnt[c] / Math.max(1, n)).toFixed(0) + '%'; }).join('  ') +
        '\nKnowing the partner column sharpens the guess. A network that minimises this loss is forced to discover which columns are coupled — i.e. which residues touch.';
    }
    make();
  };

  // ---------------------------------------------------------------- self-distillation flow
  W.selfDistill = function (host) {
    AF.head(host, 'Self-distillation: learning from its own confident predictions', 'Click through the steps (Methods, “Training regimen”).');
    const steps = [
      ['Train on the PDB', 'Train a first model on PDB chains released before 30 April 2018, sampled in inverse proportion to the size of their 40%-identity cluster, cropped to 256 residues, batch 128 (128 TPU v3 cores, ~1 week + ~4 days fine-tuning).'],
      ['Predict new proteins', 'Use it to predict structures for 355,993 sequences from Uniclust30 (with full MSAs). No experimental structure exists for most of them.'],
      ['Keep the confident ones', 'Filter the predictions to a high-confidence subset (using the model’s own confidence estimates).'],
      ['Train again from scratch', 'Train the same architecture from scratch, sampling 75% of examples from the predicted-structure set (with sub-sampled MSAs) and 25% from the PDB. Cropping and MSA subsampling make it hard to simply copy the earlier predictions.'],
      ['Result', 'The self-distilled model is noticeably more accurate (top row of Fig. 4a: roughly +2 GDT on CASP14 and +2 lDDT-Cα on recent PDB chains, read off the figure).']
    ];
    let k = 0;
    const pipe = el('div', { class: 'pipeline' }, host), out = el('div', { class: 'out', style: 'font-family:var(--font-ui)' }, host);
    function draw() {
      pipe.innerHTML = '';
      steps.forEach(function (s, i) { if (i) el('span', { class: 'arr', text: '→' }, pipe); const b = el('span', { class: 'stage' + (i === k ? ' on' : ''), text: (i + 1) + '. ' + s[0] }, pipe); b.addEventListener('click', function () { k = i; draw(); }); });
      out.innerHTML = '<b>' + steps[k][0] + '.</b> ' + steps[k][1];
    }
    draw();
  };

  // ---------------------------------------------------------------- pLDDT
  W.plddtView = function (host) {
    const toy = AF.toy();
    AF.head(host, 'pLDDT: confidence for every residue', 'AlphaFold’s confidence head is trained to predict each residue’s lDDT-Cα (0–100) from the network’s own activations. Here we show what a <i>perfect</i> confidence head would output: the true per-residue lDDT of each toy prediction, painted with AlphaFold’s colour scale.');
    const ctr = el('div', { class: 'controls' }, host);
    let kind = 'hinge';
    AF.select(ctr, 'Prediction', AF.predictionChoices, kind, function (v) { kind = v; draw(); });
    const v = P.Viewer(host, { height: 320 }), plot = el('div', null, host);
    const leg = el('div', { class: 'legend plddt-legend' }, host);
    leg.innerHTML = '<span><i style="background:#0053d6"></i>very high (&gt; 90)</span><span><i style="background:#65cbf3"></i>confident (70–90)</span><span><i style="background:#ffdb13"></i>low (50–70)</span><span><i style="background:#ff7d45"></i>very low (&lt; 50)</span>';
    const out = el('div', { class: 'note' }, host);
    v.setZoomFrom(toy.ca);
    function draw() {
      const X = AF.makePrediction(kind), ld = P.lddt(X, toy.ca);
      v.set([{ pts: X, color: function (i) { return AF.plddtColor(ld.perResidue[i]); }, mode: 'tube+points', width: 5, r: 3 }], true);
      AF.chart.bars(plot, ld.perResidue.map(function (p, i) { return { label: i % 10 === 0 ? String(i + 1) : '', value: p, note: '', color: AF.plddtColor(p) }; }), { max: 100, height: 170, ylabel: 'lDDT' });
      out.textContent = 'mean = ' + ld.global.toFixed(1) + '. lDDT is local (only neighbours within 15 Å count) and needs no superposition: in the hinge prediction, residues whose neighbourhood stayed intact keep high scores, while residues that touched the swung helix lose points. In real AlphaFold outputs, very low pLDDT (< 50) often marks intrinsically disordered regions.';
    }
    draw();
  };

  // ---------------------------------------------------------------- PAE / pTM
  W.paeView = function (host) {
    const toy = AF.toy();
    AF.head(host, 'Predicted aligned error (PAE) and pTM', 'PAE<sub>ij</sub> is the expected error in residue j’s position when the prediction and truth are aligned on residue i’s frame — exactly the quantity inside FAPE. Below is the <i>actual</i> aligned error of each toy prediction (what a perfect PAE head would predict). Dark = small error. A wrong hinge shows up as two confident blocks with high error between them.');
    const ctr = el('div', { class: 'controls' }, host);
    let kind = 'hinge';
    AF.select(ctr, 'Prediction', AF.predictionChoices, kind, function (v) { kind = v; draw(); });
    const g = el('div', { class: 'two-col' }, host), left = el('div', null, g), out = el('div', { class: 'out' }, g);
    const cv = el('canvas', { width: 400, height: 400, class: 'heat', style: 'width:100%;max-width:380px' }, left);
    el('div', { class: 'small', text: 'row i = aligned on residue i · column j = scored residue · scale 0 → 30 Å' }, left);
    function draw() {
      const X = AF.makePrediction(kind), bb = P.backboneFromCA(X), F = P.framesOf(bb), L = toy.L;
      const f = P.fape(F, X, toy.frames, toy.ca, { clamp: 1e9 }), E = f.matrix, ctx = cv.getContext('2d'), c = 400 / L;
      for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) { const t = Math.min(30, E[i][j]) / 30; ctx.fillStyle = 'rgb(' + Math.round(30 + 225 * t) + ',' + Math.round(110 + 145 * t) + ',' + Math.round(50 + 205 * t) + ')'; ctx.fillRect(j * c, i * c, c + 0.5, c + 0.5); }
      const d0 = P.d0(L); let best = 0, bi = 0;
      for (let i = 0; i < L; i++) { let s = 0; for (let j = 0; j < L; j++) s += 1 / (1 + Math.pow(E[i][j] / d0, 2)); s /= L; if (s > best) { best = s; bi = i; } }
      out.textContent = 'pTM-style estimate from these errors:\n  max_i (1/L) Σ_j 1 / (1 + (e_ij / d0)²) = ' + best.toFixed(3) + '   (best anchor residue ' + (bi + 1) + ')' +
        '\nactual TM-score of the prediction: ' + P.tmscore(X, toy.ca).toFixed(3) +
        '\nd0(L = ' + L + ') = ' + d0.toFixed(2) + ' Å' +
        '\n\nAlphaFold predicts a distribution over binned errors for every (i, j) from the pair representation and computes pTM from it (SI §1.9.7). The paper’s Fig. 2d: TM-score ≈ 0.98 × pTM + 0.07 on 10,795 recent PDB chains.';
    }
    draw();
  };

  // ---------------------------------------------------------------- calibration fits from Fig. 2c,d
  W.calibration = function (host) {
    AF.head(host, 'How trustworthy is the confidence? (paper Fig. 2c, d)', 'The paper fitted straight lines between predicted and true accuracy on 10,795 recent PDB chains. Slide a predicted value and read off what the fits say.');
    const ctr = el('div', { class: 'controls' }, host);
    let pl = 85, pt = 0.8;
    AF.slider(ctr, 'average pLDDT', 30, 100, 1, pl, function (x) { pl = x; draw(); });
    AF.slider(ctr, 'pTM', 0.2, 1, 0.01, pt, function (x) { pt = x; draw(); }, function (x) { return x.toFixed(2); });
    const g = el('div', { class: 'grid2' }, host), a = el('div', null, g), b = el('div', null, g), out = el('div', { class: 'out' }, host);
    function draw() {
      const l1 = [[30, 0.997 * 30 - 1.17], [100, 0.997 * 100 - 1.17]], l2 = [[0.2, 0.98 * 0.2 + 0.07], [1, 0.98 + 0.07]];
      AF.chart.lines(a, [{ color: 'var(--muted)', points: [[30, 30], [100, 100]], marker: 'none', dash: '3 3', width: 1 }, { color: 'var(--accent)', points: l1, marker: 'none' }, { color: '#e53e3e', points: [[pl, 0.997 * pl - 1.17]], r: 7 }], { xmin: 30, xmax: 100, ymin: 20, ymax: 100, xticks: [40, 60, 80, 100], xlabel: 'average pLDDT', ylabel: 'lDDT-Cα', height: 220 });
      AF.chart.lines(b, [{ color: 'var(--muted)', points: [[0.2, 0.2], [1, 1]], marker: 'none', dash: '3 3', width: 1 }, { color: 'var(--accent)', points: l2, marker: 'none' }, { color: '#e53e3e', points: [[pt, 0.98 * pt + 0.07]], r: 7 }], { xmin: 0.2, xmax: 1, ymin: 0.2, ymax: 1.06, xticks: [0.2, 0.4, 0.6, 0.8, 1], xlabel: 'pTM', ylabel: 'TM-score', ydec: 2, height: 220 });
      out.textContent = 'lDDT-Cα = 0.997 × pLDDT − 1.17 = ' + (0.997 * pl - 1.17).toFixed(1) + '   (Pearson r = 0.76)\nTM-score = 0.98 × pTM + 0.07 = ' + (0.98 * pt + 0.07).toFixed(2) + '   (Pearson r = 0.85)\nThe fits describe the average; individual chains scatter around them (that is what r < 1 means).';
    }
    draw();
  };

  // ---------------------------------------------------------------- CASP14 headline numbers
  W.casp14Chart = function (host) {
    AF.head(host, 'CASP14: AlphaFold versus the next best method', 'Median accuracy over the CASP14 domains (n = 87) with 95% confidence intervals, from the paper’s main text. Lower is better. The dashed line marks the width of a carbon atom (~1.4 Å).');
    const ctr = el('div', { class: 'controls' }, host);
    let mode = 'bb';
    AF.select(ctr, 'Measure', [{ value: 'bb', label: 'backbone: Cα r.m.s.d.95' }, { value: 'aa', label: 'all-atom r.m.s.d.95' }], mode, function (v) { mode = v; draw(); });
    const host2 = el('div', null, host);
    const data = { bb: [['AlphaFold', 0.96, 0.85, 1.16], ['next best', 2.8, 2.7, 4.0]], aa: [['AlphaFold', 1.5, 1.2, 1.6], ['next best', 3.5, 3.1, 4.2]] };
    function draw() {
      host2.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 640 230' }, host2), X0 = 120, sc = 110;
      for (let v = 0; v <= 4.5; v += 1) { S('line', { x1: X0 + v * sc, x2: X0 + v * sc, y1: 20, y2: 190, stroke: 'var(--line)' }, svg); S('text', { x: X0 + v * sc, y: 208, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: v + ' Å' }, svg); }
      data[mode].forEach(function (d, i) {
        const y = 45 + i * 80;
        S('rect', { x: X0, y: y, width: d[1] * sc, height: 40, rx: 4, fill: i ? '#9cc3e6' : '#1f5fa8' }, svg);
        S('line', { x1: X0 + d[2] * sc, x2: X0 + d[3] * sc, y1: y + 20, y2: y + 20, stroke: 'var(--ink)', 'stroke-width': 2 }, svg);
        [d[2], d[3]].forEach(function (e) { S('line', { x1: X0 + e * sc, x2: X0 + e * sc, y1: y + 12, y2: y + 28, stroke: 'var(--ink)', 'stroke-width': 2 }, svg); });
        S('text', { x: X0 - 8, y: y + 25, 'text-anchor': 'end', 'font-size': 13, 'font-weight': 700, class: 'svg-text', text: d[0] }, svg);
        S('text', { x: X0 + d[3] * sc + 14, y: y + 25, 'font-size': 12, class: 'svg-text', text: d[1] + ' Å  [' + d[2] + '–' + d[3] + ']' }, svg);
      });
      S('line', { x1: X0 + 1.4 * sc, x2: X0 + 1.4 * sc, y1: 20, y2: 190, stroke: '#e53e3e', 'stroke-dasharray': '5 4', 'stroke-width': 1.5 }, svg);
      S('text', { x: X0 + 1.4 * sc + 4, y: 16, 'font-size': 11, fill: '#e53e3e', text: 'carbon atom ≈ 1.4 Å' }, svg);
    }
    draw();
  };

  // ---------------------------------------------------------------- run time
  W.timing = function (host) {
    AF.head(host, 'How long does a prediction take?', 'Representative times for one model on one V100 GPU, quoted in the Methods (open-source code). Between the quoted lengths we interpolate on a log–log scale, so treat in-between values as rough.');
    const ctr = el('div', { class: 'controls' }, host);
    let n = 384, ens = false;
    AF.slider(ctr, 'residues', 100, 2500, 10, n, function (x) { n = x; draw(); });
    const cb = el('label', null, ctr), ci = el('input', { type: 'checkbox' }, cb); cb.appendChild(document.createTextNode(' CASP14-style ensembling (8× slower)'));
    ci.addEventListener('change', function () { ens = ci.checked; draw(); });
    const out = el('div', { class: 'out' }, host);
    const pts = { no: [[256, 0.6], [384, 1.1], [2500, 126]], en: [[256, 4.8], [384, 9.2], [2500, 1080]] };
    function interp(arr, x) {
      const lx = Math.log(x); let a = arr[0], b = arr[1];
      if (x > arr[1][0]) { a = arr[1]; b = arr[2]; }
      const t = (lx - Math.log(a[0])) / (Math.log(b[0]) - Math.log(a[0]));
      return Math.exp(Math.log(a[1]) + t * (Math.log(b[1]) - Math.log(a[1])));
    }
    function fmt(m) { return m < 1 ? (m * 60).toFixed(0) + ' s' : m < 90 ? m.toFixed(1) + ' min' : (m / 60).toFixed(1) + ' h'; }
    function draw() {
      const m = interp(ens ? pts.en : pts.no, n);
      out.textContent = n + ' residues → about ' + fmt(m) + ' of GPU time for the neural network' + (n > 1300 && !ens ? '\n(> ~1,300 residues exceeds a 16 GB V100 and needs unified memory across GPUs)' : '') +
        '\nQuoted: 256 res 0.6 min (4.8 with ensembling) · 384 res 1.1 min (9.2) · 2,500 res 2.1 h (18 h).' +
        '\nMSA search and the final Amber relaxation take extra CPU time. Compare: months to years for an experimental structure.';
    }
    draw();
  };

  // ---------------------------------------------------------------- ablations (approximate values read from Fig. 4a)
  W.ablations = function (host) {
    AF.head(host, 'What happens when you remove a piece? (Fig. 4a)', 'Each row retrains AlphaFold without one component and reports the change in accuracy versus the baseline. <b>Values are approximate means read off the paper’s figure</b>; see Fig. 4a (reproduced in <a href="#ch-recycling">Chapter 21</a>) for exact points and error bars.');
    const ctr = el('div', { class: 'controls' }, host);
    let set = 'casp';
    AF.select(ctr, 'Test set', [{ value: 'casp', label: 'CASP14 domains (GDT)' }, { value: 'pdb', label: 'recent PDB chains (lDDT-Cα)' }], set, function (v) { set = v; draw(); });
    const rows = [
      ['With self-distillation training', 2.4, 1.9], ['No templates', -0.2, -0.4], ['No auxiliary distogram head', -1.2, -0.3], ['No raw MSA (use pairwise frequencies)', -2.2, -1.3],
      ['No IPA (use direct projection)', -2.2, -0.3], ['No auxiliary masked MSA head', -3.0, -1.8], ['No recycling', -4.5, -1.8], ['No triangles, biasing or gating', -4.5, -1.7],
      ['No end-to-end structure gradients', -6.0, -2.7], ['No IPA and no recycling', -16.0, -3.0]
    ];
    const box = el('div', null, host);
    function draw() {
      box.innerHTML = '';
      const k = set === 'casp' ? 1 : 2, lo = set === 'casp' ? -18 : -4, hi = set === 'casp' ? 4 : 2.5;
      const svg = S('svg', { viewBox: '0 0 640 ' + (rows.length * 26 + 40) }, box), X0 = 300, X = function (v) { return X0 + (v - lo) / (hi - lo) * 320; };
      S('line', { x1: X(0), x2: X(0), y1: 6, y2: rows.length * 26 + 10, stroke: 'var(--muted)' }, svg);
      rows.forEach(function (r, i) {
        const y = 10 + i * 26, v = r[k];
        S('rect', { x: Math.min(X(0), X(v)), y: y, width: Math.abs(X(v) - X(0)), height: 18, rx: 3, fill: v > 0 ? '#2f855a' : '#c0392b', opacity: 0.85 }, svg);
        S('text', { x: X0 - 8, y: y + 13, 'text-anchor': 'end', 'font-size': 11.5, class: 'svg-text', text: r[0] }, svg);
        S('text', { x: v >= 0 ? X(v) + 5 : X(v) - 5, y: y + 13, 'text-anchor': v >= 0 ? 'start' : 'end', 'font-size': 11, class: 'svg-text', text: (v > 0 ? '+' : '') + v.toFixed(1) }, svg);
      });
      S('text', { x: X(0), y: rows.length * 26 + 30, 'text-anchor': 'middle', 'font-size': 11.5, class: 'svg-text', text: (set === 'casp' ? 'GDT' : 'lDDT-Cα') + ' difference vs baseline' }, svg);
    }
    draw();
  };

  // ---------------------------------------------------------------- sequence databases (Methods)
  W.msaSources = function (host) {
    AF.head(host, 'Removing sequence databases (Methods)', 'Re-running CASP14 with the same models but MSAs built without the metagenomic databases. Mean change in GDT.');
    AF.chart.bars(el('div', null, host), [{ label: 'no BFD', value: 0.4, note: '−0.4', color: '#e8a29a' }, { label: 'no MGnify', value: 0.7, note: '−0.7', color: '#e07b70' }, { label: 'neither', value: 6.1, note: '−6.1', color: '#c0392b' }], { height: 200, ylabel: 'GDT lost', max: 8 });
    el('div', { class: 'note', text: 'Most targets barely changed, but a few lost 20+ GDT — the ones whose MSA fell below the ~30-sequence threshold. The two databases largely overlap, so losing one hurts little; losing both hurts a lot.' }, host);
  };
})();
