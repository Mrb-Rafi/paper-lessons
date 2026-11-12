/* Part IV widgets: the AlphaFold network — representations, Evoformer, triangles, frames, IPA, structure module, FAPE. */
(function () {
  'use strict';
  const AF = window.AF, W = AF.widgets, el = AF.el, S = AF.svg, P = window.Prot;

  // apply a hinge rotation to a backbone (all three atoms of residues ≥ from)
  AF.hingeBB = function (bb, ang, from) {
    from = from == null ? 32 : from;
    const X = bb.map(function (r) { return r.CA; });
    const pivot = X[from], axis = P.unit(P.sub(X[from + 2], X[from - 2])), R = P.rotAxis(axis, ang * P.DEG);
    const f = function (x) { return P.add(P.mv(R, P.sub(x, pivot)), pivot); };
    return bb.map(function (r, i) { return i < from ? { N: r.N, CA: r.CA, C: r.C } : { N: f(r.N), CA: f(r.CA), C: f(r.C) }; });
  };
  AF.mirrorBB = function (bb) { const m = function (x) { return [-x[0], x[1], x[2]]; }; return bb.map(function (r) { return { N: m(r.N), CA: m(r.CA), C: m(r.C) }; }); };
  AF.noisyBB = function (bb, sigma, seed) {
    const r = P.rng(seed || 3);
    return bb.map(function (b) { const o = [sigma * P.gauss(r), sigma * P.gauss(r), sigma * P.gauss(r)]; return { N: P.add(b.N, o), CA: P.add(b.CA, o), C: P.add(b.C, o) }; });
  };

  // ---------------------------------------------------------------- architecture explorer
  W.archExplorer = function (host) {
    AF.head(host, 'The AlphaFold pipeline, box by box', 'This is the paper’s Fig. 1e redrawn. <b>Click any box</b> to see what it is, what shape of data flows through it, and which chapter explains it.');
    const boxes = [
      { id: 'seq', x: 10, y: 120, w: 90, h: 40, t: 'Input\nsequence', d: 'The amino-acid sequence of the protein: N<sub>res</sub> letters. It is the only input that is strictly required; everything else is looked up from databases using it.', ch: 'ch-protein' },
      { id: 'msa', x: 130, y: 30, w: 110, h: 50, t: 'Genetic search\n→ MSA', d: 'Sequence databases (UniRef90, MGnify, BFD, Uniclust30) are searched with jackhmmer and HHBlits for evolutionary relatives, which are aligned into an MSA. Search is tuned for high recall: spurious hits are expected and the network learns to cope.', ch: 'ch-msa' },
      { id: 'tpl', x: 130, y: 200, w: 110, h: 50, t: 'Structure search\n→ templates', d: 'The PDB70 database is searched (HHSearch) for already-solved proteins with similar sequences. Their 3-D coordinates become “templates”, encoded into the pair representation. AlphaFold still works well without them (ablation in Fig. 4a).', ch: 'ch-history' },
      { id: 'mrep', x: 270, y: 30, w: 110, h: 50, t: 'MSA repr.\n(s, r, c)', d: 'A 3-D array: one learned vector of c = 256 numbers for every sequence s and residue r. It starts from the raw MSA and is refined in every Evoformer block.', ch: 'ch-reps' },
      { id: 'prep', x: 270, y: 200, w: 110, h: 50, t: 'Pair repr.\n(r, r, c)', d: 'One vector of c = 128 numbers for every <i>pair</i> of residues (i, j). It starts from the sequence (relative positions) and templates, and ends up encoding distances and orientations.', ch: 'ch-reps' },
      { id: 'evo', x: 410, y: 60, w: 100, h: 160, t: 'Evoformer\n48 blocks', d: 'The trunk. 48 blocks, each with its own weights, that exchange information between the MSA and pair representations: row and column attention, outer product mean, triangle updates and triangle attention.', ch: 'ch-evoformer' },
      { id: 'single', x: 540, y: 30, w: 90, h: 50, t: 'Single repr.\n(r, c)', d: 'The first row of the MSA representation (the query sequence), linearly projected to c = 384 numbers per residue. With the pair representation it feeds the structure module.', ch: 'ch-structure' },
      { id: 'sm', x: 660, y: 60, w: 100, h: 160, t: 'Structure\nmodule\n8 blocks', d: 'Turns representations into 3-D: each residue is a rigid frame (rotation + translation), starting at the origin, updated by invariant point attention. Then side-chain torsion angles give every heavy atom. 8 blocks that share weights.', ch: 'ch-ipa' },
      { id: 'out', x: 790, y: 100, w: 90, h: 80, t: '3-D structure\n+ confidence', d: 'Coordinates of all heavy atoms, plus per-residue confidence (pLDDT) and, in the fine-tuned models, a predicted aligned error and pTM score.', ch: 'ch-confidence' },
      { id: 'rec', x: 330, y: 270, w: 440, h: 34, t: '← Recycling (three times)', d: 'The outputs (final pair and first-row MSA representations, and the predicted structure) are fed back as extra inputs and the whole network is run again — three extra times at inference.', ch: 'ch-recycling' }
    ];
    const links = [['seq', 'msa'], ['seq', 'tpl'], ['msa', 'mrep'], ['tpl', 'prep'], ['seq', 'prep'], ['mrep', 'evo'], ['prep', 'evo'], ['evo', 'single'], ['single', 'sm'], ['evo', 'sm'], ['sm', 'out']];
    const svg = S('svg', { viewBox: '0 0 890 315' }, host);
    const by = {}; boxes.forEach(function (b) { by[b.id] = b; });
    S('defs', null, svg).innerHTML = '<marker id="afarr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="var(--muted)"/></marker>';
    links.forEach(function (l) { const a = by[l[0]], b = by[l[1]]; S('line', { x1: a.x + a.w, y1: a.y + a.h / 2, x2: b.x - 2, y2: b.y + b.h / 2, stroke: 'var(--muted)', 'stroke-width': 1.5, 'marker-end': 'url(#afarr)' }, svg); });
    const info = el('div', { class: 'out', style: 'font-family:var(--font-ui)' }, host);
    let sel = 'evo';
    const rects = {};
    boxes.forEach(function (b) {
      const g = S('g', { style: 'cursor:pointer' }, svg);
      rects[b.id] = S('rect', { x: b.x, y: b.y, width: b.w, height: b.h, rx: 9, fill: 'var(--bg)', stroke: 'var(--line)', 'stroke-width': 1.5 }, g);
      const lines = b.t.split('\n');
      lines.forEach(function (ln, i) { S('text', { x: b.x + b.w / 2, y: b.y + b.h / 2 + (i - (lines.length - 1) / 2) * 14 + 4, 'text-anchor': 'middle', 'font-size': 12, class: 'svg-text', text: ln }, g); });
      g.addEventListener('click', function () { sel = b.id; draw(); });
    });
    function draw() {
      boxes.forEach(function (b) { rects[b.id].setAttribute('fill', b.id === sel ? 'var(--accent-soft)' : 'var(--bg)'); rects[b.id].setAttribute('stroke', b.id === sel ? 'var(--accent)' : 'var(--line)'); });
      const b = by[sel];
      info.innerHTML = '<b>' + b.t.replace(/\n/g, ' ') + '</b><br>' + b.d + '<br><a href="#' + b.ch + '">→ read more</a>';
    }
    draw();
  };

  // ---------------------------------------------------------------- representation shapes and memory
  W.repShapes = function (host) {
    AF.head(host, 'How big are the representations?', 'Pick a protein length and MSA depth. AlphaFold’s main arrays scale as <b>s × r</b> (MSA) and <b>r × r</b> (pair); triangle attention even touches r³ numbers per head. This is why memory grows roughly with the square of the length.');
    const ctr = el('div', { class: 'controls' }, host);
    let r = 384, s = 512;
    AF.slider(ctr, 'residues r', 50, 2700, 10, r, function (x) { r = x; draw(); });
    AF.slider(ctr, 'MSA rows s', 1, 512, 1, s, function (x) { s = x; draw(); });
    const pic = el('div', null, host), stats = el('div', null, host), out = el('div', { class: 'note' }, host);
    function mb(n) { const b = n * 4; return b > 1e9 ? (b / 1e9).toFixed(2) + ' GB' : (b / 1e6).toFixed(1) + ' MB'; }
    function draw() {
      const msa = s * r * 256, pair = r * r * 128, tri = r * r * r * 4, single = r * 384;
      stats.innerHTML = [['MSA repr. s×r×256', msa], ['pair repr. r×r×128', pair], ['single repr. r×384', single], ['one triangle-attention logit tensor r×r×r×4 heads', tri]].map(function (x) { return '<div class="stat"><b>' + mb(x[1]) + '</b><span>' + x[0] + ' (' + x[1].toExponential(2) + ' numbers)</span></div>'; }).join('');
      pic.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 640 170' }, pic), sc = 150 / 2700;
      S('rect', { x: 10, y: 10, width: Math.max(2, r * sc * 1.5), height: Math.max(2, s / 512 * 150 * 0.5), fill: '#f0a58f', stroke: '#c0392b' }, svg);
      S('text', { x: 14, y: 165, 'font-size': 11, class: 'svg-text', text: 'MSA: ' + s + ' × ' + r }, svg);
      S('rect', { x: 330, y: 10, width: Math.max(2, r * sc), height: Math.max(2, r * sc), fill: '#9cc3e6', stroke: '#1f5fa8' }, svg);
      S('text', { x: 334, y: 165, 'font-size': 11, class: 'svg-text', text: 'pair: ' + r + ' × ' + r }, svg);
      out.innerHTML = 'Counting 4 bytes per number. The real system avoids storing everything at once (it processes chunks and recomputes activations), but the paper notes that memory is “approximately quadratic in the number of residues”: a 16 GB V100 handles about <b>1,300 residues</b> without ensembling, and a 2,500-residue protein needs unified memory spanning several GPUs (Methods).';
    }
    draw();
  };

  // ---------------------------------------------------------------- row / column attention on the MSA
  W.axialAttention = function (host) {
    AF.head(host, 'Attention along rows and along columns', 'Attending from every MSA cell to every other cell would cost (s·r)² — far too much. AlphaFold instead attends <b>along a row</b> (one sequence, all residues) or <b>along a column</b> (one residue, all sequences). <b>Click a cell</b> of the MSA and switch modes.');
    const s = 6, r = 12;
    let mode = 'row', ci = [1, 4];
    const ctr = el('div', { class: 'controls' }, host);
    AF.select(ctr, 'Mode', [{ value: 'row', label: 'row-wise gated self-attention with pair bias' }, { value: 'col', label: 'column-wise gated self-attention' }], mode, function (v) { mode = v; draw(); });
    const svg = S('svg', { viewBox: '0 0 640 250' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const c = 22;
      S('text', { x: 10, y: 14, 'font-size': 12, 'font-weight': 700, class: 'svg-text', text: 'MSA representation (s × r)' }, svg);
      for (let a = 0; a < s; a++) for (let b = 0; b < r; b++) {
        const isQ = a === ci[0] && b === ci[1], inRow = mode === 'row' && a === ci[0], inCol = mode === 'col' && b === ci[1];
        const rect = S('rect', { x: 10 + b * c, y: 24 + a * c, width: c - 2, height: c - 2, rx: 3, fill: isQ ? '#e53e3e' : (inRow || inCol) ? '#f0a58f' : 'var(--bg-soft)', stroke: 'var(--line)', style: 'cursor:pointer' }, svg);
        rect.addEventListener('click', function () { ci = [a, b]; draw(); });
      }
      for (let a = 0; a < s; a++) S('text', { x: 10 + r * c + 4, y: 24 + a * c + 15, 'font-size': 10, class: 'svg-text', text: a === 0 ? 'query' : 'seq ' + (a + 1) }, svg);
      const ox = 360;
      S('text', { x: ox, y: 14, 'font-size': 12, 'font-weight': 700, class: 'svg-text', text: 'Pair representation (r × r)' }, svg);
      const pc = 17;
      for (let a = 0; a < r; a++) for (let b = 0; b < r; b++) {
        const used = mode === 'row' && a === ci[1];
        S('rect', { x: ox + b * pc, y: 24 + a * pc, width: pc - 1.5, height: pc - 1.5, rx: 2, fill: used ? '#5a9bd6' : 'var(--bg-soft)', stroke: 'var(--line)' }, svg);
      }
      if (mode === 'row') S('text', { x: ox, y: 24 + r * pc + 16, 'font-size': 11, class: 'svg-text', text: 'row i = ' + (ci[1] + 1) + ' supplies the bias b_ij for every key j' }, svg);
      else S('text', { x: ox, y: 24 + r * pc + 16, 'font-size': 11, class: 'svg-text', text: 'column attention does not read the pair representation' }, svg);
      out.textContent = mode === 'row' ?
        'Row attention, sequence s = ' + (ci[0] + 1) + ', residue i = ' + (ci[1] + 1) + ':\n  a_sij = softmax_j( q_si · k_sj / √c + b_ij ),   o_si = g_si ⊙ Σ_j a_sij v_sj\nThe query cell looks at all r residues of the SAME sequence. The bias b_ij is a linear projection of the pair representation z_ij — this is how the pair representation steers the MSA (“closing the loop”). Cost per sequence: r².' :
        'Column attention, sequence s = ' + (ci[0] + 1) + ', residue i = ' + (ci[1] + 1) + ':\n  a_sti = softmax_t( q_si · k_ti / √c ),   o_si = g_si ⊙ Σ_t a_sti v_ti\nThe query cell looks at the SAME residue in all s sequences — letting information flow between species at each position. Cost per residue: s².';
    }
    draw();
  };

  // ---------------------------------------------------------------- outer product mean
  W.outerProduct = function (host) {
    const toy = AF.toy();
    AF.head(host, 'Outer product mean: how the MSA writes into the pair representation', 'For each sequence s, take a vector for column i and a vector for column j, form their <b>outer product</b> (every entry of one times every entry of the other), and <b>average over sequences</b>. If the vectors are just one-hot codes of the residue kind, the result is exactly the table of pair frequencies from the coevolution chapter. AlphaFold uses learned 32-number vectors instead.');
    const best = AF.predictContacts(300).top.find(function (p, k) { return AF.predictContacts(300).hit[k]; }) || [3, 40];
    let i = best[0], j = best[1];
    const ctr = el('div', { class: 'controls' }, host);
    AF.slider(ctr, 'column i', 1, toy.L, 1, i + 1, function (x) { i = x - 1; draw(); });
    AF.slider(ctr, 'column j', 1, toy.L, 1, j + 1, function (x) { j = x - 1; draw(); });
    const row = el('div', { class: 'two-col' }, host), ex = el('div', null, row), res = el('div', null, row), out = el('div', { class: 'out' }, host);
    const K = 'BSKE';
    function mat(parent, M, title, max) {
      const t = el('table', { style: 'width:auto;margin:4px 0' }, parent); const cap = el('caption', { text: title, style: 'caption-side:top;text-align:left;font-size:12px;color:var(--muted)' }, t);
      const h = el('tr', null, t); el('th', { text: '' }, h); K.split('').forEach(function (k) { el('th', { text: k, style: 'padding:2px 6px' }, h); });
      M.forEach(function (rr, a) { const tr = el('tr', null, t); el('th', { text: K[a], style: 'padding:2px 6px' }, tr); rr.forEach(function (v) { el('td', { text: max === 1 ? (v ? '1' : '·') : v.toFixed(2), style: 'text-align:center;padding:2px 6px;background:' + AF.heat(0.85 * v / (max || 1)) + ';color:' + (v / (max || 1) > 0.55 ? '#fff' : 'var(--ink)') }, tr); }); });
    }
    function draw() {
      const m = AF.toyMSA(300).cls, N = m.length;
      ex.innerHTML = ''; res.innerHTML = '';
      for (let s = 0; s < 2; s++) {
        const a = K.indexOf(m[s][i]), b = K.indexOf(m[s][j]);
        const M = [0, 1, 2, 3].map(function (x) { return [0, 1, 2, 3].map(function (y) { return x === a && y === b ? 1 : 0; }); });
        mat(ex, M, 'sequence ' + (s + 1) + ': a_s,i = onehot(' + m[s][i] + ') ⊗ b_s,j = onehot(' + m[s][j] + ')', 1);
      }
      const F = [0, 1, 2, 3].map(function () { return [0, 0, 0, 0]; }), fi = [0, 0, 0, 0], fj = [0, 0, 0, 0];
      m.forEach(function (sq) { const a = K.indexOf(sq[i]), b = K.indexOf(sq[j]); F[a][b] += 1 / N; fi[a] += 1 / N; fj[b] += 1 / N; });
      const mx = Math.max.apply(null, F.map(function (r) { return Math.max.apply(null, r); }));
      mat(res, F, 'mean over all ' + N + ' sequences = pair frequencies f_ij(a, b)', mx);
      const Cov = F.map(function (r, a) { return r.map(function (v, b) { return v - fi[a] * fj[b]; }); });
      let mi = 0; F.forEach(function (r, a) { r.forEach(function (v, b) { if (v > 0) mi += v * Math.log(v / (fi[a] * fj[b])); }); });
      const covMax = Math.max.apply(null, Cov.map(function (r) { return Math.max.apply(null, r.map(Math.abs)); }));
      out.textContent = 'columns ' + (i + 1) + ' and ' + (j + 1) + ' · distance ' + P.dist(toy.ca[i], toy.ca[j]).toFixed(1) + ' Å · MI = ' + mi.toFixed(3) + ' · largest |covariance| = ' + covMax.toFixed(3) +
        '\n\nAlphaFold (SI Alg. 10):  a_si, b_sj = Linear(LayerNorm(m_si)), Linear(LayerNorm(m_sj))   (32 numbers each)' +
        '\n                         o_ij = flatten( mean_s ( a_si ⊗ b_sj ) )   (32 × 32 = 1,024 numbers)' +
        '\n                         z_ij ← z_ij + Linear(o_ij)' +
        '\nBecause a and b are learned, the network can compute frequency tables, covariances, or anything richer — and it does so in all 48 blocks, not once.';
    }
    draw();
  };

  // ---------------------------------------------------------------- triangle inequality
  W.triangleIneq = function (host) {
    AF.head(host, 'Pair distances must be consistent: the triangle inequality', 'Three residues i, j, k. Set the three pair distances freely — as a pair representation could. Only some combinations are possible in real 3-D space: every side must be shorter than the other two combined.');
    const ctr = el('div', { class: 'controls' }, host);
    const d = { ij: 10, jk: 8, ik: 11 };
    Object.keys(d).forEach(function (k) { AF.slider(ctr, 'd(' + k[0] + ',' + k[1] + ')', 1, 20, 0.5, d[k], function (x) { d[k] = x; draw(); }, function (x) { return x.toFixed(1) + ' Å'; }); });
    const svg = S('svg', { viewBox: '0 0 640 230' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const a = d.ij, b = d.jk, c = d.ik, ok = a + b >= c && a + c >= b && b + c >= a, sc = 10;
      const I = [320 - a * sc / 2, 215], J = [320 + a * sc / 2, 215];
      if (ok) {
        const x = (c * c - b * b + a * a) / (2 * a), y = Math.sqrt(Math.max(0, c * c - x * x));
        const K = [I[0] + x * sc, 215 - y * sc];
        [[I, J, 'ij'], [J, K, 'jk'], [I, K, 'ik']].forEach(function (s) { S('line', { x1: s[0][0], y1: s[0][1], x2: s[1][0], y2: s[1][1], stroke: 'var(--accent)', 'stroke-width': 3 }, svg); S('text', { x: (s[0][0] + s[1][0]) / 2 + 6, y: (s[0][1] + s[1][1]) / 2 - 6, 'font-size': 12, class: 'svg-text', text: d[s[2]].toFixed(1) }, svg); });
        [[I, 'i'], [J, 'j'], [K, 'k']].forEach(function (p) { S('circle', { cx: p[0][0], cy: p[0][1], r: 13, fill: 'var(--panel)', stroke: 'var(--ink-soft)', 'stroke-width': 2 }, svg); S('text', { x: p[0][0], y: p[0][1] + 4, 'text-anchor': 'middle', 'font-size': 13, 'font-style': 'italic', class: 'svg-text', text: p[1] }, svg); });
      } else {
        S('text', { x: 320, y: 110, 'text-anchor': 'middle', 'font-size': 16, 'font-weight': 700, fill: '#e53e3e', text: 'No triangle has these side lengths.' }, svg);
      }
      const worst = Math.max(c - a - b, b - a - c, a - b - c);
      out.textContent = ok ? 'Consistent: ' + a.toFixed(1) + ' + ' + b.toFixed(1) + ' ≥ ' + c.toFixed(1) + ' (and the other two checks).' :
        'Inconsistent by ' + worst.toFixed(1) + ' Å: one side is longer than the other two combined.\nA pair representation that encodes such distances can never be turned into one 3-D structure. AlphaFold’s triangle updates let every edge (i,j) look at the two other edges of every triangle (i,j,k) — so the network can learn to repair inconsistencies like this one.';
    }
    draw();
  };

  // ---------------------------------------------------------------- the four triangle operations
  W.triangleOps = function (host) {
    AF.head(host, 'The four triangle operations', 'The pair representation is a table of directed edges. To update edge <b>ij</b> (gold), each operation gathers information from the two other edges of every triangle (i, j, k). <b>Click a cell</b> to pick the edge to update, and switch the operation.');
    const n = 6;
    let op = 'out', e = [0, 1];
    const ctr = el('div', { class: 'controls' }, host);
    AF.select(ctr, 'Operation', [{ value: 'out', label: 'Triangle update using outgoing edges' }, { value: 'in', label: 'Triangle update using incoming edges' }, { value: 'start', label: 'Triangle attention around starting node' }, { value: 'end', label: 'Triangle attention around ending node' }], op, function (v) { op = v; draw(); });
    const svg = S('svg', { viewBox: '0 0 640 260' }, host), out = el('div', { class: 'out' }, host);
    function roles(a, b) {
      const i = e[0], j = e[1], R = [];
      for (let k = 0; k < n; k++) {
        if (op === 'out') { if (a === i && b === k) R.push('main'); if (a === j && b === k) R.push('main'); }
        if (op === 'in') { if (a === k && b === i) R.push('main'); if (a === k && b === j) R.push('main'); }
        if (op === 'start') { if (a === i && b === k) R.push('main'); if (a === j && b === k) R.push('bias'); }
        if (op === 'end') { if (a === k && b === j) R.push('main'); if (a === k && b === i) R.push('bias'); }
      }
      if (a === i && b === j) return 'self';
      return R.indexOf('main') >= 0 ? 'main' : R.indexOf('bias') >= 0 ? 'bias' : '';
    }
    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const c = 34, ox = 20, oy = 30, lab = 'ijklmn'.split('');
      S('text', { x: ox, y: 18, 'font-size': 12, 'font-weight': 700, class: 'svg-text', text: 'pair representation z (row = from, column = to)' }, svg);
      for (let a = 0; a < n; a++) {
        S('text', { x: ox - 10, y: oy + a * c + 21, 'font-size': 11, class: 'svg-text', text: a + 1 }, svg);
        S('text', { x: ox + a * c + 14, y: oy + n * c + 14, 'font-size': 11, class: 'svg-text', text: a + 1 }, svg);
        for (let b = 0; b < n; b++) {
          const ro = roles(a, b);
          const rect = S('rect', { x: ox + b * c, y: oy + a * c, width: c - 3, height: c - 3, rx: 4, fill: ro === 'self' ? '#f6c344' : ro === 'main' ? '#5a9bd6' : ro === 'bias' ? '#b8c2cc' : 'var(--bg-soft)', stroke: 'var(--line)', style: 'cursor:pointer' }, svg);
          if (a !== b) rect.addEventListener('click', function () { e = [a, b]; draw(); });
        }
      }
      // graph
      const gx = 450, gy = 130, R = 95, pos = [];
      for (let k = 0; k < n; k++) { const t = -Math.PI / 2 + k * 2 * Math.PI / n; pos.push([gx + R * Math.cos(t), gy + R * Math.sin(t)]); }
      S('defs', null, svg).innerHTML = '<marker id="afar2" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="context-stroke"/></marker>';
      for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
        if (a === b) continue; const ro = roles(a, b); if (!ro) continue;
        const p = pos[a], q = pos[b], dx = q[0] - p[0], dy = q[1] - p[1], L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L;
        S('line', { x1: p[0] + ux * 16 - uy * 3, y1: p[1] + uy * 16 + ux * 3, x2: q[0] - ux * 18 - uy * 3, y2: q[1] - uy * 18 + ux * 3, stroke: ro === 'self' ? '#d69e2e' : ro === 'main' ? '#1f5fa8' : '#9aa5b1', 'stroke-width': ro === 'self' ? 4 : 2.2, 'marker-end': 'url(#afar2)' }, svg);
      }
      pos.forEach(function (p, k) { S('circle', { cx: p[0], cy: p[1], r: 14, fill: 'var(--panel)', stroke: 'var(--ink-soft)', 'stroke-width': 1.5 }, svg); S('text', { x: p[0], y: p[1] + 4, 'text-anchor': 'middle', 'font-size': 12, class: 'svg-text', text: k + 1 }, svg); });
      const i = e[0] + 1, j = e[1] + 1;
      const txt = {
        out: 'Update edge ' + i + '→' + j + ' from, for every k, the edges ' + i + '→k and ' + j + '→k (both “outgoing”).\n  z_ij ← z_ij + g_ij ⊙ Linear( LayerNorm( Σ_k a_ik ⊙ b_jk ) )\nNo attention: a weighted sum, multiplied element-wise. Cheap and symmetric.',
        in: 'Update edge ' + i + '→' + j + ' from, for every k, the edges k→' + i + ' and k→' + j + ' (both “incoming”).\n  z_ij ← z_ij + g_ij ⊙ Linear( LayerNorm( Σ_k a_ki ⊙ b_kj ) )',
        start: 'Edge ' + i + '→' + j + ' attends over all edges ' + i + '→k that share its STARTING node; the third edge ' + j + '→k (grey) enters as a bias.\n  a_ijk = softmax_k( q_ij · k_ik / √c + b_jk ),   o_ij = g_ij ⊙ Σ_k a_ijk v_ik',
        end: 'Edge ' + i + '→' + j + ' attends over all edges k→' + j + ' that share its ENDING node; the third edge k→' + i + ' (grey) enters as a bias.\n  a_ijk = softmax_k( q_ij · k_kj / √c + b_ki ),   o_ij = g_ij ⊙ Σ_k a_ijk v_kj'
      };
      out.textContent = txt[op] + '\n\nGold = edge being updated, blue = edges it reads, grey = bias edges. Formulas: SI Algorithms 11–14.';
    }
    draw();
  };

  // ---------------------------------------------------------------- frames
  W.frameBuilder = function (host) {
    const toy = AF.toy();
    AF.head(host, 'A residue as a rigid body: its frame', 'Pick a residue. Its three backbone atoms N, Cα, C define a <b>frame</b>: an origin (Cα) and three perpendicular axes built by Gram–Schmidt. Now move the whole residue with the sliders: the <b>global</b> coordinates of a side-chain point change, but its <b>local</b> coordinates (in the residue’s own frame) do not.');
    let k = 20, a = 0, b = 0, tx = 0;
    const ctr = el('div', { class: 'controls' }, host);
    AF.slider(ctr, 'residue', 1, toy.L, 1, k + 1, function (x) { k = x - 1; draw(); });
    AF.slider(ctr, 'rotate about z', -180, 180, 5, 0, function (x) { a = x; draw(); }, function (x) { return x + '°'; });
    AF.slider(ctr, 'rotate about x', -180, 180, 5, 0, function (x) { b = x; draw(); }, function (x) { return x + '°'; });
    AF.slider(ctr, 'shift x', -6, 6, 0.5, 0, function (x) { tx = x; draw(); }, function (x) { return x + ' Å'; });
    const g = el('div', { class: 'two-col' }, host), left = el('div', null, g), out = el('div', { class: 'out' }, g);
    const v = P.Viewer(left, { height: 300, width: 400, zoom: 26 });
    const fmt = function (x) { return '(' + x.map(function (u) { return (u >= 0 ? ' ' : '') + u.toFixed(2); }).join(', ') + ')'; };
    function draw() {
      const r = toy.bb[k], cb0 = P.place(r.C, r.N, r.CA, 1.53, 110.5 * P.DEG, -122.6 * P.DEG);
      const G = { R: P.mm(P.rotAxis([0, 0, 1], a * P.DEG), P.rotAxis([1, 0, 0], b * P.DEG)), t: [tx, 0, 0] };
      const mv = function (x) { return P.add(P.mv(G.R, P.sub(x, r.CA)), G.t); };
      const N = mv(r.N), CA = mv(r.CA), C = mv(r.C), CB = mv(cb0);
      const T = P.rigidFrom3(N, CA, C), e = [0, 1, 2].map(function (c) { return [T.R[0][c], T.R[1][c], T.R[2][c]]; });
      const local = P.invApply(T, CB);
      v.set([{ pts: [N, CA, C], color: function (i) { return ['#3182ce', '#718096', '#2d3748'][i]; }, mode: 'tube+points', width: 4, r: 6,
        segments: [[CA, P.add(CA, P.scale(e[0], 2.2)), '#e53e3e', 3], [CA, P.add(CA, P.scale(e[1], 2.2)), '#2f855a', 3], [CA, P.add(CA, P.scale(e[2], 2.2)), '#1f5fa8', 3], [CA, CB, '#d69e2e', 3]],
        labels: [[N, 'N'], [CA, 'Cα'], [C, 'C'], [CB, 'side chain', '#d69e2e'], [P.add(CA, P.scale(e[0], 2.4)), 'e1', '#e53e3e'], [P.add(CA, P.scale(e[1], 2.4)), 'e2', '#2f855a'], [P.add(CA, P.scale(e[2], 2.4)), 'e3', '#1f5fa8']] }]);
      out.textContent = 'Residue ' + (k + 1) + ' (' + toy.seq[k] + ')' +
        '\nGram–Schmidt (SI Alg. 21):\n  v1 = C − Cα,  v2 = N − Cα\n  e1 = v1/‖v1‖\n  e2 = normalise(v2 − (e1·v2) e1)\n  e3 = e1 × e2\n  R = [e1 e2 e3],  t = Cα' +
        '\n\nR = ' + T.R.map(function (row) { return fmt(row); }).join('\n    ') +
        '\ndet R = ' + P.det3(T.R).toFixed(3) + '  (a proper rotation)' +
        '\n\nside-chain point, global x = ' + fmt(CB) +
        '\nside-chain point, local  T⁻¹∘x = Rᵀ(x − t) = ' + fmt(local) + '   ← never changes';
    }
    draw();
  };

  // ---------------------------------------------------------------- invariant point attention
  W.ipaInvariance = function (host) {
    const toy = AF.toy();
    AF.head(host, 'Invariant point attention: rotate the world, nothing changes', 'Residue i (red) attends to every residue j. Each residue produces a few 3-D <b>query/key points</b> in its <i>own</i> frame; the attention logit penalises the squared distance between i’s query points and j’s key points once both are placed in the global frame. Rotate and shift the whole protein: the IPA weights stay put. A naive layer that reads raw global coordinates changes.');
    let qi = 8, ang = 0, sh = 0;
    const ctr = el('div', { class: 'controls' }, host);
    AF.slider(ctr, 'query residue i', 1, toy.L, 1, qi + 1, function (x) { qi = x - 1; draw(); });
    AF.slider(ctr, 'rotate whole protein', 0, 360, 5, 0, function (x) { ang = x; draw(); }, function (x) { return x + '°'; });
    AF.slider(ctr, 'shift whole protein', -20, 20, 1, 0, function (x) { sh = x; draw(); }, function (x) { return x + ' Å'; });
    const g = el('div', { class: 'two-col' }, host), left = el('div', null, g), right = el('div', null, g);
    const v = P.Viewer(left, { height: 300, width: 400 }), plot = el('div', null, right), out = el('div', { class: 'out' }, host);
    const r = P.rng(17), NQ = 4;
    const qPts = [], kPts = [], qVec = [P.gauss(r), P.gauss(r), P.gauss(r)];
    for (let p = 0; p < NQ; p++) qPts.push([2 * P.gauss(r), 2 * P.gauss(r), 2 * P.gauss(r)]);
    for (let p = 0; p < NQ; p++) kPts.push([2 * P.gauss(r), 2 * P.gauss(r), 2 * P.gauss(r)]);
    const gamma = 0.05, wC = Math.sqrt(2 / (9 * NQ));
    function weights(frames, X) {
      const Ti = frames[qi], gq = qPts.map(function (q) { return P.apply(Ti, q); });
      const ipa = frames.map(function (Tj) { let s2 = 0; kPts.forEach(function (kp, p) { const d = P.sub(gq[p], P.apply(Tj, kp)); s2 += P.dot(d, d); }); return -gamma * wC / 2 * s2; });
      const naive = X.map(function (x) { return 0.25 * P.dot(qVec, x); });
      return { ipa: AF.softmax(ipa), naive: AF.softmax(naive), gq: gq };
    }
    const ref = weights(toy.frames, toy.ca);
    v.setZoomFrom(toy.ca.map(function (x) { return P.add(x, [0, 0, 0]); }));
    function draw() {
      const G = { R: P.rotAxis([0.3, 1, 0.2], ang * P.DEG), t: [sh, 0, 0] };
      const frames = toy.frames.map(function (T) { return P.compose(G, T); }), X = toy.ca.map(function (x) { return P.apply(G, x); });
      const w = weights(frames, X);
      const best = w.ipa.indexOf(Math.max.apply(null, w.ipa));
      v.set([{ pts: X, color: function (j) { return j === qi ? '#e53e3e' : AF.heat(0.15 + 0.85 * Math.min(1, w.ipa[j] / Math.max.apply(null, w.ipa))); }, mode: 'tube+points', width: 4, r: 3.5 },
        { pts: w.gq, color: '#e53e3e', mode: 'points', r: 2.5, segments: w.gq.map(function (q) { return [X[qi], q, '#e53e3e', 1, 0.6, '2 2']; }) }], true);
      const series = function (a, col) { return { color: col, points: a.map(function (x, j) { return [j + 1, x * 100]; }), marker: 'none', width: 2 }; };
      AF.chart.lines(plot, [series(ref.ipa, 'rgba(31,95,168,.35)'), series(w.ipa, '#1f5fa8'), series(ref.naive, 'rgba(229,62,62,.35)'), series(w.naive, '#e53e3e')], { xmin: 1, xmax: toy.L, ymin: 0, xticks: [1, 10, 20, 30, 40, 50], xlabel: 'key residue j', ylabel: 'attention weight (%)', height: 230 });
      const dI = Math.max.apply(null, w.ipa.map(function (x, j) { return Math.abs(x - ref.ipa[j]); })), dN = Math.max.apply(null, w.naive.map(function (x, j) { return Math.abs(x - ref.naive[j]); }));
      out.textContent = 'IPA point term (SI Alg. 22):  − (γ w_C / 2) Σ_p ‖ T_i ∘ q_i^p − T_j ∘ k_j^p ‖²     (blue curve)\nnaive term:  q · x_j  using raw global coordinates      (red curve)\nFaint curves = before moving the protein.\n\nlargest change in IPA weights: ' + (dI * 100).toFixed(4) + ' %     largest change in naive weights: ' + (dN * 100).toFixed(2) + ' %' +
        '\nWhy IPA is invariant: moving everything by G turns T_i into G∘T_i, and ‖G∘a − G∘b‖ = ‖a − b‖. Residue ' + (best + 1) + ' gets the most attention from residue ' + (qi + 1) + '.';
    }
    draw();
  };

  // ---------------------------------------------------------------- the residue gas / structure module
  W.residueGas = function (host) {
    const toy = AF.toy();
    AF.head(host, 'The structure module: a gas of residues finds its shape', 'All residue frames start at the origin with the identity rotation. Every block updates <b>every</b> frame at once. The chain is not enforced — watch the Cα–Cα “bonds” (red when stretched or squashed by more than 0.5 Å) break during refinement and heal at the end. <i>Illustrative trajectory, not a trained network.</i>');
    let blk = 3, play = null;
    const ctr = el('div', { class: 'controls' }, host);
    const sl = AF.slider(ctr, 'block', 0, 8, 1, 3, function (x) { blk = x; draw(); });
    AF.button(ctr, '▶ Play', function () { if (play) { clearInterval(play); play = null; return; } blk = 0; play = setInterval(function () { blk++; sl.value = blk; sl.dispatchEvent(new Event('input')); if (blk >= 8) { clearInterval(play); play = null; } }, 800); }, 'primary');
    const showTri = el('label', null, ctr), st = el('input', { type: 'checkbox', checked: 'checked' }, showTri); showTri.appendChild(document.createTextNode(' show residue triangles'));
    st.addEventListener('change', draw);
    const v = P.Viewer(host, { height: 360 }), out = el('div', { class: 'out' }, host);
    const r = P.rng(31), speed = toy.bb.map(function () { return 0.7 + 0.6 * r(); }), wob = toy.bb.map(function () { return [P.gauss(r), P.gauss(r), P.gauss(r)]; });
    v.setZoomFrom(toy.ca);
    function state(b) {
      const t = b / 8;
      return toy.bb.map(function (res, i) {
        const ti = AF.clamp(t * speed[i] + (t > 0.85 ? (t - 0.85) / 0.15 * (1 - t * speed[i]) : 0), 0, 1);
        const e = 1 - Math.pow(1 - ti, 2), jit = P.scale(wob[i], 2.2 * Math.sin(Math.PI * t) * (b < 8 ? 1 : 0));
        const ca = P.add(P.scale(res.CA, e), jit);
        const nOff = P.lerp([-1.0, 1.0, 0], P.sub(res.N, res.CA), e), cOff = P.lerp([1.5, 0, 0], P.sub(res.C, res.CA), e);
        return { N: P.add(ca, nOff), CA: ca, C: P.add(ca, cOff) };
      });
    }
    function draw() {
      const bb = state(blk), X = bb.map(function (x) { return x.CA; });
      let bad = 0; const segs = [];
      for (let i = 0; i < X.length - 1; i++) { const d = P.dist(X[i], X[i + 1]); const b2 = Math.abs(d - 3.8) > 0.5; if (b2) bad++; segs.push([X[i], X[i + 1], b2 ? '#e53e3e' : 'var(--ink-soft)', b2 ? 1.5 : 2.5, 1, b2 ? '3 3' : '']); }
      v.set([{ pts: X, mode: 'points', r: 2, color: 'var(--muted)', segments: segs, tris: st.checked ? bb.map(function (b, i) { return [b.N, b.CA, b.C, AF.rainbow(i / (bb.length - 1)), 0.9]; }) : [] }], true);
      const ref = P.rmsd(X, toy.ca);
      out.textContent = 'block ' + blk + ' / 8   ·   Cα RMSD to final = ' + ref.toFixed(2) + ' Å   ·   chain “bonds” out of range: ' + bad + ' / ' + (X.length - 1) +
        '\nEach block: IPA updates the single representation → a small network predicts, per residue, a rotation (as a quaternion (1, b, c, d)) and a translation → T_i ← T_i ∘ (R_i, t_i).' +
        '\nThe 8 blocks share weights. Peptide-bond geometry is only encouraged (by a violation loss during fine-tuning) and finally enforced by an Amber relaxation.';
    }
    draw();
  };

  // ---------------------------------------------------------------- FAPE
  W.fapeLab = function (host) {
    const toy = AF.toy();
    AF.head(host, 'FAPE: compare structures from every residue’s point of view', 'For each residue k, express all atoms in k’s local frame — once for the prediction and once for the truth — and measure how far apart they are. Average over all frames and atoms, clamping each error at 10 Å. Try the <b>mirror image</b>: distance-based scores call it perfect; FAPE does not. The 3-D view shows both structures as seen from frame k (they coincide exactly at the frame).');
    let noise = 0.5, hinge = 0, mirror = false, k = 20, clamp = 10;
    const ctr = el('div', { class: 'controls' }, host);
    AF.slider(ctr, 'random error (Å)', 0, 4, 0.1, noise, function (x) { noise = x; draw(); }, function (x) { return x.toFixed(1); });
    AF.slider(ctr, 'hinge swing', 0, 90, 5, hinge, function (x) { hinge = x; draw(); }, function (x) { return x + '°'; });
    const mi = el('label', null, ctr), mc = el('input', { type: 'checkbox' }, mi); mi.appendChild(document.createTextNode(' mirror image')); mc.addEventListener('change', function () { mirror = mc.checked; draw(); });
    AF.slider(ctr, 'view from frame k', 1, toy.L, 1, k + 1, function (x) { k = x - 1; draw(); });
    AF.slider(ctr, 'clamp', 2, 30, 1, clamp, function (x) { clamp = x; draw(); }, function (x) { return x + ' Å'; });
    const stats = el('div', null, host);
    const g = el('div', { class: 'two-col' }, host), left = el('div', null, g), right = el('div', null, g);
    const v = P.Viewer(left, { height: 300, width: 400 }), plot = el('div', null, right), out = el('div', { class: 'note' }, host);
    v.setZoomFrom(toy.ca);
    function draw() {
      let bb = AF.noisyBB(toy.bb, noise / Math.sqrt(3), 8);
      bb = AF.hingeBB(bb, hinge);
      if (mirror) bb = AF.mirrorBB(bb);
      const pF = P.framesOf(bb), pX = bb.map(function (b) { return b.CA; });
      const f = P.fape(pF, pX, toy.frames, toy.ca, { clamp: clamp });
      const perFrame = f.matrix.map(function (row) { return row.reduce(function (s, d) { return s + Math.min(clamp, d); }, 0) / row.length; });
      const m = { 'FAPE (Å, clamped)': f.meanClamped.toFixed(2), 'FAPE loss (÷ Z = 10 Å)': f.loss.toFixed(3), 'Cα RMSD after superposition (Å)': P.rmsd(pX, toy.ca).toFixed(2), 'distance-matrix RMSD (Å)': P.drmsd(pX, toy.ca).toFixed(2) };
      stats.innerHTML = Object.keys(m).map(function (q) { return '<div class="stat"><b>' + m[q] + '</b><span>' + q + '</span></div>'; }).join('');
      const lp = pX.map(function (x) { return P.invApply(pF[k], x); }), lt = toy.ca.map(function (x) { return P.invApply(toy.frames[k], x); });
      v.set([{ pts: lt, color: 'var(--muted)', width: 7, opacity: 0.45 }, { pts: lp, color: function (i) { return i === k ? '#e53e3e' : AF.rainbow(i / (toy.L - 1)); }, width: 4 }], false);
      AF.chart.bars(plot, perFrame.map(function (x, i) { return { label: (i % 10 === 0 ? String(i + 1) : ''), value: x, note: '', color: i === k ? '#e53e3e' : 'var(--accent)' }; }), { height: 220, ml: 40, ylabel: 'mean error in frame k (Å)', max: Math.max(1, clamp) });
      out.innerHTML = 'FAPE (SI Alg. 28): x<sub>ij</sub> = T<sub>i</sub><sup>−1</sup>∘x<sub>j</sub>, d<sub>ij</sub> = √(‖x<sub>ij</sub> − x<sub>ij</sub><sup>true</sup>‖² + ε), L = (1/Z)·mean<sub>ij</sub> min(d<sub>clamp</sub>, d<sub>ij</sub>). The bar chart shows the error seen from each frame. Here only Cα atoms are scored; AlphaFold scores all heavy atoms in all backbone and side-chain frames.' + (mirror ? ' <b>Mirror:</b> distances are identical (distance-matrix RMSD ≈ 0) yet FAPE is large — FAPE sees chirality.' : '');
    }
    draw();
  };
})();
