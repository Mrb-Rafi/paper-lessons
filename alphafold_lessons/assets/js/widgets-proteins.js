/* Part I widgets: proteins, folding, the structure gap, geometry, backbone angles. */
(function () {
  'use strict';
  const AF = window.AF, W = AF.widgets, el = AF.el, S = AF.svg, P = window.Prot;

  // shared toy protein + cached MSAs
  AF.toy = function () { return P.toy(); };
  AF.msaCache = {};
  AF.toyMSA = function (n, seed) {
    const k = n + ':' + (seed || 11);
    if (!AF.msaCache[k]) AF.msaCache[k] = AF.toy().msa(n, seed || 11);
    return AF.msaCache[k];
  };
  AF.ssColor = function (c) { return c === 'H' ? '#c0392b' : c === 'E' ? '#d69e2e' : '#718096'; };
  AF.seqChips = function (parent, seq, colorFn, opt) {
    opt = opt || {};
    const d = el('div', { class: 'seq' }, parent);
    seq.split('').forEach(function (a, i) {
      const s = el('span', { text: a, title: (i + 1) + ' ' + (P.AA_INFO[a] ? P.AA_INFO[a][1] : a) }, d);
      const c = colorFn ? colorFn(a, i) : P.aaColor(a);
      s.style.background = c; s.style.color = '#fff'; s.style.marginRight = '1px';
      if (opt.onclick) { s.style.cursor = 'pointer'; s.addEventListener('click', function () { opt.onclick(i); }); }
    });
    return d;
  };
  // interpolate a set of rigid frames between two states (positions linear, rotations by re-orthonormalising)
  AF.lerpBackbone = function (A, B, t) {
    return A.map(function (a, i) { return { N: P.lerp(a.N, B[i].N, t), CA: P.lerp(a.CA, B[i].CA, t), C: P.lerp(a.C, B[i].C, t) }; });
  };
  AF.triOf = function (r, col, op) { return [r.N, r.CA, r.C, col, op]; };

  // ---------------------------------------------------------------- hero: the residue gas assembles
  W.heroDemo = function (host) {
    const toy = AF.toy();
    const top = el('div', { class: 'two-col' }, host);
    const left = el('div', null, top), right = el('div', null, top);
    el('div', { class: 'small', text: 'Input: the amino-acid sequence (one letter per residue)' }, left);
    AF.seqChips(left, toy.seq);
    const stage = el('div', { class: 'out', style: 'font-family:var(--font-ui);margin-top:12px' }, left);
    const v = P.Viewer(right, { height: 330, width: 520, spin: true });
    // start: every residue frame at the origin with identity rotation (the paper's initial "black hole")
    const r = P.rng(4);
    const start = toy.bb.map(function (b) {
      const f = toy.frames[0];
      return { N: P.sub(b.N, b.CA), CA: [0, 0, 0], C: P.sub(b.C, b.CA) };
    });
    const mid = toy.bb.map(function (b) { const o = [3 * P.gauss(r), 3 * P.gauss(r), 3 * P.gauss(r)]; return { N: P.add(b.N, o), CA: P.add(b.CA, o), C: P.add(b.C, o) }; });
    v.setZoomFrom(toy.ca);
    let k = 0, timer = null;
    const total = 9;
    function frameAt(k) {
      if (k === 0) return start;
      const t = k / (total - 1);
      if (t < 0.5) return AF.lerpBackbone(start, mid, t * 2);
      return AF.lerpBackbone(mid, toy.bb, (t - 0.5) * 2);
    }
    function draw() {
      const bb = frameAt(k), done = k === total - 1;
      const tris = bb.map(function (b, i) { return AF.triOf(b, AF.rainbow(i / (bb.length - 1)), 0.9); });
      const layers = [{ pts: bb.map(function (b) { return b.CA; }), tris: tris, mode: done ? 'tube' : 'none', color: function (i) { return AF.rainbow(i / (bb.length - 1)); }, width: 5 }];
      v.set(layers, true);
      stage.innerHTML = k === 0 ? '<b>Block 0.</b> All residues start as free-floating triangles stacked at the origin — the “residue gas”.' :
        done ? '<b>Done.</b> Each residue’s triangle has found its place and orientation; the chain is drawn through the Cα atoms.' :
          '<b>Structure-module block ' + k + ' of 8.</b> Every residue moves and turns <i>at the same time</i>, guided by what the network has learned.';
    }
    function loop() { draw(); k = (k + 1) % total; timer = setTimeout(loop, k === 0 ? 2600 : 900); }
    loop();
    const c = el('div', { class: 'note', html: 'An illustration of AlphaFold’s final stage on this book’s 50-residue toy protein: each triangle is one residue’s backbone (N, Cα, C atoms). It is an animation, not a trained network — but every idea in it is built for real later in the book.' }, host);
    c.style.marginTop = '10px';
  };

  // ---------------------------------------------------------------- the 20 amino acids
  W.aminoAcids = function (host) {
    AF.head(host, 'The 20 amino acids', 'Click a letter to learn about that amino acid. Colours show its chemical character. Then type or paste any protein sequence below to colour it and count its composition.');
    const grid = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fill,minmax(54px,1fr));gap:6px' }, host);
    const info = el('div', { class: 'out' }, host);
    P.AA.split('').forEach(function (a) {
      const b = el('button', { text: a, title: P.AA_INFO[a][1], style: 'font-family:var(--font-mono);font-size:17px;color:#fff;border:none;background:' + P.aaColor(a) }, grid);
      b.addEventListener('click', function () { show(a); });
    });
    const legend = el('div', { class: 'legend' }, host);
    Object.keys(P.CLASS_COLOR).forEach(function (k) { legend.innerHTML += '<span><i style="background:' + P.CLASS_COLOR[k] + '"></i>' + k + '</span>'; });
    const notes = {
      hydrophobic: 'Oily: avoids water, so it tends to be buried in the protein’s core. The hydrophobic effect is the main force that makes proteins fold.',
      positive: 'Carries a positive charge at body pH. Likes the surface (water) and negatively charged partners.',
      negative: 'Carries a negative charge at body pH. Likes the surface and positively charged partners (a “salt bridge”).',
      polar: 'Uncharged but water-loving: can form hydrogen bonds. Usually on the surface.',
      special: 'Unusual: glycine has no side chain (very flexible), proline’s ring locks its backbone angle (a helix breaker), cysteines can link through disulfide bonds.'
    };
    function show(a) {
      const i = P.AA_INFO[a];
      info.textContent = a + ' = ' + i[1] + ' (' + i[0] + ') · ' + i[2] + '\n' + notes[i[2]] + '\nSide-chain torsion angles (χ) AlphaFold predicts for it: ' + P.CHI[a] + '.';
    }
    show('L');
    const ta = el('textarea', null, host); ta.value = AF.toy().seq;
    const out = el('div', null, host);
    function draw() {
      const s = ta.value.toUpperCase().replace(/[^A-Z]/g, '').split('').filter(function (c) { return P.AA_INFO[c]; }).join('');
      out.innerHTML = '';
      AF.seqChips(out, s);
      const cnt = {}; Object.keys(P.CLASS_COLOR).forEach(function (k) { cnt[k] = 0; });
      s.split('').forEach(function (c) { cnt[P.aaClass(c)]++; });
      const bar = el('div', { style: 'display:flex;height:16px;border-radius:4px;overflow:hidden;margin-top:8px' }, out);
      Object.keys(cnt).forEach(function (k) { if (cnt[k]) el('div', { title: k + ' ' + cnt[k], style: 'background:' + P.CLASS_COLOR[k] + ';width:' + (100 * cnt[k] / s.length) + '%' }, bar); });
      el('div', { class: 'note', text: s.length + ' residues. The default text is the toy protein used throughout this book.' }, out);
    }
    ta.addEventListener('input', draw); draw();
  };

  // ---------------------------------------------------------------- Levinthal's paradox
  W.levinthal = function (host) {
    AF.head(host, 'Levinthal’s paradox: why folding cannot be a random search', 'Suppose each residue could take just a few backbone shapes. Count every combination, and imagine trying them at a fantastically fast rate. Real proteins fold in milliseconds to seconds.');
    const ctr = el('div', { class: 'controls' }, host);
    let n = 100, k = 3, rate = 13;
    AF.slider(ctr, 'residues', 10, 400, 10, n, function (x) { n = x; draw(); });
    AF.slider(ctr, 'shapes per residue', 2, 10, 1, k, function (x) { k = x; draw(); });
    AF.slider(ctr, 'tries per second 10^', 6, 15, 1, rate, function (x) { rate = x; draw(); });
    const out = el('div', { class: 'out' }, host);
    function draw() {
      const log10 = n * Math.log10(k), secs = log10 - rate, univ = Math.log10(4.35e17);
      out.textContent = 'possible shapes = ' + k + '^' + n + ' ≈ 10^' + log10.toFixed(0) +
        '\ntime to try them all ≈ 10^' + secs.toFixed(0) + ' seconds' +
        '\nage of the universe ≈ 10^' + univ.toFixed(1) + ' seconds  →  ' + (secs > univ ? 'about 10^' + (secs - univ).toFixed(0) + ' times the age of the universe' : 'feasible!') +
        '\n\nConclusion: proteins do not search blindly. The energy landscape funnels them toward the folded state — and the folded state is encoded in the sequence.';
    }
    draw();
  };

  // ---------------------------------------------------------------- HP lattice protein
  W.hpFold = function (host) {
    AF.head(host, 'Fold a toy protein on a grid (the HP model)', 'A classic physics toy: residues are <b style="color:#d69e2e">H</b> (hydrophobic) or <b style="color:#38a169">P</b> (polar), the chain lives on a square grid and cannot cross itself. Each non-bonded H–H contact lowers the energy by 1. Build a fold with the arrow buttons, then let the computer search <i>every</i> possible fold.');
    const seq = 'HPHPPHHPHPPHPH';
    let path = [[0, 0]];
    const ctr = el('div', { class: 'controls' }, host);
    const dirs = { '↑': [0, 1], '↓': [0, -1], '←': [-1, 0], '→': [1, 0] };
    Object.keys(dirs).forEach(function (d) { AF.button(ctr, d, function () { move(dirs[d]); }); });
    AF.button(ctr, 'Undo', function () { if (path.length > 1) path.pop(); draw(); });
    AF.button(ctr, 'Reset', function () { path = [[0, 0]]; draw(); });
    AF.button(ctr, 'Random fold', function () { randomFold(); draw(); });
    const sb = AF.button(ctr, 'Search all folds', function () { search(); }, 'primary');
    const g = el('div', { class: 'two-col' }, host), pic = el('div', null, g), out = el('div', { class: 'out' }, g);
    let best = null, msg = '';
    function key(p) { return p[0] + ',' + p[1]; }
    function move(d) {
      if (path.length >= seq.length) return;
      const last = path[path.length - 1], nx = [last[0] + d[0], last[1] + d[1]];
      if (path.some(function (p) { return p[0] === nx[0] && p[1] === nx[1]; })) { msg = 'That square is taken — the chain cannot cross itself.'; draw(); return; }
      path.push(nx); msg = ''; draw();
    }
    function energy(pth) {
      const at = {}; pth.forEach(function (p, i) { at[key(p)] = i; }); let e = 0, pairs = [];
      pth.forEach(function (p, i) {
        if (seq[i] !== 'H') return;
        [[1, 0], [0, 1]].forEach(function (d) { const j = at[(p[0] + d[0]) + ',' + (p[1] + d[1])]; if (j != null && Math.abs(j - i) > 1 && seq[j] === 'H') { e--; pairs.push([i, j]); } });
      });
      return { e: e, pairs: pairs };
    }
    function randomFold() {
      for (let tries = 0; tries < 500; tries++) {
        const p = [[0, 0]], used = { '0,0': 1 }; let ok = true;
        for (let i = 1; i < seq.length; i++) {
          const last = p[i - 1], opts = Object.values(dirs).map(function (d) { return [last[0] + d[0], last[1] + d[1]]; }).filter(function (q) { return !used[key(q)]; });
          if (!opts.length) { ok = false; break; }
          const q = opts[Math.floor(Math.random() * opts.length)]; p.push(q); used[key(q)] = 1;
        }
        if (ok) { path = p; msg = 'A random self-avoiding fold.'; return; }
      }
    }
    function search() {
      sb.disabled = true; out.textContent = 'Searching…';
      setTimeout(function () {
        const t0 = performance.now(); let count = 0, bestE = 1, bestP = null;
        const p = [[0, 0], [1, 0]], used = { '0,0': 1, '1,0': 1 };
        const D = [[1, 0], [0, 1], [-1, 0], [0, -1]];
        (function rec(firstTurn) {
          if (p.length === seq.length) { count++; const e = energy(p).e; if (e < bestE) { bestE = e; bestP = p.map(function (q) { return q.slice(); }); } return; }
          const last = p[p.length - 1];
          for (let d = 0; d < 4; d++) {
            if (!firstTurn && D[d][1] < 0) continue; // symmetry: first turn goes up
            const q = [last[0] + D[d][0], last[1] + D[d][1]], k = key(q);
            if (used[k]) continue;
            used[k] = 1; p.push(q); rec(firstTurn || D[d][1] !== 0); p.pop(); delete used[k];
          }
        })(false);
        const ms = performance.now() - t0;
        best = bestP; path = bestP; sb.disabled = false;
        msg = 'Exhaustive search: ' + count.toLocaleString() + ' distinct folds (up to symmetry) in ' + ms.toFixed(0) + ' ms.\nBest energy = ' + bestE + '. Feasible for 14 residues; hopeless for 300 (see the paradox above).';
        draw();
      }, 30);
    }
    function draw() {
      pic.innerHTML = '';
      const xs = path.map(function (p) { return p[0]; }), ys = path.map(function (p) { return p[1]; });
      const minx = Math.min.apply(null, xs) - 1, maxx = Math.max.apply(null, xs) + 1, miny = Math.min.apply(null, ys) - 1, maxy = Math.max.apply(null, ys) + 1;
      const span = Math.max(maxx - minx, maxy - miny, 6), sc = 300 / span;
      const svg = S('svg', { viewBox: '0 0 320 320' }, pic);
      const X = function (x) { return 10 + (x - minx + (span - (maxx - minx)) / 2) * sc; }, Y = function (y) { return 310 - (y - miny + (span - (maxy - miny)) / 2) * sc; };
      const en = energy(path);
      en.pairs.forEach(function (pr) { S('line', { x1: X(path[pr[0]][0]), y1: Y(path[pr[0]][1]), x2: X(path[pr[1]][0]), y2: Y(path[pr[1]][1]), stroke: '#d69e2e', 'stroke-width': 3, 'stroke-dasharray': '4 3' }, svg); });
      for (let i = 0; i < path.length - 1; i++) S('line', { x1: X(path[i][0]), y1: Y(path[i][1]), x2: X(path[i + 1][0]), y2: Y(path[i + 1][1]), stroke: 'var(--ink-soft)', 'stroke-width': 3 }, svg);
      path.forEach(function (p, i) {
        S('circle', { cx: X(p[0]), cy: Y(p[1]), r: Math.min(14, sc * 0.32), fill: seq[i] === 'H' ? '#d69e2e' : '#38a169', stroke: i === 0 ? 'var(--ink)' : 'var(--panel)', 'stroke-width': 2 }, svg);
        S('text', { x: X(p[0]), y: Y(p[1]) + 4, 'text-anchor': 'middle', 'font-size': 10, fill: '#fff', text: i + 1 }, svg);
      });
      out.textContent = 'Sequence: ' + seq + '\nPlaced ' + path.length + ' / ' + seq.length + ' residues' + '\nEnergy = ' + en.e + '  (number of H–H contacts, dashed, times −1)' + (msg ? '\n\n' + msg : '');
    }
    draw();
  };

  // ---------------------------------------------------------------- the structure gap
  W.gapChart = function (host) {
    AF.head(host, 'The gap: sequences versus structures', 'Numbers quoted in the paper, on a <b>logarithmic</b> scale (each grid line is 10× the previous one). Hover a bar for its source.');
    const data = [
      { label: 'Unique protein structures solved experimentally', v: 1e5, note: '≈ 100,000', src: 'Paper, first paragraph (“around 100,000 unique proteins”)' , c: 'var(--green)' },
      { label: 'Protein families in BFD (as alignments)', v: 65983866, note: '65,983,866', src: 'Methods: BFD', c: 'var(--teal)' },
      { label: 'Sequence clusters in BFD at 30% identity', v: 345159030, note: '345,159,030', src: 'Methods: BFD construction', c: 'var(--teal)' },
      { label: 'Protein sequences covered by BFD', v: 2204359010, note: '2,204,359,010', src: 'Methods: BFD', c: 'var(--accent)' }
    ];
    const svg = S('svg', { viewBox: '0 0 660 230' }, host);
    const x = function (v) { return 230 + (Math.log10(v) - 4) / (10 - 4) * 410; };
    for (let e = 4; e <= 10; e++) { S('line', { x1: x(Math.pow(10, e)), x2: x(Math.pow(10, e)), y1: 10, y2: 195, stroke: 'var(--line)' }, svg); S('text', { x: x(Math.pow(10, e)), y: 212, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: '10^' + e }, svg); }
    data.forEach(function (d, i) {
      const y = 20 + i * 44;
      const g = S('g', null, svg);
      S('title', { text: d.src }, g);
      S('rect', { x: x(1e4), y: y, width: x(d.v) - x(1e4), height: 26, rx: 4, fill: d.c }, g);
      S('text', { x: 222, y: y + 17, 'text-anchor': 'end', 'font-size': 11.5, class: 'svg-text', text: d.label.length > 34 ? d.label.slice(0, 33) + '…' : d.label }, g);
      S('text', { x: x(d.v) + 6, y: y + 17, 'font-size': 12, 'font-weight': 700, class: 'svg-text', text: d.note }, g);
    });
    el('div', { class: 'note', html: 'Roughly <b>20,000 sequences for every solved structure</b>. Solving one structure experimentally takes “months to years of painstaking effort”; AlphaFold takes GPU minutes to hours (Chapter 24).' }, host);
  };

  // ---------------------------------------------------------------- 3-D viewer + distance map
  W.distMap = function (host) {
    const toy = AF.toy();
    AF.head(host, 'A protein in 3-D and its distance map', 'Left: the toy protein’s Cα atoms (one per residue). Right: the <b>distance matrix</b> — cell (i, j) is the distance between residues i and j, dark = close. <b>Click any cell</b> to highlight that pair in 3-D. Lower the threshold to turn distances into a <b>contact map</b>.');
    const ctr = el('div', { class: 'controls' }, host);
    let mode = 'dist', cut = 8, colorBy = 'rainbow', sel = [5, 25];
    AF.select(ctr, 'Show', [{ value: 'dist', label: 'distances' }, { value: 'contact', label: 'contacts (below threshold)' }], mode, function (v) { mode = v; draw(); });
    AF.slider(ctr, 'threshold (Å)', 4, 16, 0.5, cut, function (x) { cut = x; draw(); }, function (x) { return x.toFixed(1); });
    AF.select(ctr, 'Colour 3-D by', [{ value: 'rainbow', label: 'position N→C' }, { value: 'chem', label: 'chemistry' }, { value: 'ss', label: 'secondary structure' }], colorBy, function (v) { colorBy = v; draw(); });
    const g = el('div', { class: 'two-col' }, host), left = el('div', null, g), right = el('div', null, g);
    const v = P.Viewer(left, { height: 360, width: 420 });
    const cv = el('canvas', { width: 400, height: 400, class: 'heat', style: 'width:100%;max-width:400px;cursor:crosshair' }, right);
    const out = el('div', { class: 'out' }, host);
    const D = P.distMatrix(toy.ca), L = toy.L;
    cv.addEventListener('click', function (e) { const r = cv.getBoundingClientRect(); const i = Math.floor((e.clientY - r.top) / r.height * L), j = Math.floor((e.clientX - r.left) / r.width * L); sel = [AF.clamp(i, 0, L - 1), AF.clamp(j, 0, L - 1)]; draw(); });
    function col(i) { return colorBy === 'rainbow' ? AF.rainbow(i / (L - 1)) : colorBy === 'chem' ? P.aaColor(toy.seq[i]) : AF.ssColor(toy.ss[i]); }
    function draw() {
      const ctx = cv.getContext('2d'), c = 400 / L;
      for (let i = 0; i < L; i++) for (let j = 0; j < L; j++) {
        const d = D[i][j];
        ctx.fillStyle = mode === 'dist' ? AF.heat(1 - Math.min(d, 30) / 30) : (d < cut ? '#1f5fa8' : '#f1f4f8');
        ctx.fillRect(j * c, i * c, c + 0.5, c + 0.5);
      }
      ctx.strokeStyle = '#e53e3e'; ctx.lineWidth = 2; ctx.strokeRect(sel[1] * c, sel[0] * c, c, c); ctx.strokeRect(sel[0] * c, sel[1] * c, c, c);
      const a = toy.ca[sel[0]], b = toy.ca[sel[1]];
      v.set([{ pts: toy.ca, color: col, mode: 'tube+points', width: 5, r: 3.4, segments: [[a, b, '#e53e3e', 2.5, 1, '5 4']], labels: [[a, String(sel[0] + 1), '#e53e3e'], [b, String(sel[1] + 1), '#e53e3e']] }], true);
      const nC = P.contacts(toy.ca, cut, 1).length;
      out.textContent = 'residues ' + (sel[0] + 1) + ' (' + toy.seq[sel[0]] + ') and ' + (sel[1] + 1) + ' (' + toy.seq[sel[1]] + '): distance = ' + D[sel[0]][sel[1]].toFixed(1) + ' Å' + (D[sel[0]][sel[1]] < cut ? '  → in contact' : '') +
        '\nThe matrix is symmetric (d_ij = d_ji) and zero on the diagonal. ' + nC + ' pairs are closer than ' + cut.toFixed(1) + ' Å.' +
        '\nThe stripes next to the diagonal are the three helices; the off-diagonal blocks are helices packing against each other.';
    }
    v.setZoomFrom(toy.ca);
    draw();
  };

  // ---------------------------------------------------------------- Ramachandran + NeRF backbone builder
  W.rama = function (host) {
    AF.head(host, 'Backbone angles build the shape (Ramachandran plot)', 'Each residue’s backbone has two adjustable torsion angles, <b>φ</b> (phi) and <b>ψ</b> (psi). <b>Click anywhere on the plot</b> to give all 16 residues of a peptide those angles; the 3-D chain is built atom by atom with ideal bond lengths and angles (the “NeRF” method). Shaded regions are where real proteins’ residues mostly fall.');
    const g = el('div', { class: 'two-col' }, host), left = el('div', null, g), right = el('div', null, g);
    const ctr = el('div', { class: 'controls' }, left);
    let phi = -57, psi = -47;
    const presets = { 'α-helix': [-57, -47], 'β-strand': [-120, 130], 'polyproline II': [-75, 145], 'left-handed helix': [57, 47] };
    Object.keys(presets).forEach(function (k) { AF.button(ctr, k, function () { phi = presets[k][0]; psi = presets[k][1]; draw(); }); });
    const svg = S('svg', { viewBox: '0 0 300 300', style: 'cursor:crosshair;max-width:340px;background:var(--bg-soft);border-radius:8px' }, left);
    const X = function (a) { return 20 + (a + 180) / 360 * 270; }, Y = function (a) { return 280 - (a + 180) / 360 * 270; };
    [['α', -63, -43, 32, 28], ['β', -120, 135, 50, 35], ['Lα', 60, 45, 18, 18], ['PII', -72, 145, 16, 22]].forEach(function (r) { S('ellipse', { cx: X(r[1]), cy: Y(r[2]), rx: r[3] / 360 * 270 * 2, ry: r[4] / 360 * 270 * 2, fill: 'var(--accent-soft)', stroke: 'var(--accent)', 'stroke-width': 1 }, svg); S('text', { x: X(r[1]), y: Y(r[2]) + 4, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: r[0] }, svg); });
    S('line', { x1: X(0), x2: X(0), y1: 10, y2: 280, stroke: 'var(--line)' }, svg); S('line', { y1: Y(0), y2: Y(0), x1: 20, x2: 290, stroke: 'var(--line)' }, svg);
    S('text', { x: 155, y: 297, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: 'φ (−180° … 180°)' }, svg);
    S('text', { x: 9, y: 150, 'font-size': 11, class: 'svg-text', transform: 'rotate(-90 9 150)', 'text-anchor': 'middle', text: 'ψ' }, svg);
    const dot = S('circle', { r: 6, fill: '#e53e3e', stroke: '#fff', 'stroke-width': 2 }, svg);
    svg.addEventListener('click', function (e) { const r = svg.getBoundingClientRect(), sx = 300 / r.width; phi = Math.round(((e.clientX - r.left) * sx - 20) / 270 * 360 - 180); psi = Math.round((280 - (e.clientY - r.top) * sx) / 270 * 360 - 180); phi = AF.clamp(phi, -180, 180); psi = AF.clamp(psi, -180, 180); draw(); });
    const v = P.Viewer(right, { height: 320, width: 420 });
    const out = el('div', { class: 'out' }, host);
    function draw() {
      dot.setAttribute('cx', X(phi)); dot.setAttribute('cy', Y(psi));
      const n = 16, bb = P.buildBackbone(new Array(n).fill(phi), new Array(n).fill(psi));
      const pts = [], cols = [];
      bb.forEach(function (r) { pts.push(r.N, r.CA, r.C); cols.push('#3182ce', '#718096', '#2d3748'); });
      const segs = bb.map(function (r) { return [r.C, r.O, '#e53e3e', 3]; });
      const ca = bb.map(function (r) { return r.CA; });
      v.setZoomFrom(pts);
      v.set([{ pts: pts, color: function (i) { return cols[i]; }, mode: 'tube', width: 3.5, segments: segs }, { pts: ca, color: '#718096', mode: 'points', r: 3 }], true);
      const rise = P.dist(ca[0], ca[n - 1]) / (n - 1);
      const hb = P.dist(bb[3].O, bb[7].N);
      out.textContent = 'φ = ' + phi + '°, ψ = ' + psi + '°' +
        '\nlength gained per residue along the chain: ' + rise.toFixed(2) + ' Å  (α-helix ≈ 1.5 Å, β-strand ≈ 3.3 Å)' +
        '\ndistance O(i) → N(i+4): ' + hb.toFixed(2) + ' Å  ' + (hb < 3.3 ? '← a hydrogen bond: this is what holds an α-helix together' : '') +
        '\nBlue = N, grey = Cα, black = C, red stubs = C=O oxygens. The peptide bond itself (ω) is kept flat at 180°.';
    }
    draw();
  };
})();
