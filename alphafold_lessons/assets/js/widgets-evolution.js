/* Part III widgets: MSAs, coevolution, contacts → coordinates, superposition and accuracy metrics. */
(function () {
  'use strict';
  const AF = window.AF, W = AF.widgets, el = AF.el, S = AF.svg, P = window.Prot;
  const CLS_COL = { B: '#d69e2e', S: '#a0aec0', K: '#3182ce', E: '#e53e3e' };
  AF.toyClassColor = function (c) { return CLS_COL[c] || '#999'; };
  function letterClass(a) { for (const k in P.TOY_LETTERS) if (P.TOY_LETTERS[k].indexOf(a) >= 0) return k; return null; }
  AF.letterClass = letterClass;

  // predicted contacts from an MSA of depth n (cached)
  AF.coevoCache = {};
  AF.predictContacts = function (n, seed) {
    const k = n + ':' + (seed || 11);
    if (AF.coevoCache[k]) return AF.coevoCache[k];
    const toy = AF.toy(), m = AF.toyMSA(n, seed);
    const co = P.coevolution(m.cls, { alphabet: 'BSKE' });
    const top = P.topPairs(co.APC, 6, Math.round(toy.L * 0.6));
    const truth = new Set(toy.longRange.map(function (c) { return c[0] + ',' + c[1]; }));
    const hit = top.map(function (p) { return truth.has(p[0] + ',' + p[1]); });
    const prec = hit.filter(Boolean).length / top.length;
    return (AF.coevoCache[k] = { co: co, top: top, hit: hit, precision: prec, msa: m });
  };

  // ---------------------------------------------------------------- MSA viewer
  W.msaViewer = function (host) {
    const toy = AF.toy();
    AF.head(host, 'A multiple sequence alignment (MSA)', 'Row 1 is our query protein; the other rows are related sequences from the toy “evolution”. Each <b>column</b> is one position of the protein across species. The bar above each column shows how <b>conserved</b> it is. Increase the depth and look for columns that change together.');
    const ctr = el('div', { class: 'controls' }, host);
    let n = 30, colour = 'class';
    AF.select(ctr, 'Depth (sequences)', [10, 30, 100, 300].map(function (x) { return { value: x, label: String(x) }; }), n, function (v) { n = +v; draw(); });
    AF.select(ctr, 'Colour', [{ value: 'class', label: 'by chemistry' }, { value: 'diff', label: 'only differences from query' }], colour, function (v) { colour = v; draw(); });
    const box = el('div', { class: 'msa' }, host), out = el('div', { class: 'note' }, host);
    function draw() {
      const m = AF.toyMSA(n);
      box.innerHTML = '';
      const rows = [toy.seq].concat(m.seqs.slice(0, 15));
      const cons = [];
      for (let i = 0; i < toy.L; i++) cons.push(P.conservation(m.cls, i, 4));
      const cr = el('div', { class: 'r' }, box); el('span', { class: 'lab', text: 'conservation' }, cr);
      cons.forEach(function (c) { const s = el('span', null, cr); s.innerHTML = '<span style="display:inline-block;width:70%;height:' + Math.max(2, 14 * c) + 'px;background:var(--accent);vertical-align:bottom;border-radius:2px"></span>'; });
      const sr = el('div', { class: 'r' }, box); el('span', { class: 'lab', text: 'helix / loop' }, sr);
      toy.ss.split('').forEach(function (c) { el('span', { text: c === 'H' ? '▬' : '·', style: 'color:' + AF.ssColor(c) }, sr); });
      rows.forEach(function (s, r) {
        const row = el('div', { class: 'r' }, box); el('span', { class: 'lab', text: r === 0 ? 'query' : 'species ' + r }, row);
        s.split('').forEach(function (a, i) {
          const k = letterClass(a), diff = r > 0 && a !== toy.seq[i];
          const sp = el('span', { text: a }, row);
          if (colour === 'class' || r === 0 || diff) { sp.style.background = k ? CLS_COL[k] : P.aaColor(a); sp.style.color = '#fff'; } else sp.style.color = 'var(--muted)';
          if (r === 0) sp.style.fontWeight = '700';
        });
      });
      out.innerHTML = 'Showing 15 of ' + n + ' sequences. Toy chemistry: <b style="color:' + CLS_COL.B + '">big hydrophobic</b>, <b style="color:#718096">small</b>, <b style="color:' + CLS_COL.K + '">positive</b>, <b style="color:' + CLS_COL.E + '">negative</b>. Conserved columns are usually buried in the core; variable columns sit on the surface.';
    }
    draw();
  };

  // ---------------------------------------------------------------- one pair of columns
  W.coevoPair = function (host) {
    const toy = AF.toy();
    AF.head(host, 'Do two columns change together?', 'Pick two positions. The table counts, over 500 toy sequences, how often each combination of residue kinds appears together. If the positions touch in 3-D, evolution must keep them compatible — so some combinations become common and others rare. <b>Mutual information</b> (MI) measures that dependence.');
    const pr = AF.predictContacts(300), best = pr.top.find(function (p, i) { return pr.hit[i]; });
    let i = best ? best[0] : 3, j = best ? best[1] : 40;
    const ctr = el('div', { class: 'controls' }, host);
    const si = AF.slider(ctr, 'position i', 1, toy.L, 1, i + 1, function (x) { i = x - 1; draw(); });
    const sj = AF.slider(ctr, 'position j', 1, toy.L, 1, j + 1, function (x) { j = x - 1; draw(); });
    AF.button(ctr, 'A touching pair', function () { const c = toy.longRange[Math.floor(Math.random() * toy.longRange.length)]; set(c[0], c[1]); });
    AF.button(ctr, 'A distant pair', function () { let a, b; do { a = Math.floor(Math.random() * toy.L); b = Math.floor(Math.random() * toy.L); } while (Math.abs(a - b) < 6 || P.dist(toy.ca[a], toy.ca[b]) < 14); set(Math.min(a, b), Math.max(a, b)); });
    function set(a, b) { i = a; j = b; si.value = a + 1; sj.value = b + 1; si.dispatchEvent(new Event('input')); sj.dispatchEvent(new Event('input')); }
    const g = el('div', { class: 'two-col' }, host), tab = el('div', null, g), out = el('div', { class: 'out' }, g);
    function draw() {
      const m = AF.toyMSA(500).cls, K = 'BSKE', C = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], fi = [0, 0, 0, 0], fj = [0, 0, 0, 0];
      m.forEach(function (s) { const a = K.indexOf(s[i]), b = K.indexOf(s[j]); C[a][b]++; fi[a]++; fj[b]++; });
      const N = m.length; let mi = 0;
      for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) if (C[a][b]) { const p = C[a][b] / N; mi += p * Math.log(p / (fi[a] / N * fj[b] / N)); }
      tab.innerHTML = '';
      const t = el('table', null, tab), hr = el('tr', null, t);
      el('th', { text: 'i=' + (i + 1) + ' \\ j=' + (j + 1) }, hr); K.split('').forEach(function (k) { el('th', { text: k, style: 'color:' + CLS_COL[k] }, hr); });
      const mx = Math.max.apply(null, C.map(function (r) { return Math.max.apply(null, r); }));
      K.split('').forEach(function (k, a) {
        const tr = el('tr', null, t); el('th', { text: k, style: 'color:' + CLS_COL[k] }, tr);
        C[a].forEach(function (c, b) { const exp = fi[a] * fj[b] / N; el('td', { text: c, title: 'expected if independent: ' + exp.toFixed(0), style: 'text-align:center;background:' + AF.heat(0.85 * c / (mx || 1)) + ';color:' + (c / mx > 0.55 ? '#fff' : 'var(--ink)') }, tr); });
      });
      const d = P.dist(toy.ca[i], toy.ca[j]);
      out.textContent = 'positions ' + (i + 1) + ' and ' + (j + 1) + ' are ' + d.toFixed(1) + ' Å apart' + (d < 8 ? ' → in contact' : '') +
        '\nMI = Σ p(a,b) log[ p(a,b) / (p(a) p(b)) ] = ' + mi.toFixed(3) +
        '\n\nB = big hydrophobic, S = small, K = +, E = −.\nFor a touching pair you should see “big with small” and “+ with −” combinations dominate; for a distant pair, counts are close to what independence predicts (hover a cell).';
    }
    draw();
  };

  // ---------------------------------------------------------------- the coevolution contact map vs MSA depth
  W.coevoMap = function (host) {
    const toy = AF.toy();
    AF.head(host, 'Predicting contacts from coevolution', 'For every pair of columns compute MI with the <b>average-product correction</b> (APC), and call the top-scoring pairs “contacts”. <b>Upper triangle:</b> predictions (green = correct, red = wrong). <b>Lower triangle:</b> the true contacts from the 3-D structure. Change the number of sequences.');
    const ctr = el('div', { class: 'controls' }, host);
    let n = 30;
    const depths = [10, 30, 100, 300, 1000];
    AF.select(ctr, 'MSA depth', depths.map(function (x) { return { value: x, label: x + ' sequences' }; }), n, function (v) { n = +v; draw(); });
    const g = el('div', { class: 'two-col' }, host), left = el('div', null, g), right = el('div', null, g);
    const cv = el('canvas', { width: 400, height: 400, class: 'heat', style: 'width:100%;max-width:400px' }, left);
    const plot = el('div', null, right), out = el('div', { class: 'out' }, right);
    function draw() {
      out.textContent = 'computing…';
      setTimeout(function () {
        const pr = AF.predictContacts(n), L = toy.L, c = 400 / L, ctx = cv.getContext('2d');
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-soft') || '#eee'; ctx.fillRect(0, 0, 400, 400);
        ctx.strokeStyle = 'rgba(128,128,128,.5)'; ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(400, 400); ctx.stroke();
        toy.contacts.forEach(function (p) { ctx.fillStyle = p[1] - p[0] >= 6 ? '#1f5fa8' : 'rgba(31,95,168,.35)'; ctx.fillRect(p[0] * c, p[1] * c, c, c); });
        pr.top.forEach(function (p, k) { ctx.fillStyle = pr.hit[k] ? '#2f855a' : '#e53e3e'; ctx.fillRect(p[1] * c, p[0] * c, c, c); });
        const pts = depths.map(function (d) { return [d, AF.predictContacts(d).precision * 100]; });
        AF.chart.lines(plot, [{ color: 'var(--accent)', points: pts }, { color: '#e53e3e', points: [[n, pr.precision * 100]], r: 7 }], { logx: true, xmin: 10, xmax: 1000, ymin: 0, ymax: 100, xticks: depths, xlabel: 'sequences in the MSA (log scale)', ylabel: 'precision of top contacts (%)', height: 220 });
        out.textContent = 'MSA depth ' + n + ': ' + pr.hit.filter(Boolean).length + ' of the top ' + pr.top.length + ' predicted long-range contacts are real (precision ' + (pr.precision * 100).toFixed(0) + '%).' +
          '\nRandom guessing would score about ' + (100 * toy.longRange.length / ((L - 6) * (L - 5) / 2)).toFixed(0) + '%.' +
          '\nShallow alignments give noise; deep ones reveal the fold. AlphaFold’s accuracy also drops sharply below ~30 effective sequences (paper Fig. 5a).';
      }, 20);
    }
    draw();
  };

  // ---------------------------------------------------------------- from contacts to coordinates
  W.foldContacts = function (host) {
    const toy = AF.toy();
    AF.head(host, 'From predicted contacts to a 3-D model (the pre-2020 recipe)', 'Take the predicted contacts, add the local distances expected inside helices, and search for 3-D coordinates that satisfy them all (shortest-path distance bounds → classical multidimensional scaling → gradient descent). The <b>grey</b> chain is the true structure; the <b>coloured</b> chain is the model after optimal superposition.');
    const ctr = el('div', { class: 'controls' }, host);
    let src = '300';
    AF.select(ctr, 'Contacts from', [{ value: '10', label: 'MSA, 10 sequences' }, { value: '30', label: 'MSA, 30 sequences' }, { value: '100', label: 'MSA, 100 sequences' }, { value: '300', label: 'MSA, 300 sequences' }, { value: 'true', label: 'the true structure (cheating)' }], src, function (v) { src = v; run(); });
    const fb = AF.button(ctr, 'Run again', run, 'primary');
    const v = P.Viewer(host, { height: 360 });
    const out = el('div', { class: 'out' }, host);
    let anim = null;
    function run() {
      if (anim) clearTimeout(anim);
      const pairs = src === 'true' ? toy.longRange : AF.predictContacts(+src).top;
      const f = P.foldFromContacts(toy, pairs);
      let it = 0;
      const tick = function () {
        f.step(0.003, 60); it += 60;
        let X = f.step.coords();
        const tm = P.tmscore(X, toy.ca), tmm = P.tmscore(P.mirror(X), toy.ca);
        const flip = tmm > tm; if (flip) X = P.mirror(X);
        const T = P.superpose(X, toy.ca), Y = P.transformAll(T, X);
        v.set([{ pts: toy.ca, color: 'var(--muted)', width: 7, opacity: 0.45 }, { pts: Y, color: function (i) { return AF.rainbow(i / (toy.L - 1)); }, width: 4.5 }], true);
        out.textContent = 'refinement step ' + it + ' / 1500 · restraint violation = ' + f.step.energy.toFixed(1) +
          '\nTM-score of the model: ' + tm.toFixed(2) + '    TM-score of its mirror image: ' + tmm.toFixed(2) +
          (flip ? '\n⚠ The search converged to the MIRROR image (shown flipped back). Distances alone cannot tell left from right — Chapter 20 explains how AlphaFold avoids this.' : '') +
          '\nTM-score > 0.5 ≈ the right fold;  > 0.8 ≈ very close.';
        if (it < 1500) anim = setTimeout(tick, 16);
      };
      v.setZoomFrom(toy.ca);
      tick();
    }
    run();
  };

  // ---------------------------------------------------------------- superposition
  W.superpose = function (host) {
    const toy = AF.toy();
    AF.head(host, 'Superposition: line two structures up before comparing', 'To compare a model to the truth we first find the rotation and translation that bring them as close as possible (the least-squares optimum, found here with Horn’s quaternion method — equivalent to the Kabsch algorithm). Press <b>Superpose</b> and watch.');
    const ctr = el('div', { class: 'controls' }, host);
    let noise = 1.0, t = 0, anim = null;
    AF.slider(ctr, 'model error (Å)', 0, 4, 0.1, noise, function (x) { noise = x; make(); }, function (x) { return x.toFixed(1); });
    AF.button(ctr, 'Superpose', function () { go(); }, 'primary');
    AF.button(ctr, 'Scramble', function () { make(); });
    const v = P.Viewer(host, { height: 340 }), out = el('div', { class: 'out' }, host);
    let model, T, rnd = P.rng(5);
    function make() {
      const R = P.rotEuler(rnd() * 6, rnd() * 3, rnd() * 6), sh = [8 * (rnd() - 0.5), 8 * (rnd() - 0.5), 8 * (rnd() - 0.5)];
      model = toy.ca.map(function (x) { return P.add(P.mv(R, P.add(x, [noise * P.gauss(rnd), noise * P.gauss(rnd), noise * P.gauss(rnd)])), sh); });
      T = P.superpose(model, toy.ca); t = 0; draw();
    }
    function draw() {
      const aligned = P.transformAll(T, model), cur = model.map(function (x, i) { return P.lerp(x, aligned[i], t); });
      v.set([{ pts: toy.ca, color: 'var(--muted)', width: 7, opacity: 0.5 }, { pts: cur, color: function (i) { return AF.rainbow(i / (toy.L - 1)); }, width: 4 }], true);
      out.textContent = 'RMSD without superposition: ' + P.rmsdNoAlign(model, toy.ca).toFixed(2) + ' Å\nRMSD after optimal superposition: ' + T.rmsd.toFixed(2) + ' Å' + (t >= 1 ? '   ✓ aligned' : '') +
        '\nRMSD = √( (1/N) Σ_i ‖x_i − y_i‖² ). The remaining error after superposition is the model’s real error.';
    }
    function go() { if (anim) clearInterval(anim); t = 0; anim = setInterval(function () { t = Math.min(1, t + 0.04); draw(); if (t >= 1) clearInterval(anim); }, 30); }
    v.setZoomFrom(toy.ca);
    make();
  };

  // ---------------------------------------------------------------- the metrics lab
  AF.hinge = function (X, ang, from) {
    from = from == null ? 32 : from;
    const pivot = X[from], axis = P.unit(P.sub(X[from + 2], X[from - 2])), R = P.rotAxis(axis, ang * P.DEG);
    return X.map(function (x, i) { return i < from ? x.slice() : P.add(P.mv(R, P.sub(x, pivot)), pivot); });
  };
  W.metricsLab = function (host) {
    const toy = AF.toy();
    AF.head(host, 'Four ways to score a prediction', 'Make a “prediction” by adding random error and/or swinging the third helix around its hinge. Compare how each metric reacts. RMSD punishes one wrong hinge heavily; lDDT, being <i>local</i>, barely notices; GDT and TM-score sit in between.');
    const ctr = el('div', { class: 'controls' }, host);
    let noise = 0.8, hinge = 0;
    AF.slider(ctr, 'random error (Å)', 0, 5, 0.1, noise, function (x) { noise = x; draw(); }, function (x) { return x.toFixed(1); });
    AF.slider(ctr, 'hinge swing of helix 3', 0, 90, 5, hinge, function (x) { hinge = x; draw(); }, function (x) { return x + '°'; });
    const stats = el('div', null, host);
    const v = P.Viewer(host, { height: 320 });
    const out = el('div', { class: 'note' }, host);
    const r0 = P.rng(21), eps = toy.ca.map(function () { return [P.gauss(r0), P.gauss(r0), P.gauss(r0)]; });
    v.setZoomFrom(toy.ca);
    function draw() {
      let X = toy.ca.map(function (x, i) { return P.add(x, P.scale(eps[i], noise / Math.sqrt(3))); });
      X = AF.hinge(X, hinge);
      const T = P.superpose(X, toy.ca), Y = P.transformAll(T, X);
      const ld = P.lddt(X, toy.ca);
      const m = { 'RMSD (Å)': P.rmsd(X, toy.ca).toFixed(2), 'r.m.s.d.95 (Å)': P.rmsd95(X, toy.ca).toFixed(2), 'GDT_TS (0–100)': P.gdt(X, toy.ca).toFixed(1), 'lDDT-Cα (0–100)': ld.global.toFixed(1), 'TM-score (0–1)': P.tmscore(X, toy.ca).toFixed(3) };
      stats.innerHTML = Object.keys(m).map(function (k) { return '<div class="stat"><b>' + m[k] + '</b><span>' + k + '</span></div>'; }).join('');
      v.set([{ pts: toy.ca, color: 'var(--muted)', width: 7, opacity: 0.45 }, { pts: Y, color: function (i) { return AF.plddtColor(ld.perResidue[i]); }, width: 4.5 }], true);
      out.innerHTML = 'Model coloured by its <b>per-residue lDDT</b> (blue &gt; 90, light blue 70–90, yellow 50–70, orange &lt; 50) — the same colour scale AlphaFold uses for its confidence (Chapter 23). Grey = truth, after superposition on all residues.';
    }
    draw();
  };
})();
