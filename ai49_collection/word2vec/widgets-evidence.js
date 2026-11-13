/* Word2Vec Interactive Web Book — Part IV (evidence: §4–§7 of the paper).
   Paper numbers are typed in from the paper (arXiv v3) and cited; toy numbers are computed live and labelled. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, W2V = window.W2V, U = window.W2VUI;

  // ------------------------------------------------------------------ ch-analogy
  W.analogyCalc = function (host) {
    Book.head(host, 'The analogy calculator',
      '“a is to b as c is to ?” → X = vector(b) − vector(a) + vector(c), then the word whose vector has the highest cosine with X. Everything below is computed from the trained toy vectors.');
    const presets = [['man', 'king', 'woman'], ['france', 'paris', 'germany'], ['big', 'biggest', 'small'], ['big', 'bigger', 'fast'], ['france', 'french', 'japan'], ['girl', 'princess', 'boy']];
    const st = { a: 'man', b: 'king', c: 'woman', cbow: false, keep: false };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'preset', presets.map(function (p, i) { return { value: i, label: p[0] + ' : ' + p[1] + ' :: ' + p[2] + ' : ?' }; }), 0, function (v) {
      const p = presets[+v]; st.a = p[0]; st.b = p[1]; st.c = p[2]; sa.value = st.a; sb.value = st.b; sc.value = st.c; draw(); });
    Book.select(ctl, 'model', [{ value: 'sg', label: 'skip-gram' }, { value: 'cbow', label: 'CBOW' }], 'sg', function (v) { st.cbow = v === 'cbow'; draw(); });
    const ctl2 = el('div', { class: 'controls' }, host);
    const words = U.SETS.all.concat(W2V.places.map(function (p) { return p.food; })).sort();
    const sa = U.select(ctl2, 'a', words, st.a, function (v) { st.a = v; draw(); });
    const sb = U.select(ctl2, 'b', words, st.b, function (v) { st.b = v; draw(); });
    const sc = U.select(ctl2, 'c', words, st.c, function (v) { st.c = v; draw(); });
    const kb = el('label', null, ctl2), cb = el('input', { type: 'checkbox' }, kb); kb.appendChild(document.createTextNode(' allow the input words as answers'));
    cb.addEventListener('change', function () { st.keep = cb.checked; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const m = U.model(st.cbow), C = m.corpus, M = m.normalized();
      const r = W2V.analogy(M, C.words, C.index[st.a], C.index[st.b], C.index[st.c], 6, st.keep);
      Book.chart.bars(chart, r.top.map(function (t) { return { label: t.w, value: Math.max(0, t.cos), note: Book.fmt(t.cos, 2), color: U.color(t.w) }; }), { max: 1, height: 200, ylabel: 'cosine with X' });
      out.innerHTML = 'X = ' + st.b + ' − ' + st.a + ' + ' + st.c + ' → answer: <b>' + r.top[0].w + '</b> (cosine ' + Book.fmt(r.top[0].cos, 3) + ')' +
        '<br>cos(X, ' + st.b + ') = ' + Book.fmt(ML.cosine(r.x, M[C.index[st.b]]), 2) + ', cos(X, ' + st.c + ') = ' + Book.fmt(ML.cosine(r.x, M[C.index[st.c]]), 2) +
        (st.keep ? ' — with inputs allowed, an input word often wins, which is why the paper discards them.' : '') +
        '<br><span class="note">Toy vectors (20 dimensions, ' + (st.cbow ? 'CBOW' : 'skip-gram') + '). Search as in word-analogy.c: unit-length vectors, inputs skipped unless you tick the box.</span>';
    }
    draw();
  };

  W.offsetMap = function (host) {
    Book.head(host, 'Same relation, same arrow',
      'Each arrow goes from the first to the second word of a pair. If a relation is a consistent direction in the space, the arrows are parallel. The number is the average cosine between the arrows, measured in all 20 dimensions.');
    const rels = W2V.relations();
    const st = { r: 0, cbow: false, view: {} };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'relation', rels.map(function (r, i) { return { value: i, label: r.name }; }), st.r, function (v) { st.r = +v; st.view = {}; draw(); });
    Book.select(ctl, 'model', [{ value: 'sg', label: 'skip-gram' }, { value: 'cbow', label: 'CBOW' }], 'sg', function (v) { st.cbow = v === 'cbow'; st.view = {}; draw(); });
    const map = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const m = U.model(st.cbow), rel = rels[st.r], words = [].concat.apply([], rel.pairs);
      U.plotWords(map, m, words, st.view, { pairs: rel.pairs.map(function (p) { return [p[0], p[1], 'var(--accent)']; }), width: 520, height: 320 });
      const offs = rel.pairs.map(function (p) { return ML.sub(m.vector(p[1]), m.vector(p[0])); });
      let s = 0, n = 0; for (let i = 0; i < offs.length; i++) for (let j = i + 1; j < offs.length; j++) { s += ML.cosine(offs[i], offs[j]); n++; }
      out.innerHTML = 'average cosine between the ' + offs.length + ' offset vectors: <b>' + Book.fmt(s / n, 2) + '</b> (1 = perfectly parallel, 0 = unrelated)' +
        '<br><span class="note">Toy vectors; the 2-D picture is a projection (PCA) and can distort angles, the number cannot.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-testset
  const TABLE1 = [
    ['semantic', 'Common capital city', 'Athens', 'Greece', 'Oslo', 'Norway', 506, 'capital-common-countries'],
    ['semantic', 'All capital cities', 'Astana', 'Kazakhstan', 'Harare', 'Zimbabwe', 4524, 'capital-world'],
    ['semantic', 'Currency', 'Angola', 'kwanza', 'Iran', 'rial', 866, 'currency'],
    ['semantic', 'City-in-state', 'Chicago', 'Illinois', 'Stockton', 'California', 2467, 'city-in-state'],
    ['semantic', 'Man-Woman', 'brother', 'sister', 'grandson', 'granddaughter', 506, 'family'],
    ['syntactic', 'Adjective to adverb', 'apparent', 'apparently', 'rapid', 'rapidly', 992, 'gram1-adjective-to-adverb'],
    ['syntactic', 'Opposite', 'possibly', 'impossibly', 'ethical', 'unethical', 812, 'gram2-opposite'],
    ['syntactic', 'Comparative', 'great', 'greater', 'tough', 'tougher', 1332, 'gram3-comparative'],
    ['syntactic', 'Superlative', 'easy', 'easiest', 'lucky', 'luckiest', 1122, 'gram4-superlative'],
    ['syntactic', 'Present Participle', 'think', 'thinking', 'read', 'reading', 1056, 'gram5-present-participle'],
    ['syntactic', 'Nationality adjective', 'Switzerland', 'Swiss', 'Cambodia', 'Cambodian', 1599, 'gram6-nationality-adjective'],
    ['syntactic', 'Past tense', 'walking', 'walked', 'swimming', 'swam', 1560, 'gram7-past-tense'],
    ['syntactic', 'Plural nouns', 'mouse', 'mice', 'dollar', 'dollars', 1332, 'gram8-plural'],
    ['syntactic', 'Plural verbs', 'work', 'works', 'speak', 'speaks', 870, 'gram9-plural-verbs']];
  W.table1 = function (host) {
    Book.head(host, 'The 14 question types (Table 1) and how many questions each has',
      'Bars: number of questions per type in questions-words.txt from the official repository. Pick a type to read its two example pairs from Table 1 as a question.');
    const st = { k: 4 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'type', TABLE1.map(function (t, i) { return { value: i, label: t[1] }; }), st.k, function (v) { st.k = +v; draw(); });
    const chart = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const ch = el('div', { style: 'min-width:640px' }); chart.innerHTML = ''; chart.appendChild(ch);
      Book.chart.bars(ch, TABLE1.map(function (t, i) { return { label: String(i + 1), value: t[6], note: t[6], color: i === st.k ? 'var(--accent)' : (t[0] === 'semantic' ? 'var(--teal)' : 'var(--purple)') }; }), { height: 220, ylabel: 'questions', mb: 30 });
      const t = TABLE1[st.k];
      let sem = 0, syn = 0; TABLE1.forEach(function (r) { if (r[0] === 'semantic') sem += r[6]; else syn += r[6]; });
      out.innerHTML = '<b>' + (st.k + 1) + '. ' + t[1] + '</b> (' + t[0] + ', file section “' + t[7] + '”): ' + t[6] + ' questions' +
        '<br>question: <b>' + t[2] + '</b> is to <b>' + t[3] + '</b> as <b>' + t[4] + '</b> is to <b>?</b> (answer: ' + t[5] + ')' +
        '<br>totals: semantic (teal, 1–5) = <b>' + sem.toLocaleString() + '</b>, syntactic (purple, 6–14) = <b>' + syn.toLocaleString() + '</b> — the paper’s 8869 and 10675 (§4.1).';
    }
    draw();
  };

  W.toyTest = function (host) {
    Book.head(host, 'The toy test set, scored like the paper',
      'The toy has 6 relation types built the paper’s way: a list of pairs per relation, then every ordered pair of pairs is a question. Exact match only. Compare the two architectures.');
    const st = { cbow: false };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'model', [{ value: 'sg', label: 'skip-gram' }, { value: 'cbow', label: 'CBOW' }], 'sg', function (v) { st.cbow = v === 'cbow'; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const ev = W2V.evaluate(U.model(st.cbow));
      Book.chart.bars(chart, Object.keys(ev.per).map(function (k) { const p = ev.per[k]; return { label: k.replace('–', '–\n'), value: p.right / p.n, note: p.right + '/' + p.n, color: p.type === 'semantic' ? 'var(--teal)' : 'var(--purple)' }; }),
        { max: 1, height: 230, ylabel: 'accuracy', mb: 56, tickFmt: function (v) { return Math.round(100 * v) + '%'; } });
      out.innerHTML = 'overall <b>' + U.pct(ev.acc) + '</b> of ' + ev.n + ' · semantic ' + U.pct(ev.semantic) + ' · syntactic ' + U.pct(ev.syntactic) +
        '<br>some misses: ' + ev.misses.slice(0, 6).map(function (x) { return x.q.a + ':' + x.q.b + ' :: ' + x.q.c + ':? → ' + x.got + ' (want ' + x.q.d + ')'; }).join(' · ') +
        '<br><span class="note">Toy results; the toy corpus is small and synthetic, so do not compare these percentages with the paper’s.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-datadims
  const T2 = { dims: [50, 100, 300, 600], words: ['24M', '49M', '98M', '196M', '391M', '783M'], wn: [24, 49, 98, 196, 391, 783],
    acc: [[13.4, 15.7, 18.6, 19.1, 22.5, 23.2], [19.4, 23.1, 27.8, 28.7, 33.4, 32.2], [23.2, 29.2, 35.3, 38.6, 43.7, 45.9], [24.0, 30.1, 36.5, 40.8, 46.6, 50.4]] };
  W.table2 = function (host) {
    Book.head(host, 'Table 2: accuracy as data and dimensions grow',
      'CBOW vectors, questions restricted to the 30k most frequent words. Each line is one vector size. Pick a cell to compare “double the data” with “double-or-more the dimensions”.');
    const st = { i: 1, j: 2 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'dimensions', T2.dims, T2.dims[st.i], function (v) { st.i = T2.dims.indexOf(+v); draw(); });
    Book.select(ctl, 'training words', T2.words, T2.words[st.j], function (v) { st.j = T2.words.indexOf(v); draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const series = T2.dims.map(function (d, i) { return { name: d + 'd', color: Book.palette[i], points: T2.wn.map(function (w, j) { return [w, T2.acc[i][j]]; }) }; });
      series.push({ color: 'var(--ink)', noline: true, r: 7, points: [[T2.wn[st.j], T2.acc[st.i][st.j]]] });
      Book.chart.lines(chart, series, { logx: true, xmin: 24, xmax: 783, ymin: 0, ymax: 55, xticks: T2.wn, xfmt: function (v) { return v + 'M'; }, xlabel: 'training words (log scale)', ylabel: 'accuracy [%]', mr: 46, height: 260 });
      const a = T2.acc[st.i][st.j];
      const moreData = st.j < 5 ? T2.acc[st.i][st.j + 1] : null, moreDim = st.i < 3 ? T2.acc[st.i + 1][st.j] : null;
      out.innerHTML = T2.dims[st.i] + ' dimensions, ' + T2.words[st.j] + ' words: <b>' + a + '%</b>' +
        '<br>2× data (' + (moreData != null ? T2.words[st.j + 1] + '): ' + moreData + '% (' + (moreData - a >= 0 ? '+' : '') + Book.fmt(moreData - a, 1) + ')' : 'not in the table)') +
        '<br>more dimensions (' + (moreDim != null ? T2.dims[st.i + 1] + 'd): ' + moreDim + '% (' + (moreDim - a >= 0 ? '+' : '') + Book.fmt(moreDim - a, 1) + ')' : 'not in the table)') +
        '<br><span class="note">All numbers from paper Table 2 (§4.2). Note 100d at 783M (32.2) is below 391M (33.4): small dimensions saturate.</span>';
    }
    draw();
  };

  W.toyGrid = function (host) {
    Book.head(host, 'The same experiment on the toy',
      'Train CBOW for every combination of vector size and corpus size, then score the toy test set. 16 trainings run one after another (about two seconds).');
    const dims = [2, 5, 10, 20], sizes = [300, 600, 1200, 2400];
    const st = { cancel: null, res: {} };
    const ctl = el('div', { class: 'controls' }, host);
    Book.button(ctl, 'Run the grid', function () { run(); }, 'primary');
    const tbl = el('div', { class: 'scroll-x' }, host), out = el('div', { class: 'out' }, host);
    function draw() {
      let h = '<table><tr><th>dims \\ sentences</th>' + sizes.map(function (s) { return '<th>' + s + '</th>'; }).join('') + '</tr>';
      dims.forEach(function (d) { h += '<tr><td><b>' + d + '</b></td>' + sizes.map(function (s) { const v = st.res[d + ':' + s]; return '<td>' + (v == null ? '…' : U.pct(v)) + '</td>'; }).join('') + '</tr>'; });
      tbl.innerHTML = h + '</table>';
    }
    function run() {
      if (st.cancel) st.cancel(); st.res = {};
      const jobs = []; dims.forEach(function (d) { sizes.forEach(function (s) { jobs.push([d, s]); }); });
      let k = 0;
      out.innerHTML = 'training…';
      st.cancel = U.loop(function () {
        const j = jobs[k++];
        const corpus = W2V.makeCorpus({ sentences: j[1] });
        const m = new W2V.Model({ corpus: corpus, cbow: true, D: j[0], window: 3, epochs: 6, seed: 1 }).trainAll();
        st.res[j[0] + ':' + j[1]] = W2V.evaluate(m).acc; draw();
        if (k >= jobs.length) { out.innerHTML = 'Done. Toy accuracies (CBOW, 6 epochs, window 3). Look along a row (more data) and down a column (more dimensions): with 2 dimensions, more data barely helps; with little data, more dimensions barely help.'; return false; }
        return true;
      }, 10);
    }
    draw();
    out.innerHTML = 'Press “Run the grid”. Toy experiment, not the paper’s.';
  };

  // ------------------------------------------------------------------ ch-archs
  W.table3 = function (host) {
    Book.head(host, 'Table 3: four architectures, same data, 640 dimensions',
      'Pick which accuracy to compare. Data: LDC corpora, 320M words, 82K vocabulary (§4.3).');
    const T3 = [['RNNLM', 9, 36, 35], ['NNLM', 23, 53, 47], ['CBOW', 24, 64, 61], ['Skip-gram', 55, 59, 56]];
    const st = { k: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'accuracy', [{ value: 1, label: 'semantic' }, { value: 2, label: 'syntactic' }, { value: 3, label: 'MSR word relatedness' }], st.k, function (v) { st.k = +v; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      Book.chart.bars(chart, T3.map(function (r, i) { return { label: r[0], value: r[st.k], note: r[st.k] + '%', color: Book.palette[i] }; }), { max: 70, height: 220, ylabel: 'accuracy [%]' });
      const best = T3.slice().sort(function (a, b) { return b[st.k] - a[st.k]; })[0];
      out.innerHTML = 'best: <b>' + best[0] + '</b> (' + best[st.k] + '%). Paper Table 3.<br><span class="note">Skip-gram wins the semantic part by far; CBOW wins the syntactic part and the MSR set.</span>';
    }
    draw();
  };

  const T4 = [['Collobert-Weston NNLM', 50, '660M', 9.3, 12.3, 11.0], ['Turian NNLM', 50, '37M', 1.4, 2.6, 2.1], ['Turian NNLM', 200, '37M', 1.4, 2.2, 1.8],
    ['Mnih NNLM', 50, '37M', 1.8, 9.1, 5.8], ['Mnih NNLM', 100, '37M', 3.3, 13.2, 8.8], ['Mikolov RNNLM', 80, '320M', 4.9, 18.4, 12.7], ['Mikolov RNNLM', 640, '320M', 8.6, 36.5, 24.6],
    ['Huang NNLM', 50, '990M', 13.3, 11.6, 12.3], ['Our NNLM', 20, '6B', 12.9, 26.4, 20.3, 1], ['Our NNLM', 50, '6B', 27.9, 55.8, 43.2, 1], ['Our NNLM', 100, '6B', 34.2, 64.5, 50.8, 1],
    ['CBOW', 300, '783M', 15.5, 53.1, 36.1, 1], ['Skip-gram', 300, '783M', 50.0, 55.9, 53.3, 1]];
  W.table4 = function (host) {
    Book.head(host, 'Table 4: published word vectors versus the paper’s',
      'Sort by any accuracy. Coloured bars are the paper’s own models; grey bars are vectors other groups had published.');
    const st = { k: 5 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'sort by', [{ value: 3, label: 'semantic' }, { value: 4, label: 'syntactic' }, { value: 5, label: 'total' }], st.k, function (v) { st.k = +v; draw(); });
    const pic = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const rows = T4.slice().sort(function (a, b) { return b[st.k] - a[st.k]; });
      pic.innerHTML = '';
      const Wd = 640, rh = 24, L = 250, Ht = rows.length * rh + 10;
      const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + Ht, role: 'img', 'aria-label': 'Table 4 as bars' }, pic);
      rows.forEach(function (r, i) {
        const y = 5 + i * rh, w = r[st.k] / 70 * (Wd - L - 60);
        S('text', { x: L - 8, y: y + 16, 'text-anchor': 'end', 'font-size': 12, 'font-weight': r[6] ? 700 : 400, class: 'svg-text', text: r[0] + ' · ' + r[1] + 'd · ' + r[2] }, svg);
        S('rect', { x: L, y: y + 4, width: Math.max(1, w), height: rh - 8, rx: 3, fill: r[6] ? 'var(--accent)' : 'var(--muted)' }, svg);
        S('text', { x: L + w + 6, y: y + 16, 'font-size': 12, class: 'svg-text', text: r[st.k] }, svg);
      });
      out.innerHTML = 'Top by ' + ['', '', '', 'semantic', 'syntactic', 'total'][st.k] + ': <b>' + rows[0][0] + ' ' + rows[0][1] + 'd (' + rows[0][st.k] + '%)</b>. All values from paper Table 4; full vocabularies.' +
        '<br><span class="note">The paper’s CBOW and skip-gram were trained on one CPU (§4.3); “Our NNLM” used the DistBelief cluster and 6B words.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-scale
  const T5 = [['3 epoch CBOW', 300, '783M', 15.5, 53.1, 36.1, 1], ['3 epoch Skip-gram', 300, '783M', 50.0, 55.9, 53.3, 3], ['1 epoch CBOW', 300, '783M', 13.8, 49.9, 33.6, 0.3],
    ['1 epoch CBOW', 300, '1.6B', 16.1, 52.6, 36.1, 0.6], ['1 epoch CBOW', 600, '783M', 15.4, 53.3, 36.2, 0.7], ['1 epoch Skip-gram', 300, '783M', 45.6, 52.2, 49.2, 1],
    ['1 epoch Skip-gram', 300, '1.6B', 52.2, 55.1, 53.8, 2], ['1 epoch Skip-gram', 600, '783M', 56.7, 54.5, 55.5, 2.5]];
  W.table5 = function (host) {
    Book.head(host, 'Table 5: accuracy and training days, row by row',
      'Each bar is one row of Table 5 (bar length = accuracy, label = training days). Sort by accuracy or by days and compare the one-epoch rows with the three-epoch rows.');
    const st = { k: 5, sort: 'acc' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'accuracy', [{ value: 3, label: 'semantic' }, { value: 4, label: 'syntactic' }, { value: 5, label: 'total' }], st.k, function (v) { st.k = +v; draw(); });
    Book.select(ctl, 'sort by', [{ value: 'acc', label: 'accuracy' }, { value: 'days', label: 'training days' }], st.sort, function (v) { st.sort = v; draw(); });
    const pic = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const rows = T5.slice().sort(function (a, b) { return st.sort === 'acc' ? b[st.k] - a[st.k] : a[6] - b[6]; });
      pic.innerHTML = '';
      const Wd = 640, rh = 28, L = 250, Ht = rows.length * rh + 10;
      const svg = S('svg', { viewBox: '0 0 ' + Wd + ' ' + Ht, role: 'img', 'aria-label': 'Table 5 as bars' }, pic);
      rows.forEach(function (r, i) {
        const y = 5 + i * rh, w = r[st.k] / 60 * (Wd - L - 120), sg = r[0].indexOf('Skip') >= 0;
        S('text', { x: L - 8, y: y + 18, 'text-anchor': 'end', 'font-size': 12, 'font-weight': r[0].indexOf('3 epoch') >= 0 ? 700 : 400, class: 'svg-text', text: r[0] + ' · ' + r[1] + 'd · ' + r[2] }, svg);
        S('rect', { x: L, y: y + 5, width: Math.max(1, w), height: rh - 10, rx: 3, fill: sg ? 'var(--accent)' : 'var(--teal)' }, svg);
        S('text', { x: L + w + 6, y: y + 18, 'font-size': 12, class: 'svg-text', text: r[st.k] + '% · ' + r[6] + (r[6] === 1 ? ' day' : ' days') }, svg);
      });
      out.innerHTML = 'Skip-gram, 1 epoch on 1.6B words: <b>' + T5[6][5] + '%</b> total in 2 days, beating 3 epochs on 783M words (' + T5[1][5] + '% in 3 days).' +
        '<br>CBOW, 1 epoch on 1.6B: ' + T5[3][5] + '% in 0.6 days, the same total as 3 epochs on 783M (' + T5[0][5] + '% in 1 day). Bold rows: 3 epochs. All numbers from paper Table 5.';
    }
    draw();
  };

  W.table6 = function (host) {
    Book.head(host, 'Table 6: training on the DistBelief cluster (6B words)',
      'Bars show accuracy; the label shows the compute used, days × CPU cores, and their product in core-days.');
    const T6 = [['NNLM', 100, 34.2, 64.5, 50.8, 14, 180], ['CBOW', 1000, 57.3, 68.9, 63.7, 2, 140], ['Skip-gram', 1000, 66.1, 65.1, 65.6, 2.5, 125]];
    const st = { k: 4 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'accuracy', [{ value: 2, label: 'semantic' }, { value: 3, label: 'syntactic' }, { value: 4, label: 'total' }], st.k, function (v) { st.k = +v; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      Book.chart.bars(chart, T6.map(function (r, i) { return { label: r[0] + ' ' + r[1] + 'd\n' + r[5] + ' d × ' + r[6] + ' = ' + (r[5] * r[6]) + ' core-days', value: r[st.k], note: r[st.k] + '%', color: Book.palette[i + 1] }; }), { max: 80, height: 240, ylabel: 'accuracy [%]', mb: 50 });
      out.innerHTML = 'The NNLM with 100-dimensional vectors used <b>' + Book.fmt(14 * 180 / (2 * 140), 1) + '×</b> the core-days of CBOW with 1000-dimensional vectors, and scored lower. Paper Table 6, §4.4 (“training of NNLM with 1000-dimensional vectors would take too long to complete”).';
    }
    draw();
  };

  W.adagradAsync = function (host) {
    Book.head(host, 'Adagrad and stale gradients on a stretched bowl',
      'The loss is 0.5 × (w₁² + 25 w₂²): steep in one direction, flat in the other. Plain SGD must use a small step for the steep direction; Adagrad divides each weight’s step by the root of its summed squared gradients. “Staleness” uses gradients computed several steps ago, as asynchronous replicas do.');
    const st = { lr: 0.035, alr: 0.6, stale: 0, seed: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'SGD step', 0.005, 0.1, 0.005, st.lr, function (v) { st.lr = v; draw(); }, function (v) { return v.toFixed(3); });
    Book.slider(ctl, 'Adagrad step', 0.05, 1.5, 0.05, st.alr, function (v) { st.alr = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'staleness (steps)', 0, 6, 1, st.stale, function (v) { st.stale = v; draw(); });
    const box = el('div', { class: 'grid2' }, host), pic = el('div', null, box), chart = el('div', null, box), out = el('div', { class: 'out' }, host);
    const loss = function (w) { return 0.5 * (w[0] * w[0] + 25 * w[1] * w[1]); };
    function run(ada) {
      const r = ML.rng(st.seed), hist = [[-4, 1.5]], G = [0, 0], path = [];
      for (let t = 0; t < 80; t++) {
        const w = hist[hist.length - 1], old = hist[Math.max(0, hist.length - 1 - st.stale)];
        const g = [old[0] + r.normal(0, 0.3), 25 * old[1] + r.normal(0, 0.3)];
        let nw;
        if (ada) { G[0] += g[0] * g[0]; G[1] += g[1] * g[1]; nw = [w[0] - st.alr * g[0] / Math.sqrt(G[0] + 1e-8), w[1] - st.alr * g[1] / Math.sqrt(G[1] + 1e-8)]; }
        else nw = [w[0] - st.lr * g[0], w[1] - st.lr * g[1]];
        nw = nw.map(function (x) { return Book.clamp(x, -50, 50); });
        hist.push(nw); path.push([t + 1, Math.min(loss(nw), 40)]);
      }
      return { hist: hist, path: path };
    }
    function draw() {
      const a = run(false), b = run(true);
      pic.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 300 200', role: 'img', 'aria-label': 'Paths of SGD and Adagrad on a stretched bowl' }, pic);
      const X = function (x) { return 150 + x * 30; }, Y = function (y) { return 100 - y * 50; };
      [1, 4, 9, 16].forEach(function (c) { S('ellipse', { cx: 150, cy: 100, rx: Math.sqrt(2 * c) * 30, ry: Math.sqrt(2 * c / 25) * 50, fill: 'none', stroke: 'var(--line)' }, svg); });
      [[a.hist, 'var(--red)'], [b.hist, 'var(--blue)']].forEach(function (p) {
        S('polyline', { points: p[0].map(function (w) { return X(Book.clamp(w[0], -5, 5)) + ',' + Y(Book.clamp(w[1], -2, 2)); }).join(' '), fill: 'none', stroke: p[1], 'stroke-width': 1.5 }, svg);
      });
      Book.chart.lines(chart, [{ color: 'var(--red)', marker: 'none', points: a.path }, { color: 'var(--blue)', marker: 'none', points: b.path }],
        { width: 320, height: 200, ymin: 0, ymax: 40, xlabel: 'step', ylabel: 'loss (capped at 40)', xticks: [0, 40, 80] });
      out.innerHTML = 'after 80 noisy steps: <span style="color:var(--red)">SGD</span> loss <b>' + Book.fmt(loss(a.hist[80]), 3) + '</b>, <span style="color:var(--blue)">Adagrad</span> loss <b>' + Book.fmt(loss(b.hist[80]), 3) + '</b>' +
        '<br><span class="note">Toy problem. Push the SGD step above 0.08 (= 2 ÷ 25, the steep direction’s limit) and it diverges along the steep direction; staleness makes both less stable. The paper used Adagrad with mini-batch asynchronous updates and 50–100 replicas (§2.3, §4.4).</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-msr
  W.msrToy = function (host) {
    Book.head(host, 'Toy sentence completion: skip-gram, a bigram model, and both',
      'Each toy question is a held-out sentence with one word missing and 5 candidates of the same kind, chosen so that only one candidate fits the toy world. Skip-gram scores a candidate the paper’s way: put it at the input and add up log P of every other word in the sentence. Mix the two scores with the weight λ.');
    const C = U.corpus(), B = U.bigram();
    const pools = { people: U.SETS.people, city: W2V.places.map(function (p) { return p.city; }), country: W2V.places.map(function (p) { return p.country; }),
      language: W2V.places.map(function (p) { return p.lang; }), food: W2V.places.map(function (p) { return p.food; }), adj: U.SETS.adjectives };
    const poolOf = function (w) { for (const k in pools) if (pools[k].indexOf(w) >= 0) return pools[k]; return null; };
    const held = W2V.makeCorpus({ seed: 77, sentences: 600 }).raw, rng = ML.rng(5), qs = [];
    // every sentence the toy grammar can produce (a large sample of it), to keep only fair questions:
    // a question is kept when no wrong candidate also makes a sentence the toy world could say
    const possible = new Set(W2V.makeCorpus({ seed: 5, sentences: 30000 }).raw.map(function (x) { return x.join(' '); }));
    held.forEach(function (s) {
      const cand = s.map(function (w, i) { return poolOf(w) ? i : -1; }).filter(function (i) { return i >= 0; });
      if (!cand.length || qs.length >= 60) return;
      const pos = cand[rng.int(cand.length)], pool = poolOf(s[pos]).filter(function (w) { return w !== s[pos]; });
      rng.shuffle(pool);
      const wrong = pool.slice(0, 4);
      if (wrong.some(function (w) { const t = s.slice(); t[pos] = w; return possible.has(t.join(' ')); })) return;
      const opts = rng.shuffle([s[pos]].concat(wrong));
      qs.push({ s: s, pos: pos, opts: opts, ans: s[pos] });
    });
    const m = U.model(false);
    function sgScore(q, w) { const v = m.vector(w); let t = 0; q.s.forEach(function (x, i) { if (i !== q.pos) t += Math.log(m.hsProb(v, C.index[x])); }); return t; }
    function lmScore(q, w) {
      const s = q.s.slice(); s[q.pos] = w; let prev = B.BOS, t = 0;
      s.forEach(function (x) { const i = C.index[x]; t += Math.log(((B.big[prev * 1000 + i] || 0) + 0.1) / (B.uni[prev] + 0.1 * (C.V + 1))); prev = i; });
      return t;
    }
    const scored = qs.map(function (q) { return { q: q, sg: q.opts.map(function (w) { return sgScore(q, w); }), lm: q.opts.map(function (w) { return lmScore(q, w); }) }; });
    const acc = function (lam) { let r = 0; scored.forEach(function (x) { const s = x.q.opts.map(function (_, k) { return lam * x.sg[k] + (1 - lam) * x.lm[k]; }); if (x.q.opts[ML.argmax(s)] === x.q.ans) r++; }); return r / scored.length; };
    const st = { lam: 0.5, k: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'weight λ on skip-gram', 0, 1, 0.05, st.lam, function (v) { st.lam = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.button(ctl, 'Next question', function () { st.k = (st.k + 1) % scored.length; draw(); });
    const box = el('div', { class: 'grid2' }, host), q1 = el('div', null, box), chart = el('div', null, box), out = el('div', { class: 'out' }, host);
    const curve = ML.linspace(0, 1, 21).map(function (l) { return [l, acc(l)]; });
    function draw() {
      const x = scored[st.k], comb = x.q.opts.map(function (_, k) { return st.lam * x.sg[k] + (1 - st.lam) * x.lm[k]; }), pick = ML.argmax(comb);
      q1.innerHTML = '<div class="tokrow">' + x.q.s.map(function (w, i) { return '<span class="chip' + (i === x.q.pos ? ' hl' : '') + '">' + (i === x.q.pos ? '____' : w) + '</span>'; }).join('') + '</div>' +
        '<table><tr><th>candidate</th><th>skip-gram</th><th>bigram</th><th>mix</th></tr>' + x.q.opts.map(function (w, k) {
          return '<tr><td>' + (k === pick ? '<b>' + w + '</b>' : w) + (w === x.q.ans ? ' ✓' : '') + '</td><td>' + Book.fmt(x.sg[k], 1) + '</td><td>' + Book.fmt(x.lm[k], 1) + '</td><td>' + Book.fmt(comb[k], 1) + '</td></tr>'; }).join('') + '</table>';
      Book.chart.lines(chart, [{ color: 'var(--accent)', points: curve, marker: 'none' }, { color: 'var(--red)', noline: true, r: 6, points: [[st.lam, acc(st.lam)]] }],
        { width: 320, height: 210, xmin: 0, xmax: 1, ymin: 0, ymax: 1, xticks: [0, 0.5, 1], xlabel: 'λ (0 = bigram only, 1 = skip-gram only)', ylabel: 'toy accuracy', yfmt: function (v) { return Math.round(100 * v) + '%'; } });
      out.innerHTML = 'question ' + (st.k + 1) + ' of ' + scored.length + ' · accuracy at λ = ' + st.lam.toFixed(2) + ': <b>' + U.pct(acc(st.lam)) + '</b> (bigram alone ' + U.pct(acc(0)) + ', skip-gram alone ' + U.pct(acc(1)) + ')' +
        '<br><span class="note">Toy task and toy numbers. Scores are log-probabilities (higher is better). Skip-gram sees the whole sentence but ignores order; the bigram model sees order but only one word back, so their mistakes differ.</span>';
    }
    draw();
  };

  W.table7 = function (host) {
    Book.head(host, 'Table 7: the Microsoft Sentence Completion Challenge',
      '1040 sentences, 5 choices each, so guessing scores 20%. Grey: earlier methods; colour: the paper’s results.');
    const T7 = [['4-gram', 39], ['Average LSA\nsimilarity', 49], ['Log-bilinear\nmodel', 54.8], ['RNNLMs', 55.4], ['Skip-gram', 48.0], ['Skip-gram\n+ RNNLMs', 58.9]];
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    Book.chart.bars(chart, T7.map(function (r, i) { return { label: r[0], value: r[1], note: r[1] + '%', color: i >= 4 ? 'var(--accent)' : 'var(--muted)' }; }), { max: 70, height: 240, ylabel: 'accuracy [%]', mb: 50 });
    out.innerHTML = 'Skip-gram alone (48.0%) is below LSA (49%), but combined with the RNNLMs it reaches <b>58.9%</b>, 3.5 points above the previous best (55.4%). Dev part 59.2%, test part 58.7%. Paper Table 7, §4.5.';
  };

  // ------------------------------------------------------------------ ch-relations
  const T8 = [['France – Paris', 'Italy: Rome', 'Japan: Tokyo', 'Florida: Tallahassee'], ['big – bigger', 'small: larger', 'cold: colder', 'quick: quicker'],
    ['Miami – Florida', 'Baltimore: Maryland', 'Dallas: Texas', 'Kona: Hawaii'], ['Einstein – scientist', 'Messi: midfielder', 'Mozart: violinist', 'Picasso: painter'],
    ['Sarkozy – France', 'Berlusconi: Italy', 'Merkel: Germany', 'Koizumi: Japan'], ['copper – Cu', 'zinc: Zn', 'gold: Au', 'uranium: plutonium'],
    ['Berlusconi – Silvio', 'Sarkozy: Nicolas', 'Putin: Medvedev', 'Obama: Barack'], ['Microsoft – Windows', 'Google: Android', 'IBM: Linux', 'Apple: iPhone'],
    ['Microsoft – Ballmer', 'Google: Yahoo', 'IBM: McNealy', 'Apple: Jobs'], ['Japan – sushi', 'Germany: bratwurst', 'France: tapas', 'USA: pizza']];
  W.table8 = function (host) {
    Book.head(host, 'Table 8, and the toy’s own version',
      'Left: the paper’s examples (skip-gram, 783M words, 300 dimensions). Right: the same procedure on the toy: take the first pair of a relation as the example and apply it to every other first word.');
    const st = { r: 0 };
    const rels = W2V.relations();
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'toy relation', rels.map(function (r, i) { return { value: i, label: r.name }; }), st.r, function (v) { st.r = +v; draw(); });
    const box = el('div', { class: 'grid2' }, host), left = el('div', { class: 'scroll-x' }, box), right = el('div', null, box), out = el('div', { class: 'out' }, host);
    left.innerHTML = '<table><tr><th>relationship</th><th>ex. 1</th><th>ex. 2</th><th>ex. 3</th></tr>' + T8.map(function (r) { return '<tr><td><b>' + r[0] + '</b></td><td>' + r[1] + '</td><td>' + r[2] + '</td><td>' + r[3] + '</td></tr>'; }).join('') + '</table>';
    function draw() {
      const m = U.model(false), C = m.corpus, M = m.normalized(), rel = rels[st.r], ex = rel.pairs[0];
      let ok = 0;
      right.innerHTML = '<table><tr><th>' + ex[0] + ' – ' + ex[1] + '</th><th>toy answer</th></tr>' + rel.pairs.slice(1).map(function (p) {
        const r = W2V.analogy(M, C.words, C.index[ex[0]], C.index[ex[1]], C.index[p[0]], 1), good = r.top[0].w === p[1]; if (good) ok++;
        return '<tr><td>' + p[0] + '</td><td>' + r.top[0].w + ' ' + (good ? '✓' : '✗ (want ' + p[1] + ')') + '</td></tr>'; }).join('') + '</table>';
      out.innerHTML = 'toy: <b>' + ok + ' of ' + (rel.pairs.length - 1) + '</b> exact matches with one example pair.' +
        '<br><span class="note">Paper numbers: by exact match, Table 8 “would score only about 60%” (§5). Note its own errors, e.g. small: larger and uranium: plutonium. Toy results on the right.</span>';
    }
    draw();
  };

  W.avgOffset = function (host) {
    Book.head(host, 'Average several example pairs to define the relation',
      'Instead of one pair, average the offsets of k example pairs and add the result to the question word. Computed over every toy question for k = 1, 2, 3.');
    const st = { cbow: false };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'model', [{ value: 'sg', label: 'skip-gram' }, { value: 'cbow', label: 'CBOW' }], 'sg', function (v) { st.cbow = v === 'cbow'; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const m = U.model(st.cbow), C = m.corpus, M = m.normalized(), rels = W2V.relations(), res = [];
      for (let k = 1; k <= 3; k++) {
        let right = 0, n = 0;
        rels.forEach(function (rel) {
          const P = rel.pairs;
          P.forEach(function (target, t) {
            for (let s = 1; s < P.length; s++) {             // each starting offset gives a different set of k examples
              const ex = []; for (let j = 0; j < k; j++) ex.push(P[(t + s + j) % P.length]);
              if (ex.some(function (e) { return e === target; })) continue;
              const D = M[0].length, x = M[C.index[target[0]]].slice();
              ex.forEach(function (e) { for (let d = 0; d < D; d++) x[d] += (M[C.index[e[1]]][d] - M[C.index[e[0]]][d]) / k; });
              const skip = new Set([C.index[target[0]]]); ex.forEach(function (e) { skip.add(C.index[e[0]]); skip.add(C.index[e[1]]); });
              let best = -1, bw = -1; const xn = ML.normalize(x);
              for (let i = 0; i < M.length; i++) if (!skip.has(i)) { const c = ML.dot(xn, M[i]); if (c > best) { best = c; bw = i; } }
              n++; if (bw === C.index[target[1]]) right++;
            }
          });
        });
        res.push(right / n);
      }
      Book.chart.bars(chart, res.map(function (a, i) { return { label: 'k = ' + (i + 1), value: a, note: U.pct(a), color: 'var(--accent)' }; }), { max: 1, height: 200, ylabel: 'toy accuracy', tickFmt: function (v) { return Math.round(100 * v) + '%'; } });
      out.innerHTML = 'toy: one example ' + U.pct(res[0]) + ' → three examples ' + U.pct(res[2]) + '.' +
        '<br><span class="note">Toy numbers (the toy relations have only 4–5 pairs, so k stops at 3). The paper: ten examples instead of one improved its best models by about 10% absolute (§5).</span>';
    }
    draw();
  };

  W.oddOneOut = function (host) {
    Book.head(host, 'Which word does not belong?',
      'Average the (unit-length) vectors of a list of words, then find the word least similar to that average: the paper’s recipe for “odd one out” puzzles (§5).');
    const presets = [['paris', 'rome', 'berlin', 'queen', 'tokyo'], ['king', 'queen', 'prince', 'girl', 'sushi'], ['bigger', 'smaller', 'faster', 'taller', 'biggest'], ['france', 'italy', 'spain', 'madrid', 'japan'], ['cheese', 'pasta', 'sushi', 'paella', 'woman']];
    const st = { p: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'list', presets.map(function (p, i) { return { value: i, label: p.join(', ') }; }), 0, function (v) { st.p = +v; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const m = U.model(false), C = m.corpus, M = m.normalized(), ws = presets[st.p];
      const mean = ML.normalize(ws.reduce(function (a, w) { return ML.add(a, M[C.index[w]]); }, ML.zeros(m.D)));
      const cs = ws.map(function (w) { return ML.dot(mean, M[C.index[w]]); }), odd = ML.argmin(cs);
      Book.chart.bars(chart, ws.map(function (w, i) { return { label: w, value: Math.max(0, cs[i]), note: Book.fmt(cs[i], 2), color: i === odd ? 'var(--red)' : U.color(w) }; }), { max: 1, height: 200, ylabel: 'cosine with the average' });
      out.innerHTML = 'odd one out: <b>' + ws[odd] + '</b><br><span class="note">Toy skip-gram vectors.</span>';
    }
    draw();
  };

  // ------------------------------------------------------------------ ch-next
  W.stressTest = function (host) {
    Book.head(host, 'Starve the model: how much text does the toy need?',
      'Train skip-gram on fewer or more toy sentences and see the analogy accuracy and the king − man + woman answer. The paper’s whole argument is that more data, used cheaply, wins.');
    const st = { n: 300, D: 20 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'sentences', 50, 4800, 50, st.n, function (v) { st.n = v; }, function (v) { return v.toLocaleString(); });
    Book.slider(ctl, 'D', 2, 40, 1, st.D, function (v) { st.D = v; });
    Book.button(ctl, 'Train and test', function () { run(); }, 'primary');
    const out = el('div', { class: 'out' }, host);
    const hist = [];
    function run() {
      out.innerHTML = 'training…';
      setTimeout(function () {
        const corpus = W2V.makeCorpus({ sentences: st.n });
        const m = new W2V.Model({ corpus: corpus, cbow: false, D: st.D, window: 3, epochs: 6, seed: 1 }).trainAll();
        const ev = W2V.evaluate(m), M = m.normalized(), ix = corpus.index;
        const kq = (ix.man != null && ix.king != null && ix.woman != null) ? W2V.analogy(M, corpus.words, ix.man, ix.king, ix.woman, 1).top[0].w : '(words missing)';
        hist.unshift(st.n + ' sentences (' + corpus.T.toLocaleString() + ' words), D = ' + st.D + ': accuracy <b>' + U.pct(ev.acc) + '</b>, king − man + woman → <b>' + kq + '</b>');
        out.innerHTML = hist.slice(0, 6).join('<br>') + '<br><span class="note">Toy runs (6 epochs, window 3). Try 50, 300, 1,200 and 4,800 sentences.</span>';
      }, 20);
    }
    run();
  };
})();
