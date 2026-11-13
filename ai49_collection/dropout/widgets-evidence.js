/* Dropout Interactive Web Book — widgets for Part III (results, §6), Part IV (inside a dropout network, §7–§8)
   and Part V (beyond Bernoulli noise, §9–§11 and Appendix A). Paper numbers are typed from the JMLR version;
   values read off a figure are marked approximate. Uses Book, ML, NN, Toy and window.DOUI. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, Toy = window.Toy, U = window.DOUI;
  const P = Book.palette;

  // a results-table explorer: a select over named views, each a list of rows {label, value, hl}
  function explorer(host, title, desc, views, o) {
    o = o || {};
    Book.head(host, title, desc);
    const names = Object.keys(views); let cur = names[0];
    const ctl = el('div', { class: 'controls' }, host);
    if (names.length > 1) Book.select(ctl, o.selLabel || 'show', names, cur, function (v) { cur = v; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const v = views[cur];
      U.hbars(chart, v.rows.map(function (r) { return { label: r[0], value: r[1], note: r[1] == null ? null : (o.fmt ? o.fmt(r[1]) : String(r[1])), color: r[2] ? P[0] : 'var(--muted)', bold: r[2] === 2 }; }),
        { labelWidth: o.labelWidth || 360, max: v.max, tickFmt: o.tickFmt });
      out.innerHTML = v.note || '';
    }
    draw();
  }

  // ================================================================== PART III · RESULTS
  W.table1 = function (host) {
    const D = [['MNIST', 'Vision', 784, 60000, 10000], ['SVHN', 'Vision', 3072, 600000, 26000], ['CIFAR-10/100', 'Vision', 3072, 60000, 10000], ['ImageNet (ILSVRC-2012)', 'Vision', 65536, 1200000, 150000],
      ['TIMIT', 'Speech', 2520, 1100000, 58000], ['Reuters-RCV1', 'Text', 2000, 200000, 200000], ['Alternative Splicing', 'Genetics', 1014, 2932, 733]];
    Book.head(host, 'Table 1: seven data sets', 'The data sets of the paper (Table 1). Bars use a logarithmic scale (each grid line is 10× the previous). TIMIT counts frames, not recordings.');
    let cur = 'training set size';
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'show', ['training set size', 'test set size', 'input dimensionality'], cur, function (v) { cur = v; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const k = cur === 'training set size' ? 3 : cur === 'test set size' ? 4 : 2;
      U.hbars(chart, D.map(function (d) { return { label: d[0] + ' · ' + d[1], value: Math.log10(d[k]), note: d[k].toLocaleString('en-US'), color: P[['Vision', 'Speech', 'Text', 'Genetics'].indexOf(d[1])] }; }),
        { labelWidth: 250, max: 7, ticks: 7, tickFmt: function (v) { return v === 0 ? '1' : '10' + ['', '', '²', '³', '⁴', '⁵', '⁶', '⁷'][Math.round(v)]; } });
      out.innerHTML = 'Dimensionality (Table 1): MNIST 784 (28×28 grayscale) · SVHN and CIFAR 3072 (32×32 colour) · ImageNet 65536 (256×256 colour) · TIMIT 2520 (120-dim, 21 frames) · Reuters 2000 · Splicing 1014. Sizes as printed in Table 1 (K = thousand, M = million).';
    }
    draw();
  };

  // top-k on the toy dropout network, in the style of Paper Fig. 6
  W.topk = function (host) {
    Book.head(host, 'Top-1 and top-k error',
      'ImageNet has 1000 classes, so the paper also reports top-5 error: a test image counts as wrong only if the true label is not among the model’s five most probable labels (§6.1.4). Here the same idea on the toy dropout network, which has 10 classes, with top-1, top-2 and top-3. Bars show the four most probable labels, true label in green (like Paper Fig. 6).');
    const st = { start: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.button(ctl, 'Next four test digits', function () { st.start += 4; if (U._pairDone) draw(U._pairDone); }, 'primary');
    const row = el('div', { style: 'display:flex;flex-wrap:wrap;gap:14px' }, host), out = el('div', { class: 'out' }, host);
    function draw(Pr) {
      row.innerHTML = '';
      const te = Pr.data.test, probs = Pr.drop.net.probs(te.X);
      // pick test digits, preferring a mix of right and wrong ones
      const idx = []; for (let i = st.start; idx.length < 4; i++) { const j = (i * 37) % te.X.length; if (idx.length < 2 || ML.argmax(probs[j]) !== te.y[j] || i - st.start > 40) idx.push(j); }
      idx.forEach(function (j) {
        const card = el('div', { style: 'border:1px solid var(--line);border-radius:10px;padding:8px;width:150px;font-size:12px' }, row);
        U.digit(card, te.X[j], { px: 56, cell: 8 });
        el('div', { html: ' true label <b>' + te.y[j] + '</b>', style: 'display:inline-block;margin-left:6px' }, card);
        ML.topk(probs[j], 4).forEach(function (c) {
          const b = el('div', { style: 'display:flex;align-items:center;gap:6px;margin-top:3px' }, card);
          el('span', { text: c, style: 'width:12px;font-weight:700' }, b);
          el('span', { style: 'height:10px;border-radius:3px;background:' + (c === te.y[j] ? 'var(--green)' : 'var(--accent)') + ';width:' + Math.max(2, Math.round(100 * probs[j][c])) + 'px' }, b);
          el('span', { text: (100 * probs[j][c]).toFixed(0) + '%', style: 'color:var(--muted)' }, b);
        });
      });
      const err = [1, 2, 3].map(function (k) { let w = 0; probs.forEach(function (p, i) { if (ML.topk(p, k).indexOf(te.y[i]) < 0) w++; }); return 100 * w / probs.length; });
      out.innerHTML = 'toy dropout network on 1,000 test digits: top-1 error <b>' + U.pct(err[0]) + '</b> · top-2 <b>' + U.pct(err[1]) + '</b> · top-3 <b>' + U.pct(err[2]) + '</b>';
    }
    U.withPair(host, function (Pr) { U._pairDone = Pr; draw(Pr); });
  };

  W.table2 = function (host) {
    explorer(host, 'Table 2: the MNIST ladder', 'Test error (%) on MNIST in the permutation-invariant setting (no image structure used, no distorted copies). Top: networks trained from random weights; bottom: networks pretrained with RBM stacks (DBN) or a Deep Boltzmann Machine (DBM). Blue rows use dropout.', {
      'from random weights': { rows: [['Standard Neural Net (Simard et al., 2003) · Logistic · 2×800', 1.60], ['SVM Gaussian kernel', 1.40], ['Dropout NN · Logistic · 3×1024', 1.35, 1], ['Dropout NN · ReLU · 3×1024', 1.25, 1], ['Dropout NN + max-norm · ReLU · 3×1024', 1.06, 1], ['Dropout NN + max-norm · ReLU · 3×2048', 1.04, 1], ['Dropout NN + max-norm · ReLU · 2×4096', 1.01, 1], ['Dropout NN + max-norm · ReLU · 2×8192', 0.95, 1], ['Dropout NN + max-norm · Maxout · 2×(5×240) (Goodfellow 2013)', 0.94, 1]], max: 1.8,
        note: 'Each change helps: dropout 1.60 → 1.35, ReLU → 1.25, max-norm → 1.06, bigger nets → 0.95 (65 million parameters trained on 60,000 images, no early stopping; §6.1.1). Paper Table 2.' },
      'pretrained': { rows: [['DBN + finetuning (Hinton & Salakhutdinov, 2006) · 500-500-2000', 1.18], ['DBM + finetuning (Salakhutdinov & Hinton, 2009) · 500-500-2000', 0.96], ['DBN + dropout finetuning · 500-500-2000', 0.92, 1], ['DBM + dropout finetuning · 500-500-2000', 0.79, 2]], max: 1.8,
        note: 'Dropout finetuning improves both pretrained models; 0.79 % was “the best performance ever reported for the permutation invariant setting” at the time (§6.1.1). Paper Table 2.' }
    }, { fmt: function (v) { return v.toFixed(2) + '%'; }, tickFmt: function (v) { return v.toFixed(1); }, labelWidth: 400 });
  };

  W.fig4Toy = function (host) {
    Book.head(host, 'Figure 4, toy version: many architectures, one setting',
      'Paper Fig. 4 trains six MNIST architectures (2–4 hidden layers, 1024 or 2048 units) with and without dropout, all with the same hyperparameters, and finds two separate bands of test-error curves. Here: four toy architectures on 1,000 toy digits, each trained with and without dropout (p = 0.8 input, 0.5 hidden).');
    U.cached(host, 'fig4', function (R, view) {
      view.innerHTML = '';
      const plot = el('div', null, view), out = el('div', { class: 'out' }, view);
      Book.chart.lines(plot, R.map(function (r) { const d = /dropout/.test(r.label); return { color: d ? P[0] : P[1], marker: 'none', width: 1.6, points: r.curve.slice(1).map(function (c) { return [c[0], c[2]]; }) }; }),
        { width: 640, height: 250, xmin: 0, xmax: 4000, ymin: 5, ymax: 16, xticks: [0, 1000, 2000, 3000, 4000], xlabel: 'number of weight updates', ylabel: 'test error % (toy)' });
      out.innerHTML = R.map(function (r) { return r.label.padEnd(20) + ' final test error ' + U.pct(r.test) + ' · training error ' + U.pct(r.train); }).join('\n') +
        '\n<span class="note">Red: without dropout, blue: with dropout. In the paper the bands are far apart (≈1.6–1.9 % vs ≈1.0–1.2 %, read off Fig. 4). On the toy the bands are close but every dropout run ends below its standard twin.</span>';
    }, 25);
  };

  W.table3 = function (host) {
    explorer(host, 'Table 3: Street View House Numbers', 'Test error (%) on SVHN. Blue rows: the paper’s dropout conv nets (and maxout, which also uses dropout).', {
      all: { rows: [['Binary Features (WDCH)', 36.7], ['HOG', 15.0], ['Stacked Sparse Autoencoders', 10.3], ['KMeans', 9.4], ['Multi-stage Conv Net, average pooling', 9.06], ['Multi-stage Conv Net + L2 pooling', 5.36], ['Multi-stage Conv Net + L4 pooling + padding', 4.90], ['Conv Net + max-pooling', 3.95], ['Conv Net + max pooling + dropout in fully connected layers', 3.02, 1], ['Conv Net + stochastic pooling (Zeiler & Fergus, 2013)', 2.80], ['Conv Net + max pooling + dropout in all layers', 2.55, 1], ['Conv Net + maxout (Goodfellow et al., 2013)', 2.47, 2], ['Human performance', 2.0]],
        note: 'No dropout 3.95 % → dropout in the fully connected layers 3.02 % → dropout in all layers 2.55 % (§6.1.2). Paper Table 3.' }
    }, { fmt: function (v) { return (Math.abs(v * 10 - Math.round(v * 10)) < 1e-9 ? v.toFixed(1) : v.toFixed(2)) + '%'; }, labelWidth: 400 });
  };
  W.table4 = function (host) {
    explorer(host, 'Table 4: CIFAR-10 and CIFAR-100', 'Test error (%), no data augmentation. Blue rows use dropout.', {
      'CIFAR-10': { rows: [['Conv Net + max pooling (hand tuned)', 15.60], ['Conv Net + stochastic pooling', 15.13], ['Conv Net + max pooling (Snoek et al., 2012)', 14.98], ['Conv Net + max pooling + dropout fully connected layers', 14.32, 1], ['Conv Net + max pooling + dropout in all layers', 12.61, 1], ['Conv Net + maxout (Goodfellow et al., 2013)', 11.68, 2]], max: 18, note: 'Bayesian hyperparameter search (Snoek et al.) reached 14.98 %; dropout in every layer 12.61 %; maxout 11.68 % (§6.1.3, Table 4).' },
      'CIFAR-100': { rows: [['Conv Net + max pooling (hand tuned)', 43.48], ['Conv Net + stochastic pooling', 42.51], ['Conv Net + max pooling (Snoek et al., 2012)', null], ['Conv Net + max pooling + dropout fully connected layers', 41.26, 1], ['Conv Net + max pooling + dropout in all layers', 37.20, 2], ['Conv Net + maxout (Goodfellow et al., 2013)', 38.57, 1]], max: 50, note: '43.48 % → 37.20 % with dropout in all layers, “a huge improvement” (§6.1.3); on CIFAR-100 the plain dropout net beats maxout. Paper Table 4.' }
    }, { selLabel: 'data set', fmt: function (v) { return v.toFixed(2) + '%'; }, labelWidth: 400 });
  };
  W.layerP = function (host) {
    Book.head(host, 'A different p for every layer',
      'The SVHN network keeps units with probability p = (0.9, 0.75, 0.75, 0.5, 0.5, 0.5) going from the input through the three convolutional layers to the two fully connected layers (§6.1.2, App. B.2). Bars: expected fraction of units kept, and the noise each layer adds, measured as the variance (1 − p)/p of the multiplier 1/p·r.');
    const L = [['input (3 colour channels)', 0.9], ['conv1 (96 filters)', 0.75], ['conv2 (128 filters)', 0.75], ['conv3 (256 filters)', 0.5], ['fc1 (2048 units)', 0.5], ['fc2 (2048 units)', 0.5]];
    const st = { show: 'kept' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'show', [{ value: 'kept', label: 'fraction kept p' }, { value: 'var', label: 'noise (1 − p)/p' }], st.show, function (v) { st.show = v; draw(); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      U.hbars(chart, L.map(function (l, i) { const v = st.show === 'kept' ? l[1] : (1 - l[1]) / l[1]; return { label: l[0] + ' → p = ' + l[1], value: v, note: v.toFixed(2), color: i < 4 && i > 0 ? P[5] : P[0] }; }), { labelWidth: 290, max: st.show === 'kept' ? 1 : 1.2 });
      out.innerHTML = 'Why drop in conv layers at all? They have few parameters (Chapter 7), so they hardly overfit. The paper’s explanation: dropout there “provides noisy inputs for the higher fully connected layers which prevents them from overfitting” (§6.1.2). Inputs are dropped least: 10 % of the pixels.';
    }
    draw();
  };
  W.table56 = function (host) {
    explorer(host, 'Tables 5 and 6: ImageNet', 'Error (%) on ImageNet. ILSVRC-2010 has public test labels (Table 5); ILSVRC-2012 was the competition, scored on hidden test labels (Table 6). The conv net with dropout is Krizhevsky et al. (2012).', {
      'ILSVRC-2010 top-5 (Table 5)': { rows: [['Sparse Coding (Lin et al., 2010)', 28.2], ['SIFT + Fisher Vectors (Sanchez & Perronnin, 2011)', 25.7], ['Conv Net + dropout (Krizhevsky et al., 2012)', 17.0, 2]], max: 50, note: 'Top-1 on the same test set: 47.1 %, 45.7 % and 37.5 % (Table 5).' },
      'ILSVRC-2010 top-1 (Table 5)': { rows: [['Sparse Coding (Lin et al., 2010)', 47.1], ['SIFT + Fisher Vectors (Sanchez & Perronnin, 2011)', 45.7], ['Conv Net + dropout (Krizhevsky et al., 2012)', 37.5, 2]], max: 50, note: 'Paper Table 5.' },
      'ILSVRC-2012 top-5 test (Table 6)': { rows: [['SVM on Fisher Vectors of Dense SIFT and Color Statistics', 27.3], ['Avg of classifiers over FVs of SIFT, LBP, GIST and CSIFT', 26.2], ['Avg of 5 Conv Nets + dropout (Krizhevsky et al., 2012)', 16.4, 2]], max: 50, note: 'Standard vision features ≈ 26 % vs ≈ 16 % for conv nets with dropout, “a staggering difference”; this entry won ILSVRC-2012 (§6.1.4).' },
      'ILSVRC-2012 validation (Table 6)': { rows: [['Conv Net + dropout · top-1 (val)', 40.7, 1], ['Conv Net + dropout · top-5 (val)', 18.2, 1], ['Avg of 5 Conv Nets + dropout · top-1 (val)', 38.1, 1], ['Avg of 5 Conv Nets + dropout · top-5 (val)', 16.4, 2]], max: 50, note: 'Averaging five dropout nets (an ensemble of ensembles) still helps: 18.2 → 16.4 top-5. Paper Table 6.' }
    }, { selLabel: 'table', fmt: function (v) { return v.toFixed(1) + '%'; }, labelWidth: 400 });
  };
  W.table7 = function (host) {
    explorer(host, 'Table 7: TIMIT phone error rate', 'Phone error rate (%) on the TIMIT core test set. Blue rows use dropout.', {
      all: { rows: [['NN (6 layers) (Mohamed et al., 2010)', 23.4], ['Dropout NN (6 layers)', 21.8, 1], ['DBN-pretrained NN (4 layers)', 22.7], ['DBN-pretrained NN (6 layers) (Mohamed et al., 2010)', 22.4], ['DBN-pretrained NN (8 layers) (Mohamed et al., 2010)', 20.7], ['mcRBM-DBN-pretrained NN (5 layers) (Dahl et al., 2010)', 20.5], ['DBN-pretrained NN (4 layers) + dropout', 19.7, 2], ['DBN-pretrained NN (8 layers) + dropout', 19.7, 2]], max: 25,
        note: 'Without pretraining: 23.4 → 21.8. With pretraining: 4 layers 22.7 → 19.7; 8 layers 20.7 → 19.7 in the table (the text of §6.2 says “from 20.5 %”, which is the 5-layer mcRBM row). Paper Table 7.' }
    }, { fmt: function (v) { return v.toFixed(1) + '%'; }, labelWidth: 420 });
  };
  // code quality (App. B.6) for one tissue and one input
  W.codeQuality = function (host) {
    Book.head(host, 'Code quality, the splicing score',
      'For each exon and tissue the model predicts a distribution over 3 states (inc, exc, nc). Code quality adds up p·log₂(q / p̄): the target probability p times how many bits better the prediction q is than always predicting the average p̄ (App. B.6). Higher is better; predicting p̄ scores 0. Drag the prediction.');
    const tgt = [0.7, 0.2, 0.1], avg = [0.35, 0.35, 0.3], q = [0.5, 0.3, 0.2];
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'q(inc)', 0.01, 0.98, 0.01, q[0], function (v) { const rest = 1 - v, o = q[1] + q[2]; q[1] = rest * q[1] / o; q[2] = rest * q[2] / o; q[0] = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.slider(ctl, 'q(exc) share of the rest', 0.01, 0.99, 0.01, q[1] / (q[1] + q[2]), function (v) { const rest = 1 - q[0]; q[1] = rest * v; q[2] = rest * (1 - v); draw(); }, function (v) { return v.toFixed(2); });
    const chart = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const names = ['inc', 'exc', 'nc'], terms = names.map(function (n, s) { return tgt[s] * Math.log2(q[s] / avg[s]); }), cq = ML.sum(terms);
      const kl = ML.sum(tgt.map(function (p, s) { return p * Math.log2(p / q[s]); }));
      const data = []; names.forEach(function (n, s) { data.push({ label: n + '\ntarget', value: tgt[s], note: tgt[s].toFixed(2), color: P[2] }); data.push({ label: n + '\npredicted', value: q[s], note: q[s].toFixed(2), color: P[0] }); data.push({ label: n + '\naverage', value: avg[s], note: avg[s].toFixed(2), color: 'var(--muted)' }); });
      Book.chart.bars(chart, data, { max: 1, mb: 50, tickFmt: function (v) { return v.toFixed(2); } });
      out.innerHTML = 'terms p·log₂(q/p̄): ' + terms.map(function (t, s) { return names[s] + ' ' + U.f(t, 3); }).join(', ') + '\ncode quality for this one case = <b>' + U.f(cq, 3) + ' bits</b>' +
        '\nKL(target ‖ predicted) = ' + U.f(kl, 3) + ' bits; code quality = (bits of the average) − KL, so a smaller KL means a higher score.' +
        '\n<span class="note">Toy numbers for one exon and one tissue. The paper sums over all exons and the 4 tissue types; Table 8 reports the totals in bits.</span>';
    }
    draw();
  };
  W.table8 = function (host) {
    explorer(host, 'Table 8: alternative splicing, dropout vs a Bayesian net', 'Code quality in bits (higher is better) on the alternative splicing data, averaged over the 5 folds of Xiong et al. (2011).', {
      all: { rows: [['Neural Network (early stopping) (Xiong et al., 2011)', 440], ['Regression, PCA (Xiong et al., 2011)', 463], ['SVM, PCA (Xiong et al., 2011)', 487], ['Neural Network with dropout', 567, 1], ['Bayesian Neural Network (Xiong et al., 2011)', 623, 2]], max: 700,
        note: 'Dropout beats everything except the Bayesian net, which averages networks weighted by how well they fit (the “correct” averaging) but is slow and stays small (a few tens of units, vs 1000s in the dropout net). Paper Table 8, §6.4.' }
    }, { fmt: function (v) { return v + ' bits'; }, labelWidth: 400 });
  };
  W.table9 = function (host) {
    explorer(host, 'Table 9: dropout against other regularizers', 'MNIST test error (%), the same 784-1024-1024-2048-10 ReLU network, each regularizer tuned on a validation set (§6.5).', {
      all: { rows: [['L2', 1.62], ['L2 + L1 applied towards the end of training', 1.60], ['L2 + KL-sparsity', 1.55], ['Max-norm', 1.35], ['Dropout + L2', 1.25, 1], ['Dropout + Max-norm', 1.05, 2]], max: 1.8, note: 'Dropout + max-norm gives the lowest error. Max-norm alone already beats the weight penalties. Paper Table 9.' }
    }, { fmt: function (v) { return v.toFixed(2) + '%'; }, tickFmt: function (v) { return v.toFixed(1); } });
  };
  W.regToy = function (host) {
    Book.head(host, 'Table 9, toy version', 'The book’s base toy network (49-96-96-10, 500 training digits, 3,000 steps) with each regularizer. Strengths were set by hand for the toy, not tuned on a validation set. KL-sparsity is left out.');
    U.cached(host, 'table9', function (R, view) {
      view.innerHTML = '';
      const chart = el('div', null, view), out = el('div', { class: 'out' }, view);
      U.hbars(chart, R.map(function (r) { return { label: r.label, value: r.test, note: U.pct(r.test), color: /Dropout/.test(r.label) ? P[0] : 'var(--muted)' }; }), { labelWidth: 220, max: 14 });
      out.innerHTML = R.map(function (r) { return r.label.padEnd(20) + ' training error ' + U.pct(r.train); }).join('\n') + '\n<span class="note">Test error = mean of the last three checks, toy. Settings: L2 λ = 0.001, L1 λ = 0.0001, max-norm c = 2, dropout p = 0.8 / 0.5.</span>';
    }, 20);
  };

  // ================================================================== PART IV · INSIDE A DROPOUT NETWORK
  // two autoencoders (49 → 48 ReLU → 49), without and with dropout on the hidden layer, trained once per page
  const aeWait = []; let aeState = null;
  U.aes = function (cb, prog) {
    if (aeState && aeState.done) { cb(aeState); return; }
    aeWait.push([cb, prog]);
    if (aeState) return;
    aeState = { done: false };
    const X = Toy.digits(2000, 9191).X.concat(Toy.data(500).train.X), r = ML.rng(4), nets = [new Toy.AE(48, 1, 7), new Toy.AE(48, 0.5, 7)], steps = 3000;
    let t = 0;
    (function tick() {
      const t0 = performance.now();
      while (t < 2 * steps && performance.now() - t0 < 30) { const b = []; for (let i = 0; i < 20; i++) b.push(X[r.int(X.length)]); nets[t < steps ? 0 : 1].step(b); t++; }
      aeWait.forEach(function (w) { if (w[1]) w[1](t / (2 * steps)); });
      if (t < 2 * steps) { setTimeout(tick, 0); return; }
      const te = Toy.data(500).test.X.slice(0, 200);
      aeState = { done: true, nets: nets, te: te, err: nets.map(function (n) { return n.recError(Toy.data(500).test.X); }) };
      aeWait.splice(0).forEach(function (w) { w[0](aeState); });
    })();
  };
  function withAE(host, ready) {
    const msg = el('div', { class: 'out', text: 'Training two autoencoders on the toy digits (one time)…' }, host);
    U.aes(function (A) { msg.remove(); ready(A); }, function (f) { msg.textContent = 'Training two autoencoders on the toy digits (one time, in your browser): ' + Math.round(100 * f) + '%'; });
  }
  W.aeFeatures = function (host) {
    Book.head(host, 'Figure 7, toy version: features with and without dropout',
      'Two autoencoders with one hidden layer of 48 ReLU units learn to reproduce the toy digits (49 → 48 → 49). One drops hidden units with p = 0.5. Each small picture shows one hidden unit’s 49 incoming weights (blue +, red −): the pattern that unit responds to. Paper Fig. 7 does this on MNIST with 256 units.');
    const box = el('div', { class: 'grid2' }, host), out = el('div', { class: 'out' }, host);
    withAE(host, function (A) {
      A.nets.forEach(function (n, k) {
        const col = el('div', null, box); el('div', { html: '<b>' + (k ? 'Dropout, p = 0.5' : 'Without dropout') + '</b>', style: 'margin-bottom:4px' }, col);
        const g = el('div', { style: 'display:grid;grid-template-columns:repeat(8,auto);gap:3px;justify-content:start' }, col);
        for (let j = 0; j < 48; j++) U.digit(g, n.feature(j), { px: 30, cell: 6, gap: 0, signed: true });
      });
      out.innerHTML = 'test reconstruction error (squared, per image): without dropout <b>' + U.f(A.err[0], 3) + '</b> · with dropout <b>' + U.f(A.err[1], 3) + '</b> (toy)' +
        '\n<span class="note">Honest note: on a 7×7 toy both grids look noisy, and the difference in the pictures is hard to see by eye; the measurable effect here is the sparsity in the next widget. Also unlike the paper’s pair, the toy dropout autoencoder reconstructs clearly worse: 48 units without dropout can almost copy 49 pixels, while half of them missing cannot. The paper (MNIST, 256 units): without dropout the units co-adapt and each alone means little; with dropout they detect edges, strokes and spots (§7.1).</span>';
    });
  };
  W.sparsity = function (host) {
    Book.head(host, 'Figure 8, toy version: dropout gives sparse codes',
      'The same two autoencoders, fed 200 test digits. Left of each pair: histogram of every hidden unit’s mean activation over those digits. Right: histogram of all 200 × 48 activations. A sparse code has a tall spike at 0 and only a few large values.');
    const box = el('div', { class: 'grid2' }, host), out = el('div', { class: 'out' }, host);
    withAE(host, function (A) {
      const stats = A.nets.map(function (n, k) {
        const H = n.hidden(A.te), means = ML.range(48).map(function (j) { return ML.mean(H.map(function (r) { return r[j]; })); }), all = [].concat.apply([], H);
        const col = el('div', null, box); el('div', { html: '<b>' + (k ? 'Dropout, p = 0.5' : 'Without dropout') + '</b>' }, col);
        const mx = Math.max(1e-6, Math.max.apply(null, all));
        [[means, 'mean activation of a unit'], [all, 'activation']].forEach(function (d) {
          const nb = 16, cnt = new Array(nb).fill(0); d[0].forEach(function (v) { cnt[Math.min(nb - 1, Math.floor(v / mx * nb))]++; });
          Book.chart.bars(el('div', null, col), cnt.map(function (c, i) { return { label: i % 4 ? '' : (i * mx / nb).toFixed(1), value: c / d[0].length, note: '', color: k ? P[0] : P[1] }; }), { width: 320, height: 150, ylabel: 'fraction', mb: 30, tickFmt: function (v) { return v.toFixed(2); } });
          el('div', { class: 'note', text: d[1] }, col);
        });
        return { mean: ML.mean(all), zero: 100 * all.filter(function (v) { return v < 1e-9; }).length / all.length };
      });
      out.innerHTML = 'overall mean activation: without dropout <b>' + U.f(stats[0].mean, 3) + '</b> · with dropout <b>' + U.f(stats[1].mean, 3) + '</b>\nactivations exactly 0: without ' + U.pct(stats[0].zero) + ' · with ' + U.pct(stats[1].zero) + ' (toy)' +
        '\n<span class="note">Paper Fig. 8 (MNIST, 256 ReLUs): mean activation ≈ 2.0 without dropout, ≈ 0.7 with, and a sharp peak at zero (§7.2). The toy’s scale differs; compare the shapes.</span>';
    });
  };

  // Fig 9 (values read off the paper's plot: approximate)
  const FIG9 = { p: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    a: { test: [3.2, 1.8, 1.2, 1.12, 1.07, 1.1, 1.1, 1.08, 1.25, 1.62], train: [2.72, 0.65, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01] },
    b: { test: [2.4, 1.78, 1.5, 1.27, 1.22, 1.1, 1.2, 1.42, 1.45, 1.7], train: [1.7, 0.88, 0.48, 0.1, 0.07, 0.02, 0.01, 0.01, 0.01, 0.01] } };
  W.fig9 = function (host) {
    Book.head(host, 'Figure 9: error as a function of p (paper, approximate)',
      'MNIST, 784-2048-2048-2048-10, no input dropout. (a) The same n units per layer for every p. (b) n changed so that p·n stays fixed (256, 256, 512 units kept on average). Values read off Paper Fig. 9, so approximate.');
    let cur = 'a';
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'panel', [{ value: 'a', label: '(a) keeping n fixed' }, { value: 'b', label: '(b) keeping pn fixed' }], cur, function (v) { cur = v; draw(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const d = FIG9[cur];
      Book.chart.lines(plot, [{ name: 'test', color: P[2], points: FIG9.p.map(function (p, i) { return [p, d.test[i]]; }) }, { name: 'training', color: P[0], points: FIG9.p.map(function (p, i) { return [p, d.train[i]]; }) }],
        { width: 640, height: 240, xmin: 0, xmax: 1, ymin: 0, ymax: 3.5, yticks: 7, xticks: [0, 0.2, 0.4, 0.6, 0.8, 1], xlabel: 'probability of retaining a unit, p', ylabel: 'classification error %', mr: 70 });
      out.innerHTML = cur === 'a' ? 'Small p: few units are on during training, so even the training error is high: underfitting. Test error is flat for 0.4 ≤ p ≤ 0.8 and rises as p → 1 (no dropout) (§7.3).'
        : 'With p·n fixed, small p means a huge network: the error at p = 0.1 falls from 2.7 % to 1.7 % compared with (a) (§7.3 text; the plot in (a) shows ≈ 3.2 % at p = 0.1). Best near p = 0.6; the default 0.5 is close.';
    }
    draw();
  };
  W.rateToy = function (host) {
    Book.head(host, 'Figure 9, toy version', 'The toy network with hidden-unit retention p from 0.1 to 1 (no input dropout). (a) 96 units per hidden layer for every p. (b) 24/p units per hidden layer, so that p·n = 24 units are kept on average (p = 0.1 → 240 units).');
    let cur = 'a', R = null;
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'panel', [{ value: 'a', label: '(a) n fixed' }, { value: 'b', label: '(b) pn fixed' }], cur, function (v) { cur = v; if (R) draw(); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const rows = R.filter(function (r) { return r.label.indexOf(cur === 'a' ? 'n fixed' : 'pn fixed') === 0; }), ps = rows.map(function (r) { return +r.label.split('=')[1]; });
      Book.chart.lines(plot, [{ name: 'test', color: P[2], points: rows.map(function (r, i) { return [ps[i], r.test]; }) }, { name: 'training', color: P[0], points: rows.map(function (r, i) { return [ps[i], r.train]; }) }],
        { width: 640, height: 240, xmin: 0, xmax: 1, ymin: 0, ymax: 40, yticks: 4, xticks: [0, 0.2, 0.4, 0.6, 0.8, 1], xlabel: 'p (hidden units)', ylabel: 'error % (toy)', mr: 70 });
      let bi = 0; rows.forEach(function (r, i) { if (r.test < rows[bi].test) bi = i; });
      out.innerHTML = rows.map(function (r, i) { return 'p = ' + ps[i].toFixed(1) + ': test ' + U.pct(r.test) + ', training ' + U.pct(r.train); }).join('\n') + '\nlowest test error at p = ' + ps[bi] + ' (toy)';
    }
    U.cached(host, 'fig9', function (res) { R = res; draw(); }, 120);
  };
  const FIG10 = { n: [100, 500, 1000, 5000, 10000, 50000], with: [26.5, 13, 10.7, 3, 2.3, 1.3], without: [22.5, 13, 10.4, 4.8, 3.3, 1.7] };
  W.fig10 = function (host) {
    Book.head(host, 'Figure 10: error as a function of data set size (paper, approximate)',
      'MNIST subsets of 100 to 50,000 images, the same 784-1024-1024-2048-10 network, dropout p = 0.5 hidden, 0.8 input. Values read off Paper Fig. 10 (log scale), so approximate. Slide to compare at each size.');
    let k = 3;
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'data set size', 0, 5, 1, k, function (v) { k = v; draw(); }, function (v) { return FIG10.n[v].toLocaleString('en-US'); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const r = Book.chart.lines(plot, [{ name: 'with dropout', color: P[0], points: FIG10.n.map(function (n, i) { return [n, FIG10.with[i]]; }) }, { name: 'without', color: P[1], points: FIG10.n.map(function (n, i) { return [n, FIG10.without[i]]; }) }],
        { width: 640, height: 240, logx: true, xmin: 100, xmax: 50000, ymin: 0, ymax: 30, xticks: [100, 500, 1000, 5000, 10000, 50000], xfmt: function (v) { return v >= 1000 ? v / 1000 + 'K' : v; }, xlabel: 'data set size (log scale)', ylabel: 'classification error %', mr: 100 });
      const x = r.X(FIG10.n[k]); S('line', { x1: x, x2: x, y1: 16, y2: 218, stroke: 'var(--muted)', 'stroke-dasharray': '3 3' }, r.svg);
      const gain = FIG10.without[k] - FIG10.with[k];
      out.innerHTML = FIG10.n[k].toLocaleString('en-US') + ' images: with dropout ≈ ' + FIG10.with[k] + ' %, without ≈ ' + FIG10.without[k] + ' % → dropout ' + (gain > 0.05 ? 'helps by ≈ ' + gain.toFixed(1) + ' points' : gain < -0.05 ? 'hurts by ≈ ' + (-gain).toFixed(1) + ' points' : 'makes no clear difference') +
        '\n<span class="note">For 100 and 500 images dropout gives no improvement; the gain grows up to a point, then shrinks: a “sweet spot” (§7.4).</span>';
    }
    draw();
  };
  W.sizeToy = function (host) {
    Book.head(host, 'Figure 10, toy version', 'The toy network trained on 50 to 2,000 toy digits, with and without dropout (p = 0.8 input, 0.5 hidden), always 3,000 steps of minibatch 20.');
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    U.cached(host, 'fig10', function (R) {
      const pts = function (d) { return R.filter(function (r) { return /dropout/.test(r.label) === d; }).map(function (r) { return [+r.label.split(' ')[0], r.test]; }); };
      const a = pts(true), b = pts(false);
      Book.chart.lines(plot, [{ name: 'with dropout', color: P[0], points: a }, { name: 'without', color: P[1], points: b }],
        { width: 640, height: 240, logx: true, xmin: 50, xmax: 2000, ymin: 0, xticks: [50, 100, 200, 500, 1000, 2000], xlabel: 'training-set size (log scale)', ylabel: 'test error % (toy)', mr: 100 });
      out.innerHTML = a.map(function (p, i) { return String(p[0]).padStart(5) + ' digits: with ' + U.pct(p[1]) + ' · without ' + U.pct(b[i][1]) + ' · difference ' + U.f(b[i][1] - p[1], 1); }).join('\n');
    }, 60);
  };

  // Monte-Carlo averaging vs weight scaling on the toy dropout net
  W.mcToy = function (host) {
    Book.head(host, 'Figure 11, toy version: Monte-Carlo averaging vs weight scaling',
      'The trained toy dropout network classifies 400 test digits in two ways. Weight scaling: one pass with all units (the horizontal line). Monte-Carlo: run k randomly thinned networks and average their predicted probabilities. Repeated 3 times for each k.');
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    U.withPair(host, function (Pr) {
      const net = Pr.drop.net, sub = { X: Pr.data.test.X.slice(0, 400), y: Pr.data.test.y.slice(0, 400) }, K = 100, reps = 3;
      const ws = net.error(sub), Xt = NN.tensor(sub.X), errs = ML.zeros(K);
      let rep = 0, k = 0, acc = null; const rng = ML.rng(77);
      out.textContent = 'sampling thinned networks…';
      (function tick() {
        const t0 = performance.now();
        while (rep < reps && performance.now() - t0 < 30) {
          if (k === 0) acc = sub.X.map(function () { return ML.zeros(10); });
          const lg = net.forward(Xt, { train: true, rng: rng });
          let w = 0;
          for (let i = 0; i < sub.X.length; i++) { const q = ML.softmax(lg.row(i)); for (let c = 0; c < 10; c++) acc[i][c] += q[c]; if (ML.argmax(acc[i]) !== sub.y[i]) w++; }
          errs[k] += 100 * w / sub.X.length / reps;
          k++; if (k === K) { k = 0; rep++; }
        }
        if (rep < reps) { out.textContent = 'sampling thinned networks: repeat ' + (rep + 1) + ' of ' + reps + ', k = ' + k; setTimeout(tick, 0); return; }
        const pts = errs.map(function (e, i) { return [i + 1, e]; });
        Book.chart.lines(plot, [{ name: 'Monte-Carlo, k samples', color: P[0], marker: 'none', points: pts }, { name: 'weight scaling', color: P[1], marker: 'none', dash: '6 4', points: [[1, ws], [K, ws]] }],
          { width: 640, height: 240, xmin: 1, xmax: K, ymin: 0, xticks: [1, 20, 40, 60, 80, 100], xlabel: 'number of thinned networks averaged, k', ylabel: 'test error % (toy)', mr: 150 });
        let match = K; for (let i = 0; i < K; i++) if (errs[i] <= ws) { match = i + 1; break; }
        out.innerHTML = 'weight scaling: <b>' + U.pct(ws) + '</b> · Monte-Carlo with k = 1: ' + U.pct(errs[0]) + ', k = 10: ' + U.pct(errs[9]) + ', k = 50: ' + U.pct(errs[49]) + ', k = 100: ' + U.pct(errs[99]) +
          '\nfirst k where Monte-Carlo is as good as weight scaling: ' + (match < K ? '<b>' + match + '</b>' : 'not reached by 100') + ' (toy)' +
          '\n<span class="note">Paper Fig. 11 (MNIST): around k = 50 Monte-Carlo matches weight scaling (≈ 1.05 %, read off), then is slightly better but within one standard deviation (§7.5). Weight scaling costs one pass; Monte-Carlo costs k.</span>';
      })();
    });
  };

  // ---------- ch-rbm: enumerate a tiny RBM
  W.rbmEnumerate = function (host) {
    Book.head(host, 'A tiny RBM, every state counted',
      'An RBM with 3 visible and 2 hidden binary units has 2⁵ = 32 joint states (v, h). Each state gets an energy E = −(vᵀWh + aᵀh + bᵀv) and probability e^(−E)/Z, where Z adds e^(−E) over all 32 states (§8.1). Change three parameters and check the conditional formula by brute force.');
    const Wm = [[1.5, -0.5], [1.0, 1.2], [-1.0, 0.8]], a = [-0.5, 0.2], b = [0.3, -0.2, 0.1];
    const st = { dropH2: false };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'W₁₁', -3, 3, 0.1, Wm[0][0], function (v) { Wm[0][0] = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'hidden bias a₁', -3, 3, 0.1, a[0], function (v) { a[0] = v; draw(); }, function (v) { return v.toFixed(1); });
    Book.slider(ctl, 'visible bias b₁', -3, 3, 0.1, b[0], function (v) { b[0] = v; draw(); }, function (v) { return v.toFixed(1); });
    const tg = Book.button(ctl, 'Drop h₂ (r₂ = 0): off', function () { st.dropH2 = !st.dropH2; tg.textContent = 'Drop h₂ (r₂ = 0): ' + (st.dropH2 ? 'on' : 'off'); draw(); });
    const box = el('div', { class: 'grid2' }, host), tb = el('div', { style: 'max-height:320px;overflow:auto' }, box), out = el('div', { class: 'out' }, box);
    function draw() {
      const states = [];
      for (let m = 0; m < 32; m++) {
        const v = [m & 1, (m >> 1) & 1, (m >> 2) & 1], h = [(m >> 3) & 1, (m >> 4) & 1];
        if (st.dropH2 && h[1]) continue;                           // g(h₂, r₂ = 0) = 0 unless h₂ = 0
        let e = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) e -= v[i] * Wm[i][j] * h[j];
        e -= a[0] * h[0] + a[1] * h[1] + b[0] * v[0] + b[1] * v[1] + b[2] * v[2];
        states.push({ v: v, h: h, e: e, u: Math.exp(-e) });
      }
      const Z = ML.sum(states.map(function (s) { return s.u; }));
      states.sort(function (x, y) { return x.e - y.e; });
      tb.innerHTML = '';
      U.table(tb, [['v', 'h', 'energy E', 'P(v, h)']].concat(states.map(function (s) { return [s.v.join(''), s.h.join(''), s.e.toFixed(2), (s.u / Z).toFixed(4)]; })));
      const vq = [1, 0, 1];
      const match = states.filter(function (s) { return s.v.join('') === vq.join(''); }), pv = ML.sum(match.map(function (s) { return s.u; }));
      const ph1 = ML.sum(match.filter(function (s) { return s.h[0] === 1; }).map(function (s) { return s.u; })) / pv;
      const formula = ML.sigmoid(a[0] + Wm[0][0] * vq[0] + Wm[1][0] * vq[1] + Wm[2][0] * vq[2]);
      const ph2 = ML.sum(match.filter(function (s) { return s.h[1] === 1; }).map(function (s) { return s.u; })) / pv;
      out.innerHTML = 'states: ' + states.length + ' · Z = Σ e^(−E) = <b>' + U.f(Z, 3) + '</b>\n\nfor v = (1, 0, 1):\nP(h₁ = 1 | v) by counting states = <b>' + U.f(ph1, 4) + '</b>\nσ(a₁ + Σᵢ Wᵢ₁ vᵢ) = <b>' + U.f(formula, 4) + '</b>\nP(h₂ = 1 | v) = ' + U.f(ph2, 4) + (st.dropH2 ? '  (r₂ = 0 forces h₂ = 0)' : '') +
        '\n\n<span class="note">Listed by energy: low energy = high probability. The conditional formula holds whatever the other units do, which is why the hidden units can be sampled all at once.</span>';
    }
    draw();
  };
  // ---------- ch-rbm: one CD-1 step
  W.cd1Step = function (host) {
    Book.head(host, 'Contrastive divergence (CD-1), step by step',
      'An RBM with 16 hidden units learns one binarised toy digit. A CD-1 update: from the data v compute P(h | v) and sample h; reconstruct P(v′ | h); compute P(h′ | v′). Then W changes by ε (v·P(h|v)ᵀ − v′·P(h′|v′)ᵀ). Press “CD-1 update” a few times and watch the reconstruction approach the data.');
    const st = { k: 3, n: 0, rbm: null, keep: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'digit', ML.range(10).map(String), String(st.k), function (v) { st.k = +v; reset(); });
    Book.select(ctl, 'hidden dropout', [{ value: '1', label: 'off' }, { value: '0.5', label: 'p = 0.5' }], '1', function (v) { st.keep = +v; reset(); });
    Book.button(ctl, 'CD-1 update', function () { upd(1); }, 'primary');
    Book.button(ctl, '20 updates', function () { upd(20); });
    const row = el('div', { style: 'display:flex;flex-wrap:wrap;gap:14px;align-items:center' }, host), out = el('div', { class: 'out' }, host);
    let v;
    function reset() { st.rbm = new Toy.RBM(16, st.keep, 3); st.n = 0; v = Toy.binarize([Toy.sample(st.k, ML.rng(5))], 2)[0]; draw(); }
    function upd(n) { for (let i = 0; i < n; i++) st.rbm.cd1([v], 0.2); st.n += n; draw(); }
    function draw() {
      row.innerHTML = '';
      const rb = st.rbm, ph = rb.hidProb(v, null), h = ph.map(function (q, j) { return q > 0.5 ? 1 : 0; }), pv = rb.visProb(ph);
      const cap = function (t, x, o) { const d = el('div', { style: 'text-align:center;font-size:12px;color:var(--muted)' }, row); U.digit(d, x, o); el('div', { text: t }, d); };
      cap('data v', v, { px: 84, cell: 12 });
      cap('P(h | v), 16 units', ph, { px: 64, cell: 16 });
      cap('reconstruction P(v′ | h)', pv, { px: 84, cell: 12 });
      let err = 0; for (let i = 0; i < 49; i++) err += (v[i] - pv[i]) * (v[i] - pv[i]);
      out.innerHTML = 'updates so far: ' + st.n + ' · squared reconstruction error: <b>' + U.f(err, 3) + '</b>' + (st.keep < 1 ? '\nwith dropout, each update first draws r: dropped hidden units are clamped to 0 in both phases (§8.2)' : '') +
        '\n<span class="note">Here the reconstruction uses P(h | v) instead of a sample, to show what the model has learned. Learning pushes energy down at the data and up at the model’s own reconstructions.</span>';
    }
    reset();
  };
  // ---------- ch-drbm: train an RBM and a dropout RBM on the toy digits
  const rbmWait = []; let rbmState = null;
  U.rbms = function (cb, prog) {
    if (rbmState && rbmState.done) { cb(rbmState); return; }
    rbmWait.push([cb, prog]);
    if (rbmState) return;
    rbmState = { done: false };
    const X = Toy.binarize(Toy.digits(1500, 3131).X, 7), nets = [new Toy.RBM(36, 1, 5), new Toy.RBM(36, 0.5, 5)], upd = 1500, r = ML.rng(8);
    let t = 0;
    (function tick() {
      const t0 = performance.now();
      while (t < 2 * upd && performance.now() - t0 < 30) { const b = []; for (let i = 0; i < 20; i++) b.push(X[r.int(X.length)]); nets[t < upd ? 0 : 1].cd1(b, 0.1); t++; }
      rbmWait.forEach(function (w) { if (w[1]) w[1](t / (2 * upd)); });
      if (t < 2 * upd) { setTimeout(tick, 0); return; }
      rbmState = { done: true, nets: nets, te: Toy.binarize(Toy.data(500).test.X.slice(0, 200), 8) };
      rbmWait.splice(0).forEach(function (w) { w[0](rbmState); });
    })();
  };
  function withRBM(host, ready) {
    const msg = el('div', { class: 'out', text: 'Training two RBMs on the toy digits (one time)…' }, host);
    U.rbms(function (R) { msg.remove(); ready(R); }, function (f) { msg.textContent = 'Training two RBMs with CD-1 on the toy digits (one time, in your browser): ' + Math.round(100 * f) + '%'; });
  }
  W.drbmTrain = function (host) {
    Book.head(host, 'Figure 12, toy version: RBM features',
      'An RBM and a dropout RBM (p = 0.5 on the hidden units), each with 36 hidden units, trained with CD-1 on 1,500 binarised toy digits. Each picture is one hidden unit’s weights, ordered by their L2 norm (largest first), as in Paper Fig. 12.');
    const box = el('div', { class: 'grid2' }, host), out = el('div', { class: 'out' }, host);
    withRBM(host, function (R) {
      const stats = R.nets.map(function (n, k) {
        const col = el('div', null, box); el('div', { html: '<b>' + (k ? 'Dropout RBM, p = 0.5' : 'RBM') + '</b>', style: 'margin-bottom:4px' }, col);
        const g = el('div', { style: 'display:grid;grid-template-columns:repeat(6,auto);gap:3px;justify-content:start' }, col);
        const order = ML.range(36).sort(function (i, j) { return ML.norm(n.feature(j)) - ML.norm(n.feature(i)); });
        order.forEach(function (j) { U.digit(g, n.feature(j), { px: 36, cell: 6, gap: 0, signed: true }); });
        const act = ML.range(36).map(function (j) { return ML.mean(R.te.map(function (v) { return n.hidProb(v, null)[j]; })); });
        return { dead: act.filter(function (m) { return m < 0.02 || m > 0.98; }).length };
      });
      out.innerHTML = 'units almost never changing state (mean P(h = 1) below 0.02 or above 0.98 on 200 test digits): RBM <b>' + stats[0].dead + '</b> · dropout RBM <b>' + stats[1].dead + '</b> of 36 (toy)' +
        '\n<span class="note">Paper Fig. 12 (MNIST, 256 hidden units): the dropout RBM’s features look coarser than the sharp stroke-like features of the standard RBM, and it has very few dead units (§8.3).</span>';
    });
  };
  W.drbmSparsity = function (host) {
    Book.head(host, 'Figure 13, toy version: dropout RBMs are sparser',
      'Histograms of the hidden activation probabilities P(hⱼ = 1 | v) of both RBMs on 200 binarised test digits (all units present).');
    const box = el('div', { class: 'grid2' }, host), out = el('div', { class: 'out' }, host);
    withRBM(host, function (R) {
      const st = R.nets.map(function (n, k) {
        const all = [].concat.apply([], R.te.map(function (v) { return n.hidProb(v, null); }));
        const col = el('div', null, box); el('div', { html: '<b>' + (k ? 'Dropout RBM' : 'RBM') + '</b>' }, col);
        const nb = 10, cnt = new Array(nb).fill(0); all.forEach(function (v) { cnt[Math.min(nb - 1, Math.floor(v * nb))]++; });
        Book.chart.bars(el('div', null, col), cnt.map(function (c, i) { return { label: (i / nb).toFixed(1), value: c / all.length, note: '', color: k ? P[0] : P[1] }; }), { width: 320, height: 170, ylabel: 'fraction', mb: 30, tickFmt: function (v) { return v.toFixed(2); } });
        return { mean: ML.mean(all), low: 100 * all.filter(function (v) { return v < 0.1; }).length / all.length };
      });
      out.innerHTML = 'mean activation probability: RBM <b>' + U.f(st[0].mean, 3) + '</b> · dropout RBM <b>' + U.f(st[1].mean, 3) + '</b>\nactivations below 0.1: RBM ' + U.pct(st[0].low) + ' · dropout RBM ' + U.pct(st[1].low) + ' (toy)' +
        '\n<span class="note">Paper Fig. 13: “a large number of units have activations close to zero and very few units have high activation” with dropout (§8.4).</span>';
    });
  };

  // ================================================================== PART V · BEYOND BERNOULLI
  // a small regression problem used by the §9 widgets
  const REG = (function () {
    const r = ML.rng(13), N = 8, D = 3, X = ML.mat(N, D, function (i, j) { return +(r.normal(0, [1, 2, 0.5][j])).toFixed(2); }), wt = [1.0, -0.6, 2.0];
    const y = X.map(function (row) { return +(ML.dot(row, wt) + r.normal(0, 0.3)).toFixed(2); });
    const G2 = ML.range(D).map(function (j) { let s = 0; X.forEach(function (row) { s += row[j] * row[j]; }); return s; });   // diag(XᵀX)
    return { X: X, y: y, N: N, D: D, G2: G2 };
  })();
  W.margCheck = function (host) {
    Book.head(host, 'Dropout on the inputs of linear regression = a penalty',
      'Eight data points with three input features (the table). Dropout keeps each input entry with probability p. The average squared error over thousands of random masks R matches the paper’s closed form ‖y − pXw‖² + p(1 − p)‖Γw‖² (§9.1). Drag the weights and p.');
    const w = [0.8, -0.4, 1.5], st = { p: 0.5, rng: ML.rng(4) };
    const ctl = el('div', { class: 'controls' }, host);
    w.forEach(function (v, j) { Book.slider(ctl, 'w' + (j + 1), -2, 3, 0.1, v, function (u) { w[j] = u; draw(); }, function (u) { return u.toFixed(1); }); });
    Book.slider(ctl, 'p', 0.05, 1, 0.05, st.p, function (v) { st.p = v; draw(); }, function (v) { return v.toFixed(2); });
    const box = el('div', { class: 'grid2' }, host), tb = el('div', null, box), out = el('div', { class: 'out' }, box);
    U.table(tb, [['x₁', 'x₂', 'x₃', 'y']].concat(REG.X.map(function (row, i) { return row.concat([REG.y[i]]); })));
    function draw() {
      const p = st.p, M = 4000; let mc = 0;
      for (let t = 0; t < M; t++) { let s = 0; REG.X.forEach(function (row, i) { let pred = 0; for (let j = 0; j < 3; j++) if (st.rng.next() < p) pred += row[j] * w[j]; s += (REG.y[i] - pred) * (REG.y[i] - pred); }); mc += s / M; }
      let fit = 0; REG.X.forEach(function (row, i) { const d = REG.y[i] - p * ML.dot(row, w); fit += d * d; });
      const pen = p * (1 - p) * ML.sum(w.map(function (v, j) { return REG.G2[j] * v * v; }));
      out.innerHTML = 'Monte-Carlo average of ‖y − (R ∗ X)w‖² over ' + M + ' masks: <b>' + U.f(mc, 3) + '</b>\nclosed form ‖y − pXw‖² + p(1 − p)‖Γw‖² = ' + U.f(fit, 3) + ' + ' + U.f(pen, 3) + ' = <b>' + U.f(fit + pen, 3) + '</b>' +
        '\n\nΓ² = diag(XᵀX) = (' + REG.G2.map(function (v) { return v.toFixed(2); }).join(', ') + ')\n<span class="note">Feature 2 varies the most, so its weight pays the largest penalty.</span>';
    }
    draw();
  };
  W.ridgeEquiv = function (host) {
    Book.head(host, 'Less p, more shrinkage',
      'Solving both of the paper’s forms of the marginalised objective exactly. With w̃ = pw the dropout objective becomes ridge regression with penalty strength (1 − p)/p, scaled per feature by Γ (§9.1). Slide p: the solution w̃ shrinks towards 0 as p falls.');
    const st = { p: 0.5 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'p', 0.05, 1, 0.05, st.p, function (v) { st.p = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    const XtX = ML.mat(3, 3, function (a, c) { let s = 0; REG.X.forEach(function (r) { s += r[a] * r[c]; }); return s; }), Xty = ML.range(3).map(function (a) { let s = 0; REG.X.forEach(function (r, i) { s += r[a] * REG.y[i]; }); return s; });
    function solveW(p) { const A = ML.mat(3, 3, function (a, c) { return p * p * XtX[a][c] + (a === c ? p * (1 - p) * REG.G2[a] : 0); }); return U.solve(A, Xty.map(function (v) { return p * v; })); }
    function solveRidge(p) { const lam = (1 - p) / p; const A = ML.mat(3, 3, function (a, c) { return XtX[a][c] + (a === c ? lam * REG.G2[a] : 0); }); return U.solve(A, Xty); }
    function draw() {
      const ps = ML.linspace(0.05, 1, 40);
      Book.chart.lines(plot, [0, 1, 2].map(function (j) { return { name: 'w̃' + (j + 1), color: P[j], marker: 'none', points: ps.map(function (p) { return [p, solveRidge(p)[j]]; }) }; }),
        { width: 640, height: 230, xmin: 0, xmax: 1, xticks: [0, 0.25, 0.5, 0.75, 1], xlabel: 'p', ylabel: 'solution w̃', mr: 50 });
      const w = solveW(st.p), wt = solveRidge(st.p);
      out.innerHTML = 'minimiser of ‖y − pXw‖² + p(1−p)‖Γw‖²: w = (' + w.map(function (v) { return v.toFixed(3); }).join(', ') + ')\np · w = (<b>' + w.map(function (v) { return (st.p * v).toFixed(3); }).join(', ') + '</b>)\nridge with λ = (1 − p)/p = ' + U.f((1 - st.p) / st.p, 3) + ': w̃ = (<b>' + wt.map(function (v) { return v.toFixed(3); }).join(', ') + '</b>)   (identical)';
    }
    draw();
  };
  W.fastDropout = function (host) {
    Book.head(host, 'Fast dropout: replace the coin flips by a Gaussian',
      'A logistic unit with 30 dropped-out inputs. Its input z is a sum of many independent random terms, so its distribution is close to a bell curve with mean μ = p Σ wᵢxᵢ and variance s² = p(1 − p) Σ (wᵢxᵢ)². Wang and Manning (2013), discussed in §9.2, train with that Gaussian instead of sampling masks. Compare three estimates of the average output.');
    const r0 = ML.rng(31), wx = ML.range(30).map(function () { return r0.normal(0.1, 0.6); });
    const st = { p: 0.5, rng: ML.rng(5) };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'p', 0.1, 1, 0.05, st.p, function (v) { st.p = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const p = st.p, zs = [], M = 5000; let Ef = 0;
      for (let t = 0; t < M; t++) { let z = 0; for (let i = 0; i < 30; i++) if (st.rng.next() < p) z += wx[i]; zs.push(z); Ef += ML.sigmoid(z) / M; }
      const mu = p * ML.sum(wx), s2 = p * (1 - p) * ML.sum(wx.map(function (v) { return v * v; })), s = Math.sqrt(s2) || 1e-9;
      const lo = mu - 4 * s, hi = mu + 4 * s, nb = 30, cnt = new Array(nb).fill(0); zs.forEach(function (z) { const k = Math.floor((z - lo) / (hi - lo) * nb); if (k >= 0 && k < nb) cnt[k]++; });
      const bw = (hi - lo) / nb;
      Book.chart.lines(plot, [{ name: 'sampled', color: P[0], marker: 'none', points: cnt.map(function (c, i) { return [lo + (i + 0.5) * bw, c / M / bw]; }) },
        { name: 'Gaussian', color: P[1], marker: 'none', dash: '5 3', points: ML.linspace(lo, hi, 80).map(function (z) { return [z, Math.exp(-(z - mu) * (z - mu) / (2 * s2)) / (s * Math.sqrt(2 * Math.PI))]; }) }],
        { width: 640, height: 220, xmin: lo, xmax: hi, ymin: 0, xticks: [lo, mu, hi].map(function (v) { return +v.toFixed(1); }), xlabel: 'z', ylabel: 'density', ydec: 2, mr: 80 });
      const gauss = ML.sigmoid(mu / Math.sqrt(1 + Math.PI * s2 / 8));
      out.innerHTML = 'μ = ' + U.f(mu, 3) + ', s = ' + U.f(s, 3) + '\naverage of σ(z) over ' + M + ' masks: <b>' + U.f(Ef, 4) + '</b>\nGaussian approximation σ(μ / √(1 + πs²/8)): <b>' + U.f(gauss, 4) + '</b>\nweight scaling σ(μ): <b>' + U.f(ML.sigmoid(mu), 4) + '</b>' +
        '\n<span class="note">The Gaussian formula is the standard probit-style approximation; the paper only summarises Wang and Manning’s method and notes its assumptions weaken as layers are added.</span>';
    }
    draw();
  };
  W.noiseCompare = function (host) {
    Book.head(host, 'Bernoulli and Gaussian multipliers with the same mean and variance',
      'Dropout (inverted form) multiplies a unit by r_b, which is 1/p with probability p and 0 otherwise: mean 1, variance (1 − p)/p. Gaussian dropout multiplies by r_g ~ N(1, σ²); with σ² = (1 − p)/p it has the same mean and variance (§10). 20,000 draws of each.');
    const st = { p: 0.5, rng: ML.rng(14) };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'p', 0.1, 0.95, 0.05, st.p, function (v) { st.p = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const p = st.p, sd = Math.sqrt((1 - p) / p), N = 20000, rb = [], rg = [];
      for (let i = 0; i < N; i++) { rb.push(st.rng.next() < p ? 1 / p : 0); rg.push(1 + sd * st.rng.normal(0, 1)); }
      const lo = -3, hi = 1 / p + 1.5, nb = 40, bw = (hi - lo) / nb, h = function (a) { const c = new Array(nb).fill(0); a.forEach(function (v) { const k = Math.floor((v - lo) / bw); if (k >= 0 && k < nb) c[k]++; }); return c.map(function (x, i) { return [lo + (i + 0.5) * bw, x / N]; }); };
      Book.chart.lines(plot, [{ name: 'Bernoulli r_b', color: P[0], marker: 'none', points: h(rb) }, { name: 'Gaussian r_g', color: P[1], marker: 'none', points: h(rg) }],
        { width: 640, height: 220, xmin: lo, xmax: hi, ymin: 0, xticks: [-2, 0, 1, Math.round(1 / p * 10) / 10], xlabel: 'multiplier r', ylabel: 'fraction of draws', ydec: 2, mr: 100 });
      out.innerHTML = 'r_b: mean ' + U.f(ML.mean(rb), 3) + ', variance ' + U.f(ML.variance(rb), 3) + '\nr_g: mean ' + U.f(ML.mean(rg), 3) + ', variance ' + U.f(ML.variance(rg), 3) + '\n(1 − p)/p = <b>' + U.f((1 - p) / p, 3) + '</b> · σ = √((1 − p)/p) = <b>' + U.f(sd, 3) + '</b>' +
        '\n<span class="note">r_b takes only two values; r_g can even be negative. The paper notes that, for a fixed mean and variance, r_g has the highest entropy and r_b the lowest, and both work well.</span>';
    }
    draw();
  };
  W.gaussToy = function (host) {
    Book.head(host, 'Table 10: Bernoulli vs Gaussian dropout',
      'Paper Table 10 (error %, mean ± standard deviation over 10 random seeds), then the same comparison on the toy (3 seeds each, σ = √((1 − p)/p) per layer).');
    const chart = el('div', null, host);
    U.hbars(chart, [{ label: 'MNIST · Bernoulli (paper)', value: 1.08, note: '1.08 ± 0.04 %', color: P[0] }, { label: 'MNIST · Gaussian (paper)', value: 0.95, note: '0.95 ± 0.04 %', color: P[1] },
      { label: 'CIFAR-10 · Bernoulli (paper)', value: 12.6, note: '12.6 ± 0.1 %', color: P[0] }, { label: 'CIFAR-10 · Gaussian (paper)', value: 12.5, note: '12.5 ± 0.1 %', color: P[1] }], { labelWidth: 230, max: 14 });
    U.cached(host, 'table10', function (R, view) {
      view.innerHTML = '';
      const stat = function (kind) { const v = R.filter(function (r) { return r.label.indexOf(kind) === 0; }).map(function (r) { return r.test; }); const m = ML.mean(v); return [m, Math.sqrt(ML.variance(v)), v]; };
      const b = stat('Bernoulli'), g = stat('Gaussian');
      el('div', { class: 'out', html: 'toy, Bernoulli: ' + b[2].map(function (x) { return x.toFixed(2); }).join(', ') + ' → mean <b>' + U.pct(b[0], 2) + '</b> ± ' + U.f(b[1], 2) + '\ntoy, Gaussian:  ' + g[2].map(function (x) { return x.toFixed(2); }).join(', ') + ' → mean <b>' + U.pct(g[0], 2) + '</b> ± ' + U.f(g[1], 2) + '\n<span class="note">Same seeds, so each Gaussian run starts from the same weights as its Bernoulli twin.</span>' }, view);
    }, 20);
  };
  W.recipe = function (host) {
    Book.head(host, 'The practical guide as a calculator',
      'Appendix A in one place. Enter what works for a standard (no-dropout) network on your task, and the p you plan to use; the widget applies the paper’s rules of thumb.');
    const st = { n: 512, p: 0.5, lr: 0.01, inputType: 'real' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'best standard layer size n', 64, 4096, 64, st.n, function (v) { st.n = v; draw(); }, function (v) { return v; });
    Book.slider(ctl, 'hidden p', 0.3, 0.9, 0.05, st.p, function (v) { st.p = v; draw(); }, function (v) { return v.toFixed(2); });
    Book.select(ctl, 'standard learning rate', ['0.001', '0.01', '0.1'], String(st.lr), function (v) { st.lr = +v; draw(); });
    Book.select(ctl, 'inputs', [{ value: 'real', label: 'real-valued (pixels, speech)' }, { value: 'other', label: 'other' }], st.inputType, function (v) { st.inputType = v; draw(); });
    const out = el('div', { class: 'out' }, host);
    function draw() {
      out.innerHTML = 'A.1 network size: at least n/p = ' + st.n + '/' + st.p + ' = <b>' + Math.ceil(st.n / st.p) + '</b> units per layer (on average p·(n/p) = ' + st.n + ' are kept)' +
        '\nA.2 learning rate: 10–100 × the standard rate = <b>' + U.f(10 * st.lr, 3) + ' to ' + U.f(100 * st.lr, 3) + '</b>' +
        '\nA.2 momentum: <b>0.95 to 0.99</b> (instead of the common 0.9)' +
        '\nA.3 max-norm: incoming weight vectors limited to ‖w‖ ≤ c with <b>c ≈ 3 to 4</b>' +
        '\nA.4 dropout rate: hidden p typically <b>0.5 to 0.8</b>' + (st.p < 0.5 || st.p > 0.8 ? ' (your p is outside that range)' : '') + '; input p ' + (st.inputType === 'real' ? '<b>0.8</b> for real-valued inputs' : 'depends on the input; closer to 1 than the hidden p') +
        '\n<span class="note">Smaller p needs a bigger n, which slows training and can underfit; larger p may not drop enough to stop overfitting (A.4).</span>';
    }
    draw();
  };
  W.tradeoff = function (host) {
    Book.head(host, 'The price: slower, noisier training',
      'From the toy Figure 4 runs: the training error of each architecture with and without dropout. Dropout nets fit their training data much more slowly, because every step trains a different random architecture. Pick a target test error and count the steps each run needs to reach it.');
    let target = 9;
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, 'target test error %', 7.5, 12, 0.25, target, function (v) { target = v; draw(); }, function (v) { return v.toFixed(2); });
    const plot = el('div', null, host), out = el('div', { class: 'out' }, host);
    function draw() {
      const R = (window.TOY_RESULTS || {}).fig4; if (!R) { out.textContent = 'toy results missing'; return; }
      Book.chart.lines(plot, R.map(function (r) { const d = /dropout/.test(r.label); return { color: d ? P[0] : P[1], marker: 'none', width: 1.5, points: r.curve.slice(1).map(function (c) { return [c[0], c[1]]; }) }; }),
        { width: 640, height: 220, xmin: 0, xmax: 4000, ymin: 0, ymax: 16, xticks: [0, 1000, 2000, 3000, 4000], xlabel: 'weight updates', ylabel: 'training error % (toy)' });
      out.innerHTML = R.map(function (r) { const hit = r.curve.find(function (c) { return c[0] > 0 && c[2] <= target; }); return r.label.padEnd(20) + ' first reaches ≤ ' + target.toFixed(2) + ' % test error at ' + (hit ? 'step ' + hit[0] : 'never (in 4,000 steps)'); }).join('\n') +
        '\n<span class="note">Red: standard, blue: dropout. The paper: a dropout network “typically takes 2–3 times longer to train” than a standard one of the same architecture (§11).</span>';
    }
    draw();
  };
})();
