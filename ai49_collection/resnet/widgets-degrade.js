/* Widgets for Part III (the degradation problem) of the ResNet book. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, T = window.RNT, U = window.RNU;

  // values read by eye off the paper's Fig. 1 (approximate)
  const FIG1 = {
    tr20: [[0.4, 20], [0.6, 17], [1, 13.5], [1.5, 12], [2, 10.5], [2.5, 10.2], [3, 9.5], [3.2, 9.2], [3.3, 4.2], [3.6, 3.6], [4, 2.6], [4.5, 2.2], [5, 1.4], [5.5, 1.4], [6, 1.3], [6.4, 1.5]],
    tr56: [[1.95, 20], [2.2, 18.5], [2.5, 18], [3, 17], [3.2, 18.5], [3.35, 11.5], [3.6, 10], [4, 9], [4.5, 8.3], [5, 5.8], [5.5, 5.9], [6, 5.4], [6.4, 6]],
    te20: [[0.4, 20], [0.6, 17.5], [1, 15.5], [1.5, 14], [2, 13.3], [2.5, 13], [3, 12.7], [3.2, 12.8], [3.3, 10.2], [3.6, 9.9], [4, 10.1], [4.5, 10], [5, 9.8], [5.5, 9.7], [6, 9.7], [6.4, 9.7]],
    te56: [[2.2, 20], [2.4, 18.2], [2.6, 19.8], [3, 18.3], [3.2, 18.6], [3.35, 14.5], [3.6, 13.9], [4, 13.6], [4.5, 13.5], [5, 13.4], [5.5, 13.3], [6, 13.3], [6.4, 13.3]]
  };
  W.fig1 = function (host) {
    Book.head(host, 'Figure 1, redrawn: 20 layers beat 56 layers',
      'CIFAR-10, plain networks (no shortcuts). Left: error on the training images. Right: error on the test images. Values read by eye off the paper’s Fig. 1 (approximate). The drop near 32k iterations is where the learning rate is divided by 10.');
    const g2 = el('div', { class: 'grid2' }, host), A = el('div', null, g2), B = el('div', null, g2);
    const o = { xmin: 0, xmax: 6.4, ymin: 0, ymax: 20, xticks: [0, 1, 2, 3, 4, 5, 6], xlabel: 'iterations (×10⁴)', width: 330, height: 220 };
    U.lines(A, [{ name: '20-layer', color: U.C.gold, marker: 'none', points: FIG1.tr20 }, { name: '56-layer', color: U.C.plain, marker: 'none', points: FIG1.tr56 }], Object.assign({ ylabel: 'training error (%) ≈' }, o));
    U.lines(B, [{ name: '20-layer', color: U.C.gold, marker: 'none', points: FIG1.te20 }, { name: '56-layer', color: U.C.plain, marker: 'none', points: FIG1.te56 }], Object.assign({ ylabel: 'test error (%) ≈' }, o));
    U.note(host, 'At the end: training error ≈ 1.5% (20-layer) vs ≈ 6% (56-layer); test error ≈ 9.7% vs ≈ 13.3%. The deeper network is worse on the data it was trained on, not only on new data.');
  };

  // ---------------------------------------------------------------- live race of plain (and residual) MLPs of several depths
  W.degradeRace = function (host) {
    Book.head(host, 'Reproduce degradation: plain networks of growing depth',
      'Fully-connected networks with batch normalization, He initialization and SGD with momentum (the paper’s recipe, in miniature) learn the 5-spiral task. Tick the depths to race. Each “block” is two layers, so depth = 2 × blocks + 2. Add shortcuts to compare.');
    const depths = [6, 14, 30, 50], on = { 6: true, 14: true, 30: true, 50: true };
    let residual = false;
    const ctl = el('div', { class: 'controls' }, host);
    depths.forEach(function (d) { const lab = el('label', null, ctl); const c = el('input', { type: 'checkbox', checked: 'checked' }, lab); lab.appendChild(document.createTextNode(' ' + d + ' layers')); c.addEventListener('change', function () { on[d] = c.checked; }); });
    const rl = el('label', null, ctl); const rc = el('input', { type: 'checkbox' }, rl); rl.appendChild(document.createTextNode(' with shortcuts')); rc.addEventListener('change', function () { residual = rc.checked; });
    Book.button(ctl, 'Train ▶', function () { start(); }, 'primary');
    const plot = el('div', null, host), bars = el('div', null, host), out = U.out(host);
    const cols = { 6: U.C.gold, 14: U.C.green, 30: U.C.plain, 50: U.C.purple };
    let runner = null;
    function start() {
      if (runner) runner.stop();
      const d = T.spiralData(), sel = depths.filter(function (x) { return on[x]; });
      const jobs = sel.map(function (dep) {
        const net = new T.MLP({ inDim: 2, width: 16, blocks: (dep - 2) / 2, residual: residual, classes: 5, seed: 1 });
        const j = T.trainer(net, d.train, { steps: 1500, batch: 64, lr: 0.1, schedule: [[1000, 0.01]], seed: 11, every: 25 }); j.depth = dep; return j;
      });
      bars.innerHTML = '';
      runner = T.run(jobs, function () { draw(jobs); }, function () {
        const e = jobs.map(function (j) { return { d: j.depth, e: T.evaluate(j.net, d.train).err }; });
        Book.chart.bars(bars, e.map(function (x) { return { label: x.d + ' layers', value: 100 * x.e, color: cols[x.d], note: (100 * x.e).toFixed(1) + '%' }; }), { ylabel: 'final training error (%)', height: 180 });
        out.innerHTML = (residual ? 'With shortcuts, every depth fits the training data (toy).' : 'Plain networks: beyond a point, <b>more layers give higher training error</b> (toy). Same data, same recipe; only the depth changed.') + ' Final errors are measured on all 1,200 training points.';
      });
    }
    function draw(jobs) {
      U.lines(plot, jobs.map(function (j) { return { name: (residual ? 'residual-' : 'plain-') + j.depth, color: cols[j.depth], marker: 'none', points: j.hist.map(function (h) { return [h.step, 100 * h.err]; }) }; }),
        { xmin: 0, xmax: 1500, ymin: 0, ymax: 80, xticks: [0, 500, 1000, 1500], xlabel: 'training steps', ylabel: 'training error (%)', height: 220 });
      out.innerHTML = 'step ' + (jobs[0] ? jobs[0].step : 0) + ' / 1500';
    }
    start();
  };

  // ---------------------------------------------------------------- not overfitting, not vanishing
  W.diagnose = function (host) {
    Book.head(host, 'Two suspects ruled out',
      'Plain networks of 6 and 30 layers train on the spirals as above. Left: training and test error at the end. Right: the size of the gradient that reaches each layer’s weights, measured during training. If the deep net were overfitting, its training error would be <em>lower</em>; if its gradients vanished, the bottom layers’ bars would be tiny.');
    const g2 = el('div', { class: 'grid2' }, host), A = el('div', null, g2), B = el('div', null, g2), out = U.out(host);
    out.textContent = 'training…';
    const d = T.spiralData();
    const jobs = [6, 30].map(function (dep) { const net = new T.MLP({ inDim: 2, width: 16, blocks: (dep - 2) / 2, residual: false, classes: 5, seed: 1 }); const j = T.trainer(net, d.train, { steps: 1500, batch: 64, lr: 0.1, schedule: [[1000, 0.01]], seed: 11, every: 50 }); j.depth = dep; return j; });
    const gn = [];
    function gradNorms(net) {
      const r = ML.rng(77), idx = []; for (let i = 0; i < 256; i++) idx.push(r.int(d.train.n));
      NN.zeroGrad(net.params); const L = NN.softmaxCE(net.forward(T.rows(d.train, idx), null, 'batch'), idx.map(function (k) { return d.train.y[k]; })); L.backward();
      const Ws = [net.inW]; net.blocks.forEach(function (b) { Ws.push(b.W1, b.W2); }); Ws.push(net.out.W);
      const res = Ws.map(function (Wt, i) { let s = 0; for (let k = 0; k < Wt.grad.length; k++) s += Wt.grad[k] * Wt.grad[k]; return [i + 1, Math.sqrt(s)]; });
      NN.zeroGrad(net.params); return res;
    }
    T.run(jobs, function () { out.textContent = 'training… step ' + jobs[0].step + ' / 1500'; if (jobs[1].step === 500 && !gn.length) gn.push(gradNorms(jobs[1].net)); }, function () {
      const ev = jobs.map(function (j) { return { tr: T.evaluate(j.net, d.train).err, te: T.evaluate(j.net, d.test).err }; });
      Book.chart.bars(A, [
        { label: '6 layers\ntraining', value: 100 * ev[0].tr, color: U.C.gold, note: (100 * ev[0].tr).toFixed(1) + '%' }, { label: '6 layers\ntest', value: 100 * ev[0].te, color: U.rgba(U.C.gold, 0.55), note: (100 * ev[0].te).toFixed(1) + '%' },
        { label: '30 layers\ntraining', value: 100 * ev[1].tr, color: U.C.plain, note: (100 * ev[1].tr).toFixed(1) + '%' }, { label: '30 layers\ntest', value: 100 * ev[1].te, color: U.rgba(U.C.plain, 0.55), note: (100 * ev[1].te).toFixed(1) + '%' }
      ], { ylabel: 'error (%)', height: 220, width: 360 });
      const g = gn[0] || gradNorms(jobs[1].net);
      U.logLines(B, [{ name: 'plain-30, at step 500', color: U.C.plain, points: g, r: 2.5 }], { ymin: 1e-3, ymax: 1e1, xmin: 1, xmax: 30, xticks: [1, 10, 20, 30], xlabel: 'layer (1 = bottom)', ylabel: 'gradient size ‖∂ℓ/∂W‖', width: 360, height: 220 });
      const vals = g.map(function (p) { return p[1]; }), mx = Math.max.apply(null, vals), mn = Math.min.apply(null, vals);
      out.innerHTML = 'The 30-layer net has the higher <b>training</b> error (so it is not overfitting), and its gradients stay within a factor of ' + Math.round(mx / mn) + ' from bottom to top (so they do not vanish: batch normalization keeps them healthy, as the paper also verified). Something else makes deep plain nets hard to optimize (toy).';
    });
  };

  // ---------------------------------------------------------------- the constructed solution
  W.constructDemo = function (host) {
    Book.head(host, 'A deeper network that is at least as good, by construction',
      'Train a shallow network (6 layers). Then build a 30-layer network from it: copy its layers and insert 24 layers that do nothing (weights = the identity matrix, batch normalization set to pass values through unchanged). It makes exactly the same predictions. Finally, train a 30-layer plain network from scratch with the same recipe: the solver does not find anything as good (toy).');
    const ctl = el('div', { class: 'controls' }, host);
    Book.button(ctl, 'Run again ▶', function () { start(); }, 'primary');
    const view = el('div', null, host), out = U.out(host);
    function start() {
      const d = T.spiralData();
      const mk = function (blocks, seed) { return new T.MLP({ inDim: 2, width: 16, blocks: blocks, residual: false, classes: 5, seed: seed }); };
      const sh = mk(2, 1), deep = mk(14, 1);
      const o = { steps: 1500, batch: 64, lr: 0.1, schedule: [[1000, 0.01]], seed: 11, every: 25 };
      const jobs = [T.trainer(sh, d.train, o), T.trainer(deep, d.train, o)];
      out.textContent = 'training…';
      T.run(jobs, function () { out.textContent = 'training… step ' + jobs[0].step + ' / 1500'; }, function () {
        // construct: copy the shallow layers, identities in between
        const built = mk(14, 99), I = function (Wt) { Wt.data.fill(0); for (let i = 0; i < Wt.rows; i++) Wt.data[i * Wt.cols + i] = 1; };
        const copyBN = function (to, from) { to.g.data.set(from.g.data); to.b.data.set(from.b.data); to.run.mean.set(from.run.mean); to.run.var.set(from.run.var); };
        const passBN = function (bn) { bn.g.data.fill(1); bn.b.data.fill(0); bn.run.mean.fill(0); bn.run.var.fill(1 - 1e-5); };   // (h − 0)/√(1 − ε + ε) · 1 + 0 = h
        built.inW.data.set(sh.inW.data); copyBN(built.inBN, sh.inBN);
        built.blocks.forEach(function (b, i) {
          if (i < 2) { b.W1.data.set(sh.blocks[i].W1.data); b.W2.data.set(sh.blocks[i].W2.data); copyBN(b.bn1, sh.blocks[i].bn1); copyBN(b.bn2, sh.blocks[i].bn2); }
          else { I(b.W1); I(b.W2); passBN(b.bn1); passBN(b.bn2); }
        });
        built.out.W.data.set(sh.out.W.data); built.out.b.data.set(sh.out.b.data);
        const e = [sh, built, deep].map(function (n) { return T.evaluate(n, d.train).err; });
        view.innerHTML = '';
        Book.chart.bars(el('div', null, view), [
          { label: 'shallow (6)\ntrained', value: 100 * e[0], color: U.C.gold, note: (100 * e[0]).toFixed(1) + '%' },
          { label: '30 = 6 + 24 identity\nconstructed', value: 100 * e[1], color: U.C.green, note: (100 * e[1]).toFixed(1) + '%' },
          { label: '30 plain\ntrained from scratch', value: 100 * e[2], color: U.C.plain, note: (100 * e[2]).toFixed(1) + '%' }
        ], { ylabel: 'training error (%)', height: 220, mb: 46 });
        out.innerHTML = 'The constructed 30-layer net is exactly as good as the shallow one (identical predictions: with test-time BN set to pass values through, an identity layer followed by ReLU changes nothing, because ReLU outputs are never negative). So a 30-layer solution this good <em>exists</em>, yet training from scratch did not find it (toy).';
      });
    }
    start();
  };
})();
