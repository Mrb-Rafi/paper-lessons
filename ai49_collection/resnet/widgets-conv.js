/* Widgets for Part II (convolutions) of the ResNet book. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, T = window.RNT, U = window.RNU;

  // a clean glyph of digit d at offset (ox, oy), values 0/1
  function glyph(d, ox, oy) { const v = new Float64Array(64); for (let i = 0; i < 7; i++) for (let j = 0; j < 5; j++) if (T.FONT[d][i * 5 + j] === '1') v[(i + (oy || 0)) * 8 + j + (ox == null ? 1 : ox)] = 1; return v; }
  T.glyph = glyph;
  const KERNELS = {
    'vertical edges': [-1, 0, 1, -2, 0, 2, -1, 0, 1], 'horizontal edges': [-1, -2, -1, 0, 0, 0, 1, 2, 1],
    'blur (average)': [1, 1, 1, 1, 1, 1, 1, 1, 1].map(function (v) { return v / 9; }), 'sharpen': [0, -1, 0, -1, 5, -1, 0, -1, 0],
    'identity': [0, 0, 0, 0, 1, 0, 0, 0, 0], 'diagonal': [2, -1, -1, -1, 2, -1, -1, -1, 2]
  };
  function conv2(x, H, Wd, k, K, stride, pad) {           // single channel, returns {y, Ho, Wo}
    const Ho = T.outSize(H, K, stride, pad), Wo = T.outSize(Wd, K, stride, pad), y = new Float64Array(Ho * Wo);
    for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++) {
      let s = 0; for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) { const iy = oy * stride - pad + a, ix = ox * stride - pad + b; if (iy >= 0 && ix >= 0 && iy < H && ix < Wd) s += k[a * K + b] * x[iy * Wd + ix]; }
      y[oy * Wo + ox] = s;
    }
    return { y: y, Ho: Ho, Wo: Wo };
  }

  // ---------------------------------------------------------------- ch-pixels
  W.pixelGrid = function (host) {
    Book.head(host, 'A picture is a grid of numbers',
      'An 8×8 grey picture of a digit: each square is one number, 0 for black … 1 for full ink. Click squares to paint or erase. A colour photo has three such grids, one per colour <b>channel</b>.');
    let d = 3, numbers = true; let img = glyph(d);
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'digit', ML.range(10).map(String), d, function (v) { d = +v; img = glyph(d); draw(); });
    const cb = el('label', null, ctl); const ch = el('input', { type: 'checkbox', checked: 'checked' }, cb); cb.appendChild(document.createTextNode(' show numbers'));
    ch.addEventListener('change', function () { numbers = ch.checked; draw(); });
    const g2 = el('div', { class: 'grid2' }, host), A = el('div', null, g2), B = el('div', null, g2), out = U.out(host);
    function draw() {
      A.innerHTML = ''; U.grid(A, img, 8, 8, { cell: 34, max: 1, numbers: numbers, digits: 1, onclick: function (i) { img[i] = img[i] > 0.5 ? 0 : 1; draw(); }, label: 'grey picture: 8 × 8 × 1' });
      B.innerHTML = '';
      const cols = ['#c0392b', '#2f855a', '#2b6cb0'], names = ['red', 'green', 'blue'];
      const svgRow = el('div', { style: 'display:flex;gap:6px;flex-wrap:wrap' }, B);
      [0, 1, 2].forEach(function (c) {
        const box = el('div', { style: 'flex:1;min-width:84px' }, svgRow);
        const v = Float64Array.from(img, function (x, i) { const r = (i % 8) / 7, t = Math.floor(i / 8) / 7; return x * (c === 0 ? 1 : c === 1 ? 0.6 + 0.4 * t : 0.3 + 0.7 * r); });
        U.grid(box, v, 8, 8, { cell: 13, max: 1, hue: cols[c], label: names[c] });
      });
      el('div', { class: 'note', html: 'a colour version of the same picture: 8 × 8 × 3 numbers. ResNet’s ImageNet inputs are 224 × 224 × 3 = 150,528 numbers; CIFAR-10’s are 32 × 32 × 3 = 3,072.' }, B);
      let ink = 0; for (let i = 0; i < 64; i++) ink += img[i];
      out.innerHTML = 'numbers in this picture: <b>64</b> · squares with ink: <b>' + ink + '</b>';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-conv
  W.convSlide = function (host) {
    Book.head(host, 'Slide a 3×3 kernel over the picture',
      'At every position, multiply the 9 numbers under the window by the 9 kernel weights and add them up: that is one output number. Step through, or pick another kernel and see which strokes it lights up.');
    const st = { d: 4, kern: 'vertical edges', pos: 0, timer: null };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'digit', ML.range(10).map(String), st.d, function (v) { st.d = +v; draw(); });
    Book.select(ctl, 'kernel', Object.keys(KERNELS), st.kern, function (v) { st.kern = v; draw(); });
    const posIn = Book.slider(ctl, 'position', 0, 35, 1, st.pos, function (v) { st.pos = v; draw(); });
    const play = Book.button(ctl, 'Play ▶', function () { if (st.timer) stop(); else { play.textContent = 'Pause ❚❚'; tick(); } }, 'primary');
    function stop() { clearTimeout(st.timer); st.timer = null; play.textContent = 'Play ▶'; }
    function tick() { st.pos = (st.pos + 1) % 36; posIn.value = st.pos; posIn.dispatchEvent(new Event('input')); st.timer = setTimeout(tick, 350); }
    const row = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;align-items:start' }, host);
    const A = el('div', null, row), K = el('div', null, row), B = el('div', null, row), out = U.out(host);
    function draw() {
      const x = glyph(st.d), k = KERNELS[st.kern], r = conv2(x, 8, 8, k, 3, 1, 0), oy = Math.floor(st.pos / 6), ox = st.pos % 6;
      const win = []; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) win.push((oy + a) * 8 + ox + b);
      A.innerHTML = ''; U.grid(A, x, 8, 8, { cell: 22, max: 1, hl: win, label: 'input 8 × 8' });
      K.innerHTML = ''; U.grid(K, k, 3, 3, { cell: 40, signed: true, numbers: true, digits: 2, label: 'kernel 3 × 3 (9 weights)', maxw: 130 });
      const part = Float64Array.from(r.y, function (v, i) { return i <= st.pos ? v : 0; });
      B.innerHTML = ''; U.grid(B, part, 6, 6, { cell: 26, signed: true, max: 4, numbers: true, digits: 1, hl: [st.pos], label: 'output 6 × 6' });
      const terms = []; win.forEach(function (idx, q) { if (x[idx] !== 0 && k[q] !== 0) terms.push(T.f(k[q], 2) + '×' + x[idx]); });
      out.innerHTML = 'output at row ' + (oy + 1) + ', column ' + (ox + 1) + ' = ' + (terms.length ? terms.join(' + ') : '0 (no ink under the window)') + ' = <b>' + T.f(r.y[st.pos], 2) + '</b>. The same 9 weights are used at all 36 positions.';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-stride
  W.strideCalc = function (host) {
    Book.head(host, 'Stride, padding and the size of the output',
      'One row of an input, with padding cells (dashed) added at both ends. The kernel is placed every <em>stride</em> cells. Count the placements: that is the output size. Try the presets from ResNet.');
    const st = { n: 14, k: 3, s: 2, p: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    const sl = {};
    sl.n = Book.slider(ctl, 'input size n', 3, 24, 1, st.n, function (v) { st.n = v; draw(); });
    sl.k = Book.slider(ctl, 'kernel k', 1, 7, 1, st.k, function (v) { st.k = v; draw(); });
    sl.s = Book.slider(ctl, 'stride s', 1, 3, 1, st.s, function (v) { st.s = v; draw(); });
    sl.p = Book.slider(ctl, 'padding p', 0, 3, 1, st.p, function (v) { st.p = v; draw(); });
    const pre = el('div', { class: 'controls' }, host);
    [['3×3, stride 1, pad 1 (keeps size)', 14, 3, 1, 1], ['3×3, stride 2, pad 1 (halves)', 14, 3, 2, 1], ['1×1, stride 2 (projection)', 14, 1, 2, 0], ['7×7, stride 2, pad 3 (conv1, scaled down)', 16, 7, 2, 3]].forEach(function (p) {
      Book.button(pre, p[0], function () { st.n = p[1]; st.k = p[2]; st.s = p[3]; st.p = p[4]; ['n', 'k', 's', 'p'].forEach(function (key) { sl[key].value = st[key]; sl[key].dispatchEvent(new Event('input')); }); });
    });
    const view = el('div', { class: 'scroll-x' }, host), out = U.out(host);
    function draw() {
      view.innerHTML = '';
      const tot = st.n + 2 * st.p, c = Math.min(26, 620 / tot), outN = st.k > tot ? 0 : Math.floor((tot - st.k) / st.s) + 1;
      const H = 60 + outN * 9 + 40, svg = S('svg', { viewBox: '0 0 ' + (tot * c + 20) + ' ' + H, style: 'min-width:' + Math.min(620, tot * c + 20) + 'px' }, view);
      for (let i = 0; i < tot; i++) {
        const pad = i < st.p || i >= st.p + st.n;
        S('rect', { x: 10 + i * c + 1, y: 10, width: c - 2, height: c - 2, rx: 3, fill: pad ? 'none' : 'var(--accent-soft)', stroke: pad ? 'var(--muted)' : 'var(--accent)', 'stroke-dasharray': pad ? '3 2' : '' }, svg);
        if (pad) S('text', { x: 10 + i * c + c / 2, y: 10 + c / 2 + 4, 'text-anchor': 'middle', 'font-size': 10, class: 'svg-text', text: '0' }, svg);
      }
      for (let o = 0; o < outN; o++) {
        const x0 = 10 + o * st.s * c + 2, w = st.k * c - 4, y = 16 + c + o * 9;
        S('rect', { x: x0, y: y, width: w, height: 6, rx: 3, fill: U.stage[o % U.stage.length], opacity: 0.8 }, svg);
      }
      S('text', { x: 10, y: H - 12, 'font-size': 12, class: 'svg-text', text: outN + ' placements → output size ' + outN }, svg);
      out.innerHTML = 'output size = ⌊(n + 2p − k) / s⌋ + 1 = ⌊(' + st.n + ' + ' + 2 * st.p + ' − ' + st.k + ') / ' + st.s + '⌋ + 1 = <b>' + outN + '</b>. ' +
        'In ResNet: ⌊(224 + 6 − 7)/2⌋ + 1 = 112 for conv1, ⌊(112 + 2 − 3)/2⌋ + 1 = 56 for the max pool, and every 3×3 conv with stride 2 and padding 1 halves 56 → 28 → 14 → 7.';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-channels
  W.channelStack = function (host) {
    Book.head(host, 'Many filters, many channels: counting weights and work',
      'A convolutional layer has C<sub>out</sub> filters; each filter is k × k × C<sub>in</sub> weights, because it looks at every input channel. Change the sizes and see the cost. Then try the ResNet rule: halve the map size, double the channels.');
    const st = { cin: 64, cout: 64, k: 3, h: 56 };
    const ctl = el('div', { class: 'controls' }, host);
    const sl = {};
    sl.cin = Book.slider(ctl, 'C_in', 1, 512, 1, st.cin, function (v) { st.cin = v; draw(); });
    sl.cout = Book.slider(ctl, 'C_out', 1, 512, 1, st.cout, function (v) { st.cout = v; draw(); });
    sl.k = Book.slider(ctl, 'kernel k', 1, 7, 2, st.k, function (v) { st.k = v; draw(); });
    sl.h = Book.slider(ctl, 'output map size', 1, 112, 1, st.h, function (v) { st.h = v; draw(); });
    const pre = el('div', { class: 'controls' }, host);
    [['conv2_x (56², 64→64)', 64, 64, 3, 56], ['conv3_x (28², 128→128)', 128, 128, 3, 28], ['conv4_x (14², 256→256)', 256, 256, 3, 14], ['conv5_x (7², 512→512)', 512, 512, 3, 7]].forEach(function (p) {
      Book.button(pre, p[0], function () { st.cin = p[1]; st.cout = p[2]; st.k = p[3]; st.h = p[4]; ['cin', 'cout', 'k', 'h'].forEach(function (key) { sl[key].value = st[key]; sl[key].dispatchEvent(new Event('input')); }); });
    });
    const view = el('div', null, host), out = U.out(host);
    function draw() {
      view.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 600 120' }, view), shown = Math.min(st.cout, 12);
      for (let f = 0; f < shown; f++) {
        const x = 10 + f * 48, depth = Math.min(st.cin, 6);
        for (let dd = depth - 1; dd >= 0; dd--) S('rect', { x: x + dd * 3, y: 30 - dd * 3, width: 8 + st.k * 4, height: 8 + st.k * 4, fill: U.rgba(U.stage[f % 7], 0.25 + 0.1 * dd), stroke: U.stage[f % 7] }, svg);
      }
      S('text', { x: 10, y: 100, 'font-size': 12, class: 'svg-text', text: (st.cout > shown ? 'first ' + shown + ' of ' : '') + st.cout + ' filters, each ' + st.k + ' × ' + st.k + ' × ' + st.cin }, svg);
      const params = st.k * st.k * st.cin * st.cout, macs = st.h * st.h * params;
      out.innerHTML = 'weights = k² · C<sub>in</sub> · C<sub>out</sub> = ' + st.k * st.k + ' · ' + st.cin + ' · ' + st.cout + ' = <b>' + T.comma(params) + '</b> · multiply-adds = (map size)² · weights = ' + st.h + '² · ' + T.comma(params) + ' = <b>' + T.big(macs) + 'FLOPs</b>' +
        '<br><span class="note">Halving the map size divides the work by 4; doubling both C<sub>in</sub> and C<sub>out</sub> multiplies it by 4. So the paper’s rule “halve the size, double the filters” keeps the work per layer the same (§3.3).</span>';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-1x1
  W.oneByOne = function (host) {
    Book.head(host, 'A 1×1 convolution mixes channels, pixel by pixel',
      'Three input channels, two output channels. Each output pixel is a weighted sum of the three input values <em>at the same pixel</em>: a tiny fully-connected layer applied everywhere. Click a pixel to see its arithmetic.');
    const Wm = [[1, 0.5, 0], [-0.5, 0, 1]], st = { pix: 14 };
    const ctl = el('div', { class: 'controls' }, host);
    [0, 1].forEach(function (o) { [0, 1, 2].forEach(function (i) { Book.slider(ctl, 'w' + (o + 1) + (i + 1), -1, 1, 0.1, Wm[o][i], function (v) { Wm[o][i] = v; draw(); }, function (v) { return v.toFixed(1); }); }); });
    const row = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px' }, host), out = U.out(host);
    const X = [0, 1, 2].map(function (c) { return Float64Array.from({ length: 36 }, function (_, i) { const r = Math.floor(i / 6), q = i % 6; return c === 0 ? (r < 3 ? 1 : 0) : c === 1 ? (q < 3 ? 1 : 0) : ((r + q) % 2); }); });
    function draw() {
      row.innerHTML = '';
      const Y = Wm.map(function (w) { return Float64Array.from({ length: 36 }, function (_, i) { return w[0] * X[0][i] + w[1] * X[1][i] + w[2] * X[2][i]; }); });
      X.forEach(function (m, c) { const b = el('div', null, row); U.grid(b, m, 6, 6, { cell: 18, max: 1, hl: [st.pix], label: 'input ' + (c + 1), onclick: function (i) { st.pix = i; draw(); } }); });
      Y.forEach(function (m, c) { const b = el('div', null, row); U.grid(b, m, 6, 6, { cell: 18, signed: true, max: 2, hl: [st.pix], label: 'output ' + (c + 1), onclick: function (i) { st.pix = i; draw(); } }); });
      const p = st.pix, xs = X.map(function (m) { return m[p]; });
      out.innerHTML = 'at the highlighted pixel: inputs (' + xs.join(', ') + ') → output 1 = ' + Wm[0].map(function (w, i) { return T.f(w, 1) + '·' + xs[i]; }).join(' + ') + ' = <b>' + T.f(Y[0][p], 2) + '</b>, output 2 = <b>' + T.f(Y[1][p], 2) + '</b>. Weights: 1 · 1 · 3 · 2 = <b>6</b>. ResNet uses 1×1 convolutions to shrink and restore the number of channels (the bottleneck, Fig. 5) and to reshape shortcuts (Eq. 2).';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-pool
  W.poolDemo = function (host) {
    Book.head(host, 'Pooling summarizes each neighbourhood',
      'Max pooling keeps the largest value in each window; average pooling keeps the mean. Global average pooling (how ResNet ends) averages the whole map into a single number per channel.');
    const st = { kind: 'max', k: 2, s: 2, seed: 2 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'kind', ['max', 'average'], st.kind, function (v) { st.kind = v; draw(); });
    Book.select(ctl, 'window', ['2', '3'], st.k, function (v) { st.k = +v; draw(); });
    Book.select(ctl, 'stride', ['1', '2'], st.s, function (v) { st.s = +v; draw(); });
    Book.button(ctl, 'New map ↻', function () { st.seed++; draw(); });
    const row = el('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px' }, host), out = U.out(host);
    function draw() {
      const r = ML.rng(st.seed), x = Float64Array.from({ length: 36 }, function () { return Math.round(r.uniform(0, 9)); });
      const X = new NN.Tensor(1, 36, x), p = T.pool(X, { C: 1, H: 6, W: 6 }, st.k, st.s, 0, st.kind === 'max' ? 'max' : 'avg');
      row.innerHTML = '';
      const a = el('div', null, row), b = el('div', null, row);
      U.grid(a, x, 6, 6, { cell: 28, max: 9, numbers: true, digits: 0, hue: '#2b6cb0', label: 'feature map 6 × 6', hl: (function () { const h = []; for (let i = 0; i < st.k; i++) for (let j = 0; j < st.k; j++) h.push(i * 6 + j); return h; })() });
      U.grid(b, p.t.data, p.s.H, p.s.W, { cell: 28, max: 9, numbers: true, digits: st.kind === 'max' ? 0 : 1, hue: '#1f7a7a', label: st.kind + ' pool → ' + p.s.H + ' × ' + p.s.W, hl: [0] });
      out.innerHTML = 'output size ⌊(6 − ' + st.k + ')/' + st.s + '⌋ + 1 = <b>' + p.s.H + '</b> · global average pool of the whole map = <b>' + T.f(ML.mean(Array.from(x)), 2) + '</b> · pooling has no weights. ResNet uses one 3×3 max pool with stride 2 after conv1, and one global average pool at the end (Table 1).';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-rf
  W.rfGrow = function (host) {
    Book.head(host, 'Deeper layers see more of the picture',
      'A stack of 3×3 convolutions shown along one row of pixels. The top cell depends on the highlighted cells below it: its <b>receptive field</b>. Add layers or make some of them stride 2.');
    const st = { L: 4, s2: 1 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.slider(ctl, '3×3 layers', 1, 6, 1, st.L, function (v) { st.L = v; draw(); });
    Book.select(ctl, 'stride-2 layers', [{ value: 0, label: 'none' }, { value: 1, label: 'layer 1' }, { value: 2, label: 'layers 1 and 3' }], st.s2, function (v) { st.s2 = +v; draw(); });
    const view = el('div', { class: 'scroll-x' }, host), out = U.out(host);
    function draw() {
      const strides = []; for (let l = 0; l < st.L; l++) strides.push((st.s2 >= 1 && l === 0) || (st.s2 === 2 && l === 2) ? 2 : 1);
      // sizes bottom-up for the input row of 40 cells
      let n = 40; const sizes = [n]; strides.forEach(function (s) { n = Math.floor((n + 2 - 3) / s) + 1; sizes.push(n); });
      // receptive field of the middle top cell, propagated down
      let lo = Math.floor(sizes[st.L] / 2), hi = lo; const ranges = []; ranges[st.L] = [lo, hi];
      for (let l = st.L - 1; l >= 0; l--) { lo = Math.max(0, lo * strides[l] - 1); hi = Math.min(sizes[l] - 1, hi * strides[l] + 1); ranges[l] = [lo, hi]; }
      view.innerHTML = '';
      const c = 14, H = (st.L + 1) * 30 + 20, svg = S('svg', { viewBox: '0 0 ' + (40 * c + 70) + ' ' + H, style: 'min-width:560px' }, view);
      for (let l = 0; l <= st.L; l++) {
        const y = H - 30 - l * 30, cw = 40 * c / sizes[l];
        S('text', { x: 40 * c + 8, y: y + 11, 'font-size': 11, class: 'svg-text', text: l === 0 ? 'input' : 'layer ' + l + (strides[l - 1] === 2 ? ' (/2)' : '') }, svg);
        for (let i = 0; i < sizes[l]; i++) {
          const on = i >= ranges[l][0] && i <= ranges[l][1];
          S('rect', { x: i * cw + 1, y: y, width: Math.max(1, cw - 2), height: 14, rx: 2, fill: on ? 'var(--accent)' : 'var(--accent-soft)', opacity: on ? 0.9 : 0.6 }, svg);
        }
      }
      let rf = 1, jump = 1; strides.forEach(function (s) { rf += 2 * jump; jump *= s; });
      // ResNet-34 receptive field from Table 1 layers
      let rf34 = 1, j34 = 1; T.archLayers(34, { plain: true }).forEach(function (l) { if (l.k) { rf34 += (l.k - 1) * j34; j34 *= l.stride; } });
      out.innerHTML = 'receptive field = 1 + Σ (k − 1) × (product of earlier strides) = <b>' + rf + '</b> input pixels wide (' + (ranges[0][1] - ranges[0][0] + 1) + ' visible here). ' +
        'Stride 2 doubles how fast it grows. By the same rule the last 3×3 layer of the 34-layer network has a receptive field of ' + rf34 + ' pixels, wider than the 224-pixel input: each of its outputs can use the whole picture.';
    }
    draw();
  };

  // ---------------------------------------------------------------- ch-cnn: a trained toy CNN's feature maps
  W.featureMaps = function (host) {
    Book.head(host, 'Inside a small convolutional network',
      'An 8-layer residual network from this book’s toy (a 3×3 conv, three stages with 6, 12 and 24 filters on maps of 8 × 8, 4 × 4 and 2 × 2, a global average pool, and a 10-way layer) is trained on noisy 8×8 digits right here. Pick a test digit and look at what each stage computes.');
    const ctl = el('div', { class: 'controls' }, host);
    let idx = 0;
    const pick = Book.slider(ctl, 'test digit #', 0, 49, 1, idx, function (v) { idx = v; draw(); });
    const status = U.out(host), view = el('div', null, host), out = U.out(host);
    let net = null;
    const d = T.cnnData(), data = d.train;
    net = new T.CNN({ n: 1, w: 6, residual: true, seed: 3 });
    const job = T.trainer(net, data, { steps: 240, batch: 32, lr: 0.1, schedule: [[170, 0.01]], seed: 5, every: 20 });
    T.run([job], function () { status.innerHTML = 'training the toy network: step ' + job.step + ' / 240 · running training error ' + (100 * job.errAvg).toFixed(0) + '%'; },
      function () { net.freeze(T.rows(data, ML.range(500))); const e = T.evaluate(net, d.test); status.innerHTML = 'trained for 240 steps · test error on 500 new digits: <b>' + (100 * e.err).toFixed(1) + '%</b> (toy)'; draw(); });
    function draw() {
      if (!net || job.step < 240) return;
      const te = d.test, X = T.rows(te, [idx]), maps = [];
      const lg = net.forward(X, 'eval', null, maps), p = ML.softmax(Array.from(lg.data));
      view.innerHTML = '';
      const row = el('div', { style: 'display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start' }, view);
      const b0 = el('div', null, row); U.grid(b0, te.X.subarray(idx * 64, idx * 64 + 64), 8, 8, { cell: 11, signed: true, max: 1.2, label: 'input' });
      [0, 1, 2, 3].forEach(function (mi) {
        const m = maps[mi], P = m.s.H * m.s.W, box = el('div', null, row);
        el('div', { class: 'note', text: m.name + ': ' + m.s.C + ' maps of ' + m.s.H + '×' + m.s.W }, box);
        const g = el('div', { style: 'display:grid;grid-template-columns:repeat(6,auto);gap:3px' }, box);
        for (let c = 0; c < m.s.C; c++) { const cell = el('div', null, g); U.grid(cell, m.t.data.subarray(c * P, (c + 1) * P), m.s.H, m.s.W, { cell: mi === 0 ? 5 : mi === 1 ? 8 : mi === 2 ? 10 : 12, maxw: 48 }); }
      });
      const best = ML.argmax(p);
      out.innerHTML = 'true digit <b>' + te.y[idx] + '</b> · network says <b>' + best + '</b> with probability ' + (100 * p[best]).toFixed(0) + '%. Early maps respond to strokes at each position; later maps are smaller and more abstract; the average pool keeps one number per channel (24 numbers) for the final layer.';
    }
  };
  T.cnnData = function () { if (!T._cd) T._cd = { train: T.glyphs(2000, 11, { noise: 0.25 }), test: T.glyphs(500, 12, { noise: 0.25 }) }; return T._cd; };

  // ---------------------------------------------------------------- ch-cnn: FLOPs of VGG-19 vs the 34-layer plain net
  T.vgg19Layers = function () {
    const L = [], cfg = [[64, 2], [128, 2], [256, 4], [512, 4], [512, 4]]; let H = 224, C = 3;
    cfg.forEach(function (st, i) {
      for (let j = 0; j < st[1]; j++) { L.push({ name: 'conv' + (i + 1) + '_' + (j + 1), kind: 'conv', k: 3, cin: C, cout: st[0], outH: H, params: 9 * C * st[0] + st[0], flops: H * H * 9 * C * st[0], group: 'stage ' + (i + 1) }); C = st[0]; }
      H = H / 2; L.push({ name: 'pool', kind: 'pool', outH: H, params: 0, flops: 0, group: 'stage ' + (i + 1) });
    });
    [[7 * 7 * 512, 4096], [4096, 4096], [4096, 1000]].forEach(function (f, i) { L.push({ name: 'fc' + (6 + i), kind: 'fc', params: f[0] * f[1] + f[1], flops: f[0] * f[1], group: 'fc' }); });
    return L;
  };
  W.flopCount = function (host) {
    Book.head(host, 'Where the work goes: VGG-19 against the 34-layer plain network',
      'Every bar is one stage of the network; its height is the number of multiply-adds (FLOPs, as the paper counts them) for one 224 × 224 image. Hover a bar for its parameter count.');
    const ctl = el('div', { class: 'controls' }, host);
    let which = 'both';
    Book.select(ctl, 'show', [{ value: 'both', label: 'both networks' }, { value: 'vgg', label: 'VGG-19' }, { value: 'p34', label: '34-layer plain' }], which, function (v) { which = v; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    function groups(L) { const g = {}; L.forEach(function (l) { g[l.group] = g[l.group] || { flops: 0, params: 0 }; g[l.group].flops += l.flops; g[l.group].params += l.params; }); return g; }
    function draw() {
      const v = groups(T.vgg19Layers()), p = groups(T.archLayers(34, { plain: true, bn: false }));
      const bars = [];
      if (which !== 'p34') Object.keys(v).forEach(function (k) { bars.push({ label: 'VGG\n' + k, value: v[k].flops / 1e9, color: U.C.plain, note: (v[k].flops / 1e9).toFixed(2) }); });
      if (which !== 'vgg') Object.keys(p).forEach(function (k) { bars.push({ label: '34\n' + k, value: p[k].flops / 1e9, color: U.C.res, note: (p[k].flops / 1e9).toFixed(2) }); });
      Book.chart.bars(plot, bars, { ylabel: 'GFLOPs (10⁹ multiply-adds)', height: 240, mb: 40 });
      const vt = T.archTotals(T.vgg19Layers().map(function (l) { return Object.assign({ bn: 0 }, l); })), pt = T.archTotals(T.archLayers(34, { plain: true, bn: false }));
      out.innerHTML = 'VGG-19: <b>' + (vt.flops / 1e9).toFixed(2) + ' G</b> multiply-adds, ' + T.big(vt.params) + ' parameters · 34-layer plain: <b>' + (pt.flops / 1e9).toFixed(2) + ' G</b>, ' + T.big(pt.params) + ' parameters · ratio <b>' + (100 * pt.flops / vt.flops).toFixed(0) + '%</b>. The paper: 19.6 G vs 3.6 G, “only 18%” (§3.3). Most of VGG’s work is in its wide 224² and 112² layers; most of its parameters are in its fully-connected layers.';
    }
    draw();
  };
})();
