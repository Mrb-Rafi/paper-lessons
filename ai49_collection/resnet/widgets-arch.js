/* Widgets for Part V (the ResNet architectures and recipe) of the ResNet book. */
(function () {
  'use strict';
  const Book = window.Book, W = Book.widgets, el = Book.el, S = Book.svg, ML = window.ML, NN = window.NN, T = window.RNT, U = window.RNU;
  const GROUPS = ['conv1', 'conv2_x', 'conv3_x', 'conv4_x', 'conv5_x', 'fc'];
  const GCOL = { conv1: U.stage[0], conv2_x: U.stage[1], conv3_x: U.stage[2], conv4_x: U.stage[3], conv5_x: U.stage[4], fc: U.stage[6] };

  // ---------------------------------------------------------------- Fig. 3: three columns
  W.fig3Explorer = function (host) {
    Book.head(host, 'Figure 3, rebuilt: VGG-19, a 34-layer plain net, and ResNet-34',
      'Each box is one weighted layer, coloured by map size (224, 112, 56, 28, 14, 7). Curved lines are shortcuts: solid = identity, dashed = where dimensions increase. Hover or tap a box for its shape. The FLOPs are computed from the layers, for one 224 × 224 image.');
    const view = el('div', { class: 'scroll-x' }, host), out = U.out(host);
    const vgg = T.vgg19Layers().filter(function (l) { return l.kind !== 'pool'; });
    const p34 = T.archLayers(34, { plain: true }).filter(function (l) { return l.kind === 'conv' || l.kind === 'fc'; });
    const sizeCol = { 224: U.stage[0], 112: U.stage[1], 56: U.stage[2], 28: U.stage[3], 14: U.stage[4], 7: U.stage[5], 1: U.stage[6] };
    const bh = 11, gap = 3, colW = 150, H = 40 + 36 * (bh + gap) + 20;
    const svg = S('svg', { viewBox: '0 0 ' + (3 * colW + 40) + ' ' + H, style: 'min-width:480px' }, view);
    function column(x, layers, title, residual) {
      S('text', { x: x + 55, y: 16, 'text-anchor': 'middle', 'font-size': 12.5, 'font-weight': 700, class: 'svg-text', text: title }, svg);
      const ys = [];
      layers.forEach(function (l, i) {
        const y = 28 + i * (bh + gap), h = l.kind === 'fc' ? 1 : l.outH;
        ys.push(y);
        const r = S('rect', { x: x, y: y, width: 110, height: bh, rx: 3, fill: sizeCol[h] || U.stage[6], opacity: 0.85 }, svg);
        const t = S('title', null, r); t.textContent = l.kind === 'fc' ? 'fc ' + (l.cout || '') + ' (' + T.comma(l.params) + ' weights)' : l.k + '×' + l.k + ' conv, ' + l.cout + ' filters' + (l.stride === 2 ? ', stride 2' : '') + ' → ' + l.outH + '×' + l.outH;
        r.addEventListener('click', function () { out.innerHTML = title + ', layer ' + (i + 1) + ': ' + t.textContent; });
        S('text', { x: x + 55, y: y + bh - 2, 'text-anchor': 'middle', 'font-size': 8.5, fill: '#fff', text: l.kind === 'fc' ? 'fc' : l.k + '×' + l.k + ', ' + l.cout + (l.stride === 2 ? ', /2' : ''), 'pointer-events': 'none' }, svg);
      });
      if (residual) {
        for (let i = 1; i + 1 < layers.length - 1; i += 2) {
          const y0 = ys[i] - gap / 2, y1 = ys[i + 2] - gap / 2, dashed = layers[i].stride === 2;
          S('path', { d: 'M ' + (x + 112) + ' ' + y0 + ' C ' + (x + 132) + ' ' + y0 + ', ' + (x + 132) + ' ' + y1 + ', ' + (x + 112) + ' ' + y1, fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.2, 'stroke-dasharray': dashed ? '3 2' : '' }, svg);
        }
      }
    }
    column(10, vgg, 'VGG-19', false); column(10 + colW, p34, '34-layer plain', false); column(10 + 2 * colW, p34, '34-layer residual', true);
    const vt = T.archTotals(vgg), pt = T.archTotals(T.archLayers(34, { plain: true })), rL = T.archLayers(34, { shortcut: 'A' }), rt = T.archTotals(rL), adds = rL.reduce(function (a, l) { return a + (l.kind === 'add' ? l.flops : 0); }, 0);
    out.innerHTML = 'multiply-adds: VGG-19 <b>' + (vt.flops / 1e9).toFixed(1) + ' G</b> (paper: 19.6 G) · plain-34 <b>' + (pt.flops / 1e9).toFixed(2) + ' G</b> · ResNet-34 <b>' + (rt.flops / 1e9).toFixed(2) + ' G</b> (paper: 3.6 G for both; the 16 shortcuts add only ' + T.comma(adds) + ' additions, which the paper calls negligible). Weighted layers: 19, 34, 34.';
  };

  // ---------------------------------------------------------------- Fig. 5: block builder
  W.blockBuilder = function (host) {
    Book.head(host, 'Basic block or bottleneck block?',
      'Fig. 5 compares a basic block on 64-d maps with a bottleneck block on 256-d maps, both on 56 × 56 maps. Choose a block and a stage; compare weights and multiply-adds, and see what a projection shortcut would cost.');
    const st = { kind: 'bottleneck', stage: 0, sc: 'identity' };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'block', [{ value: 'basic', label: 'basic (ResNet-18/34)' }, { value: 'bottleneck', label: 'bottleneck (ResNet-50/101/152)' }], st.kind, function (v) { st.kind = v; draw(); });
    Book.select(ctl, 'stage', [{ value: 0, label: 'conv2_x (56×56)' }, { value: 1, label: 'conv3_x (28×28)' }, { value: 2, label: 'conv4_x (14×14)' }, { value: 3, label: 'conv5_x (7×7)' }], st.stage, function (v) { st.stage = +v; draw(); });
    Book.select(ctl, 'shortcut', [{ value: 'identity', label: 'identity' }, { value: 'projection', label: '1×1 projection' }], st.sc, function (v) { st.sc = v; draw(); });
    const view = el('div', { class: 'scroll-x' }, host), out = U.out(host);
    function draw() {
      const w = [64, 128, 256, 512][st.stage], h = [56, 28, 14, 7][st.stage];
      const layers = st.kind === 'basic' ? [[3, w, w], [3, w, w]] : [[1, 4 * w, w], [3, w, w], [1, w, 4 * w]];
      const dIn = st.kind === 'basic' ? w : 4 * w;
      view.innerHTML = '';
      const svg = S('svg', { viewBox: '0 0 520 ' + (70 + layers.length * 52), style: 'min-width:420px' }, view);
      S('text', { x: 150, y: 18, 'text-anchor': 'middle', 'font-size': 13, 'font-weight': 700, class: 'svg-text', text: dIn + '-d input, ' + h + '×' + h }, svg);
      let tp = 0, tf = 0;
      layers.forEach(function (l, i) {
        const y = 30 + i * 52, p = l[0] * l[0] * l[1] * l[2], f = h * h * p; tp += p; tf += f;
        const wd = 60 + 140 * Math.sqrt(l[2] / (4 * w));
        S('rect', { x: 150 - wd / 2, y: y, width: wd, height: 30, rx: 7, fill: U.rgba(U.stage[1 + st.stage], 0.25), stroke: U.stage[1 + st.stage], 'stroke-width': 1.6 }, svg);
        S('text', { x: 150, y: y + 19, 'text-anchor': 'middle', 'font-size': 12.5, class: 'svg-text', text: l[0] + '×' + l[0] + ', ' + l[2] + (i < layers.length - 1 ? '  → BN, ReLU' : '  → BN') }, svg);
        S('text', { x: 280, y: y + 19, 'font-size': 11.5, class: 'svg-text', style: 'font-family:var(--font-mono)', text: T.comma(p) + ' weights · ' + T.big(f) + 'FLOPs' }, svg);
      });
      const yEnd = 30 + layers.length * 52;
      S('text', { x: 150, y: yEnd + 14, 'text-anchor': 'middle', 'font-size': 12.5, class: 'svg-text', text: '⊕ add shortcut, then ReLU' }, svg);
      S('path', { d: 'M 40 24 C 10 24, 10 ' + (yEnd + 10) + ', 120 ' + (yEnd + 10), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2, 'stroke-dasharray': st.sc === 'projection' ? '5 3' : '' }, svg);
      const sp = st.sc === 'projection' ? dIn * dIn : 0, sf = h * h * sp;
      S('text', { x: 14, y: yEnd + 30, 'font-size': 11.5, class: 'svg-text', text: st.sc === 'projection' ? 'projection 1×1, ' + dIn + ': ' + T.comma(sp) + ' weights' : 'identity shortcut: 0 weights' }, svg);
      const other = st.kind === 'basic' ? 2 * 9 * w * w : (4 * w * w + 9 * w * w + 4 * w * w);
      out.innerHTML = 'block total: <b>' + T.comma(tp + sp) + '</b> weights, <b>' + T.big(tf + sf) + 'FLOPs</b> on ' + h + '×' + h + ' maps. ' +
        (st.sc === 'projection' ? 'The projection adds ' + Math.round(100 * sp / tp) + '% to this block' + (st.kind === 'bottleneck' ? ': connected to the two ' + dIn + '-d ends, it roughly doubles the block (the paper’s reason to keep identity shortcuts in bottlenecks).' : '.') : '') +
        ' Fig. 5’s pair: basic 64-d = ' + T.comma(2 * 9 * 64 * 64) + ' weights, bottleneck 256-d = ' + T.comma(4 * 64 * 64 + 9 * 64 * 64 + 4 * 64 * 64) + ': similar cost, but the bottleneck’s block is 3 layers deep and 4× wider at its ends. (This stage, the other kind: ' + T.comma(other) + '.)';
    }
    draw();
  };

  // ---------------------------------------------------------------- Table 1 explorer
  W.table1Explorer = function (host) {
    Book.head(host, 'Table 1, computed: parameters and FLOPs of every ResNet',
      'Pick a depth. The table is computed layer by layer from Table 1’s recipe (weights of every convolution, the BN scale and shift of every channel, the final fully-connected layer). The parameter totals match torchvision’s published counts exactly.');
    const st = { d: 50, sc: 'B', stride: '1x1', plain: false };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'depth', [18, 34, 50, 101, 152], st.d, function (v) { st.d = +v; draw(); });
    Book.select(ctl, 'shortcuts', [{ value: 'A', label: 'A (zero-pad)' }, { value: 'B', label: 'B (projection when needed)' }, { value: 'C', label: 'C (all projections)' }, { value: 'plain', label: 'none (plain net)' }], st.sc, function (v) { st.sc = v; draw(); });
    Book.select(ctl, 'stride-2 in bottleneck on', [{ value: '1x1', label: 'first 1×1 (paper, Caffe models)' }, { value: '3x3', label: '3×3 (torchvision “v1.5”)' }], st.stride, function (v) { st.stride = v; draw(); });
    const tbl = el('div', { class: 'table-wrap' }, host), plot = el('div', null, host), out = U.out(host);
    const PAPER = { 18: 1.8, 34: 3.6, 50: 3.8, 101: 7.6, 152: 11.3 }, TV = { 18: 11689512, 34: 21797672, 50: 25557032, 101: 44549160, 152: 60192808 };
    function draw() {
      const L = T.archLayers(st.d, { shortcut: st.sc === 'plain' ? 'A' : st.sc, plain: st.sc === 'plain', strideOn: st.stride }), tot = T.archTotals(L);
      const g = {}; L.forEach(function (l) { const k = l.group === 'conv2_x' && l.kind === 'pool' ? 'conv2_x' : l.group; g[k] = g[k] || { p: 0, f: 0, n: 0, out: l.outH, c: l.cout }; g[k].p += l.params + l.bn; if (l.kind !== 'add' && l.kind !== 'pool') g[k].f += l.flops; if (l.kind === 'conv' || l.kind === 'fc') g[k].n++; g[k].out = l.outH; g[k].c = l.cout; });
      tbl.innerHTML = '<table class="data"><tr><th>stage</th><th>output</th><th>weighted layers</th><th>parameters</th><th>multiply-adds</th></tr>' +
        GROUPS.map(function (k) { const v = g[k]; return '<tr><td>' + k + '</td><td>' + (k === 'fc' ? '1×1, 1000' : v.out + '×' + v.out + ', ' + v.c) + '</td><td>' + v.n + '</td><td>' + T.comma(v.p) + '</td><td>' + (v.f / 1e9).toFixed(3) + ' G</td></tr>'; }).join('') +
        '<tr><td><b>total</b></td><td></td><td><b>' + tot.weighted + '</b></td><td><b>' + T.comma(tot.params) + '</b></td><td><b>' + (tot.flops / 1e9).toFixed(2) + ' G</b></td></tr></table>';
      Book.chart.bars(plot, GROUPS.map(function (k) { return { label: k, value: g[k].f / 1e9, color: GCOL[k], note: (g[k].f / 1e9).toFixed(2) }; }), { ylabel: 'GFLOPs per stage', height: 190 });
      const match = st.sc === 'B' ? (tot.params === TV[st.d] ? ' <span class="ok">= torchvision’s ' + T.comma(TV[st.d]) + ' ✓</span>' : ' (torchvision, option B: ' + T.comma(TV[st.d]) + ')') : '';
      out.innerHTML = 'ResNet-' + st.d + ': <b>' + T.comma(tot.params) + '</b> parameters' + match + ' · <b>' + (tot.flops / 1e9).toFixed(2) + ' G</b> multiply-adds (paper, Table 1: ' + PAPER[st.d] + ' × 10⁹) · ' + tot.proj + ' projection shortcuts. ' +
        (st.d >= 50 && st.stride === '3x3' ? 'Moving the stride to the 3×3 makes the 3×3 run on the larger map, which costs more (torchvision lists 4.09 G for ResNet-50).' : '');
    }
    draw();
  };

  // ---------------------------------------------------------------- the recipe: augmentation
  W.augmentDemo = function (host) {
    Book.head(host, 'Scale augmentation, random crops and flips',
      'Each training step sees a new version of each picture. The shorter side is resized to a random s in [256, 480], then a random 224 × 224 crop is cut out, flipped half the time (§3.4). Press the button to draw another version.');
    const ctl = el('div', { class: 'controls' }, host);
    let seed = 1;
    Book.button(ctl, 'Another sample ↻', function () { seed++; draw(); }, 'primary');
    const view = el('div', null, host), out = U.out(host);
    function scene(ctx, x0, y0, w, h, flip) {           // a simple drawn "photo": sky, ground, sun, house
      ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, w, h); ctx.clip();
      if (flip) { ctx.translate(2 * x0 + w, 0); ctx.scale(-1, 1); }
      ctx.fillStyle = '#9ecae1'; ctx.fillRect(x0, y0, w, h * 0.62); ctx.fillStyle = '#74c476'; ctx.fillRect(x0, y0 + h * 0.62, w, h * 0.38);
      ctx.fillStyle = '#fdd835'; ctx.beginPath(); ctx.arc(x0 + w * 0.8, y0 + h * 0.2, h * 0.09, 0, 7); ctx.fill();
      ctx.fillStyle = '#c0392b'; ctx.fillRect(x0 + w * 0.22, y0 + h * 0.45, w * 0.28, h * 0.25);
      ctx.fillStyle = '#6d4c41'; ctx.beginPath(); ctx.moveTo(x0 + w * 0.18, y0 + h * 0.46); ctx.lineTo(x0 + w * 0.36, y0 + h * 0.3); ctx.lineTo(x0 + w * 0.54, y0 + h * 0.46); ctx.fill();
      ctx.fillStyle = '#4e342e'; ctx.fillRect(x0 + w * 0.32, y0 + h * 0.58, w * 0.06, h * 0.12);
      ctx.restore();
    }
    function draw() {
      const r = ML.rng(seed * 31), s = Math.round(r.uniform(256, 480)), ar = 4 / 3, Wi = Math.round(s * ar), Hi = s;
      const cx = Math.round(r.uniform(0, Wi - 224)), cy = Math.round(r.uniform(0, Hi - 224)), flip = r.next() < 0.5;
      view.innerHTML = '';
      const cv = el('canvas', { width: 640, height: 260, style: 'width:100%;max-width:640px;border-radius:8px;border:1px solid var(--line)' }, view), ctx = cv.getContext('2d');
      const k = 240 / 480;                              // pixels per image pixel in the drawing
      scene(ctx, 10, 10, Wi * k, Hi * k, false);
      ctx.strokeStyle = '#f6ad55'; ctx.lineWidth = 3; ctx.strokeRect(10 + cx * k, 10 + cy * k, 224 * k, 224 * k);
      // the crop, enlarged, as the network sees it (same zoom of the scene)
      ctx.save(); ctx.beginPath(); ctx.rect(400, 10, 224, 224); ctx.clip();
      const z = 224 / (224 * k);
      if (flip) { ctx.translate(400 + 224, 0); ctx.scale(-1, 1); ctx.translate(-400, 0); }
      scene(ctx, 400 - cx * k * z, 10 - cy * k * z, Wi * k * z, Hi * k * z, false); ctx.restore();
      ctx.strokeStyle = '#f6ad55'; ctx.strokeRect(400, 10, 224, 224);
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--ink') || '#222'; ctx.font = '12px sans-serif';
      ctx.fillText('resized photo ' + Wi + ' × ' + Hi, 12, 254); ctx.fillText('the 224 × 224 crop' + (flip ? ', flipped' : ''), 402, 254);
      out.innerHTML = 'shorter side s = <b>' + s + '</b> (random in [256, 480]) · crop at (' + cx + ', ' + cy + ') · ' + (flip ? '<b>flipped</b>' : 'not flipped') + ' · then the per-pixel mean is subtracted and the standard colour augmentation of AlexNet is applied (not drawn). On CIFAR-10: pad 4 pixels on each side, random 32 × 32 crop, random flip (§4.2).';
    }
    draw();
  };

  // ---------------------------------------------------------------- the recipe: learning-rate schedules
  W.lrSchedule = function (host) {
    Book.head(host, 'The learning-rate schedule: divide by 10',
      'Both recipes start at 0.1 and divide by 10. On ImageNet the drops happen when the error stops improving (up to 60 × 10⁴ iterations); on CIFAR-10 at fixed points, 32k and 48k, ending at 64k. The 110-layer CIFAR net warms up at 0.01 first.');
    let which = 'cifar';
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'recipe', [{ value: 'cifar', label: 'CIFAR-10 (§4.2)' }, { value: 'warm', label: 'CIFAR-10, 110 layers with warm-up' }, { value: 'imagenet', label: 'ImageNet (§3.4), drops ≈ read off Fig. 4' }], which, function (v) { which = v; draw(); });
    const plot = el('div', null, host), out = U.out(host);
    function draw() {
      let pts, xmax, xt, lab;
      if (which === 'imagenet') { pts = [[0, 0.1], [15, 0.1], [15, 0.01], [30, 0.01], [30, 0.001], [60, 0.001]]; xmax = 60; xt = [0, 15, 30, 45, 60]; lab = 'iterations (×10⁴)'; }
      else { pts = which === 'warm' ? [[0, 0.01], [0.4, 0.01], [0.4, 0.1], [32, 0.1], [32, 0.01], [48, 0.01], [48, 0.001], [64, 0.001]] : [[0, 0.1], [32, 0.1], [32, 0.01], [48, 0.01], [48, 0.001], [64, 0.001]]; xmax = 64; xt = [0, 16, 32, 48, 64]; lab = 'iterations (×1000)'; }
      U.logLines(plot, [{ color: 'var(--accent)', points: pts, marker: 'none' }], { ymin: 1e-4, ymax: 1, xmin: 0, xmax: xmax, xticks: xt, xlabel: lab, ylabel: 'learning rate', height: 200, noLegend: true });
      out.innerHTML = which === 'imagenet' ? 'ImageNet: mini-batch 256, momentum 0.9, weight decay 0.0001. The paper says only “divided by 10 when the error plateaus”; the drop points shown are approximate, read off the steps in Fig. 4.'
        : which === 'warm' ? '110 layers: 0.01 until the training error is below 80% (about 400 iterations), then back to 0.1 and the usual schedule. With 0.1 from the start it still converges, after several epochs, to similar accuracy (footnote 5).'
          : 'CIFAR-10: mini-batch 128 on two GPUs, momentum 0.9, weight decay 0.0001; schedule chosen on a 45k/5k train/validation split. 64k iterations × 128 images ≈ 164 passes over the 50k training images.';
    }
    draw();
  };

  // ---------------------------------------------------------------- the recipe: 10-crop testing
  W.tenCrop = function (host) {
    Book.head(host, 'Ten-crop testing',
      'For the comparison tables, each test image (shorter side 256) is judged ten times: four corner crops, the centre crop, and the same five mirrored. The ten predictions are averaged. Click a crop to highlight it.');
    let sel = 4;
    const view = el('div', null, host), out = U.out(host);
    const names = ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'centre'];
    function draw() {
      view.innerHTML = '';
      const k = 0.9, svg = S('svg', { viewBox: '0 0 640 260' }, view), Wi = 341, Hi = 256;
      [0, 1].forEach(function (f) {
        const x0 = 10 + f * 320;
        S('rect', { x: x0, y: 10, width: Wi * k, height: Hi * k, fill: 'var(--accent-soft)', stroke: 'var(--line)' }, svg);
        S('text', { x: x0 + 4, y: 248, 'font-size': 12, class: 'svg-text', text: f ? 'mirrored image' : 'image, 341 × 256' }, svg);
        const pos = [[0, 0], [Wi - 224, 0], [0, Hi - 224], [Wi - 224, Hi - 224], [(Wi - 224) / 2, (Hi - 224) / 2]];
        pos.forEach(function (p, i) {
          const id = f * 5 + i, on = id === sel;
          const rct = S('rect', { x: x0 + p[0] * k, y: 10 + p[1] * k, width: 224 * k, height: 224 * k, fill: on ? U.rgba('#d69e2e', 0.25) : 'none', stroke: on ? '#d69e2e' : U.stage[i], 'stroke-width': on ? 3 : 1.3, 'stroke-dasharray': on ? '' : '4 3' }, svg);
          rct.style.cursor = 'pointer'; rct.addEventListener('click', function () { sel = id; draw(); });
        });
      });
      out.innerHTML = 'crop ' + (sel + 1) + ' of 10: <b>' + names[sel % 5] + (sel >= 5 ? ', mirrored' : '') + '</b>. The final scores are the average of the ten softmax outputs. For the best single-model and ensemble results the paper instead runs the network fully-convolutionally on whole images at 5 scales (shorter side 224, 256, 384, 480, 640) and averages.';
    }
    draw();
  };

  // ---------------------------------------------------------------- the official models' layer names
  W.layerNames = function (host) {
    Book.head(host, 'Reading the official model files',
      'The authors’ Caffe files name every layer: <code>res3a_branch2a</code> is stage 3 (conv3_x), block a, residual branch, first layer; <code>branch1</code> is the projection shortcut. Pick a block to see its layers as the file lists them, and where torchvision puts the stride.');
    const st = { d: 50, stage: 3, blk: 0 };
    const ctl = el('div', { class: 'controls' }, host);
    Book.select(ctl, 'model', [{ value: 50, label: 'ResNet-50' }, { value: 101, label: 'ResNet-101' }, { value: 152, label: 'ResNet-152' }], st.d, function (v) { st.d = +v; draw(); });
    Book.select(ctl, 'stage', [2, 3, 4, 5], st.stage, function (v) { st.stage = +v; st.blk = 0; draw(); });
    const bSel = Book.slider(ctl, 'block', 0, 22, 1, st.blk, function (v) { st.blk = v; draw(); }, function (v) { return v; });
    const view = el('div', { class: 'table-wrap' }, host), out = U.out(host);
    function name(i, n) { if (n <= 3 || st.d === 50) return String.fromCharCode(97 + i); return i === 0 ? 'a' : 'b' + i; }
    function draw() {
      const nb = T.ARCH[st.d].n[st.stage - 2], w = [64, 128, 256, 512][st.stage - 2];
      bSel.max = nb - 1; if (st.blk > nb - 1) { st.blk = nb - 1; bSel.value = st.blk; }
      bSel.parentNode.querySelector('.val').textContent = st.blk;
      const nm = 'res' + st.stage + name(st.blk, nb), first = st.blk === 0, stride = first && st.stage > 2 ? 2 : 1;
      const rows = [];
      if (first) rows.push([nm + '_branch1', '1×1, ' + 4 * w + (stride === 2 ? ', stride 2' : ''), 'projection shortcut (option B)', '']);
      rows.push([nm + '_branch2a', '1×1, ' + w + (stride === 2 ? ', <b>stride 2</b>' : ''), 'reduce', stride === 2 ? 'torchvision: stride 1 here' : '']);
      rows.push([nm + '_branch2b', '3×3, ' + w, 'bottleneck', stride === 2 ? 'torchvision: <b>stride 2</b> here' : '']);
      rows.push([nm + '_branch2c', '1×1, ' + 4 * w, 'restore', '']);
      rows.push([nm, 'Eltwise (sum)', first ? 'branch1 + branch2c' : 'input + branch2c', '']);
      rows.push([nm + '_relu', 'ReLU', 'after the addition', '']);
      view.innerHTML = '<table class="data"><tr><th>layer name</th><th>operation</th><th>role</th><th>note</th></tr>' + rows.map(function (r) { return '<tr><td><code>' + r[0] + '</code></td><td>' + r[1] + '</td><td>' + r[2] + '</td><td>' + r[3] + '</td></tr>'; }).join('') + '</table>';
      out.innerHTML = 'Every convolution is followed by <code>bn…</code> (BatchNorm with stored mean and variance) and <code>scale…</code> (the learned γ and β), then ReLU, except the last one of the branch, whose ReLU comes after the sum. Stage ' + st.stage + ' of ResNet-' + st.d + ' has ' + nb + ' blocks' + (nb > 3 && st.d !== 50 ? ', named a, b1, b2, …' : ', named a, b, c, …') + '.';
    }
    draw();
  };
})();
