/* Small drawing helpers shared by this book's widgets (window.RNU): pixel grids, legends, line charts with legends and
   clamping, a log-scale line chart, and a stage-coloured palette. Colours come from CSS variables or mid-tone hues that read
   on both light and dark backgrounds. */
(function () {
  'use strict';
  const Book = window.Book, U = window.RNU = {};
  const S = function (t, a, p) { return Book.svg(t, a, p); };
  U.C = { plain: '#c0392b', res: '#2b6cb0', plain2: '#e67e22', res2: '#1f7a7a', grey: '#718096', green: '#2f855a', purple: '#6b46c1', gold: '#b7791f', pink: '#d53f8c' };
  U.stage = ['#6b46c1', '#2b6cb0', '#1f7a7a', '#2f855a', '#b7791f', '#c0392b', '#718096'];
  U.rgba = function (hex, a) { const n = parseInt(hex.replace('#', ''), 16); return 'rgba(' + (n >> 16 & 255) + ',' + (n >> 8 & 255) + ',' + (n & 255) + ',' + a + ')'; };

  // a grid of numbers as coloured cells. vals: flat array (rows·cols). o: {cell, max, signed, numbers, digits, hl:[i…], hl2:[i…], onclick(i), label}
  U.grid = function (parent, vals, rows, cols, o) {
    o = o || {}; const c = o.cell || 26, pad = 1, W = cols * c, H = rows * c + (o.label ? 16 : 0);
    const svg = S('svg', { viewBox: '0 0 ' + (W + 2) + ' ' + (H + 2), style: 'width:100%;max-width:' + (o.maxw || W + 2) + 'px;display:block' }, parent);
    let max = o.max; if (!max) { max = 1e-9; for (let i = 0; i < vals.length; i++) max = Math.max(max, Math.abs(vals[i])); }
    const top = o.label ? 16 : 0;
    if (o.label) S('text', { x: 1, y: 11, 'font-size': 11, class: 'svg-text', text: o.label }, svg);
    for (let r = 0; r < rows; r++) for (let q = 0; q < cols; q++) {
      const i = r * cols + q, v = vals[i], t = Math.max(-1, Math.min(1, v / max));
      let fill;
      if (o.signed) fill = t >= 0 ? U.rgba('#c0392b', 0.08 + 0.85 * t) : U.rgba('#2b6cb0', 0.08 + 0.85 * -t);
      else fill = U.rgba(o.hue || '#1a202c', 0.04 + 0.9 * Math.max(0, t));
      const rect = S('rect', { x: 1 + q * c + pad / 2, y: 1 + top + r * c + pad / 2, width: c - pad, height: c - pad, rx: 2, fill: o.hue || o.signed ? fill : 'var(--panel)', stroke: 'var(--line)' }, svg);
      if (!o.signed && !o.hue) { S('rect', { x: 1 + q * c + pad / 2, y: 1 + top + r * c + pad / 2, width: c - pad, height: c - pad, rx: 2, fill: 'var(--ink)', 'fill-opacity': (0.04 + 0.86 * Math.max(0, t)).toFixed(3), 'pointer-events': 'none' }, svg); }
      if (o.hl && o.hl.indexOf(i) >= 0) S('rect', { x: 1 + q * c + 1, y: 1 + top + r * c + 1, width: c - 2, height: c - 2, rx: 2, fill: 'none', stroke: o.hlColor || '#d69e2e', 'stroke-width': 2.4, 'pointer-events': 'none' }, svg);
      if (o.hl2 && o.hl2.indexOf(i) >= 0) S('rect', { x: 1 + q * c + 3, y: 1 + top + r * c + 3, width: c - 6, height: c - 6, rx: 2, fill: 'none', stroke: '#2b6cb0', 'stroke-width': 2, 'pointer-events': 'none' }, svg);
      if (o.numbers) S('text', { x: 1 + q * c + c / 2, y: 1 + top + r * c + c / 2 + 4, 'text-anchor': 'middle', 'font-size': Math.min(11, c * 0.36), class: 'svg-text', style: 'font-family:var(--font-mono);paint-order:stroke;stroke:var(--panel);stroke-width:2.5px', text: o.fmt ? o.fmt(v) : (Math.abs(v) < 0.005 ? '0' : (+v.toFixed(o.digits == null ? 1 : o.digits)).toString().replace('-', '−')), 'pointer-events': 'none' }, svg);
      if (o.onclick) { rect.style.cursor = 'pointer'; rect.addEventListener('click', function () { o.onclick(i); }); }
    }
    return svg;
  };

  U.legend = function (parent, items) {
    const d = Book.el('div', { style: 'display:flex;flex-wrap:wrap;gap:6px 14px;font:12.5px var(--font-ui);margin:4px 0 2px' }, parent);
    items.forEach(function (it) {
      const s = Book.el('span', { style: 'display:inline-flex;align-items:center;gap:6px' }, d);
      Book.el('span', { style: 'display:inline-block;width:18px;height:0;border-top:3px ' + (it.dash ? 'dashed ' : 'solid ') + it.color }, s);
      Book.el('span', { text: it.name }, s);
    });
    return d;
  };

  // line chart: clamps to [ymin, ymax] and shows a legend instead of end labels; returns {svg, X, Y}
  U.lines = function (host, series, o) {
    o = Object.assign({}, o || {}); host.innerHTML = '';
    const lo = o.ymin, hi = o.ymax;
    const ss = series.map(function (s) {
      return Object.assign({}, s, { name: o.endLabels ? s.name : '', points: s.points.filter(function (p) { return isFinite(p[1]); }).map(function (p) { return [p[0], lo != null && hi != null ? Math.max(lo, Math.min(hi, p[1])) : p[1]]; }) });
    }).filter(function (s) { return s.points.length; });
    if (!ss.length) ss.push({ points: [[o.xmin || 0, lo || 0]], color: 'transparent', marker: 'none' });
    const chartHost = Book.el('div', null, host);
    const r = Book.chart.lines(chartHost, ss, o);
    if (!o.endLabels && !o.noLegend) U.legend(host, series.filter(function (s) { return s.name; }).map(function (s) { return { name: s.name, color: s.color, dash: !!s.dash }; }));
    return r;
  };
  // log10-scale y; values must be > 0
  U.logLines = function (host, series, o) {
    o = Object.assign({}, o || {});
    const lo = o.ymin, hi = o.ymax;
    const ss = series.map(function (s) { return Object.assign({}, s, { points: s.points.map(function (p) { return [p[0], Math.log10(Math.max(lo, Math.min(hi, p[1])))]; }) }); });
    const dec = Math.round(Math.log10(hi) - Math.log10(lo));
    o.ymin = Math.log10(lo); o.ymax = Math.log10(hi); o.yticks = o.yticks || (dec > 7 && dec % 2 === 0 ? dec / 2 : dec);
    o.yfmt = function (v) { return U.p10(Math.round(v)); };
    return U.lines(host, ss, o);
  };
  const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  U.p10 = function (e) { if (e === 0) return '1'; if (e === 1) return '10'; return '10' + String(e).split('').map(function (ch) { return SUP[ch]; }).join(''); };

  // a horizontal bar list (long labels): rows [{label, value, color, note}]
  U.hbars = function (host, rows, o) {
    o = o || {}; host.innerHTML = '';
    const W = o.width || 640, rh = o.rowH || 24, lw = o.labelW || 170, H = rows.length * rh + 26;
    const svg = S('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' }, host);
    const min = o.min || 0, max = o.max || Math.max.apply(null, rows.map(function (r) { return r.value; })) * 1.1;
    const X = function (v) { return lw + (W - lw - 60) * (v - min) / (max - min); };
    (o.ticks || []).forEach(function (t) {
      S('line', { x1: X(t), x2: X(t), y1: 4, y2: H - 20, stroke: 'var(--line)' }, svg);
      S('text', { x: X(t), y: H - 6, 'text-anchor': 'middle', 'font-size': 11, class: 'svg-text', text: o.tickFmt ? o.tickFmt(t) : t }, svg);
    });
    rows.forEach(function (r, i) {
      const y = 6 + i * rh;
      S('text', { x: lw - 8, y: y + rh * 0.6, 'text-anchor': 'end', 'font-size': 12, class: 'svg-text', style: r.bold ? 'font-weight:700' : '', text: r.label }, svg);
      if (r.value == null) { S('text', { x: lw + 4, y: y + rh * 0.6, 'font-size': 11.5, class: 'svg-text', text: '—' }, svg); return; }
      S('rect', { x: lw, y: y + 3, width: Math.max(1, X(r.value) - lw), height: rh - 8, rx: 3, fill: r.color || 'var(--accent)' }, svg);
      S('text', { x: X(r.value) + 5, y: y + rh * 0.6, 'font-size': 11.5, 'font-weight': 600, class: 'svg-text', text: r.note != null ? r.note : r.value }, svg);
    });
    return svg;
  };

  // run an expensive function after the browser paints (so a "computing…" note shows first)
  U.later = function (fn) { setTimeout(fn, 20); };
  U.out = function (parent) { return Book.el('div', { class: 'out' }, parent); };
  U.note = function (parent, html) { return Book.el('div', { class: 'note', html: html }, parent); };
})();
