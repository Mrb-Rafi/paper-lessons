/* Collection navigation inside a book of "Modern AI in 49 Papers".
   Reads window.AI49 (../collection.js) and <body data-lesson="<slug>">, then:
     • fills <div id="lessonStrip"></div> (in the hero) with: lesson n of 49 · part · difficulty · builds on
     • fills <nav id="lessonPager"></nav> (before the footer) with previous / next lesson links
     • fills <dl class="glossary" data-terms="term a; term b"> with definitions from window.AI49_GLOSSARY
       (terms the book defines itself can be written as normal <dt>/<dd> pairs in the same list)
   Runs before core.js builds the contents, so filled glossary entries are in place when KaTeX renders. */
(function () {
  'use strict';
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function stars(n) { let s = ''; for (let i = 1; i <= 5; i++) s += i <= n ? '★' : '☆'; return s; }

  function fill() {
    const C = window.AI49, slug = document.body.getAttribute('data-lesson');
    if (!C || !slug) return;
    const L = C.lessons, i = L.findIndex(function (l) { return l.slug === slug; });
    if (i < 0) return;
    const me = L[i], part = C.parts.find(function (p) { return p.id === me.part; });
    const bySlug = {}; L.forEach(function (l) { bySlug[l.slug] = l; });
    const link = function (l) { return l.status === 'ready' ? '<a href="../' + l.slug + '/">' + esc(l.title) + '</a>' : '<a href="../#l-' + l.slug + '">' + esc(l.title) + '</a>'; };

    const strip = document.getElementById('lessonStrip');
    if (strip) {
      strip.innerHTML =
        '<span><b>Lesson ' + me.n + '</b> of ' + L.length + '</span>' +
        '<span>Part ' + me.part + ' · ' + esc(part ? part.title : '') + '</span>' +
        '<span class="diff" title="Difficulty ' + me.difficulty + ' of 5">' + stars(me.difficulty) + '</span>' +
        '<span>' + (me.builds_on.length ? 'Builds on: ' + me.builds_on.map(function (s) { return link(bySlug[s]); }).join(', ') + ' (optional)' : 'No prerequisites') + '</span>';
    }
    const pager = document.getElementById('lessonPager');
    if (pager) {
      const prev = L[i - 1], next = L[i + 1];
      const card = function (l, dir) {
        if (!l) return '<span></span>';
        const href = l.status === 'ready' ? '../' + l.slug + '/' : '../#l-' + l.slug;
        return '<a class="pager-card ' + dir + '" href="' + href + '"><span class="pl">' + (dir === 'prev' ? '← Previous' : 'Next →') + ' · Lesson ' + l.n + (l.status === 'ready' ? '' : ' · coming soon') + '</span>' +
          '<span class="pt">' + esc(l.title) + '</span><span class="pb">' + esc(l.blurb) + '</span></a>';
      };
      pager.innerHTML = card(prev, 'prev') + card(next, 'next');
    }
  }

  function glossary() {
    const G = window.AI49_GLOSSARY || {};
    document.querySelectorAll('dl.glossary[data-terms]').forEach(function (dl) {
      const own = {};
      dl.querySelectorAll('dt').forEach(function (dt) { own[dt.textContent.trim().toLowerCase()] = true; });
      const terms = dl.getAttribute('data-terms').split(';').map(function (t) { return t.trim(); }).filter(Boolean);
      terms.forEach(function (t) {
        if (own[t.toLowerCase()]) return;
        const key = Object.keys(G).find(function (k) { return k.toLowerCase() === t.toLowerCase(); });
        const dt = document.createElement('dt'), dd = document.createElement('dd');
        dt.textContent = key || t;
        dd.innerHTML = key ? G[key] : '<em>(missing from the shared glossary)</em>';
        dl.appendChild(dt); dl.appendChild(dd);
      });
      // keep the list alphabetical
      const pairs = []; let cur = null;
      Array.from(dl.children).forEach(function (n) { if (n.tagName === 'DT') { cur = [n]; pairs.push(cur); } else if (cur) cur.push(n); });
      pairs.sort(function (a, b) { return a[0].textContent.localeCompare(b[0].textContent, 'en', { sensitivity: 'base' }); });
      pairs.forEach(function (p) { p.forEach(function (n) { dl.appendChild(n); }); });
    });
  }

  // run as soon as the DOM is parsed, before core.js's DOMContentLoaded handler (this script is loaded first)
  document.addEventListener('DOMContentLoaded', function () { fill(); glossary(); });
})();
