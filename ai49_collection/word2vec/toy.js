/* The toy world of the Word2Vec Interactive Web Book (lesson 1 of Modern AI in 49 Papers).
   window.W2V holds:
     • a seeded generator for a small synthetic corpus (people, countries, adjectives) and its vocabulary;
     • a port of word2vec.c's Huffman tree (CreateBinaryTree) and of its training loop: CBOW and skip-gram with
       hierarchical softmax (the method of Mikolov et al. 2013a), plus the negative sampling and subsampling of
       the follow-up paper (Mikolov et al. 2013b), which the released code also contains;
     • linear learning-rate decay, the analogy search of compute-accuracy.c, and a toy analogy test set.
   Everything is plain JavaScript on Float64Arrays and runs in milliseconds. Depends on window.ML (shared/lib/math.js). */
(function () {
  'use strict';
  const ML = window.ML;
  const W2V = window.W2V = {};

  // ------------------------------------------------------------------ the toy world
  // People differ in three attributes: gender, royal or common, adult or young.
  W2V.people = [
    { w: 'king', g: 'm', royal: 1, young: 0 }, { w: 'queen', g: 'f', royal: 1, young: 0 },
    { w: 'prince', g: 'm', royal: 1, young: 1 }, { w: 'princess', g: 'f', royal: 1, young: 1 },
    { w: 'man', g: 'm', royal: 0, young: 0 }, { w: 'woman', g: 'f', royal: 0, young: 0 },
    { w: 'boy', g: 'm', royal: 0, young: 1 }, { w: 'girl', g: 'f', royal: 0, young: 1 }
  ];
  // Countries, their capitals and their languages.
  W2V.places = [
    { country: 'france', city: 'paris', lang: 'french', food: 'cheese' }, { country: 'italy', city: 'rome', lang: 'italian', food: 'pasta' },
    { country: 'germany', city: 'berlin', lang: 'german', food: 'bratwurst' }, { country: 'japan', city: 'tokyo', lang: 'japanese', food: 'sushi' },
    { country: 'spain', city: 'madrid', lang: 'spanish', food: 'paella' }
  ];
  // Adjectives with comparative and superlative forms, each tied to the animal it usually describes.
  W2V.adjectives = [
    { base: 'big', cmp: 'bigger', sup: 'biggest', animal: 'elephant' },
    { base: 'small', cmp: 'smaller', sup: 'smallest', animal: 'mouse' },
    { base: 'fast', cmp: 'faster', sup: 'fastest', animal: 'cheetah' },
    { base: 'tall', cmp: 'taller', sup: 'tallest', animal: 'giraffe' }
  ];

  function pick(r, arr) { return arr[r.int(arr.length)]; }

  // One sentence, as an array of words. Each template is a small grammar rule of the toy world.
  function sentence(r) {
    const kind = r.next();
    if (kind < 0.42) {                                   // ---- people
      const p = pick(r, W2V.people), he = p.g === 'm' ? 'he' : 'she', his = p.g === 'm' ? 'his' : 'her';
      const t = r.int(8);
      if (t === 0) return ['the', p.w, 'said', 'that', he, 'was', 'tired'];
      if (t === 1) return [he, 'is', 'the', p.w];
      if (t === 2) return ['the', p.w, 'and', his, 'dog', 'went', 'home'];
      if (t === 3 || t === 4) return p.royal ? pick(r, [['the', p.w, 'sat', 'on', 'the', 'throne'], ['the', p.w, 'wore', 'the', 'crown'], ['the', p.w, 'lives', 'in', 'the', 'palace'], ['the', p.w, 'rules', 'the', 'kingdom']])
                                             : pick(r, [['the', p.w, 'works', 'on', 'the', 'farm'], ['the', p.w, 'lives', 'in', 'the', 'village'], ['the', p.w, 'walked', 'to', 'the', 'market'], ['the', p.w, 'sells', 'bread']]);
      if (t === 5 || t === 6) return p.young ? pick(r, [['the', 'young', p.w, 'plays', 'at', 'school'], ['the', p.w, 'is', 'a', 'child'], ['the', 'little', p.w, 'plays', 'with', 'toys']])
                                              : pick(r, [['the', p.w, 'is', 'an', 'adult'], ['the', 'old', p.w, 'reads', 'the', 'news'], ['the', p.w, 'drinks', 'coffee']]);
      return ['the', p.w, 'smiled', 'and', he, 'laughed'];
    }
    if (kind < 0.72) {                                   // ---- countries and capitals
      const c = pick(r, W2V.places), t = r.int(10);
      if (t === 0) return [c.city, 'is', 'the', 'capital', 'of', c.country];
      if (t === 1) return ['people', 'in', c.country, 'speak', c.lang];
      if (t === 2) return ['people', 'in', c.city, 'speak', c.lang];
      if (t === 3) return ['we', 'visited', 'the', 'city', 'of', c.city];
      if (t === 4) return ['the', 'streets', 'of', c.city, 'are', 'busy'];
      if (t === 5) return ['we', 'crossed', 'the', 'border', 'of', c.country];
      if (t === 6) return [c.country, 'is', 'a', 'country', 'with', 'many', 'towns'];
      if (t === 7) return ['in', pick(r, [c.country, c.city]), 'we', 'ate', c.food];
      if (t === 8) return ['she', 'learned', c.lang, 'and', 'cooked', c.food];
      return ['the', c.lang, 'language', 'has', 'many', 'words'];
    }
    const a = pick(r, W2V.adjectives), t = r.int(5);     // ---- adjectives
    if (t === 0) return ['the', a.animal, 'is', a.base];
    if (t === 1) return ['a', 'very', a.base, a.animal];
    if (t === 2) return ['the', a.animal, 'is', a.cmp, 'than', 'the', 'dog'];
    if (t === 3) return ['this', a.animal, 'is', a.cmp, 'than', 'that', 'one'];
    return ['the', a.animal, 'is', 'the', a.sup, 'of', 'all'];
  }

  // Build a corpus: sentences of word ids, and a vocabulary sorted by count (most frequent first, as word2vec.c's
  // SortVocab does), so that index 0 is the most frequent word.
  W2V.makeCorpus = function (opts) {
    opts = opts || {};
    const r = ML.rng(opts.seed || 2013), n = opts.sentences || 2400;
    const raw = [];
    for (let i = 0; i < n; i++) raw.push(sentence(r));
    const counts = {};
    raw.forEach(function (s) { s.forEach(function (w) { counts[w] = (counts[w] || 0) + 1; }); });
    const words = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a] || (a < b ? -1 : 1); });
    const index = {}; words.forEach(function (w, i) { index[w] = i; });
    const sentences = raw.map(function (s) { return s.map(function (w) { return index[w]; }); });
    let T = 0; sentences.forEach(function (s) { T += s.length; });
    return { words: words, counts: words.map(function (w) { return counts[w]; }), index: index, sentences: sentences, raw: raw, T: T, V: words.length };
  };

  // ------------------------------------------------------------------ Huffman tree (port of CreateBinaryTree)
  // counts must be sorted from most to least frequent. Returns, for every word, its binary code (0/1 per level,
  // root first) and its point list (the inner nodes on its path, root first, numbered 0 … V−2).
  // Also returns the merge history so widgets can replay the construction.
  W2V.huffman = function (counts) {
    const V = counts.length;
    const count = new Float64Array(2 * V + 1), binary = new Uint8Array(2 * V + 1), parent = new Int32Array(2 * V + 1);
    for (let a = 0; a < V; a++) count[a] = counts[a];
    for (let a = V; a < 2 * V; a++) count[a] = 1e15;
    let pos1 = V - 1, pos2 = V;
    const merges = [];
    for (let a = 0; a < V - 1; a++) {
      let min1, min2;                                     // the two smallest nodes
      if (pos1 >= 0 && count[pos1] < count[pos2]) { min1 = pos1--; } else { min1 = pos2++; }
      if (pos1 >= 0 && count[pos1] < count[pos2]) { min2 = pos1--; } else { min2 = pos2++; }
      count[V + a] = count[min1] + count[min2];
      parent[min1] = V + a; parent[min2] = V + a;
      binary[min2] = 1;
      merges.push({ node: V + a, left: min1, right: min2, count: count[V + a] });
    }
    const codes = [], points = [];
    for (let a = 0; a < V; a++) {
      const code = [], point = [];
      let b = a;
      if (V === 1) { codes.push([]); points.push([]); continue; }
      while (true) { code.push(binary[b]); point.push(b); b = parent[b]; if (b === 2 * V - 2) break; }
      const L = code.length, c2 = new Array(L), p2 = new Array(L);
      // root first; inner nodes renumbered 0 … V−2 (node V+k → k), exactly as vocab[a].point in the C code
      p2[0] = V - 2;
      for (let k = 0; k < L; k++) { c2[L - k - 1] = code[k]; if (L - k < L) p2[L - k] = point[k] - V; }
      codes.push(c2); points.push(p2.slice(0, L));
    }
    return { codes: codes, points: points, merges: merges, count: Array.from(count.subarray(0, 2 * V - 1)), parent: Array.from(parent.subarray(0, 2 * V - 1)), binary: Array.from(binary.subarray(0, 2 * V - 1)), V: V };
  };
  // Fast variant for big vocabularies (up to millions of words): same two-queue merge, returns only the
  // frequency-weighted average code length, without storing codes.
  W2V.huffmanAvgLength = function (counts) {
    const V = counts.length;
    if (V < 2) return 0;
    const count = new Float64Array(2 * V), parent = new Int32Array(2 * V);
    for (let a = 0; a < V; a++) count[a] = counts[a];
    for (let a = V; a < 2 * V; a++) count[a] = 1e300;
    let pos1 = V - 1, pos2 = V;
    for (let a = 0; a < V - 1; a++) {
      let m1, m2;
      if (pos1 >= 0 && count[pos1] < count[pos2]) m1 = pos1--; else m1 = pos2++;
      if (pos1 >= 0 && count[pos1] < count[pos2]) m2 = pos1--; else m2 = pos2++;
      count[V + a] = count[m1] + count[m2]; parent[m1] = V + a; parent[m2] = V + a;
    }
    const depth = new Int32Array(2 * V - 1), root = 2 * V - 2;
    for (let n = root - 1; n >= 0; n--) depth[n] = depth[parent[n]] + 1;   // parents always have larger indices
    let s = 0, tot = 0;
    for (let a = 0; a < V; a++) { s += counts[a] * depth[a]; tot += counts[a]; }
    return s / tot;
  };
  // Average code length weighted by word frequency (= expected number of sigmoid outputs per prediction)
  W2V.avgCodeLength = function (counts, codes) {
    let s = 0, n = 0; counts.forEach(function (c, i) { s += c * codes[i].length; n += c; }); return s / n;
  };

  // ------------------------------------------------------------------ the model
  const MAX_EXP = 6;
  W2V.MAX_EXP = MAX_EXP;
  // opts: { corpus, D, cbow (bool), window, hs (bool), negative (int), alpha, seed, sample, sgCode (bool),
  //         epochs }   — defaults follow the paper's setup at toy scale.
  W2V.Model = function (opts) {
    const C = opts.corpus;
    this.corpus = C; this.V = C.V; this.D = opts.D || 20;
    this.cbow = !!opts.cbow; this.window = opts.window || 3;
    this.hs = opts.hs !== false; this.negative = opts.negative || 0;
    this.startAlpha = opts.alpha || (this.cbow ? 0.05 : 0.025);
    this.alpha = this.startAlpha; this.sample = opts.sample || 0; this.sgCode = !!opts.sgCode;
    this.epochs = opts.epochs || 5;
    this.rng = ML.rng(opts.seed || 1);
    const V = this.V, D = this.D;
    this.syn0 = new Float64Array(V * D);          // input vectors: the word vectors we keep
    this.syn1 = new Float64Array(V * D);          // hierarchical-softmax node vectors (V−1 used), start at 0
    this.syn1neg = new Float64Array(V * D);       // negative-sampling output vectors, start at 0
    for (let i = 0; i < V * D; i++) this.syn0[i] = (this.rng.next() - 0.5) / D;   // as InitNet: (u − 0.5) / size
    this.tree = W2V.huffman(C.counts);
    if (this.negative > 0) this.table = W2V.unigramTable(C.counts, 0.75, 20000);
    this.wordsDone = 0; this.epoch = 0; this.sentIdx = 0;
    this.totalWords = this.epochs * C.T;
    this.lossSum = 0; this.lossN = 0; this.history = [];
    this.neu1 = new Float64Array(D); this.neu1e = new Float64Array(D);
  };
  const sig = function (x) { return 1 / (1 + Math.exp(-x)); };

  // One hierarchical-softmax update: input vector `inp` (offset into `inArr`), target word `word`.
  // Accumulates the error for the input into neu1e and updates the node vectors, as word2vec.c lines 449–463.
  W2V.Model.prototype._hs = function (inArr, inOff, word, alpha, rec) {
    const D = this.D, code = this.tree.codes[word], point = this.tree.points[word], syn1 = this.syn1, e = this.neu1e;
    for (let d = 0; d < code.length; d++) {
      const l2 = point[d] * D;
      let f = 0; for (let c = 0; c < D; c++) f += inArr[inOff + c] * syn1[l2 + c];
      if (f <= -MAX_EXP || f >= MAX_EXP) { this.lossSum += code[d] ? (f > 0 ? f : 0) : (f < 0 ? -f : 0); continue; }  // the code skips saturated nodes
      f = sig(f);
      const label = 1 - code[d];
      this.lossSum += -Math.log(label ? f : 1 - f);
      const g = (label - f) * alpha;                // word2vec.c: g = (1 − code − f) · alpha
      if (rec) rec.push({ node: point[d], code: code[d], p: f, g: g });
      for (let c = 0; c < D; c++) e[c] += g * syn1[l2 + c];
      for (let c = 0; c < D; c++) syn1[l2 + c] += g * inArr[inOff + c];
    }
    this.lossN++;
  };
  // One negative-sampling update (follow-up paper; word2vec.c lines 465–484).
  W2V.Model.prototype._ns = function (inArr, inOff, word, alpha) {
    const D = this.D, s1 = this.syn1neg, e = this.neu1e;
    for (let d = 0; d < this.negative + 1; d++) {
      let target, label;
      if (d === 0) { target = word; label = 1; }
      else { target = this.table[this.rng.int(this.table.length)]; if (target === word) continue; label = 0; }
      const l2 = target * D;
      let f = 0; for (let c = 0; c < D; c++) f += inArr[inOff + c] * s1[l2 + c];
      let g;
      if (f > MAX_EXP) g = (label - 1) * alpha; else if (f < -MAX_EXP) g = label * alpha;
      else { const p = sig(f); g = (label - p) * alpha; this.lossSum += -Math.log(label ? p : 1 - p); }
      for (let c = 0; c < D; c++) e[c] += g * s1[l2 + c];
      for (let c = 0; c < D; c++) s1[l2 + c] += g * inArr[inOff + c];
    }
    if (!this.hs) this.lossN++;
  };

  // Subsampling (follow-up paper; word2vec.c lines 408–412): keep probability of a word with count cn.
  W2V.keepProb = function (cn, sample, T) { if (!sample) return 1; const st = sample * T; return Math.min(1, (Math.sqrt(cn / st) + 1) * st / cn); };

  // Train on one sentence (array of word ids). Returns the number of words consumed.
  W2V.Model.prototype.trainSentence = function (sen0) {
    const D = this.D, C = this.corpus, syn0 = this.syn0, neu1 = this.neu1, e = this.neu1e;
    let sen = sen0;
    if (this.sample) { const self = this; sen = sen0.filter(function (w) { return self.rng.next() < W2V.keepProb(C.counts[w], self.sample, C.T); }); }
    // linear decay over all training words, with the code's floor of startAlpha × 0.0001
    this.alpha = Math.max(this.startAlpha * (1 - this.wordsDone / (this.totalWords + 1)), this.startAlpha * 0.0001);
    const alpha = this.alpha, win = this.window;
    for (let pos = 0; pos < sen.length; pos++) {
      const word = sen[pos];
      const b = this.rng.int(win);                // reach R = window − b, uniform in 1 … window
      if (this.cbow) {
        neu1.fill(0); e.fill(0);
        let cw = 0;
        for (let a = b; a < win * 2 + 1 - b; a++) if (a !== win) {
          const c = pos - win + a; if (c < 0 || c >= sen.length) continue;
          const l = sen[c] * D; for (let k = 0; k < D; k++) neu1[k] += syn0[l + k]; cw++;
        }
        if (!cw) continue;
        for (let k = 0; k < D; k++) neu1[k] /= cw;
        if (this.hs) this._hs(neu1, 0, word, alpha);
        if (this.negative > 0) this._ns(neu1, 0, word, alpha);
        for (let a = b; a < win * 2 + 1 - b; a++) if (a !== win) {
          const c = pos - win + a; if (c < 0 || c >= sen.length) continue;
          const l = sen[c] * D; for (let k = 0; k < D; k++) syn0[l + k] += e[k];
        }
      } else {
        for (let a = b; a < win * 2 + 1 - b; a++) if (a !== win) {
          const c = pos - win + a; if (c < 0 || c >= sen.length) continue;
          // paper: input = current word, output = the context word. word2vec.c swaps the roles (sgCode).
          const inW = this.sgCode ? sen[c] : word, outW = this.sgCode ? word : sen[c];
          const l1 = inW * D;
          e.fill(0);
          if (this.hs) this._hs(syn0, l1, outW, alpha);
          if (this.negative > 0) this._ns(syn0, l1, outW, alpha);
          for (let k = 0; k < D; k++) syn0[l1 + k] += e[k];
        }
      }
    }
    this.wordsDone += sen0.length;
    return sen0.length;
  };
  // Train until `budget` words have been consumed (or training ends). Returns true while training continues.
  W2V.Model.prototype.trainWords = function (budget) {
    const S = this.corpus.sentences;
    let used = 0;
    while (used < budget && this.epoch < this.epochs) {
      used += this.trainSentence(S[this.sentIdx]);
      this.sentIdx++;
      if (this.sentIdx >= S.length) {
        this.sentIdx = 0; this.epoch++;
        this.history.push(this.lossN ? this.lossSum / this.lossN : 0);
        this.lossSum = 0; this.lossN = 0;
      }
    }
    return this.epoch < this.epochs;
  };
  W2V.Model.prototype.trainAll = function () { while (this.trainWords(1e9)); return this; };
  W2V.Model.prototype.progress = function () { return Math.min(1, this.wordsDone / this.totalWords); };
  W2V.Model.prototype.currentLoss = function () { return this.lossN ? this.lossSum / this.lossN : (this.history.length ? this.history[this.history.length - 1] : NaN); };
  W2V.Model.prototype.vector = function (w) { const i = typeof w === 'number' ? w : this.corpus.index[w]; return Array.from(this.syn0.subarray(i * this.D, (i + 1) * this.D)); };
  // Unit-length copies of all word vectors (compute-accuracy.c normalises every vector when loading)
  W2V.Model.prototype.normalized = function () {
    const V = this.V, D = this.D, M = [];
    for (let i = 0; i < V; i++) M.push(ML.normalize(Array.from(this.syn0.subarray(i * D, (i + 1) * D))));
    return M;
  };
  // Probability the model gives to word `out` given input word `inp` via hierarchical softmax (paper direction).
  W2V.Model.prototype.hsProb = function (inVec, out) {
    const D = this.D, code = this.tree.codes[out], point = this.tree.points[out];
    let p = 1;
    for (let d = 0; d < code.length; d++) {
      let f = 0; const l2 = point[d] * D; for (let c = 0; c < D; c++) f += inVec[c] * this.syn1[l2 + c];
      const s = sig(f); p *= code[d] ? 1 - s : s;
    }
    return p;
  };

  // ------------------------------------------------------------------ negative-sampling table (follow-up paper)
  W2V.unigramTable = function (counts, power, size) {
    let tot = 0; counts.forEach(function (c) { tot += Math.pow(c, power); });
    const table = new Int32Array(size); let i = 0, d1 = Math.pow(counts[0], power) / tot;
    for (let a = 0; a < size; a++) {
      table[a] = i;
      if (a / size > d1) { i++; d1 += Math.pow(counts[Math.min(i, counts.length - 1)], power) / tot; }
      if (i >= counts.length) i = counts.length - 1;
    }
    return table;
  };

  // ------------------------------------------------------------------ analogies (word-analogy.c / compute-accuracy.c)
  // a : b :: c : ?   →  X = b − a + c on unit vectors; nearest by cosine, skipping a, b and c.
  W2V.analogy = function (M, words, a, b, c, k, keepInputs) {
    const D = M[0].length, x = new Array(D);
    for (let j = 0; j < D; j++) x[j] = M[b][j] - M[a][j] + M[c][j];
    const xn = ML.normalize(x), scores = [];
    for (let i = 0; i < M.length; i++) {
      if (!keepInputs && (i === a || i === b || i === c)) continue;
      scores.push({ i: i, w: words[i], cos: ML.dot(xn, M[i]) });
    }
    scores.sort(function (p, q) { return q.cos - p.cos; });
    return { x: x, top: scores.slice(0, k || 5) };
  };
  W2V.nearest = function (M, words, i, k) {
    const out = [];
    for (let j = 0; j < M.length; j++) if (j !== i) out.push({ i: j, w: words[j], cos: ML.dot(M[i], M[j]) });
    out.sort(function (p, q) { return q.cos - p.cos; });
    return out.slice(0, k || 5);
  };

  // ------------------------------------------------------------------ the toy analogy test set
  // Built like the paper's test set (§4.1): lists of word pairs per relation, then every ordered pair of pairs.
  W2V.relations = function () {
    const P = W2V.people, by = function (w) { return P.find(function (p) { return p.w === w; }); };
    return [
      { name: 'man–woman', type: 'semantic', pairs: [['king', 'queen'], ['prince', 'princess'], ['man', 'woman'], ['boy', 'girl']] },
      { name: 'common–royal', type: 'semantic', pairs: [['man', 'king'], ['woman', 'queen'], ['boy', 'prince'], ['girl', 'princess']] },
      { name: 'capital city', type: 'semantic', pairs: W2V.places.map(function (p) { return [p.country, p.city]; }) },
      { name: 'country–language', type: 'semantic', pairs: W2V.places.map(function (p) { return [p.country, p.lang]; }) },
      { name: 'comparative', type: 'syntactic', pairs: W2V.adjectives.map(function (a) { return [a.base, a.cmp]; }) },
      { name: 'superlative', type: 'syntactic', pairs: W2V.adjectives.map(function (a) { return [a.base, a.sup]; }) }
    ].map(function (r) { void by; return r; });
  };
  W2V.questions = function () {
    const qs = [];
    W2V.relations().forEach(function (r) {
      r.pairs.forEach(function (p1, i) { r.pairs.forEach(function (p2, j) { if (i !== j) qs.push({ rel: r.name, type: r.type, a: p1[0], b: p1[1], c: p2[0], d: p2[1] }); }); });
    });
    return qs;
  };
  // Exact-match accuracy, overall and per relation, as compute-accuracy.c reports it.
  W2V.evaluate = function (model) {
    const M = model.normalized(), C = model.corpus, qs = W2V.questions(), per = {};
    let right = 0, sem = [0, 0], syn = [0, 0];
    const misses = [];
    qs.forEach(function (q) {
      const ix = [q.a, q.b, q.c, q.d].map(function (w) { return C.index[w]; });
      if (ix.some(function (i) { return i == null; })) return;
      const res = W2V.analogy(M, C.words, ix[0], ix[1], ix[2], 1);
      const ok = res.top[0].i === ix[3];
      per[q.rel] = per[q.rel] || { right: 0, n: 0, type: q.type };
      per[q.rel].n++; if (ok) { per[q.rel].right++; right++; } else misses.push({ q: q, got: res.top[0].w });
      const t = q.type === 'semantic' ? sem : syn; t[1]++; if (ok) t[0]++;
    });
    return { acc: right / qs.length, n: qs.length, per: per, semantic: sem[0] / sem[1], syntactic: syn[0] / syn[1], misses: misses };
  };

  // A shared, lazily trained model for widgets that only need "a trained toy" (cached by settings).
  W2V._cache = {};
  W2V.corpusDefault = function () { return W2V._corpus || (W2V._corpus = W2V.makeCorpus({})); };
  W2V.trained = function (opts) {
    opts = Object.assign({ cbow: false, D: 20, window: 3, epochs: 6, seed: 1 }, opts || {});
    const key = JSON.stringify(opts);
    if (!W2V._cache[key]) {
      const corpus = opts.sentences ? W2V.makeCorpus({ sentences: opts.sentences, seed: opts.corpusSeed }) : W2V.corpusDefault();
      W2V._cache[key] = new W2V.Model(Object.assign({ corpus: corpus }, opts)).trainAll();
    }
    return W2V._cache[key];
  };
})();
