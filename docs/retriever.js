// 混合检索：向量相似度 + 中文二元词 BM25，用 RRF 融合排序。
// 浏览器和 Node 通用，问题编码函数由调用方注入。

const RRF_K = 60;
const HEROINE_BONUS = 0.003;

function tokenize(text) {
  const tokens = [];
  for (const run of text.toLowerCase().match(/[\u4e00-\u9fff]+|[a-z0-9]+/g) || []) {
    if (/^[a-z0-9]/.test(run)) {
      tokens.push(run);
    } else if (run.length === 1) {
      tokens.push(run);
    } else {
      for (let i = 0; i < run.length - 1; i++) tokens.push(run.slice(i, i + 2));
    }
  }
  return tokens;
}

class BM25 {
  constructor(docs, k1 = 1.2, b = 0.75) {
    this.k1 = k1;
    this.b = b;
    this.tfs = docs.map((d) => {
      const tf = new Map();
      for (const t of tokenize(d)) tf.set(t, (tf.get(t) || 0) + 1);
      return tf;
    });
    this.lens = docs.map((d) => tokenize(d).length);
    this.avgLen = this.lens.reduce((a, b) => a + b, 0) / docs.length;
    const df = new Map();
    for (const tf of this.tfs) for (const t of tf.keys()) df.set(t, (df.get(t) || 0) + 1);
    const n = docs.length;
    this.idf = new Map([...df].map(([t, f]) => [t, Math.log(1 + (n - f + 0.5) / (f + 0.5))]));
  }

  scores(query) {
    const terms = [...new Set(tokenize(query))].filter((t) => this.idf.has(t));
    return this.tfs.map((tf, i) => {
      let s = 0;
      for (const t of terms) {
        const f = tf.get(t);
        if (!f) continue;
        const norm = f + this.k1 * (1 - this.b + (this.b * this.lens[i]) / this.avgLen);
        s += (this.idf.get(t) * f * (this.k1 + 1)) / norm;
      }
      return s;
    });
  }
}

function decodeVectors(base64, dim) {
  const bin = typeof atob === "function" ? atob(base64) : Buffer.from(base64, "base64").toString("binary");
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const flat = new Float32Array(bytes.buffer);
  const out = [];
  for (let i = 0; i < flat.length; i += dim) out.push(flat.subarray(i, i + dim));
  return out;
}

function ranks(scores) {
  const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
  const r = new Array(scores.length);
  order.forEach(([s, i], rank) => (r[i] = s > 0 ? rank : Infinity));
  return r;
}

export class Retriever {
  /**
   * @param index docs/data/index.json 的内容
   * @param embedQuery async (text) => Float32Array | number[]，需已归一化
   */
  constructor(index, embedQuery) {
    this.chunks = index.chunks;
    this.vectors = decodeVectors(index.vectors, index.dim);
    this.bm25 = new BM25(this.chunks.map((c) => c.text));
    this.embedQuery = embedQuery;
  }

  /** @param allow 可选 (chunk) => boolean，只在允许的片段里检索 */
  async search(query, k = 4, allow = () => true) {
    const q = await this.embedQuery(query);
    const dense = this.vectors.map((v) => {
      let dot = 0;
      for (let i = 0; i < v.length; i++) dot += v[i] * q[i];
      return dot;
    });
    const denseRank = ranks(dense);
    const sparseRank = ranks(this.bm25.scores(query));

    const fused = this.chunks.map((c, i) => ({
      i,
      score:
        1 / (RRF_K + denseRank[i]) +
        1 / (RRF_K + sparseRank[i]) +
        (c.text.includes("若琳") ? HEROINE_BONUS : 0),
    }));
    return fused
      .filter(({ i }) => allow(this.chunks[i]))
      .sort((a, b) => b.score - a.score)
      .slice(0, k).map(({ i, score }) => ({ ...this.chunks[i], id: i, score }));
  }
}
