// 读取 raw/chunks.json，用 bge-small-zh 计算向量，输出 docs/data/index.json。
// 模型文件需事先放在 docs/models/（见 scripts/download-model.ps1），网页从本站加载同一个模型给问题编码。
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { pipeline, env } from "@huggingface/transformers";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const EMBED_MODEL = "Xenova/bge-small-zh-v1.5";

env.localModelPath = path.join(ROOT, "docs", "models");
env.allowRemoteModels = false;

const chunks = JSON.parse(await readFile(path.join(ROOT, "raw", "chunks.json"), "utf8"));
const extractor = await pipeline("feature-extraction", EMBED_MODEL, { dtype: "q8" });

const BATCH = 16;
const vectors = [];
for (let i = 0; i < chunks.length; i += BATCH) {
  const batch = chunks.slice(i, i + BATCH).map((c) => c.text);
  const out = await extractor(batch, { pooling: "cls", normalize: true });
  vectors.push(...out.tolist());
  process.stdout.write(`\r向量化 ${Math.min(i + BATCH, chunks.length)}/${chunks.length}`);
}
process.stdout.write("\n");

const dim = vectors[0].length;
const flat = new Float32Array(vectors.length * dim);
vectors.forEach((v, i) => flat.set(v, i * dim));

const index = {
  model: EMBED_MODEL,
  dim,
  chunks: chunks.map(({ where, text }) => ({ where, text })),
  vectors: Buffer.from(flat.buffer).toString("base64"),
};

await mkdir(path.join(ROOT, "docs", "data"), { recursive: true });
const dest = path.join(ROOT, "docs", "data", "index.json");
await writeFile(dest, JSON.stringify(index));
console.log(`${chunks.length} 个片段, 维度 ${dim} -> ${dest}`);
