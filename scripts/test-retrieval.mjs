// 用法: node scripts/test-retrieval.mjs "问题1" "问题2" ...
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { pipeline, env } from "@huggingface/transformers";
import { Retriever } from "../docs/retriever.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章：";

env.localModelPath = path.join(ROOT, "docs", "models");
env.allowRemoteModels = false;

const index = JSON.parse(await readFile(path.join(ROOT, "docs", "data", "index.json"), "utf8"));
const extractor = await pipeline("feature-extraction", index.model, { dtype: "q8" });
const embed = async (text) => (await extractor(QUERY_PREFIX + text, { pooling: "cls", normalize: true })).data;
const retriever = new Retriever(index, embed);

for (const q of process.argv.slice(2)) {
  console.log(`\n=== ${q}`);
  for (const r of await retriever.search(q, 4)) {
    console.log(`[${r.score.toFixed(4)}] ${r.where} | ${r.text.slice(0, 70)}…`);
  }
}
