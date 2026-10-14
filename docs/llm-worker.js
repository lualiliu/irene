import { WebWorkerMLCEngineHandler } from "https://esm.run/@mlc-ai/web-llm@0.2.86";

// WebLLM 0.2.86 忽略 ChatOptions.prefill_chunk_size，一律用模型编译时的 1024。
// 在 GTX 1650 这类显卡上一次预填充 1024 token 会超过 Windows 的 2 秒显卡超时（TDR），
// 显卡被重置后报 "Object has already been disposed"。KV 缓存按 1024 分配，改小是安全的。
const PREFILL_CHUNK = 128;

const params = new URL(import.meta.url).searchParams;
const useMirror = params.get("mirror") === "1";
const mirrorBase = params.get("mirrorBase") || new URL("../mirror/", import.meta.url).href;
const HF_HOSTS = ["https://huggingface.co/", "https://hf-mirror.com/", "https://hf-mirror.net/"];

if (useMirror) {
  const nativeFetch = self.fetch.bind(self);
  const base = mirrorBase.endsWith("/") ? mirrorBase : mirrorBase + "/";
  self.fetch = (input, init) => {
    let url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    for (const host of HF_HOSTS) {
      if (url.startsWith(host)) {
        url = base + url.slice(host.length);
        input = typeof input === "string" || input instanceof URL ? url : new Request(url, input);
        break;
      }
    }
    return nativeFetch(input, init);
  };
}

const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg) => {
  for (const pipeline of handler.engine.loadedModelIdToPipeline?.values() ?? []) {
    if (pipeline.prefillChunkSize > PREFILL_CHUNK) pipeline.prefillChunkSize = PREFILL_CHUNK;
  }
  handler.onmessage(msg);
};
