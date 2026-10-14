import { WebWorkerMLCEngineHandler } from "https://esm.run/@mlc-ai/web-llm@0.2.86";

// WebLLM 0.2.86 忽略 ChatOptions.prefill_chunk_size，一律用模型编译时的 1024。
// 一次预填充 1024 token：在 GTX 1650 上会超过 Windows 的 2 秒显卡超时（TDR）；
// 在 iPhone 上会在第一次回复时多占一大块临时显存，系统直接回收 WebGPU 设备。
// 查询参数 prefill 表示手机：除了改 JS 分块，还要改 wasm 元数据，KV 缓存才会按小块分配。
const params = new URL(import.meta.url).searchParams;
const requestedPrefill = Number(params.get("prefill"));
const PREFILL_CHUNK = requestedPrefill > 0 ? requestedPrefill : 128;

if (requestedPrefill > 0) {
  const nativeParse = JSON.parse;
  JSON.parse = function (text, reviver) {
    const value = nativeParse.call(this, text, reviver);
    if (
      value &&
      typeof value.prefill_chunk_size === "number" &&
      typeof value.max_batch_size === "number" &&
      Array.isArray(value.params) &&
      value.prefill_chunk_size > PREFILL_CHUNK
    ) {
      value.prefill_chunk_size = PREFILL_CHUNK;
    }
    return value;
  };
}
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
