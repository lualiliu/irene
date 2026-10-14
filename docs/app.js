import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.86";
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";
import { Retriever } from "./retriever.js";

const MODELS = [
  ["Qwen3.5-2B", "4GB 显存推荐"],
  ["Qwen3.5-4B", "6GB 以上显存推荐"],
  ["Qwen3.5-9B", "8GB 以上显存，效果最好"],
  ["Qwen2.5-1.5B-Instruct", "8GB 内存的 iPhone 可试"],
  ["Qwen3-0.6B", "容易答非所问"],
  ["Qwen2.5-0.5B-Instruct", "iPhone 14 等 6GB 手机推荐"],
  ["Qwen3-1.7B", ""],
  ["Qwen3-4B", ""],
  ["Qwen3-8B", ""],
  ["Qwen2.5-3B-Instruct", ""],
  ["Qwen2.5-7B-Instruct", ""],
];

// 按上下文长度分配：检索片段数、历史对话字数上限、单次回复 token 上限。
// Qwen 分词器下中文约 0.6 token/字；人设+时期约 750 token，每段原文约 230 token。
const BUDGETS = {
  4096: { k: 4, historyChars: 2000, maxTokens: 300 },
  8192: { k: 7, historyChars: 6500, maxTokens: 400 },
  // DeepSeek 上下文 64K 以上，主要受费用约束
  api: { k: 8, historyChars: 12000, maxTokens: 400 },
};

const DEEPSEEK_URL = "https://api.deepseek.com/chat/completions";
const QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章：";
const STORAGE_KEY = "guoruolin-chat-history";
const SETTINGS_KEY = "guoruolin-chat-settings-v2";
const API_KEY_STORAGE = "guoruolin-chat-deepseek-key";

const $ = (id) => document.getElementById(id);

let engine = null;
let worker = null;
let retriever = null;
let persona = "";
let stages = [];
let stage = null;
let budget = BUDGETS[4096];
let modelId = "";
let backend = "local";
let apiKey = "";
let apiModel = "deepseek-chat";
let abortCtl = null;
let generating = false;
let history = [];

function loadHistory() {
  const saved = JSON.parse(localStorage.getItem(`${STORAGE_KEY}-${stage.id}`) || "null");
  history = saved || [{ role: "assistant", content: stage.greeting }];
}

function save() {
  localStorage.setItem(`${STORAGE_KEY}-${stage.id}`, JSON.stringify(history));
}

function allowedInStage(chunk) {
  return !stage.chapters || stage.chapters.some((c) => chunk.where.endsWith(c));
}

// ---------- 初始化设置面板 ----------

let f16 = false;
let gpuOk = false;

const isMobile =
  /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) ||
  (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);

// iPhone 上超过约 2GB 的模型生成时会被系统回收显卡；Qwen3.5 实际占用远超标称值。
// 6GB 内存的 iPhone 14 / 15 连 1.5B（约 1.6GB）的第一次回复都撑不住，自动选择只落到 0.5B。
const MOBILE_MAX_MB = 2000;
const MOBILE_SAFE_MB = 1100;

// iOS 切到后台或锁屏后会回收 WebGPU 设备
let wentHidden = false;
document.addEventListener("visibilitychange", () => {
  if (document.hidden) wentHidden = true;
});

async function detectGpu() {
  if (!navigator.gpu) {
    $("gpu-status").textContent =
      "当前浏览器不支持 WebGPU，无法本地运行（可以改用 DeepSeek API）。电脑请用 Chrome / Edge 113 以上版本；iPhone / iPad 需要 iOS 26 以上的 Safari。";
    $("gpu-status").classList.add("error");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    $("gpu-status").textContent =
      "没有找到可用的显卡（WebGPU adapter），无法本地运行（可以改用 DeepSeek API）。如果之前出现过显卡重置，浏览器会暂时禁用 WebGPU：请完全退出浏览器（包括后台进程）后重新打开。";
    $("gpu-status").classList.add("error");
    return;
  }
  gpuOk = true;
  f16 = adapter.features.has("shader-f16");
  $("gpu-status").textContent = f16
    ? "WebGPU 可用，支持 f16。"
    : "WebGPU 可用，但不支持 f16，将使用 f32 版本（占用显存更多）。";
}

function fullModelId(base) {
  return `${base}-${f16 ? "q4f16_1" : "q4f32_1"}-MLC`;
}

function fillModelSelect() {
  const known = new Map(webllm.prebuiltAppConfig.model_list.map((m) => [m.model_id, m]));
  $("model").innerHTML = "";
  for (const [base, note] of MODELS) {
    const record = known.get(fullModelId(base));
    if (!record) continue;
    if (isMobile && (base.startsWith("Qwen3.5") || record.vram_required_MB > MOBILE_MAX_MB)) continue;
    const gb = (record.vram_required_MB / 1024).toFixed(1);
    const opt = new Option(`${base}（约 ${gb}GB${note ? "，" + note : ""}）`, base);
    opt.dataset.mb = record.vram_required_MB;
    $("model").add(opt);
  }
  const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
  const options = [...$("model").options];
  if (isMobile) {
    $("ctx").value = "4096";
    $("ctx").disabled = true;
    // 旧版本会把 1.5B 自动写进设置。没手动选过模型时，不沿用那个值。
    const savedOpt = options.find((o) => o.value === saved.model);
    const keepSaved = saved.modelChosen && savedOpt;
    const fallback = options.find((o) => o.value === "Qwen2.5-0.5B-Instruct") || options[0];
    $("model").value = (keepSaved ? savedOpt : fallback)?.value || "";
  } else {
    if (options.some((o) => o.value === saved.model)) $("model").value = saved.model;
    if (saved.ctx) $("ctx").value = saved.ctx;
  }
  if (saved.mirror) $("mirror").checked = true;
}

function fillApiSettings() {
  const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
  $("backend").value = saved.backend || (gpuOk ? "local" : "deepseek");
  if (saved.apiModel) $("api-model").value = saved.apiModel;
  $("api-key").value = localStorage.getItem(API_KEY_STORAGE) || "";
  updateBackendUI();
}

function updateBackendUI() {
  const api = $("backend").value === "deepseek";
  $("local-opts").hidden = api;
  $("local-hint").hidden = api;
  $("api-opts").hidden = !api;
  $("load").textContent = api ? "开始对话" : "加载模型";
  $("load").disabled = !api && !gpuOk;
}

function selectSmallerModel() {
  const sel = $("model");
  const current = Number(sel.selectedOptions[0]?.dataset.mb);
  const smaller = [...sel.options]
    .filter((o) => Number(o.dataset.mb) < current)
    .filter((o) => !isMobile || Number(o.dataset.mb) <= MOBILE_SAFE_MB)
    .sort((a, b) => b.dataset.mb - a.dataset.mb)[0];
  if (smaller) sel.value = smaller.value;
  return smaller?.value;
}

// ---------- 加载 ----------

function setProgress(ratio, text) {
  $("progress").hidden = false;
  $("bar-fill").style.width = `${Math.round(ratio * 100)}%`;
  $("progress-text").textContent = text;
}

async function loadRetriever() {
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  // 必须是相对路径：transformers.js 4.3.1 对 http(s) 形式的本地路径会跳过存在性检查，导致分词器加载不到
  env.localModelPath = "models/";
  const [index, personaText, stageList] = await Promise.all([
    fetch("data/index.json").then((r) => r.json()),
    fetch("data/persona.md").then((r) => r.text()),
    fetch("data/stages.json").then((r) => r.json()),
  ]);
  persona = personaText.trim();
  stages = stageList;
  const extractor = await pipeline("feature-extraction", index.model, { dtype: "q8", device: "wasm" });
  const embed = async (text) =>
    (await extractor(QUERY_PREFIX + text, { pooling: "cls", normalize: true })).data;
  retriever = new Retriever(index, embed);
}

function mirrorBaseUrl() {
  // 浏览器直连 hf-mirror.net 会被 CORS 拦住（只放行 huggingface.co）。
  // 本地 scripts/serve.py 提供同域 /mirror/ 代理；GitHub Pages 等纯静态环境没有该代理。
  return new URL("mirror/", location.href).href;
}

async function mirrorProxyAvailable() {
  try {
    const r = await fetch("mirror/mlc-ai/Qwen3.5-2B-q4f16_1-MLC/resolve/main/mlc-chat-config.json", {
      method: "HEAD",
      cache: "no-store",
    });
    return r.ok;
  } catch {
    return false;
  }
}

async function loadEngine(base, ctx, mirror) {
  modelId = fullModelId(base);
  const HF = "https://huggingface.co/";
  let useMirror = mirror;
  if (useMirror && !(await mirrorProxyAvailable())) {
    setProgress(0, "当前环境没有镜像代理（需用 npm run serve 启动），改走 Hugging Face 官网…");
    useMirror = false;
    $("mirror").checked = false;
  }
  const MIRROR = mirrorBaseUrl();
  const appConfig = {
    ...webllm.prebuiltAppConfig,
    model_list: webllm.prebuiltAppConfig.model_list.map((m) => {
      if (!useMirror || !m.model.startsWith(HF)) return m;
      return { ...m, model: MIRROR + m.model.slice(HF.length) };
    }),
  };
  const workerUrl = new URL("./llm-worker.js", import.meta.url);
  if (isMobile) workerUrl.searchParams.set("prefill", "32");
  if (useMirror) {
    workerUrl.searchParams.set("mirror", "1");
    workerUrl.searchParams.set("mirrorBase", MIRROR);
  }
  worker = new Worker(workerUrl, { type: "module" });
  engine = new webllm.WebWorkerMLCEngine(worker, {
    appConfig,
    initProgressCallback: (p) => setProgress(p.progress, p.text),
  });
  // 等 worker 处理完 setAppConfig，再 reload
  await new Promise((r) => setTimeout(r, 50));
  // 已下载的分片会缓存，网络中断后重试即可接着下载
  for (let attempt = 1; ; attempt++) {
    try {
      await engine.reload(modelId, { context_window_size: ctx });
      return;
    } catch (err) {
      const msg = String(err);
      if (attempt >= 5 || !/network|fetch|Failed to fetch|Load failed|CORS/i.test(msg)) throw err;
      setProgress(0, `网络中断，正在重试（第 ${attempt} 次）… ${err.message || err}`);
      await new Promise((r) => setTimeout(r, 2000));
    }
  }
}

async function onLoad() {
  backend = $("backend").value;
  const settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
  if (backend === "deepseek") {
    apiKey = $("api-key").value.trim();
    apiModel = $("api-model").value;
    if (!apiKey) {
      $("progress").hidden = false;
      $("progress-text").textContent = "请先填写 DeepSeek API Key。";
      $("progress-text").classList.add("error");
      return;
    }
    if ($("remember-key").checked) localStorage.setItem(API_KEY_STORAGE, apiKey);
    else localStorage.removeItem(API_KEY_STORAGE);
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...settings, backend, apiModel }));
    budget = BUDGETS.api;
  } else {
    const ctx = Number($("ctx").value);
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ ...settings, backend, model: $("model").value, ctx, mirror: $("mirror").checked }),
    );
    budget = BUDGETS[ctx];
  }
  // 换模型时先释放旧模型占用的显存
  worker?.terminate();
  worker = null;
  engine = null;

  $("load").disabled = true;
  $("progress-text").classList.remove("error");
  setProgress(0, backend === "deepseek" ? "加载原著检索…" : "准备中…");
  try {
    await Promise.all([
      retriever ? null : loadRetriever(),
      backend === "deepseek" ? null : loadEngine($("model").value, Number($("ctx").value), $("mirror").checked),
    ]);
  } catch (err) {
    console.error(err);
    worker?.terminate();
    worker = null;
    $("progress-text").textContent = `加载失败：${err.message || err}`;
    $("progress-text").classList.add("error");
    $("load").disabled = false;
    return;
  }
  wentHidden = false;
  $("load").disabled = false;
  $("progress").hidden = true;
  $("setup").hidden = true;
  $("chat").hidden = false;
  $("model-name").textContent =
    backend === "deepseek" ? `DeepSeek · ${apiModel}` : `${modelId} · 上下文 ${$("ctx").value}`;
  if (!$("stage").options.length) for (const s of stages) $("stage").add(new Option(s.label, s.id));
  await switchStage(JSON.parse(localStorage.getItem(SETTINGS_KEY)).stage || stages[0].id);
  $("input").focus();
}

async function switchStage(id) {
  stage = stages.find((s) => s.id === id) || stages[0];
  $("stage").value = stage.id;
  const settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...settings, stage: stage.id }));
  loadHistory();
  await engine?.resetChat();
  renderAll();
}

// ---------- 渲染 ----------

function renderMessage(msg) {
  const el = document.createElement("div");
  el.className = `msg ${msg.role}`;
  const who = document.createElement("div");
  who.className = "who";
  who.textContent = msg.role === "assistant" ? "若琳" : "我";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = msg.content;
  el.append(who, bubble);
  if (msg.memories?.length) el.append(renderMemories(msg.memories));
  $("log").append(el);
  return bubble;
}

function renderMemories(memories) {
  const d = document.createElement("details");
  d.className = "memories";
  const s = document.createElement("summary");
  s.textContent = `参考了 ${memories.length} 段原文`;
  d.append(s);
  for (const m of memories) {
    const p = document.createElement("p");
    p.textContent = `【${m.where}】${m.text}`;
    d.append(p);
  }
  return d;
}

function renderAll() {
  $("log").innerHTML = "";
  history.forEach(renderMessage);
  scrollToBottom();
}

function scrollToBottom() {
  window.scrollTo({ top: document.body.scrollHeight });
}

// ---------- 对话 ----------

function buildSystemPrompt(memories) {
  const recalled = memories.map((m) => `【${m.where}】${m.text}`).join("\n\n");
  return `${persona}

## 此刻的你
${stage.prompt}

## 此刻浮现在你脑海里的往事
以下是与当前话题相关的往事片段，是你的记忆素材。片段里的“我”指润彬，“她”或“若琳”指你自己。用你自己的口吻自然地带出细节，不要提到“原文”“片段”“书”之类的字眼，也不要大段照搬。

${recalled}

## 回复要求
以若琳的身份回应润彬刚说的话：口语化，1~4 句，不超过 100 字；最多一处括号动作；不要重复你之前说过的话。`;
}

function recentHistory() {
  const picked = [];
  let chars = 0;
  for (let i = history.length - 1; i >= 0; i--) {
    chars += history[i].content.length;
    if (chars > budget.historyChars && picked.length) break;
    picked.unshift({ role: history[i].role, content: history[i].content });
  }
  return picked;
}

function retrievalQuery(text) {
  if (text.length >= 12) return text;
  const prevUser = [...history].reverse().find((m) => m.role === "user");
  return prevUser ? `${prevUser.content} ${text}` : text;
}

function stripThinking(text) {
  return text
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/<think>[\s\S]*$/, "")
    .replace(/^\s+/, "");
}

async function send(text) {
  const memories = await retriever.search(retrievalQuery(text), budget.k, allowedInStage);
  history.push({ role: "user", content: text });
  renderMessage(history.at(-1));

  const reply = { role: "assistant", content: "", memories: memories.map(({ where, text }) => ({ where, text })) };
  const bubble = renderMessage(reply);
  bubble.textContent = "（若琳在想……）";
  scrollToBottom();

  const recent = recentHistory();
  let system = buildSystemPrompt(memories);
  if (backend === "deepseek" && recent[0]?.role === "assistant") {
    system += `\n\n（对话开始时，你对润彬说：“${recent.shift().content}”）`;
  }
  const messages = [{ role: "system", content: system }, ...recent];

  setGenerating(true);
  try {
    let raw = "";
    const stream = backend === "deepseek" ? deepseekStream(messages) : localStream(messages);
    for await (const delta of stream) {
      raw += delta;
      reply.content = stripThinking(raw);
      bubble.textContent = reply.content || "（若琳在想……）";
      scrollToBottom();
    }
  } catch (err) {
    if (err.name !== "AbortError") return onSendError(err, text, bubble);
  } finally {
    setGenerating(false);
  }
  wentHidden = false;
  reply.content = reply.content.trim();
  if (!reply.content) {
    history.pop();
    renderAll();
    $("input").value = text;
    return;
  }
  bubble.textContent = reply.content;
  history.push(reply);
  save();
  scrollToBottom();
}

async function* localStream(messages) {
  const stream = await engine.chat.completions.create({
    messages,
    stream: true,
    temperature: 0.6,
    top_p: 0.8,
    repetition_penalty: 1.05,
    max_tokens: budget.maxTokens,
    ...(modelId.startsWith("Qwen3") ? { extra_body: { enable_thinking: false } } : {}),
  });
  for await (const chunk of stream) yield chunk.choices[0]?.delta?.content || "";
}

async function* deepseekStream(messages) {
  abortCtl = new AbortController();
  const reasoner = apiModel === "deepseek-reasoner";
  const res = await fetch(DEEPSEEK_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: apiModel,
      messages,
      stream: true,
      // reasoner 的 max_tokens 包含思考过程，且不支持 temperature
      ...(reasoner ? { max_tokens: 8000 } : { max_tokens: budget.maxTokens, temperature: 1.3 }),
    }),
    signal: abortCtl.signal,
  });
  if (!res.ok) {
    const detail = await res.json().then((j) => j.error?.message || "", () => "");
    const hints = {
      401: "API Key 无效，请点「换模型」重新填写。",
      402: "DeepSeek 账户余额不足，请先充值。",
      429: "请求太频繁，稍等一会儿再发。",
    };
    throw new Error(hints[res.status] || `DeepSeek 返回错误 ${res.status}：${detail}`);
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += value;
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (data === "[DONE]") return;
      const content = JSON.parse(data).choices?.[0]?.delta?.content;
      if (content) yield content;
    }
  }
}

function onSendError(err, text, bubble) {
  console.error(err);
  history.pop();
  if (backend === "local" && /disposed|Device was lost|ModelNotLoaded/i.test(String(err))) {
    $("input").value = text;
    onGpuLost();
    return;
  }
  const network = err instanceof TypeError && backend === "deepseek";
  bubble.textContent = network
    ? "（连不上 api.deepseek.com，请检查网络后重发。）"
    : `（出错了：${err.message || err}）`;
  bubble.classList.add("error");
  $("input").value = text;
}

// 显卡失效后回到加载面板重新加载，对话记录在 localStorage 里不会丢
function onGpuLost() {
  worker?.terminate();
  worker = null;
  engine = null;
  let msg;
  if (wentHidden) {
    msg = "页面切到后台或锁屏后，系统回收了显卡，模型失效了。点「加载模型」重新加载即可，模型文件已缓存，不用重新下载。";
  } else {
    const smaller = selectSmallerModel();
    msg = isMobile
      ? "手机内存不够，第一次回复时系统回收了显卡。iPhone 14 这类 6GB 内存的手机请用 Qwen2.5-0.5B。请先关掉其他 App 和 Safari 标签页，"
      : "显卡被重置了，模型已失效。通常是显存不够或显卡太老。请关掉其他占用显卡的网页或程序，";
    msg += smaller ? `已帮你换成更小的 ${smaller}，点「加载模型」重试。` : "然后点「加载模型」重试。";
    if (smaller) {
      const settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
      localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...settings, model: smaller, modelChosen: true }));
    }
  }
  msg += "如果仍然失败，请完全退出浏览器后重新打开。";
  $("chat").hidden = true;
  $("setup").hidden = false;
  $("progress").hidden = false;
  $("bar-fill").style.width = "0%";
  $("progress-text").textContent = msg;
  $("progress-text").classList.add("error");
  $("load").disabled = false;
  window.scrollTo({ top: 0 });
}

function setGenerating(on) {
  generating = on;
  $("send").textContent = on ? "停止" : "发送";
}

$("composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (generating) {
    if (backend === "deepseek") abortCtl?.abort();
    else engine?.interruptGenerate();
    return;
  }
  const text = $("input").value.trim();
  if (!text) return;
  $("input").value = "";
  await send(text);
});

$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});

$("reset").addEventListener("click", async () => {
  if (generating || !confirm("清空这个时间点的对话记录？")) return;
  history = [{ role: "assistant", content: stage.greeting }];
  save();
  await engine?.resetChat();
  renderAll();
});

$("change-model").addEventListener("click", () => {
  if (generating) return;
  $("chat").hidden = true;
  $("setup").hidden = false;
  $("progress").hidden = true;
  window.scrollTo({ top: 0 });
});

$("backend").addEventListener("change", updateBackendUI);

$("show-memories").addEventListener("change", (e) => {
  document.body.classList.toggle("show-memories", e.target.checked);
});

$("stage").addEventListener("change", (e) => {
  if (generating) {
    e.target.value = stage.id;
    return;
  }
  switchStage(e.target.value);
});

$("model").addEventListener("change", () => {
  const settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...settings, model: $("model").value, modelChosen: true }));
});

$("load").addEventListener("click", onLoad);

await detectGpu();
fillModelSelect();
fillApiSettings();
