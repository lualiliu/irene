# 与若琳对话

基于小说《光与太阳，风筝与风》的角色扮演网页：你是润彬，和若琳聊天。网页会检索书中原文作为她的"记忆"（RAG），回复可以由两种方式生成：

- **本地运行**：大模型通过 [WebLLM](https://github.com/mlc-ai/web-llm) 在浏览器里用 WebGPU 运行，免费，不需要服务器和 API Key，但受显存限制只能用小模型。
- **DeepSeek API**：填入自己的 [DeepSeek API Key](https://platform.deepseek.com/api_keys)，回复质量好得多，也不需要 WebGPU，任何手机都能用。

## DeepSeek API

「运行方式」选 DeepSeek API，填入 Key，点「开始对话」。

- 模型：`deepseek-chat` 适合聊天；`deepseek-reasoner` 会先思考再回答，更慢、更贵。
- Key 只保存在本浏览器的 localStorage 里（可取消「记住 Key」），由浏览器直接发给 `api.deepseek.com`，不经过任何中转服务器。
- 每次发送会带上人物设定、8 段检索到的原文和最近至多约 12000 字的对话，约 2500–10000 token，对话越长越多。
- 聊天界面下方的「换模型」可以随时切换运行方式。

## 本地运行模型

打开网页，选模型，点「加载模型」。首次会下载模型并缓存在浏览器里。

| 设备 | 浏览器 | 推荐模型 |
| --- | --- | --- |
| 电脑，4GB 显存 | Chrome / Edge 113+ | Qwen3.5-2B |
| 电脑，6GB / 8GB 以上显存 | 同上 | Qwen3.5-4B / Qwen3.5-9B |
| iPhone 14 / 15（6GB 内存） | iOS 26+ 的 Safari | Qwen2.5-0.5B（约 0.9GB） |
| iPhone 15 Pro 及更新（8GB 内存） | 同上 | 可试 Qwen2.5-1.5B（约 1.6GB） |

- 手机上只列出 2GB 以内的模型，6GB 内存的 iPhone 默认 Qwen2.5-0.5B，上下文固定 4096。1.5B 在 iPhone 14 上能加载，但第一次回复的临时显存会让系统回收显卡，页面提示模型已失效。
- 如果页面自己刷新了或提示显卡被回收：关掉其他 App 和 Safari 标签页后重试。已经在用 0.5B 仍失败的话，完全退出 Safari 再打开。
- 手机切到后台或锁屏后，系统会回收显卡，回来后需要重新点「加载模型」（模型已缓存，对话记录也不会丢）。
- 手机上不列出 Qwen3.5 系列：它们第一次回复时实际占用的内存远超标称值（0.8B 标称 1.6GB，实测约 2.8GB），会导致页面崩溃。
- 「时间点」决定若琳知道什么：初三时期只检索初中部分的章节，不会提前剧透。

## 在本机启动网页

```powershell
npm run serve        # 然后打开 http://localhost:8080
```

国内网络可勾选「经本地代理走国内镜像下载」，由 `scripts/serve.py` 转发到 hf-mirror.net。GitHub Pages 上没有这个代理。

## 重新构建索引

需要 Python（`pip install pymupdf`）和 Node 20+。

```powershell
npm install
npm run model                                   # 下载向量模型到 docs/models（只需一次）
python scripts/extract.py "./光与太阳，风筝与风.pdf"
npm run index                                   # 生成 docs/data/index.json
node scripts/test-retrieval.mjs "你想考哪所高中"   # 检查检索效果
```

## 部署

仓库 Settings → Pages，Source 选 "Deploy from a branch"，分支 `main`，目录 `/docs`。

网页带了 `noindex`，不会被搜索引擎收录；但公开仓库的源码（包括 `docs/data/index.json` 里的原文）仍可能被收录。介意的话，可以把仓库设为私有，改用 Cloudflare Pages 部署（输出目录填 `docs`）。

## 调整角色

- `docs/data/persona.md`：人物设定和说话方式（系统提示词，控制在 1000 字左右）。
- `docs/data/stages.json`：各时间点的开场白、情境和可检索的章节。
- `data/character.md`：完整人物档案，修改上面两个文件时参考。
