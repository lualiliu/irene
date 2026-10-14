# 与若琳对话

基于小说《光与太阳，风筝与风》（刘润彬 著）的角色扮演网页。大模型通过 [WebLLM](https://github.com/mlc-ai/web-llm) 在浏览器里用 WebGPU 运行，配合书中原文做检索增强（RAG），无需服务器和 API Key，可以直接部署到 GitHub Pages。

## 工作原理

1. `scripts/extract.py`：从 PDF 提取正文，去掉页眉页码，按句子切成约 320 字的片段，带上卷/章标签。
2. `scripts/build-index.mjs`：用 `bge-small-zh-v1.5` 给每个片段算向量，生成 `docs/data/index.json`。
3. 网页里：
   - 用同一个向量模型给你的问题编码，结合中文二元词 BM25 做混合检索，找出最相关的几段原文；
   - 把人物设定（`docs/data/persona.md`）、当前时间点（`docs/data/stages.json`）和检索到的原文一起作为系统提示词；
   - 交给浏览器里运行的 Qwen 模型生成回复。

“时间点”决定若琳“知道什么”：初三时期只检索第一卷和第三卷《如梦令》（她视角的初中回忆），不会提前剧透后面的悲剧。

## 本地运行

```powershell
npm run serve
```

然后用 Chrome / Edge 打开 http://localhost:8080 。

国内网络可勾选「经本地代理走国内镜像下载」：页面会通过本机 `/mirror/` 转发到 `hf-mirror.net`（浏览器不能直连该镜像，会被 CORS 拦住）。部署到 GitHub Pages 时没有这个代理，请关掉该选项，或自行挂系统代理访问 Hugging Face。

## 重新构建索引（换书或改切片规则时）

需要 Python（`pip install pymupdf`）和 Node 20+。

```powershell
npm install
npm run model                                   # 下载向量模型到 docs/models（只需一次）
python scripts/extract.py "E:\光与太阳，风筝与风.pdf"
npm run index
node scripts/test-retrieval.mjs "你想考哪所高中"   # 检查检索效果
```

## 部署到 GitHub Pages

1. 把仓库推送到 GitHub。
2. 仓库 Settings → Pages → Build and deployment，Source 选 “Deploy from a branch”，分支选 `main`，目录选 `/docs`。
3. 稍等一两分钟，访问 `https://<用户名>.github.io/<仓库名>/`。

## 防止被搜索引擎收录

本站已做的防护：

- `docs/index.html` 带 `<meta name="robots" content="noindex, nofollow, ...">`，Google、Bing 等遵守规则的搜索引擎不会把网页编入索引。
- `docs/robots.txt` 和 `docs/_headers`（`X-Robots-Tag: noindex`）只在**网站根目录**生效：部署到 Cloudflare Pages、Netlify 或自定义域名时有效；在 `<用户名>.github.io/<仓库名>/` 这种子路径下会被搜索引擎忽略，GitHub Pages 也不支持自定义响应头。

做不到的事：

- **公开仓库的 github.com 页面挡不住收录。** 仓库里的任何文件都无法阻止 Google 收录 `github.com/<用户名>/<仓库名>` 及其中的源码，包括 `docs/data/index.json` 里的原文。要彻底不被收录，必须把仓库设为 **Private**。
- 免费账号的 GitHub Pages 要求仓库公开（私有仓库开 Pages 需要 GitHub Pro）。

推荐做法（按隐私程度从高到低）：

1. **私有仓库 + Cloudflare Pages**（免费，可连接私有仓库）：构建命令留空，输出目录填 `docs`。`robots.txt` 和 `_headers` 在根目录生效，源码也不公开。
2. **私有仓库 + GitHub Pro 开 Pages**：源码不公开，网页靠 meta 标签防收录。
3. **公开仓库 + GitHub Pages**：网页本身不被收录，但仓库源码页面仍可能被收录。如果你有 `<用户名>.github.io` 这个根站仓库，可在它的根目录放 `robots.txt` 写上 `Disallow: /<仓库名>/`。

已经被收录的页面，可以在 [Google Search Console](https://search.google.com/search-console/removals) 提交移除申请。

## 调整角色

- `docs/data/persona.md`：若琳的人物设定和说话方式（会完整放进系统提示词，控制在 1000 字左右）。
- `docs/data/stages.json`：各时间点的开场白、情境描述和允许检索的章节。
- `data/character.md`：通读全书整理出的完整人物档案，供修改上面两个文件时参考。

## 硬件要求

- 桌面版 Chrome / Edge 113+（需要 WebGPU）。
- 显存：Qwen3.5-2B 约 2.2GB，Qwen3.5-4B 约 3.8GB，Qwen3.5-9B 约 6.3GB。首次加载会下载模型并缓存在浏览器里。
