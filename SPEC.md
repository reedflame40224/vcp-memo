# VCP Memo for DSH — P0 实现规格书

> 本文件是 P0 实现的**唯一权威契约**。实现者（子代理）必须严格遵守本文件的模块接口、
> 文件格式与纪律条款；不要自行发明接口。算法来源：VCPToolBox (CC BY-NC-SA 4.0)。

## 0. 定位与边界

DSH（DeepSeek Harness，Cordis 插件体系）的跨会话记忆插件。P0 范围：
`save_memory` / `recall_memory` / `memory_admin` 三个工具 + 中文友好切分 +
Ollama bge-m3 embedding + 暴力余弦 KNN + JSONL 存储 + 日记目录文件监听。

**P0 不做**：Tag 向量化、EPA、残差金字塔、传播核、被动注入、update 语义（仅 create）、管理面板。
（update 的锚点替换语义作为**延伸项**，时间允许才做，见 §7。）

## 1. 硬性约束（不可违反）

1. **零 npm 依赖**：只允许 Node.js 内置模块（`node:fs`、`node:path`、`node:crypto` 等）与全局 `fetch`。
   不引入 `@dqbd/tiktoken`：token 计数用启发式估计（见 §4）。
2. 全部文件为 **plain ESM `.mjs`**，不用 TypeScript/JSX，不用打包器。
3. 插件目录：`/home/lyy/workspace/DSH/Plugin/vcp-memo/`。数据目录独立于插件目录。
4. 真相源永远是日记 Markdown 文件；索引（`index/`）是派生产物，删掉可随时全量重建。
5. 文件写盘必须原子化：写临时文件 + `rename`。
6. 所有异步失败不得导致进程崩溃；embedding 单批失败填 `null` 占位，绝不把错误向量写进索引。

## 2. 目录结构（交付物）

```
vcp-memo/
├── SPEC.md               # 本文件
├── package.json          # name: vcp-memo, type: module, dsh.bundle.patch → ./cordis.patch.yml
├── cordis.patch.yml      # bundle patch（insert 行，name 用包名 vcp-memo）
├── vcp-memo.mjs          # 入口：export name/inject/apply
├── core/
│   └── chunker.mjs       # TextChunker 移植（零依赖改写）
├── engine/
│   ├── embed.mjs         # Ollama/OpenAI 兼容 embedding 客户端
│   └── store.mjs         # 存储 + 内存索引 + KNN + watcher + 重建
├── NOTICE.md             # 署名 lioensky/VCPToolBox + CC BY-NC-SA 4.0
└── README.md             # 用法、配置、备份说明
```

数据目录（默认 `/home/lyy/vcp-memo-data`，由 config.dataRoot 决定）：

```
vcp-memo-data/
├── diaries/<agent>/<YYYY-MM-DD>-<HH_MM_SS>[-标题].md
└── index/
    ├── chunks.jsonl      # 每行一个 chunk（含向量）
    └── meta.json         # 派生资产与一致性信息
```

## 3. 日记文件契约（严格对齐 VCP DailyNote）

- 文件名：`<YYYY-MM-DD>-<HH_MM_SS>[-<标题>].md`，标题做文件名安全清洗（去掉 `/\` 等）。
  同一秒冲突时追加 `-1`、`-2` 后缀。**一笔记一文件，同一天可多篇**。
- 文件内容（恰好四段）：

```markdown
[2026-08-29] - dsh
[14:30]
<正文，原样，不改动>

Tag: 标签甲, 标签乙, 标签丙
```

- 第 1 行 `[YYYY-MM-DD] - <agentName>`；第 2 行 `[HH:MM]`（服务器本地时间）；
  空行；末行 `Tag: ` + 逗号分隔标签。
- **Tag 纪律**：tags 数组保序写入（顺序即叙事方向，是后续 P2 的原材料）；
  只做标点清洗（全角逗号/顿号→`, `、去首尾空白、折叠连续空白），**不排序、不去重、不改写**。
  `save_memory` 的 `tags` 参数为必填且至少 1 个；正文里若模型已自带 `Tag:` 行，
  以 `tags` 参数为准（参数优先，正文自带行原样保留在正文内）。

## 4. core/chunker.mjs — 切分器

移植自 VCPToolBox `TextChunker.js` 的**算法**（句子切分 + 贪心组块 + 重叠窗口），
把 tiktoken 计数换成零依赖估计。

```js
// 导出：
export function estimateTokens(text) // 中文及全角字符 ×1.5，其余 ×0.25，向上取整（参照 VCP LightMemo._estimateTokens）
export function chunkText(text, options) // options: { maxTokens=6800, overlapTokens=680 }
  // → string[]；空/空白输入 → []
```

行为要求（与 VCP 一致）：
- 按 `/(?<=[。？！.!?\n])/` 切句，保留分隔符；
- 逐句贪心组块，加入后超 `maxTokens` 则封块；
- 单句超 `maxTokens` 时强制按 token 估计切片，优先在标点处断开；
- 块间保留 `overlapTokens` 的回溯重叠（按完整句子回溯，不足则取能放下的句子）；
- 每块 `.trim()`，丢弃空白块。

## 5. engine/embed.mjs — Embedding 客户端

```js
export function createEmbedder(config)
// config: { baseUrl, model, dimension, apiKey?, batchSize?=16, concurrency?=2, timeoutMs?=60000, retries?=2 }
// 返回 { embed, model, dimension, sig } ；sig = `${model}@${dimension}`
// embed(texts: string[]) → Promise<(Float32Array|null)[]>，长度严格等于输入，失败位 null
```

行为要求：
- POST `${baseUrl}/embeddings`，body `{ model, input: string[] }`；
  `apiKey` 存在时加 `Authorization: Bearer <apiKey>` 头；
- 响应 `data` 按 `index` 排序回填；每条 embedding 长度必须等于 `dimension`，否则该位记 null；
- 分批：每批 ≤ `batchSize` 条；批次间并发 ≤ `concurrency`（自实现 worker 池，不用第三方库）；
- 单批失败（网络错/非 2xx/超时/结构非法）：指数退避重试（1s、3s），重试耗尽后该批全部填 null，
  并 `console.error` 记录（带批次大小与错误消息，**不打印正文内容**——日记是隐私）；
- 超时用 `AbortSignal.timeout(timeoutMs)`。

## 6. engine/store.mjs — 存储与检索

```js
// store.mjs 自行 import { chunkText } from '../core/chunker.mjs'（同包相对导入）。
export async function openStore(config, embedder, log)
// config: { dataRoot, agentName, dimension, sig, watch?=true, chunker?: { maxTokens, overlapTokens } }
// log: (level, msg) => void
// 返回 store 实例（见下）
```

### 6.1 索引文件

`index/chunks.jsonl`，每行：
```json
{ "id": 1, "file": "diaries/dsh/2026-08-29-14_30_00-标题.md", "chunkIndex": 0,
  "content": "...", "vector": [0.021, ...] }
```
`file` 为相对 dataRoot 的路径。`id` 单调递增。

`index/meta.json`：
```json
{ "sig": "bge-m3@1024", "dimension": 1024, "chunkCount": 0, "updatedAt": 0 }
```

### 6.2 store API

- `ready: Promise<void>` — 启动加载（或必要的全量重建）完成。
- `saveDiary({ title, content, tags })` → `{ file, chunks: <入队即返回，不保证已索引> }`。
  落盘（原子写）后把文件推入索引队列，立即返回相对路径 `file`。
- `recall({ query, k=6, truncate=0.4 })` →
  ```js
  { blocks: [{ file, chunkIndex, score, text }],   // score 降序
    stats: { candidates, indexedChunks, ms } }
  ```
  流程：embed([query]) → 取到向量（null 则抛错"embedding 服务不可用"）→
  对内存索引做暴力余弦 → 过滤 `score < truncate` → 取 top-k。
  （`truncate` 即相似度下限阈值，与 VCP 语义一致。）
- `stats()` → `{ sig, dimension, diaries, indexedChunks, pendingFiles, lastRebuild }`。
- `rebuild()` → 全量重建：扫描 `diaries/**/*.md` → 逐文件重切分、重 embed →
  重建内存索引 → 整体重写 chunks.jsonl + meta.json。重建期间 recall 继续用旧索引（写时替换）。
- `close()` → 停 watcher、清空定时器。

### 6.3 内存索引

`{ id, file, chunkIndex, content, vector: Float32Array }` 数组 + 预归一化向量。
余弦相似度 = 归一化向量点积。朴素循环即可（个人规模数千 chunk）。

### 6.4 一致性纪律（不可违反）

1. 启动时读 `meta.json`：`sig` 或 `dimension` 与当前配置不一致 → **拒绝服务**：
   `ready` reject 或 recall/save 抛错，错误消息必须说明"嵌入模型签名已变（旧 X → 新 Y），
   索引与日记语义空间不再一致；确认切换模型请调用 memory_admin 的 rebuild 全量重建"。
   **绝不清库、绝不混用**。
2. `meta.json` 缺失但 `diaries/` 非空 → 启动时自动全量重建（记 log）。
3. 索引队列串行（单写者），chunk 向量 embed 失败的位跳过该 chunk（记 log），文件其余 chunk 正常入库。

### 6.5 文件监听（人直接编辑记忆的生命线）

- `watch: true` 时启动：对 `diaries/` 递归建 watcher（`fs.watch` 不支持 Linux 递归，
  自行遍历目录树对每个目录 `fs.watch`，新增目录时补挂）；
- 变更去抖 500ms；只处理 `.md`；
- add/change：读文件、md5 与内存中该文件旧值比对，相同则跳过；不同则
  "删除该文件全部旧 chunk → 重新切分 embed → 入索引"（与 saveDiary 同一队列）；
- unlink：删除该文件全部 chunk；
- 每次批量变更落盘后，整体重写 chunks.jsonl（去抖 2s 合并多次变更）；
- watcher 错误只 log 不抛。

## 7. 延伸项（时间允许才做）：update 语义

`updateDiary({ target, replace })`：`target` ≥15 字符；在 diaries 下按修改时间倒序找第一个
**包含** target 的文件，做子串替换（恰好 1 处，多处命中则报错让调用者提供更长的 target），
原子写回，走同一索引队列。找不到返回明确错误。**整文件覆写是禁止的。**

## 8. vcp-memo.mjs — 插件入口

```js
export const name = 'vcp-memo'
export const inject = ['tools']

export async function apply(ctx, config) {
  // 1. createEmbedder(config.embedding)；openStore(config, embedder, (l,m)=>ctx.logger[l]?.(m) ?? console[l]?.(m))
  // 2. await store.ready —— sig 不一致时：不 await 成功，三个工具仍然注册，
  //    但每次调用返回上述拒绝服务错误文本（工具可用性不因索引失效而崩掉插件）。
  // 3. ctx.effect(() => ctx.tools.register(...)) × 3：
}
```

三个工具（`parameters` 用 JSON Schema，`output: { schema: { type: 'string' }, render }`，
execute 返回 **JSON 字符串**；render 把值包成 `[{ type: 'text', text }]`)：

1. `save_memory` — description 要点："把值得长期记住的经历/结论/决定/偏好写入跨会话长期记忆。
   写入后立即可用；会在后台进入向量索引。"
   parameters: `title` (string, 必填), `content` (string, 必填), `tags` (string[] min 1, 必填，保序)。
   返回 `{ ok, file, tags }`。
2. `recall_memory` — "按语义检索跨会话长期记忆（历史日记片段）。会话开始或进入新话题时先调用。"
   parameters: `query` (必填), `k` (integer 1..20, 默认 6), `truncate` (number 0..1, 默认 0.4)。
   返回 §6.2 的结构。
3. `memory_admin` — "记忆库管理：stats 查看统计；rebuild 全量重建索引（更换 embedding 模型后必须执行）。"
   parameters: `op` (enum: ['stats','rebuild'])。

`ctx` 服务注意：只用 `ctx.tools`、`ctx.logger`、`ctx.effect`、`ctx.on`；文件与网络 IO 直接用
node 内置模块，**不**用 `ctx.fs`/`ctx.subprocess`（数据目录在工作区沙箱外，刻意为之）。

## 9. package.json / cordis.patch.yml / 文档

`package.json` 参照（无 dependencies）：
```json
{
  "name": "vcp-memo", "version": "0.1.0", "type": "module", "main": "vcp-memo.mjs",
  "license": "CC-BY-NC-SA-4.0",
  "dsh": { "bundle": { "patch": "./cordis.patch.yml" } }
}
```

`cordis.patch.yml`（bundle 层；`name` 用包名）：
```yaml
- insert:
    - id: vcp-memo
      name: vcp-memo
      config:
        dataRoot: /home/lyy/vcp-memo-data
        agentName: dsh
        embedding:
          baseUrl: http://127.0.0.1:11434/v1
          model: bge-m3
          dimension: 1024
        memory:
          kDefault: 6
          truncate: 0.4
```

`NOTICE.md`：声明 core/chunker.mjs 的算法移植自 https://github.com/lioensky/VCPToolBox
（CC BY-NC-SA 4.0），保留署名；engine/ 与入口为本项目自写。

## 10. P0 验收标准（实现者必须自测通过）

在插件目录下执行（Node ≥ 20，环境有 Ollama bge-m3 于 127.0.0.1:11434）：

1. `node --check` 全部 .mjs 通过；
2. chunker 自测：长中文文本（>2 万字符）切分，块大小不越界、重叠存在、无空块；
3. embed 自测：embed(['测试','第二条']) 返回 2×1024 Float32Array；错误端点（如 127.0.0.1:9）
   返回 null 位而不抛出；
4. store 集成自测（临时 dataRoot，`/tmp/vcp-memo-test-*`）：
   - saveDiary 3 篇（内容含"上周五和徕拉讨论了 TagMemo 的浪潮算法"等可被语义命中的文本）；
   - 等索引队列排空（轮询 stats().pendingFiles === 0）；
   - `recall({query:'浪潮算法讨论'})` 能命中对应日记的 chunk，score > 0.4；
   - 模拟人直接编辑：用 fs 改写其中一篇日记正文（换成别的主题并保存），等 3s，
     recall 原查询不再把该 chunk 排在首位、新主题查询能命中它；
   - 删掉 index/ 目录后重新 openStore → 自动全量重建，recall 仍正确；
   - 用错误 sig 的 embedder 打开同一 dataRoot → 得到拒绝服务错误且 chunks.jsonl 未被改动；
5. 全部自测脚本放 `tests/` 下（`node tests/xxx.test.mjs` 直接可跑，不用测试框架）。

## 11. 明确禁止

- 禁止引入任何 npm 依赖；禁止使用 `eval`；禁止把日记正文写进日志；
- 禁止在 recall 路径做任何写盘；禁止全局缓存检索结果（每次 recall 重新 embed query）；
- 禁止改动 `SPEC.md` 以外的设计决策；遇到规格缺口，选最小惊奇的实现并在代码注释说明。
