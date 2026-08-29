# vcp-memo

DSH（DeepSeek Harness，Cordis 插件体系）的跨会话长期记忆插件（P0）。

- **`save_memory`**：把值得长期记住的经历/结论/决定/偏好写入跨会话长期记忆。即时可写，后台进入向量索引。
- **`recall_memory`**：按语义检索历史日记片段（Ollama bge-m3 embedding + 暴力余弦 KNN + 相似度下限过滤）。
- **`memory_admin`**：`stats` 查看统计；`rebuild` 全量重建索引（更换 embedding 模型后必须执行）。

底层遵循 VCP DailyNote 日记格式：一笔记一文件（`diaries/<agent>/<YYYY-MM-DD>-<HH_MM_SS>[-标题].md`），
Markdown 文件永远是真相源，`index/` 只是可随时全量重建的派生产物。日记目录被实时监听，
人工直接编辑的记忆文件也会自动进入索引。中文友好的启发式 token 切分移植自
[VCPToolBox](https://github.com/lioensky/VCPToolBox)（见 NOTICE.md）。

零 npm 依赖（仅 Node.js 内置模块与全局 `fetch`），plain ESM，无需打包器。

## 环境要求

- Node.js ≥ 20
- 本地 Ollama 服务，已拉取 `bge-m3` 模型（默认 `http://127.0.0.1:11434/v1`，1024 维）。
  也兼容任意 OpenAI 风格 `/v1/embeddings` 服务（配置 `embedding.apiKey` 即可）。

## 安装（三层）

1. **让 DSH 能解析本包**：把本插件目录 link 进 profile 的依赖解析。
   典型做法是 `npm link`（或在 profile 的 `node_modules` 下建目录链接），
   使 `vcp-memo` 作为可解析的 npm 包存在。
2. **把包加进 profile 的 bundles**：在目标 profile 的 `package.json` 中声明：

   ```json
   {
     "dsh": {
       "profile": {
         "bundles": ["@deepseek-ai/dsh-base", "vcp-memo"]
       }
     }
   }
   ```

   DSH 的 profile composer 会按序解析每个 bundle 包，并读取其 `dsh.bundle.patch` 指向的补丁文件。
3. **本插件的 `cordis.patch.yml` 被自动应用**：`package.json` 中
   `"dsh": { "bundle": { "patch": "./cordis.patch.yml" } }` 声明了补丁位置；补丁内容为
   `insert` 一行 `id: vcp-memo` 的插件行（含默认配置）。启动 profile 后插件即注册三个工具。

如需覆盖默认配置（如更换 embedding 模型/数据目录），在 profile 的 `cordis.patch.yml`（用户层）里
追加对 `vcp-memo` 行的 `config` 覆盖，或直接改本插件的 patch 后重启。

## 配置项

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `dataRoot` | `/home/lyy/vcp-memo-data` | 数据目录（必须独立于插件目录） |
| `agentName` | `dsh` | 日记所属 agent 目录名 |
| `watch` | `true` | 是否监听 `diaries/` 目录（人工编辑自动入索引） |
| `embedding.baseUrl` | `http://127.0.0.1:11434/v1` | Ollama/OpenAI 兼容 embedding 端点 |
| `embedding.model` | `bge-m3` | embedding 模型名 |
| `embedding.dimension` | `1024` | 向量维度（必须与模型一致） |
| `embedding.apiKey` | 无 | 需要鉴权时的 Bearer key |
| `embedding.batchSize` | `16` | 单批条数 |
| `embedding.concurrency` | `2` | 批次并发数 |
| `embedding.timeoutMs` | `60000` | 单次请求超时 |
| `embedding.retries` | `2` | 指数退避重试次数（1s、3s） |
| `memory.kDefault` | `6` | `recall_memory` 默认返回条数 |
| `memory.truncate` | `0.4` | `recall_memory` 默认相似度下限 |
| `chunker.maxTokens` | `6800` | 切分块 token 上限 |
| `chunker.overlapTokens` | `680` | 块间重叠 token 数 |

> 更换 `embedding.model`/`dimension` 后索引签名（`model@dimension`）会与旧索引不一致，
> 插件将拒绝服务并提示执行 `memory_admin rebuild`。**切勿手动混用旧索引。**

## 数据目录

```
<dataRoot>/
├── diaries/<agent>/<YYYY-MM-DD>-<HH_MM_SS>[-标题].md   # 日记真相源（一笔记一文件）
└── index/
    ├── chunks.jsonl   # 每行一个 chunk（含向量；派生产物）
    └── meta.json      # 签名/维度/计数等一致性信息（派生产物）
```

## 备份建议

- **必须备份**：`diaries/`（真相源，人工与插件共同写入）。
- `index/` 是派生产物，丢了无须备份——删除整个 `index/` 目录后下次启动会自动全量重建
  （需要 embedding 服务在线）。备份时可跳过以省空间；恢复时把 `diaries/` 放回原目录即可。
- 若 `diaries/` 与 `index/` 因异常（如中途强杀）不完全一致，执行 `memory_admin` 的 `rebuild`
  即可全量对齐。

## 运维

### 数据目录布局

<dataRoot>(默认 `/home/lyy/vcp-memo-data`,由 config.dataRoot 决定,必须独立于插件目录):

```text
<dataRoot>/
├── diaries/<agent>/<YYYY-MM-DD>-<HH_MM_SS>[-标题].md   # 真相源:一笔记一文件
└── index/                        # 派生产物:可随时全量重建,不单独备份
    ├── chunks.jsonl              # 每行一个 chunk(正文 + 向量)
    ├── meta.json                 # 签名/维度/计数等一致性信息
    ├── tags.jsonl                # Tag 层:标签、向量、出现位置(P1/P2 可用)
    └── epa.json                  # EPA/金字塔派生资产(P1)
```

- **`diaries/` 是真相源**:人工与插件共同写入,必须备份;
- **`index/` 是派生产物**:被删或损坏后,启动时自动(或手动 `rebuild`)按当前 embedder 全量重建,
  不需要备份,恢复时删掉 `index/` 即可。

### CLI 四条命令

独立于 DSH 运行的管理工具 `bin/vcp-memo.mjs`(零依赖 plain ESM,复用 `engine/` 模块):

```bash
node bin/vcp-memo.mjs stats                       # 统计
node bin/vcp-memo.mjs rebuild                     # 全量重建索引
node bin/vcp-memo.mjs tags                        # Tag 列表(按出现次数降序)
node bin/vcp-memo.mjs doctor                      # 一致性体检
```

- 默认配置与 `cordis.patch.yml` 一致(dataRoot `/home/lyy/vcp-memo-data`、embedding `bge-m3@1024`);
  输入 `--dataRoot PATH` 可指向别的库(如恢复演练用临时目录);
- `stats`:打印 `sig / dimension / diaries / indexedChunks / pendingFiles / lastRebuild`
  以及 `tagCount / vectorizedTags / epaTrained` 全字段;
- `rebuild`:全量重建并打印 `files / chunks` 数;**打印警告**——若 DSH 正在运行,
  其内存索引不会自动刷新,建议重启 DSH 或改用 `memory_admin` 工具;
- `tags`:读 `index/tags.jsonl`,按出现次数降序列出「tag 名 + 文件数 + 有无向量」;
- `doctor`:体检并逐项打印 ✅/⚠️——meta sig 与当前 embedder sig 一致性、孤儿 chunk
  (索引指向不存在的文件)、未入索引的日记文件、无向量 Tag、`epa.json` 的 tagHash 与当前 tag 集一致性;
- 退出码:正常 0;doctor 发现问题 1;命令非法 2(打印用法);
- 输出全部中文、纯文本,不打印日记正文(隐私纪律见 NOTICE)。

### 备份与恢复

**备选方案(A/B 任一即可,推荐 A)**:

**A. Windows 侧 `.bat`(推荐,双机/跨发行版场景)**:直接运行 `scripts/backup-vcp-memo.bat`,
经 `\\wsl$\Arch\...` UNC 路径把 `diaries/` 镜像到备份目录(`robocopy /MIR`)。
只镜像 `diaries/` 真相源;`index/` 不备份(恢复后删掉自动重建)。
可用环境变量 `VCP_MEMO_BACKUP_DST` 覆盖目标目录。注册计划任务(每小时):

```bat
schtasks /Create /TN "vcp-memo-backup" ^
  /TR "cmd /c \"C:\path\to\vcp-memo\scripts\backup-vcp-memo.bat\"" ^
  /SC HOURLY /F
```

移除计划任务:`schtasks /Delete /TN "vcp-memo-backup" /F`。

**B. git 版本库(WSL 侧)**:把 `diaries/` 纳入 git:

```bash
cd /home/lyy/vcp-memo-data
git init && git add diaries && git commit -m "backup: $(date)"
```

> ⚠️ **隐私提醒**:日记含私人内容,若推远端,**远端仓库必须私有**
> (私有 GitHub/GitLab/自建 git 均可);`index/` 是派生数据,不要入库。

**恢复演练(推荐定期做一次)**:

1. 把备份的 `diaries/` 拷到临时目录:`cp -r <备份>/diaries /tmp/vcp-memo-restore/diaries`;
2. 用 CLI 指向临时库体检:`node bin/vcp-memo.mjs doctor --dataRoot /tmp/vcp-memo-restore`
   (首次必报“未入索引”,属预期);
3. 重建:`node bin/vcp-memo.mjs rebuild --dataRoot /tmp/vcp-memo-restore`;
4. 对比日记数:`node bin/vcp-memo.mjs stats --dataRoot /tmp/vcp-memo-restore` 与原库
   `node bin/vcp-memo.mjs stats` 的 `diaries` 数字一致,即演练通过。

**正式恢复**:停 DSH → 把 `diaries/` 放回原 dataRoot → 删除 `index/` 整目录 → 重启 DSH
(启动时自动全量重建;embedding 服务必须在线)。

### 换 embedding 模型

标准流程(例如 `bge-m3` → `bge-large-zh-v1.5`):

1. 改配置:在 profile 的 `cordis.patch.yml` 用户层追加对 `vcp-memo` 的
   `embedding.model` / `embedding.dimension` 覆盖,重启 DSH;
2. 全量重建索引(二选一):
   - DSH 内:`memory_admin` 工具选 `rebuild`;
   - 命令行:`node bin/vcp-memo.mjs rebuild`(DSH 正在运行时不刷新内存,需重启 DSH 生效);
3. 验证:`memory_admin stats`(或 `CLI stats`)中 `sig` 变为新 `model@dimension`,且
   `diaries` / `indexedChunks` 数量不变。

> 签名(旧 `bge-m3@1024` → 新)不一致时插件会拒绝服务并提示 rebuild;
> **切勿手动混用旧索引**(见「配置项」)。

### 常见问题(FAQ)

- **召回有噪声 / 结果太泛**:调高相似度下限 `memory.truncate`(默认 0.4,如调到 0.5);
  被动注入噪声则调 `injection.truncate`(默认 0.45,可上调)或调小 `injection.k`。无需重建索引。
- **DSH 运行中用 CLI rebuild 后,插件行为没变**:CLI 只改磁盘索引,DSH 内存索引不自动刷新;
  重启 DSH,或改用 `memory_admin` 工具的 `rebuild`。
- **恢复后直接启动,`index/` 空/缺失**:符合预期,启动时自动全量重建(embedding 服务需在线);
  数据量大时首次启动稍慢属正常。
- **`diaries/` 与 `index/` 不一致(如中途强杀)**:执行 `memory_admin rebuild`(或 CLI `rebuild` + 重启 DSH)全量对齐。
- **doctor 报孤儿 chunk**:通常是异常中断后的残留,`rebuild` 后应清零;持续存在再排查是否有文件被手工删除。

## 许可证

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — 非商业使用、演绎同许可。
`core/chunker.mjs` 的算法移植自 lioensky/VCPToolBox（署名见 [NOTICE.md](./NOTICE.md)）。