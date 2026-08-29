# VCP Memo for DSH — P3 实现规格书(打磨:管理 CLI / 备份 / update 入口)

> 与 SPEC.md(P0)、SPEC-P1.md、SPEC-INJECT.md、SPEC-P2.md 共同构成契约;此前 16 套测试必须保持绿色。
> P3 原则:全部为零依赖 plain ESM;CLI 复用 engine/ 模块,不复制逻辑。

## 1. bin/vcp-memo.mjs —— 管理 CLI

脱离 DSH 运行的独立管理工具:`node bin/vcp-memo.mjs <command> [--dataRoot PATH]`。

- 复用 `engine/embed.mjs` 的 `createEmbedder` 与 `engine/store.mjs` 的 `openStore`;
- 默认配置与 cordis.patch.yml 一致(dataRoot `/home/lyy/vcp-memo-data`、bge-m3@1024);
  `--dataRoot` 覆盖;`openStore` 时 `watch:false`,用完必须 `close()`;
- 命令:
  - `stats`:打印 store.stats() 全字段(sig/dimension/diaries/indexedChunks/pendingFiles/lastRebuild + tagCount/vectorizedTags/epaTrained);
  - `rebuild`:全量重建并打印结果(files/chunks 数);**打印警告**:若 DSH 正在运行,其内存索引不会自动刷新,建议重启 DSH 或改用 memory_admin 工具;
  - `tags`:读 index/tags.jsonl,按出现次数降序列出 `tag 名 + 文件数 + 有无向量`;
  - `doctor`:体检并逐项打印 ✅/⚠️——meta sig 与当前 embedder sig 一致性;孤儿 chunk(索引指向不存在的文件);未入索引的日记文件;无向量 Tag;epa.json 的 tagHash 与当前 tag 集一致性;
- 退出码:正常 0;doctor 发现问题 1;命令非法 2(打印用法);
- 输出全部中文、纯文本;不打印日记正文(隐私纪律同 embed.mjs)。

## 2. scripts/backup-vcp-memo.bat —— Windows 侧备份脚本

放在插件目录 `scripts/` 下,内容(注释用中文,带完整自说明):

```bat
@echo off
rem vcp-memo 记忆库备份(Windows 侧,WSL 通过 \\wsl$ 映射访问)
rem 用法:双击运行,或注册 schtasks 计划任务(见 README「备份」节)
set SRC=\\wsl$\Arch\home\lyy\vcp-memo-data\diaries
set DST=%~dp0..\..\backup\vcp-memo-data\diaries
rem 可用环境变量覆盖:set VCP_MEMO_BACKUP_DST=D:\backup\vcp-memo\diaries
if defined VCP_MEMO_BACKUP_DST set DST=%VCP_MEMO_BACKUP_DST%
robocopy "%SRC%" "%DST%" /MIR /R:2 /W:5 /NFL /NDL /NP /LOG+:"%~dp0backup.log"
echo 备份完成: %DATE% %TIME% >> "%~dp0backup.log"
```

- 只镜像 `diaries/`(真相源);`index/` 是派生产物不备份(README 说明恢复后删重建);
- README 补「备份与恢复」节:schtasks 注册命令示例(每小时)、git 备选方案(隐私提醒:
  远端必须私有)、**恢复演练**步骤(拷到临时目录 → CLI `--dataRoot` 指过去 → doctor + rebuild →
  与原库 stats 对比日记数)。

## 3. update_memory 工具(补齐 VCP create/update 双命令)

- `engine/store.mjs` 的 `updateDiary({target, replace})` 已存在(P0 延伸项),不改逻辑;
- `vcp-memo.mjs` 新增第四个工具 `update_memory`(`ctx.effect` 注册):
  - description: "锚点式修正已有记忆:target 是至少 15 字符的原文片段,replace 是替换内容。
    命中多篇或不命中会报错,需提供更精确的 target。"
  - parameters: `target`(string, minLength 15, 必填)、`replace`(string, 必填);
  - fault 态拒绝(与现有三工具一致);返回 `{ ok, file, replaced }` 或 `{ ok:false, error }`;
- 入口 `DISCIPLINE_TEXT` 第 2/3 条之间补一句:发现记忆过时或错误时,优先用 update_memory 修正而不是重复 save;
- `tests/entry.test.mjs` 补用例:四工具注册;update_memory 正常替换、target 过短报错、不命中报错;
  fault 态拒绝。

## 4. README.md 增补「运维」章

- 数据目录布局(diaries 真相源 / index 派生);
- CLI 用法四条命令;
- 备份(bat + schtasks + git 备选)与恢复演练;
- 换 embedding 模型的标准流程(改配置 → memory_admin rebuild 或 CLI rebuild);
- 常见问题:召回有噪声调 injection.truncate;DSH 运行时用 CLI rebuild 需重启。

## 5. 验收

1. `node --check` 全部通过;既有 16 套测试全绿;
2. `tests/cli.test.mjs`:临时 dataRoot + 真 Ollama,经 `node:child_process` 跑 CLI:
   stats 输出含关键字段且退出码 0;tags 列出播种的 tag;rebuild 后日记数一致;
   doctor 对健康库全 ✅、对人为制造的孤儿 chunk(手改 chunks.jsonl)报 ⚠️ 且退出码 1;
3. entry 测试全绿(含 update_memory 新用例);
4. `scripts/backup-vcp-memo.bat` 与 README 节内容存在且自说明完整(静态审查即可,不执行 bat)。
