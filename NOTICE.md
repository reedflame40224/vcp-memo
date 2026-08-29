# NOTICE

## 署名与许可

本插件（vcp-memo）中以下文件的**算法**移植自：

- 项目：VCPToolBox
- 地址：https://github.com/lioensky/VCPToolBox
- 许可：Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International（[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)）

移植文件清单（`core/`）：

1. `chunker.mjs` — TextChunker：句子切分 + 贪心组块 + 重叠窗口（token 计数替换为零依赖启发式估计）；
2. `EPAModule.mjs` — EPA 语义分析：K-Means 聚类 + 加权 SVD 语义主轴 + 投影熵/跨域共振（仅替换 db 接口，数学逐行保留）；
3. `ResidualPyramid.mjs` — 残差金字塔：Modified Gram-Schmidt 逐层正交解释查询向量（仅替换 tagIndex/db 接口，数学逐行保留）；
4. `ResultDeduplicator.mjs` — 结果去重器：多身份硬去重（chunkId / NFKC 规范化正文 / path-chunk）+ 可选语义近重复抑制（余弦 ≥ 0.92 贪心抑制；db 补水分支删除，候选向量直接挂对象上）。

按 CC BY-NC-SA 4.0 的要求保留上述署名；本插件整体以同一许可发布。

`engine/`（embedding 客户端与存储/检索/文件监听）与插件入口 `vcp-memo.mjs` 为本项目自写实现，不属于 VCPToolBox 的移植部分。

## 第三方依赖

本插件零 npm 依赖，仅使用 Node.js 内置模块与全局 `fetch`。

## 使用限制

CC BY-NC-SA 4.0 许可不得用于商业用途；演绎作品须以相同方式共享。