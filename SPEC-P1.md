# VCP Memo for DSH — P1 实现规格书(Tag 层)

> 本文件与 `SPEC.md`(P0)共同构成权威契约。P0 全部行为与测试必须保持绿色。
> P1 验收:**recall 返回 matchedTags 诊断字段**(不改变排序,排序增强是 P2)。
> 算法来源:`/home/lyy/workspace/vcp-src/VCPToolBox-main/` 的 `EPAModule.js`(741 行)与
> `ResidualPyramid.js`(394 行),CC BY-NC-SA 4.0,移植文件必须保留版权头。

## 0. P1 范围

1. Tag 层:从日记末行 `Tag:` 解析带 **position** 的 Tag 记录,Tag 名向量化,落 `tags.jsonl`;
2. EPA 基底训练(启动时 + Tag 集合变化时):K-Means + 加权 SVD 语义主轴;
3. Residual Pyramid 感应种子:逐层正交解释查询向量,输出种子 Tag 与 coverage/novelty/activation;
4. `recall` 响应扩展诊断字段(matchedTags / epa / pyramid),**排序逻辑与 P0 完全一致(纯 KNN)**。

**P1 不做**:共现图、传播核、查询融合、SVD 去重、被动注入、update 工具化。

## 1. core/EPAModule.mjs —— 移植(直接搬,只换接口)

源:`/home/lyy/workspace/vcp-src/VCPToolBox-main/EPAModule.js`(CommonJS,零 require,数学全自实现)。

移植规则:
- 原文件头部加版权与出处注释(VCPToolBox + CC BY-NC-SA 4.0 + 移植日期),转 ESM 导出;
- **数学代码逐行保留**(K-Means、Gram 矩阵、Power Iteration、Gram-Schmidt 再正交化、投影熵、跨域几何平均),不得重写;
- 只替换外部接口:原构造参数 `(db, config)` 改为:

```js
export class EPAModule {
  constructor(tagProvider, cache, config)
  // tagProvider: { listTagVectors(): Array<{ id:number, name:string, vector:Float32Array }> }
  // cache: { get(key:string): any, set(key:string, value:any): void }   // 同步 KV,持久化由调用方负责
  // config: { dimension:number, maxBasisDim?=32, clusterCount?=12, minTags?=8 }
  async initialize(): Promise<boolean>     // 原逻辑:缓存命中→加载;否则训练→写缓存。tag 数 < minTags → false
  project(vector): { projections, probabilities, entropy, logicDepth, dominantAxes }  // 未训练时返回 null
  detectCrossDomainResonance(vector): { resonance, bridges }  // 未训练时返回 null
  get trained(): boolean
  // 原 refreshInBackground 删除(单用户场景,训练由 taglayer 显式触发)
}
```

- 原代码中 vector 是 SQLite Buffer,改为 Float32Array 入参(移植时在读取处统一适配,注释说明);
- 缓存 key 沿用 `epa_basis_cache`;缓存值必须 JSON 可序列化(basis 转普通数组);
- 原配置默认 `dim=3072/maxBasisDim=64/clusterCount=64` 改为上表默认(1024 维、个人规模);
- **若原代码含 Math.random(K-Means Forgy 初始化),保持原样不改**(在测试规格中按不变量验证,见 §5)。

## 2. core/ResidualPyramid.mjs —— 移植(直接搬,只换接口)

源:`/home/lyy/workspace/vcp-src/VCPToolBox-main/ResidualPyramid.js`(CommonJS,零 require)。

移植规则同 §1(版权头、数学逐行保留、只换接口):

```js
export class ResidualPyramid {
  constructor(searchTags, config)
  // searchTags: async (residualVector:Float32Array, topK:number)
  //   => Array<{ id:number, name:string, vector:Float32Array, similarity:number }>
  //   (融合原 tagIndex.search + db 取向量两步;实现方是 taglayer 的暴力余弦)
  // config: { dimension:number, maxLevels?=3, topK?=10, minEnergyRatio?=0.1 }
  async analyze(queryVector): {
    levels: Array<{ level, tags: Array<{ id, name, similarity, contribution, ... }>, energyExplained, residualEnergyRatio }>,
    totalExplainedEnergy, finalResidual,
    features: { depth, coverage, novelty, coherence, tagMemoActivation, expansionSignal }
  }
}
```

- 原代码中 Rust 快路径(computeOrthogonalProjection/computeHandshakes)删除,保留 JS 实现;
- tag id 的 BigInt 转换逻辑删除(我们的 id 就是 number)。

## 3. engine/taglayer.mjs —— 自写

```js
export function createTagLayer(deps)
// deps: { dataRoot, dimension, sig, embedder, epaConfig, pyramidConfig, log }
// 返回 {
//   load(): Promise<void>,            // 读 tags.jsonl + epa.json;校验 sig,不符则抛 sig 错误
//   parseTags(text): Array<{ name, position }>,
//   updateFile(rel, text): void,      // 更新 occurrences;新 tag 标记待 embed
//   removeFile(rel): void,
//   flush(): Promise<void>,           // embed 新 tag → 重写 tags.jsonl → tagSet 变化则重训 EPA
//   searchTags(vector, topK): Promise<Array<{id,name,vector,similarity}>>,
//   epa: EPAModule,                   // trained 状态可查
//   pyramid: ResidualPyramid,
//   stats(): { tagCount, vectorizedTags, epaTrained },
// }
```

### 3.1 Tag 解析(对齐 VCP 末行 Tag 契约)

- 取正文中**最后一个**匹配 `/^\s*Tag\s*[:：]\s*(.+)$/m` 的行;
- 按 `/[,，、]/` 切分,逐项清洗(去首尾空白、折叠连续空白),丢弃空项;
- **保序、不排序、不去重**;`position` 从 **1** 开始(VCP 语义:position>0 为有序记录)。

### 3.2 tags.jsonl

每行:`{ "name": "...", "vector": [...] | null, "occurrences": [{ "file": "...", "position": 1 }] }`。
embed 失败的 tag `vector: null`,下次 flush 重试;无向量 tag 不参与 searchTags/EPA。

### 3.3 EPA 训练触发与缓存

- `tagHash` = 对"有向量的 tag 名排序列表 + sig"取 md5;
- `flush()` 后若 `tagHash` 变化且有向量 tag 数 ≥ minTags → 调 `epa.initialize()` 重训,
  结果写 `index/epa.json`(含 sig、tagHash、basis);启动 `load()` 时读回,hash 一致直接用;
- EPA 训练在后台异步进行(不阻塞 recall);未训练完成时诊断字段优雅降级(§4)。

## 4. recall 响应扩展(P1 唯一的行为变化)

`store.recall()` 返回结构在 P0 基础上扩展(**blocks 与排序完全不变**):

```js
{
  blocks: [...],                       // 与 P0 一致
  matchedTags: [                        // pyramid 种子,按 weight 降序,最多 8 个
    { tag, weight, level, notation }    // notation: `=${tag}:${weight.toFixed(2)}@seed`(VCP 记号)
  ],
  stats: {
    candidates, indexedChunks, ms,      // 与 P0 一致
    epa: { trained: true, logicDepth, entropy, resonance, dominantAxes: [{label, score}] }
       | { trained: false },
    pyramid: { depth, coverage, novelty, coherence, activation } | null,
  }
}
```

- pyramid 种子来源:各 level 的 tags,`weight = contribution`;跨层合并同名 tag 取最大 weight;
- 无 tag / pyramid 不可用时 `matchedTags: []`、`stats.pyramid: null`;
- recall 路径的新增计算(pyramid 逐层 searchTags)失败时:记 log,按无种子降级返回,**不得让 recall 整体失败**。

## 5. 移植保真验收(关键)

为证明"搬数学"没有搬错,必须做**原版 vs 移植版对照测试**(`tests/port-equivalence.test.mjs`):

- 用 `node:module` 的 `createRequire` 加载原版 CJS 文件;
- 原版 EPAModule 喂假 db(`prepare(sql)` 返回 `{ all: () => 假tag行(Buffer向量), get: () => 缓存行, run: () => {} }`);
- 原版 ResidualPyramid 喂假 tagIndex(`search` 返回 `{id, similarity}`)+ 假 db(按 id 取向量);
- 同一组合成数据集(≥16 个有聚类结构的 1024 维向量 + 一个查询向量):
  - EPA:两版 `initialize` 均成功;`project` 的 entropy/logicDepth 差异 < 1e-3;dominantAxes 数量一致
    (若 K-Means 有随机初始化导致主轴不可比,改为验证不变量:probabilities 和为 1、熵在 [0,1]、
    同版两次 project 结果一致、缓存 roundtrip 后 project 结果一致);
  - Pyramid:两版 `analyze` 的 level 数、各层 top1 tag、features.coverage 差异 < 1e-3;
- 对照测试同样是 `node tests/port-equivalence.test.mjs` 直接可跑。

## 6. store/入口集成

`engine/store.mjs` 改动(保持 P0 行为):
- 启动:`createTagLayer`,`load()` 随 ready 进行;sig 不一致走原有拒绝服务;
- `indexFile` 提交 chunk 后:`tagLayer.updateFile(rel, text)`;队列空闲时 `tagLayer.flush()`;
- `removeChunks`:`tagLayer.removeFile(rel)`;
- `doRebuild`:重建时同步重建 tagLayer(逐文件 updateFile,最后 flush);
- `recall`:按 §4 组装诊断字段;`stats()`:并入 `tagLayer.stats()`。

入口 `vcp-memo.mjs`:无需结构性改动(诊断字段随 JSON 透出);
config 新增可选节(默认值如下,缺省即用):

```yaml
tagmemo:
  epa: { minTags: 8, clusterCount: 12, maxBasisDim: 32 }
  pyramid: { maxLevels: 3, topK: 10, minEnergyRatio: 0.1 }
```

`NOTICE.md`:补充 EPAModule.mjs、ResidualPyramid.mjs 两个移植文件的署名。

## 7. P1 验收标准

1. `node --check` 全部通过;**P0 全部测试(chunker/embed/store/entry/e2e)保持绿色**;
2. `tests/port-equivalence.test.mjs` 通过(§5);
3. `tests/taglayer.test.mjs`:Tag 解析(全角变体、保序、position 1-based、取最后一行);
   tags.jsonl roundtrip;新 tag 经真 Ollama 向量化;
4. `tests/e2e-p1.test.mjs`(真 Ollama + 临时 dataRoot):
   - 预置 ≥10 篇带 Tag 的日记(构造 8+ 个不同 Tag,其中若干共享 Tag);
   - recall 返回结构含 matchedTags/epa/pyramid 字段;epa.trained === true;
   - 查询与某共享 Tag 语义相关时,该 Tag 出现在 matchedTags 中;
   - blocks 排序与"同查询下 P0 纯 KNN 结果"一致(显式断言,防回归);
   - 无 Tag 的库上 recall 优雅降级(matchedTags: [], pyramid: null);
   - sig 不一致仍拒绝服务;
5. 全部测试仍在 `tests/` 下、无框架、`node` 直接可跑。
