# VCP Memo for DSH — P2 实现规格书(浪潮核心)

> 与 SPEC.md(P0)、SPEC-P1.md(P1)、SPEC-INJECT.md(P1.5)共同构成契约。此前全部测试必须保持绿色。
> P2 验收:**固定 A/B(纯 KNN vs 增强)——自造词私有概念测试中,增强路找回 KNN 漏掉的结构记忆**。
> 公式与常数的唯一来源:`/home/lyy/workspace/vcp-src/VCPToolBox-main/TagMemoEngine.js`(已核实的行号见下文)
> 与 `rag_params.json`(`KnowledgeBaseManager.v9 / spikeRouting / orderedCooccurrence`)。
> **文档公式仅作导航,一切以本规格抄录的代码级公式为准。**

## 0. P2 范围与总管线

```
recall(query 或 vector, k, truncate, tags?)
 1. q(向量路径或 embed)
 2. EPA project → logicDepth/entropy/resonance/dominantAxes        (P1 已有)
 3. ResidualPyramid analyze → 种子 tag(带 contribution/level)+ features   (P1 已有)
 4. 显式 tags 参数 → Core Tag 强制入选(虚拟补全,§4.4)
 5. V9.1 有界传播:种子经传播核扩散 → FIR 能量场(core/seed/emergent 来源)
 6. Tag 语义去重(0.88)+ 上下文向量 c → 查询融合 q' = normalize((1−α)q + αc)
 7. 增强向量 KNN ∪ 种子/涌现 Tag 所在文件补充捞取(只加分不罚分)
 8. ResultDeduplicator 候选去重(多身份硬去重 + 0.92 余弦近重复抑制)
 9. truncate + Top-K,组装(含 matchedTags 来源记号与 wave 诊断)
```

**无 Tag / 图不可用 / wave 任一步失败 → 整体回退 P1 行为(纯 KNN + 诊断字段),recall 永不失败。**
P2 新增失败同样只降级不抛出。注入(inject.mjs)路径自动受益,无需改动。

## 1. 常数表(从 rag_params.json 挖取,config.tagmemo 可覆盖,默认值即生产值)

```js
{
  enabled: true,
  baseTagBoost: 0.15,            // VCP 被动注入链生产默认(RAGDiaryPlugin defaultTagWeight)
  cooccurrence: {
    forwardGain: 1.0, reverseGain: 0.35, minReverseGain: 0.25, maxReverseGain: 0.6,
    distanceDecay: 0.08, reverseInversionGuard: 0.9,
    semanticGainEnabled: true, semanticGainPeak: 0.65, semanticGainSigma: 0.25,
    semanticGainLowSimFallback: 0.1,
  },
  kernel: {
    outboundMass: 0.95, associationReserveMass: 0.05, evidenceCompression: 1,
    wormholeGain: 1.35, tensionThreshold: 1,
    hubPenaltyExponent: 0.3, hubPenaltyFloor: 0.55, hubPenaltyCeiling: 1.8, hubSmoothingRatio: 0.1,
  },
  spike: {
    maxSafeHops: 4, baseMomentum: 2, firingThreshold: 0.1, baseDecay: 0.25, wormholeDecay: 0.7,
    tensionThreshold: 1, maxEmergentNodes: 50, maxNeighborsPerNode: 20,
    returnFlowFactor: 0.15, firGamma: 0.6, maxPropagationStates: 2000,
  },
  fusion: {
    dynamicBoostRange: [0.3, 2.0], activationMultiplierRange: [0.5, 1.5],
    coreBoostRange: [1.2, 1.4], coreBoostFactor: 1.33,
    tagDedupThreshold: 0.88, techTagThreshold: 0.08, normalTagThreshold: 0.015,
  },
  dedup: { semanticThreshold: 0.92 },
}
```

## 2. engine/taggraph.mjs —— 共现图 + 传播核(自写,照代码公式)

数据源:tagLayer 的 tags(每个 tag 有 vector 与 occurrences[{file, position}])。

### 2.1 有向共现事实矩阵(buildDirectedCooccurrenceMatrix 等价,TagMemoEngine.js:2858-3131)

对每个文件的**有序** Tag 序列(position 1-based,共 n 个),对每个有序对 (i,j)(i≠j):
- 序位势能:`φ(pos) = 0.9 − 0.4·(pos−1)/(n−1)`(n=1 时无对);范围 [0.5, 0.9]
- 距离衰减:`base = φ(i)·φ(j)·exp(−0.08·(|pos_i − pos_j| − 1))`
- 钟形语义增益(对 tag 向量余弦 sim,无向量走 lowSimFallback=0.1):
  `sim < 0.15 → semGain = 0.4 + sim`;否则 `semGain = 0.5 + 0.8·exp(−(sim−0.65)²/(2·0.25²))`
- **顺流**(pos_i < pos_j):`w = base · forwardGain · semGain`
- **逆流**(pos_i > pos_j):`w = base · reverseGain · semGain`,
  reverseGain 夹逼到 [0.25, 0.6];再被**反转守卫**截断:`w_逆流 ≤ w_顺流(同对) · 0.9`
- 事实矩阵 `fact: Map<srcId, Map<dstId, weight>>`,**跨文件累加**(同边多次共现求和);
- Pairwise 余弦:只对实际共现的 tag 对计算(实时,内存向量,无需持久化缓存)。

### 2.2 V9.1 传播核(_buildV9PropagationKernel 等价,:448-542)

输入 fact + residualMap(**P2 无 IR,恒 1.0,接口保留**):
1. 证据压缩:`e = log1p(w · evidenceCompression)`;
2. 虫洞判定:`isWormhole = e · residual ≥ tensionThreshold(1)`(注:强累计边 e≥1 即可触发,无需 IR);
   传导率 `g = e · (isWormhole ? wormholeGain(1.35) : 1)`;
3. 入流统计:对每个目标节点累计全图未归一化入流 s_in(在归一化之前);
   `median = 正入流中位数`,`smoothing = max(1e-9, median · 0.1)`;
4. 枢纽校正:`relative = s_in / (median + smoothing)`;`penalty = clamp(relative^(−0.3), 0.55, 1.8)`;`g̃ = g · penalty`;
5. 行归一化:`mainMass = outboundMass − (有虫洞边 ? reserveMass : 0)`;
   主传导 = `mainMass · g̃/Σg̃`;虫洞边再分得 `reserveMass · g̃_wh/Σg̃_wh`;
   每源节点总出流 ≤ outboundMass(0.95);虫洞边记入 `wormholeEdges: Set<'src:dst'>`。
输出 `{ kernel: Map<srcId, Map<dstId, P>>, wormholeEdges, diagnostics:{medianInflow, targetCount} }`。

### 2.3 图生命周期

- 内存持有 `{ fact, kernel, wormholeEdges, generation }`;
- 启动:tagLayer.load 后全量构建(有 ≥2 个有向量 tag 时);
- tagLayer.flush 后若 tagSet/occurrences 变化 → 全量重建(个人规模,不做增量),generation++;
- 重建异步进行,recall 始终读**当前已发布**的整代图(写时替换,不读半成品)。

## 3. engine/propagate.mjs —— 软非回溯 FIR 传播(_propagateSpikes 移植,:683-954)

```js
export function propagateSpikes(seedTags, kernel, wormholeEdges, spikeCfg)
// seedTags: [{ id, name?, adjustedWeight, isCore }]
// kernel/wormholeEdges: §2.2 产物;spikeCfg: §1 spike 节
// → { accumulatedEnergy: Map<id, energy>, fieldProvenance: Map<id, {sourceType, originType?, hop, seedId?}>, diagnostics }
```

逐行对齐以下语义(这是 P2 保真的核心,不得走样):
1. FIR 权重:`w_hop = γ^hop / Σ_{r=0}^{maxHops} γ^r`(γ=0.6),对 hop 0..maxHops 归一化;
2. 初始:每种子 `{nodeId:id, previousNodeId:null, energy:adjustedWeight, momentum:2, sourceType:isCore?'core':'seed', hop:0}`;
   `accumulatedEnergy[id] = adjustedWeight · w_0`;provenance 记 core/seed、hop 0;
3. 每跳:对每条活跃 spike(energy ≥ 0.1 且 momentum ≥ 0):
   出边按 P 降序截前 20;对每邻居:
   - `isWormhole = wormholeEdges.has('node:neighbor')`;
   - decay = wormhole ? 0.7 : 0.25;momentumCost = wormhole ? 0 : 1;
   - **立即回流**(`neighbor === spike.previousNodeId`):流量 ×0.15(软非回溯),压制的质量记入诊断;
   - `injected = spike.energy · P · decay · (回流 ? 0.15 : 1)`;< 0.01 丢弃;
   - nextMomentum = momentum − cost;< 0 且非虫洞 → 丢弃;
   - 状态键 `node:neighbor`,同键合并(能量累加、动量取 max、hop 取小);
4. 状态总数 > 2000 → 按能量截断(记诊断);
5. 每跳结束:节点能量聚合 ×w_{hop+1} 累入 accumulatedEnergy;provenance 记 emergent + 最早 hop + originType;
6. 无有效传播(本跳最大节点能量 ≤ 0.01)提前终止。

## 4. engine/wave.mjs —— 查询融合编排(applyTagBoost 等价,:962-1432 的核心子集)

```js
export function applyTagBoost(queryVector, baseTagBoost, coreTagNames, deps)
// deps: { epa, pyramid, tagLayer, graph, cfg }  cfg = §1 全节
// → { vector: Float32Array(融合后), info: { epa, pyramid, propagation, matchedTags, alpha, ... } | null, energyField, fieldProvenance }
//   info=null 表示无种子,调用方用原向量
```

严格按序(行号为 TagMemoEngine.js 原文参照):

1. **[1-2]** EPA project + detectCrossDomainResonance + pyramid analyze(P1 实例);
2. **[3] 动态链**(:1004-1017):
   `activationMultiplier = 0.5 + tagMemoActivation · (1.5−0.5)`;
   `dynamicBoostFactor = (logicDepth · (1 + ln(1+resonance)) / (1 + 0.5·entropy)) · activationMultiplier`;
   `effectiveTagBoost = baseTagBoost · clamp(dynamicBoostFactor, 0.3, 2.0)`;
   `coreMetric = 0.5·logicDepth + 0.5·(1−coverage)`;
   `dynamicCoreBoostFactor = 1.2 + coreMetric · 0.2`;
3. **[4] 种子收集**(:1038-1103):pyramid 各层 tag:
   `layerDecay = 0.7^level`;`adjustedWeight = contribution · layerDecay · coreBoost`,
   `coreBoost = isCore ? dynamicCoreBoostFactor · (0.95 + similarity·0.1) : 1`;
   (语言置信度门控 langPenalty **不实现**,配置位预留);
4. **[4.5] 传播**(:1105-1172):propagateSpikes(§3);种子节点取
   `max(adjustedWeight, emergentEnergy)`;涌现节点按能量降序截前 50,标记 `isPullback`;
5. **[4.6] Core 补全**(:1174-1200):显式 coreTagNames 中未入选的,从 tagLayer 按名取,
   `adjustedWeight = maxBaseWeight · dynamicCoreBoostFactor`,
   `maxBaseWeight = max(allTags.map(t => t.adjustedWeight / coreBoostFactor(1.33)))`,标 `isCore/isVirtual`;
   幽灵节点(带向量的匿名 core 对象)**不实现**;
6. **[5.5] Tag 语义去重**(:1250-1295):按 adjustedWeight 降序,两两余弦 > 0.88 →
   冗余方的 20% 权重转给代表方并丢弃(core 属性转移);
7. **[6] 上下文向量**(:1297-1320):`c = normalize(Σ v_t·adjustedWeight / ΣadjustedWeight)`;
   totalWeight = 0 → info=null 返回原向量;
8. **[6] 融合**(:1336-1348):`α = min(1, effectiveTagBoost)`;`q' = normalize((1−α)·q + α·c)`;
9. **matchedTags**(:1358-1378 的过滤 + V9.1 记号):Core 始终保留;技术 Tag(非中文且
   /^[A-Za-z0-9\-_.\s]+$/)阈值 `> maxWeight·0.08`;其余 `> maxWeight·0.015`;
   每条 `{ tag, weight, sourceType: core|seed|emergent, hop, notation }`,
   notation:种子/核心 `=name:w.xx@core|@seed`,涌现 `~name:w.xx@emergent:n`。

## 5. core/ResultDeduplicator.mjs —— 移植(零依赖,db 可为 null)

源:`/home/lyy/workspace/vcp-src/VCPToolBox-main/ResultDeduplicator.js`(417 行,零 require)。
算法实质(与"SVD"无关):多身份硬去重(chunk:id / NFKC 规范化正文 / path-chunk:路径:序号,
并查归并,来源优先级>分数>完整度选代表)+ 可选语义去重(余弦 ≥ 0.92 贪心近重复抑制,无向量候选保留)。
移植规则同 EPA/金字塔:版权头、逻辑逐行保留;我们候选的来源统一 'rag',向量直接挂在候选对象上
(不走 db 补水分支,该分支删除)。导出 `deduplicate(candidates, queryVector, options)`。

## 6. store.recall 管线升级(engine/store.mjs)

1. 启动:tagLayer.load 后构建 taggraph(§2.3);flush 后联动重建;
2. `recall({query, vector, k, truncate, tags})` 新增 `tags`(string[],可空):
   - `tagmemo.enabled && graph 可用 && 有向量化 tag` → wave 路径:
     `applyTagBoost(q, baseTagBoost, tags, deps)` → 用融合向量做 KNN;
     **补充捞取**:能量场 top 节点(core/seed/emergent)所在文件的 chunk,若不在 KNN 结果中,
     以融合向量余弦补入候选(只加分不罚分:分数就是向量余弦,不做额外加权);
     候选 → ResultDeduplicator(vector 已在手上,queryVector=融合向量)→ truncate → Top-K;
     响应 `stats.wave = { alpha, effectiveBoost, logicDepth, resonance, coverage, activation,
     seeds, emergentCount, fieldNodes, graphGeneration }`;matchedTags 用 §4.9 的完整来源版;

     **§6.2 修订(2026-08-29,审查裁决)**:补充捞取块【豁免 truncate 下限】但标记
     `viaStructure: true`;语义块(≥truncate)严格 Top-K;结构补充块单独以
     `maxSupplement`(config `tagmemo.maxSupplement`,默认 2,0 则关闭)追加,
     总块数 ≤ k + maxSupplement。语义候选与结构补充分别经 ResultDeduplicator。
     k 契约与 truncate 语义对语义块严格成立,结构召回以有界、可识别的方式透出。
   - 否则 → P1 路径(完全不变);
3. wave 路径任何异常 → 记 log 回退 P1 路径重算(不得返回半成品);
4. `save_memory` 不变;`recall_memory` 工具 parameters 增加可选 `tags`(string[],Core Tag 强制入选);
5. 启动对账/rebuild/sig 纪律不变;graph 是派生资产,不落盘(启动重建)。

## 7. config 与文档

`cordis.patch.yml` 示例补 `tagmemo` 节(§1 全表);NOTICE.md 补 ResultDeduplicator.mjs 署名。

## 8. P2 验收标准

1. `node --check` 全部通过;P0/P1/P1.5 全部 11 套测试保持绿色;
2. `tests/taggraph.test.mjs`:φ 公式边界(pos=1 → 0.9,pos=n → 0.5)、距离衰减、顺逆流不对称
   (顺 > 逆)、反转守卫截断、钟形增益分段(0.1→0.5、0.65→~1.3、0.95→衰减)、跨文件累加;
   传播核:行出流和 ≤ 0.95、枢纽校正方向(高入流被压)、虫洞边进 Set;
3. `tests/propagate.test.mjs`:FIR 权重和为 1 且 γ 比正确;回流 ×0.15(构造 A→B→A);
   momentum 耗尽停止;hop ≤ 4;状态上限截断;provenance 的 core/seed/emergent + hop 正确;
4. `tests/wave.test.mjs`(合成向量):动态链公式手算对照(已知 logicDepth/entropy/resonance/
  activation → effectiveTagBoost);contextVec 加权正确;q' 归一化;无种子 → info=null 原向量;
5. `tests/dedup.test.mjs`:硬去重(同文不同 id)与语义去重(余弦 > 0.92 抑制、< 0.92 保留);
6. `tests/e2e-p2.test.mjs`(真 Ollama,临时库,**A/B 固定对照**):
   - 数据:日记 A 含自造词"泽塔波"(tag: 泽塔波, 澜沧计划);日记 B 讲澜沧计划的部署细节
     (tag: 澜沧计划, 部署),正文与查询语义疏远(不含泽塔波及近义词);再加若干无关日记;
   - 查询"泽塔波的原理与影响":纯 KNN 路径(k=6)命中 A 但**断言语义疏远的 B 不在前列**;
     增强路径(同库同查询同 k)断言 **B 被召回且 blocks 数/排序提升**(泽塔波→澜沧计划→B 的结构传导);
   - 只加分不罚分:无结构证据的无关日记分数不因 wave 路径变负或消失(候选并集断言);
   - stats.wave 字段齐全;显式 tags:['澜沧计划'] 时该 Core Tag 出现在 matchedTags 且 sourceType=core;
   - sig 拒绝服务、无 Tag 库回退 P1,两条老纪律仍在;
7. 汇报 matchedTags 记号的实际输出样例(人工排查召回质量用)。

## 9. 明确不做(P4 及以后)

Intrinsic Residual 概念锚(reverseAnchorBoost 相关增强留接口)、测地线重排、外部 Rerank、
时间过滤、RiverMemo、语言置信度门控(预留)、幽灵节点。
