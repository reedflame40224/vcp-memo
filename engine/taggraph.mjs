// engine/taggraph.mjs —— 共现图 + V9.1 传播核(规格 SPEC-P2 §2,自写)
//
// 职责:由 tagLayer 风格的 tag 记录构建“有向共现事实矩阵”(§2.1)→ V9.1
//      有界传播核(§2.2)→ 以 { fact, kernel, wormholeEdges, generation }
//      为单元的内存图生命周期(§2.3,写时替换发布,recall 永不读半成品)。
//
// 公式与常数的唯一来源:
//   /home/lyy/workspace/vcp-src/VCPToolBox-main/TagMemoEngine.js
//     · §2.1 ← buildDirectedCooccurrenceMatrix(:2858-3131)
//     · §2.2 ← _buildV9PropagationKernel(:448-542)
//   常数默认值以 SPEC-P2 §1 为准(config.tagmemo 可覆盖,默认即生产值)。
//   文档公式仅作导航,本文件逐条对齐上述代码级公式。
//
// 零依赖:仅 node 内置模块(node:crypto)。纯 ESM(.mjs),无任何 npm 依赖。
//
// 输入接口(本文件自定,注释清楚):
//   tags: Array<{ name:string, vector: Float32Array|number[]|null,
//                  occurrences: Array<{ file:string, position:number }> }>
//     —— tagLayer 记录风格:position 1-based,id 即输入数组下标(与 tagLayer 一致);
//        无向量的 tag(embed 失败)不参与向量相似度,共现时按 §2.1 走 lowSimFallback。
//   cfg?: { cooccurrence?: {...}, kernel?: {...} }  缺省用 §1 生产值,语义同
//     TagMemoEngine 的 rag_params.json(orderedCooccurrence / v9)。
//
// 已知决策(与汇报一致,偏离参考实现处均在此声明):
//   - §2.3“有 ≥2 个有向量 tag 时”才构建:门槛落在 buildGraph 入口,不足时返回
//     同构空图+skippedReason,store 据此判“图不可用 → 回退 P1”;门槛满足后,
//     无向量 tag 仍参与共现(§2.1 规定无向量走 lowSimFallback,与参考 getSimSafe 一致);
//   - 同名 tag 在同一文件多重出现(位置不同)按 §2.1“每对 (i,j)(i≠j)”只对不同
//     tag 实体建边,不产生自环(参考实现对重复 tag_id 会给自身建边,属上游脏数据
//     场景,此处防御性剔除,并已注释);
//   - 参考实现的 n>100 单文件性能保护未照搬:SPEC 未要求,个人规模不可达,照搬
//     反而会在极端场景静默丢文件(§2.1 无此条款);
//   - 参考实现的 position=0 legacy 无向回退路径未移植:P1 tagLayer 契约
//     position 恒 ≥1,该路径在我们的数据源中不可达;
//   - kernelDiagnostics 在 SPEC §2.2 的 { medianInflow, targetCount } 之外追加
//     smoothing / positiveInflowCount 与 §2.1 可观测计数(追加不改契约,供验收排障);
//   - buildV9Kernel 的 residualMap 参数按“P2 无 IR,恒 1.0,接口保留”实现为
//     可选第二参,缺省即恒 1;传入时逐目标节点取值(接口可被 P4 概念锚复用)。

import { createHash } from 'node:crypto';

// ─────────────────────────────────────────────────────────────────────────────
// §1 常数默认值(生产值;调用方可传 cfg 覆盖,覆盖语义=数值替换)
// ─────────────────────────────────────────────────────────────────────────────
const DEFAULTS = Object.freeze({
  cooccurrence: Object.freeze({
    forwardGain: 1.0,                 // 顺流增益
    reverseGain: 0.35,                // 逆流增益(会夹逼到 [minReverseGain, maxReverseGain])
    minReverseGain: 0.25,
    maxReverseGain: 0.6,
    distanceDecay: 0.08,              // 序位距离衰减
    reverseInversionGuard: 0.9,       // 反转守卫:逆流 ≤ 顺流 × 0.9
    semanticGainEnabled: true,
    semanticGainPeak: 0.65,           // 钟形增益峰值位置
    semanticGainSigma: 0.25,
    semanticGainLowSimFallback: 0.1,  // 无向量/无效 sim 的兜底相似度
  }),
  kernel: Object.freeze({
    outboundMass: 0.95,               // 每源节点总出流预算
    associationReserveMass: 0.05,     // 虫洞关联保留分量(在预算内竞争,不产生额外能量)
    evidenceCompression: 1,           // 证据压缩系数:e = log1p(w · evidenceCompression)
    wormholeGain: 1.35,               // 虫洞传导率增益
    tensionThreshold: 1,              // 虫洞判定阈值(e · residual ≥ 阈)
    hubPenaltyExponent: 0.3,          // 枢纽幂惩罚指数
    hubPenaltyFloor: 0.55,
    hubPenaltyCeiling: 1.8,
    hubSmoothingRatio: 0.1,           // 中位入流平滑比例
  }),
});

// ─────────────────────────────────────────────────────────────────────────────
// 工具:向量归一化(输入可为 Float32Array / number[] / null)
// ─────────────────────────────────────────────────────────────────────────────
function toF32(v) {
  if (v instanceof Float32Array) return v;
  if (Array.isArray(v)) return Float32Array.from(v);
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// §2.1 公式原子件(纯函数导出,供测试逐条对照)
// ─────────────────────────────────────────────────────────────────────────────

// 序位势能:φ(pos) = 0.9 − 0.4·(pos−1)/(n−1),范围 [0.5, 0.9](参考实现
// PHI_MAX=0.9 / PHI_MIN=0.5);n≤1 时无有序对,参考实现返回 PHI_MAX(0.9)。
export function phiPotential(pos, n) {
  if (!Number.isFinite(n) || n <= 1) return 0.9;
  return 0.9 - 0.4 * ((pos - 1) / (n - 1));
}

// 距离衰减:exp(−decay·(delta−1));decay 非正 → 1.0(参考实现语义);
// delta 内部夹逼到 ≥1(参考实现 delta = Math.max(1, pos差)),相邻 tag 衰减为 1。
export function positionDistanceFactor(delta, decay) {
  if (!(decay > 0)) return 1.0;
  const d = Math.max(1, delta);
  return Math.exp(-decay * (d - 1));
}

// 钟形语义增益(§2.1):sim < 0.15 → 0.4 + sim(软底);
// 否则 0.5 + 0.8·exp(−(sim−peak)²/(2σ²))(中段放大、高 sim 抑制)。
// semCfg?: { enabled=true, peak=0.65, sigma=0.25, lowSimFallback=0.1 }
export function semanticGain(sim, semCfg = {}) {
  const enabled = semCfg.enabled ?? true;
  if (!enabled) return 1.0;            // 参考实现:禁用时恒 1,不做增益
  if (!Number.isFinite(sim)) return 1.0;
  const peak = semCfg.peak ?? 0.65;
  const sigma = semCfg.sigma ?? 0.25;
  if (sim < 0.15) return 0.4 + sim;
  return 0.5 + 0.8 * Math.exp(-((sim - peak) ** 2) / (2 * sigma * sigma));
}

// Pairwise 余弦:只对实际共现的 tag 对实时计算(内存向量,无需持久化缓存)。
// 无向量 / 维度不符 / 零向量(无方向)→ null;由 getSimSafe 落入 fallback。
export function pairwiseCosine(a, b) {
  const va = toF32(a);
  const vb = toF32(b);
  if (!va || !vb || va.length !== vb.length || va.length === 0) return null;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < va.length; i++) {
    dot += va[i] * vb[i];
    na += va[i] * va[i];
    nb += vb[i] * vb[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb) || 0;
  if (denom === 0) return null;
  return dot / denom;
}

// getSimSafe(参考实现原义):语义增益启用时才计算真实余弦;
// 非有限或 ≤ 0 的相似度一律走 lowSimFallback(负相似度对叙事结构无意义)。
function makeGetSim(vectors, sem) {
  return (idA, idB) => {
    if (!sem.enabled) return sem.fallback;
    const v = pairwiseCosine(vectors[idA], vectors[idB]);
    return Number.isFinite(v) && v > 0 ? v : sem.fallback;
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// §2.1 有向共现事实矩阵(buildDirectedCooccurrenceMatrix 等价,:2858-3131)
// ─────────────────────────────────────────────────────────────────────────────
//
// 对每个文件的【有序】Tag 序列(position 1-based,共 n 个),对每个有序对
// (i,j)(i≠j,即 i<j 的无序对的两个方向):
//   · φ 序位势能(见 phiPotential)
//   · base = φ(i)·φ(j)·exp(−0.08·(|pos_i−pos_j| − 1))
//   · 钟形增益 semGain(对 tag 向量余弦,无向量 → lowSimFallback)
//   · 顺流(pos_i<pos_j):w = base · forwardGain · semGain
//   · 逆流:    w = base · reverseGain(夹逼[0.25,0.6]) · semGain,
//     再被反转守卫截断:逆流 ≤ 顺流(同对)· reverseInversionGuard
//   · fact: Map<srcId, Map<dstId, weight>>,跨文件累加(同边多次共现求和)
//   · 注意:参考实现无 reverseAnchorBoost 分支(β)= 关闭,我们亦不实现(P2 无 IR)
export function buildFactMatrix(tags, cfg = {}) {
  // 配置合并:外层覆盖 §1 默认(浅合并即可,键全为标量)
  const co = { ...DEFAULTS.cooccurrence, ...(cfg.cooccurrence ?? {}) };
  const forwardGain = co.forwardGain;
  // 逆流增益:先夹逼到 [minReverseGain, maxReverseGain](参考 :3015-3018)
  const reverseGain = Math.max(co.minReverseGain, Math.min(co.maxReverseGain, co.reverseGain));
  const inversionGuard = co.reverseInversionGuard;
  const sem = {
    enabled: co.semanticGainEnabled,
    peak: co.semanticGainPeak,
    sigma: co.semanticGainSigma,
    fallback: co.semanticGainLowSimFallback,
  };
  const distanceDecay = co.distanceDecay;

  // 节点 id = 输入数组下标(与 tagLayer 的 id=数组下标 契约一致)
  const vectors = (tags ?? []).map((t) => toF32(t?.vector));
  const getSim = makeGetSim(vectors, sem);

  // 按文件聚合并按 position 升序(file → [{ id, pos }])
  const fileGroups = new Map(); // key: file
  for (let id = 0; id < tags.length; id++) {
    const occ = tags[id]?.occurrences ?? [];
    for (const o of occ) {
      if (!o || o.file === undefined || o.file === null) continue; // 防御:occurrence 缺 file
      let g = fileGroups.get(o.file);
      if (!g) {
        g = [];
        fileGroups.set(o.file, g);
      }
      g.push({ id, pos: o.position });
    }
  }

  // 事实矩阵:Map<srcId, Map<dstId, weight>>,addEdge 跨文件累加
  const fact = new Map();
  const addEdge = (from, to, weight) => {
    if (!Number.isFinite(weight) || weight <= 0) return false;
    let row = fact.get(from);
    if (!row) {
      row = new Map();
      fact.set(from, row);
    }
    row.set(to, (row.get(to) || 0) + weight); // 同边多次共现求和
    return true;
  };

  // 可观测计数(照参考实现同名变量)
  let files = 0;
  let pairs = 0;
  let forwardEdges = 0;
  let backwardEdges = 0;
  let inversionClamped = 0;

  for (const group of fileGroups.values()) {
    const n = group.length;
    if (n < 2) continue; // n=1 时无对(§2.1)
    files++;
    group.sort((a, b) => a.pos - b.pos);
    pairs += (n * (n - 1)) / 2;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const t1 = group[i];
        const t2 = group[j];
        // 防御:同名 tag 同文件多重出现 → 只对不同的 tag 实体建边,不产生自环
        // (参考实现对重复 tag_id 会给自身建边;SPEC §2.1 只定义 (i≠j) 的有序对)
        if (t1.id === t2.id) continue;

        // 序位势能:越靠前的 tag 越像叙事源头
        const phi1 = phiPotential(t1.pos, n);
        const phi2 = phiPotential(t2.pos, n);
        // 距离衰减:|pos_i−pos_j| − 1(相邻衰减为 1)
        const distanceFactor = positionDistanceFactor(t2.pos - t1.pos, distanceDecay);
        const baseWeight = phi1 * phi2 * distanceFactor;

        // γ:语义增益(对称项,余弦天然对称)
        const sim = getSim(t1.id, t2.id);
        const semGain = semanticGain(sim, sem);

        // 顺流:pos_i < pos_j → t1 → t2
        const forwardWeight = baseWeight * forwardGain * semGain;

        // 逆流:t2 → t1;无概念锚增强(P2 无 IR,参考分支 β 保持关闭)
        const rawBackwardWeight = baseWeight * reverseGain * semGain;
        const cap = forwardWeight * inversionGuard; // 🛡️ 反转守卫:逆流 ≤ 顺流×0.9
        const backwardWeight = rawBackwardWeight > cap
          ? (inversionClamped++, cap)
          : rawBackwardWeight;

        if (addEdge(t1.id, t2.id, forwardWeight)) forwardEdges++;
        if (addEdge(t2.id, t1.id, backwardWeight)) backwardEdges++;
      }
    }
  }

  return {
    fact,
    diagnostics: {
      files,
      pairs,
      forwardEdges,
      backwardEdges,
      inversionClamped,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// §2.2 V9.1 传播核(_buildV9PropagationKernel 等价,:448-542)
// ─────────────────────────────────────────────────────────────────────────────
//
// 输入 fact + residualMap(P2 无 IR,恒 1.0,接口保留):
//   1. 证据压缩:e = log1p(w · evidenceCompression)
//   2. 虫洞判定:isWormhole = e·residual ≥ tensionThreshold(1)(强累计边 e≥1 即可触发)
//      传导率 g = e · (isWormhole ? wormholeGain : 1)
//   3. 入流统计:对每个目标节点累计全图【未归一化】入流 s_in(在行归一化之前);
//      median = 正入流中位数;smoothing = max(1e-9, median·hubSmoothingRatio)
//   4. 枢纽校正:relative = s_in/(median+smoothing);
//      penalty = clamp(relative^(−hubPenaltyExponent), floor, ceiling);g̃ = g·penalty
//   5. 行归一化:mainMass = outboundMass − (有虫洞边 ? reserveMass : 0);
//      主传导 = mainMass·g̃/Σg̃;虫洞边再分得 reserveMass·g̃_wh/Σg̃_wh;
//      每源节点总出流 ≤ outboundMass(0.95);虫洞边记入 wormholeEdges Set('src:dst')。
export function buildV9Kernel(factMatrix, residualMap = null, kernelCfg = {}) {
  // 参考实现同款参数夹逼(:451-463)
  const k = { ...DEFAULTS.kernel, ...(kernelCfg ?? {}) };
  const outboundMass = Math.max(0.01, Math.min(1, Number(k.outboundMass ?? 0.95)));
  const associationReserveMass = Math.min(
    Math.max(0, Number(k.associationReserveMass ?? 0.05)),
    outboundMass
  );
  const evidenceCompression = Math.max(0.01, Number(k.evidenceCompression ?? 1));
  const wormholeGain = Math.max(1, Number(k.wormholeGain ?? 1.35));
  const tensionThreshold = Math.max(0, Number(k.tensionThreshold ?? 1));
  const hubPenaltyExponent = Math.max(0, Math.min(1, Number(k.hubPenaltyExponent ?? 0.3)));
  const hubPenaltyFloor = Math.max(0.05, Math.min(1, Number(k.hubPenaltyFloor ?? 0.55)));
  const hubPenaltyCeiling = Math.max(1, Math.min(4, Number(k.hubPenaltyCeiling ?? 1.8)));
  const hubSmoothingRatio = Math.max(0.01, Math.min(2, Number(k.hubSmoothingRatio ?? 0.1)));

  // —— 第一遍:证据压缩 + 虫洞判定 + 全图未归一化入流统计 ——
  // 入流统计必须在行归一化之前,否则无法识别“从许多来源吸积少量质量”的通用枢纽。
  const rawRows = new Map(); // srcId → [ [targetId, rawConductance, isWormhole], ... ]
  const targetInflows = new Map(); // targetId → s_in(未归一化,含 wormholeGain 的传导率)

  for (const [sourceId, edges] of factMatrix.entries()) {
    if (!(edges instanceof Map) || edges.size === 0) continue;
    const rawEdges = [];
    for (const [targetId, compatWeight] of edges.entries()) {
      // 证据压缩:e = log1p(w·evidenceCompression)
      const evidence = Math.log1p(Math.max(0, Number(compatWeight) || 0) * evidenceCompression);
      // 残差:P2 无 IR,residualMap 缺省恒 1.0(接口保留,供概念锚复用)
      const residual = residualMap?.get(targetId) ?? 1;
      // 虫洞判定:强累计边 e≥1 即可触发,无需 IR
      const isWormhole = evidence * residual >= tensionThreshold;
      const rawConductance = evidence * (isWormhole ? wormholeGain : 1);
      if (!Number.isFinite(rawConductance) || rawConductance <= 0) continue;
      rawEdges.push([targetId, rawConductance, isWormhole]);
      targetInflows.set(targetId, (targetInflows.get(targetId) || 0) + rawConductance);
    }
    if (rawEdges.length > 0) rawRows.set(sourceId, rawEdges);
  }

  // 正入流中位数(参考实现:排序后取中间元素;无正入流 → 1)
  const positiveInflows = [...targetInflows.values()]
    .filter((v) => Number.isFinite(v) && v > 0)
    .sort((a, b) => a - b);
  const medianInflow = positiveInflows.length > 0
    ? positiveInflows[Math.floor(positiveInflows.length / 2)]
    : 1;
  const smoothing = Math.max(1e-9, medianInflow * hubSmoothingRatio);

  // —— 第二遍:枢纽校正(相对中位入流幂惩罚)+ 行归一化 ——
  const kernel = new Map();
  const wormholeEdges = new Set();
  for (const [sourceId, rawEdges] of rawRows.entries()) {
    const adjustedEdges = [];
    let adjustedSum = 0;
    let wormholeAdjustedSum = 0;

    for (const [targetId, rawConductance, isWormhole] of rawEdges) {
      const relativeInflow = (targetInflows.get(targetId) || 0) / (medianInflow + smoothing);
      const rawPenalty = hubPenaltyExponent > 0
        ? Math.pow(Math.max(1e-9, relativeInflow), -hubPenaltyExponent)
        : 1;
      const hubPenalty = Math.max(hubPenaltyFloor, Math.min(hubPenaltyCeiling, rawPenalty));
      const adjustedConductance = rawConductance * hubPenalty;
      if (!Number.isFinite(adjustedConductance) || adjustedConductance <= 0) continue;
      adjustedEdges.push([targetId, adjustedConductance, isWormhole]);
      adjustedSum += adjustedConductance;
      if (isWormhole) wormholeAdjustedSum += adjustedConductance;
    }

    if (adjustedSum <= 0) continue;
    // 虫洞在总预算内竞争:reserve 不产生额外能量
    const reserveMass = wormholeAdjustedSum > 0 ? associationReserveMass : 0;
    const mainMass = outboundMass - reserveMass;
    const normalizedEdges = new Map();
    for (const [targetId, adjustedConductance, isWormhole] of adjustedEdges) {
      const mainConductance = mainMass * adjustedConductance / adjustedSum;
      const associationConductance = isWormhole && wormholeAdjustedSum > 0
        ? reserveMass * adjustedConductance / wormholeAdjustedSum
        : 0;
      normalizedEdges.set(targetId, mainConductance + associationConductance);
      if (isWormhole) wormholeEdges.add(`${sourceId}:${targetId}`); // 'src:dst' 记号
    }
    kernel.set(sourceId, normalizedEdges);
  }

  return {
    kernel,
    wormholeEdges,
    diagnostics: {
      algorithmVersion: 'v9.1-hub-aware',
      medianInflow,
      targetCount: targetInflows.size,
      // 追加可观测(参考实现仅内部持有 smoothing,我们外放便于验收“median 平滑”)
      smoothing,
      positiveInflowCount: positiveInflows.length,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// §2.3 全量构建入口(buildGraph 一次性构建,代数恒为 1;跨代代数由 createTagGraph 持有)
// ─────────────────────────────────────────────────────────────────────────────
export function buildGraph(tags, cfg = {}) {
  const vectorizedTags = (tags ?? []).filter((t) => toF32(t?.vector) !== null).length;
  if (vectorizedTags < 2) {
    // §2.3:“启动:tagLayer.load 后全量构建(有 ≥2 个有向量 tag 时)”。
    // 不足 → 返回同构空图(结构与正常图一致),store 据此判“图不可用→回退 P1”。
    return {
      fact: new Map(),
      kernel: new Map(),
      wormholeEdges: new Set(),
      generation: 1,
      diagnostics: emptyDiagnostics(vectorizedTags, 'insufficient-vectorized-tags'),
    };
  }
  const { fact, diagnostics: factDiag } = buildFactMatrix(tags, cfg);
  const { kernel, wormholeEdges, diagnostics: kernelDiag } = buildV9Kernel(fact, null, cfg.kernel);
  return {
    fact,
    kernel,
    wormholeEdges,
    generation: 1,
    diagnostics: {
      ...factDiag,
      ...kernelDiag,
      vectorizedTags,
      sufficientVectorizedTags: true,
      skippedReason: null,
    },
  };
}

// 空图诊断(门槛未满足 / 生命周期初始代共用)
function emptyDiagnostics(vectorizedTags, skippedReason) {
  return {
    algorithmVersion: 'v9.1-hub-aware',
    // §2.1 可观测
    files: 0,
    pairs: 0,
    forwardEdges: 0,
    backwardEdges: 0,
    inversionClamped: 0,
    vectorizedTags,
    sufficientVectorizedTags: vectorizedTags >= 2,
    skippedReason,
    // §2.2 可观测
    medianInflow: 1,
    targetCount: 0,
    smoothing: 0,
    positiveInflowCount: 0,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// §2.3 图生命周期持有者:内存持有 { fact, kernel, wormholeEdges, generation }
// ─────────────────────────────────────────────────────────────────────────────
//
//   · 启动:tagLayer.load 后调 rebuild(tags)(有 ≥2 个有向量 tag 时构建,见 buildGraph);
//   · tagLayer.flush 后若 tagSet/occurrences 变化 → rebuildIfChanged(tags):
//     内容指纹不变则跳过(个人规模,不做增量),实际重建时 generation++;
//   · 重建异步进行(内部 Promise 链串行,任意时刻至多一条构建链,并发诉求排队);
//   · 发布 = 写时替换整代对象:recall 同步读 snapshot,永不读到半成品;
//     构建失败或异常保留上一代已发布对象。
export function createTagGraph(cfg = {}, log = () => {}) {
  let published = {
    fact: new Map(),
    kernel: new Map(),
    wormholeEdges: new Set(),
    generation: 0,
    diagnostics: emptyDiagnostics(0, 'not-built'),
  };
  let builtFingerprint = null; // 最近一次成功发布对应的内容指纹
  let chain = Promise.resolve(); // 重建串行链

  // 内容指纹:tag 名 + 完整向量值 + occurrences(向量 re-embed 后也会触发重建)
  function fingerprint(tags) {
    return createHash('md5')
      .update(
        JSON.stringify(
          (tags ?? []).map((t) => ({
            name: t?.name ?? '',
            vector: t?.vector ? Array.from(toF32(t.vector)) : null,
            occurrences: t?.occurrences ?? [],
          }))
        )
      )
      .digest('hex');
  }

  function runOnce(tags, ifChanged) {
    if (ifChanged && builtFingerprint !== null && fingerprint(tags) === builtFingerprint) {
      log('info', '[taggraph] 内容指纹未变化,跳过重建');
      return;
    }
    const built = buildGraph(tags, cfg); // 纯函数,总返回完整整代(含空图)
    // 写时替换:整代一起换,读者持有的旧代对象保持原样
    published = {
      fact: built.fact,
      kernel: built.kernel,
      wormholeEdges: built.wormholeEdges,
      generation: published.generation + 1, // 每次实际重建代数 +1(空图发布同样计数)
      diagnostics: built.diagnostics,
    };
    builtFingerprint = fingerprint(tags);
    log(
      'info',
      `[taggraph] 全量重建完成:generation=${published.generation},` +
        `fact=${published.fact.size},kernel=${published.kernel.size}`
    );
  }

  return {
    // 同步读当前已发布整代(只读契约;调用方不得改写)
    get snapshot() {
      return published;
    },
    get generation() {
      return published.generation;
    },
    // 无条件全量重建(启动/重建用);返回的 Promise 在该轮重建完成后 resolve
    rebuild(tags) {
      chain = chain
        .then(() => runOnce(tags, false))
        .catch((err) => log('error', `[taggraph] 重建异常,保留上一代:${err?.stack ?? err}`));
      return chain;
    },
    // flush 联动:内容指纹变化才重建(变化 → generation++)
    rebuildIfChanged(tags) {
      chain = chain
        .then(() => runOnce(tags, true))
        .catch((err) => log('error', `[taggraph] 重建异常,保留上一代:${err?.stack ?? err}`));
      return chain;
    },
  };
}