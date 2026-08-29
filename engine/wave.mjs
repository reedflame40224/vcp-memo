// engine/wave.mjs —— 查询融合编排(SPEC-P2 §4,applyTagBoost 等价,自写)
//
// 职责:把"EPA 动态链 + 残差金字塔种子 + V9.1 有界传播 + Core 补全 + Tag 语义去重
//      + 上下文向量 + 查询融合 + matchedTags 过滤与记号"串成一条无抛错的管线,
//      输出融合向量与诊断(info),供 store.recall 的 wave 路径消费(SPEC-P2 §6.2)。
//
// 公式与常数的唯一来源:
//   /home/lyy/workspace/vcp-src/VCPToolBox-main/TagMemoEngine.js 的
//   `applyTagBoost`(:962-1432 的核心子集)与 rag_params.json;
//   常数默认值以 SPEC-P2 §1 为准(config.tagmemo 可覆盖,默认即生产值)。
//   文档公式仅作导航,本文件逐条对齐 SPEC-P2 §4 抄录的代码级公式。
//
// 零依赖:仅静态 import 引擎内已交付的 ./propagate.mjs。纯 ESM(.mjs)。
//
// 接口(SPEC-P2 §4):
//   applyTagBoost(queryVector, baseTagBoost, coreTagNames, deps) → Promise<{
//     vector: Float32Array,          // 融合后(已归一化);info=null 时为原向量
//     info: {...} | null,           // null 表示无种子/不可用,调用方用原向量
//     energyField,                   // Map<id, energy> 传播能量场(未传播为空 Map)
//     fieldProvenance,               // Map<id, {sourceType,originType?,hop,seedId?}>
//   }>
//   deps = { epa, pyramid, tagLayer, graph, cfg }:
//     · epa: tagLayer.epa 实例 { project(v)→{logicDepth,entropy,dominantAxes}|null,
//                               detectCrossDomainResonance(v)→{resonance}|null }
//     · pyramid: tagLayer.pyramid 实例 { analyze(v)→Promise<{levels,features}> }
//     · tagLayer: 记录适配器 { records(): Array<{id,name,vector,occurrences}>,
//                             byName(name): record|null } —— id 即记录下标(tagLayer 契约)
//     · graph: taggraph snapshot { kernel, wormholeEdges, generation }
//     · cfg: 合并后的 §1 全节(建议用本文件导出的 mergeWaveCfg 生成)
//
// 已知决策(偏离规格处均在汇报中记录):
//   - §4 [1] 的 EPA project/resonance 在"未训练"时返回 null(P1 实例契约)→ wave
//     整体不可用,返回 info=null,由调用方回退 P1 路径(§0"wave 任一步失败→回退 P1");
//   - 模块内部 try/catch 兜底,applyTagBoost 永不抛出(SPEC-P2 §0:recall 永不失败);
//   - matchedTags 沿 P1 契约保留"最多 8 个"上限(SPEC-P2 §4.9 未写上限,e2e-p1 断言 ≤8);
//   - matchedTags 每条携带 level(种子为金字塔层号,涌现/虚拟核心为 0),仅作兼容附加;
//   - 幽灵节点(带向量的匿名 core 对象)按 §4.6 不实现;
//   - 语言置信度门控 langPenalty 按 §4 [4] 不实现(配置位预留)。

import { propagateSpikes } from './propagate.mjs';

// ─────────────────────────────────────────────────────────────────────────────
// §1 常数默认值(生产值;config.tagmemo 可覆盖,覆盖语义=数值替换)
// ─────────────────────────────────────────────────────────────────────────────
export const WAVE_DEFAULTS = Object.freeze({
  enabled: true,
  baseTagBoost: 0.15,            // VCP 被动注入链生产默认(RAGDiaryPlugin defaultTagWeight)
  cooccurrence: Object.freeze({
    forwardGain: 1.0,
    reverseGain: 0.35,
    minReverseGain: 0.25,
    maxReverseGain: 0.6,
    distanceDecay: 0.08,
    reverseInversionGuard: 0.9,
    semanticGainEnabled: true,
    semanticGainPeak: 0.65,
    semanticGainSigma: 0.25,
    semanticGainLowSimFallback: 0.1,
  }),
  kernel: Object.freeze({
    outboundMass: 0.95,
    associationReserveMass: 0.05,
    evidenceCompression: 1,
    wormholeGain: 1.35,
    tensionThreshold: 1,
    hubPenaltyExponent: 0.3,
    hubPenaltyFloor: 0.55,
    hubPenaltyCeiling: 1.8,
    hubSmoothingRatio: 0.1,
  }),
  spike: Object.freeze({
    maxSafeHops: 4,
    baseMomentum: 2,
    firingThreshold: 0.1,
    baseDecay: 0.25,
    wormholeDecay: 0.7,
    tensionThreshold: 1,
    maxEmergentNodes: 50,
    maxNeighborsPerNode: 20,
    returnFlowFactor: 0.15,
    firGamma: 0.6,
    maxPropagationStates: 2000,
  }),
  fusion: Object.freeze({
    dynamicBoostRange: [0.3, 2.0],
    activationMultiplierRange: [0.5, 1.5],
    coreBoostRange: [1.2, 1.4],
    coreBoostFactor: 1.33,
    tagDedupThreshold: 0.88,
    techTagThreshold: 0.08,
    normalTagThreshold: 0.015,
  }),
  dedup: Object.freeze({
    semanticThreshold: 0.92,
  }),
  // 结构补充块的补位上限(SPEC-P2 §6.2 修订):wave 路径 blocks = 语义 Top-K +
  // ≤maxSupplement 个 viaStructure 补充块;0 则完全关闭结构补充
  maxSupplement: 2,
});

// 把 config.tagmemo 节(§1 全表)与生产默认合并:缺省即用 §1 值,显式键覆盖。
export function mergeWaveCfg(tagmemo = {}) {
  const d = WAVE_DEFAULTS;
  return {
    enabled: tagmemo.enabled !== undefined ? !!tagmemo.enabled : d.enabled,
    baseTagBoost: Number.isFinite(Number(tagmemo.baseTagBoost))
      ? Number(tagmemo.baseTagBoost)
      : d.baseTagBoost,
    cooccurrence: { ...d.cooccurrence, ...(tagmemo.cooccurrence ?? {}) },
    kernel: { ...d.kernel, ...(tagmemo.kernel ?? {}) },
    spike: { ...d.spike, ...(tagmemo.spike ?? {}) },
    fusion: { ...d.fusion, ...(tagmemo.fusion ?? {}) },
    dedup: { ...d.dedup, ...(tagmemo.dedup ?? {}) },
    maxSupplement: Number.isFinite(Number(tagmemo.maxSupplement))
      ? Math.max(0, Math.floor(Number(tagmemo.maxSupplement)))
      : d.maxSupplement,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 工具:向量归一化(输入可为 Float32Array / number[] / null)
// ─────────────────────────────────────────────────────────────────────────────
function toF32(v) {
  if (v instanceof Float32Array) return v;
  if (Array.isArray(v)) return Float32Array.from(v);
  return null;
}

// 复制并归一化;零向量/非法输入返回 null(调用方自行决定语义)
function normalizeCopy(v) {
  const arr = toF32(v);
  if (!arr) return null;
  const out = new Float32Array(arr);
  let sum = 0;
  for (let i = 0; i < out.length; i++) sum += out[i] * out[i];
  const n = Math.sqrt(sum);
  if (!(n > 1e-9)) return null;
  for (let i = 0; i < out.length; i++) out[i] /= n;
  return out;
}

// 就地归一化(Float32Array 副本,读一致性)
function normalizeInPlace(out) {
  let sum = 0;
  for (let i = 0; i < out.length; i++) sum += out[i] * out[i];
  const n = Math.sqrt(sum);
  if (n > 1e-9) {
    for (let i = 0; i < out.length; i++) out[i] /= n;
    return true;
  }
  return false;
}

// 空能量场(未传播 / 传播未运行时返回的退化值)
function emptyField() {
  return { energyField: new Map(), fieldProvenance: new Map() };
}

// ─────────────────────────────────────────────────────────────────────────────
// §4 applyTagBoost —— 查询融合编排(逐条对齐 SPEC-P2 §4 编号步骤)
// ─────────────────────────────────────────────────────────────────────────────
export async function applyTagBoost(queryVector, baseTagBoost, coreTagNames, deps) {
  const origin = toF32(queryVector);
  if (!origin || origin.length === 0) {
    return { vector: queryVector, info: null, ...emptyField() };
  }
  const dim = origin.length;

  // 防御:任何一步失败只降级(info=null),绝不抛出(SPEC-P2 §0)。
  // 弯路说明:record 表在内部最前方解析,后续各步共用(与图同源,id=下标)。
  try {
    const epa = deps && deps.epa;
    const pyramid = deps && deps.pyramid;
    const tagLayer = deps && deps.tagLayer;
    const graph = deps && deps.graph;
    const cfg = deps && deps.cfg ? deps.cfg : {};
    const fusionCfg = cfg.fusion ?? {};
    const spikeCfg = cfg.spike ?? {};

    // 记录表(图节点域):id 即数组下标,与 tagLayer/taggraph 契约一致。
    const records = (() => {
      if (!tagLayer) return [];
      if (typeof tagLayer.records === 'function') {
        const r = tagLayer.records();
        return Array.isArray(r) ? r : [];
      }
      if (Array.isArray(tagLayer)) return tagLayer;
      return [];
    })();
    const recById = (id) =>
      Number.isInteger(id) && id >= 0 && id < records.length ? records[id] : null;

    // [1] EPA project + detectCrossDomainResonance + pyramid analyze(P1 实例)。
    // 未训练时 P1 实例返回 null → 动态链无从计算 → 整体回退(§0)。
    const proj = epa && typeof epa.project === 'function' ? epa.project(origin) : null;
    const res =
      epa && typeof epa.detectCrossDomainResonance === 'function'
        ? epa.detectCrossDomainResonance(origin)
        : null;
    if (!proj || !res) {
      return { vector: origin, info: null, ...emptyField() };
    }
    const pyramidResult = await pyramid.analyze(origin);
    const levels = Array.isArray(pyramidResult && pyramidResult.levels)
      ? pyramidResult.levels
      : [];
    const features = (pyramidResult && pyramidResult.features) || {};
    const logicDepth = Number(proj.logicDepth) || 0;
    const entropy = Number(proj.entropy) || 0;
    const resonanceValue = Number(res.resonance) || 0;

    // [3] 动态链:动态 booster 与动态 Core booster(SPEC-P2 §4 第 2 条)
    const actRange = fusionCfg.activationMultiplierRange ?? [0.5, 1.5];
    const activationMultiplier =
      actRange[0] + (Number(features.tagMemoActivation) || 0) * (actRange[1] - actRange[0]);
    const dynamicBoostFactor =
      ((logicDepth * (1 + Math.log(1 + resonanceValue))) / (1 + 0.5 * entropy)) *
      activationMultiplier;
    const boostRange = fusionCfg.dynamicBoostRange ?? [0.3, 2.0];
    const effectiveTagBoost =
      baseTagBoost * Math.max(boostRange[0], Math.min(boostRange[1], dynamicBoostFactor));
    const coreMetric = 0.5 * logicDepth + 0.5 * (1 - (Number(features.coverage) || 0));
    const coreRange = fusionCfg.coreBoostRange ?? [1.2, 1.4];
    const dynamicCoreBoostFactor = coreRange[0] + coreMetric * (coreRange[1] - coreRange[0]);

    // [4] 种子收集(金字塔各层 tag;语言置信度门控不实现)
    const coreTagSet = new Set((coreTagNames ?? []).map((n) => String(n).toLowerCase()));
    const allTags = [];
    const seenTagIds = new Set();
    for (const lv of levels) {
      const tags = Array.isArray(lv && lv.tags) ? lv.tags : [];
      for (const t of tags) {
        if (!t || seenTagIds.has(t.id)) continue;
        const tagName = String(t.name ?? '');
        const isCore = tagName.toLowerCase() !== '' && coreTagSet.has(tagName.toLowerCase());
        const individualRelevance = Number(t.similarity) || 0.5;
        const coreBoost = isCore
          ? dynamicCoreBoostFactor * (0.95 + individualRelevance * 0.1)
          : 1;
        const layerDecay = Math.pow(0.7, Number(lv.level) || 0);
        allTags.push({
          id: t.id,
          name: tagName,
          contribution: Number(t.contribution) || 0,
          similarity: individualRelevance,
          level: Number(lv.level) || 0,
          // adjustedWeight = contribution · layerDecay · coreBoost(§4.3)
          adjustedWeight: ((Number(t.contribution) || Number(t.weight) || 0) * layerDecay * coreBoost),
          isCore,
        });
        seenTagIds.add(t.id);
      }
    }
    const seedsCount = allTags.length;

    // [4.5] 传播:种子经传播核扩散(soft 非回溯 FIR,§3 交付物)
    let propagation = null;
    let accumulatedEnergy = new Map();
    let fieldProvenance = new Map();
    let emergentCount = 0;
    if (seedsCount > 0 && graph && graph.kernel instanceof Map && graph.kernel.size > 0) {
      const seedTags = allTags.map((t) => ({
        id: t.id,
        name: t.name,
        adjustedWeight: t.adjustedWeight,
        isCore: t.isCore,
      }));
      const result = propagateSpikes(seedTags, graph.kernel, graph.wormholeEdges ?? new Set(), spikeCfg);
      propagation = result;
      accumulatedEnergy = result.accumulatedEnergy;
      fieldProvenance = result.fieldProvenance;

      // 种子节点取 max(adjustedWeight, emergentEnergy)(防双向/循环共现不合理膨胀);
      // 涌现节点按能量降序截前 maxEmergentNodes,标记 isPullback(§4.4)。
      const allTagsMap = new Map(allTags.map((t) => [t.id, t]));
      const newAllTags = [];
      const emergentCandidates = [];
      seenTagIds.clear();
      for (const [nid, emergentEnergy] of accumulatedEnergy.entries()) {
        if (allTagsMap.has(nid)) {
          const existingTag = allTagsMap.get(nid);
          existingTag.adjustedWeight = Math.max(existingTag.adjustedWeight, emergentEnergy);
          newAllTags.push(existingTag);
          seenTagIds.add(nid);
        } else {
          const rec = recById(nid);
          emergentCandidates.push({
            id: nid,
            name: rec ? rec.name : null,
            adjustedWeight: emergentEnergy,
            isPullback: true, // 涌现节点标记
            isCore: false,
            similarity: 0,
            contribution: 0,
            level: 0,
          });
        }
      }
      emergentCandidates.sort((a, b) => b.adjustedWeight - a.adjustedWeight);
      const maxEmergentNodes = Number(spikeCfg.maxEmergentNodes) || 50;
      const topEmergent = emergentCandidates.slice(0, maxEmergentNodes);
      emergentCount = topEmergent.length;
      for (const t of topEmergent) {
        newAllTags.push(t);
        seenTagIds.add(t.id); // 截断掉的高能量涌现节点同种子一样不再重复出现
      }
      allTags.length = 0;
      allTags.push(...newAllTags);
    }

    // [4.6] Core 补全:显式 coreTagNames 中未入选的,从 tagLayer 按名取
    // (虚拟补全,标 isCore/isVirtual;幽灵节点不实现)。
    const coreBoostFactor = Number(fusionCfg.coreBoostFactor) || 1.33;
    if (coreTagSet.size > 0) {
      const missingCoreTags = [...coreTagSet].filter(
        (ct) => !allTags.some((at) => at.name && String(at.name).toLowerCase() === ct)
      );
      if (missingCoreTags.length > 0) {
        // maxBaseWeight = max(所有 tag 的 adjustedWeight / coreBoostFactor);空表 → 1.0
        const maxBaseWeight =
          allTags.length > 0
            ? Math.max(...allTags.map((t) => t.adjustedWeight / coreBoostFactor))
            : 1.0;
        for (const coreName of missingCoreTags) {
          const rec =
            typeof tagLayer.byName === 'function' ? tagLayer.byName(coreName) : null;
          if (!rec || seenTagIds.has(rec.id)) continue;
          allTags.push({
            id: rec.id,
            name: rec.name,
            vector: rec.vector ?? null,
            adjustedWeight: maxBaseWeight * dynamicCoreBoostFactor,
            isCore: true,
            isVirtual: true,
            isPullback: false,
            similarity: 0,
            contribution: 0,
            level: 0,
          });
          seenTagIds.add(rec.id);
        }
      }
    }

    // 无种子(金字塔无 tag / 全部被传播并入无) → info=null,调用方用原向量(§4.7)
    if (allTags.length === 0) {
      return { vector: origin, info: null, energyField: accumulatedEnergy, fieldProvenance };
    }

    // [5.5] Tag 语义去重:按 adjustedWeight 降序,两两余弦 > 阈值 →
    // 冗余方 20% 权重转给代表方并丢弃(core 属性转移);无向量 tag 直接跳过(照参考)。
    const dedupThreshold = Number(fusionCfg.tagDedupThreshold) || 0.88;
    const sortedTags = [...allTags].sort((a, b) => b.adjustedWeight - a.adjustedWeight);
    const deduplicatedTags = [];
    const normalizedVectorCache = new Map(); // id → { vec, norm }
    for (const tag of sortedTags) {
      const vec = toF32(tag.vector ?? recById(tag.id)?.vector);
      if (!vec || vec.length !== dim) continue;
      let normSq = 0;
      for (let d2 = 0; d2 < dim; d2++) normSq += vec[d2] * vec[d2];
      const norm = Math.sqrt(normSq);
      if (!(norm > 1e-9)) continue;
      normalizedVectorCache.set(tag.id, { vec, norm });

      let isRedundant = false;
      for (const existing of deduplicatedTags) {
        const existingCached = normalizedVectorCache.get(existing.id);
        if (!existingCached) continue;
        let dProd = 0;
        for (let d3 = 0; d3 < dim; d3++) dProd += vec[d3] * existingCached.vec[d3];
        const similarity = dProd / (norm * existingCached.norm);
        if (similarity > dedupThreshold) {
          isRedundant = true;
          existing.adjustedWeight += tag.adjustedWeight * 0.2;
          if (tag.isCore) existing.isCore = true;
          break;
        }
      }

      if (!isRedundant) {
        if (!tag.name) {
          const r = recById(tag.id);
          if (r && r.name) tag.name = r.name;
        }
        deduplicatedTags.push(tag);
      }
    }

    // [6] 上下文向量:c = normalize(Σ v_t·adjustedWeight / ΣadjustedWeight);
    // totalWeight = 0 → info=null 返回原向量(§4.7)。
    const contextVec = new Float32Array(dim);
    let totalWeight = 0;
    for (const t of deduplicatedTags) {
      const v = toF32(t.vector ?? recById(t.id)?.vector);
      if (!v || v.length !== dim) continue;
      for (let d4 = 0; d4 < dim; d4++) contextVec[d4] += v[d4] * t.adjustedWeight;
      totalWeight += t.adjustedWeight;
    }
    if (!(totalWeight > 0)) {
      return { vector: origin, info: null, energyField: accumulatedEnergy, fieldProvenance };
    }
    for (let d5 = 0; d5 < dim; d5++) contextVec[d5] /= totalWeight;
    normalizeInPlace(contextVec);

    // [6] 融合:α = min(1, effectiveTagBoost);q' = normalize((1−α)q + αc)(§4.8)
    const alpha = Math.min(1, effectiveTagBoost);
    const fused = new Float32Array(dim);
    for (let d6 = 0; d6 < dim; d6++) {
      fused[d6] = (1 - alpha) * origin[d6] + alpha * contextVec[d6];
    }
    normalizeInPlace(fused);

    // [4.9] matchedTags:Core 始终保留;技术 Tag(非中文且匹配技术字符集)阈值
    // > maxWeight·techTagThreshold;其余 > maxWeight·normalTagThreshold;
    // 每条 { tag, weight, sourceType, hop, notation },notation 含来源记号。
    const techThreshold = Number(fusionCfg.techTagThreshold) || 0.08;
    const normalThreshold = Number(fusionCfg.normalTagThreshold) || 0.015;
    const maxWeight = deduplicatedTags.length
      ? Math.max(...deduplicatedTags.map((t) => t.adjustedWeight))
      : 0;
    const matchedTags = deduplicatedTags
      .filter((t) => {
        const nm = String(t.name ?? '');
        const isTech = !/[\u4e00-\u9fa5]/.test(nm) && /^[A-Za-z0-9\-_.\s]+$/.test(nm);
        if (t.isCore) return true;
        const thr = maxWeight * (isTech ? techThreshold : normalThreshold);
        return t.adjustedWeight > thr;
      })
      .sort((a, b) => b.adjustedWeight - a.adjustedWeight)
      .slice(0, 8) // 沿 P1 契约保留"最多 8 个"(SPEC-P2 §4.9 未写上限)
      .map((t) => {
        const prov = fieldProvenance.get(t.id);
        const sourceType = t.isVirtual
          ? 'core'
          : prov
            ? prov.sourceType
            : t.isCore
              ? 'core'
              : t.isPullback
                ? 'emergent'
                : 'seed';
        const hop = t.isVirtual ? 0 : prov ? prov.hop : 0;
        const nm = String(t.name ?? '');
        const token = nm ? `${nm}:${t.adjustedWeight.toFixed(2)}` : `#${t.id}:${t.adjustedWeight.toFixed(2)}`;
        const notation =
          sourceType === 'emergent'
            ? `~${token}@emergent:${hop}` // 涌现:波浪号 + hop
            : `=${token}@${sourceType}`; // 种子/核心:等号 + core|seed
        return {
          tag: nm,
          weight: t.adjustedWeight,
          level: t.level, // 兼容附加(P1 断言需要;涌现/虚拟为 0)
          sourceType,
          hop,
          notation,
        };
      });

    const info = {
      matchedTags,
      alpha,
      boostFactor: effectiveTagBoost,
      coreTagsMatched: deduplicatedTags.filter((t) => t.isCore && t.name).map((t) => t.name),
      coreTagIds: deduplicatedTags.filter((t) => t.isCore).map((t) => t.id),
      epa: {
        logicDepth,
        entropy,
        resonance: resonanceValue,
        dominantAxes: Array.isArray(proj.dominantAxes)
          ? proj.dominantAxes.map((ax) => ({
              label: String(ax && ax.label),
              score: Number((ax && ax.score) || (ax && ax.energy)) || 0,
            }))
          : [],
      },
      pyramid: {
        depth: Number(features.depth) || 0,
        coverage: Number(features.coverage) || 0,
        novelty: Number(features.novelty) || 0,
        coherence: Number(features.coherence) || 0,
        activation: Number(features.tagMemoActivation) || 0,
      },
      propagation: propagation ? propagation.diagnostics : null,
      seeds: seedsCount,
      emergentCount,
      fieldNodes: accumulatedEnergy.size,
      graphGeneration: graph ? graph.generation ?? 0 : 0,
    };

    return { vector: fused, info, energyField: accumulatedEnergy, fieldProvenance };
  } catch (err) {
    // 防御:任何异常 → 原向量 + info=null(调用方回退 P1 并记日志)
    return { vector: origin, info: null, ...emptyField() };
  }
}