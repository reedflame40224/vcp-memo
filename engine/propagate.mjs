/**
 * engine/propagate.mjs —— 软非回溯 FIR 传播(浪潮核心,P2 保真核心)
 *
 * 移植自 /home/lyy/workspace/vcp-src/VCPToolBox-main/TagMemoEngine.js 的
 * `_propagateSpikes`(:683-954),原创作者与著作权归 VCPToolBox(CC BY-NC-SA 4.0)。
 * 按 SPEC-P2 §3 把类方法接口换成纯函数:
 *   propagateSpikes(seedTags, kernel, wormholeEdges, spikeCfg)
 * 数学与传播语义逐行对齐原版,不得走样。
 *
 * 与原版的差异(接口纯函数化,见汇报"偏离规格的决策"):
 *   1. 无 IR(SPEC-P2 §2.2 注:P2 无 IR,恒 1.0),wormholeEdges 恒为
 *      taggraph 产出的 Set<'src:dst'>,直接用 Set 分支,删除 tension 回退分支;
 *   2. riverEdgeFlow / strongestParentByNode / riverGraph 是原版供
 *      V10 RiverMemo 使用的请求级观测资产,不在 SPEC-P2 §3 契约内,不移植;
 *   3. 原版 Number(id) 强转是为适配 SQLite 字符串主键,本纯函数接口的
 *      id 直接透传(tagLayer/pyramid 的 id 本就是 number,强转是恒等变换)。
 *
 * 本文件零 npm 依赖,plain ESM。
 */

/**
 * 有界 FIR 传播:种子能量经图核扩散,逐跳 ×w_hop 累入能量场,带软非回溯、
 * 动量成本、状态硬上限与 provenance 来源记号。
 *
 * @param {Array<{id:number, name?:string, adjustedWeight:number, isCore:boolean}>} seedTags
 *   种子集(波源)。adjustedWeight 为进入传播前的加权能量,isCore 决定来源记号。
 * @param {Map<number, Map<number, number>>} kernel  §2.2 传播核:P 传导率矩阵。
 * @param {Set<string>} wormholeEdges  §2.2 产物,`Set<'src:dst'>` 虫洞边。
 * @param {object} [spikeCfg]  §1 spike 节常数,缺省即生产默认。
 * @returns {{ accumulatedEnergy: Map<number, number>,
 *             fieldProvenance: Map<number, {sourceType:string, originType?:string, hop:number, seedId?:number}>,
 *             diagnostics: object }}
 */
export function propagateSpikes(seedTags, kernel, wormholeEdges, spikeCfg = {}) {
  // ──────────────────────────── §3.1 常数 ────────────────────────────
  const MAX_SAFE_HOPS = spikeCfg.maxSafeHops ?? 4;                    // 最大安全跳数
  const BASE_MOMENTUM = spikeCfg.baseMomentum ?? 2;                   // 种子初始动量
  const FIRING_THRESHOLD = spikeCfg.firingThreshold ?? 0.1;           // 放电源阈值:能量低于此不激发
  const BASE_DECAY = spikeCfg.baseDecay ?? 0.25;                      // 普通边传播衰减
  const WORMHOLE_DECAY = spikeCfg.wormholeDecay ?? 0.7;               // 虫洞边衰减(高保真通道)
  const MAX_NEIGHBORS_PER_NODE = spikeCfg.maxNeighborsPerNode ?? 20;  // 出边按 P 降序截断数
  // V9.1 专用常数(夹逼保安全,与原版一致)
  const returnFlowFactor = Math.max(0, Math.min(1, Number(spikeCfg.returnFlowFactor ?? 0.15)));          // 软非回溯:立即回流 ×0.15
  const firGamma = Math.max(0.05, Math.min(0.95, Number(spikeCfg.firGamma ?? 0.6)));                     // FIR 时间核 γ
  const maxPropagationStates = Math.max(100, Math.floor(Number(spikeCfg.maxPropagationStates ?? 2000))); // 状态硬上限

  // ──────────────────── §3.1 FIR 归一化权重 ────────────────────
  // w_hop = γ^hop / Σ_{r=0}^{maxSafeHops} γ^r,hop 0..maxSafeHops 共 maxSafeHops+1 个。
  const firWeights = [];
  let firWeightSum = 0;
  for (let hop = 0; hop <= MAX_SAFE_HOPS; hop++) {
    const weight = Math.pow(firGamma, hop);
    firWeights.push(weight);
    firWeightSum += weight;
  }
  if (firWeightSum > 0) {
    for (let hop = 0; hop < firWeights.length; hop++) firWeights[hop] /= firWeightSum;
  }

  // 状态键为 prev:node —— V9.1 用带前驱记忆的边状态精确识别并抑制 i→j→i 立即回流。
  // provenance 记录查询场节点来自显式核心、原始种子还是第几跳涌现,供读出层区分
  // "直接事实证据"与"传播后的主题共振"。
  let activeSpikes = new Map();        // 当前跳活跃 spike 集合
  const accumulatedEnergy = new Map(); // 逐跳 ×w_hop 累入的节点能量场
  const fieldProvenance = new Map();   // 节点来源记号 {sourceType, originType?, hop, seedId?}

  // §3.2 初始:每种子一条 spike;能量即 adjustedWeight,动量 = baseMomentum(2)。
  for (const tag of seedTags) {
    const key = `seed:${tag.id}`;
    const sourceType = tag.isCore ? 'core' : 'seed';
    activeSpikes.set(key, {
      nodeId: tag.id,
      previousNodeId: null,            // 种子无前驱,首跳不会触发回流
      energy: tag.adjustedWeight,
      momentum: BASE_MOMENTUM,
      sourceType,
      hop: 0,
    });
    accumulatedEnergy.set(tag.id, tag.adjustedWeight * firWeights[0]);
    fieldProvenance.set(tag.id, {
      sourceType,
      hop: 0,
      seedId: tag.id,
    });
  }

  const diagnostics = {
    algorithmVersion: 'v9.1-soft-nonbacktracking-fir',
    returnFlowSuppressedMass: 0,       // 被软非回溯压制的回流质量
    stateTruncations: 0,               // 状态上限截断计数
    hopInFlightMass: [],               // 每跳在飞质量(诊断用)
  };

  for (let hop = 0; hop < MAX_SAFE_HOPS; hop++) {
    const nextSpikes = new Map();
    let propagated = false;
    let inFlightMass = 0;

    // §3.3 对每条活跃 spike(energy ≥ firingThreshold 且 momentum ≥ 0)放电源
    for (const spike of activeSpikes.values()) {
      if (spike.energy < FIRING_THRESHOLD || spike.momentum < 0) continue;
      const synapses = kernel.get(spike.nodeId);
      if (!synapses) continue;

      // 出边按 P 降序,截前 maxNeighborsPerNode(20) 条
      const sortedSynapses = Array.from(synapses.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, MAX_NEIGHBORS_PER_NODE);

      for (const [neighborId, coocWeight] of sortedSynapses) {
        // 虫洞判定:wormholeEdges 显式给定(SPEC §3),P2 无 IR 故无 tension 回退分支
        const isWormhole = wormholeEdges instanceof Set
          ? wormholeEdges.has(`${spike.nodeId}:${neighborId}`)
          : false;
        const decayFactor = isWormhole ? WORMHOLE_DECAY : BASE_DECAY;
        const momentumCost = isWormhole ? 0 : 1;

        // 立即回流(neighbor === spike.previousNodeId,即 i→j→i):
        // 流量 ×0.15 软非回溯,被压制的质量记入诊断。
        const isImmediateReturn = spike.previousNodeId !== null
          && neighborId === spike.previousNodeId;
        const flowFactor = isImmediateReturn ? returnFlowFactor : 1;
        const unpenalizedCurrent = spike.energy * coocWeight * decayFactor;
        const injectedCurrent = unpenalizedCurrent * flowFactor;
        if (isImmediateReturn) {
          diagnostics.returnFlowSuppressedMass += unpenalizedCurrent - injectedCurrent;
        }
        // 注入下限:过小质量丢弃
        if (injectedCurrent < 0.01) continue;

        // 动量扣除;耗尽(nextMomentum < 0)且非虫洞 → 丢弃(虫洞零成本可继续走)
        const nextMomentum = spike.momentum - momentumCost;
        if (nextMomentum < 0 && !isWormhole) continue;

        // 状态键 prev:node;同键合并:能量累加、动量取 max、hop 取小
        const stateKey = `${spike.nodeId}:${neighborId}`;
        const existing = nextSpikes.get(stateKey);
        if (existing) {
          existing.energy += injectedCurrent;
          existing.momentum = Math.max(existing.momentum, nextMomentum);
          if (spike.hop + 1 < existing.hop) {
            existing.hop = spike.hop + 1;
            existing.sourceType = spike.sourceType;
          }
        } else {
          nextSpikes.set(stateKey, {
            nodeId: neighborId,
            previousNodeId: spike.nodeId,
            energy: injectedCurrent,
            momentum: nextMomentum,
            sourceType: spike.sourceType,
            hop: spike.hop + 1,
          });
        }
      }
    }

    // §3.4 状态总量硬上限:超过 maxPropagationStates 按能量降序截断(记诊断)。
    if (nextSpikes.size > maxPropagationStates) {
      const retained = [...nextSpikes.entries()]
        .sort((a, b) => b[1].energy - a[1].energy)
        .slice(0, maxPropagationStates);
      diagnostics.stateTruncations += nextSpikes.size - retained.length;
      nextSpikes.clear();
      for (const [key, value] of retained) nextSpikes.set(key, value);
    }

    // §3.5 每跳结束:节点能量聚合 ×w_{hop+1} 累入 accumulatedEnergy;
    // provenance 记 emergent + 最早 hop + originType(源于哪类种子)。
    const nodeEnergyThisHop = new Map();
    for (const newSpike of nextSpikes.values()) {
      nodeEnergyThisHop.set(
        newSpike.nodeId,
        (nodeEnergyThisHop.get(newSpike.nodeId) || 0) + newSpike.energy,
      );
      const previousProvenance = fieldProvenance.get(newSpike.nodeId);
      if (!previousProvenance || newSpike.hop < previousProvenance.hop) {
        fieldProvenance.set(newSpike.nodeId, {
          sourceType: 'emergent',
          originType: newSpike.sourceType,
          hop: newSpike.hop,
        });
      }
      inFlightMass += newSpike.energy;
    }
    diagnostics.hopInFlightMass.push(inFlightMass);

    const fieldWeight = firWeights[hop + 1];
    for (const [nodeId, energy] of nodeEnergyThisHop.entries()) {
      accumulatedEnergy.set(
        nodeId,
        (accumulatedEnergy.get(nodeId) || 0) + energy * fieldWeight,
      );
      if (energy > 0.01) propagated = true;
    }

    // §3.6 无有效传播(本跳最大节点能量 ≤ 0.01)提前终止。
    if (!propagated) break;
    activeSpikes = nextSpikes;
  }

  return { accumulatedEnergy, fieldProvenance, diagnostics };
}