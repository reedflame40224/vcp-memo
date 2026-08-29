// tests/wave.test.mjs —— SPEC-P2 §8 第 4 条 wave 编排自测(node 直接运行,无框架)
//
// 覆盖:动态链公式手算对照(已知 logicDepth/entropy/resonance/activation →
//      activationMultiplier/dynamicBoostFactor/effectiveTagBoost/alpha);
//      动态 Core booster 公式;contextVec 加权正确(含 propagation 涌现节点,
//      FIR 权重独立重算);q' = normalize((1−α)q+αc) 且 |q'|=1;
//      无种子 → info=null 返回原向量;Core 补全(虚拟核心权重 = maxBaseWeight·boost);
//      Tag 语义去重 20% 权重转移;boost 夹逼 [0.3,2.0];matchedTags 记号
//      (=name@seed / =name@core / ~name@emergent:n)与阈值过滤;≥9 种子时上限 8;
//      §1 常数表与 mergeWaveCfg 覆盖语义。
// 对照对象:SPEC-P2 §4(公式行号锚定 TagMemoEngine.js applyTagBoost)。
// 用法:node tests/wave.test.mjs

import assert from 'node:assert/strict';
import { applyTagBoost, mergeWaveCfg, WAVE_DEFAULTS } from '../engine/wave.mjs';

let passed = 0;
let failed = 0;
function run(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ✔ ${name}`);
  } catch (err) {
    failed++;
    console.error(`  ✘ ${name}\n    ${err?.stack ?? err}`);
  }
}
function collectAsync(name, fn) {
  return fn()
    .then(() => {
      passed++;
      console.log(`  ✔ ${name}`);
    })
    .catch((err) => {
      failed++;
      console.error(`  ✘ ${name}\n    ${err?.stack ?? err}`);
    });
}

const DIM = 8;

// 单位轴向量
function unit(axis) {
  const v = new Float32Array(DIM);
  v[axis] = 1;
  return v;
}
// 由前几个分量构造单位向量
function normalize(components) {
  let s = 0;
  for (const x of components) s += x * x;
  const n = Math.sqrt(s);
  const v = new Float32Array(DIM);
  for (let i = 0; i < components.length; i++) v[i] = components[i] / n;
  return v;
}
function magnitude(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}
function cosine(a, b) {
  let d = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}
function approx(actual, expected, eps = 1e-9, msg) {
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `${msg ?? '浮点'}:期望 ${expected},实际 ${actual},|Δ|=${Math.abs(actual - expected)} > ${eps}`,
  );
}

// 测试内独立重算 FIR 权重(与 propagate.test 同一公式,w_hop = γ^hop/Σγ^r)
function calcFirWeights(gamma, maxHops) {
  const raw = [];
  let sum = 0;
  for (let hop = 0; hop <= maxHops; hop++) {
    raw.push(Math.pow(gamma, hop));
    sum += raw[hop];
  }
  return raw.map((w) => w / sum);
}
const W = calcFirWeights(0.6, 4); // [w0..w4]

// 记录构造:{ name, vector, occurrences? } → tagLayer 风格记录(id=下标)
function rec(name, vector, occ = []) {
  return { name, vector, occurrences: occ };
}

// 全参假 deps 工厂
function makeDeps({
  records,
  levels,
  features,
  logicDepth,
  entropy,
  resonance,
  kernel,
  wormholes,
  generation,
}) {
  return {
    epa: {
      project: () => ({ logicDepth, entropy, dominantAxes: [{ label: 'W', score: 1 }] }),
      detectCrossDomainResonance: () => ({ resonance }),
    },
    pyramid: {
      analyze: async () => ({ levels, features }),
    },
    tagLayer: {
      records: () => records,
      byName: (name) =>
        records.find(
          (r) => r.name && String(r.name).toLowerCase() === String(name).toLowerCase(),
        ) || null,
    },
    graph: {
      kernel: kernel ?? new Map(),
      wormholeEdges: wormholes ?? new Set(),
      generation: generation ?? 3,
    },
    cfg: mergeWaveCfg(),
  };
}

// 基准 EPA/金字塔输入(数值经手算核对,见各用例注释)
const EPA_BASE = { logicDepth: 0.5, entropy: 0.5, resonance: Math.E - 1 };
const FEATURES_BASE = {
  depth: 1,
  coverage: 0.6,
  novelty: 0.4,
  coherence: 0.8,
  tagMemoActivation: 0.5,
};

// ───────────────────── 汇总(所有异步用例在此统一注册并等待) ─────────────────────
Promise.all([
  // ── 1. 动态链公式手算对照 ──
  collectAsync('动态链:手算对照 activationMultiplier / dynamicBoostFactor / effectiveTagBoost / alpha', () =>
    (async () => {
      // 手算:logicDepth=0.5,entropy=0.5,resonance=e−1(ln(1+res)=1),activation=0.5
      //   activationMultiplier = 0.5 + 0.5·(1.5−0.5) = 1.0
      //   dynamicBoostFactor  = (0.5·(1+1)/(1+0.5·0.5))·1.0 = (1.0/1.25) = 0.8
      //   effectiveTagBoost   = 0.15·clamp(0.8, 0.3, 2.0) = 0.12;alpha = min(1,0.12) = 0.12
      //   coreMetric          = 0.5·0.5 + 0.5·(1−0.6) = 0.45
      //   dynamicCoreBoost    = 1.2 + 0.45·(1.4−1.2) = 1.29
      const records = [rec('阿尔法', unit(2)), rec('贝塔', unit(3))];
      const levels = [
        { level: 0, tags: [{ id: 0, name: '阿尔法', similarity: 0.5, contribution: 1.0 }] },
      ];
      const deps = makeDeps({
        records, levels, features: FEATURES_BASE, ...EPA_BASE,
        kernel: new Map(), // 无传播:只验动态链 + 融合
      });
      const q = unit(0);
      const out = await applyTagBoost(q, 0.15, [], deps);
      assert.ok(out.info, 'info 不应为 null');
      approx(out.info.alpha, 0.12, 1e-9, 'α');
      approx(out.info.boostFactor, 0.12, 1e-9, 'effectiveTagBoost');
      approx(out.info.epa.logicDepth, 0.5, 1e-9, 'logicDepth');
      approx(out.info.epa.entropy, 0.5, 1e-9, 'entropy');
      approx(out.info.epa.resonance, Math.E - 1, 1e-9, 'resonance');
      approx(out.info.pyramid.activation, 0.5, 1e-9, 'activation');
      assert.strictEqual(out.info.seeds, 1, '种子数');
      assert.strictEqual(out.info.emergentCount, 0, '涌现数');
      assert.strictEqual(out.info.fieldNodes, 0, '能量场节点数(无传播)');
      assert.strictEqual(out.info.graphGeneration, 3);
    })()),

  // ── 2. contextVec 加权 + q' 归一化 ──
  collectAsync('contextVec 加权正确(含涌现节点)+ q\' = normalize((1−α)q+αc) 且 |q\'|=1', () =>
    (async () => {
      // 手算基准(本用例全部数值由测试独立推导):
      //   FIR(γ=0.6,maxHops=4):W = [w0..w4] 见 calcFirWeights
      //   传播:种子 A(weight=1.0) 经核 P=4.0 → B:注入 = 1.0·4.0·0.25 = 1.0(momentum 2→1)
      //     acc[A] = 1·w0 ≈ 0.434 < 1.0 → A.weight 保持 1.0(Math.max 语义)
      //     acc[B] = 1·w1 → B 为 emergent(hop=1,originType=seed),weight = w1
      //   contextVec = normalize((vA·1.0 + vB·w1) / (1 + w1))
      //   α = 0.12(动态链同用例 1);q\' = normalize(0.88·q + 0.12·c)
      const records = [rec('阿尔法', unit(2)), rec('B', unit(3))];
      const levels = [
        { level: 0, tags: [{ id: 0, name: '阿尔法', similarity: 0.5, contribution: 1.0 }] },
      ];
      const kernel = new Map([[0, new Map([[1, 4.0]])]]); // A → B,强边令注入=1.0
      const deps = makeDeps({
        records, levels, features: FEATURES_BASE, ...EPA_BASE, kernel,
      });
      const q = unit(0);
      const out = await applyTagBoost(q, 0.15, [], deps);
      assert.ok(out.info, 'info 不应为 null');

      // 涌现 B 的能量 = 1·w1(FIR 独立重算)
      const wB = W[1];
      approx((out.info.matchedTags.find((m) => m.tag === 'B') || {}).weight, wB, 1e-9, 'B weight(w1)');

      // 手算 contextVec:加权平均后归一化
      const vA = unit(2);
      const vB = unit(3);
      const total = 1.0 + wB;
      const rawC = new Float32Array(DIM);
      for (let i = 0; i < DIM; i++) rawC[i] = (vA[i] * 1.0 + vB[i] * wB) / total;
      const cExpected = normalize(Array.from(rawC));
      // 通过融合向量反推 c 与手算对照:q' = normalize((1−α)·q + α·c)
      const alpha = out.info.alpha;
      approx(alpha, 0.12, 1e-9, 'α');
      const rawFused = new Float32Array(DIM);
      for (let i = 0; i < DIM; i++) rawFused[i] = (1 - alpha) * q[i] + alpha * cExpected[i];
      const fusedExpected = normalize(Array.from(rawFused));
      // |q'|=1:Float32 归一化有 ~1e-8 舍入,放宽到 1e-6(分量级比对同 1e-6)
      approx(magnitude(out.vector), 1.0, 1e-6, '|q\'|=1');
      for (let i = 0; i < DIM; i++) approx(out.vector[i], fusedExpected[i], 1e-6, `q'[${i}]`);
      assert.ok(magnitude(fusedExpected) > 0.99);

      // matchedTags:A(seed)+ B(emergent)按 weight 降序;notation 含来源记号
      const mt = out.info.matchedTags;
      assert.strictEqual(mt.length, 2, 'matchedTags 2 条');
      assert.strictEqual(mt[0].tag, '阿尔法');
      assert.strictEqual(mt[0].sourceType, 'seed');
      assert.strictEqual(mt[0].hop, 0);
      assert.strictEqual(mt[0].notation, `=阿尔法:${mt[0].weight.toFixed(2)}@seed`);
      assert.strictEqual(mt[1].tag, 'B');
      assert.strictEqual(mt[1].sourceType, 'emergent');
      assert.strictEqual(mt[1].hop, 1);
      assert.strictEqual(mt[1].notation, `~B:${mt[1].weight.toFixed(2)}@emergent:1`);
      // 能量场与 provenance
      assert.ok(out.energyField instanceof Map);
      assert.strictEqual(out.energyField.size, 2, '能量场含 A、B');
      assert.strictEqual(out.fieldProvenance.get(1).sourceType, 'emergent');
      assert.strictEqual(out.fieldProvenance.get(1).hop, 1);
    })()),

  // ── 3. 无种子 → info=null 返回原向量 ──
  collectAsync('无种子:金字塔空层 → info=null,vector 为原向量,energyField 空', () =>
    (async () => {
      const records = [rec('阿尔法', unit(2))];
      const deps = makeDeps({
        records,
        levels: [], // 金字塔无标签 → 无种子
        features: { depth: 0, coverage: 0, novelty: 1, coherence: 0, tagMemoActivation: 0 },
        ...EPA_BASE,
      });
      const q = unit(0);
      const out = await applyTagBoost(q, 0.15, [], deps);
      assert.strictEqual(out.info, null, 'info=null');
      assert.ok(out.vector instanceof Float32Array);
      for (let i = 0; i < DIM; i++) approx(out.vector[i], q[i], 1e-9, `原向量[${i}]`);
      assert.strictEqual(out.energyField.size, 0);
      assert.strictEqual(out.fieldProvenance.size, 0);
    })()),

  // ── 4. Core 补全(虚拟核心) ──
  collectAsync('Core 补全:未入选 core 名从 tagLayer 按名取,虚拟核心权重 = maxBaseWeight·dynamicCoreBoostFactor', () =>
    (async () => {
      // maxBaseWeight = max(1.0/1.33) = 0.75188;dynamicCoreBoostFactor = 1.29(用例 1 手算)
      // 西塔 weight = 0.75188·1.29 = 0.96992 → matchedTags 中为 core,notation @core
      const records = [rec('阿尔法', unit(2)), rec('贝塔', unit(3)), rec('西塔', unit(4))];
      const levels = [
        { level: 0, tags: [{ id: 0, name: '阿尔法', similarity: 0.5, contribution: 1.0 }] },
      ];
      const deps = makeDeps({
        records, levels, features: FEATURES_BASE, ...EPA_BASE,
        kernel: new Map(),
      });
      const q = unit(0);
      const out = await applyTagBoost(q, 0.15, ['西塔'], deps);
      assert.ok(out.info, 'info 不应为 null');
      const xita = out.info.matchedTags.find((m) => m.tag === '西塔');
      assert.ok(xita, '西塔 应出现在 matchedTags(Core 始终保留)');
      approx(xita.weight, (1.0 / 1.33) * 1.29, 1e-9, '虚拟核心权重(maxBaseWeight·boost)');
      assert.strictEqual(xita.sourceType, 'core');
      assert.strictEqual(xita.hop, 0);
      assert.strictEqual(xita.notation, `=西塔:${xita.weight.toFixed(2)}@core`);
      assert.ok(out.info.coreTagsMatched.includes('西塔'), 'coreTagsMatched 含西塔');
      // 阿尔法 仍在(seed)
      assert.ok(out.info.matchedTags.some((m) => m.tag === '阿尔法' && m.sourceType === 'seed'));
    })()),

  // ── 5. Tag 语义去重 20% 转移 ──
  collectAsync('Tag 语义去重:余弦>0.88 冗余方 20% 权重转移给代表方并丢弃', () =>
    (async () => {
      // A 与 D 余弦 ≈ 0.98(>0.88):D 冗余 → A.weight += D.weight·0.2;D 不在 matchedTags
      const vA = unit(1);
      const vD = normalize([0, 0.98, 0.199]); // 轴 1、2 分量:cos(unit(1),vD)=0.98/√(0.98²+0.199²) ≈ 0.98
      assert.ok(cosine(vA, vD) > 0.88, `前提:cos(A,D)=${cosine(vA, vD)} > 0.88`);
      const records = [rec('A', vA), rec('D', vD), rec('C', unit(3))];
      const levels = [
        {
          level: 0,
          tags: [
            { id: 0, name: 'A', similarity: 0.9, contribution: 1.0 },
            { id: 1, name: 'D', similarity: 0.9, contribution: 0.5 },
            { id: 2, name: 'C', similarity: 0.5, contribution: 0.8 },
          ],
        },
      ];
      const deps = makeDeps({
        records, levels, features: FEATURES_BASE, ...EPA_BASE,
        kernel: new Map(),
      });
      const q = unit(0);
      const out = await applyTagBoost(q, 0.15, [], deps);
      assert.ok(out.info, 'info 不应为 null');
      const a = out.info.matchedTags.find((m) => m.tag === 'A');
      assert.ok(a, 'A 保留为代表方');
      approx(a.weight, 1.0 + 0.5 * 0.2, 1e-9, 'A 吸收 D 的 20% 权重');
      assert.ok(!out.info.matchedTags.some((m) => m.tag === 'D'), 'D 被丢弃');
      assert.ok(out.info.matchedTags.some((m) => m.tag === 'C'), 'C 保留');
    })()),

  // ── 6. boost 夹逼 [0.3, 2.0] ──
  collectAsync('boost 夹逼:dynamicBoostFactor 上下限 → effectiveTagBoost 与 α 正确', () =>
    (async () => {
      // 上限:logicDepth=1, entropy=0, resonance=e³−1(ln(1+res)=3), activation=1
      //   activationMultiplier = 0.5 + 1·1 = 1.5;dynBoost = (1·4/1)·1.5 = 6 → clamp 2.0
      //   effBoost = 0.15·2.0 = 0.3 → α = 0.3
      const depsHi = makeDeps({
        records: [rec('甲', unit(2))],
        levels: [{ level: 0, tags: [{ id: 0, name: '甲', similarity: 0.5, contribution: 1.0 }] }],
        features: { depth: 1, coverage: 0.6, novelty: 0.4, coherence: 0.8, tagMemoActivation: 1.0 },
        logicDepth: 1,
        entropy: 0,
        resonance: Math.exp(3) - 1,
        kernel: new Map(),
      });
      const outHi = await applyTagBoost(unit(0), 0.15, [], depsHi);
      assert.ok(outHi.info);
      approx(outHi.info.boostFactor, 0.3, 1e-9, 'effBoost 上限');
      approx(outHi.info.alpha, 0.3, 1e-9, 'α 上限');

      // 下限:logicDepth→0 → dynBoost=0 → clamp 0.3 → effBoost = 0.045 → α = 0.045
      const depsLo = makeDeps({
        records: [rec('乙', unit(2))],
        levels: [{ level: 0, tags: [{ id: 0, name: '乙', similarity: 0.5, contribution: 1.0 }] }],
        features: FEATURES_BASE,
        logicDepth: 0,
        entropy: 0.5,
        resonance: 0,
        kernel: new Map(),
      });
      const outLo = await applyTagBoost(unit(0), 0.15, [], depsLo);
      assert.ok(outLo.info);
      approx(outLo.info.boostFactor, 0.045, 1e-9, 'effBoost 下限');
      approx(outLo.info.alpha, 0.045, 1e-9, 'α 下限');
    })()),

  // ── 7. matchedTags 阈值与上限 8 ──
  collectAsync('matchedTags:技术 Tag 阈值过滤(0.08)与 normalTagThreshold(0.015)区分', () =>
    (async () => {
      // A(中文,weight=1.0)→ 恒保留;E(英文技术名,weight=0.05 < 0.08·1.0)→ 过滤
      const records = [rec('中文甲', unit(1)), rec('ENGLISH_TAG', unit(2))];
      const levels = [
        {
          level: 0,
          tags: [
            { id: 0, name: '中文甲', similarity: 0.5, contribution: 1.0 },
            { id: 1, name: 'ENGLISH_TAG', similarity: 0.5, contribution: 0.05 },
          ],
        },
      ];
      const deps = makeDeps({
        records, levels, features: FEATURES_BASE, ...EPA_BASE,
        kernel: new Map(),
      });
      const out = await applyTagBoost(unit(0), 0.15, [], deps);
      assert.ok(out.info);
      const names = out.info.matchedTags.map((m) => m.tag);
      assert.ok(names.includes('中文甲'), '中文甲 应保留');
      assert.ok(!names.includes('ENGLISH_TAG'), '技术弱权 0.05 < 0.08 应被过滤');
    })()),

  collectAsync('matchedTags:种子超 9 个时上限截断为 8', () =>
    (async () => {
      // 9 个不同 tag 需要 9 个不同轴向(dim 局部取 12,避免单位轴写出界被忽略)
      const DIM12 = 12;
      const records = [];
      const levelTags = [];
      for (let i = 0; i < 9; i++) {
        const v = new Float32Array(DIM12);
        v[2 + i] = 1;
        records.push(rec(`标签${i}`, v));
        levelTags.push({ id: i, name: `标签${i}`, similarity: 0.5, contribution: 1.0 });
      }
      const deps = makeDeps({
        records,
        levels: [{ level: 0, tags: levelTags }],
        features: FEATURES_BASE,
        ...EPA_BASE,
        kernel: new Map(),
      });
      const q = new Float32Array(DIM12);
      q[0] = 1;
      const out = await applyTagBoost(q, 0.15, [], deps);
      assert.ok(out.info);
      assert.strictEqual(out.info.matchedTags.length, 8, 'matchedTags 上限 8');
    })()),
])
  .then(() => {
    console.log(`\nwave.test.mjs:通过 ${passed} 项,失败 ${failed} 项`);
    process.exit(failed ? 1 : 0);
  })
  .catch((err) => {
    console.error('\nwave.test.mjs 内部错误:', err);
    process.exit(1);
  });

// ───────────────────── 同步用例(常数表与合并语义) ─────────────────────
run('§1 常数表:WAVE_DEFAULTS 数值逐项核对', () => {
  assert.strictEqual(WAVE_DEFAULTS.enabled, true);
  assert.strictEqual(WAVE_DEFAULTS.baseTagBoost, 0.15);
  assert.strictEqual(WAVE_DEFAULTS.cooccurrence.forwardGain, 1.0);
  assert.strictEqual(WAVE_DEFAULTS.cooccurrence.reverseGain, 0.35);
  assert.strictEqual(WAVE_DEFAULTS.cooccurrence.minReverseGain, 0.25);
  assert.strictEqual(WAVE_DEFAULTS.cooccurrence.maxReverseGain, 0.6);
  assert.strictEqual(WAVE_DEFAULTS.cooccurrence.distanceDecay, 0.08);
  assert.strictEqual(WAVE_DEFAULTS.cooccurrence.reverseInversionGuard, 0.9);
  assert.strictEqual(WAVE_DEFAULTS.cooccurrence.semanticGainPeak, 0.65);
  assert.strictEqual(WAVE_DEFAULTS.cooccurrence.semanticGainSigma, 0.25);
  assert.strictEqual(WAVE_DEFAULTS.cooccurrence.semanticGainLowSimFallback, 0.1);
  assert.strictEqual(WAVE_DEFAULTS.kernel.outboundMass, 0.95);
  assert.strictEqual(WAVE_DEFAULTS.kernel.associationReserveMass, 0.05);
  assert.strictEqual(WAVE_DEFAULTS.kernel.evidenceCompression, 1);
  assert.strictEqual(WAVE_DEFAULTS.kernel.wormholeGain, 1.35);
  assert.strictEqual(WAVE_DEFAULTS.kernel.tensionThreshold, 1);
  assert.strictEqual(WAVE_DEFAULTS.kernel.hubPenaltyExponent, 0.3);
  assert.strictEqual(WAVE_DEFAULTS.kernel.hubPenaltyFloor, 0.55);
  assert.strictEqual(WAVE_DEFAULTS.kernel.hubPenaltyCeiling, 1.8);
  assert.strictEqual(WAVE_DEFAULTS.kernel.hubSmoothingRatio, 0.1);
  assert.strictEqual(WAVE_DEFAULTS.spike.maxSafeHops, 4);
  assert.strictEqual(WAVE_DEFAULTS.spike.baseMomentum, 2);
  assert.strictEqual(WAVE_DEFAULTS.spike.firingThreshold, 0.1);
  assert.strictEqual(WAVE_DEFAULTS.spike.baseDecay, 0.25);
  assert.strictEqual(WAVE_DEFAULTS.spike.wormholeDecay, 0.7);
  assert.strictEqual(WAVE_DEFAULTS.spike.maxEmergentNodes, 50);
  assert.strictEqual(WAVE_DEFAULTS.spike.maxNeighborsPerNode, 20);
  assert.strictEqual(WAVE_DEFAULTS.spike.returnFlowFactor, 0.15);
  assert.strictEqual(WAVE_DEFAULTS.spike.firGamma, 0.6);
  assert.strictEqual(WAVE_DEFAULTS.spike.maxPropagationStates, 2000);
  assert.deepStrictEqual(WAVE_DEFAULTS.fusion.dynamicBoostRange, [0.3, 2.0]);
  assert.deepStrictEqual(WAVE_DEFAULTS.fusion.activationMultiplierRange, [0.5, 1.5]);
  assert.deepStrictEqual(WAVE_DEFAULTS.fusion.coreBoostRange, [1.2, 1.4]);
  assert.strictEqual(WAVE_DEFAULTS.fusion.coreBoostFactor, 1.33);
  assert.strictEqual(WAVE_DEFAULTS.fusion.tagDedupThreshold, 0.88);
  assert.strictEqual(WAVE_DEFAULTS.fusion.techTagThreshold, 0.08);
  assert.strictEqual(WAVE_DEFAULTS.fusion.normalTagThreshold, 0.015);
  assert.strictEqual(WAVE_DEFAULTS.dedup.semanticThreshold, 0.92);
});

run('mergeWaveCfg:缺省即生产值,显式键覆盖(§1 覆盖语义)', () => {
  const d = mergeWaveCfg();
  assert.strictEqual(d.enabled, true);
  assert.strictEqual(d.baseTagBoost, 0.15);
  assert.strictEqual(d.fusion.tagDedupThreshold, 0.88);
  const overridden = mergeWaveCfg({
    enabled: false,
    baseTagBoost: 0.25,
    fusion: { tagDedupThreshold: 0.9 },
    spike: { maxEmergentNodes: 10 },
  });
  assert.strictEqual(overridden.enabled, false);
  assert.strictEqual(overridden.baseTagBoost, 0.25);
  assert.strictEqual(overridden.fusion.tagDedupThreshold, 0.9);
  assert.strictEqual(overridden.spike.maxEmergentNodes, 10);
  // 覆盖不影响未显式键
  assert.strictEqual(overridden.spike.firGamma, 0.6);
  assert.strictEqual(overridden.dedup.semanticThreshold, 0.92);
});