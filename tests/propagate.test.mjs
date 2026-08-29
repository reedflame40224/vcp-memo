// tests/propagate.test.mjs —— SPEC-P2 §8 第 3 条 传播核自测(node 直接运行,无框架)
//
// 覆盖:FIR 权重(和=1、相邻比=γ,公式 + 行为双验证)、软非回溯回流 ×0.15
//      (A→B→A)、momentum=2 普通边 2 跳停止、虫洞零动量成本豁免、
//      hop ≤ maxHops、状态上限截断、出边截断 20、firingThreshold 放电源截断、
//      注入 < 0.01 丢弃、同状态键合并(能量累加/动量取 max)、provenance
//      core/seed/emergent + 最早 hop + originType、空种子退化。
// 对照对象:TagMemoEngine.js `_propagateSpikes`(:683-954)逐行移植的
//          engine/propagate.mjs(SPEC-P2 §3)。
// 用法:node tests/propagate.test.mjs

import assert from 'node:assert/strict';
import { propagateSpikes } from '../engine/propagate.mjs';

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

// 浮点近似断言(绝对误差)
function approx(actual, expected, eps = 1e-9, msg) {
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `${msg ?? '浮点数值'}:期望 ${expected},实际 ${actual},|Δ|=${Math.abs(actual - expected)} > ${eps}`,
  );
}

// 测试内独立重算 FIR 权重:w_hop = γ^hop / Σ_{r=0}^{maxHops} γ^r
function calcFirWeights(gamma, maxHops) {
  const raw = [];
  let sum = 0;
  for (let hop = 0; hop <= maxHops; hop++) {
    raw.push(Math.pow(gamma, hop));
    sum += raw[hop];
  }
  return raw.map((w) => w / sum);
}

// 构建单一条链 kernel:node → node+1,统一权重 P
function chainKernel(nodes, P) {
  const kernel = new Map();
  for (let i = 1; i < nodes; i++) {
    kernel.set(i, new Map([[i + 1, P]]));
  }
  return kernel;
}
function chainWormholes(nodes) {
  const set = new Set();
  for (let i = 1; i < nodes; i++) set.add(`${i}:${i + 1}`);
  return set;
}

const W = calcFirWeights(0.6, 4); // [w0..w4],由测试独立推导,仅作期望值基准

// ───────────────────── 1. FIR 权重:公式与行为双验证 ─────────────────────
run('FIR 权重和 = 1 且相邻比 = γ(公式)', () => {
  let sum = 0;
  for (let i = 0; i < W.length; i++) {
    sum += W[i];
    if (i > 0) approx(W[i] / W[i - 1], 0.6, 1e-12, `w${i}/w${i - 1}`);
  }
  approx(sum, 1, 1e-12, '权重和');
  assert.strictEqual(W.length, 5, 'hop 0..4 共 5 个权重');
});

// 行为验证:虫洞链每次注入恒为 1.0 → accumulatedEnergy 依次 = w0..w4
run('FIR 权重被实际用于逐跳累入(虫洞链 A→E)', () => {
  const P = 1 / 0.7; // 注入 = 1·P·(虫洞衰减 0.7) ≡ 1.0
  const kernel = chainKernel(5, P);
  const wormholes = chainWormholes(5);
  const { accumulatedEnergy: acc, fieldProvenance: prov, diagnostics } =
    propagateSpikes([{ id: 1, adjustedWeight: 1, isCore: false }], kernel, wormholes);
  approx(acc.get(1), W[0], 1e-9, 'acc[A]');
  approx(acc.get(2), W[1], 1e-9, 'acc[B]');
  approx(acc.get(3), W[2], 1e-9, 'acc[C]');
  approx(acc.get(4), W[3], 1e-9, 'acc[D]');
  approx(acc.get(5), W[4], 1e-9, 'acc[E]');
  // 行为层面的相邻比:同一恒定注入下 adjacent ratio = γ
  approx(acc.get(5) / acc.get(4), 0.6, 1e-9, '行为相邻比');
  assert.strictEqual(prov.get(5).hop, 4);
  assert.strictEqual(diagnostics.stateTruncations, 0);
});

// ───────────────── 2. 软非回溯:A→B→A 立即回流 ×0.15 ─────────────────
run('回流 ×0.15:构造 A→B→A 检查压制的回流质量与能量场', () => {
  const kernel = new Map([
    [1, new Map([[2, 0.5]])], // A → B 强边
    [2, new Map([[1, 0.5]])], // B → A 边(构成立即回流)
  ]);
  const { accumulatedEnergy: acc, fieldProvenance: prov, diagnostics } =
    propagateSpikes([{ id: 1, adjustedWeight: 5, isCore: false }], kernel, new Set());
  // 首跳注入 B:5·0.5·0.25 = 0.625(恰好为二进制精确值)
  const toB = 5 * 0.5 * 0.25; // 0.625
  // 回流段未惩罚量:0.625·0.5·0.25 = 0.078125;×0.15 后注入 = 0.01171875
  const unpenalized = toB * 0.5 * 0.25; // 0.078125
  const injectedBack = unpenalized * 0.15; // 0.01171875
  approx(acc.get(2), toB * W[1], 1e-12, 'acc[B]');
  approx(acc.get(1), 5 * W[0] + injectedBack * W[2], 1e-12, 'acc[A](含 ×0.15 回流)');
  approx(diagnostics.returnFlowSuppressedMass, unpenalized - injectedBack, 1e-12, '压制质量');
  assert.strictEqual(diagnostics.stateTruncations, 0);
  // 回流 3 跳才回到 A:若未压制应为 ×1.0,此处断言注入确为 ×0.15 后的量
  assert.deepStrictEqual(prov.get(1), { sourceType: 'seed', hop: 0, seedId: 1 });
  assert.deepStrictEqual(prov.get(2), { sourceType: 'emergent', originType: 'seed', hop: 1 });
});

// ───────────────── 3. momentum 耗尽:普通边走 2 跳停止 ─────────────────
run('momentum=2 普通边走 2 跳后停止(链 1→2→3→4→5)', () => {
  const kernel = chainKernel(5, 1.8); // 全部普通边
  const { accumulatedEnergy: acc, diagnostics } =
    propagateSpikes([{ id: 1, adjustedWeight: 1, isCore: false }], kernel, new Set());
  approx(acc.get(2), 0.45 * W[1], 1e-12, 'acc[2]'); // 1·1.8·0.25 = 0.45
  approx(acc.get(3), 0.2025 * W[2], 1e-12, 'acc[3]'); // 0.45·1.8·0.25 = 0.2025
  assert.ok(!acc.has(4), '动量耗尽(0→-1),第 3 跳不得进入节点 4');
  assert.ok(!acc.has(5), '节点 5 不可达');
  assert.strictEqual(diagnostics.stateTruncations, 0);
});

// ───────────────── 4. 虫洞零动量成本豁免 ─────────────────
run('虫洞边消耗 0 动量,momentum 0 的 spike 仍可沿普通边走 1 跳', () => {
  const kernel = chainKernel(5, 1.8);
  const wormholes = new Set(['2:3']); // 仅 2→3 为虫洞
  const { accumulatedEnergy: acc } =
    propagateSpikes([{ id: 1, adjustedWeight: 1, isCore: false }], kernel, wormholes);
  // 1→2: 0.45(mom1);2→3 虫洞:0.45·1.8·0.7 = 0.567(mom 仍 1);3→4:0.567·1.8·0.25 = 0.25515(mom0)
  approx(acc.get(4), 0.25515 * W[3], 1e-9, 'acc[4]');
  assert.ok(!acc.has(5), '4→5 普通边在 mom0 时仍被拒(下一跳非虫洞)');
});

// ───────────────── 5. hop ≤ maxHops 上限 ─────────────────
run('hop ≤ maxSafeHops=4:虫洞长链在第 4 跳截停', () => {
  const kernel = chainKernel(7, 1 / 0.7);
  const wormholes = chainWormholes(7);
  const { accumulatedEnergy: acc, fieldProvenance: prov } =
    propagateSpikes([{ id: 1, adjustedWeight: 1, isCore: false }], kernel, wormholes);
  assert.strictEqual(prov.get(5).hop, 4, '节点 5 恰在第 4 跳被记 provenance');
  assert.ok(!acc.has(6), '第 5 跳不得存在');
  assert.ok(!acc.has(7), '第 6 跳不得存在');
  for (const [id, p] of prov) {
    assert.ok(p.hop <= 4, `provenance hop 超限:id=${id} hop=${p.hop}`);
  }
});

// ───────────────── 6. 状态上限截断(§1 maxPropagationStates=2000) ─────────────────
run('状态数 > 2000 按能量截断(101 种子 × 20 邻居 = 2020)', () => {
  const kernel = new Map();
  const seeds = [];
  for (let i = 1; i <= 101; i++) {
    seeds.push({ id: i, adjustedWeight: 1, isCore: false });
    const row = new Map();
    for (let j = 1; j <= 20; j++) {
      // 每种子 20 个叶子邻居,权重按种子区分(截断边界落在不同能量之间,结果确定)
      row.set(1000 + (i - 1) * 20 + j, 0.5 + i * 0.0001);
    }
    kernel.set(i, row);
  }
  const { accumulatedEnergy: acc, diagnostics } = propagateSpikes(seeds, kernel, new Set());
  assert.strictEqual(diagnostics.stateTruncations, 20, '2020 - 2000 = 20 个状态被截断');
  assert.strictEqual(acc.size, 101 + 2000, '能量场 = 101 种子 + 2000 幸存邻居');
  // 权重最低的种子 1 的 20 个叶子全被截掉
  for (let j = 1; j <= 20; j++) {
    assert.ok(!acc.has(1000 + j), `种子1 叶子 ${1000 + j} 应被截断`);
  }
  // 权重最高的种子 101 的叶子幸存
  approx(acc.get(3001), (0.5 + 101 * 0.0001) * 0.25 * W[1], 1e-9, '种子101 首个叶子能量');
});

// ───────────────── 7. 出边邻居截断 20(maxNeighborsPerNode) ─────────────────
run('出边按 P 降序截前 20(30 个邻居只达权重最高的 20 个)', () => {
  const kernel = new Map([[1, new Map()]]);
  const edgeP = (id) => 1.0 - (id - 2) * 0.01; // id=2→0.99 … id=31→0.70
  for (let i = 2; i <= 31; i++) kernel.get(1).set(i, edgeP(i));
  const { accumulatedEnergy: acc } =
    propagateSpikes([{ id: 1, adjustedWeight: 1, isCore: false }], kernel, new Set());
  assert.strictEqual(acc.size, 21, '种子 + 20 邻居');
  // 前 20 名 = id 2..21(权重 0.99..0.81);第 20 名即 id 21
  approx(acc.get(21), edgeP(21) * 0.25 * W[1], 1e-12, '第 20 名邻居(id=21,权重 0.81)');
  assert.ok(!acc.has(22), '第 21 名邻居(id=22,权重 0.80)不得被达');
});

// ───────────────── 8. firingThreshold 放电源截断 ─────────────────
run('低能量种子(adjustedWeight=0.05 < 0.1)不激发,邻居不可达', () => {
  const kernel = new Map([[1, new Map([[2, 0.9]])]]);
  const { accumulatedEnergy: acc, diagnostics } =
    propagateSpikes([{ id: 1, adjustedWeight: 0.05, isCore: false }], kernel, new Set());
  approx(acc.get(1), 0.05 * W[0], 1e-12, '只计入种子自身 w0 份额');
  assert.ok(!acc.has(2), '邻居 2 不可达');
  assert.deepStrictEqual(diagnostics.hopInFlightMass, [0], '首跳无在飞质量');
});

// ───────────────── 9. 注入下限(注入 < 0.01 丢弃) ─────────────────
run('注入 < 0.01 的质量丢弃(0.02·0.25 = 0.005)', () => {
  const kernel = new Map([[1, new Map([[2, 0.02]])]]);
  const { accumulatedEnergy: acc } =
    propagateSpikes([{ id: 1, adjustedWeight: 1, isCore: false }], kernel, new Set());
  assert.ok(!acc.has(2), '邻居 2 不得被达');
  approx(acc.get(1), W[0], 1e-12, '种子能量不受影响');
});

// ───────────────── 10. 同状态键合并:能量累加、动量取 max ─────────────────
run('合并:两 spike 同到节点 4,能量相加、动量取 max 后可再走 1 跳', () => {
  const kernel = new Map([
    [1, new Map([[3, 1.0]])], // 种子1 → 3(普通边,mom 2→1)
    [2, new Map([[3, 1.0]])], // 种子2 → 3(虫洞边,mom 2→2)
    [3, new Map([[4, 1.0]])],
    [4, new Map([[5, 1.0]])],
  ]);
  const wormholes = new Set(['2:3']);
  const seeds = [
    { id: 1, adjustedWeight: 2, isCore: false },
    { id: 2, adjustedWeight: 2, isCore: false },
  ];
  const { accumulatedEnergy: acc, fieldProvenance: prov, diagnostics } =
    propagateSpikes(seeds, kernel, wormholes);
  // 节点 3 两路并流:2·1·0.25 = 0.5(普通)+ 2·1·0.7 = 1.4(虫洞)= 1.9
  approx(acc.get(3), 1.9 * W[1], 1e-12, 'acc[3](能量合并)');
  // 同键 '3:4' 合并:0.5→4 注入 0.125(mom 0)与 1.4→4 注入 0.35(mom 1)合并 = 0.475,mom = max(0,1) = 1
  approx(acc.get(4), 0.475 * W[2], 1e-12, 'acc[4](能量相加)');
  // 若动量取 min(0) 则 4→5 被拒;取 max(1) 则可达 → 节点 5 存在即为动量取 max 的证明
  approx(acc.get(5), 0.11875 * W[3], 1e-12, 'acc[5](动量取 max 的证明)');
  assert.deepStrictEqual(prov.get(4), { sourceType: 'emergent', originType: 'seed', hop: 2 });
  assert.strictEqual(diagnostics.returnFlowSuppressedMass, 0);
  assert.strictEqual(diagnostics.stateTruncations, 0);
});

// ───────────────── 11. provenance:core/seed/emergent + 最早 hop ─────────────────
run('provenance:种子 isCore→core;emergent 记最早 hop 与 originType', () => {
  const kernel = new Map([
    [1, new Map([[5, 1.0]])], // core 种子直达 5(hop1)
    [2, new Map([[6, 1.0]])], // seed 种子 → 6(hop1)
    [6, new Map([[5, 1.0]])], // 6 → 5(hop2,更晚路径不得覆盖最早 hop)
  ]);
  const seeds = [
    { id: 1, adjustedWeight: 1, isCore: true },
    { id: 2, adjustedWeight: 1, isCore: false },
  ];
  const { accumulatedEnergy: acc, fieldProvenance: prov } =
    propagateSpikes(seeds, kernel, new Set());
  assert.deepStrictEqual(prov.get(1), { sourceType: 'core', hop: 0, seedId: 1 });
  assert.deepStrictEqual(prov.get(2), { sourceType: 'seed', hop: 0, seedId: 2 });
  assert.deepStrictEqual(prov.get(5), { sourceType: 'emergent', originType: 'core', hop: 1 });
  assert.deepStrictEqual(prov.get(6), { sourceType: 'emergent', originType: 'seed', hop: 1 });
  // 节点 5 在 hop1(core 直达 0.25)与 hop2(经 6 注入 0.0625)都被记录
  approx(acc.get(5), 0.25 * W[1] + 0.0625 * W[2], 1e-12, 'acc[5](两跳并流)');
  const noSeedId = prov.get(5);
  assert.ok(!('seedId' in noSeedId), 'emergent provenance 不带 seedId');
});

// ───────────────── 12. 空种子退化 ─────────────────
run('无种子:返回空能量场与清零诊断,不抛错', () => {
  const { accumulatedEnergy: acc, fieldProvenance: prov, diagnostics } =
    propagateSpikes([], new Map([[1, new Map([[2, 0.9]])]]), new Set());
  assert.strictEqual(acc.size, 0);
  assert.strictEqual(prov.size, 0);
  assert.strictEqual(diagnostics.stateTruncations, 0);
  assert.strictEqual(diagnostics.returnFlowSuppressedMass, 0);
  assert.strictEqual(diagnostics.algorithmVersion, 'v9.1-soft-nonbacktracking-fir');
});

// ───────────────── 汇总 ─────────────────
console.log(`\npropagate.test.mjs:通过 ${passed} 项,失败 ${failed} 项`);
process.exit(failed ? 1 : 0);