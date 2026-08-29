// tests/taggraph.test.mjs — SPEC-P2 §8.2 第 2 条验收(零框架,node 直接可跑)
//
// 覆盖:taggraph.mjs 的 §2.1 有向共现矩阵(φ 边界 / 距离衰减 / 顺逆流不对称 /
// 反转守卫 / 钟形增益分段 / 跨文件累加,手算对照)与 §2.2 V9.1 传播核
// (行出流和 ≤0.95 且多源节点 / 高入流枢纽 penalty 方向 / 虫洞边进 Set /
// median 平滑),以及 §2.3 图生命周期(门槛、写时替换、指纹变化重建、代数)。
// 断言失败即抛错,退出码非零。
import assert from 'node:assert/strict'
import {
  phiPotential, positionDistanceFactor, semanticGain, pairwiseCosine,
  buildFactMatrix, buildV9Kernel, buildGraph, createTagGraph,
} from '../engine/taggraph.mjs'

let passed = 0
function ok(name, fn) {
  fn()
  passed++
  console.log(`  ✓ ${name}`)
}
// 异步断言组(生命周期测试用;失败同样抛错、退出码非零)
async function okAsync(name, fn) {
  try {
    await fn()
    passed++
    console.log(`  ✓ ${name}`)
  } catch (err) {
    console.error(`  ✗ ${name}`)
    throw err
  }
}

// 近似断言:允许 float 误差 eps(默认 1e-9)
function approx(actual, expected, eps = 1e-9, msg = '') {
  assert.ok(Number.isFinite(actual), `${msg} 实际值非有限: ${actual}`)
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `${msg} 期望≈${expected} 实际=${actual} 偏差=${Math.abs(actual - expected)} > ${eps}`,
  )
}

// 构造 tag 记录:vector 可空(无向量走 lowSimFallback),occurrences [{file, position}]
const T = (name, vector, occ) => ({ name, vector: vector ?? null, occurrences: occ })
// 向 fact 加一条(src→dst,w)边(测试用 Map 直接构造,与 buildFactMatrix 的产物等价)
function addEdge(fact, src, dst, w) {
  if (!fact.has(src)) fact.set(src, new Map())
  fact.get(src).set(dst, w)
}

console.log('taggraph.test.mjs')

// ─────────────────────────────────────────────────────────────────────────────
// §8.2-1:φ 序位势能边界(pos=1 → 0.9,pos=n → 0.5)
// φ(pos) = 0.9 − 0.4·(pos−1)/(n−1);n≤1 时无对,恒 0.9
// ─────────────────────────────────────────────────────────────────────────────
console.log('—— §2.1 φ 序位势能边界 ——')
ok('φ(1)=0.9 与 φ(n)=0.5 边界', () => {
  assert.strictEqual(phiPotential(1, 5), 0.9)
  assert.strictEqual(phiPotential(5, 5), 0.5)
  assert.strictEqual(phiPotential(1, 1), 0.9) // n=1 无对,参考实现返回 PHI_MAX
})
ok('中间点位与单调递减', () => {
  assert.strictEqual(phiPotential(2, 5), 0.8) // 0.9 − 0.4·1/4
  assert.strictEqual(phiPotential(3, 5), 0.7) // 0.9 − 0.4·2/4
  assert.strictEqual(phiPotential(4, 5), 0.6) // 0.9 − 0.4·3/4
  for (let p = 1; p < 5; p++) {
    assert.ok(phiPotential(p, 5) > phiPotential(p + 1, 5), 'φ 应随 position 单调递减')
  }
})

// ─────────────────────────────────────────────────────────────────────────────
// §8.2-2:距离衰减 exp(−0.08·(|Δpos|−1)),相邻衰减为 1
// ─────────────────────────────────────────────────────────────────────────────
console.log('—— §2.1 距离衰减 ——')
ok('距离衰减公式(纯函数)', () => {
  assert.strictEqual(positionDistanceFactor(1, 0.08), 1) // 相邻:exp(0)
  approx(positionDistanceFactor(2, 0.08), Math.exp(-0.08))
  approx(positionDistanceFactor(3, 0.08), Math.exp(-0.16))
  approx(positionDistanceFactor(5, 0.08), Math.exp(-0.32)) // exp(−0.08·4)
  assert.strictEqual(positionDistanceFactor(2, 0), 1) // decay≤0 → 无衰减
  assert.strictEqual(positionDistanceFactor(2, -1), 1)
  assert.strictEqual(positionDistanceFactor(0, 0.08), 1) // delta<1 夹逼到 1
})
ok('跨距离权重衰减(A@1,B@2,C@3,全无向量 → sim=fallback 0.1 → semGain=0.5)', () => {
  const tags = [
    T('A', null, [{ file: 'f1', position: 1 }]),
    T('B', null, [{ file: 'f1', position: 2 }]),
    T('C', null, [{ file: 'f1', position: 3 }]),
  ]
  const { fact } = buildFactMatrix(tags)
  // 手算:A→B base=φ1·φ2=0.9·0.7=0.63 → w=0.63·1.0·0.5=0.315
  approx(fact.get(0).get(1), 0.315, 1e-12, 'A→B')
  // A→C base=0.9·0.5·exp(−0.08)=0.4154023… → w=base·1.0·0.5=0.20770117…
  approx(fact.get(0).get(2), 0.45 * Math.exp(-0.08) * 0.5, 1e-12, 'A→C')
  assert.ok(fact.get(0).get(2) < fact.get(0).get(1), '远距离对权重应小于相邻对')
})

// ─────────────────────────────────────────────────────────────────────────────
// §8.2-3:钟形增益分段(sim<0.15 → 0.4+sim;否则 0.5+0.8·exp(−(sim−0.65)²/(2·0.25²)))
// 三点验收:0.1→0.5、0.65→1.3、0.95→衰减
// ─────────────────────────────────────────────────────────────────────────────
console.log('—— §2.1 钟形语义增益 ——')
ok('钟形增益三点(0.1/0.65/0.95)', () => {
  approx(semanticGain(0.1), 0.5, 1e-12) // 0.4+0.1,软底分支
  approx(semanticGain(0.65), 1.3, 1e-12) // 0.5+0.8·exp(0),峰点
  const at095 = semanticGain(0.95)
  approx(at095, 0.5 + 0.8 * Math.exp(-((0.95 - 0.65) ** 2) / (2 * 0.25 ** 2)), 1e-12)
  assert.ok(at095 < semanticGain(0.65), 'sim=0.95 应衰减(高 sim 抑制)')
  approx(semanticGain(0.05), 0.45, 1e-12) // 软底 0.40~0.55 区间
})
ok('禁用语义增益 → 恒 1.0;无效 sim → 1.0', () => {
  assert.strictEqual(semanticGain(0.9, { enabled: false }), 1.0)
  assert.strictEqual(semanticGain(Number.NaN), 1.0)
})
ok('零/负相似度走 fallback(对照 getSimSafe:非有限或 ≤0 → 0.1)', () => {
  // 正交向量 sim=0 → 兜底 0.1 → semGain(0.1)=0.5 → 与无向量情形同权
  assert.strictEqual(pairwiseCosine([1, 0], [1, 0]), 1)
  approx(pairwiseCosine([1, 0], [0, 1]), 0, 1e-12)
  assert.strictEqual(pairwiseCosine(null, [1, 0]), null) // 无向量 → null
  assert.strictEqual(pairwiseCosine([], []), null) // 零向量无方向
  const tags = [
    T('A', [1, 0], [{ file: 'f1', position: 1 }]),
    T('B', [0, 1], [{ file: 'f1', position: 2 }]),
  ]
  const { fact } = buildFactMatrix(tags)
  approx(fact.get(0).get(1), 0.225, 1e-12, '正交向量走 fallback 后与无向量同权')
})

// ─────────────────────────────────────────────────────────────────────────────
// §8.2-4:顺逆流不对称(顺 > 逆,默认 reverseGain=0.35 夹逼后不触发反转守卫)
// ─────────────────────────────────────────────────────────────────────────────
console.log('—— §2.1 顺逆流不对称 ——')
ok('顺流 > 逆流(同对,同 base/semGain,只差 forward/reverse 增益)', () => {
  const tags = [
    T('A', null, [{ file: 'f1', position: 1 }]),
    T('B', null, [{ file: 'f1', position: 2 }]),
  ]
  const { fact, diagnostics } = buildFactMatrix(tags)
  // 手算:base=0.9·0.5=0.45;semGain(0.1)=0.5
  // 顺流 0→1 = 0.45·1.0·0.5 = 0.225;逆流 1→0 = 0.45·0.35·0.5 = 0.07875
  approx(fact.get(0).get(1), 0.225, 1e-12, '顺流 A→B')
  approx(fact.get(1).get(0), 0.07875, 1e-12, '逆流 B→A')
  assert.ok(fact.get(0).get(1) > fact.get(1).get(0), '顺流应大于逆流')
  // 比值 = forwardGain/reverseGain = 1.0/0.35(同 base/semGain 抵消)
  approx(fact.get(0).get(1) / fact.get(1).get(0), 1 / 0.35, 1e-9, '顺逆比')
  assert.strictEqual(diagnostics.inversionClamped, 0, '默认参数下守卫不应截断')
  assert.strictEqual(diagnostics.files, 1)
  assert.strictEqual(diagnostics.pairs, 1) // n(n−1)/2 = 1
})

// ─────────────────────────────────────────────────────────────────────────────
// §8.2-5:反转守卫截断(逆流 ≤ 顺流·0.9)
// ─────────────────────────────────────────────────────────────────────────────
console.log('—— §2.1 反转守卫 ——')
ok('raw 逆流超过顺流·0.9 时被截断', () => {
  const tags = [
    T('A', null, [{ file: 'f1', position: 1 }]),
    T('B', null, [{ file: 'f1', position: 2 }]),
  ]
  // reverseGain 2.0(夹逼区间放宽到 maxReverseGain=2)→ raw 逆流 = 0.45·2·0.5 = 0.45
  const { fact, diagnostics } = buildFactMatrix(tags, {
    cooccurrence: { reverseGain: 2, maxReverseGain: 2 },
  })
  approx(fact.get(0).get(1), 0.225, 1e-12, '顺流不变')
  approx(fact.get(1).get(0), 0.225 * 0.9, 1e-12, '逆流被截断到顺流×0.9=0.2025')
  assert.strictEqual(diagnostics.inversionClamped, 1, '应记录 1 次截断')
})

// ─────────────────────────────────────────────────────────────────────────────
// §8.2-6:跨文件累加(同边多次共现求和;方向各自独立累加,手算对照)
// ─────────────────────────────────────────────────────────────────────────────
console.log('—— §2.1 跨文件累加 ——')
ok('同序两文件 + 逆序一文件的同边求和', () => {
  // 单文件手算(无向量):A@1,B@2 → 顺流 A→B=0.225;逆流 B→A=0.07875
  const tags = [
    T('A', null, [
      { file: 'f1', position: 1 },
      { file: 'f2', position: 1 },
      { file: 'f3', position: 2 }, // f3 中 A 在后(叙事方向反转)
    ]),
    T('B', null, [
      { file: 'f1', position: 2 },
      { file: 'f2', position: 2 },
      { file: 'f3', position: 1 },
    ]),
  ]
  const { fact } = buildFactMatrix(tags)
  // f1/f2:0→1 累加 0.225×2=0.45;1→0 累加 0.07875×2=0.1575
  // f3(B@1,A@2):顺流 1→0 再 +0.225;逆流 0→1 再 +0.07875
  // 合计:0→1 = 0.45+0.07875 = 0.52875;1→0 = 0.1575+0.225 = 0.3825
  approx(fact.get(0).get(1), 0.52875, 1e-12, 'A→B 跨文件累加')
  approx(fact.get(1).get(0), 0.3825, 1e-12, 'B→A 跨文件累加')
  assert.ok(fact.get(0).get(1) > 0.225, 'A→B 应大于单文件值(累加生效)')
  assert.ok(fact.get(1).get(0) > 0.07875, 'B→A 应大于单文件值(累加生效)')
})

// ─────────────────────────────────────────────────────────────────────────────
// §8.2-7:传播核——行出流和 ≤ 0.95(多个源节点;含虫洞行 0.95,无虫洞行 0.9)
// ─────────────────────────────────────────────────────────────────────────────
console.log('—— §2.2 传播核:行出流预算 ——')
function hubFact() {
  // 目标 1 为"高入流枢纽":被 10 个源各注入 w=3.0
  // 目标 2/3/4/5 为普通目标;源 0 同时指向枢纽 1 与普通目标 2
  const fact = new Map()
  addEdge(fact, 0, 1, 2.0)
  addEdge(fact, 0, 2, 1.0)
  addEdge(fact, 1, 3, 1.0)
  addEdge(fact, 1, 4, 1.0)
  addEdge(fact, 2, 3, 1.0)
  addEdge(fact, 2, 5, 1.0)
  for (let s = 3; s <= 12; s++) addEdge(fact, s, 1, 3.0)
  return fact
}
ok('每源节点行出流和 = outboundMass(0.95),多个源节点', () => {
  const { kernel, wormholeEdges } = buildV9Kernel(hubFact())
  assert.ok(kernel.size >= 5, `应有 ≥5 个源节点行,实际 ${kernel.size}`)
  for (const [src, row] of kernel.entries()) {
    let sum = 0
    for (const w of row.values()) sum += w
    assert.ok(sum <= 0.95 + 1e-9, `源 ${src} 行出流 ${sum} 超过预算 0.95`)
    // 参考实现:mainMass = outboundMass − reserveMass;无虫洞行 reserve=0 → mainMass=0.95,
    // 虫洞行 main=0.9 + reserve 0.05(仅虫洞边) → 每行总出流恒 ≈ outboundMass
    approx(sum, 0.95, 1e-9, `源 ${src} 行总出流`)
  }
  assert.ok(wormholeEdges instanceof Set)
})

// ─────────────────────────────────────────────────────────────────────────────
// §8.2-8:枢纽校正方向(高入流被压、低入流相对增益)
// ─────────────────────────────────────────────────────────────────────────────
console.log('—— §2.2 传播核:枢纽校正方向 ——')
ok('高入流枢纽被压、低入流获得相对增益(对照 hubPenaltyExponent=0)', () => {
  const withPen = buildV9Kernel(hubFact()).kernel
  const noPen = buildV9Kernel(hubFact(), null, { hubPenaltyExponent: 0 }).kernel
  const wPen = withPen.get(0).get(1) // 枢纽目标(A 边)
  const lPen = withPen.get(0).get(2) // 普通目标(L 边)
  const wNo = noPen.get(0).get(1)
  const lNo = noPen.get(0).get(2)
  assert.ok(wPen < wNo, `高入流目标应被压:有惩罚 ${wPen} ≥ 无惩罚 ${wNo}`)
  assert.ok(lPen > lNo, `低入流目标应相对增益:有惩罚 ${lPen} ≤ 无惩罚 ${lNo}`)
  assert.ok(wPen / lPen < wNo / lNo, '枢纽/普通 权重比应因惩罚下降')
  // 极值方向:极端枢纽 penalty 触及下限 0.55(relative ≈ 26.5 → 0.374 → 夹到 0.55)
  // 无法直接观测内部分量,用"被压"方向 + 上述断言覆盖即可
})

// ─────────────────────────────────────────────────────────────────────────────
// §8.2-9:虫洞边进 Set + residualMap 接口保留 + median 平滑
// ─────────────────────────────────────────────────────────────────────────────
console.log('—— §2.2 虫洞判定与 median 平滑 ——')
ok('强累计边 e≥1 判定虫洞并写入 Set;弱边不入', () => {
  const fact = new Map()
  addEdge(fact, 0, 1, 2.0) // e=log1p(2)=1.0986 ≥ 1 → 虫洞
  addEdge(fact, 0, 2, 0.5) // e=log1p(0.5)=0.4055 < 1 → 非虫洞
  const { wormholeEdges } = buildV9Kernel(fact)
  assert.ok(wormholeEdges.has('0:1'), '强累计边应进 wormholeEdges')
  assert.ok(!wormholeEdges.has('0:2'), '弱边不应进入 Set')
})
ok('residualMap 接口保留:e·residual ≥ 阈才判虫洞', () => {
  const fact = new Map()
  addEdge(fact, 0, 7, 0.5) // 单独不够;residual 抬升后触发
  const plain = buildV9Kernel(fact)
  assert.ok(!plain.wormholeEdges.has('0:7'))
  const boosted = buildV9Kernel(fact, new Map([[7, 2.5]])) // 0.4055·2.5=1.0137 ≥ 1
  assert.ok(boosted.wormholeEdges.has('0:7'), 'residual≥2.5 时应判虫洞')
})
ok('median 平滑:中位数取值 + smoothing=median·0.1 + 1e-9 下限', () => {
  const { diagnostics } = buildV9Kernel(hubFact())
  // 正入流排序后取中间元素:5 个目标 → index 2 = log1p(1)=0.693147…
  approx(diagnostics.medianInflow, Math.log1p(1), 1e-12, 'medianInflow')
  approx(diagnostics.smoothing, Math.max(1e-9, Math.log1p(1) * 0.1), 1e-12, 'smoothing')
  assert.strictEqual(diagnostics.targetCount, 5)
  assert.strictEqual(diagnostics.positiveInflowCount, 5)
  // smoothing 下限:极小入流 w=1e-9 → median≈1e-9 → median·0.1 < 1e-9 → 取 1e-9
  const tiny = new Map()
  addEdge(tiny, 0, 1, 1e-9)
  approx(buildV9Kernel(tiny).diagnostics.smoothing, 1e-9, 1e-12, 'smoothing 下限')
})

// ─────────────────────────────────────────────────────────────────────────────
// §8.2-10:buildGraph 全链路装配 + §2.3 门槛(<2 有向量 tag → 同构空图)
// ─────────────────────────────────────────────────────────────────────────────
console.log('—— §2.3 buildGraph 全链路与门槛 ——')
ok('≥2 有向量 tag 的全量构建(fact/kernel 装配 + 手算对照)', () => {
  const v = [1, 0, 0] // 相同向量 → 余弦恒 1
  const tags = [
    T('A', v, [{ file: 'f1', position: 1 }]),
    T('B', v, [
      { file: 'f1', position: 2 },
      { file: 'f2', position: 1 },
    ]),
    T('C', v, [{ file: 'f2', position: 2 }]),
  ]
  const g = buildGraph(tags)
  const sem1 = semanticGain(1.0) // 钟形峰后衰减值(与实现同式)
  approx(g.fact.get(0).get(1), 0.45 * sem1, 1e-12, 'A→B 顺流')
  approx(g.fact.get(1).get(0), 0.45 * 0.35 * sem1, 1e-12, 'B→A 逆流')
  approx(g.fact.get(1).get(2), 0.45 * sem1, 1e-12, 'B→C 顺流(第二个文件)')
  assert.strictEqual(g.fact.size, 3) // A/B/C 均有出边
  assert.strictEqual(g.kernel.size, 3)
  assert.ok(g.wormholeEdges instanceof Set)
  assert.strictEqual(g.generation, 1)
  assert.strictEqual(g.diagnostics.sufficientVectorizedTags, true)
  assert.strictEqual(g.diagnostics.skippedReason, null)
  assert.strictEqual(g.diagnostics.vectorizedTags, 3)
  for (const row of g.kernel.values()) {
    let sum = 0
    for (const w of row.values()) sum += w
    assert.ok(sum <= 0.95 + 1e-9, '全链路行出流 ≤ 0.95')
  }
})
ok('<2 个有向量 tag → 同构空图(门槛,回退 P1 的依据)', () => {
  const g = buildGraph([
    T('A', [1, 0, 0], [{ file: 'f1', position: 1 }]), // 唯一有向量
    T('B', null, [{ file: 'f1', position: 2 }]),
  ])
  assert.strictEqual(g.fact.size, 0)
  assert.strictEqual(g.kernel.size, 0)
  assert.strictEqual(g.wormholeEdges.size, 0)
  assert.strictEqual(g.diagnostics.sufficientVectorizedTags, false)
  assert.strictEqual(g.diagnostics.skippedReason, 'insufficient-vectorized-tags')
})

// ─────────────────────────────────────────────────────────────────────────────
// §8.2-11:图生命周期(写时替换/指纹变化重建/代数递增/并发串行)
// ─────────────────────────────────────────────────────────────────────────────
console.log('—— §2.3 图生命周期 ——')
await okAsync('启动重建 → generation=1;内容未变跳过;变化重建 → 代数递增且旧代保留', async () => {
  const v = [1, 0, 0]
  const tagsGood = () => [
    T('A', v, [{ file: 'f1', position: 1 }]),
    T('B', v, [{ file: 'f1', position: 2 }]),
    T('C', v, [{ file: 'f2', position: 1 }]),
  ]
  const g = createTagGraph()
  assert.strictEqual(g.generation, 0)
  assert.strictEqual(g.snapshot.fact.size, 0)

  await g.rebuild(tagsGood())
  assert.strictEqual(g.generation, 1)
  assert.ok(g.snapshot.fact.size >= 1, '启动构建后应有 fact')

  // 内容指纹未变 → 跳过重建(代数不增,发布对象不变)
  const oldSnapshot = g.snapshot
  await g.rebuildIfChanged(tagsGood())
  assert.strictEqual(g.generation, 1, '指纹未变不应代数递增')
  assert.strictEqual(g.snapshot, oldSnapshot, '跳过重建时不应替换发布对象')

  // occurrences 变化 → 全量重建 → 代数 +1;旧代对象原样保留(写时替换)
  const tagsChanged = tagsGood().map((t, i) =>
    i === 1 ? T('B', v, [{ file: 'f1', position: 2 }, { file: 'f3', position: 1 }]) : t
  )
  await g.rebuildIfChanged(tagsChanged)
  assert.strictEqual(g.generation, 2)
  assert.strictEqual(oldSnapshot.generation, 1, '旧代快照不应被改写(写时替换)')

  // 并发诉求排队串行:两次无条件重建 → 代数 +2
  await Promise.all([g.rebuild(tagsGood()), g.rebuild(tagsChanged)])
  assert.strictEqual(g.generation, 4, '两次重建应各代数 +1')

  // 空图发布同样计数(门槛不满足也发布整代,供 store 判不可用)
  await g.rebuildIfChanged([T('A', [1, 0, 0], [{ file: 'f1', position: 1 }])])
  assert.strictEqual(g.generation, 5)
  assert.strictEqual(g.snapshot.diagnostics.skippedReason, 'insufficient-vectorized-tags')
})

console.log(`\ntaggraph.test.mjs: ${passed} 项断言组全部通过`)