// tests/pyramid.test.mjs — ResidualPyramid 移植自测（node tests/pyramid.test.mjs 直接运行，零依赖）
// 覆盖 SPEC-P1.md §2 验收：
//   构造 1024 维合成 tag 集（tagA=e1, tagB=e2, 若干随机噪声 tag），查询 = normalize(0.8e1+0.6e2)；
//   searchTags 用暴力余弦自实现（就在本文件内）；
//   第 1 层 top1 = tagA；残差递归后第 2 层 top1 = tagB；totalExplainedEnergy > 0.7；
//   残差能量 < minEnergyRatio 时正确截断（层数 < maxLevels）；
//   features 五字段（depth/coverage/novelty/coherence/tagMemoActivation）齐全且为有限数；
//   空 tag 集不抛异常（返回空 levels）。
// 断言失败即抛错，退出码非零。
import assert from 'node:assert/strict'
import { ResidualPyramid } from '../core/ResidualPyramid.mjs'

const DIM = 1024

let passed = 0
async function ok(name, fn) {
  await fn()
  passed++
  console.log(`  ✓ ${name}`)
}

async function main() {
  console.log('pyramid.test.mjs')

  // ── 工具：确定性随机数（mulberry32，保证测试可复现）──
  function mulberry32(seed) {
    let a = seed >>> 0
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0
      let t = Math.imul(a ^ (a >>> 15), 1 | a)
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }
  }

  // 单位向量：1024 维标准正态分量（Box-Muller）后归一化
  function randomUnitVector(rng) {
    const v = new Float32Array(DIM)
    let sq = 0
    for (let i = 0; i < DIM; i += 2) {
      const u1 = Math.max(rng(), 1e-12)
      const u2 = rng()
      const r = Math.sqrt(-2 * Math.log(u1))
      v[i] = r * Math.cos(2 * Math.PI * u2)
      v[i + 1] = r * Math.sin(2 * Math.PI * u2)
      sq += v[i] * v[i] + v[i + 1] * v[i + 1]
    }
    const norm = Math.sqrt(sq)
    for (let i = 0; i < DIM; i++) v[i] /= norm
    return v
  }

  function unitAxis(axis) {
    const v = new Float32Array(DIM)
    v[axis] = 1
    return v
  }

  function normalize(v) {
    let s = 0
    for (let i = 0; i < v.length; i++) s += v[i] * v[i]
    const n = Math.sqrt(s)
    const out = new Float32Array(v.length)
    for (let i = 0; i < v.length; i++) out[i] = v[i] / n
    return out
  }

  // 暴力余弦
  function cosine(a, b) {
    let dot = 0
    let na = 0
    let nb = 0
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i]
      na += a[i] * a[i]
      nb += b[i] * b[i]
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb))
  }

  // searchTags 工厂：对 tags 全体做暴力余弦，按相似度降序取 topK
  function makeSearchTags(tags) {
    return async (residualVector, topK) => {
      const scored = tags.map(t => ({
        id: t.id,
        name: t.name,
        vector: t.vector,
        similarity: cosine(t.vector, residualVector),
      }))
      scored.sort((a, b) => b.similarity - a.similarity)
      return scored.slice(0, topK)
    }
  }

  // ── 合成 tag 集：tagA=e1 / tagB=e2 + 16 个随机噪声 tag ──
  const tagA = { id: 1, name: 'tagA', vector: unitAxis(0) }
  const tagB = { id: 2, name: 'tagB', vector: unitAxis(1) }
  const rng = mulberry32(20260829)
  const noiseTags = Array.from({ length: 16 }, (_, i) => ({
    id: 3 + i,
    name: `noise${i}`,
    vector: randomUnitVector(rng),
  }))
  const tags = [tagA, tagB, ...noiseTags]
  const searchTags = makeSearchTags(tags)

  // 查询向量 = normalize(0.8*e1 + 0.6*e2)（范数恰为 1，归一化仅为显式）
  const query = normalize((() => {
    const v = new Float32Array(DIM)
    v[0] = 0.8
    v[1] = 0.6
    return v
  })())

  // ── 主场景：残差递归逐层感应 ──
  // 说明（偏离默认 topK=10 的原因，见汇报"偏离规格的决策"）：
  //   q 同时拥有 e1(0.8) 与 e2(0.6) 分量。若 topK 足够大把 tagB 也召回进首层，
  //   首层 Gram-Schmidt 投影会把 e1+e2 一并解释掉，残差趋零、能量截断即停，数学上
  //   不可能出现"第 2 层 top1 = tagB"。故此处 topK=1：首层只解释 e1 方向，
  //   残差保留 e2 方向，第 2 层才能感应出 tagB（这正是残差金字塔"逐层正交解释"的本意）。
  await ok('主场景：首层搜索 top1=tagA（相似度≈0.8）', async () => {
    const top1 = await searchTags(query, 1)
    assert.strictEqual(top1[0].name, 'tagA')
    assert.ok(Math.abs(top1[0].similarity - 0.8) < 1e-6, `similarity=${top1[0].similarity}`)
  })

  await ok('主场景：analyze 产生 2 层，首层 top1=tagA、次层 top1=tagB', async () => {
    const pyramid = new ResidualPyramid(searchTags, {
      dimension: DIM, topK: 1, maxLevels: 3, minEnergyRatio: 0.1,
    })
    const result = await pyramid.analyze(query)

    // 层结构：恰好 2 层
    assert.strictEqual(result.levels.length, 2, '应产生 2 层')
    assert.strictEqual(result.levels[0].level, 0)
    assert.strictEqual(result.levels[1].level, 1)

    // 第 1 层 top1 = tagA（相似度 ≈ 0.8，残差与 e1 的点积）
    assert.strictEqual(result.levels[0].tags[0].name, 'tagA')
    assert.ok(Math.abs(result.levels[0].tags[0].similarity - 0.8) < 1e-6)
    // 贡献度 = |<q, u0>| ≈ 0.8
    assert.ok(Math.abs(result.levels[0].tags[0].contribution - 0.8) < 1e-4)

    // 残差递归后第 2 层 top1 = tagB（残差方向 ≈ e2，与 tagB 余弦 ≈ 1）
    assert.strictEqual(result.levels[1].tags[0].name, 'tagB')
    assert.ok(result.levels[1].tags[0].similarity > 0.99, `similarity=${result.levels[1].tags[0].similarity}`)

    // 能量：第 1 层解释 e1 分量 (1-0.36≈0.64)，第 2 层解释 e2 分量 (≈0.36)
    assert.ok(result.levels[0].energyExplained > 0.6 && result.levels[0].energyExplained < 0.7,
      `level0.energyExplained=${result.levels[0].energyExplained}`)
    assert.ok(result.levels[1].energyExplained > 0.3 && result.levels[1].energyExplained < 0.4,
      `level1.energyExplained=${result.levels[1].energyExplained}`)
    assert.ok(result.totalExplainedEnergy > 0.7, `totalExplainedEnergy=${result.totalExplainedEnergy}`)

    // 残差能量逐层下降，且第 0 层残差占比 ≈ 0.36
    assert.ok(Math.abs(result.levels[0].residualEnergyRatio - 0.36) < 1e-3,
      `level0.residualEnergyRatio=${result.levels[0].residualEnergyRatio}`)
    assert.ok(result.levels[1].residualEnergyRatio < result.levels[0].residualEnergyRatio)

    // finalResidual 是合法的 Float32Array
    assert.ok(result.finalResidual instanceof Float32Array)
    assert.strictEqual(result.finalResidual.length, DIM)
  })

  await ok('主场景：残差能量 < minEnergyRatio 时正确截断（层数 < maxLevels）', async () => {
    const result = await new ResidualPyramid(searchTags, {
      dimension: DIM, topK: 1, maxLevels: 3, minEnergyRatio: 0.1,
    }).analyze(query)

    // 截断由能量阈值触发（末层残差占比 < 0.1），而非撞上 maxLevels=3
    assert.ok(result.levels.length < 3, `levels=${result.levels.length}`)
    assert.ok(result.levels[result.levels.length - 1].residualEnergyRatio < 0.1,
      `末层 residualEnergyRatio=${result.levels[result.levels.length - 1].residualEnergyRatio}`)

    // 放大量程（maxLevels=8）仍只产生 2 层：证明是能量截断而非层数上限
    const wide = await new ResidualPyramid(searchTags, {
      dimension: DIM, topK: 1, maxLevels: 8, minEnergyRatio: 0.1,
    }).analyze(query)
    assert.strictEqual(wide.levels.length, 2, 'maxLevels=8 时仍应因能量截断停在 2 层')
    assert.ok(wide.levels[1].residualEnergyRatio < 0.1)
  })

  await ok('主场景：features 五字段齐全且为有限数', async () => {
    const result = await new ResidualPyramid(searchTags, {
      dimension: DIM, topK: 1, maxLevels: 3, minEnergyRatio: 0.1,
    }).analyze(query)
    const f = result.features
    for (const key of ['depth', 'coverage', 'novelty', 'coherence', 'tagMemoActivation']) {
      assert.ok(Object.prototype.hasOwnProperty.call(f, key), `features 缺少字段 ${key}`)
      assert.ok(Number.isFinite(f[key]), `features.${key} 应为有限数: ${f[key]}`)
    }
    // 规格 §2 接口还包含扩展信号 expansionSignal（非空结果时为有限数）
    assert.ok(Object.prototype.hasOwnProperty.call(f, 'expansionSignal'))
    assert.ok(Number.isFinite(f.expansionSignal))
    assert.strictEqual(f.depth, 2)
    // 次层残差 ≈ 0 → coverage ≈ 1 → novelty 只剩方向分量 0.3
    assert.ok(f.coverage > 0.7 && f.coverage <= 1.0, `coverage=${f.coverage}`)
    assert.ok(f.novelty > 0 && f.novelty < 1, `novelty=${f.novelty}`)
    // 首层仅召回 1 个 tag → 无两两方向可比 → coherence=0 → tagMemoActivation=0
    assert.strictEqual(f.coherence, 0)
    assert.strictEqual(f.tagMemoActivation, 0)
  })

  // ── 截断场景 2：查询恰为 tagA → 首层全解释、单层即截断 ──
  await ok('查询恰为 tagA：单层全解释并按能量截断（1 层 < maxLevels=5）', async () => {
    const result = await new ResidualPyramid(searchTags, {
      dimension: DIM, topK: 1, maxLevels: 5, minEnergyRatio: 0.1,
    }).analyze(tagA.vector)
    assert.strictEqual(result.levels.length, 1)
    assert.strictEqual(result.levels[0].tags[0].name, 'tagA')
    assert.ok(result.totalExplainedEnergy > 0.99, `totalExplainedEnergy=${result.totalExplainedEnergy}`)
    assert.ok(result.levels[0].residualEnergyRatio < 0.1,
      `residualEnergyRatio=${result.levels[0].residualEnergyRatio}`)
  })

  // ── 默认 topK=10 保真：e1+e2 首层一并解释 → 单层、总能量 ≈ 1 ──
  // 与主场景同数据、同查询，仅 topK 不同。验证移植的投影/能量数学在两个"模式"下都正确。
  await ok('默认 topK=10：首层将 e1+e2 一并解释（单层，能量≈1，保真对照）', async () => {
    const result = await new ResidualPyramid(searchTags, {
      dimension: DIM, topK: 10, maxLevels: 3, minEnergyRatio: 0.1,
    }).analyze(query)
    assert.strictEqual(result.levels.length, 1, `levels=${result.levels.length}`)
    assert.strictEqual(result.levels[0].tags[0].name, 'tagA')
    assert.ok(result.totalExplainedEnergy > 0.99, `totalExplainedEnergy=${result.totalExplainedEnergy}`)
    assert.ok(result.levels[0].residualEnergyRatio < 0.1,
      `residualEnergyRatio=${result.levels[0].residualEnergyRatio}`)
  })

  // ── 空 tag 集：优雅降级，不抛异常 ──
  await ok('空 tag 集：返回空 levels，不抛异常', async () => {
    const emptySearch = async () => []
    const result = await new ResidualPyramid(emptySearch, {
      dimension: DIM, maxLevels: 3, minEnergyRatio: 0.1,
    }).analyze(query)
    assert.strictEqual(result.levels.length, 0)
    assert.strictEqual(result.totalExplainedEnergy, 0)
    assert.ok(result.finalResidual instanceof Float32Array)
    assert.strictEqual(result.finalResidual.length, DIM)
    for (const key of ['depth', 'coverage', 'novelty', 'coherence', 'tagMemoActivation']) {
      assert.ok(Object.prototype.hasOwnProperty.call(result.features, key))
      assert.ok(Number.isFinite(result.features[key]), `features.${key}=${result.features[key]}`)
    }
    assert.strictEqual(result.features.depth, 0)
    assert.strictEqual(result.features.novelty, 1)
  })

  // ── 零向量查询：_emptyResult 处理，不抛异常 ──
  await ok('零向量查询：返回空 levels（防除零）', async () => {
    const result = await new ResidualPyramid(searchTags, {
      dimension: DIM, topK: 1,
    }).analyze(new Float32Array(DIM))
    assert.strictEqual(result.levels.length, 0)
    assert.strictEqual(result.totalExplainedEnergy, 0)
  })

  // ── 普通数组入参：内部转 Float32Array ──
  await ok('普通数组查询向量：自动转换后结果一致', async () => {
    const result = await new ResidualPyramid(searchTags, {
      dimension: DIM, topK: 1,
    }).analyze(Array.from(query))
    assert.strictEqual(result.levels[0].tags[0].name, 'tagA')
  })

  // ── searchTags 中途抛错：记日志并降级为空 levels，不抛给调用方 ──
  await ok('searchTags 抛错：优雅降级为空 levels', async () => {
    const failingSearch = async () => {
      throw new Error('search boom')
    }
    const result = await new ResidualPyramid(failingSearch, {
      dimension: DIM, topK: 1,
    }).analyze(query)
    assert.strictEqual(result.levels.length, 0)
    assert.strictEqual(result.totalExplainedEnergy, 0)
    assert.strictEqual(result.features.depth, 0)
  })

  console.log(`\npyramid.test.mjs: ${passed} 项断言组全部通过`)
}

main().catch(err => {
  console.error(`\n✗ pyramid.test.mjs 失败: ${err.message}`)
  process.exitCode = 1
})