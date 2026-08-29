// tests/port-equivalence.test.mjs —— 移植保真对照测试(SPEC-P1.md §5,零依赖)
// 运行:node tests/port-equivalence.test.mjs(无框架,直接可跑)
//
// 目的:证明 core/EPAModule.mjs 与 core/ResidualPyramid.mjs 两个移植文件"搬数学没有搬错"。
// 方法:
//  - 用 node:module 的 createRequire 直接加载原版 CommonJS 文件:
//      /home/lyy/workspace/vcp-src/VCPToolBox-main/EPAModule.js(741 行)
//      /home/lyy/workspace/vcp-src/VCPToolBox-main/ResidualPyramid.js(394 行)
//  - 原版 EPAModule 喂 better-sqlite3 风格假 db(prepare(sql) → { all/get/run }),
//    向量是 Buffer(Float32 小端,4096 字节);移植版喂 { listTagVectors } + 同步 KV cache;
//  - 原版 ResidualPyramid 喂假 tagIndex(search 返回 [{id, score, similarity}])
//    + 假 db(按 id 取 tag 名与 Buffer 向量);移植版喂 searchTags 回调;
//  - 同一合成数据集(≥16 个有聚类结构的 1024 维向量 + 一个查询向量);
//  - 两处 Math.random(Forgy 质心选取 / 幂迭代起始向量)按规格保留,测试用带状态的种子
//    随机源替换全局 Math.random,并保证两版吃到的随机序列完全相同;
//  - 对照断言(§5):
//      EPA:两版 initialize 均成功;project 的 entropy/logicDepth 差异 < 1e-3;
//           dominantAxes 数量一致;probabilities 和为 1、熵在 [0,1];
//           同版两次 project 一致;各自缓存 roundtrip 后 project 仍一致;
//      Pyramid:两版 analyze 的 level 数、各层 top1 tag、features.coverage 差异 < 1e-3。

import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { EPAModule as EPAPorted } from '../core/EPAModule.mjs'
import { ResidualPyramid as PyramidPorted } from '../core/ResidualPyramid.mjs'

const ORIG_EPA = '/home/lyy/workspace/vcp-src/VCPToolBox-main/EPAModule.js'
const ORIG_PYR = '/home/lyy/workspace/vcp-src/VCPToolBox-main/ResidualPyramid.js'

// ── 合成数据集(与 tests/epa.test.mjs 同款生成器,保证可复现) ──
const DIM = 1024
const N_CLUSTERS = 4
const PER_CLUSTER = 5 // 共 20 个 tag ≥ 16(§5:同一组合成数据集 ≥16 个向量)
const NOISE_STD = 0.15 / Math.sqrt(DIM)

function mulberry32(seed) {
  let a = seed >>> 0
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gaussian(rng) {
  let u = 0
  let v = 0
  while (u === 0) u = rng()
  while (v === 0) v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function randomUnitVector(rng, dim) {
  const v = new Float32Array(dim)
  let m = 0
  for (let d = 0; d < dim; d++) {
    v[d] = gaussian(rng)
    m += v[d] * v[d]
  }
  m = Math.sqrt(m)
  for (let d = 0; d < dim; d++) v[d] /= m
  return v
}

function noisySample(center, rng, dim) {
  const v = new Float32Array(dim)
  let m = 0
  for (let d = 0; d < dim; d++) {
    v[d] = center[d] + gaussian(rng) * NOISE_STD
    m += v[d] * v[d]
  }
  m = Math.sqrt(m)
  for (let d = 0; d < dim; d++) v[d] /= m
  return v
}

const dataRng = mulberry32(20260829)
const centers = Array.from({ length: N_CLUSTERS }, () => randomUnitVector(dataRng, DIM))
// tagsF32:移植版入参(直接 Float32Array)
const tagsF32 = []
for (let c = 0; c < N_CLUSTERS; c++) {
  for (let i = 0; i < PER_CLUSTER; i++) {
    tagsF32.push({ id: c * PER_CLUSTER + i + 1, name: `tag-c${c}-${i}`, vector: noisySample(centers[c], dataRng, DIM) })
  }
}
const queryVec = noisySample(centers[0], dataRng, DIM)

// tagsBuf:原版 db 行的 Buffer 向量(Float32 小端,字节拷贝自同名 Float32Array)
// 注意 tag.vector 是 new Float32Array(DIM) 生成,其 .buffer 恰好 4096 字节
function toBuffer(f32) {
  return Buffer.from(f32.buffer)
}
const tagsBuf = tagsF32.map((t) => ({ id: t.id, name: t.name, vector: toBuffer(t.vector) }))

// ── 受控随机:安装一个"带状态"的种子随机源替换全局 Math.random,
//   在 fn(可能返回 Promise)完成前保持生效,结束后恢复原 Math.random。
//   两版必须各自吃"同一种子、独立推进"的随机流 → 分别调用本函数、同 seed。
async function withSeeded(seed, fn) {
  const real = Math.random
  const rng = mulberry32(seed)
  Math.random = () => rng() // 同一个 rng 闭包,状态逐次推进
  try {
    return await fn()
  } finally {
    Math.random = real
  }
}

// 静默初始化时的大量 [EPA] 进度日志(保留 console.error/warn),避免刷屏
function quiet(fn) {
  const real = console.log
  console.log = () => {}
  try {
    return fn()
  } finally {
    console.log = real
  }
}

// ── EPA 原版假 db(better-sqlite3 风格) ──
function makeOrigDb(tagRows) {
  const kv = new Map() // key -> JSON 字符串(模拟 kv_store 表)
  return {
    name: 'fake-db',
    prepare(sql) {
      if (sql.includes('SELECT id, name, vector FROM tags WHERE vector IS NOT NULL')) {
        return { all: () => tagRows }
      }
      if (sql.includes('SELECT COUNT(*) as count FROM tags')) {
        return { get: () => ({ count: tagRows.length }) }
      }
      if (sql.includes('SELECT value FROM kv_store WHERE key = ?')) {
        return { get: (key) => (kv.has(key) ? { value: kv.get(key) } : undefined) }
      }
      if (sql.includes('INSERT OR REPLACE INTO kv_store')) {
        return { run: (key, value) => kv.set(key, value) }
      }
      throw new Error(`未预期的原版 db SQL: ${sql}`)
    },
    _kv: kv, // 供测试读取缓存内容(非原代码调用)
  }
}

// ── EPA 移植版 cache(同步 KV,set 时强制 JSON roundtrip 模拟持久化) ──
function makeCache() {
  const m = new Map()
  return {
    get(key) {
      return m.has(key) ? m.get(key) : undefined
    },
    set(key, value) {
      m.set(key, JSON.parse(JSON.stringify(value)))
    },
  }
}

// ── 公共断言工具 ──
let passed = 0
let failed = 0
async function ok(name, fn) {
  try {
    await fn()
    passed++
    console.log(`  ✓ ${name}`)
  } catch (err) {
    failed++
    console.error(`  ✘ ${name}\n    ${err?.stack ?? err}`)
  }
}

async function main() {
  console.log('port-equivalence.test.mjs')
  console.log(`  加载原版: ${ORIG_EPA}`)
  console.log(`  加载原版: ${ORIG_PYR}`)

  const require = createRequire(import.meta.url)
  const EPAModuleOrig = require(ORIG_EPA)
  const ResidualPyramidOrig = require(ORIG_PYR)

  // 数据集固有结构检查(保障"聚类结构"前提成立,与 epa.test 同款)
  const mean = (arr) => arr.reduce((s, x) => s + x, 0) / (arr.length || 1)
  const within = []
  const across = []
  for (let a = 0; a < tagsF32.length; a++) {
    const ca = Math.floor((tagsF32[a].id - 1) / PER_CLUSTER)
    for (let b = a + 1; b < tagsF32.length; b++) {
      const cb = Math.floor((tagsF32[b].id - 1) / PER_CLUSTER)
      let dot = 0
      for (let d = 0; d < DIM; d++) dot += tagsF32[a].vector[d] * tagsF32[b].vector[d]
      if (ca === cb) within.push(dot)
      else across.push(dot)
    }
  }
  assert.ok(mean(within) > 0.9, `簇内余弦均值应>0.9,实际 ${mean(within).toFixed(4)}`)
  assert.ok(mean(across) < 0.2, `簇间余弦均值应<0.2,实际 ${mean(across).toFixed(4)}`)
  console.log(`  [观测] 簇内余弦均值=${mean(within).toFixed(4)}, 簇间余弦均值=${mean(across).toFixed(4)}`)

  // ══════════════ 第 1 部分:EPAModule 原版 vs 移植版 ══════════════
  console.log('== EPA 原版 vs 移植版 ==')

  // 两版共用同一配置(1024 维/个人规模,§1 移植默认)
  const epaCfg = { dimension: DIM, clusterCount: 12, maxBasisDim: 32 }
  const mathSeed = 20260830

  let epaOrig = null
  let epaPort = null
  let portCache = null

  // 原版实例:假 db + 同源 Buffer 向量
  const origDb = makeOrigDb(tagsBuf)
  await ok('原版 initialize 成功(20 tag ≥ 8)且 trained=true', async () => {
    const ok = await withSeeded(mathSeed, () => {
      epaOrig = new EPAModuleOrig(origDb, epaCfg)
      return quiet(() => epaOrig.initialize())
    })
    assert.strictEqual(ok, true)
    assert.strictEqual(epaOrig.initialized, true)
  })

  // 移植版实例:tagProvider + cache,吃同一随机种子序列
  await ok('移植版 initialize 成功且 trained=true', async () => {
    const provider = { calls: 0, listTagVectors() { this.calls++; return tagsF32 } }
    portCache = makeCache()
    const r = await withSeeded(mathSeed, () => {
      epaPort = new EPAPorted(provider, portCache, epaCfg)
      return quiet(() => epaPort.initialize())
    })
    assert.strictEqual(r, true)
    assert.strictEqual(epaPort.trained, true)
  })

  // —— 对照:project 的 entropy/logicDepth/dominantAxes ——
  let projOrig = null
  let projPort = null
  await ok('两版 project 对照:entropy/logicDepth 差异 < 1e-3,dominantAxes 数量一致', () => {
    const a = quiet(() => epaOrig.project(queryVec))
    const b = epaPort.project(queryVec)
    assert.ok(a !== null, '原版 trained 后 project 不应为 null')
    assert.ok(b !== null, '移植版 trained 后 project 不应为 null')
    projOrig = a
    projPort = b
    const dEntropy = Math.abs(a.entropy - b.entropy)
    const dLogic = Math.abs(a.logicDepth - b.logicDepth)
    console.log(`  [观测] entropy 原版=${a.entropy.toFixed(8)} 移植=${b.entropy.toFixed(8)} (diff=${dEntropy.toExponential(2)})`)
    console.log(`  [观测] logicDepth 原版=${a.logicDepth.toFixed(8)} 移植=${b.logicDepth.toFixed(8)} (diff=${dLogic.toExponential(2)})`)
    assert.ok(dEntropy < 1e-3, `entropy 差异应 < 1e-3,实际 ${dEntropy}`)
    assert.ok(dLogic < 1e-3, `logicDepth 差异应 < 1e-3,实际 ${dLogic}`)
    assert.strictEqual(a.dominantAxes.length, b.dominantAxes.length, 'dominantAxes 数量应一致')
    // 逐轴标签一致(只列前 3 个观测值)
    const brief = (p) => p.dominantAxes.slice(0, 3).map((x) => `${x.label}:${x.energy.toFixed(3)}`).join(' | ')
    console.log(`  [观测] dominantAxes 原版[${a.dominantAxes.length}] = ${brief(a)}`)
    console.log(`  [观测] dominantAxes 移植[${b.dominantAxes.length}] = ${brief(b)}`)
  })

  // —— 不变量(§5 容错约定):probabilities 和为 1、熵在 [0,1]、同版两次 project 一致 ——
  await ok('不变量:两版 probabilities 和≈1、entropy∈[0,1]、同版两次 project 结果一致', () => {
    for (const [tag, p] of [['原版', projOrig], ['移植版', projPort]]) {
      let sum = 0
      for (const x of p.probabilities) sum += x
      assert.ok(Math.abs(sum - 1) < 1e-4, `${tag}: probabilities 和应≈1,实际 ${sum}`)
      assert.ok(p.entropy >= 0 && p.entropy <= 1 + 1e-6, `${tag}: entropy 应在 [0,1],实际 ${p.entropy}`)
      assert.ok(p.logicDepth >= 0 && p.logicDepth <= 1 + 1e-6, `${tag}: logicDepth 应在 [0,1]`)
      // project 内无随机:同一实例连续两次结果必须完全一致
      const again = tag === '原版' ? quiet(() => epaOrig.project(queryVec)) : epaPort.project(queryVec)
      assert.deepStrictEqual(Array.from(again.projections), Array.from(p.projections), `${tag}: 同版两次 project 应一致`)
      assert.deepStrictEqual(Array.from(again.probabilities), Array.from(p.probabilities), `${tag}: probabilities 应一致`)
      assert.strictEqual(again.entropy, p.entropy)
      assert.strictEqual(again.dominantAxes.length, p.dominantAxes.length)
    }
  })

  // —— 缓存 roundtrip(两版各自):新实例同 cache/db → 直接从缓存加载,project 与首训完全一致 ——
  await ok('缓存 roundtrip:两版第二实例均走缓存加载,project 与首训一致', async () => {
    // 原版:同一假 db(kv 已有缓存)。若意外重训,随机用别的种子(9999)会得出不同基 → 断言会失败(符合"必须走缓存")
    let epaOrig2 = null
    const o2 = await withSeeded(9999, () => {
      epaOrig2 = new EPAModuleOrig(origDb, epaCfg)
      return quiet(() => epaOrig2.initialize())
    })
    assert.strictEqual(o2, true)
    const po3 = quiet(() => epaOrig2.project(queryVec))
    assert.deepStrictEqual(Array.from(po3.projections), Array.from(projOrig.projections), '原版 roundtrip 后 project 应一致')

    // 移植版:同一 cache
    const p2 = new EPAPorted({ listTagVectors: () => tagsF32 }, portCache, epaCfg)
    assert.strictEqual(await p2.initialize(), true)
    const pp = p2.project(queryVec)
    assert.deepStrictEqual(Array.from(pp.projections), Array.from(projPort.projections), '移植版 roundtrip 后 project 应一致')
    // 缓存值必须 JSON 可序列化(basis/mean 为普通数组,非 Float32Array)
    const cached = portCache.get('epa_basis_cache')
    assert.ok(cached && Array.isArray(cached.basis) && Array.isArray(cached.basis[0]), '移植版缓存 basis 应为普通数组')
    assert.ok(Array.isArray(cached.mean), '缓存 mean 应为普通数组')
  })

  // —— 未训练时的契约差异核对(§1:移植版未训练 project 返回 null;原版返回 _emptyResult)——
  await ok('契约差异:移植版未训练 project=null 且不抛异常(按 §1 接口规定)', async () => {
    const emptyDb = makeOrigDb([tagsBuf[0], tagsBuf[1]]) // 2 < 8
    const o = new EPAModuleOrig(emptyDb, epaCfg)
    assert.strictEqual(await o.initialize(), false, '原版低 tag 数 initialize 应 false')
    const p3 = new EPAPorted({ listTagVectors: () => tagsF32.slice(0, 2) }, makeCache(), epaCfg)
    assert.strictEqual(await p3.initialize(), false, '移植版低 tag 数 initialize 应 false')
    assert.strictEqual(p3.project(queryVec), null, '移植版未训练 project 应返回 null(§1)')
    assert.strictEqual(p3.detectCrossDomainResonance(queryVec), null, '移植版未训练共振应返回 null(§1)')
  })

  // ══════════════ 第 2 部分:ResidualPyramid 原版 vs 移植版 ══════════════
  console.log('== ResidualPyramid 原版 vs 移植版 ==')

  const pyrCfg = { dimension: DIM, maxLevels: 3, topK: 10, minEnergyRatio: 0.1 }

  // 统一余弦打分(两版各自的 fake 都用同一函数,保证"同一数据集同一查询"输入一致)
  function cosineScore(t, residual) {
    let dot = 0
    let nv = 0
    let nw = 0
    for (let i = 0; i < t.vector.length; i++) {
      dot += t.vector[i] * residual[i]
      nv += t.vector[i] * t.vector[i]
      nw += residual[i] * residual[i]
    }
    return dot / (Math.sqrt(nv) * Math.sqrt(nw))
  }

  // 原版 tagIndex + 假 db(按 id 取 Buffer 向量)
  const pyrDb = {
    prepare(sql) {
      if (sql.includes('SELECT id, name, vector FROM tags WHERE id IN')) {
        return {
          all: (...ids) =>
            ids.flat().map((id) => {
              const t = tagsBuf.find((r) => r.id === id)
              assert.ok(t, `原版 db 未找到 tag id=${id}`)
              return { id: t.id, name: t.name, vector: toBuffer(t.vector) }
            }),
        }
      }
      throw new Error(`未预期的原版金字塔 db SQL: ${sql}`)
    },
  }
  const origPyr = new ResidualPyramidOrig(
    {
      search: (residual, topK) =>
        tagsF32
          .map((t) => ({ id: t.id, score: cosineScore(t, residual) }))
          .sort((a, b) => b.score - a.score)
          .slice(0, topK),
    },
    pyrDb,
    pyrCfg
  )
  const portPyr = new PyramidPorted(
    async (residual, topK) =>
      tagsF32
        .map((t) => ({ ...t, similarity: cosineScore(t, residual) }))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, topK)
        .map((t) => ({ ...t, score: t.similarity })),
    pyrCfg
  )

  let pyrOrigRes = null
  let pyrPortRes = null
  await ok('两版 analyze 对照:level 数一致,各层 top1 tag 一致,coverage 差异 < 1e-3', async () => {
    pyrOrigRes = quiet(() => origPyr.analyze(queryVec)) // 原版同步
    pyrPortRes = await portPyr.analyze(queryVec) // 移植版 async
    assert.strictEqual(pyrOrigRes.levels.length, pyrPortRes.levels.length, 'level 数应一致')
    console.log(
      `  [观测] levels=${pyrOrigRes.levels.length}, totalExplainedEnergy 原版=${pyrOrigRes.totalExplainedEnergy.toFixed(6)} 移植=${pyrPortRes.totalExplainedEnergy.toFixed(6)}`
    )
    for (let l = 0; l < pyrOrigRes.levels.length; l++) {
      const lo = pyrOrigRes.levels[l]
      const lp = pyrPortRes.levels[l]
      assert.strictEqual(lo.tags[0].name, lp.tags[0].name, `第 ${l} 层 top1 tag 应一致`)
      assert.strictEqual(lo.tags.length, lp.tags.length, `第 ${l} 层 tag 数应一致`)
      const dCov = Math.abs(lo.energyExplained - lp.energyExplained)
      const dRes = Math.abs(lo.residualEnergyRatio - lp.residualEnergyRatio)
      assert.ok(dCov < 1e-3, `第 ${l} 层 energyExplained 差异 < 1e-3,实际 ${dCov}`)
      assert.ok(dRes < 1e-3, `第 ${l} 层 residualEnergyRatio 差异 < 1e-3,实际 ${dRes}`)
      for (let i = 0; i < lo.tags.length; i++) {
        assert.strictEqual(lo.tags[i].name, lp.tags[i].name, `第 ${l} 层第 ${i} 个 tag 名应一致`)
        const dSim = Math.abs(lo.tags[i].similarity - lp.tags[i].similarity)
        const dCon = Math.abs((lo.tags[i].contribution || 0) - (lp.tags[i].contribution || 0))
        assert.ok(dSim < 1e-3, `第 ${l} 层 tag ${i} similarity 差异 < 1e-3,实际 ${dSim}`)
        assert.ok(dCon < 1e-3, `第 ${l} 层 tag ${i} contribution 差异 < 1e-3,实际 ${dCon}`)
      }
    }
  })

  await ok('两版 features 对照:depth/coverage/novelty/coherence 差异 < 1e-3', () => {
    const f0 = pyrOrigRes.features
    const f1 = pyrPortRes.features
    assert.strictEqual(f0.depth, f1.depth)
    for (const key of ['coverage', 'novelty', 'coherence', 'tagMemoActivation']) {
      const d = Math.abs(f0[key] - f1[key])
      assert.ok(d < 1e-3, `features.${key} 差异 < 1e-3,实际 ${d}(原版=${f0[key]},移植=${f1[key]})`)
    }
    console.log(
      `  [观测] coverage 原版=${f0.coverage.toFixed(6)} 移植=${f1.coverage.toFixed(6)};` +
        ` novelty 原版=${f0.novelty.toFixed(6)} 移植=${f1.novelty.toFixed(6)}`
    )
  })

  // —— 空 tag 集与零向量查询的降级形态对照 ——
  await ok('空 tag 集/零向量:两版均返回空 levels 不抛异常', async () => {
    const oe = new ResidualPyramidOrig({ search: () => [] }, pyrDb, pyrCfg)
    const pe = new PyramidPorted(async () => [], pyrCfg)
    const ro = quiet(() => oe.analyze(queryVec))
    const rp = await pe.analyze(queryVec)
    assert.strictEqual(ro.levels.length, 0)
    assert.strictEqual(rp.levels.length, 0)
    assert.strictEqual(rp.totalExplainedEnergy, 0)
    // 零向量(防除零)
    const ro2 = quiet(() => oe.analyze(new Float32Array(DIM)))
    const rp2 = await pe.analyze(new Float32Array(DIM))
    assert.strictEqual(ro2.levels.length, 0)
    assert.strictEqual(rp2.levels.length, 0)
  })

  console.log(`\nport-equivalence.test.mjs: 通过 ${passed} 项,失败 ${failed} 项`)
  process.exitCode = failed ? 1 : 0
}

await main()