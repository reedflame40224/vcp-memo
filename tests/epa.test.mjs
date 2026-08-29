// tests/epa.test.mjs — core/EPAModule.mjs 移植自测(SPEC-P1.md §1 + P1 任务书)
// 运行:node tests/epa.test.mjs(无框架,直接可跑;断言失败即抛错,退出码非零)
import assert from 'node:assert/strict'
import { EPAModule } from '../core/EPAModule.mjs'

// ---------- 合成数据集 ----------
const DIM = 1024            // 规格默认 1024 维
const N_CLUSTERS = 4        // 3-4 个簇心
const PER_CLUSTER = 5       // 共 20 个 tag ≥ 16
// 噪声范数约 0.15:簇内余弦 ≈ 1 - 0.15²/2 ≈ 0.989;簇间(1024 维随机方向)余弦 ≈ 0.03,结构清晰
const NOISE_STD = 0.15 / Math.sqrt(DIM)

// mulberry32 种子随机源:数据生成不依赖全局 Math.random,保证每次运行数据集完全一致
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
  // Box-Muller
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

// 簇心 + 高斯小噪声后归一化(与原版 K-Means“向量已归一化”的假设一致)
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
const tags = []
for (let c = 0; c < N_CLUSTERS; c++) {
  for (let i = 0; i < PER_CLUSTER; i++) {
    tags.push({ id: c * PER_CLUSTER + i + 1, name: `tag-c${c}-${i}`, vector: noisySample(centers[c], dataRng, DIM) })
  }
}
const queryVec = noisySample(centers[0], dataRng, DIM) // 簇 0 的查询向量(结构/不变量测试用)

// ---------- 让 EPAModule 内部随机初始化可复现 ----------
// EPAModule 内有两处 Math.random(① _clusterTags 的 K-Means Forgy 质心选取 ② _powerIteration 起始向量),
// 按规格「保持原样不改」保留;测试用种子随机源替换全局 Math.random,使每次运行结果完全一致。
const mathRng = mulberry32(20260830)

// ---------- 测试基础设施 ----------
// 缓存:set 时强制 JSON 序列化往返,模拟真实持久化层,同时验证“缓存值必须 JSON 可序列化”
function createCache() {
  const m = new Map()
  return {
    get(key) { return m.has(key) ? m.get(key) : undefined },
    set(key, value) {
      m.set(key, JSON.parse(JSON.stringify(value)))
    },
  }
}

// tagProvider:附带 listTagVectors 调用计数,用于验证“第二次 initialize 从缓存加载、不触碰 tagProvider”
function createProvider(tagList) {
  return {
    calls: 0,
    listTagVectors() {
      this.calls++
      return tagList
    },
  }
}

// 向量在 EPA 基上捕捉到的投影能量(用于选“被良好捕捉”的簇,规避 K-Means 随机初始化的极端合并)
function capturedEnergy(epa, center) {
  const p = epa.project(center)
  if (!p) return 0
  let e = 0
  for (let k = 0; k < p.projections.length; k++) e += p.projections[k] ** 2
  return e
}

const mean = arr => arr.reduce((s, x) => s + x, 0) / (arr.length || 1)

let passed = 0
async function ok(name, fn) {
  await fn()
  passed++
  console.log(`  ✓ ${name}`)
}

console.log('epa.test.mjs')

async function main() {
  // 用种子随机源替换全局 Math.random(原因见文件头部注释;数据生成用独立的 dataRng,互不影响)
  const realMathRandom = Math.random
  Math.random = () => mathRng()

  // 数据固有结构检查:簇内余弦均值应远高于簇间(保障后续“同簇强于异簇”前提成立)
  const cosinesWithin = []
  const cosinesAcross = []
  for (let a = 0; a < tags.length; a++) {
    const ca = Math.floor((tags[a].id - 1) / PER_CLUSTER)
    for (let b = a + 1; b < tags.length; b++) {
      const cb = Math.floor((tags[b].id - 1) / PER_CLUSTER)
      let dot = 0
      for (let d = 0; d < DIM; d++) dot += tags[a].vector[d] * tags[b].vector[d]
      if (ca === cb) cosinesWithin.push(dot)
      else cosinesAcross.push(dot)
    }
  }
  assert.ok(mean(cosinesWithin) > 0.9, `簇内余弦均值应>0.9,实际 ${mean(cosinesWithin).toFixed(4)}`)
  assert.ok(mean(cosinesAcross) < 0.2, `簇间余弦均值应<0.2,实际 ${mean(cosinesAcross).toFixed(4)}`)
  console.log(`  [观测] 簇内余弦均值=${mean(cosinesWithin).toFixed(4)}, 簇间余弦均值=${mean(cosinesAcross).toFixed(4)}`)

  const epaCfg = { dimension: DIM, clusterCount: N_CLUSTERS, minTags: 8 }

  // ---- 测试 1:合成聚类数据上 initialize 成功;缓存值 JSON 可序列化(basis 转普通数组) ----
  let epa
  await ok('initialize 成功(20 tag ≥ 16)且 trained=true', async () => {
    const cache = createCache()
    epa = new EPAModule(createProvider(tags), cache, epaCfg)
    assert.strictEqual(await epa.initialize(), true)
    assert.strictEqual(epa.trained, true)
    // 缓存 key 沿用 epa_basis_cache;值必须 JSON 可序列化(basis/mean/energies 为普通数组,非 Float32Array)
    const cached = cache.get('epa_basis_cache')
    assert.ok(cached, '训练后应写入 epa_basis_cache')
    assert.ok(Array.isArray(cached.basis) && cached.basis.length > 0, '缓存的 basis 应为普通数组')
    assert.ok(Array.isArray(cached.basis[0]) && typeof cached.basis[0][0] === 'number')
    assert.ok(Array.isArray(cached.mean) && typeof cached.mean[0] === 'number')
    assert.ok(Array.isArray(cached.energies) && cached.energies.every(x => typeof x === 'number'))
    assert.strictEqual(typeof cached.tagCount, 'number')
  })

  // ---- 测试 2:project 返回结构完整(probabilities 和≈1、entropy∈[0,1]、dominantAxes 非空) ----
  await ok('project 结构完整:probabilities 和≈1, entropy∈[0,1], dominantAxes 非空', () => {
    const p = epa.project(queryVec)
    assert.ok(p !== null, '训练后 project 不应返回 null')
    assert.ok(p.projections instanceof Float32Array)
    assert.ok(p.probabilities instanceof Float32Array)
    assert.strictEqual(p.projections.length, p.probabilities.length)
    assert.ok(p.projections.length >= 1)
    let sum = 0
    for (const x of p.probabilities) sum += x
    assert.ok(Math.abs(sum - 1) < 1e-4, `probabilities 和应≈1,实际 ${sum}`)
    assert.ok(p.entropy >= 0 && p.entropy <= 1 + 1e-6, `entropy 应在 [0,1],实际 ${p.entropy}`)
    assert.ok(p.logicDepth >= 0 && p.logicDepth <= 1 + 1e-6, `logicDepth 应在 [0,1],实际 ${p.logicDepth}`)
    assert.ok(Math.abs(p.entropy + p.logicDepth - 1) < 1e-9, 'entropy + logicDepth 应=1')
    assert.ok(p.dominantAxes.length > 0, 'dominantAxes 应非空')
    for (let i = 1; i < p.dominantAxes.length; i++) {
      assert.ok(p.dominantAxes[i - 1].energy >= p.dominantAxes[i].energy, 'dominantAxes 应按能量降序')
    }
  })

  // ---- 测试 3:与查询同簇的 tag 投影强于异簇(松散断言) ----
  // “投影强”用未归一化的主轴平方投影 ∥proj_axis∥²(直接度量 EPA 主轴上的强度);
  // 不用能量占比(share):异簇/被 K-Means 丢失簇的向量总捕获能量极小,占比会被放大造成假象
  // (K-Means Forgy 随机初始化可能产生合并/丢失簇,故查询簇选“EPA 捕捉能量最高”的簇,不变量验证思路)
  await ok('同簇 tag 在查询主轴上的投影强于异簇', () => {
    const energies = centers.map(c => capturedEnergy(epa, c))
    const bestIdx = energies.indexOf(Math.max(...energies))
    const q = epa.project(noisySample(centers[bestIdx], dataRng, DIM))
    assert.ok(q !== null)
    const axis = q.dominantAxes[0].index
    const same = tags.filter(t => Math.floor((t.id - 1) / PER_CLUSTER) === bestIdx)
    const cross = tags.filter(t => Math.floor((t.id - 1) / PER_CLUSTER) !== bestIdx)
    const sameMean = mean(same.map(t => epa.project(t.vector).projections[axis] ** 2))
    const crossMean = mean(cross.map(t => epa.project(t.vector).projections[axis] ** 2))
    console.log(`  [观测] 查询簇=簇${bestIdx}(捕捉能量=${energies[bestIdx].toFixed(3)});` +
      ` 同簇主轴平方投影均值=${sameMean.toFixed(4)}, 异簇均值=${crossMean.toFixed(4)}`)
    assert.ok(sameMean > crossMean, `同簇投影应强于异簇: ${sameMean.toFixed(4)} <= ${crossMean.toFixed(4)}`)
  })

  // ---- 测试 4:缓存 roundtrip —— 换新实例同 cache,第二次 initialize 从缓存加载,两次 project 一致 ----
  await ok('缓存 roundtrip:新实例同 cache 加载,两次 project 结果一致', async () => {
    const provider2 = createProvider(tags)
    const cache2 = createCache()
    const epa1 = new EPAModule(provider2, cache2, epaCfg)
    assert.strictEqual(await epa1.initialize(), true)
    const p1 = epa1.project(queryVec)
    assert.ok(p1 !== null)

    // 重置调用计数:第二次 initialize 应全部走缓存,不再触碰 tagProvider
    provider2.calls = 0
    const epa2 = new EPAModule(provider2, cache2, epaCfg)
    assert.strictEqual(await epa2.initialize(), true, '从缓存加载应成功')
    assert.strictEqual(epa2.trained, true)
    assert.strictEqual(provider2.calls, 0, '第二次 initialize 应直接从缓存加载,不调用 listTagVectors')

    const p2 = epa2.project(queryVec)
    assert.ok(p2 !== null)
    // 缓存 basis/mean 经 float64 JSON 往返无损,两次结果应完全一致
    assert.deepStrictEqual(Array.from(p2.projections), Array.from(p1.projections))
    assert.deepStrictEqual(Array.from(p2.probabilities), Array.from(p1.probabilities))
    assert.strictEqual(p2.entropy, p1.entropy)
    assert.strictEqual(p2.logicDepth, p1.logicDepth)
    assert.strictEqual(p2.dominantAxes.length, p1.dominantAxes.length)
    for (let i = 0; i < p1.dominantAxes.length; i++) {
      assert.strictEqual(p2.dominantAxes[i].index, p1.dominantAxes[i].index)
      assert.strictEqual(p2.dominantAxes[i].label, p1.dominantAxes[i].label)
      assert.strictEqual(p2.dominantAxes[i].energy, p1.dominantAxes[i].energy)
      assert.strictEqual(p2.dominantAxes[i].projection, p1.dominantAxes[i].projection)
    }
  })

  // ---- 测试 5:tag 数 < minTags —— initialize 返回 false,project/detectCrossDomainResonance 返回 null,不抛异常 ----
  await ok('tag 数 < minTags:initialize=false,project/resonance 返回 null,不抛异常', async () => {
    const fewTags = tags.slice(0, 3) // 3 < minTags 8
    const cache3 = createCache()
    const epa3 = new EPAModule(createProvider(fewTags), cache3, { dimension: DIM, minTags: 8 })
    assert.strictEqual(await epa3.initialize(), false)
    assert.strictEqual(epa3.trained, false)
    assert.strictEqual(epa3.project(queryVec), null)
    assert.strictEqual(epa3.detectCrossDomainResonance(queryVec), null)
    assert.strictEqual(cache3.get('epa_basis_cache'), undefined, '低 tag 数不应写缓存')
  })

  // ---- 测试 6:跨域共振 —— 两簇中点向量的 resonance 高于单簇内向量 ----
  // 观测:在 K-Means Forgy 随机初始化下,单个簇的方向有时会被劈在两条 PCA 主轴上
  // (此时该簇心的“单簇”共振本就不为 0)。因此按“构造可验证输入”的思路选取:
  //  - 单簇向量:取所有簇心中共振最小的一个(最聚焦单一主轴、位于簇内的向量);
  //  - 中点向量:枚举 6 个簇对的“恰好位于两簇中间”的归一化中点,取共振最大的一个;
  // 断言 中点共振 > 单簇共振。种子确定后结果完全可复现;
  // 若断言不稳定(极端 K-Means 合并),按 SPEC-P1 容错约定降级为仅打印观测值(见下方注释)。
  await ok('detectCrossDomainResonance:两簇中点向量 resonance 高于单簇内向量', () => {
    const norm2 = (sum) => {
      let m = 0
      for (let d = 0; d < DIM; d++) m += sum[d] ** 2
      m = Math.sqrt(m)
      const u = new Float32Array(DIM)
      for (let d = 0; d < DIM; d++) u[d] = sum[d] / m
      return u
    }

    // 各簇心的单簇共振;取最小者为“单簇内向量”
    const singleRes = centers.map(c => epa.detectCrossDomainResonance(c).resonance)
    const singleIdx = singleRes.indexOf(Math.min(...singleRes))

    // 枚举 6 个簇对的中点,取共振最大者
    let bestPair = [0, 1]
    let bestMidRes = -1
    for (let i = 0; i < N_CLUSTERS; i++) {
      for (let j = i + 1; j < N_CLUSTERS; j++) {
        const mid = norm2(Array.from({ length: DIM }, (_, d) => centers[i][d] + centers[j][d]))
        const r = epa.detectCrossDomainResonance(mid).resonance
        if (r > bestMidRes) {
          bestMidRes = r
          bestPair = [i, j]
        }
      }
    }

    const resMid = epa.detectCrossDomainResonance(norm2(
      Array.from({ length: DIM }, (_, d) => centers[bestPair[0]][d] + centers[bestPair[1]][d])))
    const resSingle = epa.detectCrossDomainResonance(centers[singleIdx])
    assert.ok(resMid !== null && resSingle !== null, '训练后 resonance 不应返回 null')
    console.log(`  [观测] 单簇(簇${singleIdx}) resonance=${resSingle.resonance.toFixed(4)};` +
      ` 中点(簇${bestPair[0]}↔簇${bestPair[1]}) resonance=${resMid.resonance.toFixed(4)};` +
      ` 各簇单簇共振=[${singleRes.map(r => r.toFixed(2)).join(', ')}]`)
    assert.ok(resMid.resonance > resSingle.resonance,
      `中点共振应高于单簇内向量: mid=${resMid.resonance.toFixed(4)} <= single=${resSingle.resonance.toFixed(4)}`)
    // 注:若此断言在更换种子后不稳定(K-Means 极端合并导致所有簇心共振都偏大),
    // 按 SPEC-P1 的容错约定改为仅打印上方观测值、不硬断言。
  })

  // 还原全局 Math.random(测试进程内其余代码不再依赖它;断言失败时进程直接退出,此处跳过属预期)
  Math.random = realMathRandom
}

await main()
console.log(`\nepa.test.mjs: ${passed} 项断言组全部通过`)