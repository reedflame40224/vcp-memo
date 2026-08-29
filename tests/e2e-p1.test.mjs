// tests/e2e-p1.test.mjs —— P1 Tag 层端到端验收(SPEC-P1.md §7 第 4 条,零依赖)
// 运行:node tests/e2e-p1.test.mjs(依赖真实 Ollama bge-m3 于 127.0.0.1:11434)
//
// 覆盖(SPEC-P1 §7.4):
//   ① 预置 ≥10 篇带 Tag 的日记(构造 8+ 个不同 Tag,其中若干共享 Tag);
//   ② recall 返回结构含 matchedTags/epa/pyramid 字段;stats.epa.trained === true;
//   ③ 查询与某共享 Tag 语义相关时,该 Tag 出现在 matchedTags 中;
//   ④ blocks 排序与"同查询下 P0 纯 KNN 结果"一致(显式断言,防回归)——测试内
//      自实现一份 P0 纯 KNN 参照(读 chunks.jsonl + 归一化向量 + 暴力余弦),与
//      store.recall 的 blocks 序列逐条比对;
//   ⑤ 无 Tag 的库上 recall 优雅降级(matchedTags: [], stats.pyramid: null);
//   ⑥ sig 不一致仍拒绝服务。

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { createEmbedder } from '../engine/embed.mjs'
import { openStore } from '../engine/store.mjs'

const REAL = { baseUrl: 'http://127.0.0.1:11434/v1', model: 'bge-m3', dimension: 1024 }
const SIG = 'bge-m3@1024'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const readText = (f) => {
  try {
    return fs.readFileSync(f, 'utf8')
  } catch {
    return null
  }
}

// 轮询等待索引队列排空
async function waitPending(store, { timeout = 90000, what = '' } = {}) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeout) {
    if (store.stats().pendingFiles === 0) return store.stats()
    await sleep(100)
  }
  throw new Error(`等待索引队列排空超时(${what})，最后 stats=${JSON.stringify(store.stats())}`)
}

// 轮询等待真实结果条件
async function waitFor(what, pred, timeout = 90000) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeout) {
    if (pred()) return
    await sleep(100)
  }
  throw new Error(`等待条件超时: ${what}`)
}

function collectLog() {
  const logs = []
  return [(level, msg) => logs.push(`${level}:${msg}`), logs]
}

let passed = 0
function ok(what) {
  passed++
  console.log(`  ✓ ${what}`)
}

// ── P0 纯 KNN 参照实现(独立于 store,只读 chunks.jsonl) ──
function normalize(v) {
  const arr = v instanceof Float32Array ? v : Float32Array.from(v)
  let sum = 0
  for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i]
  const n = Math.sqrt(sum)
  if (n > 0) for (let i = 0; i < arr.length; i++) arr[i] /= n
  return arr
}
function dot(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

async function referenceKnn(embedder, chunksPath, query, k, truncate) {
  const lines = (readText(chunksPath) || '').split('\n').filter(Boolean)
  const docs = lines.map((l) => JSON.parse(l)).filter((o) => Array.isArray(o.vector))
  const [qv] = await embedder.embed([query])
  assert.ok(qv, '参照实现:query 必须能嵌入')
  const q = normalize(qv)
  const scored = []
  for (const o of docs) {
    const s = dot(q, normalize(o.vector))
    if (s >= truncate) scored.push({ file: o.file, chunkIndex: o.chunkIndex, score: s })
  }
  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, Math.max(1, Math.floor(k) || 1)).map((x) => [x.file, x.chunkIndex])
}

// 四段式日记文本
const diaryText = (body, tags) => `[2026-08-29] - dsh\n[14:30]\n${body}\n\nTag: ${tags}\n`

// ── 预置数据集:10 篇带 Tag 日记,14 个不同 tag,机器学习×4 / 数据分析×2 / 旅行×2 / 美食×2 共享 ──
const DIARIES = [
  ['2026-08-29-09_00_01-机器学习1.md',
    '使用梯度提升树与随机森林做特征选择,评估召回率与精确率的平衡。', '机器学习, 数据分析'],
  ['2026-08-29-09_00_02-机器学习2.md',
    '神经网络的训练过程需要调节学习率与批大小,并时刻关注过拟合风险。', '机器学习, 深度学习'],
  ['2026-08-29-09_00_03-机器学习3.md',
    '把电商点击数据做成特征工程管线,训练点击率预估模型并上线 A/B 测试。', '机器学习, 数据分析'],
  ['2026-08-29-09_00_04-机器学习4.md',
    '对故障日志做文本分类,用 TF-IDF 与 SVM 建立基线,再对比神经网络。', '机器学习, 自然语言处理'],
  ['2026-08-29-09_00_05-旅行1.md',
    '五一去大理和丽江,住在古城边的民宿,吃了菌子火锅,爬了苍山。', '旅行, 美食'],
  ['2026-08-29-09_00_06-旅行2.md',
    '周末爬山摄影,带了三脚架拍日出,山路很陡但景色值得。', '旅行, 摄影'],
  ['2026-08-29-09_00_07-美食1.md',
    '学会了做红烧肉和清蒸鱼,晚上还尝试烤了全麦面包。', '美食, 烹饪'],
  ['2026-08-29-09_00_08-健身1.md',
    '晨跑五公里配速五分半,回来做了二十个引体向上和一百个深蹲。', '健身, 运动'],
  ['2026-08-29-09_00_09-量子1.md',
    '量子计算的容错阈值与表面码纠错是当前研究热点,我们开了三小时讨论会。', '量子物理, 科研'],
  ['2026-08-29-09_00_10-音乐1.md',
    '晚上听贝多芬第九交响曲,合唱部分非常震撼,旋律久久不散。', '音乐, 艺术'],
]

async function main() {
  const base = fs.mkdtempSync(path.join('/tmp', 'vcp-memo-p1-'))
  console.log('== e2e-p1.test.mjs ==')
  console.log(`  临时根 = ${base}`)
  const embedder = createEmbedder(REAL)

  try {
    // ═══════════════ ①+②+③+④:带 Tag 库全链路 ═══════════════
    console.log('== ① 预置 10 篇带 Tag 日记(14 个不同 tag,含共享)→ openStore 自动重建 ==')
    const dataRootA = path.join(base, 'dataA')
    for (const [name, body, tags] of DIARIES) {
      const abs = path.join(dataRootA, 'diaries', 'dsh', name)
      fs.mkdirSync(path.dirname(abs), { recursive: true })
      fs.writeFileSync(abs, diaryText(body, tags))
    }
    const [logFnA, logsA] = collectLog()
    const storeA = await openStore(
      { dataRoot: dataRootA, agentName: 'dsh', watch: true },
      embedder,
      logFnA,
    )
    await storeA.ready
    // doRebuild 内已同步 flush tag 层;再轮询确保(以真实结果为准)
    await waitFor('EPA 训练完成', () => storeA.stats().epaTrained === true, 120000)
    const stA = await waitPending(storeA, { what: '带 Tag 库' })
    assert.ok(stA.indexedChunks >= 10, `indexedChunks >= 10（实际 ${stA.indexedChunks}）`)
    assert.ok(stA.tagCount >= 8, `tagCount >= 8（实际 ${stA.tagCount}）`)
    assert.ok(stA.vectorizedTags >= 8, `vectorizedTags >= 8（实际 ${stA.vectorizedTags}）`)
    assert.strictEqual(stA.epaTrained, true, 'EPA 应已训练')
    assert.ok(fs.existsSync(path.join(dataRootA, 'index', 'tags.jsonl')), 'tags.jsonl 应已生成')
    assert.ok(fs.existsSync(path.join(dataRootA, 'index', 'epa.json')), 'epa.json 应已生成')
    ok(`10 篇日记已索引(${stA.indexedChunks} chunks),${stA.tagCount} 个 tag 全部向量化,EPA 已训练`)

    console.log('== ② recall 结构:matchedTags/epa/pyramid 字段 + epa.trained === true ==')
    const recA = await storeA.recall({ query: '机器学习的模型训练与参数调优', k: 6, truncate: 0.4 })
    assert.ok(Array.isArray(recA.matchedTags), 'matchedTags 应为数组')
    assert.ok(recA.stats && typeof recA.stats.ms === 'number', 'stats.ms 保留')
    assert.ok(recA.stats.epa && typeof recA.stats.epa === 'object', 'stats.epa 存在')
    assert.strictEqual(recA.stats.epa.trained, true, 'stats.epa.trained === true')
    assert.strictEqual(typeof recA.stats.epa.entropy, 'number')
    assert.strictEqual(typeof recA.stats.epa.logicDepth, 'number')
    assert.strictEqual(typeof recA.stats.epa.resonance, 'number')
    assert.ok(Array.isArray(recA.stats.epa.dominantAxes), 'dominantAxes 应为数组')
    for (const ax of recA.stats.epa.dominantAxes) {
      assert.strictEqual(typeof ax.label, 'string')
      assert.strictEqual(typeof ax.score, 'number')
    }
    assert.ok(recA.stats.pyramid && typeof recA.stats.pyramid === 'object', 'stats.pyramid 存在')
    for (const k of ['depth', 'coverage', 'novelty', 'coherence', 'activation']) {
      assert.strictEqual(typeof recA.stats.pyramid[k], 'number', `stats.pyramid.${k} 应为数字`)
    }
    // matchedTags 结构:最多 8 个,含 tag/weight/level/notation,按 weight 降序
    assert.ok(recA.matchedTags.length <= 8, `matchedTags 最多 8 个（实际 ${recA.matchedTags.length}）`)
    for (const [i, m] of recA.matchedTags.entries()) {
      assert.strictEqual(typeof m.tag, 'string')
      assert.strictEqual(typeof m.weight, 'number')
      assert.strictEqual(typeof m.level, 'number')
      assert.strictEqual(m.notation, `=${m.tag}:${m.weight.toFixed(2)}@seed`, 'notation 应符合 VCP 记号')
      if (i > 0) assert.ok(recA.matchedTags[i - 1].weight >= m.weight, 'matchedTags 应按 weight 降序')
    }
    console.log(`  [观测] matchedTags=${JSON.stringify(recA.matchedTags.map((m) => m.tag))}`)
    console.log(`  [观测] epa={entropy:${recA.stats.epa.entropy.toFixed(3)},logicDepth:${recA.stats.epa.logicDepth.toFixed(3)},` +
      `resonance:${recA.stats.epa.resonance.toFixed(3)},axes:${recA.stats.epa.dominantAxes.length}};` +
      ` pyramid={depth:${recA.stats.pyramid.depth},coverage:${recA.stats.pyramid.coverage.toFixed(3)},` +
      `novelty:${recA.stats.pyramid.novelty.toFixed(3)},activation:${recA.stats.pyramid.activation.toFixed(3)}}`)
    ok('recall 结构完整:matchedTags 数组(≤8),stats.epa{trained:true,...},stats.pyramid{...}')

    console.log('== ③ 查询与共享 Tag 语义相关 → 该 Tag 出现在 matchedTags 中 ==')
    const mlHit = recA.matchedTags.some((m) => m.tag === '机器学习')
    assert.ok(mlHit, '查询"机器学习的模型训练与参数调优"时共享 Tag"机器学习"应出现在 matchedTags 中')
    const travelRec = await storeA.recall({ query: '去云南旅行爬山的计划', k: 6, truncate: 0.4 })
    console.log(`  [观测] "云南旅行" matchedTags=${JSON.stringify(travelRec.matchedTags.map((m) => m.tag))}`)
    assert.ok(
      travelRec.matchedTags.some((m) => m.tag === '旅行'),
      '查询"去云南旅行爬山的计划"时应出现共享 Tag"旅行"',
    )
    ok('语义相关查询把对应共享 Tag 带进 matchedTags')

    console.log('== ④ blocks 排序与 P0 纯 KNN 一致(防回归)==')
    const chunksPathA = path.join(dataRootA, 'index', 'chunks.jsonl')
    for (const [q, label] of [
      ['机器学习的模型训练与参数调优', '机器学习查询'],
      ['去云南旅行爬山的计划', '旅行查询'],
      ['量子计算与容错', '量子查询'],
    ]) {
      const rec = await storeA.recall({ query: q, k: 6, truncate: 0.3 })
      const ref = await referenceKnn(embedder, chunksPathA, q, 6, 0.3)
      assert.strictEqual(rec.blocks.length, ref.length, `${label}:blocks 数量应与 P0 纯 KNN 一致`)
      assert.deepStrictEqual(
        rec.blocks.map((b) => [b.file, b.chunkIndex]),
        ref,
        `${label}:blocks 序列(文件+chunkIndex)应与 P0 纯 KNN 完全一致`,
      )
    }
    // 无候选时的降级也一致(truncate 极高 → 两边都是空)
    const recEmpty = await storeA.recall({ query: '机器学习的模型训练与参数调优', k: 6, truncate: 1.01 })
    const refEmpty = await referenceKnn(embedder, chunksPathA, '机器学习的模型训练与参数调优', 6, 1.01)
    assert.deepStrictEqual(
      recEmpty.blocks.map((b) => [b.file, b.chunkIndex]),
      refEmpty,
      'truncate 过滤全部时 blocks 应两边都为空',
    )
    ok('三个查询 + 空候选场景:blocks 序列与 P0 纯 KNN 参照逐条一致')
    await storeA.close()

    // ═══════════════ ⑤ 无 Tag 库 → 优雅降级 ═══════════════
    console.log('== ⑤ 无 Tag 库:matchedTags=[] / stats.pyramid=null / epa.trained=false ==')
    const dataRootB = path.join(base, 'dataB')
    const noTagBodies = [
      '今天把 vcp-memo 的数据目录放在独立位置,与插件目录分离,并用 JSONL 作为索引格式。',
      '晚餐吃了红烧肉配米饭和青菜汤,明天计划晨跑五公里并写周报。',
      '下周要评审新方案的接口设计,需要先整理一份对比文档。',
    ]
    noTagBodies.forEach((body, i) => {
      const abs = path.join(dataRootB, 'diaries', 'dsh', `2026-08-29-10_0${i}_00-无标签${i}.md`)
      fs.mkdirSync(path.dirname(abs), { recursive: true })
      fs.writeFileSync(abs, `[2026-08-29] - dsh\n[10:0${i}]\n${body}\n`)
    })
    const [logFnB, logsB] = collectLog()
    const storeB = await openStore({ dataRoot: dataRootB, agentName: 'dsh', watch: false }, embedder, logFnB)
    await storeB.ready
    await waitPending(storeB, { what: '无 Tag 库' })
    assert.strictEqual(storeB.stats().tagCount, 0, '无 Tag 行 → tagCount 应为 0')
    const recB = await storeB.recall({ query: '数据目录放在独立位置', k: 6, truncate: 0.4 })
    assert.ok(Array.isArray(recB.blocks) && recB.blocks.length >= 1, '无 Tag 库上 recall 仍返回 blocks')
    assert.deepStrictEqual(recB.matchedTags, [], '无 Tag 库 matchedTags 应为 []')
    assert.ok(recB.stats.epa && recB.stats.epa.trained === false, 'epa.trained 应为 false')
    assert.strictEqual(recB.stats.pyramid, null, 'stats.pyramid 应为 null')
    ok('无 Tag 库优雅降级:matchedTags=[], pyramid=null, epa={trained:false}, blocks 正常')
    await storeB.close()

    // ═══════════════ ⑥ sig 不一致 → 拒绝服务 ═══════════════
    console.log('== ⑥ sig 不一致仍拒绝服务 ==')
    const dataRootC = path.join(base, 'dataC')
    fs.mkdirSync(path.join(dataRootC, 'diaries', 'dsh'), { recursive: true })
    fs.mkdirSync(path.join(dataRootC, 'index'), { recursive: true })
    fs.writeFileSync(
      path.join(dataRootC, 'index', 'meta.json'),
      JSON.stringify({ sig: 'bge-m3@768', dimension: 768, chunkCount: 0, updatedAt: 0 })
    )
    fs.writeFileSync(
      path.join(dataRootC, 'diaries', 'dsh', '2026-08-29-11_00_00.md'),
      diaryText('旧语义空间下的内容。', '旧')
    )
    const storeC = await openStore({ dataRoot: dataRootC, agentName: 'dsh', watch: false }, embedder, collectLog()[0])
    await assert.rejects(storeC.ready, (e) => {
      assert.match(e.message, /嵌入模型签名已变/, 'ready 拒绝,含签名变更说明')
      assert.match(e.message, /旧 bge-m3@768 → 新 bge-m3@1024/, '旧→新签名')
      return true
    })
    await assert.rejects(storeC.recall({ query: 'x' }), /嵌入模型签名已变/, 'recall 抛拒绝服务错误')
    assert.ok(!fs.existsSync(path.join(dataRootC, 'index', 'chunks.jsonl')), '拒绝服务期间不应写 chunks.jsonl')
    await storeC.close()
    ok('sig 不一致(bge-m3@768)拒绝服务:ready/recall 抛签名错误,不动盘')

    console.log(`\ne2e-p1.test.mjs: 通过 ${passed} 组断言`)
  } catch (err) {
    console.error('e2e-p1 测试失败:', err)
    process.exitCode = 1
  } finally {
    fs.rmSync(base, { recursive: true, force: true })
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})