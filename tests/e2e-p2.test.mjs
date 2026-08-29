// tests/e2e-p2.test.mjs —— P2 端到端验收(SPEC-P2.md §8 第 6 条,真 Ollama bge-m3,临时库)
// 运行:node tests/e2e-p2.test.mjs(依赖真实 Ollama bge-m3 于 127.0.0.1:11434)
//
// 验收核心:A/B 固定对照(SPEC-P2 §8.6)——自造词"泽塔波"私有概念测试中,
//   增强路径找回纯 KNN 漏掉的结构记忆。同一数据根开两个 store 实例:
//     · A 组:纯 KNN(tagmemo.enabled=false,同库第二实例)→ 记录 B 缺席/靠后;
//     · B 组:wave 路径(同库同查询同 k)→ 断言 B 被召回且 blocks 数/排序提升
//       (泽塔波→澜沧计划→B 的结构传导:日志 A 里 泽塔波 与 澜沧计划 共现,
//        B 的 Tag 澜沧计划 挂在该图节点上,波传播把 B 所在文件捞回)。
// 另断言:
//   ③ 只加分不罚分:无结构证据的无关日记不因 wave 消失(候选并集 ⊆ 断言),分数不变负;
//   ④ stats.wave 字段齐全(alpha/effectiveBoost/logicDepth/resonance/coverage/activation/
//      seeds/emergentCount/fieldNodes/graphGeneration);
//   ⑤ 显式 tags:['澜沧计划'] → matchedTags 含该 Core Tag 且 sourceType='core';
//   ⑥ sig 拒绝服务、无 Tag 库回退 P1 —— 两条老纪律仍在(P0/P1 契约不回归)。
// 任何一条断言失败即整组失败;发现 A/B 不成立时必须先排查图/传播/融合
// (打印 stats.wave 诊断),允许修实现但不允许放宽断言。
//
// 设计注(数据标定见 /tmp/probe-p2.mjs,真实 bge-m3 余弦):
//   query"泽塔波的原理与影响" vs A=0.65 / B=0.23 / 无关日记 0.24~0.34;
//   用 truncate=0.25 使纯路径下 B 根本不在候选集(更彻的"缺席"),wave 路径里
//   B 靠"种子/涌现 Tag 所在文件补充捞取"(豁免 truncate)被召回,分数即真实余弦。

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { createEmbedder } from '../engine/embed.mjs'
import { openStore } from '../engine/store.mjs'

const REAL = { baseUrl: 'http://127.0.0.1:11434/v1', model: 'bge-m3', dimension: 1024 }
const QUERY = '泽塔波的原理与影响'
const K = 6
const TRUNCATE = 0.25

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const readText = (f) => {
  try {
    return fs.readFileSync(f, 'utf8')
  } catch {
    return null
  }
}

async function waitPending(store, { timeout = 240000, what = '' } = {}) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeout) {
    if (store.stats().pendingFiles === 0) return store.stats()
    await sleep(200)
  }
  throw new Error(`等待索引队列排空超时(${what})，最后 stats=${JSON.stringify(store.stats())}`)
}

async function waitFor(what, pred, timeout = 240000) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeout) {
    if (pred()) return
    await sleep(200)
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

// 纯 KNN 参照实现(独立于 store,只读 chunks.jsonl):返回全部 ≥truncate 候选及其分数/排序
async function referenceKnnAll(embedder, chunksPath, query, truncate) {
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
  return scored
}

const diaryText = (body, tags) => `[2026-08-29] - dsh\n[14:30]\n${body}\n\nTag: ${tags}\n`

// ── 预置数据集:A(自造词泽塔波,Tag: 泽塔波, 澜沧计划)/ B(澜沧计划部署,Tag: 澜沧计划, 部署,
//    正文绝不出现"泽塔波"及近义词)/ 6 篇无关日记(各带 tag,撑起共现图与 EPA ≥8 条件) ──
const FILE_A = 'diaries/dsh/2026-08-29-09_00_01-泽塔波.md'
const FILE_B = 'diaries/dsh/2026-08-29-09_00_02-澜沧部署.md'
const DIARIES = [
  // A:私有概念"泽塔波"锚定在"澜沧计划"语境里(泽塔波, 澜沧计划 共现)
  [FILE_A,
    '澜沧计划的波形研究里,我们把一种全新的相位调制构想命名为泽塔波。泽塔波的原理是利用相位叠加构造更低的旁瓣,泽塔波的影响是让频谱效率在低信噪比下仍有增益。泽塔波目前只在实验室验证过。',
    '泽塔波, 澜沧计划'],
  // B:澜沧计划的部署细节,正文与查询语义疏远(不含泽塔波及近义词)
  [FILE_B,
    '澜沧计划的部署细节记录:先在边缘节点安装采集代理与监控探针,配置灰度发布与一键回滚,然后做全链路压测并观察告警阈值,最后写上线报告。',
    '澜沧计划, 部署'],
  // ── 6 篇无关日记(无结构证据,不涉及泽塔波/澜沧计划) ──
  ['diaries/dsh/2026-08-29-09_00_03-干扰抑制.md',
    '无线通信中的干扰抑制研究:比较了线性调频与相位编码两种波形的峰均比,分析抗干扰能力。',
    '通信, 波形'],
  ['diaries/dsh/2026-08-29-09_00_04-信号处理.md',
    '信号处理实验记录:用匹配滤波与脉冲压缩处理雷达回波,重点观察旁瓣与主瓣的能量比。',
    '信号处理, 实验'],
  ['diaries/dsh/2026-08-29-09_00_05-频谱感知.md',
    '频谱感知调研笔记:能量检测与循环平稳检测的对比,小样本下的检测概率分析。',
    '频谱, 调研'],
  ['diaries/dsh/2026-08-29-09_00_06-调参笔记.md',
    '神经网络训练的调参笔记:学习率衰减与批归一化对收敛速度的影响,记录实验结果。',
    '机器学习, 笔记'],
  ['diaries/dsh/2026-08-29-09_00_07-压测方案.md',
    '系统上线前的压测方案:并发量与响应时间的曲线,记录 QPS 与资源占用的变化。',
    '压测, 方案'],
  ['diaries/dsh/2026-08-29-09_00_08-告警配置.md',
    '整理了监控告警的规则表,把阈值参数统一收敛到一个配置文件,减少人工维护。',
    '监控, 配置'],
]

const WAVE_KEYS = ['alpha', 'effectiveBoost', 'logicDepth', 'resonance', 'coverage', 'activation', 'seeds', 'emergentCount', 'fieldNodes', 'graphGeneration']

async function main() {
  const base = fs.mkdtempSync(path.join('/tmp', 'vcp-memo-p2-'))
  console.log('== e2e-p2.test.mjs ==')
  console.log(`  临时根 = ${base}；查询 = "${QUERY}" k=${K} truncate=${TRUNCATE}`)
  const embedder = createEmbedder(REAL)

  try {
    // ═══════════════ ① 预置 8 篇 + 纯 KNN 实例(A 组,同库第二实例) ═══════════════
    const dataRoot = path.join(base, 'dataA')
    for (const [name, body, tags] of DIARIES) {
      const abs = path.join(dataRoot, name)
      fs.mkdirSync(path.dirname(abs), { recursive: true })
      fs.writeFileSync(abs, diaryText(body, tags))
    }
    console.log('== ① 纯 KNN 路径(tagmemo.enabled=false,同库第二实例)→ 记录 B 缺席/靠后 ==')
    const [logFnP, logsP] = collectLog()
    const storeP = await openStore(
      { dataRoot, agentName: 'dsh', watch: false, tagmemo: { enabled: false } },
      embedder,
      logFnP,
    )
    await storeP.ready
    await waitFor('EPA 训练完成', () => storeP.stats().epaTrained === true)
    const stP = await waitPending(storeP, { what: '纯 KNN 实例' })
    assert.ok(stP.indexedChunks >= 8, `indexedChunks >= 8（实际 ${stP.indexedChunks}）`)
    assert.ok(stP.tagCount >= 8, `tagCount >= 8（实际 ${stP.tagCount}）`)
    assert.ok(stP.vectorizedTags >= 8, `vectorizedTags >= 8（实际 ${stP.vectorizedTags}）`)
    assert.strictEqual(stP.epaTrained, true, 'EPA 应已训练')
    ok(`8 篇日记已索引(${stP.indexedChunks} chunks)，${stP.tagCount} 个 tag 全部向量化，EPA 已训练`)

    const pure = await storeP.recall({ query: QUERY, k: K, truncate: TRUNCATE })
    const pureFiles = pure.blocks.map((b) => b.file)
    assert.strictEqual(pure.blocks[0].file, FILE_A, `纯 KNN 首位命中 A（实际 ${pure.blocks[0].file}）`)
    assert.ok(pure.blocks[0].score > 0.5, `A 的分数应显著（实际 ${pure.blocks[0].score.toFixed(3)}）`)
    const pureHasB = pureFiles.includes(FILE_B)
    console.log(`  [观测] 纯 KNN top-${K} 序号:`)
    for (const [i, b] of pure.blocks.entries()) {
      console.log(`    [${i}] ${path.basename(b.file)} score=${b.score.toFixed(4)}`)
    }
    // P0 纯 KNN 全量参照(读 chunks.jsonl):记录 B 在完整候选排序中的秩次与分数
    const pureAll = await referenceKnnAll(embedder, path.join(dataRoot, 'index', 'chunks.jsonl'), QUERY, TRUNCATE)
    const pureRankB = pureAll.findIndex((x) => x.file === FILE_B)
    const pureScoreB = pureAll.find((x) => x.file === FILE_B)?.score ?? null
    assert.strictEqual(pureHasB, false, `纯 KNN(k=${K})中语义疏远的 B 不应在 top-${K} 前列（rank=${pureRankB}, score=${pureScoreB?.toFixed(4)}）`)
    assert.strictEqual(pureRankB, -1, 'B 连纯候选集(≥truncate)都不应进入 —— 语义疏远,纯 KNN 完全漏掉')
    console.log(`  [观测] B 纯 KNN 全量秩次=缺席(rank=${pureRankB}, score=${pureScoreB !== null ? pureScoreB.toFixed(4) : 'N/A'})；A 首位 ${pure.blocks[0].score.toFixed(4)}`)
    ok(`纯 KNN:命中 A(${pure.blocks[0].score.toFixed(3)})，B 不在候选集(score=${pureScoreB !== null ? pureScoreB.toFixed(3) : 'N/A'} < ${TRUNCATE})——A/B 对照的 B 侧成立`)

    // 低截断复核(truncate=0.2):B 在纯路径"靠后但可见"是否也不进 top-k —— 记录秩次
    const pure02 = await storeP.recall({ query: QUERY, k: K, truncate: 0.2 })
    const pure02All = await referenceKnnAll(embedder, path.join(dataRoot, 'index', 'chunks.jsonl'), QUERY, 0.2)
    const pure02HasB = pure02.blocks.some((b) => b.file === FILE_B)
    const pure02RankB = pure02All.findIndex((x) => x.file === FILE_B)
    console.log(`  [观测] truncate=0.2:纯路径 B 全量秩=${pure02RankB}/${pure02All.length}(top-${K} 内有 B? ${pure02HasB})`)
    assert.strictEqual(pure02HasB, false, `truncate=0.2 下纯路径 B 仍不在 top-${K}（靠后）`)
    assert.ok(pure02RankB >= 0, '0.2 截断下 B 应在纯候选集里(便于秩次对照)')
    await storeP.close()

    // ═══════════════ ② wave 路径(B 组,同库同查询同 k,同数据根第二实例) ═══════════════
    console.log('== ② wave 路径(同库同查询同 k)→ B 被召回且 blocks 数/排序提升 ==')
    const [logFnW, logsW] = collectLog()
    const storeW = await openStore({ dataRoot, agentName: 'dsh', watch: false, tagmemo: {} }, embedder, logFnW)
    await storeW.ready
    const stW = await waitPending(storeW, { what: 'wave 实例' })
    assert.strictEqual(stW.epaTrained, true, 'wave 实例(缓存装载)EPA 应已训练')
    const wave = await storeW.recall({ query: QUERY, k: K, truncate: TRUNCATE })
    const waveFiles = wave.blocks.map((b) => b.file)
    const waveRankB = waveFiles.indexOf(FILE_B)
    console.log('  [观测] wave blocks 序号(score 即向量余弦):')
    for (const [i, b] of wave.blocks.entries()) {
      console.log(`    [${i}] ${path.basename(b.file)} score=${b.score.toFixed(4)}`)
    }
    console.log('  [观测] stats.wave = ' + JSON.stringify(wave.stats.wave))
    assert.ok(waveRankB >= 0, 'B 必须被 wave 路径召回（泽塔波→澜沧计划→B 的结构传导）')
    // A 仍首位(融合不喧宾夺主),B 被召回,blocks 数提升
    assert.strictEqual(wave.blocks[0].file, FILE_A, `wave 首位仍应为 A（实际 ${wave.blocks[0].file}）`)
    assert.ok(wave.blocks.length > pure.blocks.length, `blocks 数提升:wave=${wave.blocks.length} > pure=${pure.blocks.length}`)
    // SPEC-P2 §6.2 修订:B 低于 truncate,应以 viaStructure 结构补位块出现(不占语义 Top-K);
    // 且总块数有界(≤ k + maxSupplement)
    const waveBlockB = wave.blocks.find((b) => b.file === FILE_B)
    assert.strictEqual(waveBlockB.viaStructure, true, 'B 应标记 viaStructure(结构传导补位)')
    assert.ok(wave.blocks.length <= K + 2, `wave blocks 有界: ${wave.blocks.length} ≤ k(${K}) + maxSupplement(2)`)
    ok(`wave:blocks=${wave.blocks.length}(纯=${pure.blocks.length}),B 被召回(秩 ${waveRankB + 1}/${wave.blocks.length},viaStructure),A 仍首位`)

    // 排序对照复核(truncate=0.2):此时 B 的融合余弦(0.234)≥ 0.2,B 是**语义候选**而非
    // 结构补充;两路径 Top-6 都不含 B(纯路径 B 秩 6/8 靠后)。为验证"B 作为语义候选真实存在
    // 且分数过阈",用 k=20 放大名额再断言——k 契约(Top-K 截断)本身在 ② 已由
    // `blocks ≤ k + maxSupplement` 断言守护。
    const wave02 = await storeW.recall({ query: QUERY, k: 20, truncate: 0.2 })
    const wave02RankB = wave02.blocks.findIndex((b) => b.file === FILE_B)
    console.log(`  [观测] truncate=0.2 (k=20 放大名额):纯路径 B 全量秩=${pure02RankB}/${pure02All.length};wave 路径 B 秩=${wave02RankB}/${wave02.blocks.length}`)
    assert.ok(wave02RankB >= 0, 'truncate=0.2 + k=20 下 wave 路径 B 作为语义候选被召回')
    assert.strictEqual(wave02.blocks.find((b) => b.file === FILE_B).viaStructure, undefined, '0.2 下 B 过阈为语义块,不带 viaStructure 标记')
    ok(`排序复核(truncate=0.2, k=20):纯路径 B 靠后(秩 ${pure02RankB + 1}/${pure02All.length});wave 路径 B 语义召回(秩 ${wave02RankB + 1}/${wave02.blocks.length},无 viaStructure)`)

    // ═══════════════ ③ 只加分不罚分:候选并集 + 分数不变负 ═══════════════
    console.log('== ③ 只加分不罚分:纯候选集(≥truncate)文件 ⊆ wave 结果,分数不变负 ==')
    const pureFilesAll = new Set(pureAll.map((x) => x.file))
    const waveFilesSet = new Set(waveFiles)
    for (const f of pureFilesAll) {
      assert.ok(waveFilesSet.has(f), `纯路径候选文件不应因 wave 消失: ${path.basename(f)}`)
    }
    for (const b of wave.blocks) {
      assert.ok(b.score >= 0, `wave 候选分数不变负（${path.basename(b.file)} score=${b.score.toFixed(4)}）`)
    }
    assert.ok(
      waveFilesSet.size >= pureFilesAll.size,
      `wave 候选是纯候选的超集（wave=${waveFilesSet.size} ≥ pure=${pureFilesAll.size}）`,
    )
    ok(`候选并集:纯候选 ${pureFilesAll.size} 个文件全部仍在 wave 结果中(${waveFilesSet.size} 个),wave 分数全部 ≥ 0`)

    // ═══════════════ ④ stats.wave 字段齐全 ═══════════════
    console.log('== ④ stats.wave 字段齐全 ==')
    const w = wave.stats.wave
    for (const key of WAVE_KEYS) {
      assert.strictEqual(typeof w[key], 'number', `stats.wave.${key} 应为数字（实际 ${typeof w[key]}）`)
      assert.ok(Number.isFinite(w[key]), `stats.wave.${key} 应有限`)
    }
    assert.ok(w.alpha >= 0 && w.alpha <= 1, `alpha ∈ [0,1]（实际 ${w.alpha}）`)
    assert.ok(w.seeds >= 1, `seeds >= 1（实际 ${w.seeds}）`)
    assert.ok(w.fieldNodes >= 1, `fieldNodes >= 1（实际 ${w.fieldNodes}）`)
    assert.ok(w.graphGeneration >= 1, `graphGeneration >= 1（实际 ${w.graphGeneration}）`)
    ok(`字段齐全:alpha=${w.alpha.toFixed(3)},effectiveBoost=${w.effectiveBoost.toFixed(3)},` +
      `logicDepth=${w.logicDepth.toFixed(2)},resonance=${w.resonance.toFixed(2)},coverage=${w.coverage.toFixed(2)},` +
      `activation=${w.activation.toFixed(3)},seeds=${w.seeds},emergentCount=${w.emergentCount},fieldNodes=${w.fieldNodes},graphGeneration=${w.graphGeneration}`)

    // ═══════════════ ⑤ matchedTags 记号 + 显式 tags Core Tag ═══════════════
    console.log('== ⑤ matchedTags 结构/记号 + 显式 tags:["澜沧计划"] → sourceType=core ==')
    assert.ok(Array.isArray(wave.matchedTags) && wave.matchedTags.length > 0, 'matchedTags 非空')
    assert.ok(wave.matchedTags.length <= 8, `matchedTags 最多 8 个（实际 ${wave.matchedTags.length}）`)
    for (const m of wave.matchedTags) {
      assert.strictEqual(typeof m.tag, 'string')
      assert.strictEqual(typeof m.weight, 'number')
      assert.strictEqual(typeof m.level, 'number')
      assert.ok(['core', 'seed', 'emergent'].includes(m.sourceType), `sourceType 合法（实际 ${m.sourceType}）`)
      assert.strictEqual(typeof m.hop, 'number')
      assert.match(m.notation, /^[=~].+@(core|seed|emergent)(:\d+)?$/, `记号格式（实际 ${m.notation}）`)
    }
    console.log('  [样例] wave 查询 matchedTags 记号(供人工排查召回质量):')
    for (const m of wave.matchedTags) console.log(`    ${m.notation}   (sourceType=${m.sourceType},hop=${m.hop})`)
    // 显式 Core Tag:tags:['澜沧计划'] 必须在 matchedTags 且 sourceType='core'
    const core = await storeW.recall({ query: QUERY, k: K, truncate: TRUNCATE, tags: ['澜沧计划'] })
    const coreEntry = core.matchedTags.find((m) => m.tag === '澜沧计划')
    assert.ok(coreEntry, '显式 tags:["澜沧计划"] 时 m.tag === "澜沧计划" 必须出现在 matchedTags')
    assert.strictEqual(coreEntry.sourceType, 'core', '该 Core Tag 的 sourceType 必须为 core')
    assert.match(coreEntry.notation, /@core$/, `其记号应以 @core 结尾（实际 ${coreEntry.notation}）`)
    assert.ok(core.blocks.some((b) => b.file === FILE_B), '显式 Core Tag 路径下 B 仍应被召回(结构传导不受影响)')
    console.log('  [样例] 显式 core 查询 matchedTags 记号:')
    for (const m of core.matchedTags) console.log(`    ${m.notation}   (sourceType=${m.sourceType},hop=${m.hop})`)
    ok('matchedTags 记号结构合法;显式 tags:["澜沧计划"] → 澜沧计划 sourceType=core')
    await storeW.close()

    // ═══════════════ ⑥ 无 Tag 库 → 回退 P1 ═══════════════
    console.log('== ⑥ 无 Tag 库(tagmemo 默认开启)→ 回退 P1:matchedTags=[], 无 stats.wave ==')
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
    const [logFnN, logsN] = collectLog()
    const storeN = await openStore({ dataRoot: dataRootB, agentName: 'dsh', watch: false, tagmemo: {} }, embedder, logFnN)
    await storeN.ready
    await waitPending(storeN, { what: '无 Tag 库' })
    assert.strictEqual(storeN.stats().tagCount, 0, '无 Tag 行 → tagCount 应 0')
    const recN = await storeN.recall({ query: '数据目录放在独立位置', k: K, truncate: 0.4 })
    assert.ok(Array.isArray(recN.blocks) && recN.blocks.length >= 1, '无 Tag 库上 recall 仍返回 blocks')
    assert.deepStrictEqual(recN.matchedTags, [], '无 Tag 库 matchedTags 应为 []')
    assert.ok(recN.stats.epa && recN.stats.epa.trained === false, 'epa.trained 应为 false')
    assert.strictEqual(recN.stats.pyramid, null, 'stats.pyramid 应为 null')
    assert.strictEqual(recN.stats.wave, undefined, '无 Tag 库不应有 stats.wave(wave 路径整体回退 P1)')
    ok('无 Tag 库优雅回退 P1:matchedTags=[], stats.wave 缺席, blocks 正常')

    // ═══════════════ ⑦ sig 不一致 → 拒绝服务 ═══════════════
    console.log('== ⑦ sig 不一致仍拒绝服务 ==')
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
    const storeC = await openStore({ dataRoot: dataRootC, agentName: 'dsh', watch: false, tagmemo: {} }, embedder, collectLog()[0])
    await assert.rejects(storeC.ready, (e) => {
      assert.match(e.message, /嵌入模型签名已变/, 'ready 拒绝,含签名变更说明')
      assert.match(e.message, /旧 bge-m3@768 → 新 bge-m3@1024/, '旧→新签名')
      return true
    })
    await assert.rejects(storeC.recall({ query: 'x' }), /嵌入模型签名已变/, 'recall 抛拒绝服务错误')
    assert.ok(!fs.existsSync(path.join(dataRootC, 'index', 'chunks.jsonl')), '拒绝服务期间不应写 chunks.jsonl')
    await storeC.close()
    ok('sig 不一致(bge-m3@768)拒绝服务:ready/recall 抛签名错误,不动盘')

    console.log(`\ne2e-p2.test.mjs: 通过 ${passed} 组断言`)
  } catch (err) {
    console.error('e2e-p2 测试失败:', err)
    process.exitCode = 1
  } finally {
    fs.rmSync(base, { recursive: true, force: true })
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})