// tests/e2e.test.mjs —— 端到端集成自测：真 Ollama bge-m3（1024 维）+ 临时 dataRoot
// 零依赖（node 直接运行）。逐项覆盖 SPEC §10 第 4 条的 P0 验收：
//   ① saveDiary 3 篇（含可被语义命中的文本）；
//   ② 等索引队列排空（轮询 stats().pendingFiles === 0）；
//   ③ recall({query:'浪潮算法讨论'}) 命中对应 chunk 且 score > 0.4，
//      另加语义命中质量："浪潮算法" 查询召回含"上周五和徕拉讨论了 TagMemo 的浪潮算法"的片段 score > 0.4；
//   ④ fs 直接改写其中一篇正文为别的主题，等 3s：原查询不再把该 chunk 排首位、新主题查询能命中；
//   ⑤ 删掉 index/ 后重新 openStore → 自动全量重建，recall 仍正确；
//   ⑥ 用错误 sig 的 embedder 打开同一 dataRoot → 拒绝服务错误且 chunks.jsonl 未被改动。
// 注：真 bge-m3 对不相关中文片段也有 ~0.3-0.5 的基线余弦；因此"编辑后原查询不再排首位"
//    用 truncate=0.3 复核首位归属（保留全部候选再做排序断言），比依赖 0.4 阈值更严谨。

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { createEmbedder } from '../engine/embed.mjs'
import { openStore } from '../engine/store.mjs'

const REAL = { baseUrl: 'http://127.0.0.1:11434/v1', model: 'bge-m3', dimension: 1024 }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const readText = (f) => {
  try {
    return fs.readFileSync(f, 'utf8')
  } catch {
    return null
  }
}

// 轮询等待索引队列排空（排空时 chunks.jsonl 已在同一微任务内同步落盘）
async function waitPending(store, { timeout = 60000, what = '' } = {}) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeout) {
    if (store.stats().pendingFiles === 0) return store.stats()
    await sleep(100)
  }
  throw new Error(`等待索引队列排空超时(${what})，最后 stats=${JSON.stringify(store.stats())}`)
}

// 轮询等待真实结果条件（watcher 驱动的变化要先过 500ms 事件去抖）
async function waitFor(what, pred, timeout = 60000) {
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

async function main() {
  // 临时 dataRoot：按规格注释的命名前缀创建
  const base = fs.mkdtempSync(path.join('/tmp', 'vcp-memo-test-'))
  const dataRoot = path.join(base, 'data')
  console.log(`== e2e.test.mjs ==`)
  console.log(`  dataRoot = ${dataRoot}`)

  const embedder = createEmbedder(REAL) // 真 Ollama bge-m3 @ 1024
  const [logFn, logs] = collectLog()
  const config = { dataRoot, agentName: 'dsh', watch: true, chunker: undefined }

  const d1 = '上周五和徕拉讨论了 TagMemo 的浪潮算法，要点是传播核与残差金字塔，结论是可以先做暴力 KNN。'
  const d2 = '今天决定把 vcp-memo 的数据目录放在独立位置，与插件目录分离，并用 JSONL 作为索引格式。'
  const d3 = '晚餐吃了红烧肉配米饭和青菜汤，明天计划晨跑五公里并写周报。'

  let store = null
  try {
    // ── ① saveDiary 3 篇 ──
    console.log('== ① saveDiary 3 篇（真 Ollama 落盘 + 入队）==')
    store = await openStore(config, embedder, logFn)
    await store.ready
    const r1 = store.saveDiary({ title: '浪潮算法讨论', content: d1, tags: ['TagMemo，浪潮算法', '算法'] })
    const r2 = store.saveDiary({ title: '数据目录规划', content: d2, tags: ['架构', '存储'] })
    const r3 = store.saveDiary({ title: '饮食', content: d3, tags: ['生活'] })
    assert.ok(r1.file.endsWith('.md') && r1.file.startsWith('diaries/'), `r1 文件路径合法: ${r1.file}`)
    assert.ok(r2.file.endsWith('.md') && r3.file.endsWith('.md'))
    assert.ok(fs.existsSync(path.join(dataRoot, r1.file)), '日记文件已落盘')
    assert.ok(readText(path.join(dataRoot, r1.file)).includes(d1), '正文原样保留')
    ok(`3 篇已保存：${path.basename(r1.file)} / ${path.basename(r2.file)} / ${path.basename(r3.file)}`)

    // ── ② 等索引队列排空 ──
    console.log('== ② 队列排空 + 索引与 meta 一致 ==')
    const st = await waitPending(store, { what: 'e2e 初始 3 篇' })
    assert.ok(st.indexedChunks >= 3, `indexedChunks >= 3（实际 ${st.indexedChunks}）`)
    const meta = JSON.parse(readText(path.join(dataRoot, 'index', 'meta.json')))
    assert.equal(meta.sig, 'bge-m3@1024')
    assert.equal(meta.dimension, 1024)
    assert.equal(meta.chunkCount, st.indexedChunks, 'meta.chunkCount 与索引一致')
    ok(`队列排空，${st.indexedChunks} chunks 已索引，meta sig=bge-m3@1024`)

    // ── ③ recall 命中 + 语义命中质量 ──
    console.log('== ③ recall 语义命中（score > 0.4）==')
    const recA = await store.recall({ query: '浪潮算法讨论', k: 6, truncate: 0.4 })
    assert.ok(recA.blocks.length >= 1, '应命中至少一块')
    assert.equal(recA.blocks[0].file, r1.file, `首位命中浪潮算法日记（实际 ${recA.blocks[0].file}）`)
    assert.ok(recA.blocks[0].score > 0.4, `score > 0.4（实际 ${recA.blocks[0].score}）`)
    assert.ok(recA.blocks[0].text.includes('浪潮算法'), '命中文本含查询主题')
    // 语义命中质量：短查询 "浪潮算法" 也能召回该片段且 score > 0.4
    const recB = await store.recall({ query: '浪潮算法', k: 6, truncate: 0.4 })
    assert.equal(recB.blocks[0].file, r1.file, `"浪潮算法" 首位命中（实际 ${recB.blocks[0].file}）`)
    assert.ok(recB.blocks[0].score > 0.4, `"浪潮算法" score > 0.4（实际 ${recB.blocks[0].score}）`)
    assert.ok(recB.blocks[0].text.includes('上周五和徕拉讨论了 TagMemo 的浪潮算法'), '召回文本即目标片段')
    ok(`recall("浪潮算法讨论")=${recA.blocks[0].score.toFixed(3)}；recall("浪潮算法")=${recB.blocks[0].score.toFixed(3)}，均命中 r1 且 > 0.4`)

    // ── ④ fs 直接改写一篇正文为别的主题，等 3s ──
    console.log('== ④ 人直接编辑 → watcher 重索引 ==')
    const newContent = '计划下个月去云南旅行，先订机票再订民宿，行程安排在大理和丽江各待三天。'
    fs.writeFileSync(
      path.join(dataRoot, r1.file),
      `[2026-08-29] - dsh\n[14:30]\n${newContent}\n\nTag: 旅行\n`
    )
    const t_edit = Date.now()
    await sleep(3000) // 规格：等 3s（覆盖 500ms 事件去抖 + 重索引）
    // 以真实结果为准继续轮询：jsonl 已更新为新内容、旧内容消失、且队列排空。
    // 注意：不能用 !jl.includes('浪潮算法') —— 文件名里就含"浪潮算法"（file 字段会带），
    // 必须针对旧正文文本本身判断。
    await waitFor(
      '编辑后重索引完成',
      () => {
        const jl = readText(path.join(dataRoot, 'index', 'chunks.jsonl')) || ''
        return (
          store.stats().pendingFiles === 0 &&
          jl.includes('云南') &&
          !jl.includes('上周五和徕拉讨论了 TagMemo 的浪潮算法')
        )
      },
      60000
    )
    assert.ok(Date.now() - t_edit >= 3000, '确实等待了 3s')
    // 原查询（truncate=0.3 保留全部候选，验证排序）不再把该 chunk 排首位
    const recOld = await store.recall({ query: '浪潮算法讨论', k: 6, truncate: 0.3 })
    assert.ok(recOld.blocks.length >= 1, '原查询仍有候选')
    assert.notEqual(recOld.blocks[0].file, r1.file, `原查询首位不再是编辑过的日记（实际 ${recOld.blocks[0].file}）`)
    assert.ok(!recOld.blocks.some((b) => b.text.includes('浪潮算法')), '原内容片段已从结果中消失')
    // 新主题查询能命中它
    const recNew = await store.recall({ query: '云南旅行', k: 6, truncate: 0.4 })
    assert.equal(recNew.blocks[0].file, r1.file, `新主题查询首位命中编辑过的日记（实际 ${recNew.blocks[0].file}）`)
    assert.ok(recNew.blocks[0].score > 0.4, `新主题 score > 0.4（实际 ${recNew.blocks[0].score}）`)
    assert.ok(recNew.blocks[0].text.includes('云南'), '命中文本为新内容')
    ok(`编辑后：原查询首位=${path.basename(recOld.blocks[0].file)}（非 r1）；新查询"云南旅行"=${recNew.blocks[0].score.toFixed(3)} 命中 r1`)

    // ── ⑤ 删掉 index/ 后重新 openStore → 自动全量重建 ──
    console.log('== ⑤ 删除 index/ → 自动全量重建 ==')
    await store.close()
    store = null
    fs.rmSync(path.join(dataRoot, 'index'), { recursive: true, force: true })
    const [logFn5, logs5] = collectLog()
    const store5 = await openStore(config, embedder, logFn5)
    await store5.ready
    assert.ok(store5.stats().indexedChunks >= 3, '重建后索引非空')
    assert.ok(logs5.join('\n').includes('重建'), '日志记录了自动全量重建')
    const rec5a = await store5.recall({ query: '云南旅行', k: 6, truncate: 0.4 })
    assert.equal(rec5a.blocks[0].file, r1.file, `重建后新主题仍命中 r1（实际 ${rec5a.blocks[0].file}）`)
    assert.ok(rec5a.blocks[0].score > 0.4, `重建后 score > 0.4（实际 ${rec5a.blocks[0].score}）`)
    const rec5b = await store5.recall({ query: '数据目录放在独立位置', k: 6, truncate: 0.4 })
    assert.equal(rec5b.blocks[0].file, r2.file, `重建后命中 r2（实际 ${rec5b.blocks[0].file}）`)
    assert.ok(rec5b.blocks[0].score > 0.4, `r2 score > 0.4（实际 ${rec5b.blocks[0].score}）`)
    ok(`重建完成：${store5.stats().indexedChunks} chunks；recall("云南旅行")=${rec5a.blocks[0].score.toFixed(3)} 命中 r1，recall("数据目录")=${rec5b.blocks[0].score.toFixed(3)} 命中 r2`)

    // ── ⑥ 错误 sig 的 embedder 打开同一 dataRoot → 拒绝服务且不动盘 ──
    console.log('== ⑥ 错误 sig → 拒绝服务 + chunks.jsonl 未被改动 ==')
    await store5.close()
    const chunksBefore = fs.readFileSync(path.join(dataRoot, 'index', 'chunks.jsonl'))
    const metaBefore = fs.readFileSync(path.join(dataRoot, 'index', 'meta.json'))
    const badEmbedder = createEmbedder({ ...REAL, dimension: 768 }) // sig = bge-m3@768 ≠ bge-m3@1024
    const storeB = await openStore(config, badEmbedder, logFn)
    await assert.rejects(storeB.ready, (e) => {
      assert.match(e.message, /嵌入模型签名已变/, 'ready 拒绝，含签名变更说明')
      assert.match(e.message, /旧 bge-m3@1024 → 新 bge-m3@768/, '旧→新签名')
      return true
    })
    await assert.rejects(storeB.recall({ query: 'x' }), /嵌入模型签名已变/, 'recall 抛拒绝服务错误')
    await storeB.close()
    assert.equal(
      fs.readFileSync(path.join(dataRoot, 'index', 'chunks.jsonl')).toString(),
      chunksBefore.toString(),
      'chunks.jsonl 未被改动',
    )
    assert.equal(
      fs.readFileSync(path.join(dataRoot, 'index', 'meta.json')).toString(),
      metaBefore.toString(),
      'meta.json 未被改动',
    )
    ok('错误 sig（bge-m3@768）拒绝服务，chunks.jsonl 与 meta.json 字节级未改动')

    console.log(`\ne2e.test.mjs: 通过 ${passed} 组断言`)
    // 不用 process.exit：它会跳过 finally 的清理（watcher 关闭 + 临时目录删除）；
    // 直接返回即可，store.close() 已关闭所有句柄，事件循环自然结束。
  } catch (err) {
    console.error('e2e 测试失败:', err)
    process.exitCode = 1
  } finally {
    if (store) await store.close().catch(() => {})
    fs.rmSync(base, { recursive: true, force: true })
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})