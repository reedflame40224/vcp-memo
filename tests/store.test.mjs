// tests/store.test.mjs — engine/store.mjs 集成自测（零依赖，node 直接运行）
// 覆盖 SPEC §6/§7：落盘格式 / 索引管线 / watcher 重索引 / 删除 / meta 缺失重建 /
// sig 不一致拒绝服务 / rebuild / updateDiary 三个错误分支与恰好一处替换。
// 注入"假 embedder"：把字符码累加进 8 个桶（确定性、语义无关，只验证管线）。
// 注：chunk 内容是整篇日记文件文本（含头部与 Tag 行，与 §3 落盘格式一致），
// 因此"精确命中"用文件的完整文本（trim 后）作为 query。
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { openStore } from '../engine/store.mjs'

const ROOT = `/tmp/vcp-memo-store-test-${process.pid}`
const DATA = path.join(ROOT, 'data-main')
const DATA2 = path.join(ROOT, 'data-aux')

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// 确定性假 embedder：字符码累加进 dimension 个桶（不做归一化，store 负责归一化）
function makeEmbedder(sig, opts = {}) {
  const dim = opts.dimension ?? 8
  return {
    sig,
    model: String(sig).split('@')[0],
    dimension: dim,
    async embed(texts) {
      return texts.map((t) => {
        const v = new Float32Array(dim)
        for (const ch of String(t ?? '')) v[ch.codePointAt(0) % dim] += ch.codePointAt(0)
        return v
      })
    },
  }
}

function collectLog() {
  const logs = []
  return [(l, m) => logs.push(`${l}:${m}`), logs]
}

// 轮询等待索引队列排空（排空时 chunks.jsonl 已在同一微任务内同步落盘）
async function waitPending(store, { timeout = 20000, what = '' } = {}) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeout) {
    if (store.stats().pendingFiles === 0) return store.stats()
    await sleep(50)
  }
  throw new Error(`等待索引队列排空超时(${what})，最后 stats=${JSON.stringify(store.stats())}`)
}

// 轮询等待任意条件成立（watcher 驱动的变化要先等 500ms 事件去抖，pendingFiles
// 在入队前一直为 0，因此不能只看队列，要轮询真实结果条件）
async function waitFor(what, pred, timeout = 20000) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeout) {
    if (pred()) return
    await sleep(50)
  }
  throw new Error(`等待条件超时: ${what}`)
}

// 当前本地日期 YYYY-MM-DD（与 store 落盘一致）
function todayStr() {
  const d = new Date()
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

const readText = (file) => {
  try {
    return fs.readFileSync(file, 'utf8')
  } catch {
    return null
  }
}

const readLines = (file) => {
  const t = readText(file)
  return t ? t.split('\n').filter(Boolean) : []
}

async function main() {
  fs.rmSync(ROOT, { recursive: true, force: true })
  fs.mkdirSync(DATA, { recursive: true })

  const embA = makeEmbedder('fake-a@8')
  const config = { dataRoot: DATA, agentName: 'dsh', watch: true, chunker: { maxTokens: 8000, overlapTokens: 800 } }

  // ---------- T1: saveDiary 落盘格式（§3） ----------
  console.log('== T1: saveDiary 落盘格式 ==')
  const storeT = await openStore(config, embA, collectLog()[0])
  await storeT.ready

  const d1 = '上周五和徕拉讨论了 TagMemo 的浪潮算法，要点是传播核与残差金字塔，结论是可以先做暴力 KNN。'
  const d2 = '今天决定把 vcp-memo 的数据目录放在独立位置，与插件目录分离，并用 JSONL 作为索引格式。'
  const d3 = '晚餐吃了红烧肉配米饭和青菜汤，明天计划晨跑五公里并写周报。'

  const r1 = storeT.saveDiary({ title: '测试 标题/1', content: d1, tags: [' 标签甲，标签乙、标签丙 ', '标签丁'] })
  assert.match(r1.file, /^diaries\/dsh\/\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2}-[^/]*\.md$/, '文件名格式 HH_MM_SS')
  assert.ok(!r1.file.includes('\\') && !r1.file.includes(':'), '文件名无路径非法字符')
  assert.ok(path.basename(r1.file).includes('测试 标题1'), '标题中非法字符（/）已移除')
  assert.ok(Number.isInteger(r1.chunks) && r1.chunks >= 1, '返回 chunk 数量')
  assert.ok(fs.existsSync(path.join(DATA, r1.file)), '日记文件已落盘')
  const t1Lines = readText(path.join(DATA, r1.file)).split('\n')
  assert.equal(t1Lines[0], `[${todayStr()}] - dsh`, '第 1 行 [YYYY-MM-DD] - agentName')
  assert.match(t1Lines[1], /^\[\d{2}:\d{2}\]$/, '第 2 行 [HH:MM]')
  assert.ok(readText(path.join(DATA, r1.file)).includes(d1), '正文原样保留')
  const tagLine = t1Lines.find((l) => l.startsWith('Tag: '))
  assert.equal(tagLine, 'Tag: 标签甲, 标签乙, 标签丙, 标签丁', 'Tag 行清洗：全角逗号/顿号→", "、保序')

  // Tag 纪律：不排序、不去重（content=d2，二次利用）
  const r2 = storeT.saveDiary({ title: '重复标签', content: d2, tags: ['乙', '甲', '乙'] })
  const t2Text = readText(path.join(DATA, r2.file))
  assert.ok(t2Text.includes('Tag: 乙, 甲, 乙'), 'Tag 保序且不去重')

  // 正文自带 Tag 行：参数优先，正文行原样保留
  const r3 = storeT.saveDiary({ title: '自带标签', content: '正文第一句。\nTag: 模型自带标签行', tags: ['参数标签'] })
  const t3Text = readText(path.join(DATA, r3.file))
  assert.ok(t3Text.includes('Tag: 模型自带标签行'), '正文自带 Tag 行原样保留')
  assert.ok(t3Text.endsWith('Tag: 参数标签\n'), '以参数 tags 的 Tag 行结尾')

  const r4 = storeT.saveDiary({ title: '饮食', content: d3, tags: ['饮食'] })

  // 空 tags 报错
  assert.throws(() => storeT.saveDiary({ title: 'x', content: 'y', tags: [] }), /tags/)
  await waitPending(storeT, { what: 'T1 多篇保存' })

  // ---------- T1b: 同一秒冲突追加 -1 后缀 ----------
  console.log('== T1b: 同秒文件冲突后缀 ==')
  const storeS = await openStore({ dataRoot: DATA2, agentName: 'dsh', watch: false, chunker: config.chunker }, embA)
  await storeS.ready
  let suffixOk = false
  for (let i = 0; i < 5 && !suffixOk; i++) {
    const a = storeS.saveDiary({ title: '同名', content: `后缀测试内容甲-${i}`, tags: ['t1'] })
    const b = storeS.saveDiary({ title: '同名', content: `后缀测试内容乙-${i}`, tags: ['t2'] })
    const base = path.basename(a.file).replace(/\.md$/, '')
    if (path.basename(b.file) === `${base}-1.md`) suffixOk = true
  }
  assert.ok(suffixOk, '同一秒保存应追加 -1 后缀')
  await storeS.close()

  // ---------- T2: 队列排空后 recall + 索引文件正确 ----------
  console.log('== T2: recall 管线 + chunks.jsonl/meta.json ==')
  const st2 = await waitPending(storeT, { what: 'T2 前' })
  assert.ok(st2.indexedChunks >= 4, `已索引 chunk 数 >= 4（实际 ${st2.indexedChunks}）`)

  const jsonlLines = readLines(path.join(DATA, 'index', 'chunks.jsonl'))
  assert.ok(jsonlLines.length >= 4, 'chunks.jsonl 有内容')
  for (const line of jsonlLines) {
    const o = JSON.parse(line) // 每行必须合法 JSON
    assert.equal(typeof o.id, 'number')
    assert.equal(typeof o.file, 'string')
    assert.equal(typeof o.chunkIndex, 'number')
    assert.equal(typeof o.content, 'string')
    assert.equal(o.vector.length, 8, '向量维度 = 8')
    assert.ok(Number.isFinite(o.vector[0]), '向量是有限数值')
  }
  const meta = JSON.parse(readText(path.join(DATA, 'index', 'meta.json')))
  assert.equal(meta.sig, 'fake-a@8')
  assert.equal(meta.dimension, 8)
  assert.equal(meta.chunkCount, jsonlLines.length, 'meta.chunkCount 与行数一致')
  assert.equal(typeof meta.updatedAt, 'number')

  // query 用文件全文（trim 后 = chunk 内容）→ 嵌入完全相同 → 余弦 ≈ 1，首位命中
  const fileText1 = readText(path.join(DATA, r1.file)).trim()
  const rec1 = await storeT.recall({ query: fileText1, k: 6, truncate: 0 })
  assert.ok(rec1.blocks.length >= 1, 'recall 返回 block')
  assert.equal(rec1.blocks[0].file, r1.file, 'query 与 chunk 内容相同 → 首位命中该文件')
  assert.ok(rec1.blocks[0].score > 0.99, `score ≈ 1（实际 ${rec1.blocks[0].score}）`)
  assert.ok(rec1.blocks[0].text.includes('浪潮算法'), '返回文本包含日记内容')
  assert.equal(rec1.stats.indexedChunks, st2.indexedChunks)
  assert.equal(typeof rec1.stats.candidates, 'number')
  assert.equal(typeof rec1.stats.ms, 'number')
  // truncate 过滤：阈值 >1 时全部过滤
  const recEmpty = await storeT.recall({ query: fileText1, k: 6, truncate: 1.01 })
  assert.equal(recEmpty.blocks.length, 0, 'truncate 高于最大相似度时过滤全部')

  // ---------- T3: watcher 对外部改动触发重索引 ----------
  console.log('== T3: 外部改写日记 → watcher 重索引 ==')
  const newD1 = '量子计算的容错阈值与表面码纠错是当前研究热点，我们开了三小时讨论会。'
  // 模拟人直接编辑：原地覆写（单文件 watch 兜底；原子替换走目录 rename 事件同样覆盖）
  fs.writeFileSync(path.join(DATA, r1.file), `[${todayStr()}] - dsh\n[10:00]\n${newD1}\n\nTag: 新标签\n`)
  // watcher 驱动：等去抖 + 重索引 + 落盘完成（以 jsonl 出现新内容为准）
  await waitFor('T3 外部改写重索引', () => {
    const st = storeT.stats()
    const jl = readLines(path.join(DATA, 'index', 'chunks.jsonl'))
    return st.pendingFiles === 0 && jl.some((l) => l.includes('表面码')) && !jl.some((l) => l.includes('浪潮算法'))
  })
  assert.equal(storeT.stats().indexedChunks, st2.indexedChunks, '改写不改变 chunk 总数（内容仍单块）')
  const newFileText = readText(path.join(DATA, r1.file)).trim()
  const recNew = await storeT.recall({ query: newFileText, k: 6, truncate: 0 })
  const hitNew = recNew.blocks.find((b) => b.file === r1.file)
  assert.ok(hitNew, '新内容查询能命中该文件')
  assert.ok(hitNew.text.includes('表面码'), '命中块是新内容')
  assert.ok(!hitNew.text.includes('浪潮算法'), '旧内容已被替换')
  const jsonlAfterEdit = readLines(path.join(DATA, 'index', 'chunks.jsonl'))
  assert.ok(jsonlAfterEdit.some((l) => l.includes('表面码')), 'chunks.jsonl 已更新为新内容')
  assert.ok(!jsonlAfterEdit.some((l) => l.includes('浪潮算法')), '旧内容不再存在于 chunks.jsonl')

  // ---------- T4: 删除日记 → chunk 从索引与 jsonl 消失 ----------
  console.log('== T4: 删除日记 → chunk 移除 ==')
  fs.unlinkSync(path.join(DATA, r1.file))
  await waitFor('T4 删除后移除索引', () => {
    const st = storeT.stats()
    const jl = readLines(path.join(DATA, 'index', 'chunks.jsonl'))
    return st.pendingFiles === 0 && st.indexedChunks === st2.indexedChunks - 1 && !jl.some((l) => l.includes(r1.file))
  })
  assert.equal(storeT.stats().indexedChunks, st2.indexedChunks - 1, '删除后 chunk 数 -1')
  const fileText2 = readText(path.join(DATA, r2.file)).trim()
  const recAfterDel = await storeT.recall({ query: fileText2, k: 6, truncate: 0 })
  assert.ok(recAfterDel.blocks.length >= 1)
  assert.equal(recAfterDel.blocks[0].file, r2.file, '剩余日记仍可命中')
  assert.ok(recAfterDel.blocks.every((b) => b.file !== r1.file), '被删文件不再出现在 recall 中')
  const jsonlAfterDel = readLines(path.join(DATA, 'index', 'chunks.jsonl'))
  assert.ok(!jsonlAfterDel.some((l) => l.includes(r1.file)), 'chunks.jsonl 不含被删文件')
  const metaAfterDel = JSON.parse(readText(path.join(DATA, 'index', 'meta.json')))
  assert.equal(metaAfterDel.chunkCount, jsonlAfterDel.length)

  // ---------- T5: 删除 index/ 后重新 openStore → 自动全量重建 ----------
  console.log('== T5: 删除 index/ 后自动重建 ==')
  await storeT.close()
  fs.rmSync(path.join(DATA, 'index'), { recursive: true, force: true })
  const log5 = collectLog()
  const store5 = await openStore(config, embA, log5[0])
  await store5.ready
  assert.ok(store5.stats().indexedChunks >= 3, '重建后索引非空')
  assert.ok(fs.existsSync(path.join(DATA, 'index', 'chunks.jsonl')), 'chunks.jsonl 已重建')
  assert.ok(log5[1].join('\n').includes('重建'), '日志记录了自动重建')
  const rec5 = await store5.recall({ query: fileText2, k: 6, truncate: 0 })
  assert.ok(rec5.blocks.length >= 1 && rec5.blocks[0].file === r2.file, '重建后 recall 仍命中已存日记')
  assert.ok(rec5.blocks[0].score > 0.99, `重建后命中 score ≈ 1（实际 ${rec5.blocks[0].score}）`)

  // ---------- T6: 不同 sig 的 embedder 打开同一 dataRoot → 拒绝服务且不动盘 ----------
  console.log('== T6: sig 不一致拒绝服务 ==')
  const chunksBytesBefore = fs.readFileSync(path.join(DATA, 'index', 'chunks.jsonl'))
  const metaBytesBefore = fs.readFileSync(path.join(DATA, 'index', 'meta.json'))
  const embB = makeEmbedder('fake-b@8') // 仅 sig 不同
  const storeB = await openStore(config, embB, collectLog()[0])
  await assert.rejects(storeB.ready, (e) => {
    assert.match(e.message, /嵌入模型签名已变/, 'ready 拒绝，消息含签名变更说明')
    assert.match(e.message, /旧 fake-a@8 → 新 fake-b@8/, '旧→新签名')
    return true
  })
  await assert.rejects(storeB.recall({ query: 'x' }), /嵌入模型签名已变/, 'recall 抛拒绝服务错误')
  assert.throws(
    () => storeB.saveDiary({ title: '拒绝', content: '内容', tags: ['t'] }),
    /嵌入模型签名已变/,
    'saveDiary 抛拒绝服务错误',
  )
  await storeB.close()
  assert.equal(
    fs.readFileSync(path.join(DATA, 'index', 'chunks.jsonl')).toString(),
    chunksBytesBefore.toString(),
    'chunks.jsonl 未被改动',
  )
  assert.equal(
    fs.readFileSync(path.join(DATA, 'index', 'meta.json')).toString(),
    metaBytesBefore.toString(),
    'meta.json 未被改动',
  )

  // ---------- T7: rebuild() 后 recall 仍正确 ----------
  console.log('== T7: rebuild() 后 recall 仍正确 ==')
  await store5.rebuild()
  assert.ok(store5.stats().lastRebuild > 0, 'lastRebuild 已更新')
  const fileText4 = readText(path.join(DATA, r4.file)).trim()
  const rec7 = await store5.recall({ query: fileText4, k: 6, truncate: 0 })
  assert.ok(rec7.blocks.length >= 1 && rec7.blocks[0].file === r4.file, 'rebuild 后 recall 仍命中')
  assert.ok(rec7.blocks[0].text.includes('红烧肉'), '命中内容正确')
  const jsonl7 = readLines(path.join(DATA, 'index', 'chunks.jsonl'))
  assert.equal(jsonl7.length, store5.stats().indexedChunks, 'rebuild 后 jsonl 与索引一致')
  await store5.close()

  // ---------- T8(延伸项 §7): updateDiary ----------
  console.log('== T8: updateDiary（延伸项）==')
  const storeU = await openStore({ dataRoot: DATA2, agentName: 'dsh', watch: true, chunker: config.chunker }, embA)
  await storeU.ready
  const TARGET = '这是一段超过十五个字符的替换目标文本片段啊'
  assert.ok(TARGET.length >= 15)

  // target 过短报错
  await assert.rejects(storeU.updateDiary({ target: '短', replace: 'x' }), /15/, 'target 必须 ≥15 字符')

  // 多处命中 → 报错
  storeU.saveDiary({ title: '多处', content: `开头${TARGET}中间${TARGET}结尾`, tags: ['u1'] })
  await waitPending(storeU, { what: 'T8 多处保存' })
  await assert.rejects(
    storeU.updateDiary({ target: TARGET, replace: 'REPLACED' }),
    /命中.*处|多处/,
    '多处命中必须报错',
  )

  // 找不到 → 报错
  await assert.rejects(
    storeU.updateDiary({ target: '这是一个不存在的超长目标文本片段啊', replace: 'x' }),
    /未找到/,
    '找不到必须报明确错误',
  )

  // 恰好一处 → 替换成功且四段式保留
  const rU = storeU.saveDiary({ title: '唯一', content: `前缀保持${TARGET}后缀保持`, tags: ['u2'] })
  await waitPending(storeU, { what: 'T8 唯一保存' })
  const up = await storeU.updateDiary({ target: TARGET, replace: 'REPLACED_ONE' })
  assert.equal(up.file, rU.file, 'updateDiary 返回被改文件')
  const upText = readText(path.join(DATA2, rU.file))
  assert.ok(upText.includes('REPLACED_ONE'), 'target 已被替换')
  assert.ok(!upText.includes(TARGET), 'target 不再存在')
  assert.ok(upText.includes('前缀保持') && upText.includes('后缀保持'), '其余内容未变（禁止整文件覆写）')
  const upLines = upText.split('\n')
  assert.match(upLines[0], /^\[\d{4}-\d{2}-\d{2}\] - dsh$/, '头部保留')
  assert.match(upLines[1], /^\[\d{2}:\d{2}\]$/, '时间行保留')
  assert.ok(upText.endsWith('Tag: u2\n'), 'Tag 行保留')
  // 替换后走同一索引队列：轮询排空并验证 recall
  await waitPending(storeU, { what: 'T8 替换后重索引' })
  const upFileText = readText(path.join(DATA2, rU.file)).trim()
  const recU = await storeU.recall({ query: upFileText, k: 6, truncate: 0 })
  const hitU = recU.blocks.find((b) => b.file === rU.file)
  assert.ok(hitU, '替换后 recall 命中')
  assert.ok(hitU.text.includes('REPLACED_ONE'), '索引内容已更新')
  assert.ok(!hitU.text.includes(TARGET), '索引中旧 target 已消失')
  await storeU.close()

  console.log('ALL STORE TESTS PASSED')
  fs.rmSync(ROOT, { recursive: true, force: true })
}

main().catch((err) => {
  console.error('测试失败:', err)
  process.exitCode = 1
})