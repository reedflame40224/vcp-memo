// tests/cli.test.mjs —— SPEC-P3 §5 第 2 条:管理 CLI 端到端验收(真 Ollama bge-m3 + 临时 dataRoot)
// 运行:node tests/cli.test.mjs(依赖真实 Ollama bge-m3 于 127.0.0.1:11434)
// 零依赖(仅 node 内置模块)。CLI 经 node:child_process spawnSync(process.execPath, ['bin/vcp-memo.mjs', ...])
// 驱动;临时 dataRoot 先用【真实 openStore】播种 3 篇日记——tag 设计正好 8 个去重标签,
// 触发 EPA 训练落盘 epa.json,让 doctor 的「epa tagHash 一致」与「无向量 tag」两项检查
// 都在有数据的真实状态下被断言。
//
// 注:store 的 tag flush 是单飞去抖,连续多篇快速 saveDiary 时"最后一批新 tag"的
// 向量化可能被丢弃(engine 既有实现行为,本次不改 engine)。播种末尾用一次锚点
// 原文替换的 updateDiary(≥15 字符 target)强制把日记重入队,队列空闲时的补 flush
// 会兜底向量化残留 tag 并训练 EPA——该路径即生产环境"编辑日记 → 重索引 → tag 层补向量"的正常流程。
//
// 逐项覆盖 §5.2:
//   ① stats 输出含关键字段且退出码 0;
//   ② tags 列出播种的 tag(名 + 有向量);
//   ③ doctor 对健康库全 ✅(退出码 0);
//   ④ rebuild 后日记数一致(stats 再看 diaries 仍 = 3)+ DSH 运行中警告;
//   ⑤ doctor 异常用例:手改 chunks.jsonl 追加一条指向不存在文件的孤儿 chunk →
//      doctor 报 ⚠️ 孤儿 chunk 且退出码 1;
//   ⑥ 非法命令 → 退出码 2 并打印用法。
// 另:断言 CLI stdout 无引擎调试噪音([EPA] 等英文行被转至 stderr),守住「输出全部中文、纯文本」。

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { createEmbedder } from '../engine/embed.mjs'
import { openStore } from '../engine/store.mjs'

const REAL = { baseUrl: 'http://127.0.0.1:11434/v1', model: 'bge-m3', dimension: 1024 }
const CLI = fileURLToPath(new URL('../bin/vcp-memo.mjs', import.meta.url))

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// 轮询等待索引队列排空(排空时 chunks.jsonl 已在同一微任务内同步落盘)
async function waitPending(store, { timeout = 240000, what = '' } = {}) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeout) {
    if (store.stats().pendingFiles === 0) return store.stats()
    await sleep(200)
  }
  throw new Error(`等待索引队列排空超时(${what})，最后 stats=${JSON.stringify(store.stats())}`)
}

// 轮询等待真实结果条件
async function waitFor(what, pred, timeout = 240000) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeout) {
    if (pred()) return
    await sleep(200)
  }
  throw new Error(`等待条件超时: ${what}`)
}

// 经 node:child_process 跑 CLI,返回 spawnSync 结果(超时提权为断言失败)
function runCli(args, opts = {}) {
  const r = spawnSync(process.execPath, [CLI, ...args], {
    encoding: 'utf8',
    timeout: 300000,
    ...opts,
  })
  assert.equal(r.error, undefined, `spawnSync 失败: ${r.error && r.error.message}`)
  assert.notEqual(r.status, null, `CLI 被信号终止(signal=${r.signal})`)
  return r
}

let passed = 0
function ok(what) {
  passed++
  console.log(`  ✓ ${what}`)
}

async function main() {
  // 临时 dataRoot(与既有测试同前缀,防误伤真实库)
  const base = fs.mkdtempSync(path.join('/tmp', 'vcp-memo-cli-test-'))
  const dataRoot = path.join(base, 'data')
  console.log(`== cli.test.mjs ==`)
  console.log(`  dataRoot = ${dataRoot}`)

  const embedder = createEmbedder(REAL)
  let store = null
  try {
    // ── 前置:Ollama bge-m3@1024 必须可用(给出明确错误而非深层断言失败)──
    const probe = await embedder.embed(['前置检查'])
    assert.ok(probe[0] && probe[0].length === 1024, '前置检查:Ollama bge-m3@1024 不可用,请先启动本地 Ollama')

    // ── ① 用真实 openStore 播种 3 篇日记(8 个去重 tag → EPA 训练门槛恰好达标)──
    console.log('== ① 播种:openStore + 3 篇日记(真 Ollama) ==')
    store = await openStore({ dataRoot, agentName: 'dsh', watch: false }, embedder, () => {})
    await store.ready
    const r1 = store.saveDiary({
      title: '澜沧计划部署窗口',
      content: '澜沧计划的部署窗口固定在每周四凌晨两点到四点。回滚预案:先切回 blue 环境,再回滚数据库快照,最后通知值班群。',
      tags: ['澜沧计划', '部署'],
    })
    const r2 = store.saveDiary({
      title: '山茶烘焙参数',
      content: '浅焙山茶的最佳参数:入豆温 175 度,一爆后发展 45 秒,出炉豆温 198 度。这个曲线能保住花香。',
      tags: ['山茶', '烘焙', '食谱'],
    })
    const r3 = store.saveDiary({
      title: '泽塔波实验记录',
      content: '今天在实验室复现了泽塔波现象:当谐振腔温度降到 4K 以下,我们自研的传感阵列捕捉到了清晰的泽塔波峰。信噪比达到 23dB。',
      tags: ['泽塔波', '实验', '安全'],
    })
    assert.ok(r1.file.startsWith('diaries/') && r2.file.startsWith('diaries/') && r3.file.startsWith('diaries/'))
    ok('3 篇日记已保存')

    const st = await waitPending(store, { what: 'cli 播种 3 篇' })
    assert.ok(st.indexedChunks >= 3, `indexedChunks >= 3(实际 ${st.indexedChunks})`)
    // 锚点原文替换的 updateDiary:强制文件重入索引队列,兜底补上最后一批可能被
    // 单飞 flush 丢弃的 tag 向量化,并触发 EPA 训练(见文件头注释)。
    const rfr = await store.updateDiary({
      target: '澜沧计划的部署窗口固定在每周四凌晨两点到四点',
      replace: '澜沧计划的部署窗口固定在每周四凌晨两点到四点',
    })
    assert.ok(rfr && rfr.replaced, `refresh 替换应成功(实际 ${JSON.stringify(rfr)})`)
    await waitPending(store, { what: 'cli refresh 重入队' })
    // 等 tag 层完成向量化 + EPA 训练,且 epa.json 真正落盘(taglayer 先置内存
    // epaTrained=true、再异步原子写 epa.json;只看内存标志可能比文件早一拍)。
    await waitFor(
      'tag 向量化 + EPA 训练完成且 epa.json 落盘',
      () => {
        const s = store.stats()
        return s.epaTrained === true && fs.existsSync(path.join(dataRoot, 'index', 'epa.json'))
      },
      240000
    )
    const st2 = store.stats()
    assert.equal(st2.diaries, 3, `diaries 应为 3(实际 ${st2.diaries})`)
    assert.equal(st2.tagCount, 8, `应恰好 8 个去重 tag(实际 ${st2.tagCount})`)
    assert.equal(st2.vectorizedTags, 8, `8 个 tag 应全部向量化(实际 ${st2.vectorizedTags})`)
    assert.equal(st2.epaTrained, true, 'EPA 应已训练(epa.json 落盘)')
    ok(`索引就绪:diaries=3, chunks=${st2.indexedChunks}, tags=8/8 有向量, EPA 已训练`)
    await store.close()
    store = null
    assert.ok(fs.existsSync(path.join(dataRoot, 'index', 'tags.jsonl')), 'tags.jsonl 已落盘')
    assert.ok(fs.existsSync(path.join(dataRoot, 'index', 'epa.json')), 'epa.json 已落盘')

    // ── ② CLI stats:含关键字段且退出码 0 ──
    console.log('== ② CLI stats ==')
    const rs = runCli(['stats', '--dataRoot', dataRoot])
    assert.equal(rs.status, 0, `stats 退出码应为 0(实际 ${rs.status}),stderr=${rs.stderr}`)
    assert.ok(rs.stdout.includes('sig: bge-m3@1024'), 'stats 含 sig 字段')
    assert.ok(rs.stdout.includes('diaries: 3'), 'stats 含 diaries: 3')
    assert.ok(rs.stdout.includes('indexedChunks:'), 'stats 含 indexedChunks 字段')
    assert.ok(rs.stdout.includes('lastRebuild:'), 'stats 含 lastRebuild 字段')
    assert.ok(rs.stdout.includes('tagCount: 8'), 'stats 含 tagCount: 8')
    assert.ok(rs.stdout.includes('vectorizedTags: 8'), 'stats 含 vectorizedTags: 8')
    assert.ok(rs.stdout.includes('epaTrained: true'), 'stats 含 epaTrained: true')
    assert.ok(rs.stdout.includes(dataRoot), 'stats 使用的 dataRoot 是传入的临时目录(防默认目录误伤)')
    ok('stats 关键字段齐备,退出码 0')

    // ── ③ CLI tags:列出播种的 tag + 有向量标记 ──
    console.log('== ③ CLI tags ==')
    const rt = runCli(['tags', '--dataRoot', dataRoot])
    assert.equal(rt.status, 0, `tags 退出码应为 0(实际 ${rt.status})`)
    assert.ok(rt.stdout.includes('共 8 个 tag'), 'tags 输出 tag 总数')
    assert.ok(rt.stdout.includes('澜沧计划'), 'tags 列出播种 tag:澜沧计划')
    assert.ok(rt.stdout.includes('泽塔波'), 'tags 列出播种 tag:泽塔波')
    assert.ok(rt.stdout.includes('食谱'), 'tags 列出播种 tag:食谱')
    assert.ok(rt.stdout.includes('有向量'), 'tags 标记有向量')
    ok('tags 按出现次数列出 tag + 向量标记')

    // ── ④ CLI doctor:健康库全 ✅,退出码 0 ──
    console.log('== ④ CLI doctor(健康库) ==')
    const rd = runCli(['doctor', '--dataRoot', dataRoot])
    assert.equal(rd.status, 0, `健康库 doctor 退出码应为 0(实际 ${rd.status})`)
    assert.ok(!rd.stdout.includes('⚠️'), '健康库不应出现任何 ⚠️')
    assert.ok(rd.stdout.includes('体检通过'), '健康库输出体检通过')
    ok('doctor 健康库全 ✅,退出码 0')

    // ── ⑤ CLI rebuild:重建完成 + DSH 警告;rebuild 后日记数一致 ──
    console.log('== ⑤ CLI rebuild ==')
    const rb = runCli(['rebuild', '--dataRoot', dataRoot])
    assert.equal(rb.status, 0, `rebuild 退出码应为 0(实际 ${rb.status})`)
    assert.ok(rb.stdout.includes('重建完成'), 'rebuild 打印结果')
    assert.ok(rb.stdout.includes('3 个文件'), 'rebuild 文件数一致')
    assert.ok(rb.stdout.includes('重启 DSH'), 'rebuild 打印 DSH 运行中的警告')
    assert.ok(!rb.stdout.includes('[EPA]'), '引擎调试噪音不进 stdout(保持中文纯文本)')
    ok('rebuild 完成,含 DSH 警告,stdout 干净')

    // rebuild 后 stats:日记数一致(§5.2「rebuild 后日记数一致」)。
    // 注:lastRebuild 是 store 进程内存值,不持久化——独立 CLI 进程打开库必然为 0,
    // 因此只核对 diaries/indexedChunks 的账面对齐。
    const rs2 = runCli(['stats', '--dataRoot', dataRoot])
    assert.equal(rs2.status, 0, `rebuild 后 stats 退出码应为 0(实际 ${rs2.status})`)
    assert.ok(rs2.stdout.includes('diaries: 3'), 'rebuild 后日记数一致')
    assert.ok(rs2.stdout.includes('indexedChunks:') && !rs2.stdout.includes('indexedChunks: 0'), 'rebuild 后索引非空')
    ok('rebuild 后 diaries 仍为 3,索引非空')

    // ── ⑥ 人为制造孤儿 chunk:手工篡改 chunks.jsonl(追加一条指向不存在文件的记录)──
    console.log('== ⑥ 篡改 chunks.jsonl 制造孤儿 chunk ==')
    const chunksPath = path.join(dataRoot, 'index', 'chunks.jsonl')
    const orphan = JSON.stringify({
      id: 99999,
      file: 'diaries/dsh/已删除的日记.md',
      chunkIndex: 0,
      content: '这条 chunk 指向的日记文件已不存在',
      vector: new Array(1024).fill(0),
    })
    fs.appendFileSync(chunksPath, orphan + '\n')
    ok('已追加 1 条孤儿 chunk(lines 含 已删除的日记.md)')

    // ── ⑦ CLI doctor 异常用例:报 ⚠️ 孤儿 chunk 且退出码 1 ──
    console.log('== ⑦ CLI doctor(孤儿 chunk 库) ==')
    const rdo = runCli(['doctor', '--dataRoot', dataRoot])
    assert.equal(rdo.status, 1, `孤儿 chunk 库 doctor 退出码应为 1(实际 ${rdo.status})`)
    assert.ok(rdo.stdout.includes('孤儿 chunk'), 'doctor 报告孤儿 chunk')
    assert.ok(rdo.stdout.includes('已删除的日记.md'), 'doctor 列出孤儿文件路径')
    assert.ok(rdo.stdout.includes('⚠️'), 'doctor 打印警示符号')
    assert.ok(rdo.stdout.includes('体检发现'), 'doctor 汇总发现问题项')
    ok('doctor 报 ⚠️ 孤儿 chunk,退出码 1')

    // ── ⑧ 非法命令 → 退出码 2 + 用法 ──
    console.log('== ⑧ 非法命令 ==')
    const rbad = runCli(['frobnicate', '--dataRoot', dataRoot])
    assert.equal(rbad.status, 2, `非法命令退出码应为 2(实际 ${rbad.status})`)
    assert.ok((rbad.stdout + rbad.stderr).includes('用法'), '非法命令打印用法')
    const rnone = runCli(['--dataRoot', dataRoot])
    assert.equal(rnone.status, 2, `缺命令退出码应为 2(实际 ${rnone.status})`)
    const rflag = runCli(['stats', '--bogus', dataRoot])
    assert.equal(rflag.status, 2, `非法参数退出码应为 2(实际 ${rflag.status})`)
    ok('非法命令/缺命令/非法参数均退出码 2 + 用法')

    // ── ⑨ dataRoot 不存在 → 友好中文错误而非 JS 堆栈,退出码 2 ──
    console.log('== ⑨ dataRoot 不存在 ==')
    const rmiss = runCli(['stats', '--dataRoot', path.join(base, 'no-such-root')])
    assert.equal(rmiss.status, 2, `不存在 dataRoot 退出码应为 2(实际 ${rmiss.status})`)
    const all = rmiss.stdout + rmiss.stderr
    assert.ok(all.includes('数据根'), '应给出友好中文错误(指出数据根不存在)')
    assert.ok(!/\.mjs:\d+/.test(all), `不应出现 JS 堆栈(实际: ${all.slice(0, 300)})`)
    ok('dataRoot 不存在:友好错误 + 退出码 2,无堆栈')

    console.log(`\ncli.test.mjs: 通过 ${passed} 组断言`)
    // 不 process.exit(会跳过 finally 清理);store 已 close,watcher 已停,事件循环自然结束。
  } catch (err) {
    console.error('cli 测试失败:', err)
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