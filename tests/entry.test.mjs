// tests/entry.test.mjs —— 插件入口 vcp-memo.mjs 自测（规格 §8 / P0 验收 + P3 §3 update_memory 用例）
// 用法：node tests/entry.test.mjs（依赖真实 Ollama bge-m3 于 127.0.0.1:11434）
// 用假 ctx（tools.register 收集 + effect 立即执行 + logger=console）import 入口并执行 apply，
// 断言四工具注册且 parameters schema 合法；用临时 dataRoot + 真 Ollama 配置验证全流程不抛异常；
// 并覆盖"索引 sig 不一致 → 工具照常注册、调用返回拒绝服务文本"的 §8 重点。

import assert from 'node:assert/strict'
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { apply, name, inject } from '../vcp-memo.mjs'

const REAL = { baseUrl: 'http://127.0.0.1:11434/v1', model: 'bge-m3', dimension: 1024 }

let passed = 0
let failed = 0
async function test(title, fn) {
  try {
    await fn()
    passed++
    console.log(`  ✔ ${title}`)
  } catch (err) {
    failed++
    console.error(`  ✘ ${title}\n    ${err?.stack ?? err}`)
  }
}

// 假 ctx（用户约定的形状）：register 收集到数组；effect 立即执行；logger 用 console。
// P1.5 §3：入口新增 ctx.on('agent/pre-step') 与 ctx.get('systemPrompt')（可选服务，
// 缺失记 log 跳过），假 ctx 补上这两个方法以保持 apply 可跑。
function fakeCtx() {
  const registered = []
  const listeners = []
  const ctx = {
    tools: { register(t) { registered.push(t) } },
    effect(fn) { fn() },
    on(name, listener) {
      listeners.push({ name, listener })
      return () => {}
    },
    get() { return undefined }, // systemPrompt 缺失分支（记 log 跳过 section）
    logger: console,
  }
  return { ctx, registered, listeners }
}

function toolByName(registered, n) {
  const t = registered.find((x) => x.name === n)
  assert.ok(t, `应注册工具 ${n}`)
  return t
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function main() {
  console.log('== entry.test.mjs ==')
  const base = await mkdtemp(path.join(os.tmpdir(), 'vcp-memo-entry-'))

  // 假 ctx 的收集结果：happy 路径注册的工具（用例 2 写入，后续端到端复用）
  let tools = null

  try {
    // ── 用例 1：导出契约 ──
    await test('导出 name/inject/apply', () => {
      assert.strictEqual(name, 'vcp-memo')
      assert.ok(Array.isArray(inject) && inject.includes('tools'), 'inject 应包含 tools')
      assert.strictEqual(typeof apply, 'function')
    })

    // ── 用例 2：假 ctx + 真 Ollama → apply 全流程不抛异常，注册 4 个工具且 schema 合法 ──
    await test('apply 不抛异常：注册 save/recall/admin/update 四工具且 parameters/output 合法', async () => {
      const { ctx, registered } = fakeCtx()
      await apply(ctx, {
        dataRoot: path.join(base, 'happy'),
        agentName: 'dsh',
        embedding: REAL,
        memory: { kDefault: 6, truncate: 0.4 },
        watch: false, // 测试进程不挂 watcher，避免残留句柄
      })
      const names = registered.map((t) => t.name).sort()
      assert.deepStrictEqual(names, ['memory_admin', 'recall_memory', 'save_memory', 'update_memory'])

      // 通用 schema 合法性
      for (const t of registered) {
        assert.ok(t.parameters && t.parameters.type === 'object', `${t.name}: parameters 应为对象`)
        assert.ok(t.parameters.properties && typeof t.parameters.properties === 'object', `${t.name}: 应有 properties`)
        assert.ok(Array.isArray(t.parameters.required), `${t.name}: 应有 required 数组`)
        assert.ok(t.output && t.output.schema && t.output.schema.type === 'string', `${t.name}: output.schema 应为 string`)
        assert.strictEqual(typeof t.output.render, 'function', `${t.name}: 应有 render`)
        assert.strictEqual(typeof t.execute, 'function', `${t.name}: 应有 execute`)
      }

      // 各自的参数约束
      const save = toolByName(registered, 'save_memory')
      assert.deepStrictEqual(save.parameters.required, ['title', 'content', 'tags'])
      assert.strictEqual(save.parameters.properties.tags.type, 'array')
      assert.strictEqual(save.parameters.properties.tags.minItems, 1)
      assert.ok(save.description.includes('写入后立即可用'), 'save 描述应符合规格要点')

      const recall = toolByName(registered, 'recall_memory')
      assert.deepStrictEqual(recall.parameters.required, ['query'])
      assert.strictEqual(recall.parameters.properties.k.type, 'integer')
      assert.strictEqual(recall.parameters.properties.k.minimum, 1)
      assert.strictEqual(recall.parameters.properties.k.maximum, 20)
      assert.strictEqual(recall.parameters.properties.k.default, 6)
      assert.strictEqual(recall.parameters.properties.truncate.minimum, 0)
      assert.strictEqual(recall.parameters.properties.truncate.maximum, 1)
      assert.strictEqual(recall.parameters.properties.truncate.default, 0.4)

      const admin = toolByName(registered, 'memory_admin')
      assert.deepStrictEqual(admin.parameters.required, ['op'])
      assert.deepStrictEqual(admin.parameters.properties.op.enum, ['stats', 'rebuild'])

      // P3 §3 update_memory：锚点式修正，target ≥15 字符
      const update = toolByName(registered, 'update_memory')
      assert.deepStrictEqual(update.parameters.required, ['target', 'replace'])
      assert.strictEqual(update.parameters.properties.target.type, 'string')
      assert.strictEqual(update.parameters.properties.target.minLength, 15)
      assert.strictEqual(update.parameters.properties.replace.type, 'string')
      assert.ok(update.description.includes('至少 15 字符'), 'update 描述应说明 target 锚点要求')

      tools = { save, recall, admin, update }
    })

    // ── 用例 3：save_memory 端到端（真 Ollama 落盘 + 入队）──
    let savedFile = null
    await test('save_memory：返回 { ok:true, file(相对路径), tags(保序清洗) }', async () => {
      const out = JSON.parse(
        await tools.save.execute({
          title: '浪潮算法讨论',
          content: '上周五和徕拉讨论了 TagMemo 的浪潮算法，我们决定在下次迭代里把残余金字塔换成传播核。',
          tags: ['TagMemo，浪潮算法', '设计,讨论'],
        })
      )
      assert.strictEqual(out.ok, true)
      assert.ok(out.file && out.file.endsWith('.md'), `file 应为日记 .md 相对路径：${out.file}`)
      assert.ok(!out.file.startsWith('/') && !/^[A-Za-z]:/.test(out.file), 'file 应为相对路径')
      assert.ok(out.file.startsWith('diaries/'), 'file 应位于 diaries/ 下')
      assert.ok(Array.isArray(out.tags) && out.tags.length === 2, 'tags 应保序返回 2 个')
      assert.ok(out.tags.every((t) => t.trim() === t), 'tags 不应带首尾空白')
      assert.ok(out.tags[0].includes('TagMemo'), 'tags 应经标点清洗（，→ , ）')
      savedFile = out.file
    })

    // ── 用例 4：等索引队列排空后 recall 命中 ──
    await test('后台索引完成后 recall_memory 语义命中（score 高）', async () => {
      // 轮询 stats 直到无 pending 且已有 chunk 入索引
      let indexed = false
      for (let i = 0; i < 300; i++) {
        const s = JSON.parse(await tools.admin.execute({ op: 'stats' }))
        if (s.ok && s.stats.pendingFiles === 0 && s.stats.indexedChunks >= 1) { indexed = true; break }
        await sleep(100)
      }
      assert.ok(indexed, '索引应在 30s 内完成')

      const out = JSON.parse(await tools.recall.execute({ query: '浪潮算法讨论', k: 6, truncate: 0.4 }))
      assert.ok(Array.isArray(out.blocks) && out.blocks.length > 0, '应命中至少一块')
      assert.ok(out.stats && typeof out.stats.ms === 'number', '应含 stats')
      assert.ok(out.blocks[0].score > 0.4, `命中分数应 > 0.4，实际 ${out.blocks[0].score}`)
      assert.strictEqual(out.blocks[0].file, savedFile, '首位命中应是刚保存的日记')
      assert.ok(out.blocks[0].text.includes('浪潮算法'), '命中文本应含查询主题')
    })

    // ── 用例 5：update_memory 端到端（复用 save 落盘的日记；P3 §3）──
    await test('update_memory：正常替换 + target 过短报错 + 不命中报错', async () => {
      // (a) 正常替换：嵌套在正文里的 21 字符原文片段，恰好一处
      const TARGET = '我们决定在下次迭代里把残余金字塔换成传播核'
      assert.ok(TARGET.length >= 15)
      const out = JSON.parse(
        await tools.update.execute({ target: TARGET, replace: '我们改用基于图的标签传播做聚类' })
      )
      assert.strictEqual(out.ok, true)
      assert.strictEqual(out.file, savedFile, 'update 应返回被改文件（与 save 同一篇）')
      assert.strictEqual(out.replaced, true)

      // 落盘内容：target 消失、replace 在位、其余正文不变（禁止整文件覆写）
      const text = await readFile(path.join(base, 'happy', out.file), 'utf8')
      assert.ok(!text.includes(TARGET), '旧 target 应被替换')
      assert.ok(text.includes('我们改用基于图的标签传播做聚类'), 'replace 应在位')
      assert.ok(text.includes('浪潮算法'), '其余正文不受影响（禁止整文件覆写）')

      // 替换走同一索引队列：轮询排空后 recall 命中新文本
      let reindexed = false
      for (let i = 0; i < 300; i++) {
        const s = JSON.parse(await tools.admin.execute({ op: 'stats' }))
        if (s.ok && s.stats.pendingFiles === 0) { reindexed = true; break }
        await sleep(100)
      }
      assert.ok(reindexed, '替换后的索引队列应在 30s 内排空')
      const rec = JSON.parse(await tools.recall.execute({ query: '基于图的标签传播', k: 6, truncate: 0 }))
      const hit = rec.blocks.find((b) => b.file === savedFile)
      assert.ok(hit && hit.text.includes('基于图的标签传播'), '替换后索引内容应更新')

      // (b) target 过短报错
      const short = JSON.parse(await tools.update.execute({ target: '太短', replace: 'x' }))
      assert.strictEqual(short.ok, false)
      assert.match(short.error, /15/, '过短 target 应报错')

      // (c) 不命中报错
      const miss = JSON.parse(
        await tools.update.execute({ target: '这是正文里不存在的超长锚点片段字符啊', replace: 'x' })
      )
      assert.strictEqual(miss.ok, false)
      assert.match(miss.error, /未找到/, '不命中应报明确错误')
    })

    // ── 用例 6：memory_admin stats / render 契约 ──
    await test('memory_admin stats 与 output.render 契约', async () => {
      const s = JSON.parse(await tools.admin.execute({ op: 'stats' }))
      assert.strictEqual(s.ok, true)
      assert.strictEqual(s.op, 'stats')
      assert.strictEqual(s.stats.sig, 'bge-m3@1024')
      assert.strictEqual(s.stats.dimension, 1024)
      assert.ok(Number.isInteger(s.stats.diaries) && s.stats.diaries >= 1)
      // render：把 execute 返回的字符串包成 text 块
      const rendered = tools.admin.output.render({}, '{"ok":true}')
      assert.deepStrictEqual(rendered, [{ type: 'text', text: '{"ok":true}' }])
    })

    // ── 用例 7：sig 不一致 → store.ready reject 被捕获，工具照常注册，调用返回拒绝服务文本 ──
    await test('sig 不一致：四工具仍注册，execute 返回含 error 的拒绝服务 JSON', async () => {
      const root = path.join(base, 'fault')
      await mkdir(path.join(root, 'index'), { recursive: true })
      await mkdir(path.join(root, 'diaries', 'dsh'), { recursive: true })
      // 预置一份"旧签名"索引（与当前 bge-m3@1024 不一致）
      await writeFile(
        path.join(root, 'index', 'meta.json'),
        JSON.stringify({ sig: 'bge-m3@768', dimension: 768, chunkCount: 0, updatedAt: 0 })
      )
      await writeFile(
        path.join(root, 'diaries', 'dsh', '2026-08-29-10_00_00.md'),
        '[2026-08-29] - dsh\n[10:00]\n旧索引语义空间下的内容。\n\nTag: 旧\n'
      )

      const { ctx, registered } = fakeCtx()
      await apply(ctx, { dataRoot: root, agentName: 'dsh', embedding: REAL, watch: false })
      const names = registered.map((t) => t.name).sort()
      assert.deepStrictEqual(names, ['memory_admin', 'recall_memory', 'save_memory', 'update_memory'], '拒绝服务时工具仍照常注册')

      const saveOut = JSON.parse(await toolByName(registered, 'save_memory').execute({ title: 't', content: 'c', tags: ['t'] }))
      assert.strictEqual(saveOut.ok, false)
      assert.match(saveOut.error, /嵌入模型签名已变/)

      const recallOut = JSON.parse(await toolByName(registered, 'recall_memory').execute({ query: 'x' }))
      assert.match(recallOut.error, /嵌入模型签名已变/)

      // P3 §3：fault 态 update_memory 同样拒绝（不动盘、不查索引）
      const updateOut = JSON.parse(
        await toolByName(registered, 'update_memory').execute({ target: '这是一段超过十五个字符的替换目标文本片段啊', replace: 'x' })
      )
      assert.strictEqual(updateOut.ok, false)
      assert.match(updateOut.error, /嵌入模型签名已变/)

      const adminOut = JSON.parse(await toolByName(registered, 'memory_admin').execute({ op: 'stats' }))
      assert.match(adminOut.error, /嵌入模型签名已变/)

      // 拒绝服务不得写索引派生产物（绝不清库、绝不混用）
      await assert.rejects(access(path.join(root, 'index', 'chunks.jsonl')), '不应创建 chunks.jsonl')

      // fault 态下 rebuild 是放行的修复通道：执行成功（真 Ollama 重建该临时库）后
      // fault 解除，save/recall 恢复服务
      const rebuildOut = JSON.parse(await toolByName(registered, 'memory_admin').execute({ op: 'rebuild' }))
      assert.strictEqual(rebuildOut.ok, true, 'fault 态下 rebuild 必须可执行')
      const saveAfter = JSON.parse(
        await toolByName(registered, 'save_memory').execute({ title: '恢复验证', content: '重建后写入恢复正常。', tags: ['恢复'] })
      )
      assert.strictEqual(saveAfter.ok, true, 'rebuild 成功后 save_memory 应恢复服务')
    })

    console.log(`\nentry.test.mjs: 通过 ${passed} 项，失败 ${failed} 项`)
    process.exitCode = failed ? 1 : 0 // 用 exitCode 而非 process.exit：后者会跳过 finally 的临时目录清理
  } finally {
    await rm(base, { recursive: true, force: true }).catch(() => {})
  }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})