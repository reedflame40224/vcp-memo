// tests/inject.test.mjs —— SPEC-INJECT §4 验收(第 1-9 项;零依赖,node 直接运行)
//
// 覆盖:
//   1. 注入消息工厂形状(UUID/role/source/递归冻结);
//   2. 注入成功:位置在 claimed 批之后、内容含 <memory> 与最高分块正文,
//      recall 收到 0.7/0.3 加权归一化向量(已知 fake 向量手算对照,容差 1e-6);
//   3. step=2 → 原样返回(同一数组引用,未 splice);
//   4. 节流:同 agent 连续两 turn 相同 userText → 第二次不注入;不同文本 → 注入;
//      buildSeedText 直测;每 agent 独立一份节流表;
//   5. 种子过滤:messages 混入上轮注入的 plugin/recall 与 tool 消息时,
//      seed 只取 source.kind==='user' 的最后一条(用 recall 收到的向量反推验证);
//   6. recall 抛错 / 超时(挂起 + timeoutMs=50) / 0 blocks → 原样返回且不抛异常;
//   7. maxChars=200:低分块被丢弃、输出长度 ≤ 预算;最高分块超预算时保留截断版;
//   8. reject / 空 messages / enabled=false / fault 非 null → 原样返回且无副作用;
//   9. 入口集成(假 ctx 同 entry.test 模式):apply 后 pre-step 监听器已注册、
//      systemPrompt 假服务收到 name 'vcp-memo:discipline' order 400 的 section,
//      section 的 disposer 挂在 ctx.effect。
// (第 10 项回归在 P0/P1 既有测试中保持。)
// 用法:node tests/inject.test.mjs(不需要 Ollama,全 fake——入口集成用空库,
//  apply 全程不触发 embed;embedding 端点指向不可达端口即可)

import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createInjector } from '../engine/inject.mjs'
import { apply } from '../vcp-memo.mjs'

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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// ---------- fakes ----------

// 确定性假 embedder:形状对齐 createEmbedder 的返回({ embed, ... } 对象);文本查表
// 返回向量(未注册 → 零向量,测试会因向量不符而显式失败);calls 记录每次 embed 收到
// 的文本数组,供"种子过滤"反推验证。
function makeFakeEmbedder(dim = 4) {
  const table = new Map()
  const calls = []
  const embed = async (texts) => {
    calls.push(texts.slice())
    return texts.map((t) => (table.has(t) ? Float32Array.from(table.get(t)) : new Float32Array(dim)))
  }
  return {
    embed,
    calls,
    set(text, v) {
      table.set(text, Float32Array.from(v))
    },
  }
}

// 假 store:记录 recall 调用参数(vector 复制到边界),可编程返回 blocks/抛错/挂起
function makeFakeStore() {
  const calls = []
  const state = { blocks: [], error: null, hang: false }
  const store = {
    calls,
    state,
    async recall(args) {
      calls.push({ ...args, vector: args.vector ? Float32Array.from(args.vector) : args.vector })
      if (state.hang) return new Promise(() => {}) // 永不 settle,配合 timeoutMs 触发超时
      if (state.error) throw state.error
      return { blocks: state.blocks, stats: { candidates: state.blocks.length, indexedChunks: 0, ms: 1 } }
    },
  }
  return store
}

function makeFakeAgent(id, events) {
  return { id, ...(events ? { session: { events } } : {}) }
}

// harness 实证形状:session.append('assistant/message', { turn, step, message, usage? })
// —— 消息在事件的 data.message,不是事件顶层(packages/core/agent-loop/src/agent.ts:417)
const assistantEvent = (text, turn = 1, step = 1) => ({
  type: 'assistant/message',
  data: { turn, step, message: { role: 'assistant', content: [{ type: 'text', text }] } },
  time: Date.now(),
})

// 造一个 source.kind==='user' 的真实 user 消息
let msgSeq = 0
function userMsg(text) {
  msgSeq++
  return { id: `u${msgSeq}`, role: 'user', content: [{ type: 'text', text }], source: { kind: 'user' } }
}

// 一轮基础 turn:一条 user 消息,decision.messages 与 messages 引用一致
function baseTurn(userText = '帮我查一下部署问题') {
  const m = userMsg(userText)
  const messages = [m]
  return { messages, decision: { kind: 'enter', messages: [...messages] } }
}

// createInjector 封装:带收集日志的假 log、默认 fault 读取器
function makeInjector({ store, embedder, config, fault } = {}) {
  const logs = []
  const injector = createInjector({
    store,
    embedder,
    config,
    log: (level, msg) => logs.push(`${level}:${msg}`),
    fault: fault || (() => null),
  })
  return { injector, logs }
}

// 把 handler 跑一轮:event = { agent, messages, step },next 返回决策
function runTurn(handler, { agent, messages, decision, step = 1 } = {}) {
  return handler({ agent, messages, step }, () => decision)
}

// 取出注入的 memory 消息(plugin-source 的最后一条)
function memoryMessageOf(decision) {
  const m = [...decision.messages].reverse().find((x) => x.source && x.source.kind === 'plugin')
  assert.ok(m, '应存在注入的 plugin-source 消息')
  return m
}

// 逐位对照向量(容差 tol)
function assertVecClose(actual, expected, tol) {
  assert.ok(actual instanceof Float32Array || Array.isArray(actual), '应收到向量数组')
  assert.strictEqual(actual.length, expected.length)
  for (let i = 0; i < expected.length; i++) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) <= tol,
      `下标 ${i}:实际 ${actual[i]} 期望 ${expected[i]}(容差 ${tol})`
    )
  }
}

// ---------- 用例 ----------

async function main() {
  console.log('== inject.test.mjs ==')

  // ── 1. 消息工厂形状 ──
  await test('消息工厂:UUID/role user/content text/source plugin recall/递归冻结', async () => {
    const store = makeFakeStore()
    store.state.blocks = [{ file: 'f.md', chunkIndex: 0, score: 0.6, text: '记忆正文' }]
    const { injector } = makeInjector({ store, embedder: makeFakeEmbedder(2) })
    const { messages, decision } = baseTurn()
    const out = await runTurn(injector.handler, { agent: makeFakeAgent('s1'), messages, decision })
    const m = memoryMessageOf(out)
    assert.match(m.id, /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i, 'id 应为 UUID')
    assert.strictEqual(m.role, 'user')
    assert.strictEqual(m.content[0].type, 'text')
    assert.ok(m.content[0].text.includes('<memory>'))
    assert.deepStrictEqual(m.source, { kind: 'plugin', plugin: 'vcp-memo', form: 'recall' })
    // 递归冻结(对齐 harness 不可变约定)
    assert.ok(Object.isFrozen(m), '消息本身应冻结')
    assert.ok(Object.isFrozen(m.content) && Object.isFrozen(m.content[0]), 'content 及其元素应递归冻结')
    assert.ok(Object.isFrozen(m.source), 'source 应冻结')
  })

  // ── 2. 注入成功:位置 / 内容 / 加权向量 ──
  await test('注入成功:位置在 claimed 批之后;recall 收到 0.7/0.3 加权归一化向量(手算对照)', async () => {
    const embedder = makeFakeEmbedder(4)
    embedder.set('帮我查一下部署问题', [1, 0, 0, 0])
    embedder.set('好的,我来帮你排查', [0, 1, 0, 0])
    embedder.set('最近怎么样', [0, 0, 1, 0]) // 若错误选用第一条 user,向量会对不上
    const store = makeFakeStore()
    const highBlock = { file: 'diaries/dsh/2026-08-29-14_11_45-foo.md', chunkIndex: 0, score: 0.83, text: '澜沧计划部署约定：周五上线，灰度 10%。' }
    store.state.blocks = [
      highBlock,
      { file: 'diaries/dsh/2026-08-29-12_00_00-bar.md', chunkIndex: 1, score: 0.61, text: '上周五和徕拉讨论了 TagMemo 的浪潮算法。' },
    ]
    const { injector } = makeInjector({ store, embedder })
    const agent = makeFakeAgent('a2', [assistantEvent('好的,我来帮你排查')])
    const user1 = userMsg('最近怎么样')
    const user2 = userMsg('帮我查一下部署问题')
    const messages = [user1, user2]
    const extra = { id: 'x1', role: 'assistant', content: [{ type: 'text', text: '下游已 claim 的消息' }], source: { kind: 'assistant' } }
    const decision = { kind: 'enter', messages: [...messages, extra] }

    const out = await runTurn(injector.handler, { agent, messages, decision })
    // 返回新对象、原 decision 未被改动(不可变拼接)
    assert.notStrictEqual(out, decision)
    assert.strictEqual(out.kind, 'enter')
    assert.strictEqual(decision.messages.length, 3, '原 decision.messages 不得被 splice 改动')
    assert.strictEqual(out.messages.length, 4)
    // 位置:最后的原消息(user2,下标 1)之后、extra 之前
    const mem = memoryMessageOf(out)
    assert.strictEqual(out.messages[2], mem, 'memory 消息应在 claimed 批之后')
    assert.strictEqual(out.messages[3], extra)
    // 内容:含 <memory> 与最高分块正文与分数记号
    assert.ok(mem.content[0].text.includes('<memory>'))
    assert.ok(mem.content[0].text.includes('澜沧计划部署约定'), '应含最高分块正文')
    assert.ok(mem.content[0].text.includes('0.83'))

    // recall 调用:只给 vector(§2 的向量路径),k/truncate 用默认
    assert.strictEqual(store.calls.length, 1)
    const call = store.calls[0]
    assert.strictEqual(call.k, 4)
    assert.strictEqual(call.truncate, 0.55)
    assert.ok(!('query' in call), '向量路径不应传 query')
    // 0.7/0.3 加权归一化手算:user→[1,0,0,0]、assistant→[0,1,0,0]
    const n = Math.sqrt(0.7 ** 2 + 0.3 ** 2)
    assertVecClose(call.vector, [0.7 / n, 0.3 / n, 0, 0], 1e-6)
    // embed 只调一次,输入为 [userText, assistantText]
    assert.deepStrictEqual(embedder.calls[0], ['帮我查一下部署问题', '好的,我来帮你排查'])

    // -1 分支(§0):decision.messages 不含任何原消息 → 拼到末尾
    const store2b = makeFakeStore()
    store2b.state.blocks = [{ file: 'f.md', chunkIndex: 0, score: 0.6, text: '正文' }]
    const { injector: inj2 } = makeInjector({ store: store2b, embedder: makeFakeEmbedder(2) })
    const t2b = baseTurn('帮我查一下部署问题')
    const out2 = await runTurn(inj2.handler, {
      agent: makeFakeAgent('a2b'),
      messages: t2b.messages,
      decision: { kind: 'enter', messages: [] },
    })
    assert.strictEqual(out2.messages.length, 1, '-1 分支应拼到末尾')
    assert.ok(memoryMessageOf(out2))
  })

  // ── 3. step=2 跳过 ──
  await test('step=2 → 原样返回(同一对象,未 splice,不调 recall)', async () => {
    const store = makeFakeStore()
    const { injector } = makeInjector({ store, embedder: makeFakeEmbedder(2) })
    const { messages, decision } = baseTurn()
    const out = await runTurn(injector.handler, { agent: makeFakeAgent('a3'), messages, decision, step: 2 })
    assert.strictEqual(out, decision)
    assert.strictEqual(decision.messages.length, 1)
    assert.strictEqual(store.calls.length, 0)
  })

  // ── 4. 节流 ──
  await test('节流:同 agent 相同 userText 第二次不注入;不同文本注入;每 agent 独立一份', async () => {
    const store = makeFakeStore()
    store.state.blocks = [{ file: 'f.md', chunkIndex: 0, score: 0.6, text: '正文' }]
    const embedder = makeFakeEmbedder(4)
    embedder.set('你好', [1, 0, 0, 0])
    embedder.set('下一个话题', [0, 1, 0, 0])
    const { injector } = makeInjector({ store, embedder })
    const agent = makeFakeAgent('a1')

    // 第 1 轮:注入
    const t1 = baseTurn('你好')
    const out1 = await runTurn(injector.handler, { agent, messages: t1.messages, decision: t1.decision })
    assert.strictEqual(out1.messages.length, 2)
    assert.ok(memoryMessageOf(out1))
    assert.strictEqual(store.calls.length, 1)

    // 第 2 轮:相同 userText → 节流,原样返回,不再调 recall
    const t2 = baseTurn('你好')
    const out2 = await runTurn(injector.handler, { agent, messages: t2.messages, decision: t2.decision })
    assert.strictEqual(out2, t2.decision)
    assert.strictEqual(store.calls.length, 1)

    // 第 3 轮:不同 userText → 再次注入
    const t3 = baseTurn('下一个话题')
    const out3 = await runTurn(injector.handler, { agent, messages: t3.messages, decision: t3.decision })
    assert.strictEqual(out3.messages.length, 2)
    assert.strictEqual(store.calls.length, 2)

    // buildSeedText 直接契约:userText + '\n' + assistantText;assistant 缺失 → 空串尾
    assert.strictEqual(injector.buildSeedText('a', 'b'), 'a\nb')
    assert.strictEqual(injector.buildSeedText('a', null), 'a\n')

    // 每 agent 独立:另一个 agent 同一文本仍注入
    const t4 = baseTurn('你好')
    const out4 = await runTurn(injector.handler, { agent: makeFakeAgent('a2'), messages: t4.messages, decision: t4.decision })
    assert.strictEqual(out4.messages.length, 2)
    assert.strictEqual(store.calls.length, 3)
  })

  // ── 5. 种子过滤 ──
  await test('种子过滤:混入 plugin/recall 与 tool 消息时,seed 只取 source.kind===\'user\' 的最后一条', async () => {
    const embedder = makeFakeEmbedder(4)
    embedder.set('第一问', [0, 1, 0, 0])
    embedder.set('第二问', [1, 0, 0, 0])
    embedder.set('上轮注入的记忆', [0, 0, 1, 0])
    embedder.set('工具结果', [0, 0, 0, 1])
    const store = makeFakeStore()
    store.state.blocks = [{ file: 'f.md', chunkIndex: 0, score: 0.6, text: '正文' }]
    const { injector } = makeInjector({ store, embedder })
    const oldRecall = { id: 'um', role: 'user', content: [{ type: 'text', text: '上轮注入的记忆' }], source: { kind: 'plugin', plugin: 'vcp-memo', form: 'recall' } }
    const oldTool = { id: 'ut', role: 'tool', content: [{ type: 'text', text: '工具结果' }], source: { kind: 'tool' } }
    // 最后一条消息是 tool/plugin,但 seed 必须取最后一条 user('第二问')
    const messages = [userMsg('第一问'), oldRecall, userMsg('第二问'), oldTool]
    const decision = { kind: 'enter', messages: [...messages] }
    const out = await runTurn(injector.handler, { agent: makeFakeAgent('a5'), messages, decision })

    // 反推验证:embed 只收到最后一条真实 user 的文本;向量即其归一化向量
    assert.deepStrictEqual(embedder.calls[0], ['第二问'], '不应 embed 上轮注入/tool 消息')
    assertVecClose(store.calls[0].vector, [1, 0, 0, 0], 1e-6) // normalize([1,0,0,0])
    // 新增的注入消息不是混入的那条旧 plugin 消息
    const mems = out.messages.filter((m) => m.source && m.source.kind === 'plugin')
    assert.strictEqual(mems.length, 2, '旧的 1 条 + 新增 1 条')
    const mem = memoryMessageOf(out)
    assert.notStrictEqual(mem, oldRecall)
    assert.ok(mem.content[0].text.includes('<memory>'))
  })

  // ── 6. 失败降级 ──
  await test('recall 抛错 → 原样返回且不抛异常(记日志)', async () => {
    const store = makeFakeStore()
    store.state.error = new Error('存储炸了')
    const embedder = makeFakeEmbedder(2)
    embedder.set('问', [1, 0])
    const { injector, logs } = makeInjector({ store, embedder })
    const { messages, decision } = baseTurn('问')
    const out = await runTurn(injector.handler, { agent: makeFakeAgent('g6a'), messages, decision })
    assert.strictEqual(out, decision)
    assert.strictEqual(store.calls.length, 1)
    assert.ok(logs.some((l) => l.includes('注入失败')), '抛错应记日志')
  })

  await test('recall 挂起 + timeoutMs=50 → 超时原样返回且不抛异常', async () => {
    const store = makeFakeStore()
    store.state.hang = true
    const { injector, logs } = makeInjector({ store, embedder: makeFakeEmbedder(2), config: { timeoutMs: 50 } })
    const { messages, decision } = baseTurn('问')
    const t0 = Date.now()
    const out = await runTurn(injector.handler, { agent: makeFakeAgent('g6b'), messages, decision })
    const elapsed = Date.now() - t0
    assert.strictEqual(out, decision)
    assert.strictEqual(store.calls.length, 1)
    assert.ok(elapsed >= 40, `应经历 50ms 超时等待,实际 ${elapsed}ms`)
    assert.ok(elapsed < 5000, '不应长时间挂起')
    assert.ok(logs.some((l) => l.includes('超时')), '超时应记日志')
  })

  await test('recall 返回 0 blocks → 原样返回且不抛异常', async () => {
    const store = makeFakeStore()
    store.state.blocks = []
    const { injector } = makeInjector({ store, embedder: makeFakeEmbedder(2) })
    const { messages, decision } = baseTurn('问')
    const out = await runTurn(injector.handler, { agent: makeFakeAgent('g6c'), messages, decision })
    assert.strictEqual(out, decision)
    assert.strictEqual(store.calls.length, 1)
    assert.strictEqual(out.messages.length, 1, '不应注入任何消息')
  })

  // ── 7. maxChars 预算 ──
  await test('maxChars=200:低分块被丢弃、输出长度≤预算;最高分块超预算时保留截断版', async () => {
    // (a) 三块总和超预算 → 丢弃中/低分块,保留最高分块
    const highBody = '澜'.repeat(80)
    const midBody = '中'.repeat(50)
    const lowBody = '低'.repeat(50)
    const store = makeFakeStore()
    store.state.blocks = [
      { file: 'f1.md', chunkIndex: 0, score: 0.9, text: highBody },
      { file: 'f2.md', chunkIndex: 0, score: 0.5, text: midBody },
      { file: 'f3.md', chunkIndex: 0, score: 0.1, text: lowBody },
    ]
    const { injector } = makeInjector({ store, embedder: makeFakeEmbedder(2), config: { maxChars: 200, truncate: 0 } })
    const { messages, decision } = baseTurn('问')
    const out = await runTurn(injector.handler, { agent: makeFakeAgent('g7a'), messages, decision })
    const text = memoryMessageOf(out).content[0].text
    assert.ok(text.length <= 200, `输出长度 ${text.length} 应 ≤ 200`)
    assert.ok(text.includes(highBody) && text.includes('</memory>'), '最高分块应完整保留')
    assert.ok(!text.includes(midBody) && !text.includes(lowBody), '低分块应被丢弃')

    // (b) 最高分块本身超预算 → 保留其截断版(至少 100 字符)
    const store2 = makeFakeStore()
    store2.state.blocks = [{ file: 'f9.md', chunkIndex: 0, score: 0.9, text: 'x'.repeat(300) }]
    const { injector: inj2 } = makeInjector({ store: store2, embedder: makeFakeEmbedder(2), config: { maxChars: 200, truncate: 0 } })
    const t2 = baseTurn('问')
    const out2 = await runTurn(inj2.handler, { agent: makeFakeAgent('g7b'), messages: t2.messages, decision: t2.decision })
    const text2 = memoryMessageOf(out2).content[0].text
    assert.ok(text2.length <= 200, `截断版长度 ${text2.length} 应 ≤ 200`)
    assert.ok(text2.startsWith('<memory>') && text2.endsWith('</memory>'))
    assert.ok(text2.includes('x'.repeat(100)), '应保留最高分块的截断正文')
  })

  // ── 8. 跳过条件 ──
  await test('reject / 空 messages / enabled=false / fault 非 null → 原样返回且无副作用', async () => {
    // (a) reject decision
    const store = makeFakeStore()
    const { injector } = makeInjector({ store, embedder: makeFakeEmbedder(2) })
    const { messages: msgsA } = baseTurn()
    const rejectDec = { kind: 'reject' }
    const outA = await runTurn(injector.handler, { agent: makeFakeAgent('g8a'), messages: msgsA, decision: rejectDec })
    assert.strictEqual(outA, rejectDec)
    assert.strictEqual(store.calls.length, 0)

    // (b) 空 messages
    const decB = { kind: 'enter', messages: [] }
    const outB = await runTurn(injector.handler, { agent: makeFakeAgent('g8b'), messages: [], decision: decB })
    assert.strictEqual(outB, decB)
    assert.strictEqual(store.calls.length, 0)

    // (c) enabled=false
    const storeC = makeFakeStore()
    const { injector: injC } = makeInjector({ store: storeC, embedder: makeFakeEmbedder(2), config: { enabled: false } })
    const { messages: msgsC, decision: decC } = baseTurn()
    const outC = await runTurn(injC.handler, { agent: makeFakeAgent('g8c'), messages: msgsC, decision: decC })
    assert.strictEqual(outC, decC)
    assert.strictEqual(storeC.calls.length, 0)

    // (d) fault 非 null(拒绝服务态)
    const storeD = makeFakeStore()
    const { injector: injD } = makeInjector({ store: storeD, embedder: makeFakeEmbedder(2), fault: () => '嵌入模型签名已变' })
    const { messages: msgsD, decision: decD } = baseTurn()
    const outD = await runTurn(injD.handler, { agent: makeFakeAgent('g8d'), messages: msgsD, decision: decD })
    assert.strictEqual(outD, decD)
    assert.strictEqual(storeD.calls.length, 0)
  })

  // ── 9. 入口集成(SPEC-INJECT §3):apply 后 pre-step 监听器已注册、
  //       systemPrompt 收到 vcp-memo:discipline section(order 400) ──
  await test('入口集成:pre-step 监听器注册 + systemPrompt 收到 name/order 正确的 section + disposer 挂 ctx.effect', async () => {
    // 假 ctx(同 entry.test 模式):tools.register 收集;effect 立即执行并记录;
    // on 捕获注册的监听器;get 返回假 systemPrompt 服务
    const registered = []
    const listeners = []
    const effects = []
    const sections = []
    const systemPrompt = {
      section(section) {
        sections.push(section)
        return () => {} // disposer:挂到 ctx.effect 后随 Fiber 生命周期撤销
      },
    }
    const ctx = {
      tools: { register(t) { registered.push(t) } },
      effect(fn) {
        effects.push(fn)
        fn()
      },
      on(name, listener) {
        listeners.push({ name, listener })
        return () => {}
      },
      get(name) {
        return name === 'systemPrompt' ? systemPrompt : undefined
      },
      logger: console,
    }
    const root = await mkdtemp(path.join(os.tmpdir(), 'vcp-memo-inject-'))
    try {
      await apply(ctx, {
        dataRoot: root,
        agentName: 'dsh',
        // 空库启动全程不触发 embed:端点指向不可达端口即可(本文件保持"不需要 Ollama")
        embedding: { baseUrl: 'http://127.0.0.1:9/v1', model: 'bge-m3', dimension: 1024 },
        watch: false,
        injection: { enabled: true, k: 4 }, // 显式给 injection 节,验证 config 流动
      })

      // (a) pre-step 监听器已注册(ctx.on 捕获)
      assert.strictEqual(listeners.length, 1, '应恰好注册 1 个 agent/pre-step 监听器')
      assert.strictEqual(listeners[0].name, 'agent/pre-step')
      assert.strictEqual(typeof listeners[0].listener, 'function', '监听器应是可调用的 handler')

      // (b) systemPrompt 假服务收到 vcp-memo:discipline,order 400(§0/§3)
      assert.strictEqual(sections.length, 1, '应恰好注册 1 个 prompt section')
      assert.strictEqual(sections[0].name, 'vcp-memo:discipline')
      assert.strictEqual(sections[0].order, 400)
      assert.ok(sections[0].text.includes('记忆使用规范'), 'section 文本应为记忆使用规范')
      assert.ok(
        sections[0].text.includes('save_memory') && sections[0].text.includes('recall_memory'),
        'section 文本应提到 save_memory 与 recall_memory'
      )

      // (c) section 注册必须包在 ctx.effect 里:effects 中存在一条调用时会再次注册 section
      //     (即"section 返回的 disposer 挂到 ctx.effect"的等价可测形态)
      assert.ok(
        effects.some((fn) => {
          const before = sections.length
          fn()
          return sections.length === before + 1
        }),
        'discipline section 应经 ctx.effect 注册(返回的 disposer 挂到 Fiber 生命周期)'
      )

      // (d) 四工具仍照常注册(§3 接线不得破坏 §8+P3 §3 工具契约)
      const names = registered.map((t) => t.name).sort()
      assert.deepStrictEqual(names, ['memory_admin', 'recall_memory', 'save_memory', 'update_memory'])

      // (e) 监听器冒烟:step=2 → 原样放行(不触发 embed/recall、不抛异常)
      const decision = { kind: 'enter', messages: [] }
      const out = await listeners[0].listener(
        {
          agent: { id: 't9' },
          messages: [{ id: 'u9', role: 'user', content: [{ type: 'text', text: '冒烟' }], source: { kind: 'user' } }],
          step: 2,
        },
        () => decision,
      )
      assert.strictEqual(out, decision, '非第一步骤应原样放行注入器')
    } finally {
      await rm(root, { recursive: true, force: true }).catch(() => {})
    }
  })

  // 收尾:给超时用例的挂起 store 一个退出呼吸(不必要,纯防御)
  await sleep(0)

  console.log(`\ninject.test.mjs: 通过 ${passed} 项,失败 ${failed} 项`)
  process.exitCode = failed ? 1 : 0
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})