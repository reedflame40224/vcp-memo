// engine/inject.mjs —— P1.5 被动注入(SPEC-INJECT §1)
//
// VCP 哲学的灵魂:"回忆先于意识"。在模型形成回答**之前**,把与当前对话相关的跨会话记忆
// 以 plugin-source 的 user 消息注入本轮请求,不再依赖模型主动调 recall_memory。
//
// 机制(§0):监听 `ctx.on('agent/pre-step', handler)`,waterfall 模式——
// 先 `await next()` 拿下游决定(PreStepDecision),再决定是否拼接;
// 注入位置照 agent-instructions 的范式:`decision.messages.toSpliced(lastClaimedIndex + 1, 0, memMsg)`,
// lastClaimedIndex = decision.messages.findLastIndex(m => messages.includes(m)),-1 时拼到末尾。
//
// 零 npm 依赖:只用 `node:crypto`;消息手工构造,递归 Object.freeze 对齐 harness 的不可变约定。
// store 的 vector 支持(SPEC-INJECT §2)由集成代理完成,本模块只按 `{ vector, k, truncate }`
// 调用 store.recall,不依赖真 store 改造(测试用 fake store 记录参数即可)。

import crypto from 'node:crypto'

// ---------- 小工具 ----------

// 数值兜底:undefined/null/非有限值 → 默认值,否则取 Number
function num(v, d) {
  if (v == null) return d
  const n = Number(v)
  return Number.isFinite(n) ? n : d
}

// 递归 Object.freeze(只冻结我们手工构造的普通对象;对齐 harness 不可变约定)。
// 先冻父再冻子:冻结后不再有任何写入,顺序无碍。
function deepFreeze(obj) {
  if (!obj || typeof obj !== 'object') return obj
  if (!Object.isFrozen(obj)) Object.freeze(obj)
  for (const k of Object.keys(obj)) deepFreeze(obj[k])
  return obj
}

// 取消息文本:content 为 text block 数组则全部拼接,为字符串(旧形态)则原样返回
function textOfMessage(m) {
  const content = m && m.content
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    let s = ''
    for (const b of content) {
      if (b && b.type === 'text' && typeof b.text === 'string') s += b.text
    }
    return s
  }
  return ''
}

// 最后一条真实 user 消息:只认 source.kind === 'user'。
// plugin/tool source(尤其我们上轮注入的 form:'recall' 消息)全部排除,防反馈回路(§1.1)。
function lastUserMessage(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (m && m.source && m.source.kind === 'user') return m
  }
  return null
}

// 最后一条 assistant 文本:反向遍历 agent.session.events(若可访问),找
// type === 'assistant/message' 的事件取其文本;取不到(或无 session)返回 null,
// 由调用方"权重全给 user"。
// 事件数据形状(harness 源码实证):user/message 的 data 即消息本身;
// assistant/message 的 data 是 { turn, step, message, usage? }——消息在 data.message。
// 做防御式兼容(data.message || data || ev.message || ev)。
function lastAssistantText(agent) {
  const events = agent && agent.session && agent.session.events
  if (!Array.isArray(events)) return null
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i]
    if (!ev || ev.type !== 'assistant/message') continue
    const carrier = (ev.data && (ev.data.message || ev.data)) || ev.message || ev
    const t = textOfMessage(carrier)
    if (t) return t
  }
  return null
}

// 归一化(原地,与 store 内部约定一致:归一化向量,点积即余弦)。
// 零向量保持原样(此时任何点积为 0,自然低于 truncate,recall 会返回 0 块并优雅跳过)。
function normalize(v) {
  if (!v) return null
  const arr = v instanceof Float32Array ? v : Array.isArray(v) ? Float32Array.from(v) : null
  if (!arr || arr.length === 0) return null
  let sum = 0
  for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i]
  const n = Math.sqrt(sum)
  if (n > 0) {
    for (let i = 0; i < arr.length; i++) arr[i] /= n
  }
  return arr
}

// 加权求和后归一化:q = normalize(userWeight*vUser + assistantWeight*vAssistant)(§1.1)
function combineVectors(wUser, vUser, wAssistant, vAssistant) {
  const len = Math.max(vUser.length, vAssistant.length)
  const out = new Float32Array(len)
  for (let i = 0; i < len; i++) {
    out[i] = (i < vUser.length ? wUser * vUser[i] : 0) + (i < vAssistant.length ? wAssistant * vAssistant[i] : 0)
  }
  return normalize(out)
}

// 查询向量:embed([userText, assistantText?]) → 加权归一化;缺 assistant 时只用 vUser。
// 返回 null 表示查询向量构建失败(embed 返回 null 位等)。
async function buildQueryVector(embedder, userText, assistantText, wUser, wAssistant) {
  if (!embedder || typeof embedder.embed !== 'function') {
    throw new Error('createInjector: embedder.embed 缺失')
  }
  const texts = assistantText ? [userText, assistantText] : [userText]
  const vecs = await embedder.embed(texts)
  if (!Array.isArray(vecs) || !vecs[0]) return null
  const vUser = vecs[0]
  // 缺 assistant(文本或向量任一缺失)→ 权重全给 user(§1.1)
  if (texts.length === 1 || !vecs[1]) return normalize(vUser)
  return combineVectors(wUser, vUser, wAssistant, vecs[1])
}

// 限时执行 store.recall:store.recall 不接受 signal(§2 只扩展了入参),用 Promise.race
// 等价实现 AbortSignal.timeout 语义。recall 迟到 reject 会被兜底 catch 吞掉,
// 避免"超时分支下 recall 稍后才拒绝"产生进程级未处理拒绝(注入永远不得弄垮对话轮)。
function recallWithTimeout(recallPromise, timeoutMs) {
  let timer = null
  const guard = new Promise((_, reject) => {
    timer = setTimeout(() => {
      const e = new Error(`store.recall 超时(${timeoutMs}ms)`)
      e.code = 'ERR_MEMO_RECALL_TIMEOUT'
      reject(e)
    }, Math.max(0, timeoutMs))
  })
  recallPromise.catch(() => {}) // 吞晚到的拒绝(race 的另一路仍会正常拿到结果/错误)
  return Promise.race([recallPromise, guard]).finally(() => {
    if (timer) clearTimeout(timer)
  })
}

// 注入消息的形状(§0):手工构造,零依赖;构造后递归冻结。
function makeRecallMessage(text) {
  return deepFreeze({
    id: crypto.randomUUID(),
    role: 'user',
    content: [{ type: 'text', text }],
    source: { kind: 'plugin', plugin: 'vcp-memo', form: 'recall' },
  })
}

// 拼接位置(§0):insert 到 claimed 批之后;decision.messages 与 messages 引用相交的
// 最后一条的下标即 lastClaimedIndex,-1(不相交)时拼到末尾。
function spliceDecisionMessages(decision, messages, memMsg) {
  const dist = decision.messages
  if (!Array.isArray(dist)) return [...messages, memMsg] // 防御:异常结构则拼到末尾
  const claimed = dist.findLastIndex((m) => messages.includes(m))
  if (claimed === -1) return [...dist, memMsg]
  return dist.toSpliced(claimed + 1, 0, memMsg)
}

// agent 身份键:优先 agent.id(字符串)保证同一会话跨轮稳定;缺失时退化用对象本身
function agentKey(agent) {
  return agent && typeof agent.id === 'string' ? agent.id : agent
}

// ---------- createInjector ----------

/**
 * 创建被动注入器。
 * deps: {
 *   store,               // openStore 实例(用其 recall)
 *   embedder,            // createEmbedder 实例(构造查询向量)
 *   config: {            // 来自插件 config.injection,全部可选,括号为默认(§1)
 *     enabled?=true, k?=4, truncate?=0.55, maxChars?=2000, timeoutMs?=1500,
 *     userWeight?=0.7, assistantWeight?=0.3,
 *   },
 *   log, fault: () => (string|null),  // 拒绝服务态读取器;非 null 时全部跳过
 * }
 * 返回 { handler, renderMemoryBlock, buildSeedText }。
 */
export function createInjector(deps) {
  const d = deps || {}
  if (!d.store || typeof d.store.recall !== 'function') {
    throw new TypeError('createInjector: deps.store 必须提供 recall')
  }
  const cfg = {
    enabled: d.config ? d.config.enabled !== false : true,
    k: Math.max(1, Math.floor(num(d.config && d.config.k, 4))),
    truncate: Math.min(1, Math.max(0, num(d.config && d.config.truncate, 0.55))),
    maxChars: Math.max(1, Math.floor(num(d.config && d.config.maxChars, 2000))),
    timeoutMs: Math.max(0, num(d.config && d.config.timeoutMs, 1500)),
    userWeight: num(d.config && d.config.userWeight, 0.7),
    assistantWeight: num(d.config && d.config.assistantWeight, 0.3),
  }
  // 日志兜底:log 自身失败不影响主流程
  const log = (level, msg) => {
    try {
      if (typeof d.log === 'function') d.log(level, msg)
    } catch {
      /* 忽略 */
    }
  }
  const faultReader = typeof d.fault === 'function' ? d.fault : () => null
  // 节流表(每 agent 一份):记录该 agent 上次注入(或失败)的 seedKey,
  // 同一话题的连续轮不重复注入(§1.1 第 4 步)。
  const lastSeedByAgent = new Map()

  async function handler(event, next) {
    // 1. 先拿下游决定(waterfall 范式);next() 的错误属于下游,不由注入器吞掉,
    //    否则会掩盖下游真实故障(这与"注入不阻塞对话轮"不矛盾——吞掉后我们也
    //    拿不到合法 decision 可返回)。
    const decision = await next()
    const payload = event || {}
    const { agent, messages, step } = payload
    let seedKey = null // 进到查询阶段后才有值;供错误路径记录 seed
    let seedOwner = undefined
    try {
      // 2. 任一命中即"原样返回 decision、不做任何副作用"(§1.1 第 2 步)
      if (!cfg.enabled) return decision
      if (faultReader()) return decision
      if (!decision || decision.kind === 'reject') return decision
      if (decision.kind !== 'enter') return decision // 防御:非 reject 非 enter 一律不动
      if (step !== 1) return decision
      if (!Array.isArray(messages) || messages.length === 0) return decision

      // 3. 构造查询(§1.1 第 3 步)
      const userMsg = lastUserMessage(messages)
      const userText = userMsg ? textOfMessage(userMsg) : ''
      if (!userText) return decision // 无 userText → 原样返回
      const assistantText = lastAssistantText(agent) // 取不到 → 权重全给 user
      seedKey = buildSeedText(userText, assistantText)
      seedOwner = agentKey(agent)

      // 4. 节流(§1.1 第 4 步):与上次注入的 seedKey 相同 → 原样返回
      if (lastSeedByAgent.get(seedOwner) === seedKey) return decision

      // 5. 查询向量 + 限时 recall(§1.1 第 5 步):
      //    失败(embed 失败/超时/抛错/0 blocks)都记 seedKey,避免同话题每轮重试同一失败
      const q = await buildQueryVector(d.embedder, userText, assistantText, cfg.userWeight, cfg.assistantWeight)
      if (!q) {
        log('warn', 'vcp-memo 查询向量构建失败(embed 结果为空),跳过本轮注入')
        lastSeedByAgent.set(seedOwner, seedKey)
        return decision
      }
      const out = await recallWithTimeout(d.store.recall({ vector: q, k: cfg.k, truncate: cfg.truncate }), cfg.timeoutMs)
      // 注入侧再过滤:wave 路径的 viaStructure 结构补充块豁免 truncate(对主动 recall
      // 合理——那是结构证据;对被动注入太吵——基线噪声带内的块不该进上下文)。
      // 被动注入的哲学是"宁缺毋滥":只让 ≥ truncate 的块进入 <memory>。
      const blocks = (out && Array.isArray(out.blocks) ? out.blocks : [])
        .filter((b) => typeof b.score === 'number' && b.score >= cfg.truncate)
      if (blocks.length === 0) {
        log('debug', `vcp-memo recall 无可用记忆块(${cfg.k}/${cfg.truncate}),跳过本轮注入`)
        lastSeedByAgent.set(seedOwner, seedKey)
        return decision
      }

      // 6. 渲染 <memory> 区块、构造消息、按 §0 拼入(§1.1 第 6 步)
      const memText = renderMemoryBlock(blocks, cfg.maxChars)
      const memMsg = makeRecallMessage(memText)
      const newMessages = spliceDecisionMessages(decision, messages, memMsg)
      lastSeedByAgent.set(seedOwner, seedKey)
      return { ...decision, messages: newMessages }
    } catch (err) {
      // 7. 全流程兜底:任何异常只跳过本轮,注入永远不得阻塞或弄垮对话轮(§1.1 第 7 步)
      log('error', `vcp-memo 注入失败,已跳过本轮注入: ${err && err.message || err}`)
      if (seedKey !== null && seedOwner !== undefined) lastSeedByAgent.set(seedOwner, seedKey)
      return decision
    }
  }

  return { handler, renderMemoryBlock, buildSeedText }
}

// ---------- §1.2 渲染与 §1.1 节流种子 ----------

/**
 * 节流种子(§1.1 第 4 步):seedKey = userText + '\n' + assistantText。
 * assistant 缺失(取不到或权重全给 user)时按空串,避免 'null' 字样污染 key。
 */
export function buildSeedText(userText, assistantText) {
  return `${String(userText)}\n${assistantText == null ? '' : String(assistantText)}`
}

/**
 * 渲染 <memory> 区块(§1.2):
 * - 按 score 降序逐个加入;超出 maxChars 时优先丢弃最低分块(只可能从尾部丢,低分在后);
 * - 至少保留最高分块的截断版(首个块本身放不下预算时,正文截断后仍保留);
 * - 正文原样,不做改写;预算含 <memory> 头与 </memory> 尾,输出长度恒 ≤ maxChars。
 * 示例格式:
 *   <memory>
 *   以下是系统按当前对话自动唤起的跨会话记忆(历史日记片段,供参考而非绝对真理):
 *
 *   [1] (score 0.83 · diaries/dsh/2026-08-29-14_11_45-xxx.md)
 *   <chunk 正文>
 *
 *   [2] ...
 *   </memory>
 */
export function renderMemoryBlock(blocks, maxChars) {
  const limit = Math.max(1, Math.floor(num(maxChars, 2000)))
  const HEAD = '<memory>\n以下是系统按当前对话自动唤起的跨会话记忆(历史日记片段,供参考而非绝对真理):\n\n'
  const FOOT = '</memory>'
  const valid = (blocks || [])
    .filter((b) => b && typeof b === 'object' && typeof b.text === 'string' && typeof b.score === 'number')
    .sort((a, b) => b.score - a.score) // 降序;store 本就降序,这里防御性重排
  const budget = limit - HEAD.length - FOOT.length
  if (budget <= 0) return HEAD + FOOT // 极端小预算:只保留头尾标记
  let out = HEAD
  let room = budget
  let idx = 0
  for (const b of valid) {
    const file = b.file == null ? '' : String(b.file)
    const head = `\n[${idx + 1}] (score ${b.score.toFixed(2)} · ${file})\n`
    const overlay = head.length + 1 // 行首 + 末尾换行
    if (overlay + b.text.length <= room) {
      out += head + b.text + '\n'
      room -= overlay + b.text.length
      idx++
    } else if (idx === 0) {
      // 最高分块本身放不下 → 保留截断版(§1.2"至少保留最高分块的截断版")
      const avail = room - overlay
      if (avail > 0) {
        out += head + b.text.slice(0, avail) + '\n'
        idx++
      }
      break
    } else {
      break // 当前及后续分数更低 → 丢弃
    }
  }
  return out + FOOT
}