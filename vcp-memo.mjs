// vcp-memo.mjs —— DSH 跨会话长期记忆插件入口（规格 §8）
// 零 npm 依赖：只用 node 内置模块与全局 fetch；文件/网络 IO 直接用 node（不走 ctx.fs，
// 数据目录在工作区沙箱外，属于刻意设计）。
// 职责：装配 embedding 客户端与 store，注册 save_memory / recall_memory / memory_admin /
// update_memory 四个工具；索引 sig 不一致时（store.ready reject）工具照常注册，调用时返回拒绝服务文本。

import { readFileSync } from 'node:fs'
import path from 'node:path'
import { createEmbedder } from './engine/embed.mjs'
import { createInjector } from './engine/inject.mjs'
import { openStore } from './engine/store.mjs'

export const name = 'vcp-memo'
export const inject = ['tools']

// 拒绝服务错误消息（规格 §6.4.1 的固定措辞；与 engine/store.mjs 的 buildRefusal 一致）
const REFUSAL_BANNER = '嵌入模型签名已变'

// 四工具共用 output 契约：schema 声明为字符串，render 把 JSON 字符串包成 text 块
const TEXT_OUTPUT = {
  schema: { type: 'string' },
  render: (args, value) => [{ type: 'text', text: String(value) }],
}

// P1.5 §3:记忆使用规范 prompt section 文本(照 SPEC-INJECT §3 原文,verbatim)
const DISCIPLINE_TEXT = [
  '## 记忆使用规范',
  '1. <memory> 区块(若存在)是系统按当前对话自动唤起的跨会话记忆,先读它再回答;它是历史记录而非绝对真理,与当前事实冲突时以当前事实为准,必要时用 save_memory 写入修正。',
  '2. 结束回合前,若有值得长期记住的经历/结论/决定/偏好,调用 save_memory(title, content, tags[]) 写入;tags 保序、优先复用已召回记忆中的既有 Tag、不堆砌同义词。',
  '3. 发现记忆过时或错误时,优先用 update_memory 修正而不是重复 save。',
  '4. 需要主动回忆更多时,调用 recall_memory(query, k?, truncate?)。',
].join('\n')

// 取错误消息的通用兜底
function errMsg(err) {
  return err && err.message ? err.message : String(err)
}

// 数值钳制
function clampInt(v, min, max, fallback) {
  const n = Math.round(Number(v))
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback
}
function clampNum(v, min, max, fallback) {
  const n = Number(v)
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback
}

// 标签清洗（规格 §3 Tag 纪律）：只做标点清洗——全角逗号/顿号→", "、去首尾空白、
// 折叠连续空白；不排序、不去重、不改写；保序写出。
function cleanTags(tags) {
  return tags.map((t) => String(t).replace(/[，、]/g, ', ').replace(/\s+/g, ' ').trim())
}

export async function apply(ctx, config = {}) {
  // 1) embedding 客户端（默认值对齐 cordis.patch.yml 的示例配置，便于最小配置启动）
  const embedding = {
    baseUrl: 'http://127.0.0.1:11434/v1',
    model: 'bge-m3',
    dimension: 1024,
    ...(config.embedding || {}),
  }
  const embedder = createEmbedder(embedding)

  // 日志：优先 ctx.logger，缺失时回退 console（规格 §8 的写法）
  const log = (level, msg) => {
    const logger = ctx.logger || console
    return (logger[level] || console[level] || console.log)(msg)
  }

  // 2) 打开存储（迁移 readiness 检查发生在 store.ready 内）
  const memory = config.memory || {}
  const dataRoot = config.dataRoot || '/home/lyy/vcp-memo-data'
  const store = await openStore(
    {
      dataRoot,
      agentName: config.agentName || 'dsh',
      dimension: embedder.dimension,
      sig: embedder.sig,
      watch: config.watch !== undefined ? !!config.watch : true,
      chunker: config.chunker,
      // P1 §6:可选 tagmemo 节(缺省即用:{ epa:{minTags:8,clusterCount:12,maxBasisDim:32},
      // pyramid:{maxLevels:3,topK:10,minEnergyRatio:0.1} },默认值在 taglayer 内应用)
      tagmemo: config.tagmemo,
    },
    embedder,
    log,
  )

  // 3) 等待 ready：sig 不一致等问题会 reject——捕获为 fault，
  //    不阻止工具注册；每次调用返回含 error 字段的拒绝服务 JSON 文本。
  let fault = null
  try {
    await store.ready
  } catch (err) {
    fault = errMsg(err)
    log('error', `vcp-memo 拒绝服务：${fault}`)
  }

  // 兼容 store 的两种实现姿态（规格 §6.4.1 允许"ready reject 或 recall/save 抛错"）：
  // 若 store 选择"ready resolve + 操作抛错"，stats() 这类只读操作不会抛——
  // 入口自行核对 index/meta.json 的签名（只用 node 内置模块，§8 允许），
  // 不一致时同样进入 fault 拒绝服务态，保证 memory_admin 的每次调用也返回拒绝文本。
  if (!fault) {
    let meta = null
    try {
      meta = JSON.parse(readFileSync(path.join(dataRoot, 'index', 'meta.json'), 'utf8'))
    } catch {
      meta = null // meta 缺失：store 已自行处理重建/初始化，这里不越权
    }
    if (meta && (meta.sig !== embedder.sig || meta.dimension !== embedder.dimension)) {
      const oldKey = meta.sig !== embedder.sig ? meta.sig : `dimension=${meta.dimension}`
      const newKey = meta.sig !== embedder.sig ? embedder.sig : `dimension=${embedder.dimension}`
      fault = `${REFUSAL_BANNER}（旧 ${oldKey} → 新 ${newKey}），索引与日记语义空间不再一致；确认切换模型请调用 memory_admin 的 rebuild 全量重建`
      log('error', `vcp-memo 拒绝服务：${fault}`)
    }
  }

  // ---- P1.5 §3:被动注入接线("回忆先于意识") ----
  // createInjector 在 store 就绪(无论成败)后创建;fault 读取器让拒绝服务态下
  // 监听器原样放行——不构造查询、不调 recall,注入永远不碰坏对话轮(SPEC-INJECT §1.1)。
  const injector = createInjector({
    store,
    embedder,
    config: config.injection, // 全部可选,默认值在 inject.mjs 内应用(§1)
    log,
    fault: () => fault,
  })
  // ctx.on 注册随 Fiber 生命周期撤销(§3)
  ctx.on('agent/pre-step', injector.handler)
  // vcp-memo:discipline prompt section(order 400):systemPrompt 服务缺失则跳过(记 log);
  // section 返回 disposer,经 ctx.effect 挂到 Fiber 生命周期(§0,同 persona preset 范式)。
  const systemPrompt = ctx.get('systemPrompt')
  if (systemPrompt && typeof systemPrompt.section === 'function') {
    ctx.effect(() => systemPrompt.section({ name: 'vcp-memo:discipline', order: 400, text: DISCIPLINE_TEXT }))
  } else {
    log('warn', 'vcp-memo: systemPrompt 服务缺失,跳过 vcp-memo:discipline prompt section 注册')
  }

  const kDefault = memory.kDefault !== undefined ? Number(memory.kDefault) : 6
  const truncateDefault = memory.truncate !== undefined ? Number(memory.truncate) : 0.4

  // ---- save_memory ----
  const saveTool = {
    name: 'save_memory',
    description:
      '把值得长期记住的经历/结论/决定/偏好写入跨会话长期记忆。写入后立即可用；会在后台进入向量索引。',
    parameters: {
      type: 'object',
      properties: {
        title: { type: 'string', description: '标题：一句话概括这条记忆的主题' },
        content: { type: 'string', description: '正文：经历、结论、决定、偏好等，保留细节与背景' },
        tags: {
          type: 'array',
          items: { type: 'string' },
          minItems: 1,
          description: '标签数组（必填，至少 1 个）；顺序即叙事方向，按原序保序写入',
        },
      },
      required: ['title', 'content', 'tags'],
    },
    output: TEXT_OUTPUT,
    async execute(args = {}) {
      try {
        if (fault) return JSON.stringify({ ok: false, error: fault })
        if (!Array.isArray(args.tags) || args.tags.length === 0)
          return JSON.stringify({ ok: false, error: 'tags 参数必填且至少 1 个' })
        if (typeof args.content !== 'string' || !args.content.trim())
          return JSON.stringify({ ok: false, error: 'content 参数必填' })
        const tags = cleanTags(args.tags)
        const r = await store.saveDiary({ title: args.title ?? '', content: args.content, tags })
        return JSON.stringify({
          ok: true,
          file: r.file,
          tags: r.tags !== undefined ? r.tags : tags, // store 自算的清洗结果优先，兜底用本地结果
        })
      } catch (err) {
        return JSON.stringify({ ok: false, error: errMsg(err) })
      }
    },
  }

  // ---- recall_memory ----
  const recallTool = {
    name: 'recall_memory',
    description:
      '按语义检索跨会话长期记忆（历史日记片段）。会话开始或进入新话题时先调用。',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: '检索查询：一段要回忆的内容描述' },
        k: {
          type: 'integer',
          minimum: 1,
          maximum: 20,
          default: kDefault,
          description: '返回片段条数（1..20）',
        },
        truncate: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          default: truncateDefault,
          description: '相似度下限（0..1），低于此阈值的片段不返回',
        },
        // P2 §6.4:显式 tags = Core Tag 强制入选(wave 路径;可空,缺省自动感应)
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: '可选：显式指定核心标签（Core Tag 强制入选），如 ["澜沧计划"]',
        },
      },
      required: ['query'],
    },
    output: TEXT_OUTPUT,
    async execute(args = {}) {
      try {
        if (fault) return JSON.stringify({ error: fault })
        if (typeof args.query !== 'string' || !args.query.trim())
          return JSON.stringify({ error: 'query 参数必填' })
        const tags = Array.isArray(args.tags)
          ? args.tags.map((t) => String(t)).filter((t) => t.trim().length > 0)
          : undefined
        const r = await store.recall({
          query: args.query,
          k: clampInt(args.k, 1, 20, kDefault),
          truncate: clampNum(args.truncate, 0, 1, truncateDefault),
          tags,
        })
        return JSON.stringify(r) // 结构与规格 §6.2 一致：{ blocks, stats }
      } catch (err) {
        return JSON.stringify({ error: errMsg(err) })
      }
    },
  }

  // ---- memory_admin ----
  const adminTool = {
    name: 'memory_admin',
    description:
      '记忆库管理：stats 查看统计；rebuild 全量重建索引（更换 embedding 模型后必须执行）。',
    parameters: {
      type: 'object',
      properties: {
        op: {
          type: 'string',
          enum: ['stats', 'rebuild'],
          description: 'stats：查看统计；rebuild：全量重建索引（更换 embedding 模型后必须执行）',
        },
      },
      required: ['op'],
    },
    output: TEXT_OUTPUT,
    async execute(args = {}) {
      try {
        // rebuild 是 sig 不一致后的官方修复通道（拒绝服务文本正是引导用户调它），
        // 即使在 fault 态也必须放行；重建成功后 store 已用新签名重写 meta 并解除内部
        // refusal，这里同步清除本地 fault，save/recall 随即恢复服务。
        if (args.op === 'rebuild') {
          await store.rebuild()
          fault = null
          const s = await store.stats()
          return JSON.stringify({ ok: true, op: 'rebuild', stats: s })
        }
        if (fault) return JSON.stringify({ error: fault })
        const s = await store.stats()
        return JSON.stringify({ ok: true, op: 'stats', stats: s })
      } catch (err) {
        return JSON.stringify({ error: errMsg(err) })
      }
    },
  }

  // ---- update_memory（P3 §3：锚点式修正，复用 P0 延伸项 store.updateDiary，不复制逻辑）----
  const updateTool = {
    name: 'update_memory',
    description:
      '锚点式修正已有记忆:target 是至少 15 字符的原文片段,replace 是替换内容。命中多篇或不命中会报错,需提供更精确的 target。',
    parameters: {
      type: 'object',
      properties: {
        target: {
          type: 'string',
          minLength: 15,
          description: '锚点：至少 15 字符、须在某一篇日记正文中恰好出现一次的连续原文片段',
        },
        replace: { type: 'string', description: '替换内容：将 target 命中处替换为该文本（只做子串替换，禁止整文件覆写）' },
      },
      required: ['target', 'replace'],
    },
    output: TEXT_OUTPUT,
    async execute(args = {}) {
      try {
        if (fault) return JSON.stringify({ ok: false, error: fault })
        // 参数校验与命中语义全部交给 store.updateDiary（同一条中文报错，保持单一实现源）
        const r = await store.updateDiary({ target: args.target, replace: args.replace })
        return JSON.stringify({ ok: true, file: r.file, replaced: r.replaced })
      } catch (err) {
        return JSON.stringify({ ok: false, error: errMsg(err) })
      }
    },
  }

  // 4) 注册工具：register 必须包在 ctx.effect 里（随 Fiber 生命周期撤销）
  ctx.effect(() => {
    ctx.tools.register(saveTool)
    ctx.tools.register(recallTool)
    ctx.tools.register(adminTool)
    ctx.tools.register(updateTool)
  })
}