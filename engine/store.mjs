// engine/store.mjs — 存储 + 内存索引 + 暴力余弦 KNN + 文件监听 + 全量重建
// 见 SPEC.md §6。零 npm 依赖：仅 node: 内置模块。
// 设计要点：
// - 日记 Markdown 文件是真相源，index/ 是派生产物，可随时重建；
// - 索引队列串行（单写者），chunk 向量失败填 null 跳过，绝不写错误向量；
// - 写入全部原子化（临时文件 + rename）；
// - watcher 对每个目录 fs.watch（Linux 不支持递归），并为 .md 文件补挂单文件
//   watch，以覆盖"原地覆写"这类不触发目录 rename 事件的编辑；
// - chunks.jsonl 重写：队列空闲时立即落盘，繁忙时 2s 去抖合并（满足 §6.5）。

import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import { chunkText } from '../core/chunker.mjs'
import { createTagLayer } from './taglayer.mjs'
// P2 §6:taggraph 生命周期 + wave 查询融合 + ResultDeduplicator 候选去重
import { createTagGraph } from './taggraph.mjs'
import { applyTagBoost, mergeWaveCfg } from './wave.mjs'
import { deduplicate } from '../core/ResultDeduplicator.mjs'

const FLUSH_DEBOUNCE_MS = 2000 // chunks.jsonl/meta.json 去抖重写时间
const EVENT_DEBOUNCE_MS = 500 // watcher 事件去抖时间
const REFUSAL_BANNER = '嵌入模型签名已变'

// ---------- 小工具 ----------

function md5(text) {
  return crypto.createHash('md5').update(text, 'utf8').digest('hex')
}

function pad2(n) {
  return String(n).padStart(2, '0')
}

function fmtDate(d) {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`
}

function fmtTime(d) {
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`
}

// 文件名用时间：HH_MM_SS（下划线，见 §2/§6.1 文件名契约）
function fmtFileTime(d) {
  return fmtTime(d).replace(/:/g, '_')
}

// 标题做文件名安全清洗：去掉路径非法字符与控制字符、折叠空白、防隐藏文件
function sanitizeTitle(t) {
  return String(t)
    .replace(/[\\/:*?"<>|\u0000-\u001f]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^\.+/, '')
    .replace(/\.+$/, '')
}

// Tag 清洗：全角逗号/顿号 → ', '，去首尾空白、折叠连续空白。
// 保序、不排序、不去重、不改写（§3 Tag 纪律）。
function cleanTag(raw) {
  return String(raw).replace(/[，、]/g, ', ').replace(/\s+/g, ' ').trim()
}

// 原子写盘：写临时文件 + rename
function atomicWrite(file, data) {
  const dir = path.dirname(file)
  const tmp = path.join(dir, `.${path.basename(file)}.${process.pid}.${Date.now()}.tmp`)
  fs.writeFileSync(tmp, data)
  try {
    fs.renameSync(tmp, file)
  } catch (err) {
    fs.rmSync(tmp, { force: true })
    throw err
  }
}

// 归一化向量（复制后原地归一化；零向量保持原样）。非法输入返回 null。
function normalize(v) {
  if (!v) return null
  const arr = v instanceof Float32Array ? v : Array.isArray(v) ? Float32Array.from(v) : null
  if (!arr) return null
  let sum = 0
  for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i]
  const n = Math.sqrt(sum)
  if (n > 0) {
    for (let i = 0; i < arr.length; i++) arr[i] /= n
  }
  return arr
}

function dot(a, b) {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

// 递归扫描目录树下的 .md，返回相对 dataRoot 的路径（Linux 分隔符）
function scanMdFiles(rootAbs, dataRoot) {
  const out = []
  const walk = (dir) => {
    let ents
    try {
      ents = fs.readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const e of ents) {
      const full = path.join(dir, e.name)
      if (e.isDirectory()) walk(full)
      else if (e.isFile() && e.name.endsWith('.md')) out.push(path.relative(dataRoot, full))
    }
  }
  walk(rootAbs)
  return out
}

// 统计子串出现次数（非重叠）
function countOccurrences(text, needle) {
  let c = 0
  let at = 0
  while ((at = text.indexOf(needle, at)) !== -1) {
    c++
    at += needle.length
  }
  return c
}

// ---------- openStore ----------

export async function openStore(config, embedder, log) {
  if (!config || typeof config.dataRoot !== 'string') {
    throw new TypeError('openStore: config.dataRoot 必须是非空字符串')
  }
  if (typeof config.agentName !== 'string' || !config.agentName) {
    throw new TypeError('openStore: config.agentName 必须是非空字符串')
  }
  const dataRoot = config.dataRoot
  const agentName = config.agentName
  const diariesDir = path.join(dataRoot, 'diaries')
  const indexDir = path.join(dataRoot, 'index')
  const chunksPath = path.join(indexDir, 'chunks.jsonl')
  const metaPath = path.join(indexDir, 'meta.json')
  const watchEnabled = config.watch !== false // 默认开启
  const chunkerOpts = config.chunker || {}

  // 向量语义空间取自 embedder（真实来源），config 值仅作缺省回退
  const effSig = (embedder && embedder.sig) || config.sig
  const effDimension = (embedder && embedder.dimension) || config.dimension
  if (!effSig || !effDimension) {
    throw new TypeError('openStore: 无法确定 sig/dimension（需来自 embedder 或 config）')
  }

  const logger = typeof log === 'function' ? log : () => {}
  const L = (level, msg) => {
    try {
      logger(level, msg)
    } catch {
      /* log 自身失败不影响主流程 */
    }
  }

  // ---------- Tag 层(P1 §6) ----------
  // 与 chunks 索引共享 embedder 与语义空间(sig/dimension);load() 随 ready 进行。
  // P2 §6.2 门槛字面读取:"tagmemo.enabled && 图可用 && 有向量化 tag"——config 里没有
  // tagmemo 节时 tagmemo.enabled 为 undefined(假)→ wave 路径不启用,recall 走 P1
  // (纯 KNN + 诊断字段);生产部署由 cordis.patch.yml 的 tagmemo 节(SPEC-P2 §7 全表)
  // 显式开启,节内 enabled 缺省 true(§1 表内默认值)。此政策使既有 P0/P1/P1.5 测试
  // (裸 store 配置、不含 tagmemo 节)保持零回归,同时保留 P2 增强路径的正式开关。
  const tagmemoSection = config.tagmemo
  const tagmemo = tagmemoSection || {}
  const tagLayer = createTagLayer({
    dataRoot,
    dimension: effDimension,
    sig: effSig,
    embedder,
    epaConfig: tagmemo.epa,
    pyramidConfig: tagmemo.pyramid,
    log: L,
  })

  // ---------- P2 §6.1:taggraph 生命周期(派生资产,不落盘,启动重建) ----------
  // waveCfg = config.tagmemo 节(§1 全表)与生产默认合并(缺省即生产值);
  // tagGraph 以写时替换发布整代图,recall 只读 snapshot,永不读半成品。
  const waveCfg = mergeWaveCfg(tagmemoSection == null ? { enabled: false } : tagmemo)
  const tagGraph = createTagGraph({ cooccurrence: waveCfg.cooccurrence, kernel: waveCfg.kernel }, L)
  // 与"当前已发布图"同源的 tag 记录(id=数组下标,与 tagLayer/taggraph 契约一致);
  // 由 refreshTagGraph 在每次重建/指纹命中后刷新,recall 用它解析能量场节点与 Core 名称。
  let graphTagRecords = []

  // ---------- 可变状态 ----------
  let currentIndex = [] // { id, file, chunkIndex, content, vector: Float32Array(已归一化) }
  let nextId = 1
  let pendingCount = 0
  let queueTail = Promise.resolve()
  let dirty = false
  let flushTimer = null
  let eventTimer = null
  let refusalMsg = null
  let lastRebuild = 0
  let closed = false
  let watcherStarted = false
  let tagDirty = false // tag 层有待 flush 的变更(updateFile/removeFile 置位)
  let tagFlushing = false // tagLayer.flush 正在运行(单飞)
  let tagFlushPending = false // flush 期间又来了变更,结束后补一次
  const fileMd5 = new Map() // relPath -> md5（watcher 变更比对基线）
  const watchHandles = new Map() // 目录 abs -> FSWatcher
  const fileWatches = new Map() // relPath -> FSWatcher（单文件 watch）
  const pendingEvents = new Map() // relPath -> true（500ms 去抖桶）

  const relOf = (abs) => path.relative(dataRoot, abs)

  // ---------- 索引队列（串行单写者） ----------
  // 返回的 promise 可被调用方 await（rebuild 等），未 await 也不会产生未处理拒绝。
  function enqueue(task, label = 'task') {
    pendingCount++
    const run = queueTail.then(() => task())
    queueTail = run.catch((err) => L('error', `队列任务(${label})失败: ${err && err.message || err}`))
    run
      .finally(() => {
        pendingCount--
        maybeFlushNow()
        maybeTagFlush() // P1 §6:队列空闲时 tagLayer.flush()
      })
      .catch(() => {}) // 抑制 finally 链上的未处理拒绝
    return run
  }

  // ---------- Tag 层落盘/重训(P1 §6) ----------
  // 只在有实际变更(tagDirty)或显式 force 时启动;force 用于启动期把
  // "load 时 tagHash 不一致而滞留未训练"的 EPA 拉起(flush 内部自行判断是否有活)。
  // tagLayer.flush 是异步的 embed+训练,不进入索引队列(避免互相阻塞),单飞防重入;
  // flush 期间又来了 tag 变更 → 置 tagFlushPending,结束后补一次。
  // embed 失败(整批 null)后 tagDirty 已被清除 → 不会无限重试,重试发生在下一次
  // 真实 tag 变更(符合 §3.2 "下次 flush 重试")。
  function maybeTagFlush(force = false) {
    if (!tagLayer || closed) return
    if (!tagDirty && !force) return
    if (tagFlushing) {
      // 能走到这里只有两种情况:tagDirty 为 true(非 force 调用已被上一行
      // 早退挡过)或显式 force——无论哪种,flush 期间都来了新变更,结束后必须补一次。
      // (原实现只在 force 时置 pending:连续快速 saveDiary 时,前一篇的 flush 还在跑,
      // 后一篇的非 force 调用在此被丢弃 → 最后一批新 tag 永远等不到向量化。)
      tagFlushPending = true
      return
    }
    tagFlushing = true
    tagDirty = false
    tagLayer
      .flush()
      .catch((err) => L('error', `tagLayer.flush 失败: ${err && err.message || err}`))
      .finally(() => {
        tagFlushing = false
        // P2 §6.1:flush 后联动的 taggraph 重建(内容指纹未变则跳过;异步排队,不阻塞)
        refreshTagGraph().catch(() => {})
        if (tagFlushPending) {
          tagFlushPending = false
          maybeTagFlush(true)
        }
      })
  }

  // ---------- P2 §6.1:taggraph 重建辅助 ----------
  // tagLayer 不暴露内存记录表(另一代理的交付物,不改),这里读它持久化的
  // tags.jsonl(SPEC-P1 §3.2 契约格式)重建同构记录;id=行序=tagLayer 的数组下标。
  // 读取失败按空记录处理(→ 空图 → recall 回退 P1),绝不令图生命周期抛错。
  function readTagRecords() {
    try {
      const content = fs.readFileSync(path.join(indexDir, 'tags.jsonl'), 'utf8')
      const out = []
      for (const line of content.split(/\r?\n/)) {
        const row = line.trim()
        if (!row) continue
        const parsed = JSON.parse(row)
        out.push({
          id: out.length,
          name: parsed.name,
          vector: parsed.vector ? Float32Array.from(parsed.vector) : null,
          occurrences: Array.isArray(parsed.occurrences) ? parsed.occurrences : [],
        })
      }
      return out
    } catch (err) {
      if (err.code !== 'ENOENT') L('warn', `读取 tags.jsonl 失败(按空记录继续): ${err.message}`)
      return []
    }
  }

  // 重建(force)或指纹联动重建(rebuildIfChanged);成功后刷新 graphTagRecords,
  // 保证 recall 读到的记录与当前已发布图代数同源(id 与图节点一一对应)。
  async function refreshTagGraph(opts = {}) {
    try {
      const records = readTagRecords()
      if (opts.force) await tagGraph.rebuild(records)
      else await tagGraph.rebuildIfChanged(records)
      graphTagRecords = records
    } catch (err) {
      L('error', `taggraph 重建异常(保留上一代): ${err && err.message || err}`)
    }
  }

  // wave 路径的 deps 装配(SPEC-P2 §4:applyTagBoost 接口)
  function buildWaveDeps(tagRecords, graph) {
    return {
      epa: tagLayer.epa,
      pyramid: tagLayer.pyramid,
      tagLayer: {
        records: () => tagRecords,
        byName: (name) =>
          tagRecords.find(
            (r) => r.name && String(r.name).toLowerCase() === String(name).toLowerCase()
          ) || null,
      },
      graph,
      cfg: waveCfg,
    }
  }

  // 补充捞取的文件集合:能量场 top 节点(core/seed/emergent)所在文件 + 显式 Core Tag
  // (虚拟补全)所在文件;按能量降序截前 50(与 spike.maxEmergentNodes 同界,防膨胀)。
  function collectFieldFiles(energyField, tagRecords, coreTagIds) {
    const nodes = [...energyField.entries()]
      .filter(([, e]) => e > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 50)
    const files = new Set()
    const addRec = (id) => {
      const r = tagRecords[id]
      for (const o of (r && r.occurrences) || []) {
        if (o && o.file) files.add(o.file)
      }
    }
    for (const [id] of nodes) addRec(id)
    for (const id of (coreTagIds || [])) addRec(id)
    return files
  }

  // ── P2 §6.2:wave 召回路径 ──
  // 条件:taggraph 可用(enabled && 已发布代 && 有向量化 tag 达标)&& EPA 已训练
  //      (§4 [1] 需要 project/resonance,未训练时 P1 实例返回 null,动态链无从计算)
  // 成功 → 返回 wave 结果(调用方组装响应);任何失败/不可用 → null(调用方回退 P1)。
  async function tryWaveRecall(q, idx, tags, truncate, kWave) {
    try {
      if (!waveCfg.enabled) return null // tagmemo.enabled=false → 整体关闭
      const graph = tagGraph.snapshot
      const gDiag = graph.diagnostics || {}
      if (!(graph.generation > 0 && gDiag.sufficientVectorizedTags === true && graph.kernel.size > 0)) {
        return null // 图不可用(门槛未满足/空图/未构建)→ 回退 P1
      }
      if (!(tagLayer.epa && tagLayer.epa.trained)) return null // EPA 未训练 → 动态链不可用
      if (!graphTagRecords.some((r) => r.vector !== null)) return null // 无向量化 tag

      const coreTagNames = Array.isArray(tags)
        ? tags.map((t) => String(t).trim()).filter(Boolean)
        : []
      const w = await applyTagBoost(q, waveCfg.baseTagBoost, coreTagNames, buildWaveDeps(graphTagRecords, graph))
      if (!w.info) return null // 无种子/融合不可用 → P1

      const fused = normalize(
        w.vector instanceof Float32Array ? Float32Array.from(w.vector) : w.vector
      )
      if (!fused) return null

      // 1. 融合向量 KNN(与 P0 同语义:s ≥ truncate 才入候选,分数就是向量余弦)
      // 2. 补充捞取:能量场 top 节点所在文件的 chunk,若不在 KNN 结果中,以融合向量
      //    余弦补入候选。只加分不罚分:补充候选【豁免 truncate 下限】——它们是结构
      //    传导捞回来的"KNN 漏掉"的证据(SPEC-P2 §8.6 的 A/B 验收),分数就是它们的
      //    真实余弦,不做加权;标记 viaStructure:true,不占语义块的 Top-K 名额
      //    (单独以 maxSupplement 补位,见下方 blocks 组装)。
      const seen = new Set()
      const candidates = []      // ≥ truncate 的语义候选
      const supplements = []     // 结构补充候选(豁免 truncate,标记 viaStructure)
      const pushCandidate = (c, list) => {
        const key = `${c.file}\u0000${c.chunkIndex}`
        if (seen.has(key)) return
        seen.add(key)
        list.push(c)
      }
      for (const c of idx) {
        const s = dot(fused, c.vector)
        if (s >= truncate) {
          pushCandidate({ file: c.file, chunkIndex: c.chunkIndex, score: s, text: c.content, vector: c.vector }, candidates)
        }
      }
      const fieldFiles = collectFieldFiles(w.energyField, graphTagRecords, w.info.coreTagIds)
      if (fieldFiles.size > 0) {
        for (const c of idx) {
          if (!fieldFiles.has(c.file)) continue
          const s = dot(fused, c.vector)
          if (s >= truncate) {
            pushCandidate({ file: c.file, chunkIndex: c.chunkIndex, score: s, text: c.content, vector: c.vector }, candidates)
          } else {
            pushCandidate({ file: c.file, chunkIndex: c.chunkIndex, score: s, text: c.content, vector: c.vector, viaStructure: true }, supplements)
          }
        }
      }
      candidates.sort((a, b) => b.score - a.score)
      supplements.sort((a, b) => b.score - a.score)

      // 3. ResultDeduplicator 候选去重(多身份硬去重 + 0.92 余弦近重复抑制;
      //    queryVector=融合向量,§6.2);语义候选与结构补充分别去重
      const deduped = await deduplicate(candidates, fused, {
        semanticThreshold: waveCfg.dedup.semanticThreshold,
        stage: 'wave',
      })
      const dedupedSupp = await deduplicate(supplements, fused, {
        semanticThreshold: waveCfg.dedup.semanticThreshold,
        stage: 'wave-supplement',
      })
      // 4. Top-K + 结构补位(SPEC-P2 §6.2 修订):语义块严格 Top-K;结构补充块以
      //    maxSupplement(默认 2)追加并标记 viaStructure,总块数 ≤ k + maxSupplement
      const maxSupplement = Math.max(0, Math.floor(Number(waveCfg.maxSupplement ?? 2)))
      const blocks = [
        ...deduped.slice(0, Math.max(1, Math.floor(kWave) || 1)),
        ...dedupedSupp.slice(0, maxSupplement),
      ].map((c) => ({
        file: c.file,
        chunkIndex: c.chunkIndex,
        score: c.score,
        text: c.text,
        ...(c.viaStructure ? { viaStructure: true } : {}),
      }))

      return {
        blocks,
        matchedTags: w.info.matchedTags, // §4.9 完整来源版(含 sourceType/hop/notation)
        candidates: deduped.length,
        epaStats: {
          trained: true,
          logicDepth: w.info.epa.logicDepth,
          entropy: w.info.epa.entropy,
          resonance: w.info.epa.resonance,
          dominantAxes: w.info.epa.dominantAxes,
        },
        pyramidStats: {
          depth: w.info.pyramid.depth,
          coverage: w.info.pyramid.coverage,
          novelty: w.info.pyramid.novelty,
          coherence: w.info.pyramid.coherence,
          activation: w.info.pyramid.activation,
        },
        // §6.2:stats.wave
        stats: {
          alpha: w.info.alpha,
          effectiveBoost: w.info.boostFactor,
          logicDepth: w.info.epa.logicDepth,
          resonance: w.info.epa.resonance,
          coverage: w.info.pyramid.coverage,
          activation: w.info.pyramid.activation,
          seeds: w.info.seeds,
          emergentCount: w.info.emergentCount,
          fieldNodes: w.info.fieldNodes,
          graphGeneration: w.info.graphGeneration,
        },
      }
    } catch (err) {
      // §6.3:wave 路径任何异常 → 记 log 回退 P1 重算(不得返回半成品)
      L('warn', `wave 路径异常,回退 P1: ${err && err.message || err}`)
      return null
    }
  }

  // ---------- 落盘 ----------

  function writeIndexFiles() {
    const snapshot = [...currentIndex].sort((a, b) => a.id - b.id)
    const lines = snapshot.map((c) =>
      JSON.stringify({
        id: c.id,
        file: c.file,
        chunkIndex: c.chunkIndex,
        content: c.content,
        vector: Array.from(c.vector),
      }),
    )
    atomicWrite(chunksPath, lines.length ? lines.join('\n') + '\n' : '')
    atomicWrite(
      metaPath,
      JSON.stringify({ sig: effSig, dimension: effDimension, chunkCount: snapshot.length, updatedAt: Date.now() }, null, 2) + '\n',
    )
  }

  // 队列空闲时立即落盘；繁忙时 2s 去抖合并（§6.5）。
  function maybeFlushNow() {
    if (closed) return
    if (!dirty) return
    if (pendingCount > 0) {
      scheduleFlush()
      return
    }
    if (flushTimer) {
      clearTimeout(flushTimer)
      flushTimer = null
    }
    dirty = false
    try {
      writeIndexFiles()
    } catch (err) {
      L('error', `chunks.jsonl/meta.json 落盘失败: ${err.message}`)
      dirty = true
    }
  }

  function scheduleFlush() {
    if (closed || flushTimer) return
    flushTimer = setTimeout(() => {
      flushTimer = null
      maybeFlushNow()
    }, FLUSH_DEBOUNCE_MS)
  }

  // ---------- 索引操作 ----------

  function removeChunks(rel) {
    const before = currentIndex.length
    currentIndex = currentIndex.filter((c) => c.file !== rel)
    if (currentIndex.length !== before) dirty = true
  }

  // 文件入队索引任务：读文件 → md5 比对 → 重切分 → embed → 提交。
  // 提交前复核 md5：处理期间文件又被改动则丢弃本次结果（等待下一次事件）。
  async function indexFile(rel, opts = {}) {
    if (refusalMsg) return // 拒绝服务期间不入索引
    const abs = path.join(dataRoot, rel)
    let text
    try {
      text = fs.readFileSync(abs, 'utf8')
    } catch {
      // 文件已不存在（可能被并发删除）→ 移除旧索引
      if (!opts.force) {
        removeChunks(rel)
        tagLayer.removeFile(rel) // P1 §6:同步清理该文件的 tag 发生记录
        tagDirty = true
        fileMd5.delete(rel)
        closeFileWatcher(rel)
      }
      return
    }
    const hash = md5(text)
    const prev = fileMd5.get(rel)
    if (!opts.force && prev === hash) return // 内容未变，跳过
    let chunks = chunkText(text, chunkerOpts)
    let vectors
    try {
      vectors = await embedder.embed(chunks)
    } catch (err) {
      L('error', `文件 ${rel} 的 embedding 调用失败（${chunks.length} chunks），该文件跳过`)
      vectors = null
    }
    if (!vectors || !Array.isArray(vectors)) vectors = chunks.map(() => null)
    // 提交前复核：文件在 embed 期间是否又被改写
    let curHash = null
    try {
      curHash = md5(fs.readFileSync(abs, 'utf8'))
    } catch {
      curHash = null
    }
    if (curHash !== hash) {
      L('debug', `文件 ${rel} 处理期间又被改动，本次结果丢弃`)
      return
    }
    removeChunks(rel) // 删除旧 chunk（重建期间 recall 仍可读旧索引，提交点才生效）
    chunks.forEach((content, i) => {
      const raw = vectors[i]
      if (!raw) {
        L('warn', `文件 ${rel} 的 chunk ${i} 向量失败，跳过该 chunk`)
        return
      }
      const v = normalize(raw)
      if (!v) return
      currentIndex.push({ id: nextId++, file: rel, chunkIndex: i, content, vector: v })
    })
    fileMd5.set(rel, hash)
    // P1 §6:提交 chunk 后同步 tag 层(解析末行 Tag,更新 occurrences;队列空闲时 flush)
    tagLayer.updateFile(rel, text)
    tagDirty = true
    if (chunks.length > 0) dirty = true
    ensureFileWatcher(rel) // 有 chunk 即补挂单文件 watch，覆盖原地覆写场景
  }

  function enqueueIndex(rel, opts) {
    return enqueue(() => indexFile(rel, opts || {}), `index:${rel}`)
  }

  function enqueueRemove(rel) {
    return enqueue(() => {
      removeChunks(rel)
      tagLayer.removeFile(rel) // P1 §6:文件删除 → 清理其 tag 发生记录
      tagDirty = true
      fileMd5.delete(rel)
      closeFileWatcher(rel)
      L('info', `日记文件已删除，移除其索引: ${rel}`)
    }, `remove:${rel}`)
  }

  // ---------- 拒绝服务（§6.4.1） ----------

  function buildRefusal(meta) {
    let oldKey
    let newKey
    if (meta.sig !== effSig) {
      oldKey = meta.sig
      newKey = effSig
    } else if (meta.dimension !== effDimension) {
      oldKey = `dimension=${meta.dimension}`
      newKey = `dimension=${effDimension}`
    } else {
      oldKey = meta.sig ?? String(meta.dimension)
      newKey = effSig
    }
    return `${REFUSAL_BANNER}（旧 ${oldKey} → 新 ${newKey}），索引与日记语义空间不再一致；确认切换模型请调用 memory_admin 的 rebuild 全量重建`
  }

  function assertRefusal() {
    if (refusalMsg) {
      const e = new Error(refusalMsg)
      e.code = 'ERR_MEMO_SIG_MISMATCH'
      throw e
    }
  }

  // ---------- 全量重建（§6.2 rebuild / §6.4.2） ----------

  async function doRebuild() {
    const files = scanMdFiles(diariesDir, dataRoot)
    const built = []
    const md5s = new Map()
    let nid = 1
    // P1 §6:重建时同步重建 tag 层——清空内存与派生产物,逐文件 updateFile,最后 flush
    await tagLayer.reset()
    tagDirty = false
    for (const rel of files) {
      let text
      try {
        text = fs.readFileSync(path.join(dataRoot, rel), 'utf8')
      } catch (err) {
        L('warn', `重建: 读取 ${rel} 失败，跳过: ${err.message}`)
        continue
      }
      const chunks = chunkText(text, chunkerOpts)
      let vectors
      try {
        vectors = await embedder.embed(chunks)
      } catch (err) {
        L('error', `重建: ${rel} 的 embedding 调用失败（${chunks.length} chunks），全部跳过`)
        vectors = null
      }
      if (!vectors || !Array.isArray(vectors)) vectors = chunks.map(() => null)
      if (vectors.length !== chunks.length) {
        L('error', `重建: ${rel} 的 embed 返回数量不符（期望 ${chunks.length}）`)
        vectors = chunks.map(() => null)
      }
      md5s.set(rel, md5(text))
      tagLayer.updateFile(rel, text) // P1 §6:重建期间逐文件同步 tag 发生记录
      tagDirty = true
      chunks.forEach((content, i) => {
        const raw = vectors[i]
        if (!raw) {
          L('warn', `重建: ${rel} 的 chunk ${i} 向量失败，跳过该 chunk`)
          return
        }
        const v = normalize(raw)
        if (!v) return
        built.push({ id: nid++, file: rel, chunkIndex: i, content, vector: v })
      })
      if (chunks.length > 0) ensureFileWatcher(rel)
    }
    // 写时替换：整个重建期间 recall 用旧索引，最后才整体切换
    currentIndex = built
    nextId = nid
    for (const [k] of fileMd5) fileMd5.delete(k)
    for (const [k, v] of md5s) fileMd5.set(k, v)
    refusalMsg = null // 重建即换模型后的修复手段
    lastRebuild = Date.now()
    dirty = false
    if (flushTimer) {
      clearTimeout(flushTimer)
      flushTimer = null
    }
    writeIndexFiles()
    // P1 §6:重建即换模型后的修复手段——tag 层也随新 sig 重 embed、重训,落 tags.jsonl/epa.json
    if (tagDirty) {
      await tagLayer.flush()
      tagDirty = false
    }
    // P2 §6.1:全量重建后从新派生的 tag 记录重建 taggraph(整代写时替换)
    if (!refusalMsg) await refreshTagGraph({ force: true })
    L('info', `全量重建完成: ${files.length} 个文件 → ${built.length} 个 chunk`)
    startWatcher()
    return { files: files.length, chunks: built.length }
  }

  // ---------- 文件监听（§6.5） ----------

  function makeDirHandler(dirAbs) {
    return (eventType, filename) => {
      const name = typeof filename === 'string' ? filename : filename ? filename.toString('utf8') : ''
      if (!name) return
      const full = path.join(dirAbs, name)
      if (eventType === 'rename') {
        let st = null
        try {
          st = fs.statSync(full)
        } catch {
          st = null
        }
        if (st && st.isDirectory()) {
          // 新目录：补挂 watcher 并索引其中的 .md
          attachWatcher(full, { indexExisting: true })
          return
        }
        if (!st && watchHandles.has(full)) closeWatcherTree(full) // 目录被移除
      }
      if (name.endsWith('.md')) scheduleMdEvent(relOf(full))
    }
  }

  function attachWatcher(dirAbs, opts = {}) {
    if (closed) return
    if (watchHandles.has(dirAbs)) return
    let w
    try {
      w = fs.watch(dirAbs, { persistent: false }, makeDirHandler(dirAbs))
    } catch (err) {
      L('warn', `watcher 无法监听 ${dirAbs}: ${err.message}`)
      return
    }
    w.on('error', (err) => L('error', `watcher 错误 ${dirAbs}: ${err.message}`))
    watchHandles.set(dirAbs, w)
    if (opts.indexExisting) {
      // 新出现的目录：其中已有 .md 文件不会产生 watch 事件，显式入队（md5 去重）
      for (const rel of scanMdFiles(dirAbs, dataRoot)) enqueueIndex(rel)
    }
    let subs = []
    try {
      subs = fs.readdirSync(dirAbs, { withFileTypes: true }).filter((e) => e.isDirectory()).map((e) => e.name)
    } catch {
      /* 目录可能刚消失，忽略 */
    }
    for (const n of subs) attachWatcher(path.join(dirAbs, n), { indexExisting: false })
  }

  function closeWatcherTree(abs) {
    for (const [dir, w] of [...watchHandles]) {
      if (dir === abs || dir.startsWith(abs + path.sep)) {
        try {
          w.close()
        } catch {
          /* 已关闭 */
        }
        watchHandles.delete(dir)
      }
    }
  }

  function closeAllWatchers() {
    for (const w of watchHandles.values()) {
      try {
        w.close()
      } catch {
        /* 已关闭 */
      }
    }
    watchHandles.clear()
    for (const w of fileWatches.values()) {
      try {
        w.close()
      } catch {
        /* 已关闭 */
      }
    }
    fileWatches.clear()
  }

  // 单文件 watch：覆盖"原地覆写"（不产生目录 rename 事件的编辑）。
  // 属于 §6.5 文件监听的生命线的一部分：watch: false 时（§6.2 openStore config）
  // 整体不监听，任何目录/文件 watcher 都不应建立。
  function ensureFileWatcher(rel) {
    if (closed || !watchEnabled || fileWatches.has(rel)) return
    const abs = path.join(dataRoot, rel)
    let st = null
    try {
      st = fs.statSync(abs)
    } catch {
      return
    }
    if (!st.isFile()) return
    let w
    try {
      w = fs.watch(abs, { persistent: false }, () => scheduleMdEvent(rel))
    } catch (err) {
      L('warn', `无法监听文件 ${rel}: ${err.message}`)
      return
    }
    w.on('error', (err) => L('error', `文件 watcher 错误 ${rel}: ${err.message}`))
    fileWatches.set(rel, w)
  }

  function closeFileWatcher(rel) {
    const w = fileWatches.get(rel)
    if (w) {
      try {
        w.close()
      } catch {
        /* 已关闭 */
      }
      fileWatches.delete(rel)
    }
  }

  function scheduleMdEvent(rel) {
    pendingEvents.set(rel, true)
    if (eventTimer) return
    eventTimer = setTimeout(() => {
      eventTimer = null
      processMdEvents()
    }, EVENT_DEBOUNCE_MS)
  }

  // 去抖批量处理：处理时刻用 stat 判定 add/change 还是 unlink
  function processMdEvents() {
    const batch = [...pendingEvents.keys()]
    pendingEvents.clear()
    for (const rel of batch) {
      let st = null
      try {
        st = fs.statSync(path.join(dataRoot, rel))
      } catch {
        st = null
      }
      if (st && st.isFile()) enqueueIndex(rel)
      else enqueueRemove(rel)
    }
  }

  function startWatcher() {
    if (closed || !watchEnabled || watcherStarted || refusalMsg) return
    watcherStarted = true
    if (!fs.existsSync(diariesDir)) {
      try {
        fs.mkdirSync(diariesDir, { recursive: true })
      } catch (err) {
        L('warn', `创建 diaries 目录失败: ${err.message}`)
      }
    }
    attachWatcher(diariesDir, { indexExisting: false })
  }

  // ---------- 启动加载（ready） ----------

  function readMeta() {
    try {
      return JSON.parse(fs.readFileSync(metaPath, 'utf8'))
    } catch {
      return null
    }
  }

  function loadIndex() {
    let lines = []
    try {
      lines = fs.readFileSync(chunksPath, 'utf8').split('\n').filter(Boolean)
    } catch {
      L('warn', 'chunks.jsonl 读取失败，视为空索引')
    }
    let maxId = 0
    let dropped = 0
    const loaded = []
    for (const line of lines) {
      try {
        const o = JSON.parse(line)
        if (
          !o ||
          typeof o.id !== 'number' ||
          typeof o.content !== 'string' ||
          !Array.isArray(o.vector) ||
          o.vector.length !== effDimension
        ) {
          dropped++
          continue
        }
        const v = normalize(o.vector)
        if (!v) {
          dropped++
          continue
        }
        loaded.push({ id: o.id, file: String(o.file), chunkIndex: Number(o.chunkIndex) || 0, content: o.content, vector: v })
        if (o.id > maxId) maxId = o.id
      } catch {
        dropped++
      }
    }
    if (dropped > 0) L('warn', `chunks.jsonl 中有 ${dropped} 行无效，已丢弃`)
    currentIndex = loaded
    nextId = maxId + 1
    // 孤儿清理：日记文件已不存在的 chunk 移除并重写（覆盖"停机期间被删"场景）
    for (const rel of new Set(currentIndex.map((c) => c.file))) {
      let text = null
      try {
        text = fs.readFileSync(path.join(dataRoot, rel), 'utf8')
      } catch {
        text = null
      }
      if (text === null) {
        removeChunks(rel)
        tagLayer.removeFile(rel) // P1 §6:孤儿文件清理同样清 tag 发生记录
        tagDirty = true
        L('warn', `日记文件缺失，移除其索引: ${rel}`)
      } else {
        fileMd5.set(rel, md5(text)) // 建立 watcher 去重基线
      }
    }
    if (dirty) maybeFlushNow()
  }

  const store = {
    saveDiary({ title = '', content, tags }) {
      assertRefusal() // 拒绝服务期间不落盘
      if (typeof content !== 'string') throw new Error('saveDiary: content 必须是字符串')
      if (!Array.isArray(tags) || tags.length === 0) {
        throw new Error('saveDiary: tags 必须是非空数组（至少 1 个，保序）')
      }
      const cleaned = tags.map(cleanTag).filter((t) => t.length > 0)
      if (cleaned.length === 0) throw new Error('saveDiary: tags 清洗后为空')

      const now = new Date()
      const date = fmtDate(now)
      const fileTime = fmtFileTime(now) // 文件名用 HH_MM_SS
      const hm = fmtTime(now).slice(0, 5) // 正文第 2 行只用 HH:MM
      const titlePart = sanitizeTitle(title)
      const stem = titlePart ? `${date}-${fileTime}-${titlePart}` : `${date}-${fileTime}`
      const relDir = path.join('diaries', agentName)
      const absDir = path.join(dataRoot, relDir)
      fs.mkdirSync(absDir, { recursive: true })

      // 同一秒冲突追加 -1、-2 后缀
      let name = `${stem}.md`
      for (let i = 1; fs.existsSync(path.join(absDir, name)); i++) name = `${stem}-${i}.md`
      const rel = path.join(relDir, name)

      // 四段式内容（§3）：正文原样保留；参数 tags 优先（正文自带 Tag 行不动）
      const body = content.endsWith('\n') ? content : content + '\n'
      const text = `[${date}] - ${agentName}\n[${hm}]\n${body}\nTag: ${cleaned.join(', ')}\n`
      atomicWrite(path.join(dataRoot, rel), text)

      const chunkCount = chunkText(content, chunkerOpts).length
      enqueueIndex(rel, { force: true }) // 入队即返回，不保证已索引
      L('info', `saveDiary: 已写入 ${rel}（${chunkCount} chunks 入队）`)
      return { file: rel, chunks: chunkCount }
    },

    async recall({ query, vector, k = 6, truncate = 0.4, tags = null }) {
      assertRefusal()
      // P1.5 §2:vector 入参——存在时跳过 embed、归一化后直接走原 KNN;与 query 至少给一个。
      // 校验只在"vector 缺失且 query 非法"时失败,保持 P0 错误消息不变。
      if (vector == null && (typeof query !== 'string' || !query.trim())) {
        throw new Error('recall: query 必须是非空字符串')
      }
      const t0 = Date.now()
      let q = null
      if (vector != null) {
        // 向量路径:对副本归一化(不原地改动调用方的 Float32Array,最小惊奇);
        // 零向量保持原样(余弦全 0,自然被 truncate 阈值过滤)
        q = normalize(vector instanceof Float32Array ? Float32Array.from(vector) : vector)
        if (!q) throw new Error('recall: vector 无法归一化')
      } else {
        // 原 query 路径:每次 recall 重新 embed query，不做全局缓存（§11）
        let qv = null
        try {
          qv = await embedder.embed([query])
        } catch {
          qv = null
        }
        q = qv && qv[0] ? normalize(qv[0]) : null
        if (!q) throw new Error('embedding 服务不可用')
      }
      const idx = currentIndex // 快照引用：重建期间读到旧索引/新索引的整块，互不混用

      // ── P2 §6.2:wave 路径(增强召回;tags 为显式 Core Tag,可空)──
      // 任一失败/不可用 → tryWaveRecall 返回 null → 走下方 P1 路径(完全不变),
      // recall 永不失败(§0)。wave 成功时 stats.epa/pyramid 字段照常透出(沿 P1 契约)。
      const wave = await tryWaveRecall(q, idx, tags, truncate, k)
      if (wave) {
        return {
          blocks: wave.blocks,
          matchedTags: wave.matchedTags, // §4.9 完整来源版
          stats: {
            candidates: wave.candidates,
            indexedChunks: idx.length,
            ms: Date.now() - t0,
            epa: wave.epaStats,
            pyramid: wave.pyramidStats,
            wave: wave.stats,
          },
        }
      }

      const scored = []
      for (const c of idx) {
        const s = dot(q, c.vector) // 向量已预归一化，点积即余弦
        if (s >= truncate) scored.push({ file: c.file, chunkIndex: c.chunkIndex, score: s, text: c.content })
      }
      scored.sort((a, b) => b.score - a.score)
      // P1 §4:响应扩展——matchedTags(pyramid 种子)与 stats.epa/stats.pyramid 诊断字段;
      // 排序逻辑与 P0 完全一致(纯 KNN),新增计算只做诊断,失败只降级、绝不令 recall 失败
      let matchedTags = []
      let pyramidStats = null
      let epaStats = { trained: false }
      try {
        // EPA 诊断:仅在有训练基底时提供(未训练 → { trained: false })
        const epaObj = tagLayer.epa
        if (epaObj && epaObj.trained) {
          const proj = epaObj.project(q)
          if (proj) {
            const res = epaObj.detectCrossDomainResonance(q)
            epaStats = {
              trained: true,
              logicDepth: proj.logicDepth,
              entropy: proj.entropy,
              resonance: res ? res.resonance : 0,
              dominantAxes: (proj.dominantAxes || []).map((a) => ({ label: a.label, score: a.energy })),
            }
          }
        }
        // 残差金字塔:种子来源(与 EPA 训练与否无关,只要有向量化 tag 即可工作)
        if (tagLayer.stats().vectorizedTags > 0) {
          const pyr = await tagLayer.pyramid.analyze(q)
          if (pyr && pyr.levels.length > 0) {
            // 跨层合并同名 tag 取最大 weight(weight = contribution)
            const merged = new Map()
            for (const lv of pyr.levels) {
              for (const t of lv.tags) {
                const w = t.contribution || 0
                const prev = merged.get(t.name)
                if (!prev || w > prev.weight) merged.set(t.name, { tag: t.name, weight: w, level: lv.level })
              }
            }
            matchedTags = [...merged.values()]
              .sort((a, b) => b.weight - a.weight)
              .slice(0, 8)
              .map((m) => ({ ...m, notation: `=${m.tag}:${m.weight.toFixed(2)}@seed` })) // VCP 记号
            pyramidStats = {
              depth: pyr.features.depth,
              coverage: pyr.features.coverage,
              novelty: pyr.features.novelty,
              coherence: pyr.features.coherence,
              activation: pyr.features.tagMemoActivation,
            }
          }
        }
      } catch (err) {
        // §4:新增计算失败 → 记 log、按无种子降级
        L('warn', `recall 诊断字段计算失败,降级为无种子: ${err && err.message || err}`)
        matchedTags = []
        pyramidStats = null
        epaStats = { trained: false }
      }
      return {
        blocks: scored.slice(0, Math.max(1, Math.floor(k) || 1)),
        matchedTags,
        stats: {
          candidates: scored.length,
          indexedChunks: idx.length,
          ms: Date.now() - t0,
          epa: epaStats,
          pyramid: pyramidStats,
        },
      }
    },

    stats() {
      const base = {
        sig: effSig,
        dimension: effDimension,
        diaries: scanMdFiles(diariesDir, dataRoot).length,
        indexedChunks: currentIndex.length,
        pendingFiles: pendingCount,
        lastRebuild,
      }
      // P1 §6:并入 tag 层统计(tagCount/vectorizedTags/epaTrained)
      const ts = tagLayer.stats()
      return { ...base, tagCount: ts.tagCount, vectorizedTags: ts.vectorizedTags, epaTrained: ts.epaTrained }
    },

    rebuild() {
      return enqueue(doRebuild, 'rebuild')
    },

    // 延伸项（§7）：按修改时间倒序找第一个包含 target 的文件，恰好一处才替换
    async updateDiary({ target, replace }) {
      assertRefusal()
      if (typeof target !== 'string' || target.length < 15) {
        throw new Error('updateDiary: target 必须是不短于 15 字符的字符串')
      }
      if (typeof replace !== 'string') throw new Error('updateDiary: replace 必须是字符串')
      const files = scanMdFiles(diariesDir, dataRoot)
        .map((rel) => {
          let st = null
          try {
            st = fs.statSync(path.join(dataRoot, rel))
          } catch {
            st = null
          }
          return st ? { rel, mtime: st.mtimeMs } : null
        })
        .filter(Boolean)
        .sort((a, b) => b.mtime - a.mtime)
      for (const { rel } of files) {
        let text
        try {
          text = fs.readFileSync(path.join(dataRoot, rel), 'utf8')
        } catch (err) {
          L('warn', `updateDiary: 读取 ${rel} 失败: ${err.message}`)
          continue
        }
        const n = countOccurrences(text, target)
        if (n > 1) throw new Error(`updateDiary: "${target}" 在 ${rel} 中命中 ${n} 处，请提供更长的 target 消除歧义`)
        if (n === 1) {
          // 只做子串替换，禁止整文件覆写；用函数式 replace 避免 $ 转义
          atomicWrite(path.join(dataRoot, rel), text.replace(target, () => replace))
          enqueueIndex(rel, { force: true }) // 走同一索引队列
          L('info', `updateDiary: ${rel} 已替换 1 处`)
          return { file: rel, replaced: true }
        }
      }
      throw new Error(`updateDiary: 未找到包含 "${target}" 的日记文件`)
    },

    async close() {
      if (closed) return
      closeAllWatchers()
      if (eventTimer) {
        clearTimeout(eventTimer)
        eventTimer = null
      }
      if (flushTimer) {
        clearTimeout(flushTimer)
        flushTimer = null
      }
      await queueTail.catch(() => {}) // 等队列排空（最后一批落盘已在 quiescence 完成）
      closed = true
      pendingEvents.clear()
    },
  }

  // ---------- ready：启动加载 / 自动重建 / 拒绝服务（§6.4） ----------
  // 注意：必须放在 store 对象定义之后，IIFE 同步段会引用 store。
  store.ready = (async () => {
    fs.mkdirSync(dataRoot, { recursive: true })
    fs.mkdirSync(indexDir, { recursive: true })
    fs.mkdirSync(diariesDir, { recursive: true })
    const meta = readMeta()
    if (meta && (meta.sig !== effSig || meta.dimension !== effDimension)) {
      // §6.4.1 拒绝服务：ready reject + recall/save 抛错；绝不清库、绝不混用、绝不动盘。
      // 工具仍可用：调用方（插件入口）catch 住 ready，三个工具每次调用返回该错误文本。
      refusalMsg = buildRefusal(meta)
      L('error', refusalMsg)
      const e = new Error(refusalMsg)
      e.code = 'ERR_MEMO_SIG_MISMATCH'
      throw e
    }
    // P1 §6:tag 层随 ready 装载。eca.json 的 sig 与当前不一致 → 走原有拒绝服务
    // (错误消息本身含"嵌入模型签名已变",入口 catch 后同样进入 fault 态);
    // 其它装载错误(如 tags.jsonl 被篡改)记日志后以空 tag 层继续,不拖垮 chunks 服务。
    try {
      await tagLayer.load()
    } catch (err) {
      if (err && /嵌入模型签名已变/.test(err.message)) {
        // sig 不一致 → 走原有拒绝服务(错误消息本身含"嵌入模型签名已变",入口 catch 后同样进入 fault 态)
        refusalMsg = err.message
        L('error', refusalMsg)
        throw err
      }
      // 其它装载错误(如 tags.jsonl 被篡改):记日志后以空 tag 层继续,不拖垮 chunks 服务;
      // 下次 tag 变更 flush 时从日记真相源重新派生(或由 rebuild 全量重建)
      L('error', `tag 层装载失败(按空 tag 层继续): ${err && err.message || err}`)
    }
    // P2 §6.1:tagLayer.load 后全量构建 taggraph(图不可用 → recall 回退 P1)
    if (!refusalMsg) await refreshTagGraph({ force: true })
    if (!meta) {
      const diaryFiles = scanMdFiles(diariesDir, dataRoot)
      if (diaryFiles.length > 0) {
        // §6.4.2 meta 缺失但 diaries 非空 → 自动全量重建
        L('info', 'index/meta.json 缺失但 diaries 非空，开始自动全量重建')
        await enqueue(doRebuild, 'rebuild')
      } else {
        writeIndexFiles() // 全新数据根：写空索引与 meta
      }
    } else if (meta.chunkCount > 0 && !fs.existsSync(chunksPath)) {
      L('info', 'meta.json 存在但 chunks.jsonl 缺失，执行全量重建')
      await enqueue(doRebuild, 'rebuild')
    } else {
      loadIndex()
    }
    // 启动对账：把"磁盘上有但索引中没有"的日记补入索引队列
    // （覆盖上次进程在 saveDiary 落盘后、索引完成前退出的窗口；上次 embed 全失败的
    //   文件也会在此获得重试）。doRebuild 分支的文件已全部在 fileMd5 中，天然跳过。
    if (!refusalMsg) {
      for (const rel of scanMdFiles(diariesDir, dataRoot)) {
        if (!fileMd5.has(rel)) enqueueIndex(rel)
      }
    }
    // 平凡加载后也补挂单文件 watch（覆盖启动后的原地覆写编辑）
    if (!refusalMsg) {
      for (const rel of new Set(currentIndex.map((c) => c.file))) ensureFileWatcher(rel)
    }
    startWatcher()
    // P1 §6:启动末尾强制一次 tag 层 flush——把"load 时 tagHash 不一致而滞留未训练"
    // 的 EPA 拉起重训(flush 内部自行判断是否有活,无事则立即返回)
    if (!refusalMsg) maybeTagFlush(true)
    return store
  })()
  store.ready.catch((err) => L('error', `store 初始化失败: ${err && err.message || err}`))

  return store
}