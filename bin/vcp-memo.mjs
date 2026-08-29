#!/usr/bin/env node
// bin/vcp-memo.mjs —— vcp-memo 管理 CLI(SPEC-P3 §1)
// 脱离 DSH 运行的独立管理工具:node bin/vcp-memo.mjs <command> [--dataRoot PATH]
// 零 npm 依赖(仅 node 内置模块)、plain ESM、中文注释。
// 复用 engine/embed.mjs 的 createEmbedder 与 engine/store.mjs 的 openStore;
// 默认配置与 cordis.patch.yml 一致(dataRoot /home/lyy/vcp-memo-data、bge-m3@1024);
// openStore 时 watch:false,用完必须 close()。
//
// 命令:
//   stats   打印 store.stats() 全字段(sig/dimension/diaries/indexedChunks/pendingFiles/
//           lastRebuild + tagCount/vectorizedTags/epaTrained)
//   rebuild 全量重建并打印结果(files/chunks 数),并打印 DSH 运行中的警告
//   tags    读 index/tags.jsonl,按出现次数降序列出「tag 名 + 文件数 + 有无向量」
//   doctor  体检:meta 签名一致/孤儿 chunk/未入索引日记/无向量 tag/epa tagHash 一致
// 退出码:正常 0;doctor 发现问题 1;命令非法 2(打印用法);执行失败(含 dataRoot 不存在)2。
// 隐私纪律(与 engine/embed.mjs 一致):只打印统计与文件名,绝不打印日记正文。

import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import { createEmbedder } from '../engine/embed.mjs'
import { openStore } from '../engine/store.mjs'

// 默认配置(与 cordis.patch.yml 一致):dataRoot/agentName/embedding。CLI 不调用 recall,
// 故 tagmemo(浪潮)节与召回无关,不随本 CLI 装配(规格 §1 只点名 dataRoot 与 bge-m3@1024)。
const DEFAULT_CONFIG = {
  dataRoot: '/home/lyy/vcp-memo-data',
  agentName: 'dsh',
  embedding: { baseUrl: 'http://127.0.0.1:11434/v1', model: 'bge-m3', dimension: 1024 },
}

// EPA 训练门槛:与 engine/taglayer.mjs 的 epaCfg.minTags 默认值(=8)保持一致,
// doctor 用它判断「epa.json 缺失是否正常」。
const EPA_MIN_TAGS = 8

// ---------- 小工具 ----------

// 递归扫描目录树下的 .md,返回相对 dataRoot 的路径(Linux 分隔符)。
// 说明:engine/store.mjs 内部有同名等价实现但未导出,CLI 无法直接复用,
// 此处保留最小等价实现(约 12 行纯 fs 遍历),其余逻辑一律复用 engine 模块(见偏离决策)。
function scanMdFiles(rootAbs, dataRoot) {
  const out = []
  const walk = (dir) => {
    let ents
    try {
      ents = fs.readdirSync(dir, { withFileTypes: true })
    } catch {
      return // 目录不存在/无权限:按空处理
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

// 读 dataRoot/index/ 下的派生资产;文件缺失返回 null,不抛错
function readIndex(dataRoot, name) {
  try {
    return fs.readFileSync(path.join(dataRoot, 'index', name), 'utf8')
  } catch {
    return null
  }
}

// 解析 JSONL 文本 → { rows, corrupt }(corrupt = 损坏行数,doctor 用于发现被篡改的索引)
function parseJsonl(text) {
  const rows = []
  let corrupt = 0
  if (text == null) return { rows, corrupt }
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim()
    if (!t) continue
    try {
      rows.push(JSON.parse(t))
    } catch {
      corrupt++
    }
  }
  return { rows, corrupt }
}

// 时间戳格式化:0 表示从未重建
function fmtTs(ts) {
  if (!ts) return '0(从未重建)'
  const d = new Date(ts)
  const p = (n) => String(n).padStart(2, '0')
  return `${ts}(${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())})`
}

// 与 engine/taglayer.mjs 的 computeTagHash 同一算法:有向量 tag 名排序 + sig → md5。
// doctor 第 5 项用它核对 index/epa.json 的 tagHash 是否与当前 tag 集一致。
function computeTagHash(tagRows, sig) {
  const names = tagRows
    .filter((t) => Array.isArray(t.vector) && t.vector.length > 0)
    .map((t) => String(t.name))
    .sort()
  return crypto.createHash('md5').update(JSON.stringify({ sig, names })).digest('hex')
}

// 手写参数解析(不引第三方库):<command> [--dataRoot PATH] 或 --dataRoot=PATH;
// 非法参数 → 抛错 → main 打印用法并以退出码 2 退出(§1「命令非法 2」)。
function parseArgs(argv) {
  const COMMANDS = new Set(['stats', 'rebuild', 'tags', 'doctor'])
  let command = null
  let dataRoot = DEFAULT_CONFIG.dataRoot
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--dataRoot') {
      if (i + 1 >= argv.length || argv[i + 1].startsWith('--')) {
        throw new Error('--dataRoot 缺少取值')
      }
      dataRoot = argv[++i]
    } else if (a.startsWith('--dataRoot=')) {
      dataRoot = a.slice('--dataRoot='.length)
      if (!dataRoot) throw new Error('--dataRoot 取值不能为空')
    } else if (a.startsWith('--')) {
      throw new Error(`未知参数: ${a}`)
    } else if (command === null) {
      command = a
    } else {
      throw new Error(`多余参数: ${a}`)
    }
  }
  if (command === null) throw new Error('缺少命令')
  if (!COMMANDS.has(command)) throw new Error(`未知命令: ${command}`)
  return { command, dataRoot }
}

// 用法说明(打印到 stderr:它是错误路径的提示,不进结果流)
function usage() {
  const text = [
    'vcp-memo 管理 CLI(SPEC-P3 §1)',
    '用法: node bin/vcp-memo.mjs <command> [--dataRoot PATH]',
    '',
    '命令:',
    '  stats    打印库统计:sig/dimension/diaries/indexedChunks/pendingFiles/lastRebuild/tagCount/vectorizedTags/epaTrained',
    '  rebuild  全量重建索引并打印 files/chunks 数(DSH 运行中时其内存索引不会自动刷新,建议重启或改用 memory_admin)',
    '  tags     按出现次数降序列出 index/tags.jsonl 中的 tag(名 + 文件数 + 有无向量)',
    '  doctor   体检:meta 签名一致/孤儿 chunk/未入索引日记/无向量 tag/epa tagHash 一致(发现问题退出码 1)',
    '',
    '参数:',
    '  --dataRoot PATH  覆盖默认数据根(默认 /home/lyy/vcp-memo-data)',
    '',
    '退出码: 0 正常;1 doctor 发现问题;2 命令非法或执行失败',
  ]
  console.error(text.join('\n'))
}

// openStore 的日志回调:引擎日志一律走 stderr,保证 stdout 上只有 CLI 自己的中文结果输出
function logToStderr(level, msg) {
  try {
    process.stderr.write(`[vcp-memo:${level}] ${msg}\n`)
  } catch {
    /* stderr 不可写时不阻塞主流程 */
  }
}

// 把引擎内部的 console.log 调试噪音(主要是 EPAModule 训练的英文进度行)转到 stderr,
// 维持 §1「输出全部中文、纯文本」:stdout 只承载命令结果。
// 注意:必须 await fn() 的返回值——fn 是 async 时,等它整个异步阶段跑完再恢复,
// 否则补丁在 fn 首次挂起时就被拆除,await 之后的引擎日志仍会漏到 stdout。
async function quietConsoleDuring(fn) {
  const original = console.log
  console.log = (...args) => {
    process.stderr.write(args.map(String).join(' ') + '\n')
  }
  try {
    return await fn()
  } finally {
    console.log = original
  }
}

// 打开 store(watch:false)执行 fn,结束后一定 close();open→fn→close 全程置于
// 同一个 quiet 窗口,覆盖 store.ready 的启动装载与 rebuild 的 EPA 训练噪音。
// ready 可能因索引签名不一致而 reject——交由各命令自行处理(rebuild 正是官方修复手段)。
async function withStore(dataRoot, embedder, fn) {
  return quietConsoleDuring(async () => {
    const store = await openStore(
      { dataRoot, agentName: DEFAULT_CONFIG.agentName, watch: false },
      embedder,
      logToStderr
    )
    try {
      return await fn(store)
    } finally {
      await store.close().catch(() => {})
    }
  })
}

// ---------- 命令实现 ----------

// stats:打印 store.stats() 全字段(§1)。store 打开时若触发自动全量重建(meta 缺失而
// diaries 非空,store §6.4.2 契约),属 openStore 固有行为,本命令如实呈现结果。
async function cmdStats(dataRoot, embedder) {
  let s = null
  let refused = false
  await withStore(dataRoot, embedder, async (store) => {
    await store.ready.catch(() => {
      refused = true // 签名不一致 → ready reject,stats() 本身仍可用
    })
    s = store.stats()
  })
  if (s == null) throw new Error('stats 获取失败')
  let indexedChunks = s.indexedChunks
  if (refused) {
    // 拒绝服务时 store 内存索引为空(loadIndex 未执行),indexedChunks 回退为
    // chunks.jsonl 实际条数,避免把「未加载」误导成「0 条」。
    indexedChunks = parseJsonl(readIndex(dataRoot, 'chunks.jsonl')).rows.length
    logToStderr(
      'warn',
      `meta 签名与当前 embedder 不一致,store 处于拒绝服务态;indexedChunks 已按 chunks.jsonl 实际条数(${indexedChunks})展示,请执行 rebuild 修复`
    )
  }
  console.log('== vcp-memo stats ==')
  console.log(`数据根: ${dataRoot}`)
  console.log(`sig: ${s.sig}`)
  console.log(`dimension: ${s.dimension}`)
  console.log(`diaries: ${s.diaries}`)
  console.log(`indexedChunks: ${indexedChunks}`)
  console.log(`pendingFiles: ${s.pendingFiles}`)
  console.log(`lastRebuild: ${fmtTs(s.lastRebuild)}`)
  console.log(`tagCount: ${s.tagCount}`)
  console.log(`vectorizedTags: ${s.vectorizedTags}`)
  console.log(`epaTrained: ${s.epaTrained}`)
}

// rebuild:全量重建并打印 files/chunks 结果 + DSH 运行中警告(§1)
async function cmdRebuild(dataRoot, embedder) {
  let result = null
  await withStore(dataRoot, embedder, async (store) => {
    // ready 因签名不一致 reject 也不阻塞 rebuild——rebuild 正是换模型后的修复通道
    await store.ready.catch(() => {})
    result = await store.rebuild()
  })
  if (result == null) throw new Error('rebuild 未返回结果')
  console.log('== vcp-memo rebuild ==')
  console.log(`重建完成: ${result.files} 个文件 → ${result.chunks} 个 chunk`)
  console.log('⚠️ 警告:若 DSH 正在运行,其内存索引不会自动刷新;建议重启 DSH 或改用 memory_admin 工具完成重建。')
}

// tags:读 index/tags.jsonl,按出现次数降序列出 tag 名 + 文件数 + 有无向量(§1)。
// 不打开 store(纯读派生资产,规格即写「读 index/tags.jsonl」)。
async function cmdTags(dataRoot, embedder) {
  void embedder // tags 命令不需要 embedder;保留签名保持一致
  const { rows } = parseJsonl(readIndex(dataRoot, 'tags.jsonl'))
  const list = rows
    .filter((t) => t && typeof t.name === 'string')
    .map((t) => {
      const occ = Array.isArray(t.occurrences) ? t.occurrences.length : 0
      const files = new Set(
        (Array.isArray(t.occurrences) ? t.occurrences : [])
          .filter((o) => o && o.file)
          .map((o) => String(o.file))
      ).size
      const hasVector = Array.isArray(t.vector) && t.vector.length > 0
      return { name: t.name, occ, files, hasVector }
    })
    .sort((a, b) => b.occ - a.occ || a.name.localeCompare(b.name))
  console.log('== vcp-memo tags ==')
  console.log(`共 ${list.length} 个 tag(按出现次数降序):`)
  for (const r of list) {
    console.log(`  ${r.name}  出现 ${r.occ} 次,覆盖 ${r.files} 个文件  ${r.hasVector ? '有向量' : '无向量'}`)
  }
}

// doctor:体检并逐项打印 ✅/⚠️(§1)。纯读派生资产 + fs 对账,不打开 store——
// 这样即使库处于拒绝服务态(签名不一致)也能给出完整诊断。返回发现问题数(0 则退出码 0)。
async function cmdDoctor(dataRoot, embedder) {
  const problems = []
  const println = (bad, text) => {
    console.log(`${bad ? '⚠️' : '✅'} ${text}`)
    if (bad) problems.push(text)
  }

  console.log('== vcp-memo doctor ==')
  console.log(`数据根: ${dataRoot}`)
  console.log('')

  // 1) meta sig 与当前 embedder sig 一致性
  const metaText = readIndex(dataRoot, 'meta.json')
  if (metaText == null) {
    const diaryCount = scanMdFiles(path.join(dataRoot, 'diaries'), dataRoot).length
    if (diaryCount === 0) {
      console.log('✅ meta 签名:index/meta.json 缺失(空库,尚未初始化,属正常)')
    } else {
      println(true, 'meta 签名:index/meta.json 缺失但 diaries 非空,索引未建立(需 rebuild,store 下次打开会自动重建)')
    }
  } else {
    let meta = null
    try {
      meta = JSON.parse(metaText)
    } catch (err) {
      println(true, `meta 签名:index/meta.json 无法解析(${err && err.message || err})`)
    }
    if (meta) {
      if (meta.sig === embedder.sig && meta.dimension === embedder.dimension) {
        console.log(`✅ meta 签名一致:index/meta.json 的 sig=${embedder.sig} 与当前 embedder 一致`)
      } else {
        println(
          true,
          `meta 签名不一致:index/meta.json 为 sig=${meta.sig ?? '(无)'}/dim=${meta.dimension ?? '(无)'},当前 embedder 为 sig=${embedder.sig};请执行 rebuild 全量重建`
        )
      }
    }
  }

  // 2) 孤儿 chunk:chunks.jsonl 索引指向不存在的文件
  const chunksParsed = parseJsonl(readIndex(dataRoot, 'chunks.jsonl'))
  if (chunksParsed.corrupt > 0) {
    println(true, `孤儿 chunk:index/chunks.jsonl 存在 ${chunksParsed.corrupt} 行损坏(索引被篡改或写坏)`)
  }
  const indexedFiles = new Set(
    chunksParsed.rows.filter((o) => o && typeof o.file === 'string').map((o) => String(o.file))
  )
  const missingFiles = []
  let orphanChunks = 0
  for (const f of indexedFiles) {
    let exists = false
    try {
      exists = fs.statSync(path.join(dataRoot, f)).isFile()
    } catch {
      exists = false
    }
    if (!exists) {
      missingFiles.push(f)
      orphanChunks += [...chunksParsed.rows].filter((o) => o && String(o.file) === f).length
    }
  }
  if (orphanChunks === 0) {
    console.log(`✅ 孤儿 chunk:共 ${chunksParsed.rows.length} 个 chunk,指向的文件均存在`)
  } else {
    println(true, `孤儿 chunk:${orphanChunks} 个 chunk 指向不存在的文件:[${missingFiles.join(', ')}]`)
  }

  // 3) 未入索引的日记文件:diaries 里有 .md 但 chunks.jsonl 无其任何 chunk
  const diaryFiles = scanMdFiles(path.join(dataRoot, 'diaries'), dataRoot)
  const unindexed = diaryFiles.filter((f) => !indexedFiles.has(f))
  if (unindexed.length === 0) {
    console.log(`✅ 日记索引:${diaryFiles.length} 个日记文件均已入索引`)
  } else {
    println(true, `未入索引的日记文件 ${unindexed.length} 个:[${unindexed.join(', ')}](需 rebuild 或等待 watcher 补索引)`)
  }

  // 4) 无向量 Tag:tags.jsonl 中 vector 为空/缺失的 tag
  const tagsParsed = parseJsonl(readIndex(dataRoot, 'tags.jsonl'))
  const tagRows = tagsParsed.rows.filter((t) => t && typeof t.name === 'string')
  if (tagsParsed.corrupt > 0) {
    println(true, `tag 向量:index/tags.jsonl 存在 ${tagsParsed.corrupt} 行损坏(派生资产被篡改)`)
  }
  const noVec = tagRows.filter((t) => !(Array.isArray(t.vector) && t.vector.length > 0))
  if (noVec.length === 0) {
    console.log(`✅ tag 向量:全部 ${tagRows.length} 个 tag 均已向量化`)
  } else {
    println(true, `无向量 Tag ${noVec.length} 个:[${noVec.map((t) => t.name).join(', ')}](embed 失败或未 flush,重试会在下次变更/rebuild 时补上)`)
  }

  // 5) epa.json 的 tagHash 与当前 tag 集一致性
  const curHash = computeTagHash(tagRows, embedder.sig)
  const epaText = readIndex(dataRoot, 'epa.json')
  if (epaText == null) {
    const vectorized = tagRows.filter((t) => Array.isArray(t.vector) && t.vector.length > 0).length
    if (vectorized < EPA_MIN_TAGS) {
      console.log(
        `✅ epa tagHash:index/epa.json 缺失(已向量化 tag ${vectorized} < minTags(${EPA_MIN_TAGS}),未达 EPA 训练门槛,属正常)`
      )
    } else {
      println(true, `epa tagHash:index/epa.json 缺失但已向量化 tag ${vectorized} ≥ ${EPA_MIN_TAGS}(EPA 未训练,需 rebuild 拉起)`)
    }
  } else {
    let epa = null
    try {
      epa = JSON.parse(epaText)
    } catch (err) {
      println(true, `epa tagHash:index/epa.json 无法解析(${err && err.message || err})`)
    }
    if (epa) {
      if (epa.sig !== embedder.sig) {
        println(true, `epa tagHash:index/epa.json 的 sig=${epa.sig} 与当前 ${embedder.sig} 不一致(需 rebuild 重训)`)
      } else if (epa.tagHash === curHash) {
        console.log('✅ epa tagHash:index/epa.json 的 tagHash 与当前 tag 集一致')
      } else {
        println(
          true,
          `epa tagHash:index/epa.json 的 tagHash(${String(epa.tagHash).slice(0, 8)}…)与当前 tag 集(${curHash.slice(0, 8)}…)不一致(EPA 基底过期,需 rebuild 重训)`
        )
      }
    }
  }

  console.log('')
  if (problems.length === 0) {
    console.log('体检通过:全部 5 项检查正常。')
  } else {
    console.log(`体检发现问题 ${problems.length} 项,建议执行 node bin/vcp-memo.mjs rebuild 全量重建。`)
  }
  return problems.length
}

// ---------- 入口 ----------

async function main() {
  let parsed
  try {
    parsed = parseArgs(process.argv.slice(2))
  } catch (err) {
    // 命令非法:打印用法,退出码 2(§1)
    usage()
    process.exitCode = 2
    return
  }
  const { command, dataRoot } = parsed

  // dataRoot 不存在/不是目录 → 友好中文错误而非 JS 堆栈(退出码 2,属"执行失败",
  // 见文件头偏离决策)。不静默当作空库:那会把 stats 误报成 diaries:0,误导用户
  // 以为自己记忆丢了;doctor 更会错误地输出"体检通过"。
  let rootStat = null
  try {
    rootStat = fs.statSync(dataRoot)
  } catch {
    rootStat = null // 路径不存在(或父级不可达):按不存在处理
  }
  if (!rootStat || !rootStat.isDirectory()) {
    const what = rootStat ? (rootStat.isFile() ? '是文件而非目录' : '不是目录') : '不存在'
    console.error(`vcp-memo: 数据根${what}: ${dataRoot}`)
    console.error(
      `        请确认 --dataRoot 指向正确的数据目录(默认 ${DEFAULT_CONFIG.dataRoot});` +
        `首次使用可由 DSH 插件自动建库,或手动创建该目录后重试。`
    )
    process.exitCode = 2
    return
  }

  const embedder = createEmbedder({ ...DEFAULT_CONFIG.embedding })
  try {
    if (command === 'stats') return await cmdStats(dataRoot, embedder)
    if (command === 'rebuild') return await cmdRebuild(dataRoot, embedder)
    if (command === 'tags') return await cmdTags(dataRoot, embedder)
    if (command === 'doctor') {
      const n = await cmdDoctor(dataRoot, embedder)
      if (n > 0) process.exitCode = 1 // doctor 发现问题 → 退出码 1(§1)
      return
    }
    // 不可达(parseArgs 已校验),保底
    usage()
    process.exitCode = 2
  } catch (err) {
    // 执行期失败:规格只定义 0/1/2 三个退出码,2 除「命令非法」外同时承载执行失败(见偏离决策)
    console.error(`vcp-memo: 命令执行失败: ${err && err.message || err}`)
    process.exitCode = 2
  }
}

main()