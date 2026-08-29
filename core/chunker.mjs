// core/chunker.mjs — 文本切分器（零依赖）
// 算法移植自 VCPToolBox TextChunker（句子切分 + 贪心组块 + 重叠窗口），
// token 计数换成启发式估计（参照 VCP LightMemo._estimateTokens）。见 SPEC.md §4。

// 中文及全角字符：全角空格/CJK 标点（3000-303F）、CJK 扩展 A/CJK 统一表意（3400-9FFF）、
// 兼容表意（F900-FAFF）、全角形式（FF00-FFEF）。非 BMP 字符按代理对逐 code point 判断，启发式足够。
const CJK_RE = /[\u3000-\u303F\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF]/

// 切句正则：句末标点或换行后断开，lookbehind 使分隔符保留在句尾
const SENTENCE_SPLIT_RE = /(?<=[。？！.!?\n])/

// 强制切片时优先断开的句内标点（含中英文逗号/分号/冒号/叹问号/顿号/句点）
const PREFERRED_CUT_RE = /[,，;；:：!！?？、。．.]/

// 单字符权重：中文及全角字符 ×1.5，其余 ×0.25
function charWeight(ch) {
  return CJK_RE.test(ch) ? 1.5 : 0.25
}

/**
 * 估计文本 token 数：中文及全角字符 ×1.5，其余 ×0.25，向上取整。
 * 非字符串输入按 0 处理。
 */
export function estimateTokens(text) {
  if (typeof text !== 'string' || text.length === 0) return 0
  let sum = 0
  for (const ch of text) sum += charWeight(ch)
  return Math.ceil(sum)
}

// 文本 token 权重浮点和。estimateTokens 与它仅差一次向上取整，
// 且 ceil(sum) ≤ maxTokens ⟺ sum ≤ maxTokens（maxTokens 为整数），
// 故内部贪心用浮点和判断与外部 estimateTokens 单调一致。
function textWeight(text) {
  let sum = 0
  for (const ch of text) sum += charWeight(ch)
  return sum
}

/**
 * 强制按 token 上限切分一段超长文本，优先在标点处断开。
 * 每次取"最长可行前缀"，再从尾部反向找最近的句内标点作为断点；
 * 无标点则直接按最长前缀切断。切分无损（各片拼接 = 原文）。
 */
function forceSlice(text, maxTokens) {
  const pieces = []
  let start = 0
  while (start < text.length) {
    // 求 [start, end) 内最长可行前缀
    let sum = 0
    let end = start
    while (end < text.length && sum + charWeight(text[end]) <= maxTokens) {
      sum += charWeight(text[end])
      end++
    }
    if (end === start) {
      // 防御：单字符也超限（如 maxTokens < 1）时强取 1 字符，避免死循环
      pieces.push(text.slice(start, start + 1))
      start++
      continue
    }
    if (end === text.length) {
      pieces.push(text.slice(start))
      break
    }
    // 优先在标点处断开：从 end-1 反向扫，取离上限最近的标点（标点留在片尾）
    let cut = end
    for (let i = end - 1; i > start; i--) {
      if (PREFERRED_CUT_RE.test(text[i])) {
        cut = i + 1
        break
      }
    }
    pieces.push(text.slice(start, cut))
    start = cut
  }
  return pieces
}

/**
 * 文本切分：按句子切分 → 逐句贪心组块 → 块间回溯重叠。
 * @param {string} text 输入文本
 * @param {{maxTokens?: number, overlapTokens?: number}} options
 *   maxTokens 默认 6800；overlapTokens 默认 680
 * @returns {string[]} 块数组；空/空白输入 → []
 *
 * 规则（SPEC §4）：
 * - 句末标点之后切句，分隔符保留；
 * - 逐句贪心组块，加入后估计超 maxTokens 则封块；
 * - 单句超 maxTokens 强制切片（优先标点处断开），各片独立成块；
 * - 封块时按完整句子从块尾回溯重叠，重叠上限取 min(overlapTokens, maxTokens - 下一句权重)，
 *   保证重叠内容与新块容纳得下（"不足则取能放下的句子"）；
 * - 每块 trim 后丢弃空白块。
 */
export function chunkText(text, options = {}) {
  let maxTokens = options.maxTokens ?? 6800
  let overlapTokens = options.overlapTokens ?? 680
  // 防御非法参数：回退默认值
  if (!Number.isFinite(maxTokens) || maxTokens <= 0) maxTokens = 6800
  if (!Number.isFinite(overlapTokens) || overlapTokens < 0) overlapTokens = 680

  // 空/空白输入 → []
  if (typeof text !== 'string' || text.trim() === '') return []

  const sentences = text.split(SENTENCE_SPLIT_RE)
  const chunks = []
  // 当前块：单位数组（句子或强制切片片），各带权重；curSum 为块权重和
  let curUnits = []
  let curSum = 0

  // 封块：拼接内容并 trim，丢弃空白块
  const flush = () => {
    const joined = curUnits.map(u => u.t).join('').trim()
    if (joined) chunks.push(joined)
    curUnits = []
    curSum = 0
  }

  // 以重叠单位为起始开新块（旧块已 flush；复制单位，避免共享引用）
  const beginWithOverlap = (units) => {
    curUnits = units.map(u => ({ t: u.t, w: u.w }))
    curSum = 0
    for (const u of curUnits) curSum += u.w
  }

  for (const sent of sentences) {
    if (!sent.trim()) continue // 跳过空白句（如换行产生的空段）
    const w = textWeight(sent)
    if (w > maxTokens) {
      // 单句超限：先封当前块，再把整句强制切片成独立块
      if (curUnits.length) flush()
      for (const piece of forceSlice(sent, maxTokens)) {
        const t = piece.trim()
        if (t) chunks.push(t)
      }
      continue
    }
    if (curSum + w <= maxTokens) {
      curUnits.push({ t: sent, w })
      curSum += w
      continue
    }
    // 封块并按完整句子回溯重叠。overlapCap = min(overlapTokens, maxTokens - w)，
    // 使重叠内容 + 下一句 ≤ maxTokens，重叠单位总能与下一句同块。
    const overlapCap = Math.min(overlapTokens, maxTokens - w)
    const overlap = []
    let oSum = 0
    for (let i = curUnits.length - 1; i >= 0; i--) {
      const u = curUnits[i]
      if (oSum + u.w > overlapCap) break // 放不下的句子不取
      overlap.unshift(u)
      oSum += u.w
      if (oSum >= overlapCap) break
    }
    flush()
    beginWithOverlap(overlap)
    curUnits.push({ t: sent, w })
    curSum += w // = 重叠权重 + w ≤ maxTokens，恒成立
  }
  flush()
  return chunks
}