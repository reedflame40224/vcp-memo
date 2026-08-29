// tests/chunker.test.mjs — chunker 自测（node tests/chunker.test.mjs 直接运行，零依赖）
// 断言失败即抛错，退出码非零。覆盖 SPEC.md §4 与 P0 验收标准 §10.2。
import assert from 'node:assert/strict'
import { estimateTokens, chunkText } from '../core/chunker.mjs'

const SENT_RE = /(?<=[。？！.!?\n])/ // 与实现一致的切句正则（仅测试内部使用）

let passed = 0
function ok(name, fn) {
  fn()
  passed++
  console.log(`  ✓ ${name}`)
}

console.log('chunker.test.mjs')

// ---- 测试 1：空输入/空白输入 → [] ----
ok('空输入 → []', () => {
  assert.deepStrictEqual(chunkText(''), [])
  assert.deepStrictEqual(chunkText(undefined), [])
  assert.deepStrictEqual(chunkText(null), [])
})

ok('空白输入 → []', () => {
  assert.deepStrictEqual(chunkText('   '), [])
  assert.deepStrictEqual(chunkText('\n\t \r\n'), [])
})

// ---- 测试 2：短文本 → 单块原样返回（trim 后）----
ok('短文本 → 单块原样（trim 后）', () => {
  assert.deepStrictEqual(chunkText('你好，世界！'), ['你好，世界！'])
  assert.deepStrictEqual(chunkText('  记忆与总结  '), ['记忆与总结'])
  assert.deepStrictEqual(chunkText('a short note.'), ['a short note.'])
})

// ---- 测试 3：>20000 字符中文，maxTokens=200, overlapTokens=40 ----
ok('长中文文本：块不越界 / 相邻重叠 / 无空块 / 覆盖原文', () => {
  const maxTokens = 200
  const overlapTokens = 40

  // 构造 >20000 字符中文文本。每句约 12 字符（≤40 token），含唯一序号保证可定位；
  // 句子 token 权重 ≤ overlapTokens，保证回溯重叠总是至少取回 1 句。
  const sents = []
  let total = 0
  let n = 0
  while (total < 20000) {
    n++
    const s = `第${n}号记忆：浪潮算法。`
    sents.push(s)
    total += s.length
  }
  const text = sents.join('')
  assert.ok(text.length > 20000, '文本应超过 20000 字符')
  assert.ok(sents.length > 100, '句子数量应足够多')

  const chunks = chunkText(text, { maxTokens, overlapTokens })
  assert.ok(chunks.length > 1, '长文本应产生多块')

  // 1) 每块 estimateTokens ≤ maxTokens
  for (const c of chunks) {
    assert.ok(estimateTokens(c) <= maxTokens, `块超限: ${estimateTokens(c)} > ${maxTokens}`)
  }

  // 2) 相邻块存在重叠内容：上一块末句（按完整句子取）必在下一块中出现
  for (let i = 0; i < chunks.length - 1; i++) {
    const lastSent = chunks[i].split(SENT_RE).filter(s => s.trim()).pop()
    assert.ok(lastSent, `块 ${i} 不应为空（取末句）`)
    assert.ok(
      chunks[i + 1].includes(lastSent),
      `相邻块 ${i}/${i + 1} 应包含重叠句子: "${lastSent}"`,
    )
  }

  // 3) 无空白块
  for (const c of chunks) {
    assert.ok(c.trim().length > 0, '不应存在空白块')
  }

  // 4) 覆盖原文：原文每条句子至少出现在一个块中（重叠允许重复；忽略空白差异）
  for (const s of sents) {
    assert.ok(chunks.some(c => c.includes(s)), `句子未被任何块覆盖: ${s}`)
  }
  assert.ok(chunks.length < sents.length, '组块应合并较短的句子')
})

// ---- 测试 4：单句超长（5000 字符无标点）强制切片且每片 ≤ maxTokens ----
ok('单句超长强制切片且每片 ≤ maxTokens', () => {
  const maxTokens = 200
  const long = '测'.repeat(5000) // 5000 中文字符，无标点，estimateTokens = 7500 > 200
  assert.ok(estimateTokens(long) > maxTokens, '单句应确实超限')

  const pieces = chunkText(long, { maxTokens, overlapTokens: 40 })
  assert.ok(pieces.length > 1, '超长单句应被切成多片')
  for (const p of pieces) {
    assert.ok(estimateTokens(p) <= maxTokens, `切片超限: ${estimateTokens(p)} > ${maxTokens}`)
    assert.ok(p.trim().length > 0, '切片不应为空白')
  }
  // 强制切片应无损：拼接 = 原文（纯中文无空白，无需归一化）
  assert.strictEqual(pieces.join(''), long)
})

// ---- 补充：默认参数可用 ----
ok('默认参数可用', () => {
  assert.deepStrictEqual(chunkText('默认参数测试'), ['默认参数测试'])
  assert.ok(estimateTokens('你好') > 0)
})

console.log(`\nchunker.test.mjs: ${passed} 项断言组全部通过`)