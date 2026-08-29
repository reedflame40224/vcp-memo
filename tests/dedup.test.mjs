// tests/dedup.test.mjs — ResultDeduplicator 移植自测（node tests/dedup.test.mjs 直接运行，零依赖）
// 覆盖 SPEC-P2.md §8-5 与 §5 验收：
//   多身份硬去重（同文同 chunk 不同对象合并、保留高分/高完整度代表、并查归并传递性身份链）；
//   同文不同 chunk 保留；NFKC 规范化等价正文去重；
//   语义去重：余弦 ≥0.92 近重复抑制（保留高分、贴近查询者）、<0.92 双保留；
//   无向量（及维度不符向量）候选一律保留；maxResults 截断；semantic=false 只走硬去重。
// 断言失败即抛错，退出码非零。
import assert from 'node:assert/strict'
import { deduplicate } from '../core/ResultDeduplicator.mjs'

const DIM = 1024

let passed = 0
async function ok(name, fn) {
  await fn()
  passed++
  console.log(`  ✓ ${name}`)
}

// 单位轴向量：v[axis] = 1
function unitAxis(axis) {
  const v = new Float32Array(DIM)
  v[axis] = 1
  return v
}

// 由前几个分量构造单位向量（其余为 0），用于精确控制两向量的余弦
function normalize(components) {
  let s = 0
  for (const x of components) s += x * x
  const n = Math.sqrt(s)
  const v = new Float32Array(DIM)
  for (let i = 0; i < components.length; i++) v[i] = components[i] / n
  return v
}

// 测试侧余弦（用于断言前提，如 cos 确实 ≥/＜ 阈值）
function cosine(a, b) {
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

async function main() {
  console.log('dedup.test.mjs')

  // ── 语义向量前提：与 unitAxis(0) 的余弦精确可控 ──
  // vHigh ≈ e0；vNear 与 e0 余弦 ≈ 0.969（≥0.92）；vMid 与 e0 余弦 ≈ 0.849（<0.92）
  const vHigh = unitAxis(0)
  const vNear = normalize([0.97, 0.2473])
  const vMid = normalize([0.85, 0.53])
  assert.ok(cosine(vHigh, vNear) >= 0.92, `cos(vHigh,vNear)=${cosine(vHigh, vNear)} 应 ≥0.92`)
  assert.ok(cosine(vHigh, vMid) < 0.92, `cos(vHigh,vMid)=${cosine(vHigh, vMid)} 应 <0.92`)

  // ── 输入卫生 ──
  await ok('空/非法输入：返回空数组，不抛异常', async () => {
    assert.deepEqual(await deduplicate([]), [])
    assert.deepEqual(await deduplicate(null), [])
    assert.deepEqual(await deduplicate(undefined), [])
    assert.deepEqual(await deduplicate({}), [])
  })

  // ── 硬去重：多身份 ──
  await ok('硬去重：同文同 chunk 不同对象合并，保留高分代表（原对象引用不变）', async () => {
    const low = { file: 'diary/01.md', chunkIndex: 0, score: 0.77, text: '同文同 chunk 第一版', vector: null }
    const high = { file: 'diary/01.md', chunkIndex: 0, score: 0.81, text: '同文同 chunk 第一版', vector: null }
    const out = await deduplicate([low, high])
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0], high, '代表应为高分原对象（无克隆）')
    assert.strictEqual(out[0].score, 0.81)
  })

  await ok('硬去重：分数相同时按完整度选代表（matchedTags 胜出）', async () => {
    const plain = { file: 'diary/02.md', chunkIndex: 0, score: 0.6, text: '同正文', vector: null }
    const rich = { file: 'diary/02.md', chunkIndex: 0, score: 0.6, text: '同正文', vector: null, matchedTags: ['澜沧计划'] }
    const out = await deduplicate([plain, rich])
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0], rich)
  })

  await ok('硬去重：并查归并——传递性身份链合并为单槽，取最高分', async () => {
    const t1 = { file: 'f.md', chunkIndex: 0, score: 0.5, text: '正文A' }
    const t2 = { file: 'f.md', chunkIndex: 0, score: 0.6, text: '正文B' }  // 与 t1 共享 path-chunk
    const t3 = { file: 'g.md', chunkIndex: 2, score: 0.7, text: '正文B' }  // 与 t2 共享正文身份
    const out = await deduplicate([t1, t2, t3])
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0], t3)
    assert.strictEqual(out[0].score, 0.7)
  })

  await ok('硬去重：source 显式存在时按优先级表取值（rag 50 胜 unknown 0）', async () => {
    const unknown = { file: 'u.md', chunkIndex: 0, score: 0.5, text: '同正文', source: 'unknown' }
    const rag = { file: 'v.md', chunkIndex: 0, score: 0.5, text: '同正文', source: 'rag' }
    const out = await deduplicate([unknown, rag])
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0], rag)
  })

  // ── 硬去重：保留路径 ──
  await ok('同文不同 chunk：各自保留（path-chunk 身份不同）', async () => {
    const out = await deduplicate([
      { file: 'diary/03.md', chunkIndex: 1, score: 0.5, text: '第一段内容', vector: null },
      { file: 'diary/03.md', chunkIndex: 2, score: 0.4, text: '第二段内容', vector: null },
    ])
    assert.strictEqual(out.length, 2)
    assert.deepEqual(out.map(c => c.chunkIndex), [1, 2])
  })

  await ok('NFKC 规范化等价正文：跨文件合并去重（全角→半角、CRLF/连续空行折叠）', async () => {
    const n1 = { file: 'diary/04.md', chunkIndex: 0, score: 0.6, text: 'Ｆｕｌｌｗｉｄｔｈ　ＡＢＣ　Ｔｅｓｔ\r\n\r\n\r\n行' }
    const n2 = { file: 'diary/other.md', chunkIndex: 5, score: 0.55, text: 'fullwidth abc test\n\n行' }
    const out = await deduplicate([n1, n2])
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0], n1, '高分代表 n1 胜出')
  })

  await ok('正文规范化：空白折叠（多空格/Tab → 单空格）等价去重', async () => {
    const w1 = { file: 'x.md', chunkIndex: 0, score: 0.5, text: '多  空格\t与Tab' }
    const w2 = { file: 'y.md', chunkIndex: 0, score: 0.45, text: '多 空格 与Tab' }
    const out = await deduplicate([w1, w2])
    assert.strictEqual(out.length, 1)
  })

  // ── 语义去重：近重复抑制 ──
  await ok('语义：余弦≈0.969 ≥0.92 抑制低分近重复（保留高分）', async () => {
    const out = await deduplicate([
      { file: 'a.md', chunkIndex: 0, score: 0.9, text: '相似甲', vector: vHigh },
      { file: 'b.md', chunkIndex: 0, score: 0.8, text: '相似乙', vector: vNear },
    ])
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].score, 0.9)
    assert.strictEqual(out[0].text, '相似甲')
  })

  await ok('语义：提供 queryVector 时按查询相似度排序，保留贴近查询者', async () => {
    const out = await deduplicate([
      { file: 'a.md', chunkIndex: 0, score: 0.9, text: '相似甲', vector: vHigh },
      { file: 'b.md', chunkIndex: 0, score: 0.8, text: '相似乙', vector: vNear },
    ], vNear)
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].score, 0.8, '查询贴近 vNear → 低分者反而入选为近重复代表')
    assert.strictEqual(out[0].text, '相似乙')
  })

  await ok('语义：余弦≈0.849 <0.92 双保留', async () => {
    const out = await deduplicate([
      { file: 'a.md', chunkIndex: 0, score: 0.9, text: '甲', vector: vHigh },
      { file: 'b.md', chunkIndex: 0, score: 0.85, text: '乙', vector: vMid },
    ])
    assert.strictEqual(out.length, 2)
    assert.deepEqual(out.map(c => c.text), ['甲', '乙'], '输出按分数降序')
  })

  await ok('语义：options.semanticThreshold=0.8 时，cos≈0.849 的对也被抑制', async () => {
    const out = await deduplicate([
      { file: 'a.md', chunkIndex: 0, score: 0.9, text: '甲', vector: vHigh },
      { file: 'c.md', chunkIndex: 0, score: 0.85, text: '丙', vector: vMid },
    ], null, { semanticThreshold: 0.8 })
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].text, '甲')
  })

  await ok('组合：先硬去重再语义去重（同正文近重复先被硬去重并掉）', async () => {
    const out = await deduplicate([
      { file: 'a.md', chunkIndex: 0, score: 0.9, text: '相同正文', vector: vHigh },
      { file: 'b.md', chunkIndex: 0, score: 0.85, text: '相同正文', vector: vNear },  // 同正文 → 硬去重并入
      { file: 'c.md', chunkIndex: 0, score: 0.7, text: '其他', vector: vMid },
    ])
    assert.strictEqual(out.length, 2)
    assert.strictEqual(out[0].score, 0.9)
    assert.ok(out.some(c => c.text === '其他'))
  })

  // ── 无向量候选：一律保留 ──
  await ok('确定性：无向量候选一律保留，不被语义抑制误伤', async () => {
    const out = await deduplicate([
      { file: 'a.md', chunkIndex: 0, score: 0.9, text: '有向量高分', vector: vHigh },
      { file: 'b.md', chunkIndex: 0, score: 0.85, text: '无向量', vector: null },
      { file: 'c.md', chunkIndex: 0, score: 0.8, text: '近重复有向量', vector: vNear },
    ])
    assert.strictEqual(out.length, 2)
    assert.ok(out.some(c => c.text === '有向量高分'))
    assert.ok(out.some(c => c.text === '无向量'), '无向量候选必须保留')
  })

  await ok('确定性：维度不符的向量视为无向量，同样原样保留', async () => {
    const short = { file: 'd.md', chunkIndex: 0, score: 0.7, text: '短向量', vector: new Float32Array([1, 0, 0]) }
    const out = await deduplicate([
      { file: 'a.md', chunkIndex: 0, score: 0.9, text: '有向量高分', vector: vHigh },
      short,
    ])
    assert.strictEqual(out.length, 2)
    assert.ok(out.includes(short), '短向量候选按无向量保留（对象引用不变）')
  })

  // ── maxResults 截断 ──
  await ok('maxResults：截断为前 3（优先级→分数降序）', async () => {
    const items = Array.from({ length: 5 }, (_, i) => ({
      file: `doc${i}.md`, chunkIndex: 0, score: 0.9 - i * 0.05, text: `内容${i}`, vector: unitAxis(i),
    }))
    const out = await deduplicate(items, null, { maxResults: 3 })
    assert.strictEqual(out.length, 3)
    assert.deepEqual(out.map(c => c.score), [0.9, 0.85, 0.8])
  })

  await ok('maxResults + semantic=false：只硬去重，按输入顺序截断', async () => {
    const items = [0, 1, 2, 3].map(i => ({
      file: `d${i}.md`, chunkIndex: i, score: 0.9 - i * 0.2, text: `t${i}`, vector: unitAxis(i),
    }))
    const out = await deduplicate(items, null, { semantic: false, maxResults: 2 })
    assert.strictEqual(out.length, 2)
    assert.deepEqual(out.map(c => c.file), ['d0.md', 'd1.md'])
  })

  await ok('semantic=false：近重复对全部保留（不做语义抑制）', async () => {
    const out = await deduplicate([
      { file: 'a.md', chunkIndex: 0, score: 0.9, text: '相似甲', vector: vHigh },
      { file: 'b.md', chunkIndex: 0, score: 0.8, text: '相似乙', vector: vNear },
    ], null, { semantic: false })
    assert.strictEqual(out.length, 2)
  })

  console.log(`\ndedup.test.mjs: ${passed} 项断言组全部通过`)
}

main().catch(err => {
  console.error(`\n✗ dedup.test.mjs 失败: ${err.message}`)
  process.exitCode = 1
})