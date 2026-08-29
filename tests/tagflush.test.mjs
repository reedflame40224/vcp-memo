// tests/tagflush.test.mjs — maybeTagFlush 竞态回归(P3 审查修复)
//
// 场景:连续快速 saveDiary。前一篇的 tag flush(慢 embed)还在进行时,
// 后一篇的队列空闲回调以非 force 方式调 maybeTagFlush——修复前该调用被直接
// 丢弃(tagFlushPending 只在 force 时置位),最后一批新 tag 永远等不到向量化。
// 修复后:flush 期间的任何调用都置 pending,结束后必补一次。
//
// 运行:node tests/tagflush.test.mjs

import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { openStore } from '../engine/store.mjs'

// 差速假 embedder:tag 向量化(整批都是 ≤6 字符短文本)走 250ms 慢车道,
// chunk 向量化走 10ms 快车道——保证"flush 进行中,后续 save 的队列空闲回调
// 以非 force 方式打进来"这一竞态窗口 100% 打开(修复前该调用被直接丢弃)。
// 向量本身:对字符码累加进 8 桶再归一化(语义无关,只验证管线)。
function slowFakeEmbedder() {
  const dim = 8
  return {
    model: 'fake', dimension: dim, sig: `fake@${dim}`,
    async embed(texts) {
      const tagLike = texts.every((t) => String(t).length <= 6)
      await new Promise((r) => setTimeout(r, tagLike ? 250 : 10))
      return texts.map((t) => {
        const v = new Float32Array(dim)
        for (const ch of String(t)) v[ch.codePointAt(0) % dim] += 1
        const n = Math.hypot(...v) || 1
        for (let i = 0; i < dim; i++) v[i] /= n
        return v
      })
    },
  }
}

async function waitFor(cond, timeoutMs, what) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeoutMs) {
    if (await cond()) return
    await new Promise((r) => setTimeout(r, 100))
  }
  assert.fail(`等待超时: ${what}`)
}

function readTags(dataRoot) {
  const p = path.join(dataRoot, 'index', 'tags.jsonl')
  if (!fs.existsSync(p)) return []
  return fs.readFileSync(p, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l))
}

test('连续快速 saveDiary:所有新 tag 最终都被向量化(flush 竞态回归)', async () => {
  const dataRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'vcp-memo-tagflush-'))
  try {
    const store = await openStore(
      { dataRoot, agentName: 'dsh', dimension: 8, sig: 'fake@8', watch: false },
      slowFakeEmbedder(),
      () => {},
    )
    await store.ready

    // 三篇快速连发,各带一个新 tag
    store.saveDiary({ title: '第一篇', content: '第一篇正文。', tags: ['甲标签'] })
    store.saveDiary({ title: '第二篇', content: '第二篇正文。', tags: ['乙标签'] })
    store.saveDiary({ title: '第三篇', content: '第三篇正文。', tags: ['丙标签'] })

    // 等索引队列排空,再等 tag flush(含补做)全部完成:
    // 三个 tag 都出现在 tags.jsonl 且向量非 null
    await waitFor(
      () => {
        const tags = readTags(dataRoot)
        const names = new Set(tags.map((t) => t.name))
        return (
          names.has('甲标签') && names.has('乙标签') && names.has('丙标签')
          && tags.every((t) => Array.isArray(t.vector) && t.vector.length === 8)
        )
      },
      15000,
      '三个新 tag 全部向量化(修复前:最后一批永远停在 vector:null)',
    )

    const tags = readTags(dataRoot)
    assert.equal(tags.length, 3)
    assert.ok(tags.every((t) => Array.isArray(t.vector)), '全部 tag 有向量')
    await store.close()
    console.log('  ✔ 三篇快速连发:三个新 tag 全部向量化,无丢失')
  } finally {
    fs.rmSync(dataRoot, { recursive: true, force: true })
  }
})

console.log('tagflush.test.mjs: 全部通过')
