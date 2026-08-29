// tests/embed.test.mjs —— 规格 §5 embedding 客户端自测（node 直接运行，无测试框架）
// 用法：node tests/embed.test.mjs （依赖真实 Ollama bge-m3 于 127.0.0.1:11434）

import { createEmbedder } from '../engine/embed.mjs';
import { createServer } from 'node:http';
import assert from 'node:assert/strict';

// 真实环境（规格 §10：Ollama bge-m3 @ 1024 维）
const REAL = { baseUrl: 'http://127.0.0.1:11434/v1', model: 'bge-m3', dimension: 1024 };

let passed = 0;
let failed = 0;
function reportOk(name) {
  passed++;
  console.log(`  ✔ ${name}`);
}
async function test(name, fn) {
  try {
    await fn();
    reportOk(name);
  } catch (err) {
    failed++;
    console.error(`  ✘ ${name}\n    ${err?.stack ?? err}`);
  }
}

// 向量是否含有非零元素
function isNonZero(v) {
  for (let i = 0; i < v.length; i++) if (v[i] !== 0) return true;
  return false;
}

// 余弦相似度（bge-m3 向量未归一，需手动归一化）
function cosSim(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

// 本地 mock HTTP 服务器工具（用于维度过错/超时/apiKey 断言，不依赖外网）
function listen(server) {
  return new Promise((resolve) =>
    server.listen(0, '127.0.0.1', () => resolve(server.address().port))
  );
}
function closeServer(server) {
  return new Promise((resolve) => server.close(() => resolve()));
}

async function main() {
  console.log('== embed.test.mjs ==');

  // ── 用例 1：基础 2 条 → 2×1024 非零 Float32Array；空输入 → [] ──
  await test('基础：embed(["测试记忆","第二条文本"]) 返回 2 个 1024 维非零向量', async () => {
    const e = createEmbedder(REAL);
    assert.strictEqual(e.sig, 'bge-m3@1024');
    const out = await e.embed(['测试记忆', '第二条文本']);
    assert.strictEqual(out.length, 2);
    for (const v of out) {
      assert.ok(v instanceof Float32Array, '应为 Float32Array');
      assert.strictEqual(v.length, 1024, '维度应为 1024');
      assert.ok(isNonZero(v), '向量不应全零');
    }
    assert.deepStrictEqual(await e.embed([]), [], '空输入应返回空数组');
  });

  // ── 用例 2：40 条（batchSize=16, concurrency=2）→ 40 个向量，顺序与输入一致 ──
  await test('分批并发：40 条文本顺序与输入一致', async () => {
    const e = createEmbedder({ ...REAL, batchSize: 16, concurrency: 2 });
    const texts = Array.from(
      { length: 40 },
      (_, i) => `记忆条目编号 ${i}：这是第 ${i + 1} 条用于验证批处理顺序的测试文本，内容各不相同以便区分向量。`
    );
    const out = await e.embed(texts);
    assert.strictEqual(out.length, 40);
    out.forEach((v, i) => {
      assert.ok(v instanceof Float32Array && v.length === 1024, `第 ${i} 条维度错误`);
      assert.ok(isNonZero(v), `第 ${i} 条为空向量`);
    });
    // 抽查 3 个位置：与单条重嵌的向量高度相似 ⇒ 批量结果未错位
    for (const i of [0, 19, 39]) {
      const single = await e.embed([texts[i]]);
      const sim = cosSim(out[i], single[0]);
      assert.ok(sim > 0.9, `第 ${i} 条疑似顺序错位（相似度 ${sim}）`);
    }
  });

  // ── 用例 3：错误端点 → 对应位 null，embed 不抛出 ──
  await test('错误端点：embed 不抛出且全部位为 null', async () => {
    const e = createEmbedder({ ...REAL, baseUrl: 'http://127.0.0.1:9/v1', retries: 1 });
    const out = await e.embed(['第一条', '第二条', '第三条']);
    assert.strictEqual(out.length, 3);
    assert.ok(out.every((v) => v === null), '失败批次整批应为 null');
  });

  // ── 用例 4：mock 服务返回乱序 data 且含维度不符条目 ──
  // 输入 3 条（dimension=2）：data 乱序给出 index=2/1/0，
  // index=1 的 embedding 是 3 维（≠2）→ 该位 null；其余按 index 正确回填。
  await test('维度不符记 null 且按 index 回填（mock）', async () => {
    let requestCount = 0;
    const server = createServer((req, res) => {
      let body = '';
      req.on('data', (c) => (body += c));
      req.on('end', () => {
        requestCount++;
        const { input } = JSON.parse(body);
        const data = [
          { index: 2, embedding: [2, 2] },
          { index: 1, embedding: [0.1, 0.2, 0.3] }, // 3 维 ≠ dimension=2
          { index: 0, embedding: [1, 1] },
        ].slice(0, input.length);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ data }));
      });
    });
    const port = await listen(server);
    try {
      const e = createEmbedder({
        baseUrl: `http://127.0.0.1:${port}/v1`,
        model: 'bge-m3',
        dimension: 2,
        batchSize: 16,
        retries: 0,
      });
      const out = await e.embed(['a', 'b', 'c']);
      assert.strictEqual(out.length, 3);
      assert.ok(out[0] instanceof Float32Array, 'index=0 应回填');
      assert.deepStrictEqual(Array.from(out[0]), [1, 1], 'index=0 内容应为 [1,1]');
      assert.strictEqual(out[1], null, '维度不符的位应为 null');
      assert.ok(out[2] instanceof Float32Array, 'index=2 应回填（data 乱序）');
      assert.deepStrictEqual(Array.from(out[2]), [2, 2], 'index=2 内容应为 [2,2]');
      assert.strictEqual(requestCount, 1, '3 条应合为一批');
    } finally {
      await closeServer(server);
    }
  });

  // ── 用例 5：mock 服务超时（timeoutMs 内无响应）→ null ──
  await test('超时：AbortSignal.timeout 生效，超时位为 null', async () => {
    const server = createServer((req, res) => {
      req.resume();
      setTimeout(() => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end('{}');
      }, 500); // 响应远慢于 timeoutMs=50
    });
    const port = await listen(server);
    try {
      const e = createEmbedder({
        baseUrl: `http://127.0.0.1:${port}/v1`,
        model: 'm',
        dimension: 8,
        timeoutMs: 50,
        retries: 0,
      });
      const out = await e.embed(['超时测试']);
      assert.strictEqual(out.length, 1);
      assert.strictEqual(out[0], null, '超时位应为 null');
    } finally {
      await closeServer(server);
    }
  });

  // ── 用例 6：apiKey 存在时携带 Authorization: Bearer 头 ──
  await test('apiKey：请求携带 Bearer 头（mock）', async () => {
    let sawAuth = null;
    const server = createServer((req, res) => {
      sawAuth = req.headers.authorization ?? null;
      req.resume();
      req.on('end', () => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ data: [{ index: 0, embedding: [0.5] }] }));
      });
    });
    const port = await listen(server);
    try {
      const e = createEmbedder({
        baseUrl: `http://127.0.0.1:${port}/v1`,
        model: 'm',
        dimension: 1,
        apiKey: 'secret-key',
        retries: 0,
      });
      const out = await e.embed(['带鉴权']);
      assert.strictEqual(out.length, 1);
      assert.ok(out[0] instanceof Float32Array);
      assert.strictEqual(sawAuth, 'Bearer secret-key', '应携带 Bearer 头');
    } finally {
      await closeServer(server);
    }
  });

  console.log(`\n通过 ${passed} 项，失败 ${failed} 项`);
  process.exit(failed ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});