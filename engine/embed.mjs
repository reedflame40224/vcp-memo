// engine/embed.mjs —— Ollama/OpenAI 兼容 embedding 客户端（规格 §5）
// 零依赖：只用 node 内置与全局 fetch；批次并发 worker 池自实现，不引任何第三方库。

/**
 * 创建 embedding 客户端。
 * config: { baseUrl, model, dimension, apiKey?, batchSize?=16, concurrency?=2, timeoutMs?=60000, retries?=2 }
 * 返回 { embed, model, dimension, sig }，sig = `${model}@${dimension}`。
 * embed(texts) → Promise<(Float32Array|null)[]>，长度恒等于输入；
 * 失败位一律 null，绝不把错误数据当成向量，也不向调用方抛出（参数类型错误除外）。
 */
export function createEmbedder(config) {
  const {
    baseUrl,
    model,
    dimension,
    apiKey,
    batchSize = 16,
    concurrency = 2,
    timeoutMs = 60000,
    retries = 2,
  } = config;

  const sig = `${model}@${dimension}`;
  // 去掉尾部斜杠，拼出 OpenAI 兼容 embeddings 端点
  const endpoint = `${String(baseUrl).replace(/\/+$/, '')}/embeddings`;
  // 指数退避间隔（毫秒）：首次失败等 1s、再次 3s，之后保持 3s
  const backoff = [1000, 3000];

  // 请求一批文本（≤ batchSize 条）；重试耗尽返回整批 null，绝不抛出。
  async function fetchBatch(batch) {
    let lastError = null;
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const headers = { 'Content-Type': 'application/json' };
        if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
        const res = await fetch(endpoint, {
          method: 'POST',
          headers,
          body: JSON.stringify({ model, input: batch }),
          signal: AbortSignal.timeout(timeoutMs),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        const json = await res.json();
        if (!json || !Array.isArray(json.data)) throw new Error('响应结构非法：缺少 data 数组');
        // 按 data 元素的 index 字段回填；缺失或维度 ≠ dimension 的位记 null（规格 §5）
        const byIndex = new Map();
        for (const item of json.data) {
          if (item && typeof item === 'object') byIndex.set(item.index, item.embedding);
        }
        const out = new Array(batch.length);
        for (let i = 0; i < batch.length; i++) {
          const emb = byIndex.get(i);
          out[i] = Array.isArray(emb) && emb.length === dimension ? Float32Array.from(emb) : null;
        }
        return out;
      } catch (err) {
        lastError = err;
        if (attempt < retries) {
          const wait = backoff[Math.min(attempt, backoff.length - 1)];
          await new Promise((r) => setTimeout(r, wait));
        }
      }
    }
    // 重试耗尽：整批填 null。只记批次大小与错误消息，绝不打印正文（日记是隐私）。
    console.error(
      `[vcp-memo] embedding 批次失败（批次大小 ${batch.length}）：${lastError?.message ?? String(lastError)}`
    );
    return new Array(batch.length).fill(null);
  }

  async function embed(texts) {
    if (!Array.isArray(texts)) throw new TypeError('embed 需要 string[] 参数');
    const input = texts.map((t) => String(t));
    const result = new Array(input.length);
    if (input.length === 0) return result;
    // 切批
    const batches = [];
    for (let i = 0; i < input.length; i += batchSize) {
      batches.push({ offset: i, texts: input.slice(i, i + batchSize) });
    }
    // 自实现 worker 池：并发 ≤ concurrency，共享游标逐个领批（JS 单线程，游标无需锁）
    let cursor = 0;
    const workers = Array.from({ length: Math.min(concurrency, batches.length) }, async () => {
      for (;;) {
        const batch = batches[cursor++];
        if (!batch) break;
        const out = await fetchBatch(batch.texts);
        for (let j = 0; j < out.length; j++) result[batch.offset + j] = out[j];
      }
    });
    await Promise.all(workers);
    return result;
  }

  return { embed, model, dimension, sig };
}