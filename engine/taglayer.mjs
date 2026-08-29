// engine/taglayer.mjs —— Tag 层(规格 SPEC-P1 §3,自写)
//
// 职责:日记末行 "Tag: " 解析(带 position)→ tag 记录 → 新 tag 向量化 →
//      tags.jsonl 原子持久化 → EPA 基底训练触发与缓存 → 暴力余弦 searchTags →
//      残差金字塔装配(种子来源,供给 P1 的 recall 诊断字段)。
// 零依赖:仅 node 内置模块(node:crypto / node:fs/promises / node:path)。
//
// 纪律(继承 P0):
//  - 文件写盘必须原子化:写临时文件 + rename(P0 §1.5);
//  - embed 失败位 vector 保持 null,下次 flush 重试,绝不把错误向量写进索引(P0 §1.6 / P1 §3.2);
//  - flush 假定由上层"索引队列空闲"串行调用(store §6),本层不做并发互斥。
//
// 已知决策(偏离规格之处,汇报中同样记录):
//  - tag 记录只增不删:occurrences 可为空,记录与向量保留,避免重复 embedding 成本;
//  - tagHash 变化时先清空 epa 缓存再 initialize():EPAModule(移植版)缓存命中会跳过训练,
//    必须在 tag 集合变化时强制重训(规格 §3.3 "hash 一致直接用" 的反面);
//  - load() 只"装载"不训练:hash 不一致时 EPA 保持未训练,由首个空闲 flush 重训
//    (§3.3 训练触发点在 flush 之后,不在 load);
//  - searchTags 额外返回 score 字段(与 similarity 同值):移植版 ResidualPyramid 原样
//    保留了原代码对 `res.score` 的读取,而 SPEC-P1 §2 契约写的是 similarity,补别名以兼容;
//  - tags.jsonl 解析遇到损坏行直接抛错(派生资产被篡改时失败出声),不做静默丢弃。

import { createHash } from 'node:crypto';
import { readFile, writeFile, rename, mkdir, rm } from 'node:fs/promises';
import path from 'node:path';
import { EPAModule } from '../core/EPAModule.mjs';
import { ResidualPyramid } from '../core/ResidualPyramid.mjs';

export function createTagLayer(deps) {
  const {
    dataRoot,
    dimension,
    sig,
    embedder,
    epaConfig = {},
    pyramidConfig = {},
    log = () => {},
  } = deps;

  // ── 配置(规格 §6 tagmemo 节,缺省即用) ──
  const epaCfg = {
    dimension,
    minTags: epaConfig.minTags ?? 8,
    clusterCount: epaConfig.clusterCount ?? 12,
    maxBasisDim: epaConfig.maxBasisDim ?? 32,
  };
  const pyrCfg = {
    dimension,
    maxLevels: pyramidConfig.maxLevels ?? 3,
    topK: pyramidConfig.topK ?? 10,
    minEnergyRatio: pyramidConfig.minEnergyRatio ?? 0.1,
  };
  const indexDir = path.join(dataRoot, 'index');
  const tagsFile = path.join(indexDir, 'tags.jsonl');
  const epaFile = path.join(indexDir, 'epa.json');

  // ── 内存状态 ──
  let records = []; // [{ id, name, vector: Float32Array|null, occurrences: [{file, position}] }]
  let dirty = false; // tags.jsonl 有待落盘变更
  let lastHash = null; // 最近一次成功训练的 tagHash(未训练为 null)
  let epaCacheValue = null; // 'epa_basis_cache' 的当前值(JSON 可序列化;持久化由本层负责)
  let epa = null;
  let pyramid = null;

  // ── 辅助 ──
  async function atomicWrite(file, content) {
    // 原子写:临时文件 + rename(P0 §1.5)
    const tmp = `${file}.${process.pid}.${Date.now()}.tmp`;
    await writeFile(tmp, content, 'utf8');
    await rename(tmp, file);
  }

  // ── §3.1 Tag 解析(对齐 VCP 末行 Tag 契约) ──
  function parseTags(text) {
    // 取正文中最后一个匹配 /^\s*Tag\s*[:：]\s*(.+)$/m 的行
    let last = null;
    const source = String(text ?? '');
    for (const line of source.split(/\r?\n/)) {
      const m = /^\s*Tag\s*[:：]\s*(.+)$/.exec(line);
      if (m) last = m[1];
    }
    if (last === null) return [];
    // 按 /[,，、]/ 切分;逐项清洗(去首尾空白、折叠连续空白)、丢弃空项;
    // 保序、不排序、不去重;position 从 1 开始(VCP 语义:position>0 为有序记录)
    const tags = [];
    for (const raw of last.split(/[,，、]/)) {
      const name = raw.trim().replace(/\s+/g, ' ');
      if (name.length > 0) tags.push({ name, position: tags.length + 1 });
    }
    return tags;
  }

  // ── §3.3 tagHash:"有向量的 tag 名排序列表 + sig" 取 md5 ──
  function computeTagHash() {
    const names = records
      .filter((r) => r.vector !== null)
      .map((r) => r.name)
      .sort();
    return createHash('md5').update(JSON.stringify({ sig, names })).digest('hex');
  }

  // ── 供 EPAModule 消费的数据源与同步 KV 缓存 ──
  const tagProvider = {
    listTagVectors() {
      // 只给有向量的记录(规格 §1:Array<{id, name, vector:Float32Array}>)
      return records
        .filter((r) => r.vector !== null)
        .map((r) => ({ id: r.id, name: r.name, vector: r.vector }));
    },
  };
  const cache = {
    get(key) {
      return key === 'epa_basis_cache' ? epaCacheValue : null;
    },
    set(key, value) {
      if (key === 'epa_basis_cache') epaCacheValue = value;
    },
  };

  // ── §3.2 tags.jsonl 落盘 ──
  async function writeTagsFile() {
    await mkdir(indexDir, { recursive: true });
    const lines = records.map((r) =>
      JSON.stringify({
        name: r.name,
        vector: r.vector ? Array.from(r.vector) : null, // Float32Array → 普通数组入 JSONL
        occurrences: r.occurrences, // [{ file, position }]
      })
    );
    // 空库写空文件(load 时按空行跳过)
    await atomicWrite(tagsFile, lines.length ? lines.join('\n') + '\n' : '');
  }

  // ── §3 更新/删除 ──
  function updateFile(rel, text) {
    const tags = parseTags(text);
    // 文件级替换:先移除该文件全部旧 occurrence,再按本次解析写入
    for (const rec of records) {
      if (rec.occurrences.some((o) => o.file === rel)) {
        rec.occurrences = rec.occurrences.filter((o) => o.file !== rel);
      }
    }
    for (const t of tags) {
      let rec = records.find((r) => r.name === t.name);
      if (!rec) {
        // 新 tag:vector 置 null → 下次 flush 待 embed(失败则下次继续重试)
        rec = { id: records.length, name: t.name, vector: null, occurrences: [] };
        records.push(rec);
      }
      rec.occurrences.push({ file: rel, position: t.position });
    }
    dirty = true;
  }

  function removeFile(rel) {
    let changed = false;
    for (const rec of records) {
      if (rec.occurrences.some((o) => o.file === rel)) {
        rec.occurrences = rec.occurrences.filter((o) => o.file !== rel);
        changed = true;
      }
    }
    if (changed) dirty = true;
  }

  // ── §3 暴力余弦 searchTags(无向量 tag 不参与,规格 §3.2) ──
  async function searchTags(vector, topK = 10) {
    const q = vector instanceof Float32Array ? vector : new Float32Array(vector);
    const qNorm = Math.sqrt(dot(q, q)) || 1;
    const hits = [];
    for (const rec of records) {
      if (rec.vector === null) continue;
      if (rec.vector.length !== q.length) continue; // 维度不一致防御(正常不会发生)
      const n = Math.sqrt(dot(rec.vector, rec.vector)) || 1;
      let s = 0;
      for (let i = 0; i < q.length; i++) s += (q[i] / qNorm) * (rec.vector[i] / n);
      hits.push({
        id: rec.id,
        name: rec.name,
        vector: rec.vector,
        similarity: s,
        // 兼容别名:移植版 ResidualPyramid 原样保留 `res.score` 读取
        score: s,
      });
    }
    hits.sort((a, b) => b.similarity - a.similarity);
    return hits.slice(0, topK);
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  // ── §3.3 EPA 重训触发(仅在 flush 内调用;load 只装载) ──
  async function maybeRetrain() {
    const hash = computeTagHash();
    const vectorized = records.filter((r) => r.vector !== null).length;
    if (hash === lastHash) return; // tag 集合未变,不需要重训
    if (vectorized < epaCfg.minTags) {
      log('info', `[taglayer] 有向量 tag ${vectorized} < minTags(${epaCfg.minTags}),EPA 暂不训练`);
      return;
    }
    // hash 变化:清空旧缓存强制重训(否则 EPAModule 缓存命中会跳过训练)
    epaCacheValue = null;
    try {
      const ok = await epa.initialize();
      if (ok && epa.trained) {
        lastHash = hash;
        await mkdir(indexDir, { recursive: true });
        // 结果写 index/epa.json(含 sig、tagHash、basis;basis 即 epa_basis_cache 缓存值)
        await atomicWrite(
          epaFile,
          JSON.stringify({ sig, tagHash: hash, basis: epaCacheValue }, null, 2)
        );
        log('info', `[taglayer] EPA 训练完成:${vectorized} 个有向量 tag,hash=${hash.slice(0, 8)}…`);
      } else {
        // 训练失败(内部捕获)或 tag 不足:保持未训练,下次 flush 重试
        log('info', '[taglayer] EPA 训练未完成,保留未训练状态,下次 flush 重试');
      }
    } catch (err) {
      log('error', `[taglayer] EPA 训练异常:${err?.stack ?? err}`);
    }
  }

  // ── §3 flush:embed 新 tag → 重写 tags.jsonl → tagSet 变化则重训 EPA ──
  async function flush() {
    const pending = records.filter((r) => r.vector === null);
    const vectorized = records.length - pending.length;
    // 早退条件:无待落盘变更、无待 embed 的 tag、且不需要重训 EPA。
    // 最后一项对齐本文件头部"由首个空闲 flush 重训"的契约:load() 时 epa.json 的
    // tagHash 与当前 tag 集合不一致 → 保持未训练,首个空闲 flush 必须在此拉起训练
    // (否则未训练状态会永久滞留)。
    const hashDirty = lastHash !== computeTagHash() && vectorized >= epaCfg.minTags;
    if (!dirty && pending.length === 0 && !hashDirty) return;

    // 1. 新 tag 向量化(失败位留 null → 下次 flush 重试)
    if (pending.length > 0) {
      const names = pending.map((r) => r.name);
      let vectors = null;
      try {
        vectors = await embedder.embed(names);
      } catch (err) {
        // embedder 契约本就不抛(失败位 null),此处防御再兜一层
        log('warn', `[taglayer] tag 向量化调用异常:${err?.message ?? err}`);
      }
      if (vectors) {
        for (let i = 0; i < pending.length; i++) {
          const v = vectors[i];
          if (v && v.length === dimension) pending[i].vector = v;
        }
      }
    }

    // 2. 原子重写 tags.jsonl
    await writeTagsFile();
    dirty = false;

    // 3. tagHash 变化且有向量 tag 数 ≥ minTags → 重训 EPA(结果写 epa.json)
    await maybeRetrain();
  }

  // ── §3 load:读 tags.jsonl + epa.json;校验 sig,不符则抛 sig 错误 ──
  async function load() {
    await mkdir(indexDir, { recursive: true });

    // 先构造 EPA 与金字塔(与数据无关):即使后续 tags.jsonl 解析抛错,
    // 层对象也已就绪,flush/降级路径仍可用
    epa = new EPAModule(tagProvider, cache, epaCfg);
    pyramid = new ResidualPyramid(searchTags, pyrCfg);

    // 1. 读 tags.jsonl(缺失视为空库)
    records = [];
    dirty = false;
    try {
      const content = await readFile(tagsFile, 'utf8');
      for (const line of content.split(/\r?\n/)) {
        const t = line.trim();
        if (!t) continue;
        const row = JSON.parse(t); // 损坏行直接抛错(派生资产被篡改,失败出声)
        records.push({
          id: records.length, // id 即数组下标(只增不删,稳定)
          name: row.name,
          vector: row.vector ? Float32Array.from(row.vector) : null,
          occurrences: Array.isArray(row.occurrences) ? row.occurrences : [],
        });
      }
    } catch (err) {
      if (err.code !== 'ENOENT') throw err;
    }

    // 2. 读 epa.json;sig 不一致 → 抛 sig 错误(与 P0 拒绝服务语义一致)
    let epaJson = null;
    try {
      const content = await readFile(epaFile, 'utf8');
      if (content.trim()) epaJson = JSON.parse(content);
    } catch (err) {
      if (err.code !== 'ENOENT') throw err;
    }
    if (epaJson) {
      if (epaJson.sig !== sig) {
        throw new Error(
          `嵌入模型签名已变(旧 ${epaJson.sig} → 新 ${sig}),Tag 向量与日记语义空间不再一致;` +
            `确认切换模型请先调用 memory_admin 的 rebuild 全量重建`
        );
      }
      const hash = computeTagHash();
      // hash 一致且缓存结构可校验(至少含 mean 与 basis 数组,对齐 EPAModule._loadFromCache 校验)
      // → 直接用缓存基底,load 阶段不训练
      if (
        epaJson.tagHash === hash &&
        epaJson.basis &&
        epaJson.basis.mean &&
        Array.isArray(epaJson.basis.basis)
      ) {
        epaCacheValue = epaJson.basis;
        lastHash = hash;
      } else if (epaJson.tagHash !== hash) {
        log('info', `[taglayer] epa.json 的 tagHash 与当前 tag 集合不一致,等待首个空闲 flush 重训`);
      }
    }

    // 3. 有可用缓存时装载(load 只装载、不训练)
    if (epaCacheValue !== null) {
      try {
        await epa.initialize();
        if (!epa.trained) {
          // 缓存装载失败(理论罕见):丢弃坏缓存,恢复未训练,由 flush 重训
          log('warn', '[taglayer] EPA 缓存基底装载失败,恢复未训练状态,下次 flush 重训');
          epaCacheValue = null;
          lastHash = null;
        }
      } catch (err) {
        log('error', `[taglayer] EPA 缓存基底装载异常:${err?.stack ?? err}`);
        epaCacheValue = null;
        lastHash = null;
      }
    }
  }

  // ── §6 store.doRebuild 的"清空重建"入口:丢弃全部内存状态与派生产物,
//    由调用方随后逐文件 updateFile + flush 从日记真相源重新派生 ──
  async function reset() {
    records = [];
    dirty = false;
    lastHash = null;
    epaCacheValue = null;
    try {
      await rm(tagsFile, { force: true });
    } catch { /* 文件不存在或删除失败都不阻止重建 */ }
    try {
      await rm(epaFile, { force: true });
    } catch { /* 同上 */ }
    // 旧层对象可能持有训练态,重建为未训练的新对象
    epa = new EPAModule(tagProvider, cache, epaCfg);
    pyramid = new ResidualPyramid(searchTags, pyrCfg);
  }

  function stats() {
    return {
      tagCount: records.length,
      vectorizedTags: records.filter((r) => r.vector !== null).length,
      epaTrained: !!(epa && epa.trained),
    };
  }

  return {
    load,
    reset,
    parseTags,
    updateFile,
    removeFile,
    flush,
    searchTags,
    get epa() {
      return epa;
    },
    get pyramid() {
      return pyramid;
    },
    stats,
  };
}