// tests/taglayer.test.mjs —— 规格 SPEC-P1 §3 Tag 层自测(node 直接运行,无框架)
//
// 覆盖:parseTags(全角变体/保序/position 1-based/取最后一行/无 Tag 行)、
//      updateFile+flush 落盘 tags.jsonl(10 文件 8+ 不同 tag 含共享)、
//      embed 失败 vector:null 且下次 flush 重试、searchTags(真 Ollama 语义 top1)、
//      EPA 训练触发与 epa.json 落盘、重启 load 后不重复训练、sig 不符抛错。
// 依赖:真实 Ollama bge-m3 于 127.0.0.1:11434(core/EPAModule.mjs 与
//      core/ResidualPyramid.mjs 为移植代理已交付的真实移植版)。
// 用法:node tests/taglayer.test.mjs

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createTagLayer } from '../engine/taglayer.mjs';
import { createEmbedder } from '../engine/embed.mjs';

const REAL = { baseUrl: 'http://127.0.0.1:11434/v1', model: 'bge-m3', dimension: 1024 };
const SIG = 'bge-m3@1024';
const DIM = 1024;
const ROOT = `/tmp/vcp-memo-taglayer-test-${process.pid}`;

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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const diaryText = (tags) =>
  `[2026-08-29] - dsh\n[14:30]\n正文内容若干。用于测试 Tag 层的回忆片段。\n\nTag: ${tags}\n`;

// 建临时数据目录
function freshDir(name) {
  const dir = path.join(ROOT, name);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

// 共享 tag 数据集:10 个文件、12 个不同 tag(机器学习/旅行/美食 等为共享 tag)
const FILES = [
  ['f01.md', '机器学习, 旅行, 美食'],
  ['f02.md', '机器学习, 量子物理'],
  ['f03.md', '旅行, 摄影'],
  ['f04.md', '美食, 健身'],
  ['f05.md', '音乐, 机器学习'],
  ['f06.md', '哲学, 旅行'],
  ['f07.md', '编程, 机器学习'],
  ['f08.md', '阅读, 写作'],
  ['f09.md', '美食, 阅读'],
  ['f10.md', '健康, 健身'],
];
const DISTINCT_TAGS = FILES.flatMap(([, t]) => t.split(',').map((s) => s.trim()));

// 8 个不同 tag(EPA 测试用,minTags=8)
const EPA_FILES = [
  ['e01.md', '机器学习, 量子物理'],
  ['e02.md', '机器学习, 美食烹饪'],
  ['e03.md', '旅行摄影, 音乐欣赏'],
  ['e04.md', '健身运动, 哲学思考'],
  ['e05.md', '编程开发, 机器学习'],
  ['e06.md', '量子物理, 音乐欣赏'],
  ['e07.md', '美食烹饪, 健身运动'],
  ['e08.md', '旅行摄影, 哲学思考'],
  ['e09.md', '编程开发, 量子物理'],
  ['e10.md', '机器学习, 健身运动'],
];

// 读 tags.jsonl 返回记录数组
function readTagLines(dir) {
  const file = path.join(dir, 'index', 'tags.jsonl');
  if (!fs.existsSync(file)) return null;
  return fs
    .readFileSync(file, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

async function main() {
  console.log('== taglayer.test.mjs ==');
  fs.rmSync(ROOT, { recursive: true, force: true });

  // ── 组 1:parseTags(纯函数,不依赖 IO) ──
  await test('parseTags:常规半角行,保序且 position 1-based', () => {
    const tagLayer = createTagLayer({
      dataRoot: freshDir('p1'), dimension: DIM, sig: SIG,
      embedder: createEmbedder(REAL),
    });
    const tags = tagLayer.parseTags('正文\n\nTag: 甲, 乙, 丙\n');
    assert.deepStrictEqual(tags, [
      { name: '甲', position: 1 },
      { name: '乙', position: 2 },
      { name: '丙', position: 3 },
    ]);
  });

  await test('parseTags:全角冒号/全角逗号/顿号混用 + 空白折叠 + 丢弃空项', () => {
    const tagLayer = createTagLayer({
      dataRoot: freshDir('p2'), dimension: DIM, sig: SIG,
      embedder: createEmbedder(REAL),
    });
    // 全角冒号、全角逗号、顿号、半角逗号混用;含连续空白与孤立分隔项
    const tags = tagLayer.parseTags('正文\n\nTag：标签  甲，标签乙、标签丙, , 、 标签丁\n');
    assert.deepStrictEqual(tags, [
      { name: '标签 甲', position: 1 }, // 连续空白折叠为单空格
      { name: '标签乙', position: 2 },
      { name: '标签丙', position: 3 },
      { name: '标签丁', position: 4 }, // 空项被丢弃
    ]);
  });

  await test('parseTags:取最后一个 Tag 行,中间 Tag 行被忽略', () => {
    const tagLayer = createTagLayer({
      dataRoot: freshDir('p3'), dimension: DIM, sig: SIG,
      embedder: createEmbedder(REAL),
    });
    const text =
      '[2026-08-29] - dsh\n[14:30]\n今天聊了 Tag 相关话题。\nTag: 甲, 乙\n' +
      '后面还有一段正文,不算 Tag 行: Tagline 不是标签。\nTag: 丙, 丁\n';
    const tags = tagLayer.parseTags(text);
    assert.deepStrictEqual(tags, [
      { name: '丙', position: 1 },
      { name: '丁', position: 2 },
    ]);
  });

  await test('parseTags:无 Tag 行返回空数组;不去重(保序保留重复项)', () => {
    const tagLayer = createTagLayer({
      dataRoot: freshDir('p4'), dimension: DIM, sig: SIG,
      embedder: createEmbedder(REAL),
    });
    assert.deepStrictEqual(tagLayer.parseTags('[2026-08-29] - dsh\n[14:30]\n正文无标签。\n'), []);
    assert.deepStrictEqual(tagLayer.parseTags(''), []);
    // 不去重:同一行内重复的 tag 名都保留,各自带 position
    const dup = tagLayer.parseTags('正文\n\nTag: 甲, 乙, 甲\n');
    assert.deepStrictEqual(dup, [
      { name: '甲', position: 1 },
      { name: '乙', position: 2 },
      { name: '甲', position: 3 },
    ]);
  });

  // ── 组 2:updateFile + flush 落盘 ──
  await test('updateFile+flush:10 文件 12 个不同 tag(含共享)落盘,结构/向量/occurrences 正确', async () => {
    const dir = freshDir('flush');
    const embedder = createEmbedder(REAL);
    const tagLayer = createTagLayer({
      dataRoot: dir, dimension: DIM, sig: SIG, embedder,
      epaConfig: { minTags: 8 },
      log: () => {},
    });
    await tagLayer.load();
    for (const [file, tags] of FILES) {
      tagLayer.updateFile(`diaries/dsh/${file}`, diaryText(tags));
    }
    await tagLayer.flush();

    // tags.jsonl 行数与结构
    const lines = readTagLines(dir);
    assert.ok(lines, 'tags.jsonl 应已生成');
    assert.strictEqual(lines.length, 12, `应有 12 个不同 tag,实际 ${lines.length}`);
    const names = new Set(DISTINCT_TAGS);
    assert.strictEqual(new Set(lines.map((l) => l.name)).size, 12, 'tag 名应全部不同');

    for (const row of lines) {
      assert.equal(typeof row.name, 'string');
      assert.ok(row.name.length > 0);
      // 所有 tag 均已向量化(真 Ollama)→ 1024 维普通数组
      assert.ok(Array.isArray(row.vector), `${row.name} 的 vector 应为数组`);
      assert.strictEqual(row.vector.length, DIM, `${row.name} 的 vector 应为 ${DIM} 维`);
      assert.ok(Array.isArray(row.occurrences));
      for (const o of row.occurrences) {
        assert.equal(typeof o.file, 'string');
        assert.ok(o.file.startsWith('diaries/dsh/'));
        assert.ok(Number.isInteger(o.position) && o.position >= 1, `position 应从 1 开始`);
      }
    }

    // 共享 tag 断言:机器学习 4 个 occurrence,旅行 3 个,各自 position 正确(保序 1-based)
    const byName = new Map(lines.map((l) => [l.name, l]));
    const ml = byName.get('机器学习');
    assert.strictEqual(ml.occurrences.length, 4, '机器学习出现于 4 个文件');
    const occByFile = Object.fromEntries(ml.occurrences.map((o) => [o.file.split('/').pop(), o.position]));
    assert.deepStrictEqual(occByFile, { 'f01.md': 1, 'f02.md': 1, 'f05.md': 2, 'f07.md': 2 });
    const travel = byName.get('旅行');
    assert.strictEqual(travel.occurrences.length, 3);
    assert.deepStrictEqual(
      Object.fromEntries(travel.occurrences.map((o) => [o.file.split('/').pop(), o.position])),
      { 'f01.md': 2, 'f03.md': 1, 'f06.md': 2 }
    );
    const food = byName.get('美食');
    assert.strictEqual(food.occurrences.length, 3);

    // stats
    const st = tagLayer.stats();
    assert.strictEqual(st.tagCount, 12);
    assert.strictEqual(st.vectorizedTags, 12);
    assert.strictEqual(st.epaTrained, true, '12 个有向量 tag ≥ minTags,EPA 应已训练');
  });

  // ── 组 3:embed 失败 → vector:null → 下次 flush 重试 ──
  await test('embed 失败:vector:null 落盘,下次 flush 重试成功向量化', async () => {
    const dir = freshDir('retry');
    const realEmbedder = createEmbedder(REAL);
    let broken = true; // 模拟端点故障:整批返回 null(与错误端点行为一致)
    const embedder = {
      sig: SIG,
      model: 'bge-m3',
      dimension: DIM,
      async embed(texts) {
        if (broken) return texts.map(() => null);
        return realEmbedder.embed(texts);
      },
    };
    const tagLayer = createTagLayer({
      dataRoot: dir, dimension: DIM, sig: SIG, embedder,
      epaConfig: { minTags: 8 },
      log: () => {},
    });
    await tagLayer.load();
    tagLayer.updateFile('diaries/dsh/g01.md', diaryText('机器学习, 量子物理'));
    await tagLayer.flush();

    // 失败:所有 tag vector 为 null,下次 flush 重试
    let lines = readTagLines(dir);
    assert.ok(lines && lines.length === 2, '应有 2 条 tag 记录');
    for (const row of lines) {
      assert.strictEqual(row.vector, null, `失败时应 vector:null(${row.name})`);
    }
    assert.strictEqual(tagLayer.stats().vectorizedTags, 0);

    // 恢复端点,flush 重试 → 全部向量化
    broken = false;
    await tagLayer.flush();
    lines = readTagLines(dir);
    for (const row of lines) {
      assert.ok(Array.isArray(row.vector) && row.vector.length === DIM, `${row.name} 应已向量化为 ${DIM} 维`);
    }
    assert.strictEqual(tagLayer.stats().vectorizedTags, 2);
  });

  // ── 组 4:searchTags(真 Ollama) ──
  await test('searchTags:语义相近查询向量把相关 tag 排 top1,自匹配相似度 ~1', async () => {
    const dir = freshDir('search');
    const embedder = createEmbedder(REAL);
    const tagLayer = createTagLayer({
      dataRoot: dir, dimension: DIM, sig: SIG, embedder,
      epaConfig: { minTags: 8 },
      log: () => {},
    });
    await tagLayer.load();
    const names = ['机器学习', '量子物理', '美食烹饪', '旅行摄影', '音乐欣赏', '健身运动'];
    names.forEach((n, i) => tagLayer.updateFile(`diaries/dsh/s${i}.md`, diaryText(n)));
    await tagLayer.flush();

    // 语义相近:查询与"机器学习"语义相关但不是 tag 原文
    const [qvec] = await embedder.embed(['机器学习的算法与模型应用']);
    const hits = await tagLayer.searchTags(qvec, 10);
    assert.strictEqual(hits.length, 6, '应返回全部 6 个有向量 tag');
    assert.strictEqual(hits[0].name, '机器学习', `top1 应为语义相关的 机器学习,实际 ${hits[0].name}`);
    assert.ok(hits[0].similarity > 0.4, `相似度应明显为正,实际 ${hits[0].similarity.toFixed(3)}`);
    assert.strictEqual(hits[0].score, hits[0].similarity, 'score 别名应与 similarity 同值(移植版金字塔兼容)');
    assert.strictEqual(hits[0].id, 0, 'id 应为记录下标');
    assert.ok(hits[0].vector instanceof Float32Array && hits[0].vector.length === DIM);
    // 递减序检查
    for (let i = 1; i < hits.length; i++) {
      assert.ok(hits[i - 1].similarity >= hits[i].similarity, '应按相似度降序');
    }

    // 自匹配:查询恰好是某 tag 名 → top1 即该 tag,相似度接近 1
    const [q2] = await embedder.embed(['健身运动']);
    const hits2 = await tagLayer.searchTags(q2, 10);
    assert.strictEqual(hits2[0].name, '健身运动');
    assert.ok(hits2[0].similarity > 0.9, `自匹配相似度应 >0.9,实际 ${hits2[0].similarity.toFixed(4)}`);
  });

  // ── 组 5:EPA 训练触发与缓存 ──
  await test('EPA:tagHash 变化触发重训,epa.json 落盘(含 sig/tagHash/basis)', async () => {
    const dir = freshDir('epa');
    const tagLayer = createTagLayer({
      dataRoot: dir, dimension: DIM, sig: SIG,
      embedder: createEmbedder(REAL),
      epaConfig: { minTags: 8, clusterCount: 12, maxBasisDim: 32 },
      log: () => {},
    });
    await tagLayer.load();
    for (const [file, tags] of EPA_FILES) tagLayer.updateFile(`diaries/dsh/${file}`, diaryText(tags));
    await tagLayer.flush();

    const epaFile = path.join(dir, 'index', 'epa.json');
    assert.ok(fs.existsSync(epaFile), 'epa.json 应已落盘');
    const epa1 = JSON.parse(fs.readFileSync(epaFile, 'utf8'));
    assert.strictEqual(epa1.sig, SIG);
    assert.match(epa1.tagHash, /^[0-9a-f]{32}$/);
    assert.ok(epa1.basis && Array.isArray(epa1.basis.basis) && epa1.basis.basis.length > 0, 'basis 应有主轴');
    assert.ok(Array.isArray(epa1.basis.mean));
    assert.strictEqual(tagLayer.stats().epaTrained, true);

    // 新增一个全新 tag → tagHash 变化 → 再次训练
    tagLayer.updateFile('diaries/dsh/e11.md', diaryText('徒步旅行'));
    await tagLayer.flush();
    const epa2 = JSON.parse(fs.readFileSync(epaFile, 'utf8'));
    assert.notStrictEqual(epa2.tagHash, epa1.tagHash, 'tag 集合变化后 tagHash 应改变');
    assert.strictEqual(tagLayer.stats().vectorizedTags, 9);
    assert.strictEqual(tagLayer.stats().epaTrained, true);
  });

  await test('EPA:重启 load 后 trained 且不重复训练(epa.json 字节不变)', async () => {
    const dir = freshDir('epareload');
    const make = () =>
      createTagLayer({
        dataRoot: dir, dimension: DIM, sig: SIG,
        embedder: createEmbedder(REAL),
        epaConfig: { minTags: 8 },
        log: () => {},
      });
    // 第一次:训练并落盘
    const tagA = make();
    await tagA.load();
    for (const [file, tags] of EPA_FILES) tagA.updateFile(`diaries/dsh/${file}`, diaryText(tags));
    await tagA.flush();
    assert.strictEqual(tagA.stats().epaTrained, true);
    const epaFile = path.join(dir, 'index', 'epa.json');
    const before = fs.readFileSync(epaFile);

    // 重启:新实例 load() → 直接装载缓存基底,不训练、不重写 epa.json
    const tagB = make();
    await tagB.load();
    assert.strictEqual(tagB.stats().epaTrained, true, 'load 后应直接处于 trained');
    const afterLoad = fs.readFileSync(epaFile);
    assert.ok(afterLoad.equals(before), 'load 不应重写 epa.json(未重复训练)');

    // 再做一次 tag 集无变化的 flush(仅 occurrence 变更)→ 仍不重训
    tagB.updateFile('diaries/dsh/e01.md', diaryText('机器学习, 量子物理')); // 与 e01 相同 tag
    await tagB.flush();
    const afterFlush = fs.readFileSync(epaFile);
    assert.ok(afterFlush.equals(before), 'tagHash 未变时 flush 不应重训/重写 epa.json');
    assert.strictEqual(tagB.stats().epaTrained, true);
  });

  // ── 组 6:sig 校验 ──
  await test('load:epa.json 中 sig 不符抛签名错误,含新旧签名', async () => {
    const dir = freshDir('sig');
    const indexDir = path.join(dir, 'index');
    fs.mkdirSync(indexDir, { recursive: true });
    fs.writeFileSync(
      path.join(indexDir, 'tags.jsonl'),
      JSON.stringify({ name: '机器学习', vector: null, occurrences: [] }) + '\n'
    );
    fs.writeFileSync(
      path.join(indexDir, 'epa.json'),
      JSON.stringify({ sig: 'old-model@512', tagHash: 'abc', basis: null })
    );
    const tagLayer = createTagLayer({
      dataRoot: dir, dimension: DIM, sig: SIG,
      embedder: createEmbedder(REAL),
    });
    let caught = null;
    try {
      await tagLayer.load();
    } catch (e) {
      caught = e;
    }
    assert.ok(caught, 'sig 不符应抛错');
    assert.match(caught.message, /签名已变/);
    assert.ok(caught.message.includes('old-model@512'), '错误应含旧签名');
    assert.ok(caught.message.includes(SIG), '错误应含新签名');
  });

  console.log(`\n通过 ${passed} 项,失败 ${failed} 项`);
  fs.rmSync(ROOT, { recursive: true, force: true });
  process.exit(failed ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  fs.rmSync(ROOT, { recursive: true, force: true });
  process.exit(1);
});