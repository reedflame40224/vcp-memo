// tests/docs.test.mjs —— 规格 P3 §2/§4 文档与备份脚本静态断言(node 直接运行,无测试框架)
// 用法:node tests/docs.test.mjs(纯静态检查,不执行 bat、不依赖 Ollama)
// 断言:scripts/backup-vcp-memo.bat 存在且含 robocopy 与 \\wsl$ 路径;
//       README 含「运维」「备份与恢复」节与四条 CLI 命令名。

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');

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

async function main() {
  console.log('== docs.test.mjs ==');

  // ── §2:scripts/backup-vcp-memo.bat ──
  const batPath = join(ROOT, 'scripts', 'backup-vcp-memo.bat');
  await test('§2: scripts/backup-vcp-memo.bat 存在', () => {
    assert.ok(existsSync(batPath), '缺少 scripts/backup-vcp-memo.bat');
  });
  const bat = existsSync(batPath) ? readFileSync(batPath, 'utf8') : '';

  await test('§2: bat 内含 robocopy 备份命令', () => {
    assert.ok(bat.includes('robocopy'), '未找到 robocopy 命令');
  });
  await test('§2: bat 含 \\\\wsl$ UNC 源路径', () => {
    assert.ok(bat.includes('\\\\wsl$'), '未找到 \\\\wsl$ 路径');
  });
  await test('§2: bat 带中文自说明注释(用法/源/目标/隐私)', () => {
    assert.ok(bat.includes('rem'), '无 rem 注释');
    assert.ok(bat.includes('用法'), '注释未说明用法');
    assert.ok(bat.includes('diaries'), '注释未说明只备份 diaries');
  });
  await test('§2 改进: bat 有 robocopy 退出码说明', () => {
    assert.ok(bat.includes('退出码'), '未说明 robocopy 退出码含义');
    assert.ok(/8\s*\+/.test(bat) || bat.includes('GEQ 8'), '未说明 8+ 为失败');
  });
  await test('§2 改进: bat 对源目录不存在有友好报错', () => {
    assert.ok(bat.includes('不存在') && (bat.includes('错误') || bat.includes('[错误]')),
      '源不可达时缺少友好报错');
  });
  await test('§2: bat 可用环境变量覆盖目标目录', () => {
    assert.ok(bat.includes('VCP_MEMO_BACKUP_DST'), '缺少 VCP_MEMO_BACKUP_DST 覆盖');
  });

  // ── §4:README「运维」章 ──
  const readmePath = join(ROOT, 'README.md');
  await test('§4: README 存在', () => {
    assert.ok(existsSync(readmePath), '缺少 README.md');
  });
  const readme = existsSync(readmePath) ? readFileSync(readmePath, 'utf8') : '';

  await test('§4: README 含「运维」章(二级标题)', () => {
    assert.ok(/^## 运维$/m.test(readme), '缺少 ## 运维 章');
  });
  await test('§4: README 含「备份与恢复」节', () => {
    assert.ok(/^### 备份与恢复$/m.test(readme), '缺少「备份与恢复」节');
  });
  await test('§4: README 四条 CLI 命令名齐备(stats/rebuild/tags/doctor)', () => {
    for (const cmd of ['stats', 'rebuild', 'tags', 'doctor']) {
      assert.ok(readme.includes(`vcp-memo.mjs ${cmd}`), `缺少 CLI 命令文档: ${cmd}`);
    }
  });
  await test('§4: README 含 CLI 用法/退出码说明', () => {
    assert.ok(readme.includes('--dataRoot'), '未说明 --dataRoot 覆盖');
    assert.ok(readme.includes('doctor 发现问题 1') || readme.includes('退出码'), '未说明退出码');
  });
  await test('§4: README 含 schtasks 注册示例与 git 备选(私有提醒)', () => {
    assert.ok(readme.includes('schtasks /Create'), '缺少 schtasks 注册命令示例');
    assert.ok(readme.includes('git') && readme.includes('私有'), '缺少 git 备选或隐私提醒');
  });
  await test('§4: README 含恢复演练步骤(临时目录 → doctor/rebuild → 对比日记数)', () => {
    assert.ok(readme.includes('恢复演练'), '缺少恢复演练');
    assert.ok(readme.includes('--dataRoot') && readme.includes('临时目录'), '演练未用临时 dataRoot');
  });
  await test('§4: README 含换模型流程(config 覆盖 → rebuild → 验证)', () => {
    assert.ok(readme.includes('换 embedding 模型'), '缺少换模型节');
    assert.ok(readme.includes('memory_admin') && readme.includes('rebuild'), '换模型未提 rebuild');
  });
  await test('§4: README FAQ 含噪声调 truncate 与 DSH 运行时 CLI rebuild 需重启', () => {
    assert.ok(readme.includes('truncate'), 'FAQ 未提调 truncate');
    assert.ok(readme.includes('重启 DSH') || readme.includes('重启'), 'FAQ 未提 rebuild 后重启');
  });

  console.log(`\n通过 ${passed} 项，失败 ${failed} 项`);
  process.exit(failed ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});