# VCP Memo for DSH — P1.5 实现规格书(被动注入)

> 与 SPEC.md(P0)、SPEC-P1.md(Tag 层)共同构成契约。此前全部测试必须保持绿色。
> 目标:VCP 哲学的灵魂——"回忆先于意识"。每轮对话在模型形成回答**之前**,
> 把相关记忆以 plugin-source 的 user message 注入请求,不再依赖模型主动调 recall_memory。

## 0. 机制(已在 DSH 源码中验证)

- 钩子:`ctx.on('agent/pre-step', async ({ agent, messages, turn, step, signal }, next) => PreStepDecision)`,
  waterfall 模式;先 `const decision = await next()` 拿下游决定,再决定是否拼接。
  `PreStepDecision = { kind:'reject' } | { kind:'enter', messages: UserMessage[] }`。
- 注入位置(照 agent-instructions 的范式):`decision.messages.toSpliced(lastClaimedIndex + 1, 0, memMsg)`,
  其中 `lastClaimedIndex = decision.messages.findLastIndex(m => messages.includes(m))`;
  -1 时拼到末尾。
- 消息手工构造(零依赖,无需 import 任何 dsh 包):
  ```js
  {
    id: crypto.randomUUID(),            // node:crypto
    role: 'user',
    content: [{ type: 'text', text }],
    source: { kind: 'plugin', plugin: 'vcp-memo', form: 'recall' },
  }
  ```
  构造后递归 `Object.freeze`(对齐 harness 的不可变约定)。
- prompt 纪律:`ctx.get('systemPrompt')?.section({ name: 'vcp-memo:discipline', order: 400, text })`,
  服务缺失则跳过(记 log)。section 返回 disposer,必须挂到 `ctx.effect`。

## 1. engine/inject.mjs —— 自写

```js
export function createInjector(deps)
// deps: {
//   store,               // openStore 实例(用其 recall)
//   embedder,            // createEmbedder 实例(构造查询向量)
//   config: {            // 来自插件 config.injection,全部可选,括号为默认
//     enabled?=true, k?=4, truncate?=0.45, maxChars?=2000, timeoutMs?=1500,
//     userWeight?=0.7, assistantWeight?=0.3,
//   },
//   log, fault: () => (string|null),   // 拒绝服务态读取器;返回非 null 时全部跳过
// }
// 返回 { handler, renderMemoryBlock, buildSeedText } // handler 即 pre-step 监听器
```

### 1.1 handler 流程(严格按序)

1. `const decision = await next()`;
2. 任一命中即原样返回 decision(不做任何副作用):
   `!config.enabled` / `fault()` 非 null / `decision.kind === 'reject'` / `step !== 1` /
   `messages` 为空;
3. **构造查询**:
   - 最后一条真实 user 消息:`messages` 中最后一个 `source.kind === 'user'` 的消息
     (**排除** plugin/tool source——尤其要排除我们自己上轮注入的 `form:'recall'` 消息,
     防反馈回路);取其全部 text block 拼接为 userText;
   - 最后一条 assistant 消息:遍历 `agent.session.events`(若可访问)反向找
     `type === 'assistant/message'` 的事件,取其文本;取不到则权重全给 user;
   - 无 userText → 原样返回;
   - 向量:`embed([userText, assistantText?])` →
     `q = normalize(userWeight * vUser + assistantWeight * vAssistant)`(缺 assistant 时只用 vUser);
4. **节流**(每 agent 一份):seedKey = userText + '\n' + assistantText;
   与该 agent 上次注入的 seedKey 相同 → 原样返回(同一话题的连续轮不重复注入);
5. `store.recall({ vector: q, k, truncate })`(见 §2 store 扩展)与
   `AbortSignal.timeout`/Promise.race 限时 `timeoutMs`;超时/抛错/0 blocks →
   记 seedKey(避免每轮重试同一失败)、原样返回;
6. 渲染 `<memory>` 区块(§1.2),构造消息,按 §0 拼入 decision.messages 返回;
7. 全流程 try/catch:任何异常记 log 并原样返回 decision——**注入永远不得阻塞或弄垮对话轮**。

### 1.2 renderMemoryBlock(blocks, maxChars)

```
<memory>
以下是系统按当前对话自动唤起的跨会话记忆(历史日记片段,供参考而非绝对真理):

[1] (score 0.83 · diaries/dsh/2026-08-29-14_11_45-澜沧计划部署约定.md)
<chunk 正文>

[2] ...
</memory>
```

- 按 score 降序逐个加入,超出 maxChars 时优先丢弃最低分块(至少保留最高分块的截断版);
- 正文原样,不做改写。

## 2. store 扩展(小改)

`store.recall` 支持 `vector` 入参:`recall({ query, vector, k, truncate })`——
`vector`(Float32Array/number[])存在时跳过 embed,直接归一化后走原 KNN;
与 `query` 至少给一个。诊断字段(matchedTags/epa/pyramid)仍照常计算(用同一向量)。

## 3. 入口接线(vcp-memo.mjs)

- `createInjector` 在 store 就绪后创建;`ctx.on('agent/pre-step', injector.handler)`(用 `ctx.on`
  注册即随 Fiber 生命周期撤销);
- 注册 `vcp-memo:discipline` prompt section(order 400),文本:

```markdown
## 记忆使用规范
1. <memory> 区块(若存在)是系统按当前对话自动唤起的跨会话记忆,先读它再回答;它是历史记录而非绝对真理,与当前事实冲突时以当前事实为准,必要时用 save_memory 写入修正。
2. 结束回合前,若有值得长期记住的经历/结论/决定/偏好,调用 save_memory(title, content, tags[]) 写入;tags 保序、优先复用已召回记忆中的既有 Tag、不堆砌同义词。
3. 需要主动回忆更多时,调用 recall_memory(query, k?, truncate?)。
```

- config 新增 `injection` 节(默认值见 §1);`cordis.patch.yml` 与 profile 热重载行的示例同步更新。

## 4. tests/inject.test.mjs 验收

无框架、node 直接可跑。fake store(记录 recall 调用参数、可编程返回 blocks/抛错/挂起)、
fake embedder(返回确定性向量)、fake agent(session.events 数组可控)、
真实测试 `next` 返回 `{ kind:'enter', messages:[...] }`:

1. 消息工厂形状:id 为 UUID、role user、content[0].type 'text'、source `{kind:'plugin', plugin:'vcp-memo', form:'recall'}`;冻结(Object.isFrozen);
2. step=1 + 有 user 消息 + recall 返回 2 blocks → decision.messages 增加 1 条且位置在 claimed 批之后、内容含 `<memory>` 与最高分块正文;recall 收到的 vector 是 0.7/0.3 加权的归一化结果(用已知 fake 向量手算对照);
3. step=2 → 原样返回(同一数组引用,未 splice);
4. 节流:连续两 turn 相同 userText → 第二次不注入;不同文本 → 注入;
5. 种子过滤:messages 里混入上轮注入的 plugin/recall 消息与 tool 消息时,seed 只取真实 user 消息(用 recall 收到的向量反推验证);
6. recall 抛错 / 超时(挂起 + timeoutMs=50)/ 0 blocks → 原样返回且不抛异常;
7. maxChars=200:低分块被丢弃,输出长度 ≤ 预算;
8. reject decision / 空 messages / enabled=false / fault 非 null → 原样返回;
9. 入口集成(fake ctx 同 entry.test 模式):apply 后 pre-step 监听器已注册、systemPrompt 假服务收到 name 'vcp-memo:discipline' order 400 的 section;
10. **回归**:P0/P1 全部测试保持绿色;`node --check` 全部通过。

## 5. 明确不做

- 不做"最近 N 天日记全文附带"(廉价 OneRing,后续里程碑);
- 不做 per-agent/per-preset 的注入过滤(后续需要时加 config);
- 不修改 P1 诊断字段语义;排序仍由 recall 内部决定。
