# 高并发章节汇总改造蓝图（段落级调用，章节级提交）

## 1. 背景与目标

当前流程以“章节”为主循环，章节内做分段并发。问题是：
- 外层章节串行，整体吞吐受长章节拖慢。
- 并发参数提升后，实际在飞任务数受单章分段数限制。
- 失败处理与进度恢复粒度偏粗，难以做高效调度。

本次改造目标：
1. **最小调用单元**：段落（或文本分片）。
2. **最小提交单元**：章节总结（入库与进度写入仍按章节）。
3. **调度模型**：滑动窗口 + 状态驱动，允许章节乱序完成。
4. **最终一致性**：最终章节结果按原始章节顺序输出。

---

## 2. 关键约束

- 不改变对外 CLI 行为（`ingest/resume` 参数兼容）。
- 不改变最终数据格式语义（章节列表 + 全书总结）。
- 保持断点续做（progress）可恢复、可幂等。
- 限流/重试仍遵循现有 provider_pool 策略。

---

## 3. 目标架构（概览）

分为四层：

1. **切分层（Splitter）**
   - 将章节切成段落任务（segment task）。

2. **调度层（Window Scheduler）**
   - 按窗口预算（并发/QPS/provider 健康）持续派发任务。
   - 每轮回收完成任务并更新章节状态。

3. **归约层（Reducer）**
   - 同一章节内若有多个段落总结，触发章节二次汇总。
   - 若只有一个结果，直接作为章节总结。

4. **提交层（Committer）**
   - 章节状态 `DONE` 后写入 progress。
   - 全章节 `DONE` 后生成最终全书总结并写入正式记录。

---

## 4. 核心数据结构

### 4.1 ChapterTaskState

```python
@dataclass
class ChapterTaskState:
    chapter_id: str
    chapter_index: int
    title: str

    status: Literal[
        "PENDING",           # 尚未切分
        "SEGMENTS_READY",    # 已切分，待派发
        "RUNNING",           # 有分片在执行
        "REDUCE_READY",      # 分片结果齐全，待章节归约
        "REDUCING",          # 章节归约中
        "DONE",              # 章节总结完成
        "FAILED",            # 章节失败（可继续全局）
    ]

    segment_texts: list[str]
    segment_results: dict[int, str]         # segment_idx -> summary
    segment_errors: dict[int, str]          # segment_idx -> last_error

    chapter_summary: str
    attempts_segment: dict[int, int]
    attempts_reduce: int

    updated_at: float
```

### 4.2 WorkStateMap（仅持久化状态）

```python
class WorkStateMap:
    chapters: dict[str, ChapterTaskState]
    inflight_segment_tasks: int
    inflight_reduce_tasks: int

    # 可选：按 chapter_index 维护顺序视图，输出时使用
    order: list[str]

   # 仅状态字段，不保存 asyncio task/future
   # 目标：可序列化、可恢复、无事件循环耦合
```

### 4.3 RuntimeTaskRegistry（仅内存中的异步句柄）

```python
@dataclass
class RuntimeTaskRegistry:
   # segment 任务：future -> (chapter_id, segment_idx)
   segment_futures: dict[asyncio.Task, tuple[str, int]]

   # reduce 任务：future -> chapter_id
   reduce_futures: dict[asyncio.Task, str]

   # 反向索引（可选）：(chapter_id, segment_idx) -> future
   segment_slots: dict[tuple[str, int], asyncio.Task]

   # 章节归约任务索引（可选）：chapter_id -> future
   reduce_slots: dict[str, asyncio.Task]

   # 运行时统计
   created_at: float
   last_tick_at: float
```

> 关键约束：`Future/Task` 只存在于 `RuntimeTaskRegistry`，不进入 progress 文件。

### 4.4 WindowBudget

```python
@dataclass
class WindowBudget:
    global_inflight_limit: int
    per_provider_inflight_limit: dict[str, int]
    qps_limit: float
```

---

## 5. 调度算法（滑动窗口）

主循环每个 tick 执行：

1. **补充可运行任务**
   - `PENDING` -> 切分 -> `SEGMENTS_READY`
   - `SEGMENTS_READY/RUNNING` 中找尚未成功且可重试的分片任务

2. **按预算派发**
   - 若连接/在飞预算未满，继续派发 segment/reduce 任务
   - 通过 `asyncio.create_task(...)` 创建任务后，立即注册到 `RuntimeTaskRegistry`
   - 同步更新 `WorkStateMap.inflight_*` 计数

3. **回收完成事件**
   - 使用 `asyncio.wait(futures, return_when=FIRST_COMPLETED)` 回收已完成 future
   - 成功：写入 `segment_results`
   - 失败：记录错误、累计重试、必要时进入 `FAILED`
   - 回收后从 `RuntimeTaskRegistry` 删除对应 future 与索引

4. **章节归约判定**
   - 某章节所有分片成功：
     - 结果数=1：`DONE`
     - 结果数>1：`REDUCE_READY`，再发 reduce

5. **提交进度**
   - 对新进入 `DONE` 的章节写 progress
   - 全章节完成后触发全书总结

6. **窗口自适应（可选）**
   - 429 比例升高：缩小窗口
   - 持续稳定：缓慢放大窗口

---

## 6. 状态迁移规则

- `PENDING -> SEGMENTS_READY`：切分完成。
- `SEGMENTS_READY -> RUNNING`：至少一个分片已派发。
- `RUNNING -> REDUCE_READY`：全部分片成功且 >1 条结果。
- `RUNNING -> DONE`：全部分片成功且仅 1 条结果。
- `REDUCE_READY -> REDUCING -> DONE`：章节归约成功。
- 任意状态 -> `FAILED`：超过重试上限或不可恢复异常。

失败章节默认策略：
- 写空总结并标记失败原因，流程继续。

---

## 7. 进度文件设计（兼容增强）

保持现有结构的同时新增可恢复字段：

```json
{
  "analysis_progress": {
    "chapter_index": 123,
    "chapter_summaries": [...],
    "characters": [...],
    "work_map": {
      "chapters": {
        "ch_0001": {
          "status": "RUNNING",
          "segment_results": {"0": "..."},
          "segment_errors": {"2": "429"},
          "attempts_segment": {"2": 1},
          "attempts_reduce": 0
        }
      }
    }
  }
}
```

说明：
- `work_map` 仅保存可序列化状态（章节状态、重试次数、已完成结果等）。
- `RuntimeTaskRegistry` 不落盘；进程重启后根据 `work_map` 重建待执行任务。

恢复逻辑：
- `DONE` 章节直接跳过。
- `RUNNING/REDUCING` 视作中断任务，重入队列继续执行并重新注入 future。

---

## 8. 分阶段改造计划（建议）

### Phase 1：引入状态对象与 work_map（不改变主流程语义）
- 新增状态结构与序列化/反序列化。
- 仍保持章节外层串行，仅替换内部数据组织。

**验收**：结果与现网一致，resume 可恢复。

### Phase 2：章节内任务事件化（段落任务 + reduce）
- 章节内部从同步等待改为事件回收。
- 单章节可真实跑满 `segment_concurrency`。

**验收**：单章节吞吐提升，失败不崩溃。

### Phase 3：全局滑动窗口（跨章节调度）
- 去掉章节串行主循环，改为全局调度器。
- 支持章节乱序完成、顺序输出。

**验收**：多章节吞吐显著提升，CPU/网络利用率提高。

### Phase 4：自适应窗口与监控指标
- 基于 429/超时率动态调节窗口。
- 增加日志指标：inflight、ready、done、429 rate、P95 latency。

**验收**：高压下更稳定，平均耗时下降。

---

## 9. 代码落点（拟改文件）

- `novel_kb/services/ingest_service.py`
  - 新增 `ChapterTaskState` / `WorkMap` / Scheduler 主循环。
  - 改造 `_analyze_hierarchical` 为状态驱动。

- `novel_kb/knowledge_base/repository.py`
  - 保持接口不变，仅确认 progress 读写可承载 `work_map`。

- `novel_kb/llm/provider_pool.py`
  - 保持现有逻辑；可选增加窗口指标日志。

- `tests/`（新增）
  - 状态迁移单元测试
  - 中断恢复测试
  - 乱序完成一致性测试

---

## 10. 兼容性与回滚

- 增加开关：
  - `storage.scheduler_mode: legacy | window`
- 默认 `legacy`，灰度切 `window`。
- 出现异常可一键回滚 legacy，不影响既有数据。

---

## 11. 验收标准（必须满足）

1. 正确性
   - 最终章节数、章节顺序、最终总结与 legacy 逻辑一致（语义层面）。

2. 稳定性
   - 单章节失败不导致整书中断（按策略写空总结继续）。

3. 恢复能力
   - 中断后 resume 可继续，且不重复提交已完成章节。

4. 性能
   - 在同 provider 配额下，总耗时显著优于章节串行版本。

---

## 12. 本周落地建议

- 第 1 天：Phase 1（结构与持久化）
- 第 2~3 天：Phase 2（章节内事件化）
- 第 4~5 天：Phase 3（全局窗口调度）
- 第 6 天：压测 + 回归
- 第 7 天：灰度上线（开关控制）

---

## 13. 备注

这份蓝图的核心思想是：
- **段落级并发执行**解决吞吐；
- **章节级状态提交**保证一致性；
- **窗口化调度**保证资源利用与限流稳定。
