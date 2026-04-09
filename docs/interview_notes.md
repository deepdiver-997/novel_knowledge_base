# Novel Knowledge Base 项目技术文档

## 项目概述

一个基于 MCP (Model Context Protocol) 的小说知识库系统，支持小说解析、向量化存储、智能问答。

---

## 技术栈

### 后端
- **Python 3.10+**
- **MCP Server** - 使用 `mcp` Python SDK 实现 MCP 协议
- **异步框架** - `asyncio` + `aiohttp`

### LLM 集成
- **Gateway 架构** - 轻量级 LLM 网关服务
  - 支持多 Provider 负载均衡（阿里云 Qwen、讯飞 Xunfei、OpenAI 兼容接口）
  - Tier 路由（low/medium/high 三级）
  - 交叉请求重试 + 配额管理
- **向量嵌入** - 本地 embedding 模型 (Ollama nomic-embed-text)

### 数据存储
- **向量数据库** - Chroma (本地持久化)
- **结构化数据** - JSON 文件存储
- **知识图谱** - 简易 JSON 图结构（可扩展为 Neo4j）

### 检索技术
- **混合检索** - 向量相似度 (60%) + BM25 关键词匹配 (40%)
- **中文分词** - 字符级 + 2-gram 组合
- **元数据过滤** - 支持按小说 ID、章节范围精确定过滤
- **768 维向量** - 支持多粒度检索：全书摘要 → 卷级剧情 → 章节总结 → 原文段落

---

## 核心模块

### 1. novel_kb/ - 主知识库模块

| 文件 | 职责 |
|------|------|
| `mcp/server.py` | MCP 协议服务入口 |
| `mcp/handlers/tool_handler.py` | 18 个 MCP 工具实现 |
| `services/ingest_service.py` | 小说解析 + 章节摘要生成 |
| `services/search_service.py` | 分层检索服务 |
| `knowledge_base/vector_store.py` | Chroma 向量存储封装 |

### 2. gateway/ - LLM 网关

| 文件 | 职责 |
|------|------|
| `main.py` | aiohttp HTTP 服务 |
| `routes.py` | `/v1/analyze`, `/v1/embed` 路由 |
| `tier_router.py` | 两级轮询负载均衡 + 失败冷却 |

### 3. 解析与分割

| 模块 | 职责 |
|------|------|
| `segmenters/chapter_segmenter.py` | 章节分割 |
| `parsers/epub_parser.py` | EPUB 解析 |
| `analyzers/embedding_builder.py` | 向量构建 |

---

## 设计亮点

### 1. Gateway 模式
```
Cherry Studio → MCP → novel_kb → Gateway (8747) → 各 LLM Provider
```
优点：解耦 + 统一监控 + 故障转移

### 2. 分层检索架构
```
用户问题
    ↓
┌─────────────────────────────────────┐
│ Level 1: 全书概述 (summary)          │ ← 快速回答
│ Level 2: 卷级剧情 (plot_summaries)  │
│ Level 3: 章节总结 (chapter_summaries)│
│ Level 4: 原文段落 (paragraphs)      │ ← 精确细节
└─────────────────────────────────────┘
```
智能决定用哪层，平衡速度与精度

### 3. Query 改写机制
将用户模糊问题分解为多个针对性子查询，例如：
- "强力体质有哪些" → ["强力体质", "荒古圣体", "太虚圣体", "特殊体质"]

### 4. 综合问答 (Comprehensive Answer)
一次调用完成所有搜索和汇总，解决多次调用导致上下文爆炸的问题：
```
用户问题 → comprehensive_answer → 并行搜索所有层级 → 汇总 → LLM 生成答案
```
- 并行搜索：概述、剧情总结、章节摘要、原文段落
- 智能分类：列举类/对比类/因果类/通用类
- 单次返回：答案 + 引用来源 + 原始结果

### 5. 断点续传
解析长篇小说时可中断重启，已处理章节自动跳过

### 6. 关键词预召回机制
针对列举类问题（如"有哪些强力体质"），先通过 LLM 生成多个相关关键词，大面积召回原文片段，再用语义过滤筛选：

```
用户问题："有哪些强力体质"
    ↓
LLM 关键词生成：["荒古圣体", "太虚圣体", "先天圣体", "特殊体质", "强大体质"]
    ↓
每个关键词 → BM25/向量搜索 → 合并去重 → 大量上下文片段
    ↓
语义过滤 → CrossEncoder Rerank → 精筛出真正"强大"的体质
```
解决思路：语义相似度高的文档不一定包含枚举关键词，关键词预召回保证召回率

---

## 关键技术实现

### 异步并发控制
```python
# 令牌桶限流
qps_limit=5.0
# 并发上限
concurrency_limit=10
# 重试机制
retry_limit=3
```

### 向量检索流程
1. 小说文本 → 分段 (paragraph_min_chars=100)
2. 每段 → 向量嵌入 (768维)
3. 存储到 Chroma
4. 检索时：query → 向量 → 最近邻 Top-K

### 混合检索实现 (Hybrid Search)
```python
# 混合检索公式
score = 0.6 * norm_vector_score + 0.4 * norm_bm25_score

# 中文分词策略
- 英文: 按空格分词
- 中文: 字符级 + 2-gram 组合
- 示例: "荒古圣体" → ["荒", "古", "圣", "体", "荒古", "古圣", "圣体"]
```

### 关键词预召回 (Keyword Pre-Recall)
针对列举类问题，设计两阶段召回：
```python
# Stage 1: 关键词预召回
keywords = llm.generate_keywords(question)  # LLM 生成多个关键词
for kw in keywords:
    results |= bm25_search(kw)  # 每个关键词搜索，合并去重

# Stage 2: 语义精筛
reranked = cross_encoder.rerank(question, results)  # CrossEncoder 精排
```

核心问题：语义相似度高的文档不一定包含枚举关键词（如"荒古圣体"）

### 元数据过滤 (Metadata Filtering)
```python
# 支持按维度过滤
search_paragraphs(
    query="荒古圣体",
    novel_id="遮天",           # 限定小说
    chapter_range=(100, 200)  # 限定章节范围
)

# 过滤原理
1. 向量检索后按 chapter_index 过滤
2. BM25 检索时跳过不在范围内的章节
3. 减少无效计算，加速检索
```

### 两阶段检索 + Rerank (CrossEncoder)
```
Stage 1 (召回): query → 向量 + BM25 混合召回 Top-20
Stage 2 (精排): query + doc → CrossEncoder → 精确排序

# 与双塔模型的对比
| 模型 | 输入方式 | 特点 |
|------|----------|------|
| 向量模型（双塔）| query 和 doc 分别编码 | 快，适合召回，但信息有损失 |
| 交叉编码器 | query + doc 一起输入 | 慢，但能捕捉精细交互 |

# 示例
query="叶凡的成长"
向量模型：相似度 0.65（信息损失）
交叉编码器：相似度 0.92（精细匹配）
```

### 章节摘要生成
```python
章节文本 → 分割成多个 segment (120-8000 chars)
    ↓
每个 segment → LLM analyze_plot → 摘要
    ↓
合并所有 segment 摘要 → 章节总结
```

---

## 可扩展方向

1. ~~**混合检索**~~ - ✅ 已实现：向量 + BM25 关键词混合
2. ~~**元数据过滤**~~ - ✅ 已实现：按小说 ID、章节范围过滤
3. ~~**Rerank**~~ - ✅ 已实现：CrossEncoder 交叉编码器精排
4. **多跳推理** - 支持关联推理问题
5. **知识图谱** - Neo4j 存储角色关系
6. **流式输出** - SSE 实时返回进度

---

## 快速命令

```bash
# 导入小说
python -m novel_kb.main ingest nover.epub --strict-mode

# 断点续传
python -m novel_kb.main resume novel.epub --strict-mode

# 启动 MCP
./start_mcp_server.sh

# 启动 Gateway
./start_gateway.sh start
```
