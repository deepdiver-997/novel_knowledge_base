# Novel Knowledge Base

## 架构概览
本项目是一个面向小说知识抽取与同人写作辅助的本地知识库系统，提供 MCP stdio 服务以供上层应用调用。系统以配置驱动，支持本地模型与云端模型的统一接口调用，并支持章节级分段、特征标签抽取、人物关系分析、剧情概要生成与向量检索。

## 设计目标
- 清晰的分层结构，职责单一
- 配置驱动，易于切换 LLM 提供商
- MCP stdio 通信，便于本地集成
- 章节优先分段，避免信息丢失
- 可扩展的存储层（向量、关系图、缓存）

## 数据流
1. CLI 启动并加载配置文件
2. 解析器读取小说文件并按章节分段
3. 章节内容通过 LLM Provider 进行分析与向量化
4. 分析结果进入向量库和关系图存储
5. MCP Server 对外提供检索、分析、推荐等工具

## 配置策略
- 默认配置路径: ~/.novel_knowledge_base/config.yaml
- 支持命令行参数指定配置文件
- 若配置不存在自动创建默认配置
- 通过配置文件选择 LLM Provider 并指定参数

## 模块分层
- config: 配置加载与校验
- parsers: EPUB/TXT 等输入解析
- segmenters: 章节分段与分段数据模型
- llm: 统一的 Provider 抽象与工厂
- analyzers: 人物关系、剧情摘要、特征标签、向量构建
- knowledge_base: 向量库与关系图存储
- mcp: MCP stdio 服务与工具注册
- utils: 日志与异常

## 扩展方式
- 新增 LLM Provider: 实现 LLMProvider 接口并在工厂注册
- 新增解析器: 继承 BaseParser 并加入解析器注册表
- 新增工具: 在 mcp/tools 中实现并注册到工具表

## 目录结构
项目结构将在代码中保持与上述模块一致，保证易读性和扩展能力。

## 运行方式
- 命令行启动: python -m novel_kb.main
- 指定配置: python -m novel_kb.main --config /path/to/config.yaml
- 导入小说: python -m novel_kb.main ingest /path/to/book.epub
- 断点续做: python -m novel_kb.main resume /path/to/book.epub
- 列出小说: python -m novel_kb.main list
- 摘要质检: python -m novel_kb.main audit-summaries /path/to/<novel>.json

### EPUB 清洗
- 在导入前先清洗（输出 *_cleaned.epub 并用清洗后的文件继续处理）:
	- python -m novel_kb.main ingest /path/to/book.epub --clean
	- python -m novel_kb.main resume /path/to/book.epub --clean
- 仅清洗，不入库:
	- python -m novel_kb.main ingest /path/to/book.epub --clean-only
	- python -m novel_kb.main resume /path/to/book.epub --clean-only

## 分段分析与限流
分段流程固定开启：章节原文会先分段再汇总成章节总结，随后按卷聚合并生成全书总概述。
并发、QPS 与重试策略由配置文件控制，CLI 可覆盖数值。

人物抽取可通过配置关闭：storage.characters_enabled=false。

示例:
```
python -m novel_kb.main ingest /path/to/book.epub \
	--segment-min-chars 120 \
	--segment-max-chars 0 \
	--segment-concurrency 4 \
	--segment-qps 2 \
	--segment-retries 3 \
	--segment-retry-interval 1
```

## 断点续做
- 续做使用临时文件保存进度: ~/.novel_knowledge_base/data/novels/<novel_id>.progress.json
- 完成后写入正式文件: ~/.novel_knowledge_base/data/novels/<novel_id>.json
- 章节级别进度保存，章节摘要用于最终全书摘要汇总

### 运行模式
- 普通模式（默认）：只做章节总结并持续写入 `.progress.json`，不生成最终全书总结，也不写入正式 `<novel_id>.json`。
- 最严格模式（`--strict-mode`）：
	- 所有章节总结必须完成；
	- 使用指纹库进行无关性检查，命中可疑摘要会自动重生成；
	- 通过检查后再生成卷级剧情总结与全书总结；
	- 最终写入正式 `<novel_id>.json` 并清理 progress。

最严格模式示例：
```
python -m novel_kb.main resume /path/to/book.epub \
	--strict-mode \
	--fingerprints ./wrong_fliter.txt \
	--segment-min-chars 120 \
	--segment-max-chars 0 \
	--segment-concurrency 4 \
	--segment-qps 2 \
	--segment-retries 0 \
	--segment-retry-interval 2 \
	--audit-min-score 0.9 \
	--audit-similarity-threshold 0.72 \
	--audit-min-length 20
```

## 摘要指纹筛检（无 LLM）
- 目标：在最终汇总前快速找出“固定错误回复/模板回复/无关回复”。
- 指纹库文件：默认读取项目根目录 `wrong_fliter.txt`（每行一条指纹，可随时手工增删）。
- 支持检查 `*.json` 与 `*.progress.json`。

示例：
```
# 仅报告可疑项（默认）
python -m novel_kb.main audit-summaries ~/.novel_knowledge_base/data/novels/完美世界.progress.json

# 自动删除所有可疑项（会先生成 .bak 备份）
python -m novel_kb.main audit-summaries ~/.novel_knowledge_base/data/novels/完美世界.progress.json --action delete

# 逐条确认是否删除
python -m novel_kb.main audit-summaries ~/.novel_knowledge_base/data/novels/完美世界.progress.json --action confirm-delete

# 调整阈值
python -m novel_kb.main audit-summaries /path/to/file.json --similarity-threshold 0.76 --min-length 24

# 仅处理高风险项（例如评分 >= 1.5）
python -m novel_kb.main audit-summaries /path/to/file.json --action delete --min-score 1.5
```

## 分层检索与推荐
- 开启嵌入: storage.embedding_enabled=true
- 章节嵌入: storage.embed_chapters=true
- 分层分析: storage.hierarchical_analysis_enabled=true
- MCP 工具:
  - 基础检索: search_novel, search_chapters, search_paragraphs
  - 剧情总结: search_plot_summaries, get_novel_hierarchy
  - 章节管理: search_chapters_by_range, find_chapter_by_title
  - 智能检索: comprehensive_answer（推荐）, query_rewrite_search, hierarchical_search, answer_question
  - 综合问答: comprehensive_answer - 一次完成所有搜索和汇总
  - 推荐: recommend_novels

## 健康检查
- MCP 工具: health_check

## 智能检索工具

### comprehensive_answer（推荐）
**综合问答工具**，一次完成所有层级搜索和汇总。自动并行搜索所有层级（概述→剧情→章节→原文），适合列举类、对比类、分析类问题。

### query_rewrite_search
智能改写工具，将用户问题分解为多个针对性子查询，适合查询具体实体（体质、法宝、人物）。

### hierarchical_search
分层检索工具，模拟人类阅读模式，按照以下流程进行智能检索：

1. **总概述检索** - 先看小说整体概览
2. **剧情总结检索** - 查找大段剧情总结（卷级）
3. **章节范围确定** - 根据剧情总结确定相关章节范围
4. **章节总结遍历** - 遍历范围内的单章节总结
5. **原文回退** - 必要时跳转到原文段落

### 使用示例
```python
# 智能分层检索
result = hierarchical_search(
    question="主角在哪个章节获得了神器？",
    novel_id="zhetian",
    max_overviews=3,
    max_plot_summaries=5,
    max_chapters=10,
    max_paragraphs=15
)

# 返回结构包含：
# - answer: 最终答案
# - source: 信息来源层级
# - search_steps: 详细的搜索步骤
# - confidence: 答案置信度
# - 各层级的搜索结果
```

## RAG 增强功能

### 混合检索 (Hybrid Search)
结合向量相似度与 BM25 关键词匹配，兼顾语义理解与精确关键词检索：
```python
score = 0.6 * vector_similarity + 0.4 * bm25_score
```

### CrossEncoder Rerank
两阶段检索：先用混合检索召回 Top-20，再用 CrossEncoder 交叉编码器精排。

### 关键词预召回
针对列举类问题（如"有哪些X"），先通过 LLM 生成多个相关关键词，大面积召回原文片段，再用语义过滤筛选。

### 综合问答 (comprehensive_answer)
一次调用完成所有层级搜索和汇总，解决多次调用导致上下文爆炸的问题。

## 与 novel_mcp 的对接
本知识库提供 `export_content_chunks` MCP 工具，可输出与 novel_mcp 的 `ContentChunk` 结构一致的字段，便于直接交给 novel_mcp 的 Global/Local RAG 进行索引。

返回结构要点:
- id, content, content_type, source_id, source_title, metadata, embedding, created_at
- content_type 使用 "novel_summary" 与 "chapter"

## 用户认证与访问控制

### 概述
系统支持用户认证和 KB 文件访问控制。Root 用户可访问所有 KB 文件，普通用户只能访问被分配的知识库。

### 数据库
- 位置: `~/.novel_knowledge_base/kb_auth.db`
- 表结构:
  - `users`: 用户表 (user_id, name, api_key, created_at)
  - `kb_ownership`: KB 归属表 (name, novel_id)

### Root 用户配置
在 `~/.novel_knowledge_base/config.yaml` 中配置:

```yaml
root_user:
  name: "admin"
  api_key: "your-secret-key"
```

### CLI 命令

```bash
# 注册新用户 (API key 会用 SHA256 哈希存储)
python -m novel_kb.main register_user <name> <api_key>

# 删除用户及所有 KB 归属记录
python -m novel_kb.main delete_user <name>

# 分配 KB 文件给用户
python -m novel_kb.main assign_kb <name> <novel_id>

# 列出用户可访问的 KB 文件
python -m novel_kb.main list_user_kb <name>
```

### 使用示例

```bash
# 1. 注册用户
python -m novel_kb.main register_user alice my-secret-key

# 2. 分配 KB 文件
python -m novel_kb.main assign_kb alice 遮天
python -m novel_kb.main assign_kb alice 完美世界

# 3. 查看用户可访问的 KB
python -m novel_kb.main list_user_kb alice

# 4. 删除用户
python -m novel_kb.main delete_user alice
```

### MCP stdio 调用
MCP 服务器通过 stdio 协议与 LLM 交互，启动脚本已配置好:

```bash
./start_mcp_server.sh
```

或直接运行:
```bash
python -m novel_kb.main
```

### 访问控制逻辑
- Root 用户 (config.yaml 中配置): 可访问所有 KB 文件
- 普通用户: 只能在 kb_ownership 表中被分配 novel_id 才能访问
