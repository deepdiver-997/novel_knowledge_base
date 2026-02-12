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
3. 分段内容通过 LLM Provider 进行分析与向量化
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
- 列出小说: python -m novel_kb.main list

## 语义检索与推荐
- 开启嵌入: storage.embedding_enabled=true
- 章节嵌入: storage.embed_chapters=true
- MCP 工具: search_novel, search_chapters, search_paragraphs, recommend_novels

## 健康检查
- MCP 工具: health_check

## 与 novel_mcp 的对接
本知识库提供 `export_content_chunks` MCP 工具，可输出与 novel_mcp 的 `ContentChunk` 结构一致的字段，便于直接交给 novel_mcp 的 Global/Local RAG 进行索引。

返回结构要点:
- id, content, content_type, source_id, source_title, metadata, embedding, created_at
- content_type 使用 "novel_summary" 与 "chapter"
