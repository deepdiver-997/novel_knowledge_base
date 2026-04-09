# Novel Knowledge Base MCP 服务器启动指南

## 当前状态
- MCP 服务器已就绪，支持向量搜索
- LLM 调用全部走本地 Gateway 服务
- 向量数据库（Chroma）已配置并导入数据
- 18 个 MCP 工具可用

## 快速启动

### 1. 启动 Gateway 服务
```bash
./start_gateway.sh start      # 启动
./start_gateway.sh status    # 查看状态
./start_gateway.sh stop      # 停止
./start_gateway.sh restart   # 重启
```

### 2. 启动 MCP 服务器
```bash
./start_mcp_server.sh
```

### 3. 在 Cherry Studio 中配置
```json
{
  "mcpServers": {
    "novel-kb": {
      "command": "/Users/zhuhongrui/llm/novel_knowledge_base/start_mcp_server.sh"
    }
  }
}
```

## 已导入的小说

| 小说 | 向量 chunks |
|------|------------|
| 遮天 | 1825 |
| 完美世界 | 2083 |

## 可用的 MCP 工具

### 向量搜索工具
1. **search_plot_summaries** - 搜索剧情总结（卷级）✓
2. **search_chapters** - 搜索章节总结 ✓
3. **search_paragraphs** - 搜索段落内容
4. **search_novel** - 综合搜索

### 基础工具
5. **list_novels** - 列出已导入的小说
6. **health_check** - 检查服务器状态
7. **get_novel_hierarchy** - 获取小说层级结构
8. **find_chapter_by_title** - 按标题查找章节

### 管理工具
9. **ingest_novel_file** - 导入新小说
10. **resume_novel_file** - 恢复部分导入
11. **export_content_chunks** - 导出内容块

### 分析工具
12. **hierarchical_search** - 智能分层搜索
13. **analyze_novel** - 分析小说特征
14. **extract_characters** - 提取人物关系
15. **get_summary** - 获取剧情摘要
16. **answer_question** - 回答问题
17. **search_chapters_by_range** - 按范围搜索章节
18. **recommend_novels** - 推荐小说

## 搜索示例

```json
// 搜索遮天的剧情总结
{
  "name": "search_plot_summaries",
  "arguments": {
    "novel_id": "遮天",
    "query": "修炼境界 突破",
    "k": 3
  }
}

// 搜索遮天的章节
{
  "name": "search_chapters",
  "arguments": {
    "novel_id": "遮天",
    "query": "叶凡 突破",
    "k": 3
  }
}
```

## 架构说明

```
┌─────────────────┐     HTTP      ┌─────────────────┐
│  Cherry Studio  │─────────────▶│   MCP Server   │
│  (或其他 MCP 客户端) │             │  novel_kb      │
└─────────────────┘               └────────┬────────┘
                                           │
                                    LLM 调用│
                                           ▼
                                  ┌─────────────────┐
                                  │  LLM Gateway    │
                                  │  (端口 8747)    │
                                  └────────┬────────┘
                                           │
                                   ┌───────┴───────┐
                                   ▼               ▼
                            ┌──────────┐    ┌──────────────┐
                            │  Ollama  │    │ 向量数据库    │
                            │ (本地模型) │    │   (Chroma)   │
                            │ nomic-   │    │              │
                            │ embed    │    │  768 维向量   │
                            └──────────┘    └──────────────┘
```

## 故障排除

### Gateway 无法启动
```bash
./start_gateway.sh log   # 查看日志
```

### MCP 连接失败
```bash
# 确认 Gateway 正在运行
./start_gateway.sh status

# 确认 MCP 服务器已启动
curl http://127.0.0.1:8747/v1/health
```

### 向量搜索返回空
- 确认小说已完成导入和向量化
- 确认 `embedding_enabled: true` 在配置中

## 配置文件位置

- MCP 配置: `~/.novel_knowledge_base/config.yaml`
- Gateway 配置: `~/.novel_knowledge_base/gateway_config.yaml`
- Gateway 日志: `/tmp/gateway.log`
- 认证数据库: `~/.novel_knowledge_base/kb_auth.db`

## 用户认证与 KB 访问控制

### Root 用户配置
编辑 `~/.novel_knowledge_base/config.yaml`:

```yaml
root_user:
  name: "admin"
  api_key: "your-secret-key"
```

### 用户管理命令

```bash
# 注册新用户
python -m novel_kb.main register_user alice my-secret-key

# 分配 KB 文件给用户
python -m novel_kb.main assign_kb alice 遮天
python -m novel_kb.main assign_kb alice 完美世界

# 查看用户可访问的 KB
python -m novel_kb.main list_user_kb alice

# 删除用户
python -m novel_kb.main delete_user alice
```

### 访问控制说明
- **Root 用户**: 在 config.yaml 中配置，可访问所有 KB 文件
- **普通用户**: 只能在 kb_ownership 表中被分配 novel_id 才能访问
- **API Key**: 使用 SHA256 哈希存储，传输时使用明文，存储时不会暴露
