import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from novel_kb.analyzers.embedding_builder import EmbeddingBuilder
from novel_kb.config.config_schema import KnowledgeBaseConfig
from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.knowledge_base.vector_store import create_vector_store
from novel_kb.llm.factory import LLMFactory
from novel_kb.llm.provider import LLMProvider
from novel_kb.services.export_service import ExportService
from novel_kb.services.ingest_service import AnalysisOptions, IngestService
from novel_kb.services.search_service import SearchService
from novel_kb.utils.text import truncate_text


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable


class ToolHandler:
    def __init__(self, config: KnowledgeBaseConfig) -> None:
        self.config = config
        self.repository = NovelRepository(config.storage.data_dir)
        self.provider = self._build_provider()

        # 创建向量存储（用于语义搜索）
        vector_store = None
        if config.storage.embedding_enabled:
            vector_store = create_vector_store(
                store_type=config.storage.vector_db_type,
                persist_directory=config.storage.data_dir / "vector_store",
            )
            vector_store.initialize()

        self.ingest_service = IngestService(config, self.provider, self.repository)
        self.export_service = ExportService(self.repository, EmbeddingBuilder(self.provider))
        self.search_service = SearchService(
            self.repository,
            config.storage,
            EmbeddingBuilder(self.provider),
            vector_store=vector_store,
            llm_provider=self.provider,
        )
        self.tools: List[ToolSpec] = [
            ToolSpec("ingest_novel_file", "Ingest a local novel file", self.ingest_novel_file),
            ToolSpec("resume_novel_file", "Resume a partial ingest", self.resume_novel_file),
            ToolSpec("list_novels", "List indexed novels", self.list_novels),
            ToolSpec("export_content_chunks", "Export chunks for novel_mcp", self.export_content_chunks),
            ToolSpec("search_novel", "Search novel content (overview level)", self.search_novel),
            ToolSpec("search_chapters", "Search chapter summaries. Best for: plot progression, character arcs, event summaries. NOT suitable for: specific facts, named entities, exact phrases.", self.search_chapters),
            ToolSpec("search_paragraphs", "Search original text paragraphs. Best for: specific facts, named entities (constitutions, weapons, techniques, locations), exact phrases, dialogues. Supports chapter_range filter [start, end] to limit search scope. USE THIS when query contains specific terms or names.", self.search_paragraphs),
            ToolSpec("search_plot_summaries", "Search volume/arc-level plot summaries. Best for: major plot arcs, character development arcs, world-building summaries.", self.search_plot_summaries),
            ToolSpec("search_chapters_by_range", "Search chapters by index range", self.search_chapters_by_range),
            ToolSpec("get_novel_hierarchy", "Get novel hierarchy structure. WITHOUT parameters: only returns high-level summaries (overall_summary + plot_summaries, NO chapter details). WITH chapter_start+chapter_end: returns specific chapter summaries. This is a DATA EXPLORATION tool, NOT a QA system - you must explore the data yourself to find answers.", self.get_novel_hierarchy),
            ToolSpec("find_chapter_by_title", "Find chapter by title", self.find_chapter_by_title),
            ToolSpec("recommend_novels", "Recommend novels by preference", self.recommend_novels),
            ToolSpec("analyze_novel", "Analyze novel features", self.analyze_novel),
            ToolSpec("extract_characters", "Extract character relationships", self.extract_characters),
            ToolSpec("get_summary", "Summarize novel plot", self.get_summary),
            ToolSpec("answer_question", "Answer a user question with hierarchical retrieval. Automatically decides whether to use chapter summaries or original paragraphs based on query type. For specific facts/named entities, it will search original text.", self.answer_question),
            ToolSpec("hierarchical_search", "Hierarchical search: overview→plot summaries→chapter summaries→original text. Automatically escalates to deeper levels if shallow search is insufficient. Best for complex questions needing comprehensive answers.", self.hierarchical_search),
            ToolSpec("query_rewrite_search", "Intelligent search that rewrites user query into multiple targeted sub-queries. Best for: (1) questions about specific entities (constitutions, weapons, techniques, characters), (2) comparative questions, (3) listing questions. Automatically searches both chapter summaries AND original text, then aggregates results.", self.query_rewrite_search),
            ToolSpec("comprehensive_answer", "COMPREHENSIVE question answering. Searches ALL levels (overview→plot→chapters→paragraphs), aggregates all results, and generates a complete answer in ONE call. Best for: listing questions (e.g., '有哪些体质'), comparative questions, analysis questions. Use this INSTEAD of multiple tool calls.", self.comprehensive_answer),
            ToolSpec("health_check", "Check knowledge base status", self.health_check),
        ]

    def register(self, server) -> None:
        if hasattr(server, "tool"):
            for tool in self.tools:
                server.tool(name=tool.name, description=tool.description)(tool.func)
            return
        if hasattr(server, "add_tool"):
            for tool in self.tools:
                server.add_tool(tool.name, tool.func, tool.description)
            return
        raise RuntimeError("Unsupported MCP server interface")

    def ingest_novel_file(
        self,
        file_path: str,
        novel_id: Optional[str] = None,
        title: Optional[str] = None,
        overwrite: bool = False,
        segment_min_chars: Optional[int] = None,
        segment_max_chars: Optional[int] = None,
        segment_concurrency: Optional[int] = None,
        segment_qps: Optional[float] = None,
        segment_retries: Optional[int] = None,
        segment_retry_interval: Optional[float] = None,
    ) -> Dict[str, object]:
        analysis_options = self._build_analysis_options(
            segment_min_chars,
            segment_max_chars,
            segment_concurrency,
            segment_qps,
            segment_retries,
            segment_retry_interval,
        )
        record = self.ingest_service.ingest_file(
            file_path,
            novel_id=novel_id,
            title=title,
            overwrite=overwrite,
            analysis_options=analysis_options,
        )
        return {
            "novel_id": record.novel_id,
            "title": record.title,
            "chapters": len(record.chapters),
        }

    def resume_novel_file(
        self,
        file_path: str,
        novel_id: Optional[str] = None,
        title: Optional[str] = None,
        segment_min_chars: Optional[int] = None,
        segment_max_chars: Optional[int] = None,
        segment_concurrency: Optional[int] = None,
        segment_qps: Optional[float] = None,
        segment_retries: Optional[int] = None,
        segment_retry_interval: Optional[float] = None,
    ) -> Dict[str, object]:
        analysis_options = self._build_analysis_options(
            segment_min_chars,
            segment_max_chars,
            segment_concurrency,
            segment_qps,
            segment_retries,
            segment_retry_interval,
        )
        record = self.ingest_service.resume_file(
            file_path,
            novel_id=novel_id,
            title=title,
            analysis_options=analysis_options,
        )
        return {
            "novel_id": record.novel_id,
            "title": record.title,
            "chapters": len(record.chapters),
        }

    def list_novels(self) -> Dict[str, object]:
        return {"novels": self.export_service.export_novel_list()}

    def export_content_chunks(
        self,
        novel_id: str,
        include_chapters: bool = True,
        include_embeddings: bool = False,
    ) -> Dict[str, object]:
        chunks = self.export_service.export_content_chunks(
            novel_id,
            include_chapters=include_chapters,
            include_embeddings=include_embeddings,
        )
        return {"chunks": chunks}

    def search_novel(self, query: str, k: int = 5) -> Dict[str, object]:
        return {"results": self.search_service.search_novels(query, k=k)}

    def search_chapters(self, query: str, k: int = 5, novel_id: Optional[str] = None) -> Dict[str, object]:
        results = self.search_service.search_chapters(query, k=k, novel_id=novel_id)
        return {"results": results}

    def search_paragraphs(self, query: str, k: int = 5, novel_id: Optional[str] = None, chapter_range: Optional[List[int]] = None) -> Dict[str, object]:
        """搜索原文段落

        Args:
            query: 搜索查询
            k: 返回结果数量
            novel_id: 指定小说 ID
            chapter_range: 章节范围 [start_index, end_index]，可选
        """
        chapter_range_tuple = None
        if chapter_range and len(chapter_range) == 2:
            chapter_range_tuple = (chapter_range[0], chapter_range[1])

        results = self.search_service.search_paragraphs(
            query, k=k, novel_id=novel_id, chapter_range=chapter_range_tuple
        )
        return {"results": results}

    def search_plot_summaries(self, query: str, k: int = 5, novel_id: Optional[str] = None) -> Dict[str, object]:
        """搜索大段剧情总结（卷级）"""
        results = self.search_service.search_plot_summaries(query, k=k, novel_id=novel_id)
        return {"results": results}

    def search_chapters_by_range(
        self,
        novel_id: str,
        start_chapter_index: int,
        end_chapter_index: int,
        query: Optional[str] = None,
    ) -> Dict[str, object]:
        """根据章节范围检索章节"""
        results = self.search_service.search_chapters_by_range(
            novel_id, start_chapter_index, end_chapter_index, query
        )
        return {"results": results}

    def get_novel_hierarchy(
        self,
        novel_id: str,
        chapter_start: Optional[int] = None,
        chapter_end: Optional[int] = None,
    ) -> Dict[str, object]:
        """获取小说层级结构。无参数时只返回高层总结（总概述+剧情总结），有参数时返回指定范围章节。

        WARNING: 不带参数调用只会返回高层概述，不会返回所有章节详情！
        需要查看具体章节时，必须指定 chapter_start 和 chapter_end 范围。
        """
        hierarchy = self.search_service.get_novel_hierarchy(
            novel_id, chapter_start=chapter_start, chapter_end=chapter_end
        )
        return {"hierarchy": hierarchy}

    def find_chapter_by_title(
        self,
        novel_id: str,
        chapter_title: str,
    ) -> Dict[str, object]:
        """根据章节标题查找章节"""
        chapter = self.search_service.find_chapter_by_title(novel_id, chapter_title)
        return {"chapter": chapter}

    def recommend_novels(self, preference_text: str, k: int = 5) -> Dict[str, object]:
        return {"results": self.search_service.search_novels(preference_text, k=k)}

    def analyze_novel(self, text: str) -> Dict[str, object]:
        if (
            not self.provider
            or not self.config.storage.analysis_enabled
            or not self.config.storage.features_enabled
        ):
            return {"features": []}
        result = self.provider.extract_features(
            truncate_text(text, max_chars=self.config.storage.analysis_max_chars)
        )
        return {"features": result.content}

    def extract_characters(self, text: str) -> Dict[str, object]:
        if not self.provider or not self.config.storage.analysis_enabled:
            return {"characters": []}
        result = self.provider.analyze_characters(
            truncate_text(text, max_chars=self.config.storage.analysis_max_chars)
        )
        return {"characters": result.content}

    def get_summary(self, novel_id: Optional[str] = None, text: Optional[str] = None) -> Dict[str, object]:
        if novel_id:
            record = self.repository.load_novel(novel_id)
            return {"summary": record.summary or truncate_text(record.chapters[0].content, max_chars=800)}
        if text and self.provider and self.config.storage.analysis_enabled:
            result = self.provider.analyze_plot(
                truncate_text(text, max_chars=self.config.storage.analysis_max_chars)
            )
            return {"summary": result.content}
        return {"summary": ""}

    def answer_question(self, question: str, k_overview: int = 3, k_chapters: int = 5, k_paragraphs: int = 10) -> Dict[str, object]:
        """分层检索回答：
        1) 在全局概述（novel summaries）中检索并尝试用 LLM 回答；
        2) 若信息不足，检索章节摘要并重试；
        3) 若仍不足，检索段落/原文并重试；
        返回结构包含 answer（可能为空）、source（哪个层级命中）和各层检索结果。
        """
        # 1) 全局概述检索
        overviews = self.search_service.search_novels(question, k=k_overview)
        overview_texts = [str(item.get("summary", "") or "") for item in overviews]
        overview_context = "\n\n".join([t for t in overview_texts if t])

        def _extract_answer_from_result(res):
            if not res:
                return ""
            content = getattr(res, "content", res)
            if isinstance(content, dict):
                return str(content.get("summary") or content.get("answer") or "")
            return str(content)

        # helper to call provider (synchronously via asyncio.run)
        if self.provider and self.config.storage.analysis_enabled:
            try:
                import asyncio

                if overview_context:
                    prompt = f"Use the following OVERVIEW context to answer concisely. If the context is insufficient to answer, reply only with the single token 'INSUFFICIENT'.\n\nContext:\n{overview_context}\n\nQuestion: {question}"
                    result = asyncio.run(self.provider.analyze_plot(truncate_text(prompt, max_chars=self.config.storage.analysis_max_chars)))
                    answer = _extract_answer_from_result(result)
                    if answer and answer.strip() and "INSUFFICIENT" not in answer.upper():
                        return {"answer": answer, "source": "overview", "overviews": overviews}
            except Exception:
                pass

        # 2) 章节摘要检索（SearchService 已会先尝试章节摘要）
        chapter_hits = self.search_service.search_chapters(question, k=k_chapters)
        chapter_texts = [str(item.get("snippet", "") or item.get("summary", "") or "") for item in chapter_hits]
        chapter_context = "\n\n".join([t for t in chapter_texts if t])

        if self.provider and self.config.storage.analysis_enabled:
            try:
                import asyncio

                if chapter_context:
                    prompt = f"Use the following CHAPTER SUMMARY context to answer concisely. If the context is insufficient, reply only with 'INSUFFICIENT'.\n\nContext:\n{chapter_context}\n\nQuestion: {question}"
                    result = asyncio.run(self.provider.analyze_plot(truncate_text(prompt, max_chars=self.config.storage.analysis_max_chars)))
                    answer = _extract_answer_from_result(result)
                    if answer and answer.strip() and "INSUFFICIENT" not in answer.upper():
                        return {"answer": answer, "source": "chapter_summaries", "chapters": chapter_hits}
            except Exception:
                pass

        # 3) 段落/原文检索
        paragraphs = self.search_service.search_paragraphs(question, k=k_paragraphs)
        para_texts = [str(item.get("snippet", "") or "") for item in paragraphs]
        para_context = "\n\n".join([t for t in para_texts if t])

        if para_context and self.provider and self.config.storage.analysis_enabled:
            try:
                import asyncio

                prompt = f"Use the following PARAGRAPHS (original text) to answer the question.\n\nContext:\n{para_context}\n\nQuestion: {question}"
                result = asyncio.run(self.provider.analyze_plot(truncate_text(prompt, max_chars=self.config.storage.analysis_max_chars)))
                answer = _extract_answer_from_result(result)
                if answer and answer.strip():
                    return {"answer": answer, "source": "paragraphs", "paragraphs": paragraphs}
            except Exception:
                pass

        # 最后回退：返回检索到的上下文供客户端进一步处理
        return {"answer": "", "source": "not_found", "overviews": overviews, "chapters": chapter_hits, "paragraphs": paragraphs}

    def hierarchical_search(
        self,
        question: str,
        novel_id: Optional[str] = None,
        max_overviews: int = 3,
        max_plot_summaries: int = 5,
        max_chapters: int = 10,
        max_paragraphs: int = 15,
        confidence_threshold: float = 0.7,
    ) -> Dict[str, object]:
        """
        分层搜索：模拟人类阅读模式
        1. 先看总概述（overview）
        2. 再找大段剧情总结（plot_summaries）
        3. 确定章节范围，遍历章节总结
        4. 必要时跳转到原文
        """
        import asyncio

        def _extract_answer_from_result(res):
            if not res:
                return ""
            content = getattr(res, "content", res)
            if isinstance(content, dict):
                return str(content.get("summary") or content.get("answer") or "")
            return str(content)

        def _is_confident_answer(answer: str) -> bool:
            """判断答案是否可信"""
            if not answer or not answer.strip():
                return False
            
            # 排除不确定的回答
            uncertain_indicators = [
                "我不知道", "无法回答", "不确定", "不清楚", "没有找到",
                "I don't know", "cannot answer", "not sure", "not found",
                "INSUFFICIENT"
            ]
            
            answer_lower = answer.lower()
            for indicator in uncertain_indicators:
                if indicator.lower() in answer_lower:
                    return False
            
            return len(answer.strip()) > 10

        search_steps = []
        
        # 步骤1: 查看总概述
        step1_overviews = self.search_service.search_novels(question, k=max_overviews)
        overview_texts = [str(item.get("summary", "") or "") for item in step1_overviews]
        overview_context = "\n\n".join([t for t in overview_texts if t])
        
        if overview_context and self.provider and self.config.storage.analysis_enabled:
            try:
                prompt = f"""基于以下小说总概述，简要回答这个问题。如果概述中的信息不足以回答，请直接回复"信息不足"。

问题：{question}

概述：
{overview_context}

回答："""
                
                result = asyncio.run(self.provider.analyze_plot(
                    truncate_text(prompt, max_chars=self.config.storage.analysis_max_chars)
                ))
                answer = _extract_answer_from_result(result)
                
                if _is_confident_answer(answer):
                    search_steps.append({
                        "step": "overview",
                        "answer": answer,
                        "context_size": len(overview_context),
                        "results_count": len(step1_overviews)
                    })
                    return {
                        "answer": answer,
                        "source": "overview",
                        "search_steps": search_steps,
                        "confidence": "high",
                        "overviews": step1_overviews
                    }
            except Exception:
                pass
        
        search_steps.append({
            "step": "overview",
            "answer": "信息不足",
            "context_size": len(overview_context),
            "results_count": len(step1_overviews)
        })
        
        # 步骤2: 查找大段剧情总结（卷级）
        step2_plots = self.search_service.search_plot_summaries(question, k=max_plot_summaries, novel_id=novel_id)
        plot_texts = [str(item.get("snippet", "") or "") for item in step2_plots]
        plot_context = "\n\n".join([t for t in plot_texts if t])
        
        if plot_context and self.provider and self.config.storage.analysis_enabled:
            try:
                prompt = f"""基于以下大段剧情总结（按卷划分），回答这个问题。如果总结中的信息不足以回答，请直接回复"信息不足"。

问题：{question}

剧情总结：
{plot_context}

回答："""
                
                result = asyncio.run(self.provider.analyze_plot(
                    truncate_text(prompt, max_chars=self.config.storage.analysis_max_chars)
                ))
                answer = _extract_answer_from_result(result)
                
                if _is_confident_answer(answer):
                    search_steps.append({
                        "step": "plot_summaries",
                        "answer": answer,
                        "context_size": len(plot_context),
                        "results_count": len(step2_plots)
                    })
                    return {
                        "answer": answer,
                        "source": "plot_summaries",
                        "search_steps": search_steps,
                        "confidence": "medium",
                        "plot_summaries": step2_plots
                    }
            except Exception:
                pass
        
        search_steps.append({
            "step": "plot_summaries",
            "answer": "信息不足",
            "context_size": len(plot_context),
            "results_count": len(step2_plots)
        })
        
        # 步骤3: 确定章节范围，遍历相关章节
        if novel_id and self.repository.exists(novel_id):
            record = self.repository.load_novel(novel_id)
            
            # 根据剧情总结确定可能的章节范围
            relevant_chapter_indices = set()
            for plot in step2_plots:
                if novel_id and str(plot.get("novel_id")) == novel_id:
                    # 尝试根据章节标题确定范围
                    start_chapter = str(plot.get("start_chapter", "") or "")
                    end_chapter = str(plot.get("end_chapter", "") or "")
                    
                    # 查找章节索引
                    start_idx = -1
                    end_idx = -1
                    
                    for idx, chapter in enumerate(record.chapters):
                        if start_chapter and start_chapter in chapter.title:
                            start_idx = idx
                        if end_chapter and end_chapter in chapter.title:
                            end_idx = idx
                    
                    if start_idx >= 0 and end_idx >= 0:
                        for idx in range(start_idx, end_idx + 1):
                            relevant_chapter_indices.add(idx)
            
            # 如果没有找到范围，搜索相关章节
            if not relevant_chapter_indices:
                step3_chapters = self.search_service.search_chapters(question, k=max_chapters, novel_id=novel_id)
                for chapter in step3_chapters:
                    chapter_id = str(chapter.get("chapter_id", ""))
                    for idx, ch in enumerate(record.chapters):
                        if ch.chapter_id == chapter_id:
                            relevant_chapter_indices.add(idx)
                            break
            
            # 遍历相关章节
            if relevant_chapter_indices:
                chapter_contexts = []
                for idx in sorted(relevant_chapter_indices):
                    if idx < len(record.chapters):
                        chapter = record.chapters[idx]
                        summary = self.search_service._get_chapter_summary(record, chapter.chapter_id)
                        if summary:
                            chapter_contexts.append(f"第{idx+1}章 {chapter.title}: {summary}")
                
                if chapter_contexts:
                    chapter_context = "\n\n".join(chapter_contexts)
                    
                    if self.provider and self.config.storage.analysis_enabled:
                        try:
                            prompt = f"""基于以下相关章节的总结，回答这个问题。

问题：{question}

章节总结：
{chapter_context}

回答："""
                            
                            result = asyncio.run(self.provider.analyze_plot(
                                truncate_text(prompt, max_chars=self.config.storage.analysis_max_chars)
                            ))
                            answer = _extract_answer_from_result(result)
                            
                            if _is_confident_answer(answer):
                                search_steps.append({
                                    "step": "chapter_summaries",
                                    "answer": answer,
                                    "context_size": len(chapter_context),
                                    "chapters_count": len(relevant_chapter_indices)
                                })
                                return {
                                    "answer": answer,
                                    "source": "chapter_summaries",
                                    "search_steps": search_steps,
                                    "confidence": "medium",
                                    "relevant_chapters": list(sorted(relevant_chapter_indices))
                                }
                        except Exception:
                            pass
        
        # 步骤4: 最后回退到原文检索
        step4_paragraphs = self.search_service.search_paragraphs(question, k=max_paragraphs, novel_id=novel_id)
        para_texts = [str(item.get("snippet", "") or "") for item in step4_paragraphs]
        para_context = "\n\n".join([t for t in para_texts if t])
        
        if para_context and self.provider and self.config.storage.analysis_enabled:
            try:
                prompt = f"""基于以下原文段落，回答这个问题。

问题：{question}

原文段落：
{para_context}

回答："""
                
                result = asyncio.run(self.provider.analyze_plot(
                    truncate_text(prompt, max_chars=self.config.storage.analysis_max_chars)
                ))
                answer = _extract_answer_from_result(result)
                
                if answer and answer.strip():
                    search_steps.append({
                        "step": "paragraphs",
                        "answer": answer,
                        "context_size": len(para_context),
                        "results_count": len(step4_paragraphs)
                    })
                    return {
                        "answer": answer,
                        "source": "paragraphs",
                        "search_steps": search_steps,
                        "confidence": "low",
                        "paragraphs": step4_paragraphs
                    }
            except Exception:
                pass
        
        # 最终回退
        search_steps.append({
            "step": "paragraphs",
            "answer": "未找到足够信息",
            "context_size": len(para_context),
            "results_count": len(step4_paragraphs)
        })

        return {
            "answer": "",
            "source": "not_found",
            "search_steps": search_steps,
            "confidence": "very_low",
            "overviews": step1_overviews,
            "plot_summaries": step2_plots,
            "paragraphs": step4_paragraphs
        }

    def query_rewrite_search(self, query: str, novel_id: Optional[str] = None, k_per_query: int = 5) -> Dict[str, object]:
        """
        智能查询改写搜索：
        1. 分析用户问题类型
        2. 生成多个针对性子查询
        3. 同时搜索章节总结和原文
        4. 汇总去重结果
        """
        import asyncio
        import re

        # Query 类型识别和改写 prompt
        rewrite_prompt = f"""分析以下用户问题，生成3-5个针对性搜索子查询。

要求：
1. 每个子查询应该是独立的、针对性的
2. 包含原始问题的核心概念
3. 适当扩展同义词和相关术语
4. 对于"列举类"问题，生成多个角度的查询

输出格式（只输出JSON数组，不要其他内容）：
["子查询1", "子查询2", "子查询3"]

用户问题：{query}

JSON数组："""

        # 生成子查询
        sub_queries = [query]  # 默认包含原始查询
        if self.provider and self.config.storage.analysis_enabled:
            try:
                result = asyncio.run(self.provider.analyze_plot(
                    truncate_text(rewrite_prompt, max_chars=2000)
                ))
                content = getattr(result, "content", result)
                if isinstance(content, dict):
                    text = str(content.get("content", "") or "")
                else:
                    text = str(content)

                # 尝试从结果中提取 JSON 数组
                try:
                    import json
                    # 找 JSON 数组
                    match = re.search(r'\[.*\]', text, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group())
                        if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                            sub_queries = parsed[:5]  # 最多5个子查询
                except Exception:
                    pass
            except Exception:
                pass

        # 执行所有子查询（章节总结 + 原文）
        all_chapter_results = []
        all_paragraph_results = []
        seen_paragraph_ids = set()

        for sq in sub_queries:
            # 搜索章节总结
            chapters = self.search_service.search_chapters(sq, k=k_per_query, novel_id=novel_id)
            all_chapter_results.extend(chapters)

            # 搜索原文段落
            paragraphs = self.search_service.search_paragraphs(sq, k=k_per_query, novel_id=novel_id)
            for p in paragraphs:
                pid = p.get("chunk_id") or p.get("id") or str(p.get("text", ""))[:100]
                if pid not in seen_paragraph_ids:
                    seen_paragraph_ids.add(pid)
                    all_paragraph_results.append(p)

        # 按相关性排序（简单策略：原文优先，章节总结作为补充）
        # 如果原文结果足够多，只返回原文
        if len(all_paragraph_results) >= k_per_query:
            final_paragraphs = all_paragraph_results[:k_per_query * 2]  # 多返回一些
            paragraph_context = "\n\n".join([
                str(p.get("snippet", "") or p.get("text", "") or "")[:500]
                for p in final_paragraphs
            ])
        else:
            final_paragraphs = all_paragraph_results
            paragraph_context = "\n\n".join([
                str(p.get("snippet", "") or p.get("text", "") or "")[:500]
                for p in final_paragraphs
            ])

        chapter_context = "\n\n".join([
            str(c.get("snippet", "") or c.get("summary", "") or "")[:300]
            for c in all_chapter_results[:k_per_query]
        ])

        # 生成最终答案
        answer = ""
        answer_source = "none"

        if paragraph_context and self.provider and self.config.storage.analysis_enabled:
            try:
                prompt = f"""基于以下搜索结果，回答用户问题。优先使用原文段落（paragraphs）中的具体信息。
如果原文信息不足，再参考章节总结（chapter summaries）。

用户问题：{query}

原始查询：{sub_queries}

原文段落：
{paragraph_context}

章节总结：
{chapter_context}

请给出全面、准确的回答。如果搜索结果不足以回答，请明确说明。"""

                result = asyncio.run(self.provider.analyze_plot(
                    truncate_text(prompt, max_chars=self.config.storage.analysis_max_chars)
                ))
                content = getattr(result, "content", result)
                if isinstance(content, dict):
                    answer = str(content.get("content", "") or content.get("summary", "") or "")
                else:
                    answer = str(content)
                if answer.strip():
                    answer_source = "paragraphs_with_chapters"
            except Exception:
                pass

        # 如果没有生成答案，但有搜索结果，返回汇总信息
        if not answer and (all_paragraph_results or all_chapter_results):
            combined = []
            if final_paragraphs:
                combined.append(f"=== 原文段落 ({len(final_paragraphs)} 条) ===")
                for i, p in enumerate(final_paragraphs[:10], 1):
                    text = str(p.get("snippet", "") or p.get("text", "") or "")[:300]
                    combined.append(f"{i}. {text}...")
            if all_chapter_results:
                combined.append(f"\n=== 章节总结 ({len(all_chapter_results)} 条) ===")
                for i, c in enumerate(all_chapter_results[:5], 1):
                    text = str(c.get("snippet", "") or c.get("summary", "") or "")[:200]
                    combined.append(f"{i}. {text}...")
            answer = "\n".join(combined)
            answer_source = "raw_results"

        return {
            "query": query,
            "sub_queries": sub_queries,
            "answer": answer,
            "answer_source": answer_source,
            "paragraphs": final_paragraphs[:15],  # 限制数量
            "chapters": all_chapter_results[:10],
            "total_paragraphs": len(all_paragraph_results),
            "total_chapters": len(all_chapter_results),
        }

    def comprehensive_answer(
        self,
        question: str,
        novel_id: Optional[str] = None,
        k: int = 10,
    ) -> Dict[str, object]:
        """综合问答：一次调用完成所有层级的搜索和汇总

        核心策略：
        1. 对于列举类问题：先关键词预召回，再用 LLM 语义过滤"强大/重要"的
        2. 对于所有问题：并行搜索所有层级，汇总上下文，一次生成答案
        3. 返回预估结果数量，让调用方知道是否需要补充

        适用于：
        - 列举类问题（"有哪些体质"）
        - 对比类问题（"两种体质的区别"）
        - 分析类问题（"为什么 XXX"）
        """
        import asyncio

        question_type = self._classify_question(question)

        # 1. 关键词预召回（针对列举类问题）
        all_paragraphs: List[Dict] = []
        if question_type == "listing" and self.provider and self.config.storage.analysis_enabled:
            all_paragraphs = self._keyword_recall_question(question, novel_id)

        # 2. 并行搜索所有层级
        overviews = self.search_service.search_novels(question, k=k)
        plot_summaries = self.search_service.search_plot_summaries(question, k=k, novel_id=novel_id)
        chapters = self.search_service.search_chapters(question, k=k, novel_id=novel_id)
        paragraphs = self.search_service.search_paragraphs(question, k=k, novel_id=novel_id)

        # 如果有关键词召回结果，合并到 paragraphs
        if all_paragraphs:
            existing_keys = set(
                f"{p.get('chapter_id', '')}:{p.get('paragraph_index', '')}"
                for p in paragraphs
            )
            for p in all_paragraphs:
                key = f"{p.get('chapter_id', '')}:{p.get('paragraph_index', '')}"
                if key not in existing_keys:
                    paragraphs.append(p)

        # 3. 构建分层的上下文（从详细到概括）
        context_parts = []
        total_results = len(paragraphs) + len(chapters) + len(plot_summaries) + len(overviews)

        # 原文段落（最详细）
        if paragraphs:
            para_texts = [f"[段落{i+1}] {p.get('snippet', '') or p.get('text', '')}"
                         for i, p in enumerate(paragraphs[:15])
                         if p.get('snippet') or p.get('text')]
            if para_texts:
                context_parts.append("=== 原文段落（最详细信息）===\n" + "\n\n".join(para_texts))

        # 章节总结
        if chapters:
            chapter_texts = [f"[章节{i+1}] {c.get('snippet', '') or c.get('summary', '')}"
                            for i, c in enumerate(chapters[:8])
                            if c.get('snippet') or c.get('summary')]
            if chapter_texts:
                context_parts.append("=== 章节总结 ===\n" + "\n\n".join(chapter_texts))

        # 剧情总结
        if plot_summaries:
            plot_texts = [f"[剧情{i+1}] {p.get('snippet', '') or p.get('summary', '')}"
                         for i, p in enumerate(plot_summaries[:5])
                         if p.get('snippet') or p.get('summary')]
            if plot_texts:
                context_parts.append("=== 剧情总结 ===\n" + "\n\n".join(plot_texts))

        # 全书概述
        if overviews:
            overview_texts = [f"[概述{i+1}] {o.get('summary', '') or o.get('content', '')}"
                             for i, o in enumerate(overviews[:3])
                             if o.get('summary') or o.get('content')]
            if overview_texts:
                context_parts.append("=== 全书概述 ===\n" + "\n\n".join(overview_texts))

        full_context = "\n\n".join(context_parts)

        # 4. 生成答案
        answer_text = ""
        if full_context and self.provider and self.config.storage.analysis_enabled:
            if question_type == "listing":
                prompt = f"""基于以下小说内容，回答用户的列举类问题。

要求：
1. 列出所有相关信息，给出具体名称和描述
2. 如果有多种/多个，按重要程度或出现顺序排列
3. 引用原文片段作为证据
4. 如果信息不足以完整回答，说明哪些方面缺少信息

用户问题：{question}

--- 搜索到的内容 ---
{full_context}

请给出完整、详细的回答："""
            elif question_type == "comparative":
                prompt = f"""基于以下小说内容，进行对比分析。

要求：
1. 清晰对比各方的异同点
2. 引用原文作为证据
3. 给出你的分析结论

用户问题：{question}

--- 搜索到的内容 ---
{full_context}

请给出完整、详细的对比分析："""
            elif question_type == "causal":
                prompt = f"""基于以下小说内容，分析因果关系。

要求：
1. 找出原因和结果
2. 引用原文作为证据
3. 如果原因不明，说明可能的解释

用户问题：{question}

--- 搜索到的内容 ---
{full_context}

请给出完整、详细的因果分析："""
            else:
                prompt = f"""基于以下小说内容，回答用户问题。

要求：
1. 给出完整详细的回答
2. 引用原文作为证据
3. 如果信息不足，说明哪些方面缺少信息

用户问题：{question}

--- 搜索到的内容 ---
{full_context}

请给出完整、详细的回答："""

            try:
                result = asyncio.run(self.provider.analyze_plot(
                    truncate_text(prompt, max_chars=self.config.storage.analysis_max_chars)
                ))
                content = getattr(result, "content", result)
                if isinstance(content, dict):
                    answer_text = str(content.get("content", "") or content.get("summary", "") or "")
                else:
                    answer_text = str(content)
            except Exception:
                answer_text = ""

        # 5. 如果没有生成答案且有搜索结果，返回原始结果
        if not answer_text and (paragraphs or chapters or plot_summaries):
            raw_results = []
            if paragraphs:
                raw_results.append(f"=== 原文段落 ({len(paragraphs)} 条) ===")
                for i, p in enumerate(paragraphs[:10], 1):
                    text = str(p.get("snippet", "") or "")[:300]
                    if text:
                        raw_results.append(f"{i}. {text}...")
            if chapters:
                raw_results.append(f"\n=== 章节总结 ({len(chapters)} 条) ===")
                for i, c in enumerate(chapters[:5], 1):
                    text = str(c.get("snippet", "") or c.get("summary", "") or "")[:200]
                    if text:
                        raw_results.append(f"{i}. {text}...")
            answer_text = "\n".join(raw_results) or "未找到相关信息"

        # 6. 返回结果（包含预估数量）
        return {
            "question": question,
            "answer": answer_text,
            "question_type": question_type,
            "estimated_results": {
                "total": total_results,
                "paragraphs": len(paragraphs),
                "chapters": len(chapters),
                "plot_summaries": len(plot_summaries),
                "overviews": len(overviews),
            },
            "enough_for_answer": total_results >= 5,  # 少于5条可能不够
            "sources": {
                "overviews": len(overviews),
                "plot_summaries": len(plot_summaries),
                "chapters": len(chapters),
                "paragraphs": len(paragraphs),
            },
            "results": {
                "overviews": overviews[:3],
                "plot_summaries": plot_summaries[:5],
                "chapters": chapters[:8],
                "paragraphs": paragraphs[:15],
            }
        }

    def _keyword_recall_question(
        self,
        question: str,
        novel_id: Optional[str] = None,
    ) -> List[Dict]:
        """关键词预召回：生成多个相关关键词搜索，提升召回率

        对于"有哪些强力体质"这类问题：
        1. 分析问题，生成多个相关关键词：体质、圣体、禁忌体质等
        2. 用每个关键词搜索原文段落
        3. 合并去重
        """
        import asyncio
        import re

        # 生成关键词
        keywords = self._generate_keywords_for_question(question)

        if not keywords:
            return []

        all_results: List[Dict] = []
        seen_keys: Set[str] = set()

        for kw in keywords[:5]:  # 最多5个关键词
            try:
                # 用关键词搜索原文
                results = self.search_service.search_paragraphs(kw, k=20, novel_id=novel_id)
                for r in results:
                    key = f"{r.get('chapter_id', '')}:{r.get('paragraph_index', '')}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_results.append(r)
            except Exception:
                continue

        return all_results

    def _generate_keywords_for_question(self, question: str) -> List[str]:
        """为问题生成多个相关关键词"""
        if not self.provider or not self.config.storage.analysis_enabled:
            return []

        import asyncio

        prompt = f"""分析以下用户问题，生成多个搜索关键词。

要求：
1. 生成与问题相关的关键词列表
2. 包含核心词的同义词、相关词
3. 最多5个关键词
4. 用逗号分隔

问题：{question}

关键词："""

        try:
            result = asyncio.run(self.provider.analyze_plot(
                truncate_text(prompt, max_chars=500)
            ))
            content = getattr(result, "content", result)
            if isinstance(content, dict):
                text = str(content.get("content", "") or content.get("summary", "") or "")
            else:
                text = str(content)

            # 解析关键词（逗号分隔）
            keywords = [kw.strip() for kw in text.split(",") if kw.strip()]
            return keywords[:5]
        except Exception:
            pass

        # fallback：简单提取问题中的词
        words = re.findall(r'[\u4e00-\u9fff]+', question)
        return words[:5]

    @staticmethod
    def _classify_question(question: str) -> str:
        """判断问题类型"""
        q = question.lower()
        if any(k in q for k in ["有哪些", "有什么", "列举", "列出", "哪些", "多少", "几个"]):
            return "listing"
        if any(k in q for k in ["区别", "不同", "比较", "对比", "差异"]):
            return "comparative"
        if any(k in q for k in ["为什么", "原因", "为什么", "如何导致", "来历", "来源"]):
            return "causal"
        return "general"

    def health_check(self) -> Dict[str, object]:
        provider_ok = False
        try:
            # 异步调用健康检查，但不要阻塞太久
            import asyncio
            try:
                if asyncio.iscoroutinefunction(self.provider.health_check):
                    provider_ok = asyncio.run(self.provider.health_check())
                else:
                    provider_ok = bool(self.provider.health_check())
            except Exception:
                provider_ok = False
        except Exception:
            provider_ok = False
        return {
            "provider": str(self.config.llm.provider),
            "provider_ok": provider_ok,
            "data_dir": str(self.config.storage.data_dir),
            "analysis_enabled": self.config.storage.analysis_enabled,
            "features_enabled": self.config.storage.features_enabled,
            "embedding_enabled": self.config.storage.embedding_enabled,
            "paragraph_enabled": self.config.storage.paragraph_enabled,
        }

    def _build_provider(self) -> LLMProvider:
        """构建 provider（支持 gateway 模式和直接 provider 池）"""
        from novel_kb.llm.noop_provider import NoOpProvider
        if getattr(self.config.llm, "use_gateway", False):
            from novel_kb.gateway_client import GatewayClient
            gateway_url = getattr(self.config.llm, "gateway_url", "http://127.0.0.1:8747")
            gateway_tier = getattr(self.config.llm, "gateway_tier", "medium")
            return GatewayClient(base_url=gateway_url, tier=gateway_tier)

        import asyncio
        from novel_kb.llm.provider_pool import ProviderPool
        try:
            providers_list = self.config.llm.get_providers()
            providers = []

            for provider_type in providers_list:
                models = self.config.llm.get_models_for_provider(provider_type)
                if not models:
                    models = [None]
                for model_name in models:
                    try:
                        provider = LLMFactory.create(
                            self.config.llm,
                            provider_type=provider_type,
                            model_override=model_name,
                        )
                        if asyncio.run(provider.health_check()):
                            providers.append(provider)
                    except Exception:
                        pass

            if not providers:
                return NoOpProvider()

            if len(providers) == 1:
                return providers[0]

            return ProviderPool(providers)
        except Exception:
            pass
        return NoOpProvider()

    def _build_analysis_options(
        self,
        segment_min_chars: Optional[int],
        segment_max_chars: Optional[int],
        segment_concurrency: Optional[int],
        segment_qps: Optional[float],
        segment_retries: Optional[int],
        segment_retry_interval: Optional[float],
    ) -> AnalysisOptions:
        storage = self.config.storage
        min_chars = storage.segment_min_chars if segment_min_chars is None else segment_min_chars
        max_chars = storage.segment_max_chars if segment_max_chars is None else segment_max_chars
        concurrency = storage.segment_concurrency if segment_concurrency is None else segment_concurrency
        qps = storage.segment_qps if segment_qps is None else segment_qps
        retries = storage.segment_retries if segment_retries is None else segment_retries
        retry_interval = (
            storage.segment_retry_interval if segment_retry_interval is None else segment_retry_interval
        )
        return AnalysisOptions(
            segment_enabled=True,
            segment_min_chars=min_chars,
            segment_max_chars=max_chars or None,
            concurrency_limit=concurrency,
            qps_limit=qps,
            retry_limit=retries,
            retry_interval=retry_interval,
        )

