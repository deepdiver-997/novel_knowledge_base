from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from novel_kb.analyzers.embedding_builder import EmbeddingBuilder
from novel_kb.config.config_schema import StorageConfig
from novel_kb.knowledge_base.repository import NovelRepository
from novel_kb.knowledge_base.vector_store import ContentType, VectorStore
from novel_kb.utils.segment import split_paragraphs
from novel_kb.utils.text import truncate_text

if TYPE_CHECKING:
    from novel_kb.llm.provider import LLMProvider

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class SearchService:
    # 混合检索权重配置
    VECTOR_WEIGHT = 0.6  # 向量相似度权重
    BM25_WEIGHT = 0.4   # BM25 关键词权重
    # Rerank 配置
    RERANK_TOP_K = 20   # 召回阶段取 Top-K 用于 rerank
    RERANK_MODEL = "BAAI/bge-reranker-base"  # 默认 rerank 模型

    def __init__(
        self,
        repository: NovelRepository,
        storage_config: StorageConfig,
        embedding_builder: Optional[EmbeddingBuilder] = None,
        vector_store: Optional[VectorStore] = None,
        llm_provider: Optional["LLMProvider"] = None,
    ) -> None:
        self.repository = repository
        self.storage_config = storage_config
        self.embedding_builder = embedding_builder
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        # BM25 索引缓存: {novel_id: {"paragraphs": [...], "bm25": BM25Okapi, "tokenized": [[...]] }}
        self._bm25_cache: Dict[str, Dict] = {}
        # CrossEncoder 缓存
        self._cross_encoder = None
        self._cross_encoder_model = None

    def search_novels(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        if self._can_vector_search():
            return self._search_novels_vector(query, k)
        if self._can_embed():
            return self._search_novels_semantic(query, k)
        return self._search_novels_keyword(query, k)

    def search_chapters(self, query: str, k: int = 5, novel_id: Optional[str] = None) -> List[Dict[str, object]]:
        summary_hits = self._search_chapter_summaries_keyword(query, k, novel_id)
        if len(summary_hits) >= k:
            return summary_hits[:k]

        exclude_ids = {
            str(item.get("chapter_id"))
            for item in summary_hits
            if isinstance(item.get("chapter_id"), str)
        }

        if self._can_vector_search():
            fallback = self._search_chapters_vector(query, k, novel_id, exclude_ids)
        elif self._can_embed() and self.storage_config.embed_chapters:
            fallback = self._search_chapters_semantic(query, k, novel_id, exclude_ids)
        else:
            fallback = self._search_chapters_keyword(query, k, novel_id, exclude_ids)

        return (summary_hits + fallback)[:k]

    def search_paragraphs(
        self,
        query: str,
        k: int = 5,
        novel_id: Optional[str] = None,
        chapter_range: Optional[tuple[int, int]] = None,
    ) -> List[Dict[str, object]]:
        """搜索原文段落

        Args:
            query: 搜索查询
            k: 返回结果数量
            novel_id: 指定小说 ID（可选）
            chapter_range: 章节范围过滤 (start_index, end_index)，可选
        """
        if not self.storage_config.paragraph_enabled:
            return []
        # 优先使用混合检索（向量 + BM25）
        if self._can_vector_search() and BM25_AVAILABLE:
            return self._search_paragraphs_hybrid(query, k, novel_id, chapter_range)
        if self._can_vector_search():
            return self._search_paragraphs_vector(query, k, novel_id, chapter_range)
        if self._can_embed() and self.storage_config.paragraph_semantic_enabled:
            return self._search_paragraphs_semantic(query, k, novel_id, chapter_range)
        return self._search_paragraphs_keyword(query, k, novel_id, chapter_range)

    def search_plot_summaries(self, query: str, k: int = 5, novel_id: Optional[str] = None) -> List[Dict[str, object]]:
        """检索大段剧情总结（卷级）"""
        if self._can_vector_search():
            return self._search_plot_summaries_vector(query, k, novel_id)
        if self._can_embed():
            return self._search_plot_summaries_semantic(query, k, novel_id)
        return self._search_plot_summaries_keyword(query, k, novel_id)

    def search_chapters_by_range(
        self,
        novel_id: str,
        start_chapter_index: int,
        end_chapter_index: int,
        query: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """根据章节范围检索章节"""
        if not self.repository.exists(novel_id):
            return []

        record = self.repository.load_novel(novel_id)
        results: List[Dict[str, object]] = []

        for index, chapter in enumerate(record.chapters):
            if start_chapter_index <= index <= end_chapter_index:
                results.append({
                    "novel_id": record.novel_id,
                    "title": record.title,
                    "chapter_id": chapter.chapter_id,
                    "chapter_title": chapter.title,
                    "chapter_index": index,
                    "summary": self._get_chapter_summary(record, chapter.chapter_id),
                    "content": truncate_text(chapter.content, max_chars=500),
                })

        if query:
            filtered = [
                r for r in results
                if query.lower() in r.get("chapter_title", "").lower()
                or query.lower() in r.get("summary", "").lower()
                or query.lower() in r.get("content", "").lower()
            ]
            return filtered

        return results

    def get_novel_hierarchy(
        self,
        novel_id: str,
        chapter_start: Optional[int] = None,
        chapter_end: Optional[int] = None,
    ) -> Dict[str, object]:
        """获取小说层级结构。无参数时只返回高层总结（总概述+剧情总结），有参数时返回指定范围章节。

        Args:
            novel_id: 小说ID
            chapter_start: 起始章节索引（可选），不提供则只返回高层总结
            chapter_end: 结束章节索引（可选），不提供则只返回高层总结
        """
        if not self.repository.exists(novel_id):
            return {"error": "Novel not found"}

        record = self.repository.load_novel(novel_id)
        summaries = self._extract_summaries(record)

        result = {
            "novel_id": record.novel_id,
            "title": record.title,
            "overall_summary": record.summary or "",
            "plot_summaries": summaries.get("plots", []),
            "chapter_count": len(record.chapters),
        }

        # 只有指定了章节范围，才返回章节详情
        if chapter_start is not None and chapter_end is not None:
            result["chapters"] = [
                {
                    "chapter_id": ch.chapter_id,
                    "chapter_title": ch.title,
                    "chapter_index": idx,
                    "summary": self._get_chapter_summary(record, ch.chapter_id),
                }
                for idx, ch in enumerate(record.chapters)
                if chapter_start <= idx <= chapter_end
            ]
        else:
            # 无参数时只返回章节数量，不返回详情
            result["chapters"] = []
            result["warning"] = "Use chapter_start and chapter_end to get specific chapter summaries"

        return result

    def _search_novels_semantic(self, query: str, k: int) -> List[Dict[str, object]]:
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return self._search_novels_keyword(query, k)

        results: List[Dict[str, object]] = []
        for record in self.repository.list_novels():
            summary = record.summary or ""
            if not summary:
                continue
            embedding = record.summary_embedding or self._embed_text(summary)
            if not embedding:
                continue
            score = self._cosine_similarity(query_embedding, embedding)
            results.append({
                "novel_id": record.novel_id,
                "title": record.title,
                "score": score,
                "summary": truncate_text(summary, max_chars=400),
                "features": record.features,
                "characters": record.characters,
            })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _search_chapters_semantic(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
        exclude_chapter_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, object]]:
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return self._search_chapters_keyword(query, k, novel_id, exclude_chapter_ids)

        results: List[Dict[str, object]] = []
        excluded = exclude_chapter_ids or set()
        records = self._filtered_records(novel_id)
        for record in records:
            for chapter in record.chapters:
                if chapter.chapter_id in excluded:
                    continue
                if not chapter.content:
                    continue
                embedding = chapter.embedding
                if not embedding:
                    continue
                score = self._cosine_similarity(query_embedding, embedding)
                results.append({
                    "novel_id": record.novel_id,
                    "title": record.title,
                    "chapter_id": chapter.chapter_id,
                    "chapter_title": chapter.title,
                    "score": score,
                    "source": "chapter_content",
                    "snippet": truncate_text(chapter.content, max_chars=300),
                })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    # -------------------------------------------------------------------------
    # Vector Store Based Search (new)
    # -------------------------------------------------------------------------

    def _can_vector_search(self) -> bool:
        """是否可以使用向量数据库搜索"""
        return (
            self.vector_store is not None
            and self.llm_provider is not None
            and self.storage_config.embedding_enabled
        )

    def _search_novels_vector(self, query: str, k: int) -> List[Dict[str, object]]:
        """使用向量数据库搜索全书总结"""
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return self._search_novels_keyword(query, k)

        hits = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
            content_type=ContentType.OVERALL_SUMMARY,
        )
        return self._vector_results_to_dict(hits)

    def _search_chapters_vector(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
        exclude_ids: Optional[set[str]],
    ) -> List[Dict[str, object]]:
        """使用向量数据库搜索章节总结"""
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return self._search_chapters_keyword(query, k, novel_id, exclude_ids)

        hits = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
            content_type=ContentType.CHAPTER_SUMMARY,
            novel_id=novel_id,
        )

        excluded = exclude_ids or set()
        results = []
        for hit in hits:
            if hit.chapter_id and hit.chapter_id in excluded:
                continue
            results.append({
                "novel_id": hit.novel_id,
                "title": hit.novel_title,
                "chapter_id": hit.chapter_id,
                "chapter_title": hit.chapter_title,
                "score": hit.score,
                "source": "chapter_summary",
                "snippet": truncate_text(hit.content, max_chars=300),
            })
        return results

    def _search_plot_summaries_vector(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
    ) -> List[Dict[str, object]]:
        """使用向量数据库搜索剧情总结"""
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return self._search_plot_summaries_keyword(query, k, novel_id)

        hits = self.vector_store.search(
            query_embedding=query_embedding,
            k=k,
            content_type=ContentType.PLOT_SUMMARY,
            novel_id=novel_id,
        )
        return [
            {
                "novel_id": hit.novel_id,
                "title": hit.novel_title,
                "volume_title": hit.volume_title,
                "start_chapter": hit.metadata.get("start_chapter", ""),
                "end_chapter": hit.metadata.get("end_chapter", ""),
                "score": hit.score,
                "source": "plot_summary",
                "snippet": truncate_text(hit.content, max_chars=500),
            }
            for hit in hits
        ]

    def _search_paragraphs_vector(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
        chapter_range: Optional[tuple[int, int]] = None,
    ) -> List[Dict[str, object]]:
        """使用向量数据库搜索原文段落"""
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return self._search_paragraphs_keyword(query, k, novel_id, chapter_range)

        hits = self.vector_store.search(
            query_embedding=query_embedding,
            k=k * 2,  # 多取一些，因为后面要过滤
            content_type=ContentType.PARAGRAPH,
            novel_id=novel_id,
        )

        results = [
            {
                "novel_id": hit.novel_id,
                "title": hit.novel_title,
                "chapter_id": hit.chapter_id,
                "chapter_index": hit.chapter_index,
                "chapter_title": hit.chapter_title,
                "paragraph_index": hit.metadata.get("paragraph_index", 0),
                "score": hit.score,
                "snippet": truncate_text(hit.content, max_chars=300),
            }
            for hit in hits
        ]

        # 按 chapter_range 过滤
        if chapter_range:
            start_idx, end_idx = chapter_range
            results = [
                r for r in results
                if r.get("chapter_index") is not None
                and start_idx <= r["chapter_index"] <= end_idx
            ]

        return results[:k]

    def _search_paragraphs_hybrid(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
        chapter_range: Optional[tuple[int, int]] = None,
    ) -> List[Dict[str, object]]:
        """混合检索 + Rerank：向量相似度 + BM25 关键词匹配 + 交叉编码器精排

        Stage 1 (召回): 向量 + BM25 混合召回 Top-K
        Stage 2 (精排): CrossEncoder 对候选结果重新排序
        """
        # 1. 向量检索 (多取一些用于 rerank)
        vector_results = self._search_paragraphs_vector(
            query, self.RERANK_TOP_K, novel_id, chapter_range
        )

        # 2. BM25 检索 (多取一些用于 rerank)
        bm25_results = self._search_paragraphs_bm25(
            query, self.RERANK_TOP_K, novel_id, chapter_range
        )

        # 3. 如果没有 BM25 结果，退回到纯向量
        if not bm25_results:
            return vector_results[:k]

        # 4. 合并结果
        # 构建段落 ID -> 索引的映射
        vector_map = {
            self._paragraph_key(r): (i, r)
            for i, r in enumerate(vector_results)
        }
        bm25_map = {
            self._paragraph_key(r): (i, r)
            for i, r in enumerate(bm25_results)
        }

        # 归一化分数并合并
        all_keys = set(vector_map.keys()) | set(bm25_map.keys())

        # 向量分数归一化 (Chroma 返回的 score 已经是 0-1)
        max_vector_score = max((r["score"] for r in vector_results), default=1.0)
        max_vector_score = max(max_vector_score, 0.001)  # 防止除零

        # BM25 分数归一化
        max_bm25_score = max((r["score"] for r in bm25_results), default=1.0)
        max_bm25_score = max(max_bm25_score, 0.001)  # 防止除零

        combined_scores: List[Tuple[float, Dict[str, object]]] = []

        for key in all_keys:
            combined_score = 0.0
            result_data: Dict[str, object] = {}

            if key in vector_map:
                idx, vec_r = vec_r = vector_map[key]
                norm_vector = vec_r["score"] / max_vector_score
                combined_score += self.VECTOR_WEIGHT * norm_vector
                result_data = dict(vec_r)

            if key in bm25_map:
                idx, bm25_r = bm25_map[key]
                norm_bm25 = bm25_r["score"] / max_bm25_score
                combined_score += self.BM25_WEIGHT * norm_bm25
                # 如果这个 key 已经在 result_data 中，更新分数
                if not result_data:
                    result_data = dict(bm25_r)

            result_data["score"] = combined_score
            combined_scores.append((combined_score, result_data))

        # 按混合分数排序
        combined_scores.sort(key=lambda x: x[0], reverse=True)

        # 取 Top-K 候选进行 Rerank
        candidates = [r for _, r in combined_scores[:self.RERANK_TOP_K]]

        # 如果只有少量候选或 CrossEncoder 不可用，直接返回
        if len(candidates) <= k or not CROSS_ENCODER_AVAILABLE:
            return candidates[:k]

        # Rerank: 使用交叉编码器精排
        reranked = self._rerank(query, candidates, k)
        return reranked

    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, object]],
        k: int,
    ) -> List[Dict[str, object]]:
        """使用 CrossEncoder 对候选结果进行精排

        CrossEncoder 将 query 和 doc 一起输入，能够捕捉更精细的语义匹配
        """
        if not CROSS_ENCODER_AVAILABLE or not candidates:
            return candidates[:k]

        # 懒加载模型
        if self._cross_encoder is None:
            try:
                self._cross_encoder = CrossEncoder(self.RERANK_MODEL)
                self._cross_encoder_model = self.RERANK_MODEL
            except Exception:
                return candidates[:k]

        # 准备 (query, doc) 对
        pairs = [
            (query, str(candidate.get("snippet", "") or ""))
            for candidate in candidates
        ]

        try:
            # 获取交叉编码器分数
            scores = self._cross_encoder.predict(pairs)

            # 将分数添加到结果中并按分数排序
            for i, candidate in enumerate(candidates):
                candidate["rerank_score"] = float(scores[i]) if hasattr(scores[i], '__float__') else scores[i]

            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return candidates[:k]
        except Exception:
            # 如果 rerank 失败，返回原始排序
            return candidates[:k]

    def _search_paragraphs_bm25(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
        chapter_range: Optional[tuple[int, int]] = None,
    ) -> List[Dict[str, object]]:
        """使用 BM25 算法搜索原文段落"""
        if not BM25_AVAILABLE:
            return self._search_paragraphs_keyword(query, k, novel_id, chapter_range)

        # 确保 BM25 索引已构建
        records = self._filtered_records(novel_id)
        for record in records:
            if record.novel_id not in self._bm25_cache:
                self._build_bm25_index(record)

        # 执行 BM25 搜索
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self._search_paragraphs_keyword(query, k, novel_id, chapter_range)

        results: List[Dict[str, object]] = []
        for record in records:
            cache = self._bm25_cache.get(record.novel_id)
            if not cache:
                continue

            bm25 = cache["bm25"]
            scores = bm25.get_scores(query_tokens)

            # 获取 top-k * 2（因为后面要过滤）
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 2]

            start_idx, end_idx = chapter_range if chapter_range else (None, None)

            for idx in top_indices:
                if scores[idx] <= 0:
                    continue
                para_info = cache["paragraphs"][idx]

                # 按 chapter_range 过滤
                if start_idx is not None and para_info["chapter_index"] < start_idx:
                    continue
                if end_idx is not None and para_info["chapter_index"] > end_idx:
                    continue

                results.append({
                    "novel_id": record.novel_id,
                    "title": record.title,
                    "chapter_id": para_info["chapter_id"],
                    "chapter_index": para_info["chapter_index"],
                    "chapter_title": para_info["chapter_title"],
                    "paragraph_index": para_info["paragraph_index"],
                    "score": scores[idx],
                    "snippet": truncate_text(para_info["text"], max_chars=300),
                })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _build_bm25_index(self, record) -> None:
        """为小说构建 BM25 索引"""
        if not BM25_AVAILABLE:
            return

        paragraphs: List[Dict] = []
        tokenized_corpus: List[List[str]] = []

        for chapter_idx, chapter in enumerate(record.chapters):
            if not chapter.content:
                continue
            paras = split_paragraphs(
                chapter.content,
                min_len=self.storage_config.paragraph_min_chars,
            )
            for para_idx, para_text in enumerate(paras):
                paragraphs.append({
                    "chapter_id": chapter.chapter_id,
                    "chapter_index": chapter_idx,
                    "chapter_title": chapter.title,
                    "paragraph_index": para_idx,
                    "text": para_text,
                })
                tokenized_corpus.append(self._tokenize(para_text))

        if not tokenized_corpus:
            return

        bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_cache[record.novel_id] = {
            "paragraphs": paragraphs,
            "bm25": bm25,
        }

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中英文混合分词

        策略:
        - 英文: 按空格 + 标点分割
        - 中文: 按字符级别 + 2-gram 组合
        """
        if not text:
            return []

        # 移除非文本字符但保留空格用于英文分词
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)

        tokens: List[str] = []

        # 英文按空格分词并转小写
        english_parts = text.split()
        for part in english_parts:
            if re.match(r'^[a-zA-Z]+$', part):
                tokens.append(part.lower())
            else:
                # 中文或混合: 字符级 + 2-gram
                chinese_chars = re.findall(r'[\u4e00-\u9fff]', part)
                for char in chinese_chars:
                    tokens.append(char)
                # 添加 2-gram
                for i in range(len(chinese_chars) - 1):
                    tokens.append(chinese_chars[i] + chinese_chars[i + 1])

        return tokens

    @staticmethod
    def _paragraph_key(result: Dict[str, object]) -> str:
        """生成段落唯一标识"""
        return (
            f"{result.get('novel_id', '')}:"
            f"{result.get('chapter_id', '')}:"
            f"{result.get('paragraph_index', 0)}"
        )

    def _vector_results_to_dict(self, hits: list) -> List[Dict[str, object]]:
        """将向量搜索结果转换为 dict 格式"""
        results: List[Dict[str, object]] = []
        for hit in hits:
            r: Dict[str, object] = {
                "novel_id": hit.novel_id,
                "title": hit.novel_title,
                "score": hit.score,
            }
            # 根据 content_type 添加不同字段
            if hit.content_type == ContentType.OVERALL_SUMMARY.value:
                r["summary"] = truncate_text(hit.content, max_chars=400)
            results.append(r)
        return results

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _search_novels_keyword(self, query: str, k: int) -> List[Dict[str, object]]:
        query_lower = query.lower()
        results: List[Dict[str, object]] = []
        for record in self.repository.list_novels():
            summaries = self._extract_summaries(record)
            chapter_summaries = summaries.get("chapters", [])
            plot_summaries = summaries.get("plots", [])

            score = 0
            snippets: List[str] = []

            overall_summary = record.summary or ""
            if overall_summary:
                s = overall_summary.lower().count(query_lower)
                score += s * 3
                if s > 0:
                    snippets.append(overall_summary)

            for item in plot_summaries if isinstance(plot_summaries, list) else []:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("summary", "") or "")
                if not text:
                    continue
                s = text.lower().count(query_lower)
                score += s * 2
                if s > 0 and len(snippets) < 2:
                    snippets.append(text)

            for item in chapter_summaries if isinstance(chapter_summaries, list) else []:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("summary", "") or "")
                if not text:
                    continue
                s = text.lower().count(query_lower)
                score += s
                if s > 0 and len(snippets) < 3:
                    snippets.append(text)

            if score == 0:
                fallback_content = record.chapters[0].content if record.chapters else ""
                score = fallback_content.lower().count(query_lower)
                if score > 0 and fallback_content:
                    snippets.append(fallback_content)

            if score == 0:
                continue
            display = "\n".join(snippets) if snippets else (record.summary or "")
            results.append({
                "novel_id": record.novel_id,
                "title": record.title,
                "score": score,
                "summary": truncate_text(display, max_chars=400),
            })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _search_chapters_keyword(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
        exclude_chapter_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, object]]:
        query_lower = query.lower()
        results: List[Dict[str, object]] = []
        excluded = exclude_chapter_ids or set()
        records = self._filtered_records(novel_id)
        for record in records:
            for chapter in record.chapters:
                if chapter.chapter_id in excluded:
                    continue
                content = chapter.content or ""
                score = content.lower().count(query_lower)
                if score == 0:
                    continue
                results.append({
                    "novel_id": record.novel_id,
                    "title": record.title,
                    "chapter_id": chapter.chapter_id,
                    "chapter_title": chapter.title,
                    "score": score,
                    "source": "chapter_content",
                    "snippet": truncate_text(content, max_chars=300),
                })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _search_chapter_summaries_keyword(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
    ) -> List[Dict[str, object]]:
        query_lower = query.lower()
        results: List[Dict[str, object]] = []
        records = self._filtered_records(novel_id)
        for record in records:
            chapter_map = {chapter.chapter_id: chapter for chapter in record.chapters}
            summaries = self._extract_summaries(record).get("chapters", [])
            if not isinstance(summaries, list):
                continue
            for item in summaries:
                if not isinstance(item, dict):
                    continue
                chapter_id = str(item.get("chapter_id", "") or "")
                summary_text = str(item.get("summary", "") or "")
                if not chapter_id or not summary_text:
                    continue
                score = summary_text.lower().count(query_lower)
                if score == 0:
                    continue
                chapter = chapter_map.get(chapter_id)
                chapter_title = str(item.get("title", "") or "")
                if chapter and chapter.title:
                    chapter_title = chapter.title
                results.append({
                    "novel_id": record.novel_id,
                    "title": record.title,
                    "chapter_id": chapter_id,
                    "chapter_title": chapter_title,
                    "score": score,
                    "source": "chapter_summary",
                    "snippet": truncate_text(summary_text, max_chars=300),
                })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _search_paragraphs_keyword(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
        chapter_range: Optional[tuple[int, int]] = None,
    ) -> List[Dict[str, object]]:
        query_lower = query.lower()
        results: List[Dict[str, object]] = []
        records = self._filtered_records(novel_id)

        start_idx, end_idx = chapter_range if chapter_range else (None, None)

        for record in records:
            for chapter_idx, chapter in enumerate(record.chapters):
                # 按 chapter_range 过滤
                if start_idx is not None and chapter_idx < start_idx:
                    continue
                if end_idx is not None and chapter_idx > end_idx:
                    continue

                paragraphs = split_paragraphs(
                    chapter.content,
                    min_len=self.storage_config.paragraph_min_chars,
                )
                for para_idx, paragraph in enumerate(paragraphs):
                    score = paragraph.lower().count(query_lower)
                    if score == 0:
                        continue
                    results.append({
                        "novel_id": record.novel_id,
                        "title": record.title,
                        "chapter_id": chapter.chapter_id,
                        "chapter_index": chapter_idx,
                        "chapter_title": chapter.title,
                        "paragraph_index": para_idx,
                        "score": score,
                        "snippet": truncate_text(paragraph, max_chars=300),
                    })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _search_paragraphs_semantic(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
        chapter_range: Optional[tuple[int, int]] = None,
    ) -> List[Dict[str, object]]:
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return self._search_paragraphs_keyword(query, k, novel_id, chapter_range)

        results: List[Dict[str, object]] = []
        records = self._filtered_records(novel_id)

        start_idx, end_idx = chapter_range if chapter_range else (None, None)

        for record in records:
            for chapter_idx, chapter in enumerate(record.chapters):
                # 按 chapter_range 过滤
                if start_idx is not None and chapter_idx < start_idx:
                    continue
                if end_idx is not None and chapter_idx > end_idx:
                    continue

                paragraphs = split_paragraphs(
                    chapter.content,
                    min_len=self.storage_config.paragraph_min_chars,
                )
                for para_idx, paragraph in enumerate(paragraphs):
                    embedding = self._embed_text(paragraph)
                    if not embedding:
                        continue
                    score = self._cosine_similarity(query_embedding, embedding)
                    results.append({
                        "novel_id": record.novel_id,
                        "title": record.title,
                        "chapter_id": chapter.chapter_id,
                        "chapter_index": chapter_idx,
                        "chapter_title": chapter.title,
                        "paragraph_index": para_idx,
                        "score": score,
                        "snippet": truncate_text(paragraph, max_chars=300),
                    })
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _filtered_records(self, novel_id: Optional[str]):
        if novel_id:
            return [self.repository.load_novel(novel_id)]
        return self.repository.list_novels()

    def _embed_text(self, text: str) -> List[float]:
        if not self.llm_provider:
            if self.embedding_builder:
                payload = truncate_text(text, max_chars=self.storage_config.embedding_max_chars)
                return self.embedding_builder.build(payload).vector
            return []
        payload = truncate_text(text, max_chars=self.storage_config.embedding_max_chars)
        try:
            # Use sync method if available (GatewayClient), otherwise fall back to async
            if hasattr(self.llm_provider, 'generate_embedding_sync'):
                embedding = self.llm_provider.generate_embedding_sync(payload)
            else:
                loop = asyncio.new_event_loop()
                embedding = loop.run_until_complete(self.llm_provider.generate_embedding(payload))
                loop.close()
            return embedding.vector if embedding else []
        except Exception:
            return []

    def _can_embed(self) -> bool:
        return self.storage_config.embedding_enabled and self.embedding_builder is not None

    @staticmethod
    def _extract_summaries(record) -> Dict[str, object]:
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        summaries = metadata.get("summaries")
        if isinstance(summaries, dict):
            return summaries
        return {}

    def _search_plot_summaries_keyword(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
    ) -> List[Dict[str, object]]:
        """关键词检索剧情总结"""
        query_lower = query.lower()
        results: List[Dict[str, object]] = []
        records = self._filtered_records(novel_id)

        for record in records:
            summaries = self._extract_summaries(record)
            plot_summaries = summaries.get("plots", [])
            if not isinstance(plot_summaries, list):
                continue

            for item in plot_summaries:
                if not isinstance(item, dict):
                    continue
                summary_text = str(item.get("summary", "") or "")
                volume_title = str(item.get("volume_title", "") or "")
                start_chapter = str(item.get("start_chapter", "") or "")
                end_chapter = str(item.get("end_chapter", "") or "")

                combined_text = f"{volume_title} {start_chapter} {end_chapter} {summary_text}"
                score = combined_text.lower().count(query_lower)

                if score == 0:
                    continue

                results.append({
                    "novel_id": record.novel_id,
                    "title": record.title,
                    "volume_title": volume_title,
                    "start_chapter": start_chapter,
                    "end_chapter": end_chapter,
                    "score": score,
                    "source": "plot_summary",
                    "snippet": truncate_text(summary_text, max_chars=500),
                })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _search_plot_summaries_semantic(
        self,
        query: str,
        k: int,
        novel_id: Optional[str],
    ) -> List[Dict[str, object]]:
        """语义检索剧情总结"""
        query_embedding = self._embed_text(query)
        if not query_embedding:
            return self._search_plot_summaries_keyword(query, k, novel_id)

        results: List[Dict[str, object]] = []
        records = self._filtered_records(novel_id)

        for record in records:
            summaries = self._extract_summaries(record)
            plot_summaries = summaries.get("plots", [])
            if not isinstance(plot_summaries, list):
                continue

            for item in plot_summaries:
                if not isinstance(item, dict):
                    continue
                summary_text = str(item.get("summary", "") or "")
                if not summary_text:
                    continue

                embedding = self._embed_text(summary_text)
                if not embedding:
                    continue

                score = self._cosine_similarity(query_embedding, embedding)

                results.append({
                    "novel_id": record.novel_id,
                    "title": record.title,
                    "volume_title": str(item.get("volume_title", "") or ""),
                    "start_chapter": str(item.get("start_chapter", "") or ""),
                    "end_chapter": str(item.get("end_chapter", "") or ""),
                    "score": score,
                    "source": "plot_summary",
                    "snippet": truncate_text(summary_text, max_chars=500),
                })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def _get_chapter_summary(self, record, chapter_id: str) -> str:
        """获取指定章节的摘要"""
        summaries = self._extract_summaries(record)
        chapter_summaries = summaries.get("chapters", [])
        if not isinstance(chapter_summaries, list):
            return ""

        for item in chapter_summaries:
            if not isinstance(item, dict):
                continue
            if str(item.get("chapter_id", "")) == chapter_id:
                return str(item.get("summary", "") or "")
        return ""

    def find_chapter_by_title(
        self,
        novel_id: str,
        chapter_title: str,
    ) -> Optional[Dict[str, object]]:
        """根据章节标题查找章节"""
        if not self.repository.exists(novel_id):
            return None

        record = self.repository.load_novel(novel_id)
        for index, chapter in enumerate(record.chapters):
            if chapter_title in chapter.title:
                return {
                    "novel_id": record.novel_id,
                    "title": record.title,
                    "chapter_id": chapter.chapter_id,
                    "chapter_title": chapter.title,
                    "chapter_index": index,
                    "summary": self._get_chapter_summary(record, chapter.chapter_id),
                    "content": chapter.content,
                }
        return None
