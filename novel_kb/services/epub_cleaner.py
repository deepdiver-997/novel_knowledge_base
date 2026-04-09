"""EPUB 清洗器：保留原始 HTML 结构，通过规则 + LLM 去除广告/水印段落。"""

import asyncio
import html as html_module
import json
import re
from pathlib import Path
from typing import Optional

import aiohttp

from novel_kb.utils.logger import logger

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_OLLAMA_BASE_URL = "http://localhost:11434"
_OLLAMA_MODEL = "qwen3:8b"
_OLLAMA_TIMEOUT = 120  # seconds per LLM call
_LLM_BATCH_SIZE = 15  # paragraphs per LLM call

# 规则匹配关键词 ── 命中即视为广告/作者求票
_AD_KEYWORDS: list[str] = [
    "推荐票", "月票", "收藏本站", "加入书签", "更新最快", "最新章节",
    "求订阅", "求月票", "求推荐", "求收藏", "投推荐票", "投月票",
    "手打", "无弹窗", "笔趣", "手机看", "手机阅读", "手机用户",
    "正版订阅", "正版首发", "首发更新", "首发网站",
    "请收藏", "请投", "请支持", "请投票",
    "微信公众号", "微信搜索", "微信上搜",
    "下载app", "下载APP", "APP免费",
    "粉丝节", "起点币", "起点中文网首发",
    "公众号", "公众账号",
]

# 整行匹配正则（命中 → 广告）
_AD_REGEXES: list[re.Pattern[str]] = [
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"www\.\S+", re.IGNORECASE),
    re.compile(r"（?本文来自.+）?"),
    re.compile(r"（?求.{0,6}票.{0,6}）"),
    re.compile(r"PS[\.\s：:].{0,80}(月票|推荐|粉丝节|投票|支持|赞赏)"),
]

# CSS 样式 ── 防止字体过大
_EPUB_CSS = """\
body { font-family: serif; font-size: 1em; line-height: 1.8; margin: 1em; }
h1, h2, h3 { font-size: 1.2em; font-weight: bold; margin: 1em 0 0.5em; }
p { text-indent: 2em; margin: 0.3em 0; }
"""

# HTML 解析正则
_TITLE_TAG_RE = re.compile(r"<h[1-3][^>]*>(.*?)</h[1-3]>", re.DOTALL | re.IGNORECASE)
_PARA_RE = re.compile(r"<p[^>]*>(.*?)</p>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_CHAPTER_TITLE_RE = re.compile(r"^第.{1,20}[章节卷回]")


class EPUBCleaner:
    """EPUB 清洗器：移除空章节和广告段落，输出 *_cleaned.epub。"""

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @staticmethod
    def cleaned_output_path(input_path: str) -> Path:
        source = Path(input_path).expanduser()
        return source.with_name(f"{source.stem}_cleaned{source.suffix}")

    @staticmethod
    def clean(input_path: str, output_path: Optional[str] = None) -> str:
        """
        清洗 EPUB 文件。

        1. 保留原始 HTML 段落结构（不再 strip_html 全文）
        2. 规则匹配去除广告/求票段落
        3. 调用 LLM（ollama qwen3:8b）识别剩余可疑段落
        4. 添加 CSS 统一字体/排版
        """
        try:
            from ebooklib import ITEM_DOCUMENT, epub
        except ImportError as exc:
            raise RuntimeError("ebooklib is required for EPUB cleaning") from exc

        source = Path(input_path).expanduser()
        if source.suffix.lower() != ".epub":
            raise ValueError("EPUB cleaning only supports .epub files")
        if not source.exists():
            raise FileNotFoundError(f"Input file not found: {source}")

        target = (
            Path(output_path).expanduser()
            if output_path
            else EPUBCleaner.cleaned_output_path(str(source))
        )
        target.parent.mkdir(parents=True, exist_ok=True)

        original = epub.read_epub(str(source))

        # ---- 元数据 ----
        title_meta = original.get_metadata("DC", "title")
        title = title_meta[0][0] if title_meta else source.stem

        language_meta = original.get_metadata("DC", "language")
        language = language_meta[0][0] if language_meta else "zh"

        identifier_meta = original.get_metadata("DC", "identifier")
        identifier = identifier_meta[0][0] if identifier_meta else source.stem

        # ---- 逐章清洗 ----
        items = list(original.get_items_of_type(ITEM_DOCUMENT))

        rule_removed = 0
        llm_removed = 0

        # Phase 1: 规则清洗 + 收集可疑段落
        chapter_data: list[dict] = []  # {title, paragraphs: [(text, keep)]}
        suspicious_global: list[tuple[int, int, str]] = []  # (ch_idx, p_idx, text)

        for ch_idx, item in enumerate(items):
            raw_html = item.get_content().decode("utf-8", errors="ignore")
            ch_title = _extract_title(raw_html, ch_idx + 1)
            paragraphs = _extract_paragraphs(raw_html)

            tagged: list[tuple[str, bool]] = []  # (text, keep?)
            for p_idx, p_text in enumerate(paragraphs):
                if _is_ad_by_rule(p_text):
                    tagged.append((p_text, False))
                    rule_removed += 1
                elif _is_suspicious(p_text, p_idx, len(paragraphs)):
                    tagged.append((p_text, True))  # 暂定保留
                    suspicious_global.append((ch_idx, len(tagged) - 1, p_text))
                else:
                    tagged.append((p_text, True))

            chapter_data.append({"title": ch_title, "paragraphs": tagged})

        # Phase 2: LLM 识别可疑段落
        if suspicious_global:
            logger.info(
                "Rule-based removed %d paragraphs; %d suspicious → sending to LLM",
                rule_removed,
                len(suspicious_global),
            )
            llm_decisions = _llm_classify_suspicious(suspicious_global)
            for (ch_idx, p_idx, _text), is_ad in zip(suspicious_global, llm_decisions):
                if is_ad:
                    paras = chapter_data[ch_idx]["paragraphs"]
                    if p_idx < len(paras):
                        paras[p_idx] = (paras[p_idx][0], False)
                        llm_removed += 1
        else:
            logger.info(
                "Rule-based removed %d paragraphs; no suspicious paragraphs for LLM",
                rule_removed,
            )

        # ---- 构建新 EPUB ----
        cleaned_book = epub.EpubBook()
        cleaned_book.set_identifier(str(identifier))
        cleaned_book.set_title(title)
        cleaned_book.set_language(language)

        # 添加 CSS
        css_item = epub.EpubItem(
            uid="style_default",
            file_name="style/default.css",
            media_type="text/css",
            content=_EPUB_CSS.encode("utf-8"),
        )
        cleaned_book.add_item(css_item)

        chapters = []
        kept_count = 0
        removed_count = 0

        for ch_idx, ch in enumerate(chapter_data):
            kept_paragraphs = [t for t, keep in ch["paragraphs"] if keep]
            if not kept_paragraphs:
                removed_count += 1
                continue

            kept_count += 1
            chapter = epub.EpubHtml(
                title=ch["title"],
                file_name=f"chap_{kept_count:04d}.xhtml",
                lang=language,
            )
            chapter.add_item(css_item)
            chapter.set_content(
                _build_chapter_html(ch["title"], kept_paragraphs)
            )
            cleaned_book.add_item(chapter)
            chapters.append(chapter)

        if not chapters:
            raise RuntimeError("No non-empty chapters found after cleaning")

        cleaned_book.toc = list(chapters)
        cleaned_book.spine = ["nav", *chapters]
        cleaned_book.add_item(epub.EpubNcx())
        cleaned_book.add_item(epub.EpubNav())

        epub.write_epub(str(target), cleaned_book)

        logger.info(
            "EPUB cleaned: input=%s output=%s chapters=%d→%d "
            "paragraphs_removed=%d(rule=%d,llm=%d)",
            source,
            target,
            len(items),
            kept_count,
            rule_removed + llm_removed,
            rule_removed,
            llm_removed,
        )
        return str(target)


# ---------------------------------------------------------------------------
# HTML 解析辅助
# ---------------------------------------------------------------------------


def _strip_tags(html_fragment: str) -> str:
    """去除 HTML 标签，返回纯文本。"""
    return _TAG_RE.sub("", html_fragment).strip()


def _extract_title(html_content: str, fallback_index: int) -> str:
    """从 HTML 中提取章节标题（h1-h3 标签）。"""
    m = _TITLE_TAG_RE.search(html_content)
    if m:
        return _strip_tags(m.group(1)).strip()
    # 尝试从第一个 <p> 提取
    pm = _PARA_RE.search(html_content)
    if pm:
        first_p = _strip_tags(pm.group(1)).strip()
        if _CHAPTER_TITLE_RE.match(first_p) and len(first_p) <= 50:
            return first_p
        if len(first_p) <= 30 and first_p:
            return first_p
    return f"第{fallback_index}章"


def _extract_paragraphs(html_content: str) -> list[str]:
    """提取所有 <p> 标签的纯文本内容。"""
    results = []
    for m in _PARA_RE.finditer(html_content):
        text = _strip_tags(m.group(1)).strip()
        if text:
            results.append(text)
    return results


# ---------------------------------------------------------------------------
# 广告识别
# ---------------------------------------------------------------------------


def _is_ad_by_rule(text: str) -> bool:
    """基于关键词和正则判断是否为广告。"""
    for kw in _AD_KEYWORDS:
        if kw in text:
            return True
    for pattern in _AD_REGEXES:
        if pattern.search(text):
            return True
    return False


def _is_suspicious(text: str, para_index: int, total_paragraphs: int) -> bool:
    """
    判断段落是否"可疑"（需要 LLM 进一步审查）。

    可疑条件：出现在章首/章末 + 较短 + 包含弱特征词。
    """
    if len(text) > 200:
        return False

    is_boundary = para_index < 2 or para_index >= total_paragraphs - 3
    if not is_boundary:
        return False

    # 弱特征词 ── 不确定是广告，需要 LLM 确认
    weak_signals = [
        "兄弟姐妹", "书友", "感谢", "拜谢", "多谢", "致歉",
        "抱歉", "请假", "今天", "明天", "昨天",
        "更新", "上传", "补上", "加更", "爆更",
        "新书", "完结", "上架", "免费",
        "冲榜", "成绩", "订阅", "读者",
        "辰东", "作者", "作品",
    ]
    return any(w in text for w in weak_signals)


# ---------------------------------------------------------------------------
# LLM 分类
# ---------------------------------------------------------------------------


def _llm_classify_suspicious(
    items: list[tuple[int, int, str]],
) -> list[bool]:
    """
    调用 ollama qwen3:8b 对可疑段落进行广告/正文分类。

    返回 list[bool]，True = 是广告，False = 是正文。
    """
    texts = [t for _, _, t in items]

    # 去重以减少 LLM 调用
    unique_texts = list(dict.fromkeys(texts))
    logger.info("LLM classifying %d unique suspicious paragraphs", len(unique_texts))

    # 批量分类
    decisions: dict[str, bool] = {}
    try:
        decisions = asyncio.run(_batch_classify(unique_texts))
    except Exception as exc:
        logger.warning(
            "LLM classification failed (%s), falling back to rule-based only", exc
        )
        return [False] * len(items)

    return [decisions.get(t, False) for t in texts]


async def _batch_classify(texts: list[str]) -> dict[str, bool]:
    """分批发送段落到 LLM 进行分类。"""
    results: dict[str, bool] = {}
    timeout = aiohttp.ClientTimeout(total=_OLLAMA_TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for batch_start in range(0, len(texts), _LLM_BATCH_SIZE):
            batch = texts[batch_start : batch_start + _LLM_BATCH_SIZE]
            batch_results = await _classify_batch(session, batch)
            results.update(batch_results)
            ad_count = sum(1 for v in batch_results.values() if v)
            logger.info(
                "LLM batch %d-%d done, %d/%d classified as ad",
                batch_start,
                batch_start + len(batch),
                ad_count,
                len(batch),
            )

    return results


async def _classify_batch(
    session: aiohttp.ClientSession,
    paragraphs: list[str],
) -> dict[str, bool]:
    """单次 LLM 调用，分类一批段落。"""
    numbered = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(paragraphs))
    prompt = (
        "你是小说文本清洗助手。下面是从网络小说中提取的若干段落，请判断每个段落"
        "是【广告/作者求票/推广/非正文内容】还是【正文内容】。\n\n"
        "判断标准：\n"
        "- 作者向读者求推荐票、月票、收藏、订阅等 → 广告\n"
        "- 推广游戏、APP、微信公众号等 → 广告\n"
        "- 作者请假、道歉、感言等非小说叙事 → 广告\n"
        "- 小说本身的叙事、对话、描写 → 正文\n\n"
        f"段落列表：\n{numbered}\n\n"
        "请用 JSON 格式回复，只有一个 key \"ads\"，值是广告段落的编号数组。"
        "例如 {{\"ads\": [1, 3]}} 表示第1和第3段是广告。"
        "如果全部是正文，返回 {{\"ads\": []}}。\n"
        "/no_think"
    )

    try:
        async with session.post(
            f"{_OLLAMA_BASE_URL}/api/generate",
            json={
                "model": _OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": 0.1,
                "format": "json",
                "stream": False,
            },
        ) as response:
            response.raise_for_status()
            payload = await response.json()
            raw = payload.get("response", "{}")

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # 尝试提取 JSON
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    data = json.loads(raw[start : end + 1])
                else:
                    logger.warning("LLM returned non-JSON: %s", raw[:200])
                    return {p: False for p in paragraphs}

            ad_indices = set(data.get("ads", []))
            return {
                p: (i + 1) in ad_indices for i, p in enumerate(paragraphs)
            }

    except Exception as exc:
        logger.warning("LLM classify batch failed: %s", exc)
        return {p: False for p in paragraphs}


# ---------------------------------------------------------------------------
# HTML 输出
# ---------------------------------------------------------------------------


def _build_chapter_html(title: str, paragraphs: list[str]) -> str:
    """构建章节 HTML body 内容，保持段落结构。

    只提供 <html><body> 内容，不加 XML 声明，让 ebooklib 自行包装模板。
    """
    body_parts = [f"<h2>{html_module.escape(title)}</h2>"]
    for p in paragraphs:
        # 跳过标题段（避免标题和正文重复）
        if p.strip() == title.strip():
            continue
        body_parts.append(f"<p>{html_module.escape(p)}</p>")
    # 确保 body 不为空（至少有标题）
    body = "\n".join(body_parts)
    return (
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        "<head>"
        f"<title>{html_module.escape(title)}</title>"
        '<link rel="stylesheet" type="text/css" href="style/default.css"/>'
        "</head>"
        f"<body>\n{body}\n</body>"
        "</html>"
    )
