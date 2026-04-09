from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Literal


ActionMode = Literal["report", "delete", "confirm-delete"]


DEFAULT_FAILURE_PATTERNS = [
    r"请提供.*(内容|主题|文本)",
    r"未提供.*(主题|内容|文本)",
    r"无法生成.*(总结|摘要)",
    r"我是一个ai助手",
    r"这是一个(示例|通用|简短).*(总结|摘要)",
    r"rate\s*limit|429|too\s*many\s*requests|api\s*error|providererror",
]


@dataclass
class Fingerprint:
    text: str
    normalized: str


@dataclass
class SuspiciousSummary:
    index: int
    chapter_id: str
    title: str
    summary: str
    score: float
    reasons: list[str]


def _normalize(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", "", lowered)
    return lowered


def _short(text: str, max_chars: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "…"


def load_fingerprints(path: Path) -> list[Fingerprint]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    cleaned = [line for line in lines if line and not line.startswith("#")]
    return [Fingerprint(text=line, normalized=_normalize(line)) for line in cleaned]


def _max_similarity(summary_normalized: str, fingerprints: list[Fingerprint]) -> tuple[float, str]:
    best_score = 0.0
    best_text = ""
    for fp in fingerprints:
        score = SequenceMatcher(None, summary_normalized, fp.normalized).ratio()
        if score > best_score:
            best_score = score
            best_text = fp.text
    return best_score, best_text


def _collect_summary_entries(data: dict[str, Any], path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.name.endswith(".progress.json"):
        progress = data.get("analysis_progress", {}) if isinstance(data.get("analysis_progress"), dict) else {}
        items = progress.get("chapter_summaries", []) if isinstance(progress.get("chapter_summaries"), list) else []
        return items, "progress"

    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    summaries = metadata.get("summaries", {}) if isinstance(metadata.get("summaries"), dict) else {}
    items = summaries.get("chapters", []) if isinstance(summaries.get("chapters"), list) else []
    return items, "novel"


def _is_failure_pattern(text: str) -> bool:
    for pattern in DEFAULT_FAILURE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def find_suspicious_summaries(
    summary_items: list[dict[str, Any]],
    fingerprints: list[Fingerprint],
    min_length: int,
    similarity_threshold: float,
) -> list[SuspiciousSummary]:
    suspicious: list[SuspiciousSummary] = []
    for index, item in enumerate(summary_items):
        if not isinstance(item, dict):
            continue

        summary = str(item.get("summary", "") or "")
        chapter_id = str(item.get("chapter_id", "") or "")
        title = str(item.get("title", "") or "")
        normalized = _normalize(summary)

        reasons: list[str] = []
        score = 0.0

        if not summary.strip():
            continue

        if len(normalized) < min_length:
            reasons.append(f"too_short<{min_length}")
            score += 0.45

        if _is_failure_pattern(summary):
            reasons.append("matches_failure_pattern")
            score += 0.9

        if fingerprints:
            sim_score, matched_fp = _max_similarity(normalized, fingerprints)
            if sim_score >= similarity_threshold:
                reasons.append(f"fingerprint_sim={sim_score:.2f}:{_short(matched_fp, 50)}")
                score += max(0.3, sim_score)

            for fp in fingerprints:
                if fp.normalized and fp.normalized in normalized:
                    reasons.append(f"contains_fingerprint:{_short(fp.text, 50)}")
                    score += 1.0
                    break

        if re.search(r"(.)\1{8,}", summary):
            reasons.append("repeated_chars")
            score += 0.3

        if reasons:
            suspicious.append(
                SuspiciousSummary(
                    index=index,
                    chapter_id=chapter_id,
                    title=title,
                    summary=summary,
                    score=round(score, 3),
                    reasons=reasons,
                )
            )

    suspicious.sort(key=lambda item: item.score, reverse=True)
    return suspicious


def _delete_entries(items: list[dict[str, Any]], indexes: set[int]) -> list[dict[str, Any]]:
    return [item for idx, item in enumerate(items) if idx not in indexes]


def _rewrite_with_items(data: dict[str, Any], mode: str, items: list[dict[str, Any]]) -> None:
    if mode == "progress":
        progress = data.setdefault("analysis_progress", {})
        progress["chapter_summaries"] = items
        return
    metadata = data.setdefault("metadata", {})
    summaries = metadata.setdefault("summaries", {})
    summaries["chapters"] = items


def audit_summary_file(
    file_path: Path,
    fingerprint_path: Path,
    action: ActionMode = "report",
    min_length: int = 20,
    similarity_threshold: float = 0.72,
    min_score: float = 0.0,
    backup: bool = True,
    confirm_callback: Callable[[SuspiciousSummary], bool] | None = None,
) -> tuple[list[SuspiciousSummary], int]:
    data = json.loads(file_path.read_text(encoding="utf-8"))
    items, mode = _collect_summary_entries(data, file_path)
    fingerprints = load_fingerprints(fingerprint_path)

    suspicious = find_suspicious_summaries(
        summary_items=items,
        fingerprints=fingerprints,
        min_length=min_length,
        similarity_threshold=similarity_threshold,
    )
    if min_score > 0:
        suspicious = [item for item in suspicious if item.score >= min_score]

    removed_count = 0
    if action == "report" or not suspicious:
        return suspicious, removed_count

    delete_indexes: set[int] = set()
    if action == "delete":
        delete_indexes = {item.index for item in suspicious}
    elif action == "confirm-delete":
        callback = confirm_callback or (lambda _: False)
        for item in suspicious:
            if callback(item):
                delete_indexes.add(item.index)

    if not delete_indexes:
        return suspicious, removed_count

    if backup:
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        backup_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    new_items = _delete_entries(items, delete_indexes)
    _rewrite_with_items(data, mode, new_items)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    removed_count = len(delete_indexes)
    return suspicious, removed_count
