"""从小说 PDF 提取正文，清洗页眉页脚后按句子切片，输出 raw/chunks.json。

用法: python scripts/extract.py [PDF路径]
"""
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = r"E:\光与太阳，风筝与风.pdf"

CHUNK_CHARS = 320
OVERLAP_SENTENCES = 1

VOLUME_RE = re.compile(r"^第([一二三])卷：$")
CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十]+章\s+\S+$")
NOISE_RE = re.compile(r"^(·\d+·|光与太阳，风筝与风|GuangYuTaiYang,FengZhengYuFeng\.)$")
SENTENCE_RE = re.compile(r"[^。！？…；]*(?:[。！？；]+|…+)[”」』）]*|[^。！？…；]+$")


def read_pages(pdf_path):
    doc = fitz.open(pdf_path)
    return [page.get_text() for page in doc]


def parse_sections(pages):
    """返回 [(卷, 章节标题, 正文)]，跳过目录等前置内容。"""
    sections = []
    volume, title, buf = None, None, []
    section_volume = None
    started = False

    def flush():
        # 目录里的“后记”没有制表符，会把页码收进来，靠“不含汉字”过滤掉
        if title and buf and re.search(r"[\u4e00-\u9fff]", "".join(buf)):
            sections.append((section_volume, title, "".join(buf)))

    for page in pages:
        for raw in page.splitlines():
            line = raw.strip()
            if not line:
                continue
            if m := VOLUME_RE.match(line):
                volume = f"第{m.group(1)}卷"
                continue
            if NOISE_RE.match(line):
                continue
            if CHAPTER_RE.match(line) or line == "后记":
                if "\t" in raw:  # 目录行带制表符和页码
                    continue
                if line == "后记" and title == "后记":
                    continue
                flush()
                title, buf, started = re.sub(r"\s+", " ", line), [], True
                section_volume = volume
                continue
            if started:
                buf.append(line)
    flush()
    return sections


def split_sentences(text):
    return [s for s in SENTENCE_RE.findall(text) if s.strip()]


def chunk_section(text):
    sentences = split_sentences(text)
    chunks, cur = [], []
    for s in sentences:
        cur.append(s)
        if sum(map(len, cur)) >= CHUNK_CHARS:
            chunks.append("".join(cur))
            cur = cur[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES else []
    if cur and (not chunks or "".join(cur) not in chunks[-1]):
        chunks.append("".join(cur))
    return chunks


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    sections = parse_sections(read_pages(pdf_path))
    out = []
    for volume, title, text in sections:
        where = f"{volume}·{title}" if volume and title != "后记" else title
        for c in chunk_section(text):
            out.append({"id": len(out), "where": where, "text": c})

    dest = ROOT / "raw" / "chunks.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(len(c["text"]) for c in out)
    print(f"{len(sections)} 个章节, {len(out)} 个片段, 共 {total} 字 -> {dest}")


if __name__ == "__main__":
    main()
