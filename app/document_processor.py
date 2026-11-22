"""Part 1: load the handbook PDF, extract and clean its text, and split it into chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    number: int  # 1-based page number in the PDF
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    page: int


_SPACED_CAPS = re.compile(r"\b(?:[A-Z] ){2,}[A-Z]\b")  # "E X C I T I N G" -> "EXCITING"
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d(?:\s?\d){0,2}\s*$")  # lines like "0 4" or "12"


def clean_text(text: str) -> str:
    """Normalise raw PDF text: drop page-number lines, join spaced capitals, tidy whitespace."""
    lines = []
    for line in text.splitlines():
        if _PAGE_NUMBER_LINE.match(line):
            continue
        line = _SPACED_CAPS.sub(lambda m: m.group(0).replace(" ", ""), line)
        line = re.sub(r"[ \t ]+", " ", line).strip()
        lines.append(line)
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def load_pdf(path: str | Path) -> list[Page]:
    """Extract the text of every page that contains any."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Handbook not found at {path}")
    reader = PdfReader(str(path))
    pages = []
    for index, pdf_page in enumerate(reader.pages, start=1):
        text = clean_text(pdf_page.extract_text() or "")
        if text:
            pages.append(Page(number=index, text=text))
    return pages


def _split_units(text: str) -> list[str]:
    """Split text into small units (lines, then sentences) that chunks are built from."""
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        units.extend(s for s in re.split(r"(?<=[.!?])\s+", line) if s)
    return units


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """Greedily pack units into chunks of at most ``chunk_size`` characters with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    units: list[str] = []
    for unit in _split_units(text):
        # Hard-wrap any single unit that is longer than a chunk.
        while len(unit) > chunk_size:
            cut = unit.rfind(" ", 0, chunk_size)
            cut = cut if cut > 0 else chunk_size
            units.append(unit[:cut].strip())
            unit = unit[cut:].strip()
        if unit:
            units.append(unit)

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        added = len(unit) + (1 if current else 0)
        if current and length + added > chunk_size:
            chunks.append("\n".join(current))
            # Carry trailing units forward as overlap.
            overlap: list[str] = []
            overlap_len = 0
            for prev in reversed(current):
                if overlap_len + len(prev) + 1 > chunk_overlap:
                    break
                overlap.insert(0, prev)
                overlap_len += len(prev) + 1
            current, length = overlap, max(overlap_len - 1, 0)
            added = len(unit) + (1 if current else 0)
        current.append(unit)
        length += added
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_pages(pages: list[Page], chunk_size: int = 800, chunk_overlap: int = 150) -> list[Chunk]:
    """Chunk each page separately so every chunk maps to exactly one source page."""
    chunks = []
    for page in pages:
        for i, text in enumerate(split_text(page.text, chunk_size, chunk_overlap)):
            chunks.append(Chunk(id=f"page-{page.number}-chunk-{i}", text=text, page=page.number))
    return chunks
