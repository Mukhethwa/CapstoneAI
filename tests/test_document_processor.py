from pathlib import Path

import pytest

from app.config import get_settings
from app.document_processor import Page, chunk_pages, clean_text, load_pdf, split_text


def test_clean_text_removes_page_numbers_and_joins_spaced_capitals():
    raw = "0 4\nE X C I T I N G  times   await\n\n\n\nFEES\n12"
    assert clean_text(raw) == "EXCITING times await\n\nFEES"


def test_split_text_respects_chunk_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = split_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_split_text_overlaps_consecutive_chunks():
    text = " ".join(f"Sentence {i}." for i in range(60))
    chunks = split_text(text, chunk_size=100, chunk_overlap=30)
    for previous, current in zip(chunks, chunks[1:]):
        last_unit = previous.splitlines()[-1]
        assert last_unit in current


def test_split_text_hard_wraps_very_long_words_and_lines():
    chunks = split_text("word " * 500, chunk_size=120, chunk_overlap=20)
    assert all(len(c) <= 120 for c in chunks)
    assert "".join(chunks).count("word") >= 500


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_split_text_rejects_bad_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=size, chunk_overlap=overlap)


def test_chunk_pages_keeps_page_numbers_and_unique_ids():
    pages = [Page(1, "Intro text."), Page(2, "More text. " * 200)]
    chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=50)
    assert {c.page for c in chunks} == {1, 2}
    assert len({c.id for c in chunks}) == len(chunks)


def test_load_pdf_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_pdf(Path("does-not-exist.pdf"))


@pytest.mark.skipif(not get_settings().handbook_path.exists(), reason="handbook PDF not present")
def test_load_real_handbook_extracts_known_content():
    pages = load_pdf(get_settings().handbook_path)
    assert len(pages) >= 20
    fees_page = next(p for p in pages if "R 38,950" in p.text)
    assert fees_page.number == 16
