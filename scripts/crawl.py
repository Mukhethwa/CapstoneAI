"""Crawl the ZAIO website and save a cleaned snapshot to data/website_pages.json.

Usage:  python -m scripts.crawl [--max-pages 80] [--include-blog]
"""

from __future__ import annotations

import argparse
import logging

from app.config import get_settings
from app.web_crawler import WebsiteCrawler, save_pages


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=settings.website_url)
    parser.add_argument("--max-pages", type=int, default=settings.website_max_pages)
    parser.add_argument("--include-blog", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    crawler = WebsiteCrawler(args.url, max_pages=args.max_pages, include_blog=args.include_blog)
    pages = crawler.crawl()
    save_pages(pages, settings.website_snapshot)
    total = sum(len(p.text) for p in pages)
    print(f"Saved {len(pages)} pages ({total:,} characters) to {settings.website_snapshot}")


if __name__ == "__main__":
    main()
