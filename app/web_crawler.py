"""Capstone Part 1: crawl the ZAIO website and extract clean page text.

zaio.io is a server-rendered Next.js site, so the full page content is already in the HTML
that the server returns. A plain HTTP client plus BeautifulSoup is therefore enough (no
headless browser such as Puppeteer is needed).

The crawler:
- respects robots.txt and only visits pages on the same host
- seeds from sitemap.xml and the start URL, then follows internal links (breadth first)
- skips blog posts by default (``include_blog=True`` to keep them)
- removes navigation, headers, footers, scripts, buttons and decorative (aria-hidden) markup
- removes boilerplate lines that repeat across most pages (calls to action, banners)
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENT = "CapstoneRAGCrawler/1.0 (student project; respects robots.txt)"

# Tags whose content is never useful page content.
REMOVE_TAGS = [
    "script", "style", "noscript", "svg", "iframe", "form", "footer", "nav", "aside",
    "dialog", "template", "picture", "video", "audio",
]
# ARIA roles for site chrome.
REMOVE_ROLES = ["navigation", "banner", "contentinfo", "dialog", "alert", "search"]
# class/id fragments that indicate chrome (cookie banners, announcement bars, menus).
CHROME_HINTS = re.compile(r"cookie|announcement|navbar|menu|breadcrumb|skip-?link|toast|modal", re.I)
SKIP_EXTENSIONS = re.compile(r"\.(pdf|jpe?g|png|gif|svg|webp|zip|mp4|mp3|css|js|xml|ico)$", re.I)
MIN_LINE_LENGTH = 3


@dataclass(frozen=True)
class WebPage:
    url: str
    title: str
    text: str


def normalise_url(url: str) -> str:
    """Drop query strings, fragments and trailing slashes so each page is visited once."""
    parsed = urlparse(url)
    path = re.sub(r"/+$", "", parsed.path) or "/"
    return f"{parsed.scheme}://{parsed.netloc.lower()}{path}"


def _is_chrome(tag) -> bool:
    if tag.attrs is None:
        return False
    if tag.get("aria-hidden") == "true" or tag.get("hidden") is not None:
        return True
    if tag.get("role") in REMOVE_ROLES:
        return True
    hints = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
    return bool(CHROME_HINTS.search(hints))


def extract_content(html: str | BeautifulSoup) -> tuple[str, str]:
    """Return (title, cleaned text) for one HTML page. A parsed soup is modified in place."""
    soup = html if isinstance(html, BeautifulSoup) else BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    title = re.sub(r"\s*[|–—-]\s*Zaio.*$", "", title, flags=re.I).strip() or title

    for tag in soup(REMOVE_TAGS):
        tag.decompose()
    # The site header is chrome, but <header> elements inside <main> are section headings.
    for tag in soup.find_all("header"):
        if tag.find_parent("main") is None:
            tag.decompose()
    for tag in soup.find_all(_is_chrome):
        tag.decompose()

    # Prefer the <main> element(s); fall back to <body>.
    mains = soup.find_all("main")
    roots = mains or [soup.body or soup]
    text = "\n".join(root.get_text("\n") for root in roots)
    return title, clean_lines(text)


def clean_lines(text: str) -> str:
    """Normalise whitespace, drop tiny fragments and consecutive duplicate lines."""
    lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"[\s ​⁠]+", " ", line).strip()
        if len(line) < MIN_LINE_LENGTH or (lines and lines[-1] == line):
            continue
        lines.append(line)
    return "\n".join(lines)


def remove_boilerplate(pages: list[WebPage], max_share: float = 0.5, min_pages: int = 4) -> list[WebPage]:
    """Remove lines that appear on more than ``max_share`` of pages (site-wide chrome)."""
    if len(pages) < min_pages:
        return pages
    counts = Counter(line for p in pages for line in set(p.text.splitlines()))
    limit = max_share * len(pages)
    common = {line for line, n in counts.items() if n > limit}
    cleaned = []
    for page in pages:
        text = "\n".join(line for line in page.text.splitlines() if line not in common)
        cleaned.append(WebPage(url=page.url, title=page.title, text=text))
    return cleaned


def extract_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    return [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True)]


def _sitemap_urls(client: httpx.Client, sitemap_url: str) -> list[str]:
    try:
        response = client.get(sitemap_url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Could not read sitemap %s: %s", sitemap_url, exc)
        return []
    return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", response.text)


class WebsiteCrawler:
    def __init__(
        self,
        start_url: str = "https://www.zaio.io/",
        max_pages: int = 80,
        include_blog: bool = False,
        delay_seconds: float = 0.3,
        client: httpx.Client | None = None,
    ):
        self.start_url = normalise_url(start_url)
        self.host = urlparse(self.start_url).netloc
        self.max_pages = max_pages
        self.include_blog = include_blog
        self.delay_seconds = delay_seconds
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=20
        )
        self.robots = RobotFileParser()

    def _load_robots(self) -> None:
        root = f"{urlparse(self.start_url).scheme}://{self.host}"
        try:
            response = self.client.get(f"{root}/robots.txt")
            self.robots.parse(response.text.splitlines() if response.status_code == 200 else [])
        except httpx.HTTPError:
            self.robots.parse([])

    def should_visit(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or parsed.netloc.lower() != self.host:
            return False
        if SKIP_EXTENSIONS.search(parsed.path):
            return False
        if not self.include_blog and re.match(r"^/blog/.+", parsed.path):
            return False
        return self.robots.can_fetch(USER_AGENT, url)

    def crawl(self) -> list[WebPage]:
        self._load_robots()
        root = f"{urlparse(self.start_url).scheme}://{self.host}"
        seeds = [self.start_url] + _sitemap_urls(self.client, f"{root}/sitemap.xml")

        queue = deque(normalise_url(u) for u in seeds)
        seen: set[str] = set()
        pages: list[WebPage] = []
        while queue and len(pages) < self.max_pages:
            url = queue.popleft()
            if url in seen or not self.should_visit(url):
                continue
            seen.add(url)
            try:
                response = self.client.get(url)
            except httpx.HTTPError as exc:
                logger.warning("Failed to fetch %s: %s", url, exc)
                continue
            final_url = normalise_url(str(response.url))
            content_type = response.headers.get("content-type", "")
            if response.status_code != 200 or "text/html" not in content_type:
                continue
            if final_url != url and final_url in {p.url for p in pages}:
                continue  # redirected to a page we already have
            soup = BeautifulSoup(response.text, "html.parser")
            links = extract_links(soup, final_url)  # collect links before chrome is removed
            title, text = extract_content(soup)
            if text:
                pages.append(WebPage(url=final_url, title=title, text=text))
                logger.info("Crawled %s (%d chars)", final_url, len(text))
            for link in links:
                link = normalise_url(link)
                if link not in seen:
                    queue.append(link)
            time.sleep(self.delay_seconds)
        return remove_boilerplate(pages)


def save_pages(pages: list[WebPage], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in pages], indent=2, ensure_ascii=False), encoding="utf-8")


def load_pages(path: str | Path) -> list[WebPage]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Website snapshot not found at {path}. Run: python -m scripts.crawl")
    return [WebPage(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
