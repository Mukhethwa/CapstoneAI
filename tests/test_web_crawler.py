import httpx
import pytest

from app.web_crawler import (
    WebPage,
    WebsiteCrawler,
    clean_lines,
    extract_content,
    load_pages,
    normalise_url,
    remove_boilerplate,
    save_pages,
)

PAGE = """
<html><head><title>Fullstack AI Engineer | Zaio</title><script>var x = 1;</script></head>
<body>
  <header><a href="/">Home</a><a href="/programmes">Programmes</a></header>
  <nav>Schools Bootcamps Login</nav>
  <div class="announcement-bar">2027 Applications are now open</div>
  <main>
    <header><h1>Fullstack AI Engineer</h1></header>
    <p>A seven-month   bootcamp.</p>
    <div aria-hidden="true">Decorative dashboard mock-up</div>
    <p>Pay upfront R42,950</p>
    <p>Pay upfront R42,950</p>
    <svg><text>icon</text></svg>
    <a href="/refund-policy">Refund policy</a>
  </main>
  <footer>Copyright Zaio 2026</footer>
</body></html>
"""


def test_extract_content_keeps_main_content_and_drops_chrome():
    title, text = extract_content(PAGE)
    assert title == "Fullstack AI Engineer"
    assert "Fullstack AI Engineer" in text  # header inside <main> is kept
    assert "A seven-month bootcamp." in text  # whitespace normalised
    assert text.count("Pay upfront R42,950") == 1  # consecutive duplicates removed
    for chrome in ["Programmes", "Schools", "Applications are now open",
                   "Decorative", "Copyright", "var x", "icon"]:
        assert chrome not in text


def test_extract_content_keeps_each_card_on_one_line():
    html = """<html><body><main>
      <article><h3>Cloud &amp; DevOps</h3><p>Certificate of completion</p><span>7 months</span></article>
      <article><h3>Fullstack Web Development</h3><p>NQF Level 4</p><span>6 months</span></article>
      <ul><li>Short item</li></ul>
    </main></body></html>"""
    _, text = extract_content(html)
    lines = text.splitlines()
    assert "Cloud & DevOps · Certificate of completion · 7 months" in lines
    assert "Fullstack Web Development · NQF Level 4 · 6 months" in lines
    assert "Short item" in lines  # single-part items are left alone


def test_clean_lines_drops_tiny_fragments():
    assert clean_lines("  a \n\nHello   world\n⁠\nok\n") == "Hello world"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://WWW.zaio.io/bootcamps/", "https://www.zaio.io/bootcamps"),
        ("https://www.zaio.io/intakes?ref=nav#top", "https://www.zaio.io/intakes"),
        ("https://www.zaio.io", "https://www.zaio.io/"),
    ],
)
def test_normalise_url(url, expected):
    assert normalise_url(url) == expected


def test_remove_boilerplate_strips_lines_shared_by_most_pages():
    pages = [WebPage(f"https://x.io/{i}", f"P{i}", f"Speak to an advisor\nUnique line {i}")
             for i in range(5)]
    cleaned = remove_boilerplate(pages)
    assert all(p.text == f"Unique line {i}" for i, p in enumerate(cleaned))


def test_save_and_load_pages_round_trip(tmp_path):
    pages = [WebPage("https://x.io/a", "A", "Text a")]
    save_pages(pages, tmp_path / "pages.json")
    assert load_pages(tmp_path / "pages.json") == pages


def test_load_pages_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pages(tmp_path / "missing.json")


def _mock_site(handler_pages: dict[str, str], robots: str = "User-agent: *\nAllow: /") -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text=robots)
        if path == "/sitemap.xml":
            return httpx.Response(200, text="<urlset><url><loc>https://site.test/about</loc></url></urlset>")
        if path in handler_pages:
            return httpx.Response(200, text=handler_pages[path], headers={"content-type": "text/html"})
        return httpx.Response(404, text="not found")

    return httpx.Client(transport=httpx.MockTransport(handler))


def _html(body: str) -> str:
    return f"<html><head><title>T</title></head><body><main>{body}</main></body></html>"


def test_crawler_follows_internal_links_and_respects_rules():
    site = {
        "/": _html('<p>Home page text</p><a href="/courses">c</a><a href="/admin">a</a>'
                   '<a href="https://other.test/x">ext</a><a href="/blog/post-1">b</a>'
                   '<a href="/brochure.pdf">pdf</a><a href="/prospectus/print">print</a>'),
        "/prospectus/print": _html("<p>Print edition duplicate</p>"),
        "/about": _html("<p>About us text</p>"),
        "/courses": _html("<p>Courses text</p>"),
        "/admin": _html("<p>Secret admin</p>"),
        "/blog/post-1": _html("<p>Blog post</p>"),
    }
    crawler = WebsiteCrawler("https://site.test/", delay_seconds=0,
                             client=_mock_site(site, "User-agent: *\nDisallow: /admin"))
    urls = {p.url for p in crawler.crawl()}
    assert urls == {"https://site.test/", "https://site.test/about", "https://site.test/courses"}


def test_crawler_stops_at_max_pages():
    site = {"/": _html('<p>Home text</p><a href="/courses">c</a>'),
            "/about": _html("<p>About text</p>"), "/courses": _html("<p>Courses text</p>")}
    crawler = WebsiteCrawler("https://site.test/", max_pages=2, delay_seconds=0,
                             client=_mock_site(site))
    assert len(crawler.crawl()) == 2


def test_crawler_can_include_blog_posts():
    site = {"/": _html('<p>Home text</p><a href="/blog/post-1">b</a>'),
            "/blog/post-1": _html("<p>Blog post text</p>")}
    crawler = WebsiteCrawler("https://site.test/", include_blog=True, delay_seconds=0,
                             client=_mock_site(site))
    assert "https://site.test/blog/post-1" in {p.url for p in crawler.crawl()}
