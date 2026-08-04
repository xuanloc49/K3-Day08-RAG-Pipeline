"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Nguồn: trang công khai RMIT Vietnam, phủ đủ 4 chủ đề gợi ý
# (sự kiện, thư viện, hỗ trợ sinh viên, học bổng)
ARTICLE_URLS = [
    ("https://www.rmit.edu.vn/libraryvn/about-us/library-events/2026/rmit-library-seminar-2026", "library-seminar-2026-rmit"),
    ("https://www.rmit.edu.vn/libraryvn/student-support", "library-student-support-rmit"),
    ("https://www.rmit.edu.vn/students/student-news-and-events/student-events-2026/orientation-week-sem-2", "orientation-week-sem2-2026-rmit"),
    ("https://www.rmit.edu.vn/news/all-news/2026/jan/rmit-vietnam-announces-record-2026-scholarships-worth-more-than-200-billion-vnd", "scholarships-record-2026-rmit"),
    ("https://www.rmit.edu.vn/students/student-news-and-events/student-news/2026/newbie-101-unlock-library-power", "newbie-101-library-power-rmit"),
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        return {
            "url": url,
            "title": result.metadata.get("title", "Unknown"),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown,
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, (url, slug) in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as e:
            print(f"  ✗ Lỗi crawl {url}: {e}")
            continue

        # Lưu file JSON
        filename = f"{slug}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath} ({len(article['content_markdown'])} chars)")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
