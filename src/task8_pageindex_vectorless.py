"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document (cây mục lục) thay vì embedding.

Cài đặt:
    pip install pageindex fpdf2

SDK thật (kiểm tra trực tiếp qua `inspect` trên package đã cài, KHÔNG đoán từ code mẫu cũ):
    PageIndexClient(api_key).submit_document(pdf_path)   -> {"doc_id": ...}
    PageIndexClient(api_key).is_retrieval_ready(doc_id)   -> bool
    PageIndexClient(api_key).submit_query(doc_id, query)  -> {"retrieval_id": ...}
    PageIndexClient(api_key).get_retrieval(retrieval_id)  -> {
        "status": "completed", "retrieved_nodes": [
            {"title": ..., "node_id": ..., "relevant_contents": [
                {"page_index": ..., "relevant_content": ...}, ...
            ]}, ...
        ]
    }

Lưu ý: PageIndex chỉ nhận PDF, không nhận .md trực tiếp — convert markdown sang PDF
đơn giản bằng fpdf2 trước khi upload. Cần font Unicode (không dùng core font "helvetica"
của fpdf2 vì không hỗ trợ dấu tiếng Việt, sẽ lỗi khi gặp ký tự ngoài Latin-1).
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PDF_STAGING_DIR = Path(__file__).parent.parent / "data" / "_pageindex_pdf"
DOC_IDS_FILE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"

# Số document tối đa sẽ query mỗi lần search (tránh gọi API quá nhiều lần cho 1 câu hỏi)
MAX_DOCS_PER_QUERY = 5
POLL_INTERVAL_SECONDS = 3
POLL_MAX_ATTEMPTS = 40  # ~2 phút chờ tối đa mỗi bước (upload hoặc retrieval)

# Font Unicode để fpdf2 render được tiếng Việt có dấu — thử vài đường dẫn phổ biến
# theo từng hệ điều hành, cho phép override qua biến môi trường PAGEINDEX_PDF_FONT_PATH.
_UNICODE_FONT_CANDIDATES = [
    os.getenv("PAGEINDEX_PDF_FONT_PATH", ""),
    "/Library/Fonts/Arial Unicode.ttf",                       # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",   # macOS (một số bản)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",        # Linux (fonts-dejavu-core)
    "C:\\Windows\\Fonts\\arial.ttf",                          # Windows
]


def _find_unicode_font() -> str:
    for path in _UNICODE_FONT_CANDIDATES:
        if path and Path(path).exists():
            return path
    raise RuntimeError(
        "Không tìm thấy font Unicode hỗ trợ tiếng Việt để tạo PDF. "
        "Cài fonts-dejavu-core (Linux) hoặc set biến môi trường "
        "PAGEINDEX_PDF_FONT_PATH trỏ tới 1 file .ttf hỗ trợ Unicode."
    )


def _markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    """Convert 1 file markdown sang PDF đơn giản (chỉ lấy text thô) để upload PageIndex."""
    from fpdf import FPDF

    text = md_path.read_text(encoding="utf-8")
    font_path = _find_unicode_font()

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Unicode", "", font_path)
    pdf.set_font("Unicode", size=11)
    pdf.multi_cell(0, 6, text)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))


def upload_documents() -> dict:
    """
    Convert markdown -> PDF rồi upload toàn bộ lên PageIndex.
    Lưu mapping {filename: doc_id} vào DOC_IDS_FILE để pageindex_search() dùng lại,
    tránh phải upload lại mỗi lần search (upload + xử lý cây mục lục tốn thời gian).

    Returns:
        dict {filename: doc_id}
    """
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env — đăng ký tại https://pageindex.ai/")

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_ids = {}

    md_files = list(STANDARDIZED_DIR.rglob("*.md"))
    for md_file in md_files:
        pdf_path = PDF_STAGING_DIR / (md_file.stem + ".pdf")
        _markdown_to_pdf(md_file, pdf_path)

        resp = client.submit_document(str(pdf_path))
        doc_id = resp["doc_id"]
        print(f"  ⏳ Uploaded: {md_file.name} -> {doc_id}, đang chờ xử lý cây mục lục...")

        # Đợi PageIndex xử lý xong (tree generation + OCR) trước khi có thể query
        for _ in range(POLL_MAX_ATTEMPTS):
            if client.is_retrieval_ready(doc_id):
                break
            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            print(f"  ⚠ {md_file.name}: chưa sẵn sàng sau {POLL_MAX_ATTEMPTS * POLL_INTERVAL_SECONDS}s, vẫn lưu doc_id để dùng sau")

        doc_ids[md_file.name] = doc_id
        print(f"  ✓ Sẵn sàng: {md_file.name} -> {doc_id}")

    DOC_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOC_IDS_FILE.write_text(json.dumps(doc_ids, ensure_ascii=False, indent=2))
    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    from pageindex import PageIndexClient

    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong .env — đăng ký tại https://pageindex.ai/")

    if not DOC_IDS_FILE.exists():
        raise RuntimeError(
            "Chưa có document nào trên PageIndex — chạy upload_documents() "
            "(hoặc `python -m src.task8_pageindex_vectorless`) trước."
        )

    doc_ids = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    results = []
    # Query từng document (API PageIndex retrieval nhận 1 doc_id/lần) rồi gộp lại,
    # giới hạn MAX_DOCS_PER_QUERY để không gọi API quá nhiều cho 1 câu hỏi.
    for filename, doc_id in list(doc_ids.items())[:MAX_DOCS_PER_QUERY]:
        resp = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = resp["retrieval_id"]

        retrieval = None
        for _ in range(POLL_MAX_ATTEMPTS):
            retrieval = client.get_retrieval(retrieval_id)
            if retrieval.get("status") == "completed":
                break
            time.sleep(POLL_INTERVAL_SECONDS)

        if not retrieval or retrieval.get("status") != "completed":
            continue

        for node in retrieval.get("retrieved_nodes", []):
            for item in node.get("relevant_contents", []):
                results.append({
                    "content": item.get("relevant_content", ""),
                    "metadata": {
                        "source": filename,
                        "section": node.get("title"),
                        "page_index": item.get("page_index"),
                    },
                    "source": "pageindex",
                })

    # PageIndex không trả điểm số trực tiếp — tự gán theo thứ hạng xuất hiện
    # (kết quả đầu tiên của mỗi doc coi là liên quan nhất trong doc đó).
    for i, r in enumerate(results):
        r["score"] = round(max(0.0, 1.0 - i * 0.05), 4)

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
