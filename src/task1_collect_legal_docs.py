"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang công khai của một trường đại học.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý văn bản (chủ đề dịch vụ đại học):
    - Học phí & phương thức thanh toán (Tuition Fees)
    - Chính sách học bổng (Scholarship eligibility)
    - Quy định ký túc xá / văn hóa học đường
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def verify_collected_files():
    """Kiểm tra và in danh sách các file pháp luật đã thu thập."""
    valid_extensions = {".pdf", ".docx", ".doc"}
    files = [f for f in DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    
    print(f"\n--- Đã tìm thấy {len(files)} văn bản pháp luật trong {DATA_DIR} ---")
    for idx, f in enumerate(files, 1):
        size_kb = f.stat().st_size / 1024
        print(f"  {idx}. {f.name} ({size_kb:.2f} KB)")
        
    if len(files) >= 3:
        print("✓ Đạt yêu cầu Task 1 (Tối thiểu 3 văn bản).")
    else:
        print(f"✗ Chưa đạt: Cần tối thiểu 3 file (Hiện có {len(files)}).")


if __name__ == "__main__":
    setup_directory()
    verify_collected_files()
