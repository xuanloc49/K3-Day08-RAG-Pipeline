"""
RAG Chatbot — University Services (Starter Template)
Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task10_generation import generate_with_citation

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIDEBAR — INFO & SETTINGS
# =============================================================================

with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption("Trợ lý hỏi đáp về dịch vụ và chính sách đại học (học phí, học bổng, ký túc xá, thư viện)")

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    suggestions = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Điều kiện xin học bổng Academic Achievement?",
        "Dịch vụ hỗ trợ chỗ ở cho sinh viên như thế nào?",
        "Cách đăng ký học phần qua myRMIT?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:20]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.subheader("⚙️ Cấu Hình Retrieval (Demo Mode)")
    
    use_semantic = st.toggle("🧠 Semantic Search (Dense)", value=True, help="Tìm kiếm ngữ nghĩa theo Vector Embeddings")
    use_bm25 = st.toggle("🔍 BM25 Search (Lexical)", value=True, help="Tìm kiếm từ khóa chính xác bằng thuật toán BM25")
    use_rerank = st.toggle("⚡ Reranking (RRF / Ranker)", value=True, help="Tái sắp xếp thứ hạng kết quả từ các phương pháp tìm kiếm")
    
    top_k = st.slider("Số chunks retrieval (top_k)", 1, 10, 5)

    st.divider()
    st.caption("**Trạng thái Pipeline:**")
    active_modes = []
    if use_semantic:
        active_modes.append("Dense")
    if use_bm25:
        active_modes.append("BM25")
    mode_str = " + ".join(active_modes) if active_modes else "Vectorless Only"
    if use_rerank and active_modes:
        mode_str += " → Rerank"
    st.info(f"📍 Mode hiện tại: **{mode_str}**")

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🎓 University Services RAG Chatbot")
st.caption("Hệ thống hỏi đáp thông tin dịch vụ đại học (Học phí, Học bổng, Ký túc xá, Thư viện)")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            retrieval_src = msg.get("retrieval_source", "")
            header = f"📚 Nguồn tham khảo ({len(msg['sources'])} chunks" + (f" | Mode: {retrieval_src}" if retrieval_src else "") + ")"
            with st.expander(header):
                for i, src in enumerate(msg["sources"], 1):
                    meta = src.get("metadata", {})
                    source_name = meta.get("source", "Unknown")
                    doc_type = meta.get("type", "unknown")
                    score = src.get("score", 0)
                    chunk_source = src.get("source", retrieval_src)
                    st.markdown(f"**[{i}] {source_name}** `{doc_type}` | score: `{score:.4f}` | search: `{chunk_source}`")
                    st.text(src.get("content", "")[:300] + "...")
                    st.divider()

# =============================================================================
# QUERY HANDLING
# =============================================================================

# Xử lý khi bấm nút gợi ý hoặc nhập câu hỏi mới
user_input = st.chat_input("Nhập câu hỏi của bạn về chính sách/dịch vụ đại học...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    # Hiển thị câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Sinh câu trả lời từ RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            retrieval_source = mode_str
            try:
                from src.task10_generation import generate_with_citation
                response = generate_with_citation(
                    query,
                    top_k=top_k,
                    use_semantic=use_semantic,
                    use_bm25=use_bm25,
                    use_rerank=use_rerank,
                )
                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", [])
                retrieval_source = response.get("retrieval_source", mode_str)

            except NotImplementedError as e:
                answer = f"⚠️ **Pipeline chưa sẵn sàng:** {e}. Kiểm tra Task 5-9 (retrieval) đã implement chưa."
                sources = []
            except Exception as e:
                answer = f"❌ **Lỗi khi chạy RAG Pipeline:** {e}"
                sources = []

            st.markdown(answer)

            if sources:
                header = f"📚 Nguồn tham khảo ({len(sources)} chunks | Mode: {retrieval_source})"
                with st.expander(header):
                    for i, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", "Unknown")
                        doc_type = meta.get("type", "unknown")
                        score = src.get("score", 0)
                        chunk_source = src.get("source", retrieval_source)
                        st.markdown(f"**[{i}] {source_name}** `{doc_type}` | score: `{score:.4f}` | search: `{chunk_source}`")
                        st.text(src.get("content", "")[:300] + "...")
                        st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })

