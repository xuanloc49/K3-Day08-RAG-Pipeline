"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# CHUNKING STRATEGY:
# - CHUNK_SIZE = 500: Đủ chứa 1 đoạn văn hoàn chỉnh (~80-120 từ), giúp giữ trọn vẹn ý nghĩa
#   của quy định/chính sách đại học mà không làm loãng thông tin embedding.
# - CHUNK_OVERLAP = 50: Chiếm 10% chunk size, đảm bảo các câu hoặc thuật ngữ nằm ở ranh giới
#   giữa 2 chunk không bị cắt đứt quãng ngữ nghĩa.
# - CHUNKING_METHOD = "recursive": Sử dụng RecursiveCharacterTextSplitter để ưu tiên cắt theo
#   xuống dòng, dấu chấm câu trước khi cắt theo từ, giữ cấu trúc tự nhiên của văn bản.
CHUNK_SIZE = 500        
CHUNK_OVERLAP = 50      
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# EMBEDDING MODEL:
# - EMBEDDING_MODEL = "BAAI/bge-m3": Model embedding đa ngôn ngữ mạnh mẽ (multilingual),
#   hỗ trợ rất tốt cho cả tiếng Việt lẫn tiếng Anh (đặc biệt phù hợp với tài liệu RMIT/Đại học).
# - EMBEDDING_DIM = 1024: Kích thước vector 1024 biểu diễn ngữ nghĩa chi tiết.
EMBEDDING_MODEL = "BAAI/bge-m3"  
EMBEDDING_DIM = 1024

# VECTOR STORE:
# - VECTOR_STORE = "chromadb": Lưu trữ vector tại local (persistent), cài đặt đơn giản,
#   không phụ thuộc vào Docker hay cloud service.
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "university_services_docs"


# Global cache cho embedding model & chroma client
_embedding_model_instance = None
_chroma_client_instance = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL):
    """
    Singleton / Lazy getter cho embedding model (SentenceTransformer).
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model_instance = SentenceTransformer(model_name)
    return _embedding_model_instance


def get_chroma_client():
    """
    Singleton / Lazy getter cho ChromaDB PersistentClient.
    """
    global _chroma_client_instance
    if _chroma_client_instance is None:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client_instance = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client_instance


def get_collection(collection_name: str = COLLECTION_NAME):
    """
    Lấy hoặc tạo ChromaDB collection với metric HNSW cosine distance.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> List[Dict[str, Any]]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.is_file():
            content = md_file.read_text(encoding="utf-8")
            relative_path_str = str(md_file.relative_to(STANDARDIZED_DIR))
            doc_type = "legal" if "legal" in relative_path_str else "news"
            documents.append({
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type
                }
            })
    return documents


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    if CHUNKING_METHOD == "recursive":
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        except ImportError:
            # Fallback nếu chưa cài langchain-text-splitters
            class SimpleSplitter:
                def __init__(self, size, overlap):
                    self.size = size
                    self.overlap = overlap
                def split_text(self, text):
                    chunks = []
                    start = 0
                    while start < len(text):
                        end = start + self.size
                        chunks.append(text[start:end])
                        start += (self.size - self.overlap)
                    return chunks
            splitter = SimpleSplitter(CHUNK_SIZE, CHUNK_OVERLAP)
            
        chunks = []
        for doc in documents:
            splits = splitter.split_text(doc["content"])
            for i, chunk_text in enumerate(splits):
                if chunk_text.strip():
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {
                            **doc["metadata"],
                            "chunk_index": i
                        }
                    })
        return chunks
    else:
        raise ValueError(f"Unsupported chunking method: {CHUNKING_METHOD}")


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    model = get_embedding_model(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        
    return chunks


def index_to_vectorstore(chunks: List[Dict[str, Any]]):
    """
    Lưu chunks vào vector store đã chọn (ChromaDB).
    """
    if not chunks:
        print("No chunks to index.")
        return

    collection = get_collection(COLLECTION_NAME)

    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    
    embeddings = None
    if "embedding" in chunks[0]:
        embeddings = [c["embedding"] for c in chunks]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
