from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from typing import Any

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .schemas import AcademicQueryPlan, NormalizedDocument
except ImportError:
    from schemas import AcademicQueryPlan, NormalizedDocument


MD_OUTPUT_FOLDER = "./md"
BM25_TOP_K = 15
DENSE_TOP_K = 15
RRF_TOP_K = 20
FINAL_TOP_K = 5
RRF_K = 60


class RAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None

    def setup_embeddings(self, embedding_model_path="/root/.cache/modelscope/hub/models/BAAI/bge-m3"):
        """Load the embedding model used by the local retriever."""
        print("Loading embedding model...")

        try:
            if os.path.exists(embedding_model_path):
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_path,
                    model_kwargs={
                        "device": "cuda:1",
                        "trust_remote_code": True,
                    },
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": 2,
                    },
                )
                test_emb = self.embeddings.embed_documents(["test"])
                if test_emb and len(test_emb[0]) > 0:
                    print("Embedding model loaded.")
                    return

            print("Embedding model not found, falling back to FakeEmbeddings.")
            self.embeddings = FakeEmbeddings(size=384)

        except Exception as exc:
            print(f"Failed to load embeddings: {exc}")
            self.embeddings = FakeEmbeddings(size=384)

    def load_md_documents(self):
        """Load markdown documents from the indexed md directory."""
        print("Loading markdown documents...")

        if not os.path.exists(MD_OUTPUT_FOLDER):
            print(f"Markdown directory does not exist: {MD_OUTPUT_FOLDER}")
            return []

        md_files = glob.glob(os.path.join(MD_OUTPUT_FOLDER, "*.md"))
        if not md_files:
            print(f"No markdown files found in {MD_OUTPUT_FOLDER}")
            return []

        documents = []
        for md_file in md_files:
            try:
                loader = TextLoader(md_file, encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = os.path.basename(md_file)
                documents.extend(loaded)
                print(f"Loaded {os.path.basename(md_file)}")
            except Exception as exc:
                print(f"Failed to load {md_file}: {exc}")

        return documents

    def retriever_vector_store(self):
        """Load the existing FAISS index if it exists."""
        if not os.path.exists("./faiss"):
            return

        try:
            from langchain_community.vectorstores import FAISS

            self.vectorstore = FAISS.load_local(
                "./faiss",
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            print(f"Failed to load FAISS, index will need rebuilding: {exc}")

    def _is_academic_content(self, text):
        """Filter obvious noise blocks before indexing."""
        exclusion_patterns = [
            r"^\d+$",
            r"^[ivxlcdm]+$",
            r"^(abstract|keywords|references?)$",
            r"^.*\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec).*\d{4}.*$",
        ]

        for pattern in exclusion_patterns:
            if re.match(pattern, text.lower().strip()):
                return False

        return len(text.split()) >= 5

    def setup_vector_store_optimized_fallback(self, documents):
        """Fallback chunking strategy for rebuilding the FAISS index."""

        def clean_text_content(text):
            if not text or not text.strip():
                return ""
            cleaned = re.sub(r"\s+", " ", text.strip())
            cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned)
            return str(cleaned) if cleaned else ""

        for doc in documents:
            doc.page_content = clean_text_content(doc.page_content)

        documents = [doc for doc in documents if doc.page_content.strip()]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
        )

        texts = text_splitter.split_documents(documents)
        processed_texts = []
        for text in texts:
            content = clean_text_content(text.page_content)
            if content and len(content.strip()) >= 50 and self._is_academic_content(content):
                text.page_content = content
                processed_texts.append(text)

        print(f"Fallback chunking finished with {len(processed_texts)} chunks.")

        try:
            from langchain_community.vectorstores import FAISS

            vectorstore = FAISS.from_documents(processed_texts, self.embeddings)
            vectorstore.save_local("./faiss")
            print("FAISS index rebuilt.")
            return vectorstore
        except Exception as exc:
            print(f"Failed to build FAISS index: {exc}")
            return None

    def setup_vector_store_semantic_arxiv(self, documents):
        """Semantic chunking strategy used for arXiv-style academic papers."""
        if not documents:
            print("No documents available for semantic chunking.")
            return None

        def clean_arxiv_content(text):
            if not text or not text.strip():
                return ""

            cleaned = re.sub(r"\s+", " ", text.strip())
            cleaned = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", lambda m: m.group(0).replace("\n", " "), cleaned)
            cleaned = re.sub(r"\$\$.*?\$\$", lambda m: m.group(0).replace("\n", " "), cleaned)
            cleaned = re.sub(r"\$.*?\$", lambda m: m.group(0).replace("\n", " "), cleaned)
            cleaned = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", cleaned)
            return str(cleaned) if cleaned else ""

        for doc in documents:
            doc.page_content = clean_arxiv_content(doc.page_content)

        documents = [doc for doc in documents if doc.page_content.strip()]

        try:
            text_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
            )
            texts = text_splitter.split_documents(documents)

            processed_texts = []
            for text in texts:
                content = clean_arxiv_content(text.page_content)
                if not content or len(content.strip()) < 50:
                    continue

                if len(content) > 1200:
                    secondary_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=150,
                        separators=["\n", ". ", "? ", "! ", "; ", ", ", " "],
                    )
                    sub_chunks = secondary_splitter.split_text(content)
                    for sub_chunk in sub_chunks:
                        if len(sub_chunk.strip()) >= 50 and self._is_academic_content(sub_chunk):
                            sub_doc = text.copy()
                            sub_doc.page_content = sub_chunk
                            processed_texts.append(sub_doc)
                elif self._is_academic_content(content):
                    text.page_content = content
                    processed_texts.append(text)

            print(f"Semantic chunking finished with {len(processed_texts)} chunks.")

            from langchain_community.vectorstores import FAISS

            vectorstore = FAISS.from_documents(processed_texts, self.embeddings)
            vectorstore.save_local("./faiss")
            print("FAISS index rebuilt.")
            return vectorstore
        except Exception as exc:
            print(f"Semantic chunking failed, using fallback chunking: {exc}")
            return self.setup_vector_store_optimized_fallback(documents)

    def _safe_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        safe = {}
        for key, value in (metadata or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[key] = value
            elif isinstance(value, list):
                safe[key] = [str(item) for item in value]
            else:
                safe[key] = str(value)
        return safe

    def _normalize_langchain_doc(self, doc) -> NormalizedDocument:
        metadata = self._safe_metadata(getattr(doc, "metadata", {}) or {})
        source = (
            str(metadata.get("source", "")).strip()
            or str(metadata.get("title", "")).strip()
            or "unknown"
        )
        content = str(getattr(doc, "page_content", "")).strip()
        return NormalizedDocument(
            content=content,
            source=source,
            score=None,
            metadata=metadata,
        )

    def _normalize_document_key(self, source: str, content: str) -> str:
        normalized_source = re.sub(r"\s+", " ", (source or "").strip().lower())
        normalized_content = re.sub(r"\s+", " ", (content or "").strip().lower())
        return f"{normalized_source}::{normalized_content}"

    def get_all_documents_from_faiss(self):
        """Read all indexed chunks back from FAISS for BM25 construction."""
        if self.vectorstore is None:
            print("FAISS vector store is not initialized.")
            return []

        documents = []
        for _, doc in self.vectorstore.docstore._dict.items():
            documents.append(doc)

        print(f"Read {len(documents)} chunks from FAISS.")
        return documents

    def create_bm25_retriever_from_faiss(self, k=BM25_TOP_K):
        """Build BM25 over the currently indexed FAISS chunks."""
        documents = self.get_all_documents_from_faiss()
        if not documents:
            print("Cannot build BM25 retriever without indexed chunks.")
            return None

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        try:
            bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
            bm25_retriever.k = k
            print("BM25 retriever built.")
            return bm25_retriever
        except Exception as exc:
            print(f"Failed to build BM25 retriever: {exc}")
            return None

    def setup_fallback_retriever(self, k=FINAL_TOP_K):
        """Create a dense retriever for compatibility checks and fallback use."""
        if self.vectorstore is None:
            return None
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def _coerce_query_plan(self, query_plan: AcademicQueryPlan | dict[str, Any]) -> AcademicQueryPlan:
        return query_plan if isinstance(query_plan, AcademicQueryPlan) else AcademicQueryPlan(**query_plan)

    def _build_bm25_query(self, query_plan: AcademicQueryPlan | dict[str, Any]) -> str:
        plan = self._coerce_query_plan(query_plan)
        parts = [plan.retrieval_query_en, *plan.keywords_en]
        deduped = []
        seen = set()
        for part in parts:
            text = str(part or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return " ".join(deduped)

    def _run_bm25_branch(self, query_plan: AcademicQueryPlan) -> tuple[str, list[NormalizedDocument]]:
        retriever = self.create_bm25_retriever_from_faiss(k=BM25_TOP_K)
        query = self._build_bm25_query(query_plan)
        if retriever is None or not query:
            return query, []
        raw_docs = retriever.invoke(query)
        return query, [self._normalize_langchain_doc(doc) for doc in raw_docs[:BM25_TOP_K]]

    def _run_dense_branch(self, query: str, top_k: int = DENSE_TOP_K) -> list[NormalizedDocument]:
        retriever = self.setup_fallback_retriever(k=top_k)
        if retriever is None or not str(query or "").strip():
            return []
        raw_docs = retriever.invoke(query)
        return [self._normalize_langchain_doc(doc) for doc in raw_docs[:top_k]]

    def _merge_metadata(self, base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base or {})
        merged.update(extra)
        return merged

    def _fuse_with_rrf(
        self,
        branch_results: dict[str, list[NormalizedDocument]],
        *,
        rrf_k: int = RRF_K,
        rrf_top_k: int = RRF_TOP_K,
        final_top_k: int = FINAL_TOP_K,
    ) -> tuple[list[NormalizedDocument], dict[str, Any]]:
        fused_docs: dict[str, NormalizedDocument] = {}
        fused_scores: dict[str, float] = defaultdict(float)
        branch_ranks: dict[str, dict[str, int]] = defaultdict(dict)
        branch_hits: dict[str, list[str]] = defaultdict(list)

        for branch_name, docs in branch_results.items():
            for rank, doc in enumerate(docs, start=1):
                key = self._normalize_document_key(doc.source, doc.content)
                if key not in fused_docs:
                    fused_docs[key] = doc
                fused_scores[key] += 1.0 / (rrf_k + rank)
                branch_ranks[key][branch_name] = rank
                if branch_name not in branch_hits[key]:
                    branch_hits[key].append(branch_name)

        sorted_keys = sorted(
            fused_docs.keys(),
            key=lambda key: (
                -fused_scores[key],
                min(branch_ranks[key].values()) if branch_ranks[key] else 9999,
                fused_docs[key].source,
                fused_docs[key].content[:120],
            ),
        )

        pool_keys = sorted_keys[:rrf_top_k]
        final_docs = []
        for key in pool_keys[:final_top_k]:
            doc = fused_docs[key]
            score = round(fused_scores[key], 6)
            debug_metadata = {
                "branch_hits": branch_hits[key],
                "branch_ranks": branch_ranks[key],
                "rrf_score": score,
            }
            final_docs.append(
                doc.model_copy(
                    update={
                        "score": score,
                        "metadata": self._merge_metadata(
                            doc.metadata,
                            {"retrieval_debug": debug_metadata, "rrf_score": score},
                        ),
                    }
                )
            )

        debug = {
            "branch_counts": {branch: len(docs) for branch, docs in branch_results.items()},
            "rrf_pool_count": len(pool_keys),
            "returned_count": len(final_docs),
        }
        return final_docs, debug

    def retrieve_with_query_plan(
        self,
        query_plan: AcademicQueryPlan | dict[str, Any],
        *,
        final_top_k: int = FINAL_TOP_K,
    ) -> tuple[list[NormalizedDocument], dict[str, Any]]:
        plan = self._coerce_query_plan(query_plan)
        bm25_query, bm25_docs = self._run_bm25_branch(plan)
        dense_zh_docs = self._run_dense_branch(plan.retrieval_query_zh, top_k=DENSE_TOP_K)
        dense_en_docs = self._run_dense_branch(plan.retrieval_query_en, top_k=DENSE_TOP_K)

        final_docs, debug = self._fuse_with_rrf(
            {
                "bm25_en": bm25_docs,
                "dense_zh": dense_zh_docs,
                "dense_en": dense_en_docs,
            },
            final_top_k=final_top_k,
        )
        debug.update(
            {
                "bm25_query": bm25_query,
                "retrieval_query_zh": plan.retrieval_query_zh,
                "retrieval_query_en": plan.retrieval_query_en,
            }
        )
        return final_docs, debug

    def initialize(self):
        """Initialize the retriever from the existing local index."""
        self.setup_embeddings()
        self.retriever_vector_store()

        if self.vectorstore is None:
            return False

        self.retriever = self.setup_fallback_retriever(k=FINAL_TOP_K)
        return self.retriever is not None

    def update_rag_system(self, chunk_strategy="semantic_arxiv"):
        """Rebuild the FAISS index and refresh the retriever."""
        documents = self.load_md_documents()
        if chunk_strategy == "optimized":
            self.vectorstore = self.setup_vector_store_optimized_fallback(documents)
        else:
            self.vectorstore = self.setup_vector_store_semantic_arxiv(documents)

        if self.vectorstore is None:
            return False

        self.retriever = self.setup_fallback_retriever(k=FINAL_TOP_K)
        return self.retriever is not None


_global_rag_system = None


def setup_rag_system():
    """Create and cache the local retriever system."""
    global _global_rag_system
    if _global_rag_system is None:
        _global_rag_system = RAGSystem()
        success = _global_rag_system.initialize()
        if not success:
            print("RAG system initialization failed.")
            return None
    return _global_rag_system
