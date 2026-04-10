from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from typing import Any

import requests

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .schemas import AcademicQueryPlan, NormalizedDocument
    from .ssh_service import (
        build_ssh_service_config,
        discover_openai_model,
        ensure_ssh_openai_base_url,
        is_ssh_tunnel_enabled,
    )
except ImportError:
    from schemas import AcademicQueryPlan, NormalizedDocument
    from ssh_service import (
        build_ssh_service_config,
        discover_openai_model,
        ensure_ssh_openai_base_url,
        is_ssh_tunnel_enabled,
    )


MD_OUTPUT_FOLDER = "./md"
BM25_TOP_K = 15
DENSE_TOP_K = 15
RRF_TOP_K = 20
FINAL_TOP_K = 5
RRF_K = 60
MIN_INDEX_CHARS = 20
SEMANTIC_PRECHUNK_SIZE = 6000
SEMANTIC_PRECHUNK_OVERLAP = 600


class VLLMOpenAIEmbeddings(Embeddings):
    """Minimal OpenAI-compatible embeddings client for local vLLM services."""

    def __init__(self, *, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 60.0):
        self.base_url = str(base_url or "").rstrip("/")
        self.model = str(model or "").strip()
        self.api_key = str(api_key or "EMPTY").strip() or "EMPTY"
        self.timeout = float(timeout)

    def _post_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not self.base_url:
            raise ValueError("Embedding base_url is empty.")
        if not self.model:
            raise ValueError("Embedding model is empty.")

        url = f"{self.base_url}/embeddings"
        resp = requests.post(
            url,
            timeout=self.timeout,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "input": texts,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or []
        vectors = [item.get("embedding") for item in data if isinstance(item, dict) and item.get("embedding")]
        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding response count mismatch: expected {len(texts)}, got {len(vectors)}."
            )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cleaned = [str(text or "") for text in texts]
        return self._post_embeddings(cleaned)

    def embed_query(self, text: str) -> list[float]:
        return self._post_embeddings([str(text or "")])[0]

    def __call__(self, text: str) -> list[float]:
        """Backward-compatible callable path for vector stores expecting a function."""
        return self.embed_query(text)


class RAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None

    def _resolve_embedding_ssh_config(self) -> dict[str, Any]:
        return build_ssh_service_config(
            "embedding",
            default_remote_port=int(os.getenv("RAG_EMBEDDING_REMOTE_PORT", "8000")),
            default_local_port=int(os.getenv("RAG_EMBEDDING_LOCAL_PORT", "18000")),
        )

    def _resolve_embedding_base_url(self) -> str:
        explicit_base_url = str(os.getenv("RAG_EMBEDDING_BASE_URL") or "").strip().rstrip("/")
        if explicit_base_url:
            return explicit_base_url

        ssh_config = self._resolve_embedding_ssh_config()
        if is_ssh_tunnel_enabled(ssh_config):
            try:
                return ensure_ssh_openai_base_url("embedding", ssh_config)
            except Exception as exc:
                print(f"Failed to establish embedding SSH tunnel: {exc}")

        embedding_host = str(os.getenv("RAG_EMBEDDING_HOST") or "").strip()
        embedding_port = str(os.getenv("RAG_EMBEDDING_PORT") or "").strip()
        embedding_scheme = str(os.getenv("RAG_EMBEDDING_SCHEME", "http")).strip() or "http"
        if embedding_host and embedding_port:
            return f"{embedding_scheme}://{embedding_host}:{embedding_port}/v1"
        return ""

    def _resolve_embedding_model(self) -> str:
        configured = str(os.getenv("RAG_EMBEDDING_MODEL") or "").strip()
        if configured:
            return configured

        base_url = self._resolve_embedding_base_url()
        if not base_url:
            return ""

        discovered = discover_openai_model(
            base_url,
            api_key=str(os.getenv("RAG_EMBEDDING_API_KEY", "EMPTY")).strip() or "EMPTY",
        )
        if discovered:
            return discovered

        print("Failed to discover embedding model from service.")
        return ""

    def _resolve_embedding_model_path(self, embedding_model_path=None):
        candidates = [
            embedding_model_path,
            os.getenv("RAG_EMBEDDING_MODEL_PATH"),
            "/data/202225220617/bge-m3",
            "/data/202225220617/bge-m3/",
            "/root/.cache/modelscope/hub/models/BAAI/bge-m3",
        ]
        for candidate in candidates:
            path = str(candidate or "").strip()
            if path and os.path.exists(path):
                return path
        return str(candidates[-1] or "").strip()

    def setup_embeddings(self, embedding_model_path=None):
        """Load the embedding model used by the retriever, preferring the remote service."""
        print("Loading embedding model...")
        embedding_base_url = self._resolve_embedding_base_url()
        embedding_service_model = self._resolve_embedding_model()
        embedding_api_key = str(os.getenv("RAG_EMBEDDING_API_KEY", "EMPTY")).strip() or "EMPTY"

        if embedding_base_url and embedding_service_model:
            try:
                self.embeddings = VLLMOpenAIEmbeddings(
                    base_url=embedding_base_url,
                    model=embedding_service_model,
                    api_key=embedding_api_key,
                )
                test_emb = self.embeddings.embed_documents(["test"])
                if test_emb and len(test_emb[0]) > 0:
                    print(
                        f"Embedding service connected: {embedding_base_url} model={embedding_service_model}."
                    )
                    return
            except Exception as exc:
                print(f"Failed to use embedding service, fallback to local model path: {exc}")

        resolved_model_path = self._resolve_embedding_model_path(embedding_model_path)
        embedding_device = str(os.getenv("RAG_EMBEDDING_DEVICE", "cuda:0")).strip() or "cuda:0"

        try:
            if resolved_model_path and os.path.exists(resolved_model_path):
                from langchain_huggingface import HuggingFaceEmbeddings

                self.embeddings = HuggingFaceEmbeddings(
                    model_name=resolved_model_path,
                    model_kwargs={
                        "device": embedding_device,
                        "trust_remote_code": True,
                    },
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": 2,
                    },
                )
                test_emb = self.embeddings.embed_documents(["test"])
                if test_emb and len(test_emb[0]) > 0:
                    print(f"Embedding model loaded from {resolved_model_path} on {embedding_device}.")
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

        md_files = sorted(glob.glob(os.path.join(MD_OUTPUT_FOLDER, "*.md")))
        if not md_files:
            print(f"No markdown files found in {MD_OUTPUT_FOLDER}")
            return []

        documents = []
        for md_file in md_files:
            try:
                loader = TextLoader(md_file, encoding="utf-8")
                loaded = loader.load()
                sidecar_metadata = self._load_md_sidecar_metadata(md_file)
                md_filename = os.path.basename(md_file)
                fallback_title = str(
                    sidecar_metadata.get("title") or os.path.splitext(md_filename)[0]
                ).strip()
                fallback_url = str(
                    sidecar_metadata.get("url") or sidecar_metadata.get("pdf_link") or ""
                ).strip()
                fallback_pdf_link = str(
                    sidecar_metadata.get("pdf_link") or fallback_url
                ).strip()
                source_file = str(
                    sidecar_metadata.get("source_file") or md_filename
                ).strip() or md_filename
                for doc in loaded:
                    metadata = dict(doc.metadata or {})
                    metadata.update(sidecar_metadata)
                    metadata["title"] = fallback_title
                    metadata["url"] = fallback_url
                    metadata["pdf_link"] = fallback_pdf_link
                    metadata["source_file"] = source_file
                    metadata["origin"] = str(
                        sidecar_metadata.get("origin") or metadata.get("origin") or "local_kb"
                    ).strip() or "local_kb"
                    metadata["source"] = fallback_title or md_filename
                    doc.metadata = metadata
                documents.extend(loaded)
                print(f"Loaded {os.path.basename(md_file)}")
            except Exception as exc:
                print(f"Failed to load {md_file}: {exc}")

        return documents

    def _metadata_sidecar_path(self, path: str) -> str:
        stem, _ = os.path.splitext(path)
        return f"{stem}.metadata.json"

    def _load_md_sidecar_metadata(self, md_file: str) -> dict[str, Any]:
        sidecar_path = self._metadata_sidecar_path(md_file)
        if not os.path.exists(sidecar_path):
            return {}
        try:
            with open(sidecar_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception as exc:
            print(f"Failed to load metadata sidecar {sidecar_path}: {exc}")
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): value for key, value in payload.items()}

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

        word_count = len(text.split())
        cjk_char_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        return word_count >= 5 or cjk_char_count >= 10

    def _min_index_chars(self) -> int:
        configured = str(os.getenv("RAG_MIN_CHUNK_CHARS", "")).strip()
        if configured.isdigit():
            return max(1, int(configured))
        return MIN_INDEX_CHARS

    def _collect_indexable_docs(self, documents, cleaner):
        min_chars = self._min_index_chars()
        processed_docs = []
        for doc in documents:
            content = cleaner(doc.page_content)
            if not content or len(content.strip()) < min_chars:
                continue
            if not self._is_academic_content(content):
                continue
            doc.page_content = content
            processed_docs.append(doc)
        return processed_docs

    def _prepare_semantic_chunk_inputs(self, documents, cleaner):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=SEMANTIC_PRECHUNK_SIZE,
            chunk_overlap=SEMANTIC_PRECHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
        )
        prechunked_docs = splitter.split_documents(documents)
        return self._collect_indexable_docs(prechunked_docs, cleaner)

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
        min_chars = self._min_index_chars()
        for text in texts:
            content = clean_text_content(text.page_content)
            if content and len(content.strip()) >= min_chars and self._is_academic_content(content):
                text.page_content = content
                processed_texts.append(text)

        print(f"Fallback chunking finished with {len(processed_texts)} chunks.")
        if not processed_texts:
            processed_texts = self._collect_indexable_docs(documents, clean_text_content)
            print(f"Fallback direct-doc indexing prepared {len(processed_texts)} documents.")
        if not processed_texts:
            print("No indexable chunks were produced from ./md. Expand the markdown content and retry.")
            return None

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
        min_chars = self._min_index_chars()

        try:
            semantic_input_docs = self._prepare_semantic_chunk_inputs(documents, clean_arxiv_content)
            print(f"Semantic pre-chunking prepared {len(semantic_input_docs)} docs.")
            if not semantic_input_docs:
                print("No semantic pre-chunks were produced from ./md. Expand the markdown content and retry.")
                return None

            text_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,
            )
            texts = text_splitter.split_documents(semantic_input_docs)

            processed_texts = []
            for text in texts:
                content = clean_arxiv_content(text.page_content)
                if not content or len(content.strip()) < min_chars:
                    continue

                if len(content) > 1200:
                    secondary_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=150,
                        separators=["\n", ". ", "? ", "! ", "; ", ", ", " "],
                    )
                    sub_chunks = secondary_splitter.split_text(content)
                    for sub_chunk in sub_chunks:
                        if len(sub_chunk.strip()) >= min_chars and self._is_academic_content(sub_chunk):
                            sub_doc = text.copy()
                            sub_doc.page_content = sub_chunk
                            processed_texts.append(sub_doc)
                elif self._is_academic_content(content):
                    text.page_content = content
                    processed_texts.append(text)

            print(f"Semantic chunking finished with {len(processed_texts)} chunks.")
            if not processed_texts:
                processed_texts = self._collect_indexable_docs(documents, clean_arxiv_content)
                print(f"Semantic direct-doc indexing prepared {len(processed_texts)} documents.")
            if not processed_texts:
                print("No indexable chunks were produced from ./md. Expand the markdown content and retry.")
                return None

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
        title = str(metadata.get("title") or metadata.get("source") or "").strip()
        url = str(metadata.get("url") or metadata.get("pdf_link") or "").strip()
        origin = str(metadata.get("origin") or "local_kb").strip() or "local_kb"
        source = (
            str(metadata.get("source", "")).strip()
            or title
            or url
            or "unknown"
        )
        metadata["title"] = title or source
        metadata["url"] = url
        metadata["origin"] = origin
        content = str(getattr(doc, "page_content", "")).strip()
        return NormalizedDocument(
            content=content,
            source=source,
            score=None,
            title=title or source,
            url=url,
            origin=origin,
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
