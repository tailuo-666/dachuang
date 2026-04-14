from __future__ import annotations

import glob
import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import fitz

if TYPE_CHECKING:
    from .pdf_processor import PDFProcessor
    from .rag_system import RAGSystem


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_MD_DIR = "./md"
DEFAULT_PDF_ROOT = "./paper_results"
DEFAULT_UPLOAD_DIR = "./paper_results"


@dataclass
class PaperRecord:
    paper_id: int
    title: str
    source_file: str
    time: str
    size: int
    pages: int
    url: str = ""
    pdf_link: str = ""
    origin: str = "local_kb"
    md_path: str = ""
    metadata_path: str = ""
    stored_pdf_path: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "source_file": self.source_file,
            "time": self.time,
            "size": self.size,
            "pages": self.pages,
        }

    def to_sidecar_payload(self) -> dict[str, Any]:
        payload = {
            "paper_id": self.paper_id,
            "title": self.title,
            "url": self.url,
            "pdf_link": self.pdf_link,
            "source_file": self.source_file,
            "origin": self.origin,
            "time": self.time,
            "size": self.size,
            "pages": self.pages,
        }
        if self.stored_pdf_path:
            payload["stored_pdf_path"] = self.stored_pdf_path
        for key, value in self.extra_metadata.items():
            if key not in payload:
                payload[key] = value
        return payload


class KnowledgeBaseManager:
    def __init__(
        self,
        *,
        md_dir: str = DEFAULT_MD_DIR,
        pdf_root_dir: str = DEFAULT_PDF_ROOT,
        upload_dir: str = DEFAULT_UPLOAD_DIR,
        pdf_search_dirs: list[str] | None = None,
        rag_system: "RAGSystem" | None = None,
        pdf_processor: "PDFProcessor" | None = None,
    ) -> None:
        self.md_dir = os.path.abspath(md_dir)
        self.pdf_root_dir = os.path.abspath(pdf_root_dir)
        self.upload_dir = os.path.abspath(upload_dir)
        self.pdf_search_dirs = self._normalize_search_dirs(pdf_search_dirs)
        self.rag_system = rag_system
        self.pdf_processor = pdf_processor
        self.lock = threading.RLock()

        self.total_numbers = 0
        self.total_pages = 0
        self.total_size = 0
        self.next_paper_id = 1
        self._papers: dict[int, PaperRecord] = {}

        os.makedirs(self.md_dir, exist_ok=True)
        os.makedirs(self.pdf_root_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)

    def _normalize_search_dirs(self, pdf_search_dirs: list[str] | None) -> list[str]:
        candidates = [self.upload_dir, self.pdf_root_dir, self.md_dir, *(pdf_search_dirs or [])]
        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            resolved = os.path.abspath(candidate)
            if resolved in seen:
                continue
            seen.add(resolved)
            normalized.append(resolved)
        return normalized

    def _metadata_sidecar_path(self, md_path: str) -> str:
        stem, _ = os.path.splitext(md_path)
        return f"{stem}.metadata.json"

    def _normalize_string(self, value: Any) -> str:
        return str(value or "").strip()

    def _normalize_int(self, value: Any, *, default: int = 0) -> int:
        if isinstance(value, bool):
            return int(value)
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return default

    def _relative_path(self, path: str) -> str:
        if not path:
            return ""
        try:
            return os.path.relpath(path, os.getcwd())
        except ValueError:
            return os.path.abspath(path)

    def _safe_parse_time(self, value: Any) -> datetime | None:
        text = self._normalize_string(value)
        if not text:
            return None
        try:
            return datetime.strptime(text, TIME_FORMAT)
        except ValueError:
            return None

    def _format_timestamp(self, timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).strftime(TIME_FORMAT)

    def _build_md_filename(self, paper_id: int, source_file: str) -> str:
        stem = os.path.splitext(os.path.basename(source_file))[0]
        cleaned = re.sub(r'[<>:"/\\|?*]+', "_", stem).strip(" ._") or f"paper_{paper_id}"
        return f"paper_{paper_id}_{cleaned}.md"

    def _clean_pdf_filename(self, filename: str, *, max_len: int = 80) -> str:
        clean_name = re.sub(r'[<>:"/\\|?*]', "", filename or "")
        clean_name = re.sub(r"\s+", " ", clean_name).strip()
        if len(clean_name) > max_len:
            clean_name = clean_name[:max_len]
        return clean_name

    def build_upload_pdf_path(self, original_filename: str) -> tuple[str, str]:
        basename = os.path.basename(original_filename or "").strip()
        stem, ext = os.path.splitext(basename)
        normalized_ext = ext if ext.lower() == ".pdf" else ".pdf"
        cleaned_stem = self._clean_pdf_filename(stem) or f"paper_{self.next_paper_id}"
        stored_filename = f"{cleaned_stem}{normalized_ext}"
        return stored_filename, os.path.join(self.upload_dir, stored_filename)

    def find_paper_by_source_file(self, source_file: str) -> PaperRecord | None:
        normalized = os.path.basename(source_file or "").strip()
        if not normalized:
            return None
        for record in self._papers.values():
            if record.source_file == normalized:
                return record
        return None

    def _load_json(self, path: str) -> dict[str, Any]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception as exc:
            print(f"Failed to read metadata sidecar {path}: {exc}")
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_json(self, path: str, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _resolve_existing_pdf_path(self, payload: dict[str, Any], sidecar_path: str) -> str:
        stored_pdf_path = self._normalize_string(payload.get("stored_pdf_path"))
        if stored_pdf_path:
            candidate = stored_pdf_path
            if not os.path.isabs(candidate):
                candidate = os.path.abspath(os.path.join(os.getcwd(), candidate))
            if os.path.exists(candidate):
                return candidate

        source_file = self._normalize_string(payload.get("source_file"))
        basename = os.path.basename(source_file)
        fallback_stem = os.path.splitext(os.path.basename(sidecar_path))[0].replace(".metadata", "")

        lookup_names = [name for name in [basename, f"{fallback_stem}.pdf"] if name]
        for directory in self.pdf_search_dirs:
            for name in lookup_names:
                candidate = os.path.join(directory, name)
                if os.path.exists(candidate):
                    return candidate

        for directory in self.pdf_search_dirs:
            if not os.path.isdir(directory):
                continue
            for root, _, files in os.walk(directory):
                for filename in files:
                    if not filename.lower().endswith(".pdf"):
                        continue
                    if filename == basename:
                        return os.path.join(root, filename)
        return ""

    def _read_pdf_stats(self, pdf_path: str) -> tuple[int, int]:
        if not pdf_path or not os.path.exists(pdf_path):
            return 0, 0
        size = os.path.getsize(pdf_path)
        try:
            doc = fitz.open(pdf_path)
            try:
                pages = doc.page_count
            finally:
                doc.close()
        except Exception as exc:
            print(f"Failed to read PDF page count for {pdf_path}: {exc}")
            pages = 0
        return size, pages

    def _create_record_from_payload(
        self,
        payload: dict[str, Any],
        metadata_path: str,
        *,
        default_paper_id: int | None = None,
    ) -> tuple[PaperRecord, bool]:
        changed = False
        md_path = metadata_path.replace(".metadata.json", ".md")
        pdf_path = self._resolve_existing_pdf_path(payload, metadata_path)
        source_file = self._normalize_string(payload.get("source_file"))
        if not source_file and pdf_path:
            source_file = os.path.basename(pdf_path)
            changed = True
        if not source_file:
            source_file = os.path.basename(md_path)

        title = self._normalize_string(payload.get("title")) or os.path.splitext(source_file)[0]
        url = self._normalize_string(payload.get("url"))
        pdf_link = self._normalize_string(payload.get("pdf_link")) or url
        origin = self._normalize_string(payload.get("origin")) or "local_kb"

        time_text = self._normalize_string(payload.get("time"))
        if not time_text:
            time_text = self._format_timestamp(os.path.getmtime(metadata_path))
            changed = True

        paper_id = self._normalize_int(payload.get("paper_id"), default=0)
        if paper_id <= 0 and default_paper_id is not None:
            paper_id = default_paper_id
            changed = True

        size, pages = self._read_pdf_stats(pdf_path)
        payload_size = self._normalize_int(payload.get("size"), default=-1)
        payload_pages = self._normalize_int(payload.get("pages"), default=-1)
        if payload_size != size:
            changed = True
        if payload_pages != pages:
            changed = True

        stored_pdf_path = self._normalize_string(payload.get("stored_pdf_path"))
        if pdf_path:
            resolved_pdf_path = self._relative_path(pdf_path)
            if stored_pdf_path != resolved_pdf_path:
                stored_pdf_path = resolved_pdf_path
                changed = True

        known_keys = {
            "paper_id",
            "title",
            "url",
            "pdf_link",
            "source_file",
            "origin",
            "time",
            "size",
            "pages",
            "stored_pdf_path",
        }
        extra_metadata = {key: value for key, value in payload.items() if key not in known_keys}

        record = PaperRecord(
            paper_id=paper_id,
            title=title,
            source_file=source_file,
            time=time_text,
            size=size,
            pages=pages,
            url=url,
            pdf_link=pdf_link,
            origin=origin,
            md_path=md_path,
            metadata_path=metadata_path,
            stored_pdf_path=stored_pdf_path,
            extra_metadata=extra_metadata,
        )
        return record, changed

    def refresh_state(self, *, rebuild_if_needed: bool = True) -> dict[str, Any]:
        with self.lock:
            sidecar_paths = sorted(glob.glob(os.path.join(self.md_dir, "*.metadata.json")))
            raw_entries: list[dict[str, Any]] = []
            for metadata_path in sidecar_paths:
                payload = self._load_json(metadata_path)
                existing_paper_id = self._normalize_int(payload.get("paper_id"), default=0)
                time_value = self._safe_parse_time(payload.get("time"))
                raw_entries.append(
                    {
                        "metadata_path": metadata_path,
                        "payload": payload,
                        "existing_paper_id": existing_paper_id,
                        "time_value": time_value,
                        "mtime": os.path.getmtime(metadata_path),
                    }
                )

            max_existing_id = max((entry["existing_paper_id"] for entry in raw_entries), default=0)
            next_assigned_id = max_existing_id + 1 if max_existing_id > 0 else 1

            missing_entries = [entry for entry in raw_entries if entry["existing_paper_id"] <= 0]
            missing_entries.sort(
                key=lambda entry: (
                    entry["time_value"] or datetime.fromtimestamp(entry["mtime"]),
                    entry["metadata_path"],
                )
            )

            assigned_ids: dict[str, int] = {}
            for entry in missing_entries:
                assigned_ids[entry["metadata_path"]] = next_assigned_id
                next_assigned_id += 1

            records: dict[int, PaperRecord] = {}
            sidecars_changed = False
            for entry in raw_entries:
                metadata_path = entry["metadata_path"]
                default_paper_id = assigned_ids.get(metadata_path)
                record, changed = self._create_record_from_payload(
                    entry["payload"],
                    metadata_path,
                    default_paper_id=default_paper_id,
                )
                if changed:
                    self._write_json(metadata_path, record.to_sidecar_payload())
                    sidecars_changed = True
                records[record.paper_id] = record

            self._papers = records
            self.total_numbers = len(records)
            self.total_pages = sum(record.pages for record in records.values())
            self.total_size = sum(record.size for record in records.values())
            self.next_paper_id = (max(records.keys()) + 1) if records else 1

            rebuilt = False
            if sidecars_changed and rebuild_if_needed and self.rag_system is not None:
                rebuilt = bool(self.rag_system.update_rag_system(chunk_strategy="semantic_arxiv"))

            return {
                "paper_count": self.total_numbers,
                "sidecars_changed": sidecars_changed,
                "rebuild_triggered": rebuilt,
            }

    def get_totals(self) -> dict[str, int]:
        return {
            "total_numbers": self.total_numbers,
            "total_pages": self.total_pages,
            "total_size": self.total_size,
        }

    def list_papers(self, keyword: str | None = None) -> dict[str, Any]:
        normalized_keyword = self._normalize_string(keyword).lower()
        papers = sorted(self._papers.values(), key=lambda record: record.paper_id, reverse=True)
        if normalized_keyword:
            papers = [
                record
                for record in papers
                if normalized_keyword in record.source_file.lower()
            ]
        return {
            **self.get_totals(),
            "papers": [record.to_public_dict() for record in papers],
        }

    def _ensure_dependencies(self) -> None:
        if self.pdf_processor is None:
            raise RuntimeError("PDF processor is not initialized.")
        if self.rag_system is None:
            raise RuntimeError("RAG system is not initialized.")

    def ingest_pdf(self, pdf_path: str, *, source_file: str | None = None) -> dict[str, Any]:
        self._ensure_dependencies()
        with self.lock:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

            normalized_source_file = os.path.basename(source_file or pdf_path)
            paper_id = self.next_paper_id
            size, pages = self._read_pdf_stats(pdf_path)
            time_text = datetime.now().strftime(TIME_FORMAT)
            stored_pdf_path = self._relative_path(pdf_path)
            metadata = {
                "paper_id": paper_id,
                "title": os.path.splitext(normalized_source_file)[0],
                "url": "",
                "pdf_link": "",
                "source_file": normalized_source_file,
                "origin": "local_kb",
                "time": time_text,
                "size": size,
                "pages": pages,
                "stored_pdf_path": stored_pdf_path,
            }
            out_md = os.path.join(self.md_dir, self._build_md_filename(paper_id, normalized_source_file))

            created_paths: list[str] = []
            try:
                result = self.pdf_processor.process_pdf(
                    pdf_path,
                    out_md=out_md,
                    extra_metadata=metadata,
                )
                created_paths.extend([result["md_path"], self._metadata_sidecar_path(result["md_path"])])
                rebuild_success = bool(self.rag_system.update_rag_system(chunk_strategy="semantic_arxiv"))
                if not rebuild_success:
                    raise RuntimeError("FAISS index rebuild failed after PDF ingestion.")
            except Exception:
                for created_path in created_paths:
                    if created_path and os.path.exists(created_path):
                        os.remove(created_path)
                raise

            metadata_path = self._metadata_sidecar_path(out_md)
            record_payload = self._load_json(metadata_path)
            record, _ = self._create_record_from_payload(record_payload, metadata_path)
            self._papers[record.paper_id] = record
            self.total_numbers += 1
            self.total_pages += record.pages
            self.total_size += record.size
            self.next_paper_id = max(self.next_paper_id, record.paper_id + 1)

            return {
                "paper": record.to_public_dict(),
                **self.get_totals(),
            }

    def _doc_belongs_to_paper(self, doc: Any, paper_id: int, record: PaperRecord) -> bool:
        metadata = getattr(doc, "metadata", {}) or {}
        doc_paper_id = self._normalize_int(metadata.get("paper_id"), default=0)
        if doc_paper_id > 0:
            return doc_paper_id == paper_id
        doc_source_file = self._normalize_string(metadata.get("source_file"))
        if doc_source_file and doc_source_file == record.source_file:
            return True
        doc_title = self._normalize_string(metadata.get("title"))
        return bool(doc_title and doc_title == record.title)

    def delete_paper(self, paper_id: int) -> dict[str, Any]:
        self._ensure_dependencies()
        with self.lock:
            target_record = self._papers.get(paper_id)
            if target_record is None:
                raise KeyError(f"Paper {paper_id} not found.")

            all_docs = self.rag_system.get_all_documents_from_faiss()
            retained_docs = [
                doc for doc in all_docs if not self._doc_belongs_to_paper(doc, paper_id, target_record)
            ]
            rebuild_success = self.rag_system.rebuild_from_documents(retained_docs)
            if rebuild_success is False:
                raise RuntimeError(f"Failed to rebuild FAISS index after deleting paper {paper_id}.")

            removable_paths = [
                target_record.md_path,
                target_record.metadata_path,
            ]
            if target_record.stored_pdf_path:
                stored_pdf_path = target_record.stored_pdf_path
                if not os.path.isabs(stored_pdf_path):
                    stored_pdf_path = os.path.abspath(os.path.join(os.getcwd(), stored_pdf_path))
                removable_paths.append(stored_pdf_path)
            else:
                pdf_path = self._resolve_existing_pdf_path(target_record.to_sidecar_payload(), target_record.metadata_path)
                if pdf_path:
                    removable_paths.append(pdf_path)

            for removable_path in removable_paths:
                if removable_path and os.path.exists(removable_path):
                    os.remove(removable_path)

            self._papers.pop(paper_id, None)
            self.total_numbers = max(0, self.total_numbers - 1)
            self.total_pages = max(0, self.total_pages - target_record.pages)
            self.total_size = max(0, self.total_size - target_record.size)

            return {
                "deleted_paper_id": paper_id,
                **self.get_totals(),
            }
