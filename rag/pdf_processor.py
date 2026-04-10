from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import Counter
from typing import Any

import fitz

try:
    from .ocr_client import RemoteOCRClient, create_default_ocr_client
except ImportError:
    from ocr_client import RemoteOCRClient, create_default_ocr_client


SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "method",
    "methods",
    "approach",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "references",
    "acknowledgments",
    "acknowledgements",
}


class PDFProcessor:
    """Convert PDF papers into cleaned markdown using the remote OCR service."""

    def __init__(
        self,
        output_dir: str = "./md",
        lang: str = "en",
        dpi: int = 220,
        *,
        ocr_client: RemoteOCRClient | None = None,
        ocr_max_tokens: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.lang = lang
        self.dpi = dpi
        self.ocr_client = ocr_client
        configured_max_tokens = str(ocr_max_tokens or os.getenv("RAG_OCR_MAX_TOKENS", "2048")).strip()
        self.ocr_max_tokens = int(configured_max_tokens) if configured_max_tokens.isdigit() else 2048
        self._paper_manifest_cache: dict[str, dict[str, dict[str, str]]] = {}
        os.makedirs(self.output_dir, exist_ok=True)

    def _metadata_sidecar_path(self, path: str) -> str:
        stem, _ = os.path.splitext(path)
        return f"{stem}.metadata.json"

    def _clean_filename(self, filename: str, max_len: int = 80) -> str:
        clean_name = re.sub(r'[<>:"/\\|?*]', "", str(filename or ""))
        clean_name = re.sub(r"\s+", " ", clean_name).strip()
        if len(clean_name) > max_len:
            clean_name = clean_name[:max_len]
        return clean_name

    def _safe_metadata_value(self, value: Any) -> str:
        return str(value or "").strip()

    def _load_json_metadata(self, path: str) -> dict[str, str]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception as exc:
            print(f"Failed to read metadata sidecar: {path} ({exc})")
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): self._safe_metadata_value(value) for key, value in payload.items()}

    def _load_manifest_index(self, folder: str) -> dict[str, dict[str, str]]:
        folder_key = os.path.abspath(folder)
        cached = self._paper_manifest_cache.get(folder_key)
        if cached is not None:
            return cached

        manifest_path = os.path.join(folder, "paper_result.csv")
        index: dict[str, dict[str, str]] = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, newline="", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if not isinstance(row, dict):
                            continue
                        title = self._safe_metadata_value(row.get("title"))
                        if not title:
                            continue
                        normalized_row = {
                            "title": title,
                            "pdf_link": self._safe_metadata_value(row.get("pdf_link")),
                        }
                        lookup_keys = {
                            title.lower(),
                            self._clean_filename(title).lower(),
                        }
                        for lookup_key in lookup_keys:
                            if lookup_key:
                                index.setdefault(lookup_key, normalized_row)
            except Exception as exc:
                print(f"Failed to read paper_result.csv: {manifest_path} ({exc})")

        self._paper_manifest_cache[folder_key] = index
        return index

    def _resolve_manifest_metadata(self, pdf_path: str) -> dict[str, str]:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        manifest_index = self._load_manifest_index(os.path.dirname(pdf_path))
        lookup_keys = [stem.lower(), self._clean_filename(stem).lower()]
        for lookup_key in lookup_keys:
            if lookup_key and lookup_key in manifest_index:
                return dict(manifest_index[lookup_key])
        return {}

    def _normalize_pdf_metadata(self, pdf_path: str, doc: fitz.Document | None = None) -> dict[str, str]:
        sidecar = self._load_json_metadata(self._metadata_sidecar_path(pdf_path))
        manifest = self._resolve_manifest_metadata(pdf_path)

        doc_title = ""
        if doc is not None:
            try:
                doc_title = self._safe_metadata_value((doc.metadata or {}).get("title"))
            except Exception:
                doc_title = ""

        title = (
            sidecar.get("title")
            or manifest.get("title")
            or doc_title
            or os.path.splitext(os.path.basename(pdf_path))[0]
        )
        pdf_link = sidecar.get("pdf_link") or manifest.get("pdf_link") or sidecar.get("url") or ""
        url = sidecar.get("url") or pdf_link

        return {
            "title": self._safe_metadata_value(title),
            "url": self._safe_metadata_value(url),
            "pdf_link": self._safe_metadata_value(pdf_link or url),
            "source_file": os.path.basename(pdf_path),
            "origin": "local_kb",
        }

    def _write_md_metadata(self, out_md: str, metadata: dict[str, Any]) -> None:
        payload = {
            "title": self._safe_metadata_value(metadata.get("title")),
            "url": self._safe_metadata_value(metadata.get("url")),
            "pdf_link": self._safe_metadata_value(metadata.get("pdf_link")),
            "source_file": self._safe_metadata_value(metadata.get("source_file")),
            "origin": self._safe_metadata_value(metadata.get("origin") or "local_kb"),
        }
        with open(self._metadata_sidecar_path(out_md), "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _get_ocr_client(self) -> RemoteOCRClient:
        if self.ocr_client is None:
            self.ocr_client = create_default_ocr_client(max_tokens=self.ocr_max_tokens)
        return self.ocr_client

    def _render_page_image_bytes(self, page: fitz.Page) -> bytes:
        pix = page.get_pixmap(dpi=self.dpi)
        return pix.tobytes("png")

    def _ocr_page_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        client = self._get_ocr_client()
        return client.extract_from_image_bytes(image_bytes, max_tokens=self.ocr_max_tokens)

    def _is_noise_line(self, line: str) -> bool:
        cleaned = str(line or "").strip()
        if not cleaned:
            return False
        if re.fullmatch(r"[#=\-_*~`]+", cleaned):
            return True
        if re.fullmatch(r"\d{1,4}", cleaned):
            return True
        if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", cleaned, re.IGNORECASE):
            return True
        if len(cleaned) <= 2 and not re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned):
            return True
        compact = re.sub(r"[\W_]+", "", cleaned)
        if len(compact) <= 1 and not re.search(r"[A-Za-z\u4e00-\u9fff]", cleaned):
            return True
        return False

    def _normalize_ocr_text_to_lines(self, text: str) -> list[str]:
        cleaned = str(text or "")
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.replace("\\n", "\n")
        cleaned = cleaned.replace("\u00ad", "")
        cleaned = re.sub(r"<\|LOC_\d+\|>", "", cleaned)
        cleaned = re.sub(r"\\([()\[\]{}])", r"\1", cleaned)
        cleaned = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", cleaned)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        lines: list[str] = []
        for raw_line in cleaned.split("\n"):
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                if lines and lines[-1] != "":
                    lines.append("")
                continue
            if self._is_noise_line(line):
                continue
            lines.append(line)
        while lines and lines[-1] == "":
            lines.pop()
        return lines

    def _common_boundary_lines(self, pages: list[list[str]]) -> set[str]:
        if len(pages) < 3:
            return set()

        counts: Counter[str] = Counter()
        for lines in pages:
            non_empty = [line for line in lines if line]
            boundaries = set(non_empty[:3] + non_empty[-3:])
            for line in boundaries:
                counts[line] += 1

        threshold = max(2, int(0.3 * len(pages)))
        return {
            line
            for line, count in counts.items()
            if count >= threshold and len(line) <= 120
        }

    def _is_section_heading(self, line: str) -> bool:
        cleaned = str(line or "").strip()
        if not cleaned:
            return False
        lowered = cleaned.lower().rstrip(":")
        normalized = re.sub(r"^\d+(?:\.\d+)*\s+", "", lowered).strip()
        if normalized in SECTION_HEADINGS:
            return True
        if re.fullmatch(r"\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 .,\-()/:]{0,59}", cleaned):
            return True
        if len(cleaned) <= 60 and re.fullmatch(r"[A-Z][A-Za-z0-9 .,\-()/:]{0,59}", cleaned):
            return True
        return False

    def _should_start_new_paragraph(self, previous_line: str, current_line: str) -> bool:
        prev = str(previous_line or "").strip()
        curr = str(current_line or "").strip()
        if not prev or not curr:
            return False
        if self._is_section_heading(curr):
            return True
        if self._is_section_heading(prev):
            return True
        if prev.endswith(":"):
            return True
        if len(prev) <= 80 and not re.search(r"[.!?;,:)]$", prev) and curr[:1].isupper():
            return True
        return False

    def _merge_paragraph_lines(self, lines: list[str]) -> str:
        if not lines:
            return ""
        merged = lines[0]
        for line in lines[1:]:
            stripped = line.lstrip()
            if merged.endswith("-"):
                merged = merged[:-1] + stripped
            else:
                merged = f"{merged} {stripped}".strip()
        merged = re.sub(r"\s+", " ", merged).strip()
        merged = re.sub(r"\s+([,.;:!?])", r"\1", merged)
        return merged

    def _reflow_page_lines(self, lines: list[str]) -> str:
        paragraphs: list[str] = []
        current: list[str] = []
        previous_non_empty = ""

        for line in lines:
            if not line:
                if current:
                    paragraphs.append(self._merge_paragraph_lines(current))
                    current = []
                previous_non_empty = ""
                continue

            if current and self._should_start_new_paragraph(previous_non_empty, line):
                paragraphs.append(self._merge_paragraph_lines(current))
                current = [line]
            else:
                current.append(line)
            previous_non_empty = line

        if current:
            paragraphs.append(self._merge_paragraph_lines(current))

        page_text = "\n\n".join([paragraph for paragraph in paragraphs if paragraph.strip()])
        page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
        return page_text

    def _clean_pages(self, page_lines: list[list[str]]) -> list[str]:
        common_boundary_lines = self._common_boundary_lines(page_lines)
        cleaned_pages: list[str] = []
        for lines in page_lines:
            filtered_lines = [line for line in lines if line and line not in common_boundary_lines]
            page_text = self._reflow_page_lines(filtered_lines)
            cleaned_pages.append(page_text)
        return cleaned_pages

    def _pdf_to_text(self, pdf_path: str, out_md: str) -> dict[str, Any]:
        doc = fitz.open(pdf_path)
        doc_metadata = self._normalize_pdf_metadata(pdf_path, doc)
        page_count = doc.page_count
        page_lines: list[list[str]] = []

        try:
            for index in range(page_count):
                page = doc[index]
                image_bytes = self._render_page_image_bytes(page)
                ocr_result = self._ocr_page_bytes(image_bytes)
                page_lines.append(self._normalize_ocr_text_to_lines(ocr_result.get("text", "")))
        finally:
            doc.close()

        cleaned_pages = self._clean_pages(page_lines)
        os.makedirs(os.path.dirname(out_md) or self.output_dir, exist_ok=True)
        md_parts = [f"## Page {index + 1}\n\n{page_text}" for index, page_text in enumerate(cleaned_pages)]
        with open(out_md, "w", encoding="utf-8") as file:
            file.write("\n\n".join(md_parts))
        self._write_md_metadata(out_md, doc_metadata)
        return {
            "pdf_path": pdf_path,
            "md_path": out_md,
            "page_count": page_count,
            "nonempty_pages": sum(1 for page_text in cleaned_pages if page_text.strip()),
        }

    def process_pdf(self, pdf_path: str, out_md: str | None = None) -> dict[str, Any]:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
        target_md = out_md or os.path.join(
            self.output_dir,
            os.path.splitext(os.path.basename(pdf_path))[0] + ".md",
        )
        return self._pdf_to_text(pdf_path, target_md)

    def process_pdf_folder(self, pdf_folder_path: str) -> list[str]:
        if not os.path.exists(pdf_folder_path):
            raise FileNotFoundError(f"PDF folder does not exist: {pdf_folder_path}")

        pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
        pdf_files += glob.glob(os.path.join(pdf_folder_path, "*.PDF"))
        deduped_pdf_files: list[str] = []
        seen_paths: set[str] = set()
        for pdf_file in pdf_files:
            normalized = os.path.normcase(os.path.abspath(pdf_file))
            if normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            deduped_pdf_files.append(pdf_file)
        pdf_files = deduped_pdf_files
        if not pdf_files:
            print(f"No PDF files found in {pdf_folder_path}")
            return []

        processed_files: list[str] = []
        for pdf_file in sorted(pdf_files):
            try:
                print(f"Processing: {pdf_file}")
                self.process_pdf(pdf_file)
                processed_files.append(pdf_file)
                print(f"Finished: {pdf_file}")
            except Exception as exc:
                print(f"Failed to process {pdf_file}: {exc}")
        return processed_files


if __name__ == "__main__":
    processor = PDFProcessor(output_dir="./md", lang="en", dpi=220)
    pdf_folder = "./pdf"
    result = processor.process_pdf_folder(pdf_folder)
    print(f"Processed {len(result)} PDF files")
