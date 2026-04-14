from __future__ import annotations

import glob
import json
import os
import tempfile
import time
import unittest

try:
    import fitz
except ModuleNotFoundError:
    fitz = None

from fastapi.testclient import TestClient

try:
    from rag.api_server import BackendServices, create_app
    from rag.kb_manager import KnowledgeBaseManager
except ImportError:
    from api_server import BackendServices, create_app
    from kb_manager import KnowledgeBaseManager


class DummyChunkDoc:
    def __init__(self, content: str, metadata: dict[str, object]):
        self.page_content = content
        self.metadata = dict(metadata)


class FakePDFProcessor:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.calls: list[dict[str, object]] = []

    def process_pdf(self, pdf_path: str, out_md: str | None = None, *, extra_metadata=None):
        self.calls.append(
            {
                "pdf_path": pdf_path,
                "out_md": out_md,
                "extra_metadata": dict(extra_metadata or {}),
            }
        )
        target_md = out_md or os.path.join(
            self.output_dir,
            os.path.splitext(os.path.basename(pdf_path))[0] + ".md",
        )
        os.makedirs(os.path.dirname(target_md), exist_ok=True)
        with open(target_md, "w", encoding="utf-8") as file:
            file.write("## Page 1\n\nSynthetic OCR content for testing.")
        metadata_path = target_md.replace(".md", ".metadata.json")
        payload = {
            "title": os.path.splitext((extra_metadata or {}).get("source_file") or os.path.basename(pdf_path))[0],
            "url": "",
            "pdf_link": "",
            "source_file": (extra_metadata or {}).get("source_file") or os.path.basename(pdf_path),
            "origin": "local_kb",
        }
        payload.update(extra_metadata or {})
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return {"md_path": target_md, "pdf_path": pdf_path}


class FakeRAGSystem:
    def __init__(self, *, documents=None):
        self.documents = list(documents or [])
        self.update_calls: list[str] = []
        self.rebuild_calls: list[list[DummyChunkDoc]] = []

    def update_rag_system(self, chunk_strategy="semantic_arxiv"):
        self.update_calls.append(chunk_strategy)
        return True

    def get_all_documents_from_faiss(self):
        return list(self.documents)

    def rebuild_from_documents(self, documents):
        retained = list(documents)
        self.rebuild_calls.append(retained)
        self.documents = retained
        return True


class KnowledgeBaseManagerTests(unittest.TestCase):
    def _temp_workspace(self) -> tuple[str, str, str]:
        if fitz is None:
            raise unittest.SkipTest("fitz is not available in the current test environment")
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        workspace = temp_dir.name
        md_dir = os.path.join(workspace, "md")
        pdf_root = os.path.join(workspace, "paper_results")
        upload_dir = pdf_root
        os.makedirs(md_dir, exist_ok=True)
        os.makedirs(pdf_root, exist_ok=True)
        os.makedirs(upload_dir, exist_ok=True)
        return md_dir, pdf_root, upload_dir

    def _create_pdf(self, path: str, *, pages: int, title: str) -> int:
        doc = fitz.open()
        for index in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"{title} page {index + 1}")
        doc.set_metadata({"title": title})
        doc.save(path)
        doc.close()
        return os.path.getsize(path)

    def _write_legacy_sidecar(self, md_dir: str, filename: str, payload: dict[str, object]) -> str:
        md_path = os.path.join(md_dir, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as file:
            file.write("Legacy markdown content.")
        sidecar_path = os.path.join(md_dir, f"{filename}.metadata.json")
        with open(sidecar_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        return sidecar_path

    def test_refresh_state_backfills_legacy_metadata_and_totals(self):
        md_dir, pdf_root, upload_dir = self._temp_workspace()
        pdf_one = os.path.join(pdf_root, "Paper One.pdf")
        pdf_two = os.path.join(pdf_root, "Paper Two.pdf")
        size_one = self._create_pdf(pdf_one, pages=3, title="Paper One")
        size_two = self._create_pdf(pdf_two, pages=5, title="Paper Two")

        sidecar_one = self._write_legacy_sidecar(
            md_dir,
            "Paper One",
            {
                "title": "Paper One",
                "url": "https://example.com/paper-one.pdf",
                "pdf_link": "https://example.com/paper-one.pdf",
                "source_file": "Paper One.pdf",
                "origin": "local_kb",
            },
        )
        sidecar_two = self._write_legacy_sidecar(
            md_dir,
            "Paper Two",
            {
                "title": "Paper Two",
                "url": "https://example.com/paper-two.pdf",
                "pdf_link": "https://example.com/paper-two.pdf",
                "source_file": "Paper Two.pdf",
                "origin": "local_kb",
            },
        )
        now = time.time()
        os.utime(sidecar_one, (now - 120, now - 120))
        os.utime(sidecar_two, (now - 60, now - 60))

        fake_rag_system = FakeRAGSystem()
        manager = KnowledgeBaseManager(
            md_dir=md_dir,
            pdf_root_dir=pdf_root,
            upload_dir=upload_dir,
            rag_system=fake_rag_system,
            pdf_processor=FakePDFProcessor(md_dir),
        )

        result = manager.refresh_state(rebuild_if_needed=True)

        self.assertTrue(result["sidecars_changed"])
        self.assertEqual(fake_rag_system.update_calls, ["semantic_arxiv"])
        self.assertEqual(manager.total_numbers, 2)
        self.assertEqual(manager.total_pages, 8)
        self.assertEqual(manager.total_size, size_one + size_two)
        self.assertEqual(manager.next_paper_id, 3)

        with open(sidecar_one, "r", encoding="utf-8") as file:
            payload_one = json.load(file)
        with open(sidecar_two, "r", encoding="utf-8") as file:
            payload_two = json.load(file)

        self.assertEqual(payload_one["paper_id"], 1)
        self.assertEqual(payload_two["paper_id"], 2)
        self.assertEqual(payload_one["size"], size_one)
        self.assertEqual(payload_two["pages"], 5)
        self.assertIn("time", payload_one)
        self.assertIn("stored_pdf_path", payload_one)

    def test_ingest_pdf_updates_sidecar_and_totals(self):
        md_dir, pdf_root, upload_dir = self._temp_workspace()
        uploaded_pdf = os.path.join(upload_dir, "uploaded.pdf")
        expected_size = self._create_pdf(uploaded_pdf, pages=4, title="Uploaded Paper")

        fake_processor = FakePDFProcessor(md_dir)
        fake_rag_system = FakeRAGSystem()
        manager = KnowledgeBaseManager(
            md_dir=md_dir,
            pdf_root_dir=pdf_root,
            upload_dir=upload_dir,
            rag_system=fake_rag_system,
            pdf_processor=fake_processor,
        )
        manager.refresh_state(rebuild_if_needed=False)

        result = manager.ingest_pdf(uploaded_pdf, source_file="Uploaded Paper.pdf")

        self.assertEqual(result["paper"]["paper_id"], 1)
        self.assertEqual(result["paper"]["pages"], 4)
        self.assertEqual(result["total_numbers"], 1)
        self.assertEqual(result["total_pages"], 4)
        self.assertEqual(result["total_size"], expected_size)
        self.assertEqual(fake_rag_system.update_calls, ["semantic_arxiv"])
        self.assertEqual(len(fake_processor.calls), 1)

        sidecars = glob.glob(os.path.join(md_dir, "*.metadata.json"))
        self.assertEqual(len(sidecars), 1)
        with open(sidecars[0], "r", encoding="utf-8") as file:
            payload = json.load(file)
        self.assertEqual(payload["paper_id"], 1)
        self.assertEqual(payload["source_file"], "Uploaded Paper.pdf")
        self.assertEqual(payload["size"], expected_size)
        self.assertEqual(payload["pages"], 4)
        self.assertIn("time", payload)

    def test_delete_paper_rebuilds_faiss_from_retained_chunks(self):
        md_dir, pdf_root, upload_dir = self._temp_workspace()
        pdf_one = os.path.join(pdf_root, "Paper One.pdf")
        pdf_two = os.path.join(pdf_root, "Paper Two.pdf")
        size_one = self._create_pdf(pdf_one, pages=2, title="Paper One")
        size_two = self._create_pdf(pdf_two, pages=4, title="Paper Two")

        md_one = os.path.join(md_dir, "paper_1_Paper One.md")
        md_two = os.path.join(md_dir, "paper_2_Paper Two.md")
        with open(md_one, "w", encoding="utf-8") as file:
            file.write("Paper one markdown.")
        with open(md_two, "w", encoding="utf-8") as file:
            file.write("Paper two markdown.")
        with open(md_one.replace(".md", ".metadata.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "paper_id": 1,
                    "title": "Paper One",
                    "url": "",
                    "pdf_link": "",
                    "source_file": "Paper One.pdf",
                    "origin": "local_kb",
                    "time": "2026-04-14 10:00:00",
                    "size": size_one,
                    "pages": 2,
                    "stored_pdf_path": os.path.relpath(pdf_one, os.getcwd()),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        with open(md_two.replace(".md", ".metadata.json"), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "paper_id": 2,
                    "title": "Paper Two",
                    "url": "",
                    "pdf_link": "",
                    "source_file": "Paper Two.pdf",
                    "origin": "local_kb",
                    "time": "2026-04-14 10:05:00",
                    "size": size_two,
                    "pages": 4,
                    "stored_pdf_path": os.path.relpath(pdf_two, os.getcwd()),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        fake_rag_system = FakeRAGSystem(
            documents=[
                DummyChunkDoc("chunk 1", {"paper_id": 1, "title": "Paper One", "source_file": "Paper One.pdf"}),
                DummyChunkDoc("chunk 2", {"paper_id": 1, "title": "Paper One", "source_file": "Paper One.pdf"}),
                DummyChunkDoc("chunk 3", {"paper_id": 2, "title": "Paper Two", "source_file": "Paper Two.pdf"}),
            ]
        )
        manager = KnowledgeBaseManager(
            md_dir=md_dir,
            pdf_root_dir=pdf_root,
            upload_dir=upload_dir,
            rag_system=fake_rag_system,
            pdf_processor=FakePDFProcessor(md_dir),
        )
        manager.refresh_state(rebuild_if_needed=False)

        result = manager.delete_paper(1)

        self.assertEqual(result["deleted_paper_id"], 1)
        self.assertEqual(result["total_numbers"], 1)
        self.assertEqual(result["total_pages"], 4)
        self.assertEqual(result["total_size"], size_two)
        self.assertEqual(manager.next_paper_id, 3)
        self.assertEqual(len(fake_rag_system.rebuild_calls), 1)
        retained_docs = fake_rag_system.rebuild_calls[0]
        self.assertEqual(len(retained_docs), 1)
        self.assertEqual(retained_docs[0].metadata["paper_id"], 2)
        self.assertFalse(os.path.exists(md_one))
        self.assertFalse(os.path.exists(md_one.replace(".md", ".metadata.json")))
        self.assertFalse(os.path.exists(pdf_one))
        self.assertTrue(os.path.exists(pdf_two))

    def test_list_papers_supports_keyword_search(self):
        md_dir, pdf_root, upload_dir = self._temp_workspace()
        pdf_alpha = os.path.join(pdf_root, "Alpha Theory.pdf")
        pdf_beta = os.path.join(pdf_root, "Beta Practice.pdf")
        self._create_pdf(pdf_alpha, pages=2, title="Alpha Theory")
        self._create_pdf(pdf_beta, pages=3, title="Beta Practice")
        self._write_legacy_sidecar(
            md_dir,
            "Alpha Theory",
            {
                "title": "Alpha Theory",
                "url": "",
                "pdf_link": "",
                "source_file": "Alpha Theory.pdf",
                "origin": "local_kb",
            },
        )
        self._write_legacy_sidecar(
            md_dir,
            "Beta Practice",
            {
                "title": "Beta Practice",
                "url": "",
                "pdf_link": "",
                "source_file": "Beta Practice.pdf",
                "origin": "local_kb",
            },
        )

        manager = KnowledgeBaseManager(
            md_dir=md_dir,
            pdf_root_dir=pdf_root,
            upload_dir=upload_dir,
            rag_system=FakeRAGSystem(),
            pdf_processor=FakePDFProcessor(md_dir),
        )
        manager.refresh_state(rebuild_if_needed=False)

        all_result = manager.list_papers()
        search_result = manager.list_papers(keyword="beta")

        self.assertEqual(all_result["total_numbers"], 2)
        self.assertEqual(len(all_result["papers"]), 2)
        self.assertEqual(len(search_result["papers"]), 1)
        self.assertEqual(search_result["papers"][0]["source_file"], "Beta Practice.pdf")


class KnowledgeBaseApiTests(unittest.TestCase):
    def _temp_workspace(self) -> tuple[str, str, str]:
        if fitz is None:
            raise unittest.SkipTest("fitz is not available in the current test environment")
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        workspace = temp_dir.name
        md_dir = os.path.join(workspace, "md")
        pdf_root = os.path.join(workspace, "paper_results")
        upload_dir = pdf_root
        os.makedirs(md_dir, exist_ok=True)
        os.makedirs(pdf_root, exist_ok=True)
        os.makedirs(upload_dir, exist_ok=True)
        return md_dir, pdf_root, upload_dir

    def _create_pdf(self, path: str, *, pages: int, title: str) -> None:
        doc = fitz.open()
        for index in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"{title} page {index + 1}")
        doc.set_metadata({"title": title})
        doc.save(path)
        doc.close()

    def test_kb_api_upload_list_search_and_delete(self):
        md_dir, pdf_root, upload_dir = self._temp_workspace()
        sample_pdf = os.path.join(pdf_root, "api-upload-source.pdf")
        self._create_pdf(sample_pdf, pages=3, title="API Upload")

        fake_processor = FakePDFProcessor(md_dir)
        fake_rag_system = FakeRAGSystem()
        manager = KnowledgeBaseManager(
            md_dir=md_dir,
            pdf_root_dir=pdf_root,
            upload_dir=upload_dir,
            rag_system=fake_rag_system,
            pdf_processor=fake_processor,
        )
        manager.refresh_state(rebuild_if_needed=False)

        services = BackendServices(
            tasks_db={},
            pdf_processor=fake_processor,
            rag_system=fake_rag_system,
            kb_manager=manager,
            runtime_initialized=True,
        )
        app = create_app(services=services, initialize_on_startup=False)

        with TestClient(app) as client:
            with open(sample_pdf, "rb") as file:
                upload_response = client.post(
                    "/api/kb/papers/upload",
                    files={"file": ("API Upload.pdf", file, "application/pdf")},
                )
            self.assertEqual(upload_response.status_code, 200)
            upload_payload = upload_response.json()
            self.assertEqual(upload_payload["message"], "文件已完成加载")
            self.assertEqual(upload_payload["paper"]["paper_id"], 1)
            self.assertEqual(upload_payload["paper"]["pages"], 3)
            self.assertEqual(upload_payload["paper"]["source_file"], "API Upload.pdf")
            self.assertEqual(fake_processor.calls[0]["pdf_path"], os.path.join(pdf_root, "API Upload.pdf"))
            self.assertTrue(os.path.exists(os.path.join(pdf_root, "API Upload.pdf")))

            list_response = client.get("/api/kb/papers")
            self.assertEqual(list_response.status_code, 200)
            list_payload = list_response.json()
            self.assertEqual(list_payload["total_numbers"], 1)
            self.assertEqual(len(list_payload["papers"]), 1)
            self.assertEqual(list_payload["papers"][0]["source_file"], "API Upload.pdf")

            search_response = client.get("/api/kb/papers", params={"keyword": "upload"})
            self.assertEqual(search_response.status_code, 200)
            search_payload = search_response.json()
            self.assertEqual(len(search_payload["papers"]), 1)
            self.assertEqual(search_payload["papers"][0]["paper_id"], 1)

            delete_response = client.delete("/api/kb/papers/1")
            self.assertEqual(delete_response.status_code, 200)
            delete_payload = delete_response.json()
            self.assertEqual(delete_payload["message"], "文件已删除")
            self.assertEqual(delete_payload["deleted_paper_id"], 1)
            self.assertEqual(delete_payload["total_numbers"], 0)

            final_list_response = client.get("/api/kb/papers")
            self.assertEqual(final_list_response.status_code, 200)
            self.assertEqual(final_list_response.json()["papers"], [])

    def test_kb_api_rejects_duplicate_uploaded_filename_when_already_indexed(self):
        md_dir, pdf_root, upload_dir = self._temp_workspace()
        existing_pdf = os.path.join(pdf_root, "Duplicate Paper.pdf")
        self._create_pdf(existing_pdf, pages=2, title="Duplicate Paper")

        fake_processor = FakePDFProcessor(md_dir)
        fake_rag_system = FakeRAGSystem()
        manager = KnowledgeBaseManager(
            md_dir=md_dir,
            pdf_root_dir=pdf_root,
            upload_dir=upload_dir,
            rag_system=fake_rag_system,
            pdf_processor=fake_processor,
        )
        manager.ingest_pdf(existing_pdf, source_file="Duplicate Paper.pdf")

        upload_source = os.path.join(pdf_root, "duplicate-upload-source.pdf")
        self._create_pdf(upload_source, pages=2, title="Duplicate Paper Upload")

        services = BackendServices(
            tasks_db={},
            pdf_processor=fake_processor,
            rag_system=fake_rag_system,
            kb_manager=manager,
            runtime_initialized=True,
        )
        app = create_app(services=services, initialize_on_startup=False)

        with TestClient(app) as client:
            with open(upload_source, "rb") as file:
                response = client.post(
                    "/api/kb/papers/upload",
                    files={"file": ("Duplicate Paper.pdf", file, "application/pdf")},
                )

        self.assertEqual(response.status_code, 409)
        self.assertIn("文件已存在于知识库中", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
