from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from rag.schemas import NormalizedDocument
    from rag.testing.diagnostic_cli import combine_branch_documents, detect_text_issues, load_query_from_args
except ImportError:
    from schemas import NormalizedDocument
    from testing.diagnostic_cli import combine_branch_documents, detect_text_issues, load_query_from_args


class DummyArgs:
    def __init__(self, *, query: str = "", query_file: str = "") -> None:
        self.query = query
        self.query_file = query_file


class DiagnosticCLITests(unittest.TestCase):
    def test_detect_text_issues_flags_repeated_question_marks(self):
        issues = detect_text_issues("fading memory????HA-GNN????")

        self.assertTrue(issues["has_issues"])
        self.assertIn("contains_repeated_question_marks", issues["issue_flags"])

    def test_load_query_from_file_uses_utf8_sig(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "query.txt"
            path.write_text("fading memory\uff08\u8870\u51cf\u8bb0\u5fc6\uff09\n", encoding="utf-8-sig")

            query, source = load_query_from_args(DummyArgs(query_file=str(path)))

        self.assertEqual(query, "fading memory\uff08\u8870\u51cf\u8bb0\u5fc6\uff09")
        self.assertEqual(source["source"], "query_file")
        self.assertTrue(source["path"].endswith("query.txt"))

    def test_combine_branch_documents_dedupes_and_keeps_branch_order(self):
        shared_doc = NormalizedDocument(
            content="same content",
            source="paper_a.md",
            metadata={"title": "A"},
        )
        other_doc = NormalizedDocument(
            content="different content",
            source="paper_b.md",
            metadata={"title": "B"},
        )
        branch_results = {
            "bm25_en": {"ok": True, "documents": [shared_doc]},
            "dense_zh": {"ok": True, "documents": [shared_doc, other_doc]},
            "dense_en": {"ok": False, "documents": []},
        }

        combined = combine_branch_documents(branch_results, ["bm25_en", "dense_zh", "dense_en"], max_docs=5)

        self.assertEqual(len(combined), 2)
        self.assertEqual(combined[0].metadata["diagnostic_branch"], "bm25_en")
        self.assertEqual(combined[1].metadata["diagnostic_branch"], "dense_zh")


if __name__ == "__main__":
    unittest.main()
