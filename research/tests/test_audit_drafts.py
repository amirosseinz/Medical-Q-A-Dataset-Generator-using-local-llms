import json
import os
import sqlite3
import subprocess
import sys
import unittest
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from app.services.prompt_builder import _retrieved_context, format_evidence_for_prompt
from research.audit_drafts import audit


class DraftAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name, "medqa.db")
        self.connection = sqlite3.connect(self.path)
        self.addCleanup(self.connection.close)
        self.connection.executescript('''
            CREATE TABLE projects (id TEXT PRIMARY KEY);
            CREATE TABLE sources (id TEXT PRIMARY KEY, project_id TEXT, filename TEXT);
            CREATE TABLE chunks (id TEXT PRIMARY KEY, project_id TEXT, source_id TEXT, content TEXT);
            CREATE TABLE qa_pairs (
                id TEXT PRIMARY KEY, project_id TEXT, question TEXT, answer TEXT, metadata TEXT
            );
            INSERT INTO projects VALUES ('project');
            INSERT INTO projects VALUES ('other');
        ''')
        self.chunks = [
            SimpleNamespace(chunk_id="one", source_id="source-one", source_filename="Study Δ",
                            content="A fictional study with uncertain results."),
            SimpleNamespace(chunk_id="two", source_id="source-two", source_filename="Study B",
                            content="Another fictional passage."),
        ]
        for chunk in self.chunks:
            self.connection.execute("INSERT INTO sources VALUES (?, ?, ?)",
                                    (chunk.source_id, "project", chunk.source_filename))
            self.connection.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)",
                                    (chunk.chunk_id, "project", chunk.source_id, chunk.content))
        ordered = self.chunks[::-1]
        self.manifest = _retrieved_context(ordered, format_evidence_for_prompt(ordered))
        self.add_draft("draft", {"retrieved_context": self.manifest})
        self.connection.commit()

    def add_draft(self, pair_id, metadata):
        self.connection.execute("INSERT INTO qa_pairs VALUES (?, ?, ?, ?, ?)",
                                (pair_id, "project", "Fictional question?", "Fictional answer.",
                                 json.dumps(metadata, ensure_ascii=False)))

    def run_audit(self):
        self.connection.commit()
        return audit(self.path, "project")

    def test_current_generator_manifest_and_rotated_unicode_context_match(self):
        before = sha256(self.path.read_bytes()).hexdigest()
        result = self.run_audit()
        self.assertEqual(result["counts"], {"context_matches": 1})
        self.assertEqual(result["rows"][0]["passage_count"], 2)
        self.assertEqual(before, sha256(self.path.read_bytes()).hexdigest())
        self.assertNotIn("Fictional answer", json.dumps(result))

    def test_content_change_and_filename_change_are_detected(self):
        self.connection.execute("UPDATE chunks SET content = 'Changed content' WHERE id = 'one'")
        result = self.run_audit()["rows"][0]
        self.assertEqual(result["status"], "context_mismatch")
        self.assertEqual(result["changed_chunk_ids"], ["one"])
        self.connection.execute("UPDATE chunks SET content = ? WHERE id = 'one'", (self.chunks[0].content,))
        self.connection.execute("UPDATE sources SET filename = 'Renamed' WHERE id = 'source-one'")
        result = self.run_audit()["rows"][0]
        self.assertEqual(result["status"], "context_mismatch")
        self.assertEqual(result["changed_chunk_ids"], [])

    def test_changed_order_is_detected(self):
        manifest = deepcopy(self.manifest)
        manifest["passages"].reverse()
        self.add_draft("order", {"retrieved_context": manifest})
        self.assertEqual(self.run_audit()["counts"], {"context_matches": 1, "context_mismatch": 1})

    def test_cross_project_sources_are_reported_as_missing(self):
        self.connection.execute("UPDATE sources SET project_id = 'other' WHERE id = 'source-one'")
        result = self.run_audit()["rows"][0]
        self.assertEqual(result["status"], "missing_source")
        self.assertEqual(result["missing_chunk_ids"], ["one"])

    def test_legacy_and_malformed_rows_do_not_disappear_from_denominator(self):
        self.add_draft("legacy", {})
        self.add_draft("bad-json", {})
        self.connection.execute("UPDATE qa_pairs SET metadata = '{bad' WHERE id = 'bad-json'")
        for i, manifest in enumerate((None, {}, {**self.manifest, "passages": []},
                                      {**self.manifest, "version": True},
                                      {**self.manifest, "context_sha256": "not-a-hash"})):
            self.add_draft(f"invalid-{i}", {"retrieved_context": manifest})
        result = self.run_audit()
        self.assertEqual(result["total_drafts"], 8)
        self.assertEqual(result["counts"], {
            "context_matches": 1, "missing_manifest": 1, "invalid_metadata": 1, "invalid_manifest": 5,
        })

    def test_cli_exit_codes_and_read_only_access(self):
        root = Path(__file__).resolve().parents[2]
        env = {**os.environ, "PYTHONPATH": str(root / "backend")}

        def invoke(project="project", path=None):
            return subprocess.run(
                [sys.executable, "-m", "research.audit_drafts", "--database", str(path or self.path),
                 "--project-id", project], cwd=root, env=env, capture_output=True, text=True,
            )

        self.assertEqual(invoke().returncode, 0)
        self.add_draft("legacy", {})
        self.connection.commit()
        self.assertEqual(invoke().returncode, 1)
        self.assertEqual(invoke("other").returncode, 1)
        self.assertEqual(invoke("unknown").returncode, 2)
        absent = self.path.with_name("absent.db")
        self.assertEqual(invoke(path=absent).returncode, 2)
        self.assertFalse(absent.exists())

    def test_draft_revision_changes_when_answer_is_edited(self):
        before = self.run_audit()["rows"][0]
        self.connection.execute("UPDATE qa_pairs SET answer = 'Revised answer'")
        after = self.run_audit()["rows"][0]
        self.assertNotEqual(before["draft_sha256"], after["draft_sha256"])
        self.assertEqual(after["status"], "context_matches")


if __name__ == "__main__":
    unittest.main()
