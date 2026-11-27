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

    def review(self, label="supported"):
        row = self.run_audit()["rows"][0]
        return {"qa_pair_id": row["qa_pair_id"], "draft_sha256": row["draft_sha256"],
                "context_sha256": row["context_sha256"], "label": label}

    def test_review_summary_keeps_unreviewed_rows_in_coverage_denominator(self):
        reviews = [self.review()]
        for label in ("contradicted", "insufficient", "conflicting"):
            self.add_draft(label, {"retrieved_context": self.manifest})
            reviews.append({**reviews[0], "qa_pair_id": label, "label": label})
        self.add_draft("unreviewed", {})
        self.connection.commit()
        report = audit(self.path, "project", reviews)["evidence_reviews"]
        self.assertEqual(report["counts"], {
            "supported": 1, "contradicted": 1, "insufficient": 1, "conflicting": 1, "unreviewed": 1,
        })
        self.assertEqual(report["coverage"], 0.8)
        self.assertEqual(report["supported_fraction_among_current_reviews"], 0.25)
        self.assertEqual(report["current_reviews"], 4)

    def test_edited_draft_cannot_reuse_old_review(self):
        review = self.review()
        self.connection.execute("UPDATE qa_pairs SET answer = 'Edited answer'")
        self.connection.commit()
        report = audit(self.path, "project", [review])["evidence_reviews"]
        self.assertEqual(report["counts"], {"stale_review": 1})
        self.assertEqual(report["coverage"], 0)
        self.assertIsNone(report["supported_fraction_among_current_reviews"])

    def test_changed_or_missing_context_does_not_count_as_current_review(self):
        review = self.review()
        self.connection.execute("DELETE FROM sources WHERE id = 'source-one'")
        self.connection.commit()
        report = audit(self.path, "project", [review])["evidence_reviews"]
        self.assertEqual(report["counts"], {"context_unverified": 1})
        self.assertEqual(report["current_reviews"], 0)

    def test_changed_manifest_cannot_reuse_review_even_when_context_matches(self):
        review = self.review()
        self.chunks[0].content = "A replacement passage."
        self.connection.execute("UPDATE chunks SET content = ? WHERE id = 'one'", (self.chunks[0].content,))
        manifest = _retrieved_context(self.chunks[::-1], format_evidence_for_prompt(self.chunks[::-1]))
        self.connection.execute("UPDATE qa_pairs SET metadata = ?", (json.dumps({"retrieved_context": manifest}),))
        self.connection.commit()
        report = audit(self.path, "project", [review])
        self.assertEqual(report["counts"], {"context_matches": 1})
        self.assertEqual(report["evidence_reviews"]["counts"], {"stale_review": 1})

    def test_ambiguous_or_malformed_review_files_fail(self):
        review = self.review()
        cases = [[review, review], [{**review, "qa_pair_id": "other-project-row"}],
                 [{**review, "label": "approved"}], [{**review, "draft_sha256": "bad-hash"}],
                 [{**review, "context_sha256": None}], [None]]
        for reviews in cases:
            with self.subTest(reviews=reviews):
                with self.assertRaises(ValueError):
                    audit(self.path, "project", reviews)
        report = audit(self.path, "project", [])
        self.assertEqual(report["evidence_reviews"]["counts"], {"unreviewed": 1})

    def test_review_cli_status_depends_on_revision_and_label(self):
        root = Path(__file__).resolve().parents[2]
        path = self.path.with_name("reviews.jsonl")
        review = self.review()

        def invoke(records):
            path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            return subprocess.run(
                [sys.executable, "-m", "research.audit_drafts", "--database", str(self.path),
                 "--project-id", "project", "--reviews", str(path)], cwd=root,
                env={**os.environ, "PYTHONPATH": str(root / "backend")}, capture_output=True, text=True,
            )

        self.assertEqual(invoke([review]).returncode, 0)
        self.assertEqual(invoke([{**review, "label": "contradicted"}]).returncode, 1)
        self.assertEqual(invoke([]).returncode, 1)
        self.assertEqual(invoke([{**review, "label": "unknown"}]).returncode, 2)
        self.assertEqual(invoke([review, review]).returncode, 2)

    def packet_cli(self, path):
        root = Path(__file__).resolve().parents[2]
        return subprocess.run(
            [sys.executable, "-m", "research.audit_drafts", "--database", str(self.path),
             "--project-id", "project", "--review-packet", str(path)], cwd=root,
            env={**os.environ, "PYTHONPATH": str(root / "backend")}, capture_output=True, text=True,
        )

    def test_review_packet_preserves_exact_context_order_and_round_trips_labels(self):
        before = sha256(self.path.read_bytes()).hexdigest()
        path = self.path.with_name("review.jsonl")
        process = self.packet_cli(path)
        self.assertEqual(process.returncode, 0, process.stderr)
        row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["question"], "Fictional question?")
        self.assertEqual(row["answer"], "Fictional answer.")
        self.assertEqual(row["project_id"], "project")
        self.assertEqual(row["rendered_context"], format_evidence_for_prompt(self.chunks[::-1]))
        self.assertEqual([passage["chunk_id"] for passage in row["passages"]], ["two", "one"])
        self.assertEqual(row["passages"][1]["source_filename"], "Study Δ")
        self.assertEqual(sha256(row["rendered_context"].encode()).hexdigest(), row["context_sha256"])
        self.assertIsNone(row["label"])
        with self.assertRaises(ValueError):
            audit(self.path, "project", [row])
        report = audit(self.path, "project", [{**row, "label": "supported"}])
        self.assertEqual(report["evidence_reviews"]["counts"], {"supported": 1})
        self.assertEqual(before, sha256(self.path.read_bytes()).hexdigest())
        self.assertNotIn("Fictional", process.stdout)
        self.assertNotIn("Study Δ", process.stdout)
        self.assertEqual(json.loads(process.stdout)["review_packet"],
                         {"written_drafts": 1, "excluded_drafts": 0})
        if os.name == "posix":
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_review_packet_excludes_unverified_context_without_losing_denominator(self):
        self.add_draft("legacy", {})
        changed = deepcopy(self.manifest)
        changed["context_sha256"] = "0" * 64
        self.add_draft("changed", {"retrieved_context": changed})
        self.connection.commit()
        path = self.path.with_name("review.jsonl")
        process = self.packet_cli(path)
        self.assertEqual(process.returncode, 1, process.stderr)
        report = json.loads(process.stdout)
        self.assertEqual(report["total_drafts"], 3)
        self.assertEqual(report["review_packet"], {"written_drafts": 1, "excluded_drafts": 2})
        self.assertEqual([json.loads(line)["qa_pair_id"] for line in path.read_text().splitlines()],
                         ["draft"])

    def test_review_packet_refuses_overwrite_and_symlink(self):
        path = self.path.with_name("existing.jsonl")
        path.write_text("Existing work", encoding="utf-8")
        process = self.packet_cli(path)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stdout, "")
        self.assertEqual(path.read_text(), "Existing work")
        if os.name == "posix":
            link = path.with_name("link.jsonl")
            link.symlink_to(path)
            self.assertEqual(self.packet_cli(link).returncode, 2)
            self.assertEqual(path.read_text(), "Existing work")

    def test_standard_report_never_includes_review_text(self):
        report = self.run_audit()
        self.assertNotIn("review_packet", report)
        for text in ("Fictional", "Study Δ", self.chunks[0].content, "rendered_context"):
            self.assertNotIn(text, json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
