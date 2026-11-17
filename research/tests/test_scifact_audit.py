import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from research.scifact_audit import Example, examples, score, withholding


class SciFactAuditTests(unittest.TestCase):
    def test_only_labelled_or_cited_documents_become_examples(self):
        row = {
            "id": 1,
            "claim": "The median fell.",
            "evidence": {"20": [{"label": "SUPPORT", "sentences": [1]}]},
            "cited_doc_ids": [20, 30],
        }
        corpus = {
            20: ("The authors measured the median.", "The median fell."),
            30: ("The authors measured the mean.",),
            40: ("The authors measured the mode.",),
        }
        with TemporaryDirectory() as directory:
            path = Path(directory, "claims.jsonl")
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            annotated, insufficient = examples(path, corpus)

        self.assertEqual(annotated.label, "support")
        self.assertEqual(annotated.without_rationale().context, corpus[20][0])
        self.assertEqual(insufficient.document_id, 30)
        self.assertEqual(insufficient.label, "insufficient")

    def test_claim_gate_metrics_count_unsupported_and_contradicted_accepts(self):
        items = [
            Example(i, i, "A claim", ("A passage.",), label, frozenset())
            for i, label in enumerate(("support", "contradict", "insufficient"))
        ]
        gate = score(items, ["support", "support", "support"])["support_gate"]
        self.assertEqual(gate["false_accepts"], 2)
        self.assertEqual(gate["false_accepts_contradicted"], 1)
        self.assertEqual(gate["false_accepts_insufficient"], 1)
        self.assertEqual(gate["supported_retention"], 1.0)
        self.assertEqual(withholding(items[1:], ["support", "support"], lambda _: [])[
            "annotated_support_cases"
        ], 0)
