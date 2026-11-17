"""Optional CPU baseline using a pinned DeBERTa ONNX model."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

if TYPE_CHECKING:
    from pathlib import Path

    from research.scifact_audit import Example

LABELS = {"entailment": "support", "contradiction": "contradict", "neutral": "insufficient"}


class DebertaNLI:
    def __init__(self, model_dir: Path) -> None:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.labels = {int(index): LABELS[label.lower()] for index, label in config["id2label"].items()}
        if set(self.labels) != set(range(3)) or set(self.labels.values()) != set(LABELS.values()):
            raise ValueError("Expected an entailment, contradiction and neutral label mapping")

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512, strategy="only_first")
        pad_token = "[PAD]"
        pad_id = self.tokenizer.token_to_id(pad_token)
        if pad_id is None:
            raise ValueError("Tokenizer must define [PAD]")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.inputs = {node.name for node in self.session.get_inputs()}

    def __call__(self, items: list[Example]) -> list[str]:
        return self.classify(items)[0]

    def classify(self, items: list[Example]) -> tuple[list[str], list[float]]:
        labels = ["insufficient"] * len(items)
        confidence = [0.0] * len(items)
        support_index = next(index for index, label in self.labels.items() if label == "support")
        nonempty = [(index, item) for index, item in enumerate(items) if item.context]
        for start in range(0, len(nonempty), 8):
            batch = nonempty[start:start + 8]
            encoded = self.tokenizer.encode_batch([(item.context, item.claim) for _, item in batch])
            inputs = {
                name: np.array([getattr(encoding, attribute) for encoding in encoded], dtype=np.int64)
                for name, attribute in (
                    ("input_ids", "ids"), ("attention_mask", "attention_mask"), ("token_type_ids", "type_ids"),
                )
                if name in self.inputs
            }
            logits = self.session.run(["logits"], inputs)[0]
            exponents = np.exp(logits - logits.max(axis=1, keepdims=True))
            support_scores = exponents[:, support_index] / exponents.sum(axis=1)
            for (index, _), label_id, support_score in zip(
                batch, logits.argmax(axis=1), support_scores, strict=True,
            ):
                labels[index] = self.labels[int(label_id)]
                confidence[index] = float(support_score)
        return labels, confidence
