"""A source-disjoint evidence-verification pilot on the public SciFact dev split."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

LABELS = ("support", "contradict", "insufficient")
SCIFACT_LABELS = {"SUPPORT": "support", "CONTRADICT": "contradict"}


@dataclass(frozen=True)
class Example:
    claim_id: int
    document_id: int
    claim: str
    sentences: tuple[str, ...]
    label: str
    rationale: frozenset[int]

    @property
    def context(self) -> str:
        return " ".join(self.sentences)

    def without_rationale(self) -> Example:
        # This variant is unlabelled: unannotated sentences may still contain support.
        return replace(
            self,
            sentences=tuple(t for i, t in enumerate(self.sentences) if i not in self.rationale),
        )


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def examples(path: Path, corpus: dict[int, tuple[str, ...]]) -> list[Example]:
    result = []
    for row in read_jsonl(path):
        groups_by_doc = {int(doc): groups for doc, groups in row["evidence"].items()}
        # A cited abstract without a rationale was inspected by SciFact annotators.
        # Do not label arbitrary unjudged corpus documents as insufficient.
        for doc in sorted(set(row["cited_doc_ids"]) | groups_by_doc.keys()):
            sentences = corpus[doc]
            groups = groups_by_doc.get(doc, [])
            labels = {SCIFACT_LABELS[group["label"]] for group in groups}
            if len(labels) > 1:
                raise ValueError(f"Conflicting SciFact annotations for claim {row['id']} / {doc}")
            rationale = frozenset(position for group in groups for position in group["sentences"])
            if groups and (not rationale or any(i < 0 or i >= len(sentences) for i in rationale)):
                raise ValueError(f"Invalid rationale for claim {row['id']} / {doc}")
            result.append(Example(
                row["id"], doc, row["claim"], sentences,
                next(iter(labels), "insufficient"), rationale,
            ))
    return result


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.casefold())


def features(example: Example, paired: bool) -> dict[str, float]:
    claim_tokens = tokens(example.claim)
    values = {f"claim:{word}": 1.0 for word in claim_tokens}
    if not paired:
        return values

    claim_words = set(claim_tokens)
    context_tokens = tokens(example.context)
    context_words = set(context_tokens)
    shared = claim_words & context_words
    values.update({f"shared:{word}": 1.0 for word in shared})
    values["overlap"] = len(shared) / max(len(claim_words), 1)
    values["best_sentence_overlap"] = max(
        (len(claim_words.intersection(tokens(s))) / max(len(claim_words), 1)
         for s in example.sentences),
        default=0.0,
    )
    values["matched_number_fraction"] = (
        sum(word[0].isdigit() for word in shared)
        / max(sum(word[0].isdigit() for word in claim_words), 1)
    )
    return values


def fit(train: list[Example], paired: bool):
    vectorizer = DictVectorizer()
    matrix = vectorizer.fit_transform(features(item, paired) for item in train)
    model = LogisticRegression(class_weight="balanced", max_iter=600, random_state=0)
    model.fit(matrix, [item.label for item in train])
    return model, vectorizer


def predict(fitted, items: list[Example], paired: bool) -> list[str]:
    model, vectorizer = fitted
    raw = model.predict(vectorizer.transform(features(item, paired) for item in items))
    # A paired verifier cannot claim support or contradiction with an empty input.
    return ["insufficient" if paired and not item.context else str(label)
            for item, label in zip(items, raw, strict=True)]


def support_scores(fitted, items: list[Example], paired: bool) -> list[float]:
    model, vectorizer = fitted
    matrix = vectorizer.transform(features(item, paired) for item in items)
    support_index = list(model.classes_).index("support")
    return model.predict_proba(matrix)[:, support_index].tolist()


def rank_at_fixed_coverage(items: list[Example], scores: list[float]) -> list[dict]:
    """Compare score rankings at equal observed coverage without fitting a threshold."""
    ranked = sorted(
        zip(items, scores, strict=True),
        key=lambda pair: (-pair[1], pair[0].claim_id, pair[0].document_id),
    )
    supported = sum(item.label == "support" for item in items)
    results = []
    for fraction in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        selected = ranked[:max(1, round(fraction * len(items)))]
        true_accepts = sum(item.label == "support" for item, _ in selected)
        results.append({
            "target_coverage": fraction,
            "accepted": len(selected),
            "false_accepts": len(selected) - true_accepts,
            "false_accept_fraction": 1 - true_accepts / len(selected),
            "supported_retention": true_accepts / supported if supported else None,
        })
    return results


def score(items: list[Example], predictions: list[str]) -> dict:
    actual = [item.label for item in items]
    counts = Counter(zip(actual, predictions, strict=True))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predictions, labels=list(LABELS), zero_division=0,
    )
    accepted = sum(value == "support" for value in predictions)
    true_accepted = counts["support", "support"]
    return {
        "accuracy": sum(counts[label, label] for label in LABELS) / len(items),
        "macro_f1": float(f1.mean()),
        "support_gate": {
            "accepted": accepted,
            "false_accepts": accepted - true_accepted,
            "false_accepts_contradicted": counts["contradict", "support"],
            "false_accepts_insufficient": counts["insufficient", "support"],
            "false_accept_fraction": (accepted - true_accepted) / accepted if accepted else None,
            "supported_retention": float(recall[0]),
            "coverage": accepted / len(items),
        },
        "by_label": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "count": int(n)}
            for label, p, r, f, n in zip(LABELS, precision, recall, f1, support, strict=True)
        },
        "confusion": {
            actual_label: {guess: counts[actual_label, guess] for guess in LABELS}
            for actual_label in LABELS
        },
    }


def withholding(
    items: list[Example],
    predictions: list[str],
    predict_items: Callable[[list[Example]], list[str]],
) -> dict:
    supported, before = [], []
    for item, prediction in zip(items, predictions, strict=True):
        if item.label == "support":
            supported.append(item)
            before.append(prediction)
    if not supported:
        return {
            "annotated_support_cases": 0,
            "support_predictions_original": 0,
            "support_predictions_after_rationale_removal": 0,
            "original_support_predictions_retained_after_removal": 0,
            "support_predictions_with_empty_context": 0,
        }
    without = predict_items([item.without_rationale() for item in supported])
    empty = predict_items([replace(item, sentences=()) for item in supported])
    kept = sum(label == "support" for label in before)
    return {
        "annotated_support_cases": len(supported),
        "support_predictions_original": kept,
        "support_predictions_after_rationale_removal": sum(label == "support" for label in without),
        "original_support_predictions_retained_after_removal": sum(
            a == b == "support" for a, b in zip(before, without, strict=True)
        ),
        "support_predictions_with_empty_context": sum(label == "support" for label in empty),
    }


def sha256_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def audit(data: Path, nli_model_dir: Path | None = None) -> dict:
    corpus_path = data / "corpus.jsonl"
    train_path = data / "claims_train.jsonl"
    dev_path = data / "claims_dev.jsonl"
    corpus = {row["doc_id"]: tuple(row["abstract"]) for row in read_jsonl(corpus_path)}
    train_all = examples(train_path, corpus)
    dev = examples(dev_path, corpus)
    held_out_documents = {item.document_id for item in dev}
    held_out_claims = {tuple(tokens(item.claim)) for item in dev}
    # Drop the entire training claim when any of its documents cross the boundary.
    excluded_claims = {
        item.claim_id for item in train_all
        if item.document_id in held_out_documents or tuple(tokens(item.claim)) in held_out_claims
    }
    train = [item for item in train_all if item.claim_id not in excluded_claims]
    claim_only = fit(train, paired=False)
    claim_context = fit(train, paired=True)

    def presence(items: list[Example]) -> list[str]:
        return ["support" if item.context else "insufficient" for item in items]

    def claim_model(items: list[Example]) -> list[str]:
        return predict(claim_only, items, paired=False)

    def paired_model(items: list[Example]) -> list[str]:
        return predict(claim_context, items, paired=True)

    methods: dict[str, Callable[[list[Example]], list[str]]] = {
        "context_presence": presence, "claim_only": claim_model, "claim_context": paired_model,
    }
    predictions = {name: fn(dev) for name, fn in methods.items()}
    rankings = {
        "claim_only": rank_at_fixed_coverage(dev, support_scores(claim_only, dev, paired=False)),
        "claim_context": rank_at_fixed_coverage(dev, support_scores(claim_context, dev, paired=True)),
    }
    model_sha256 = {}
    if nli_model_dir is not None:
        from research.nli import DebertaNLI

        nli = DebertaNLI(nli_model_dir)
        methods["deberta_nli"] = nli
        labels, confidence = nli.classify(dev)
        predictions["deberta_nli"] = labels
        rankings["deberta_nli"] = rank_at_fixed_coverage(dev, confidence)
        model_sha256 = {
            name: sha256_file(nli_model_dir / name)
            for name in ("model.onnx", "tokenizer.json", "config.json")
        }

    return {
        "protocol": "medqa-forge-scifact-v1",
        "nli_files_sha256": model_sha256,
        "scikit_learn": sklearn.__version__,
        "inputs_sha256": {
            path.name: sha256_file(path)
            for path in (corpus_path, train_path, dev_path)
        },
        "train_count": len(train),
        "excluded_train_claims": len(excluded_claims),
        "dev_count": len(dev),
        "dev_claims": len({item.claim_id for item in dev}),
        "dev_documents": len(held_out_documents),
        "natural_context": {name: score(dev, labels) for name, labels in predictions.items()},
        "fixed_coverage_rankings": rankings,
        "withholding_diagnostic": {
            name: withholding(dev, predictions[name], fn) for name, fn in methods.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Extracted SciFact data directory")
    parser.add_argument("--output", type=Path, help="Write aggregate JSON instead of printing")
    parser.add_argument("--nli-model-dir", type=Path, help="Optional local ONNX model directory")
    args = parser.parse_args()
    output = json.dumps(audit(args.data, args.nli_model_dir), sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
