# Evidence verification pilot

Can a verifier distinguish supported, contradicted and insufficient claims when it sees the cited scientific abstract? The pilot compares a context-presence sanity baseline, a claim-only lexical control, a claim-and-context lexical classifier, and an optional pretrained DeBERTa natural-language inference model. The production generation path records retrieval provenance. The pilot remains an offline evaluation.

## Audit stored generated drafts

Run the context audit against the application's SQLite database from the repository root. Supply the project ID from the application. Python 3.11+ is sufficient for this command, with no additional packages required.

```bash
PYTHONPATH=backend python -m research.audit_drafts \
  --database backend/data/medqa.db --project-id YOUR_PROJECT_ID
```

Use the actual path to your deployment's database. The command opens it read-only and holds a single read transaction while checking the project's drafts. Each manifest is checked against the stored chunk contents, owning source, filename and passage order using the generator's evidence formatter. A missing or modified passage fails the audit. Legacy rows without a manifest and malformed manifests remain in the report's denominator. The report identifies each draft by its database ID and a hash of its current question and answer. It omits question text, answer text and evidence text.

Exit code 0 means every draft's retrieval context matches the stored sources. Code 1 means at least one draft requires attention, or the project has no drafts. Code 2 indicates an input or database error. A matching context establishes that the stored passages can reproduce the recorded context. Factual support for the answer still requires a separate claim-level assessment. Editing a draft changes its revision hash while its context may continue to match. Historical rows cannot recover missing generation-time manifests through this audit.

The audit has been verified with temporary database fixtures using manifests produced by the current generator, including rotated Unicode passages, changed text and filenames, missing and cross-project sources, malformed metadata, and CLI exit codes. Real generated datasets still require access to the deployment database. Reproduce the fixture checks with the command below.

```bash
PYTHONPATH=backend python -m unittest research.tests.test_audit_drafts -v
```

## Reproduce

From the repository root with Python 3.12 on x86-64 with AVX2:

```bash
python -m pip install -r research/requirements-nli.txt
work_dir="$(mktemp -d)"
mkdir -p "$work_dir/model"
curl -fsSL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o "$work_dir/scifact.tar.gz"
tar -xzf "$work_dir/scifact.tar.gz" -C "$work_dir" \
  data/corpus.jsonl data/claims_train.jsonl data/claims_dev.jsonl

model_base=https://huggingface.co/cross-encoder/nli-deberta-v3-small/resolve/fa2804872c3b4bd748f38c0185cc85775361e735
curl -fsSL "$model_base/config.json" -o "$work_dir/model/config.json"
curl -fsSL "$model_base/tokenizer.json" -o "$work_dir/model/tokenizer.json"
curl -fsSL "$model_base/onnx/model_quint8_avx2.onnx?download=true" -o "$work_dir/model/model.onnx"

OPENBLAS_NUM_THREADS=1 python -m research.scifact_audit \
  --data "$work_dir/data" --nli-model-dir "$work_dir/model" --output "$work_dir/results.json"
```

For the lexical baselines, install `research/requirements.txt` and omit `--nli-model-dir`. The [recorded run](scifact_results.json) contains model and data SHA-256 hashes, denominators, confusion matrices, and coverage measurements. Source data and weights are obtained from their upstream projects. [SciFact paper](https://aclanthology.org/2020.emnlp-main.609/) · [SciFact data schema](https://github.com/allenai/scifact/blob/master/doc/data.md) · [SciFact licence](https://github.com/allenai/scifact/blob/master/LICENSE.md) · [DeBERTa model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small).

## Protocol

- Evaluate on the 339 labelled claim–abstract pairs derived from 300 public SciFact development claims. Use the cited abstracts and the abstracts with annotated rationales. Treat a cited abstract without annotated evidence as insufficient under the SciFact schema. Leave unrelated, unjudged abstracts out.
- For both trained lexical baselines, drop every training claim with a document or normalised claim appearing on the development side. This leaves 609 train pairs after dropping 272 training claims.
- The supplied DeBERTa checkpoint was fine-tuned on general-domain NLI and evaluated as supplied, with up to 512 tokens per abstract–claim pair. The input preserves the claim when truncating the abstract. The reported model has AVX2 quantisation. Model pretraining overlap is unknown.
- The default decision accepts a claim if its predicted class is support. The fixed-coverage comparison ranks the same full set by each method's support score. These scores are uncalibrated and the ranking is retrospective. Ties break by claim and document ID. A deployable confidence threshold requires a separate calibration split.
- Remove the annotated supporting sentences from supported abstracts and count changes in the decisions. The remaining text may still support the claim. Empty context is a separate sanity intervention. The context-aware baselines explicitly treat empty input as insufficient.

## Results

| Method | Macro F1 | Accepted | False accepts among accepted | Contradicted claims accepted | Supported claims retained |
|---|---:|---:|---:|---:|---:|
| Context present | 0.193 | 339 | 201/339 (59.3%) | 71 | 100.0% |
| Claim-only lexical | 0.384 | 165 | 85/165 (51.5%) | 34 | 58.0% |
| Claim and context lexical | 0.485 | 152 | 67/152 (44.1%) | 35 | 61.6% |
| DeBERTa NLI | 0.487 | 70 | 24/70 (34.3%) | 11 | 33.3% |

**At matched coverage the apparent DeBERTa advantage largely disappeared.** When the three scored methods each retained 68 of 339 pairs (about 20%), the claim-and-context lexical model and DeBERTa each accepted 24 unsupported or contradicted pairs. The claim-only control accepted 35. This is a retrospective ranking comparison at a fixed count.

DeBERTa accepted 46 of 138 supported pairs with the original abstracts. It kept 16 of those 46 support decisions after the annotated rationale sentences had been removed. The residual sentences may still support some claims. Empty context separately triggered an explicit abstention rule.

The paired-deletion numbers diagnose sensitivity to annotated sentences. A retained support decision after deletion can reflect redundant evidence or a model error. The pilot uses cited abstracts and a public development split. It does not measure clinical accuracy, retrieval quality, or the error rate of the app's generated exports. A publication-grade comparison would relabel variants, predeclare operating points on independent document families, and measure uncertainty on a new human-reviewed sample of generated rows.
