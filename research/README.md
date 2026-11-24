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

## Export splits by cited source

Before splitting, evidence-review labels can be summarised against the project's current draft and context revisions using the audit command's optional `--reviews` argument. See the review protocol below.

The export endpoint accepts `split_strategy: "source"` to keep every recorded source ID in a single split. A row citing sources A and B connects their other rows into one indivisible group. The service uses all source IDs in `retrieved_context.passages`, with `citations` as a fallback for older rows without a context manifest. Missing or incomplete source IDs return HTTP 422. This groups recorded provenance and does not establish factual support for a draft.

```bash
curl -X POST http://localhost:8000/api/v1/projects/YOUR_PROJECT_ID/export \
  -H 'Content-Type: application/json' \
  -d '{"format":"jsonl","split_strategy":"source","train_split":0.8,"val_split":0.1,"test_split":0.1}'
```

The response includes `split_strategy` and `split_counts` alongside the download filenames. Fractions are approximate because source groups remain intact. A large connected group can leave validation or test empty, so inspect the returned counts before evaluating. The API sorts rows by ID and uses a fixed local seed for reproducible assignment on the same candidate set. The default `row` strategy retains the existing row-shuffle behaviour. In the project's Export tab, select **Keep cited sources together** to use this option for project and generation exports. A successful download reports the actual split counts. A failed split download reports an error and identifies the file to retry.

Different source IDs can refer to the same article, and overlapping text can appear in different documents. Canonical DOI/PMID resolution and duplicate-text checks are still needed for a study claiming generalisation to unseen articles. Split creation alone does not establish separation from a model's pretraining data.

The browser client's export regression tests run with `node --test tests/export-api.test.mjs` from `frontend/` after installing its dependencies. They cover source strategy submission, actual split counts, failed downloads, missing-source errors, empty file lists and single-file exports.

## Summarise evidence-review labels

First run the draft audit to obtain each row's `qa_pair_id`, `draft_sha256` and, for matching contexts, `context_sha256`. A reviewer assesses every material claim in the question and answer against the recorded evidence. Preserve those three identifiers alongside the adjudicated label in a local JSONL file, with one object per row. Supported means every material claim is supported within its stated population, outcome and uncertainty. Use `contradicted` for a materially contradicted claim, `insufficient` for an unsupported claim or presupposition, and `conflicting` for unresolved opposing evidence. Resolve disagreements into one final label per draft before summarising.

```json
{"qa_pair_id":"COPY_ROW_ID","draft_sha256":"COPY_64_CHARACTER_DRAFT_HASH","context_sha256":"COPY_64_CHARACTER_CONTEXT_HASH","label":"supported"}
```

The placeholders above must be replaced with identifiers from the audit. Then run the following command using the actual database and project ID.

```bash
PYTHONPATH=backend python -m research.audit_drafts \
  --database backend/data/medqa.db --project-id YOUR_PROJECT_ID --reviews reviews.jsonl
```

The report adds `review_status` per row and an `evidence_reviews` summary. Coverage uses all project drafts as its denominator. The supported fraction uses only current reviews with matching source contexts. Edited drafts and changed context hashes produce `stale_review`; a context that cannot be verified produces `context_unverified`. Missing labels remain `unreviewed`. Duplicate or unknown row IDs, invalid hashes and unknown labels fail with exit code 2. With `--reviews`, code 0 requires all project drafts to have current supported labels and matching contexts; incomplete or adverse results return code 1. Without `--reviews`, the original source-integrity exit codes apply.

This command validates label alignment and summarises supplied judgments. It does not assign evidence labels or independently validate reviewer judgments. Counts from a selectively reviewed sample describe that sample, and cannot establish the unsupported rate for the whole project. Tests use fictional temporary database fixtures. Real dataset review still requires the deployment database and independent annotations.

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
