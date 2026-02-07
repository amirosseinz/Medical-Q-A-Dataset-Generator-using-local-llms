# User Guide

## Getting Started

After installation (see [INSTALL.md](INSTALL.md)), open **http://localhost:3000** in your browser.

## Workflow Overview

```
1. Create Project  →  2. Upload Sources  →  3. Configure & Generate  →  4. Review Q&A Pairs  →  5. Export Dataset
```

## 1. Create a Project

1. Click **New Project** on the Dashboard
2. Enter a **name** (e.g., "Cardiology QA Dataset")
3. Enter the **medical domain** (e.g., "cardiology")
4. Optionally add a description
5. Click **Create Project**

## 2. Upload Sources

In your project, go to the **Sources** tab:

- **Drag & drop** files onto the upload area, or click to browse
- Supported formats:
  - **PDF** — Medical papers, textbooks, guidelines
  - **XML** — MedQuAD format question-answer datasets
  - **DOCX** — Word documents with medical content

Files are processed on upload — text is extracted and chunked automatically.

## 3. Generate Q&A Pairs

Click **Generate** to open the configuration dialog:

| Setting | Description | Default |
|---------|-------------|---------|
| **Model** | Ollama model to use | First available |
| **Pairs per Chunk** | Q&A pairs generated per text chunk | 3 |
| **Difficulty** | Question difficulty level | Mixed |
| **Chunk Size** | Words per text chunk | 500 |
| **Temperature** | LLM creativity (0 = deterministic, 1.5 = creative) | 0.7 |
| **Min Quality Score** | Minimum quality to accept a pair | 60% |
| **Include PubMed** | Fetch additional articles from PubMed | Off |

Click **Start Generation** to begin. Progress is shown in real-time.

### Generation Pipeline

1. **Process XML sources** — Extract Q&A from MedQuAD format
2. **Process documents** — Extract text from PDFs/DOCX, chunk into segments
3. **Fetch PubMed** — (if enabled) Download and chunk PubMed abstracts
4. **Store direct Q&A** — Save pre-existing Q&A from XML sources
5. **AI Generation** — Send chunks to Ollama, parse Q&A responses
6. **Quality check** — Score each pair, reject below threshold

## 4. Review Q&A Pairs

Click **Review Q&A Pairs** to enter the review interface:

### Filtering & Search
- **Search** — Full-text search across questions and answers
- **Status filter** — All, Pending, Approved, Rejected, Needs Review
- **Source filter** — PDF, XML, PubMed, DOCX
- **Sort** — By date, quality score, or status

### Individual Actions
- **Approve** (✓) — Mark as ready for export
- **Reject** (✗) — Mark as unusable
- **Edit** (pencil icon) — Modify question or answer text

### Batch Actions
1. Select multiple pairs using checkboxes
2. Click **Approve**, **Reject**, or **Needs Review** in the batch action bar

## 5. Export Dataset

Go to the **Export** tab in your project:

| Format | Description |
|--------|-------------|
| CSV | Simple tabular format |
| JSON | Structured JSON array |
| JSONL | One JSON object per line |
| Alpaca | Stanford Alpaca training format |
| OpenAI | OpenAI fine-tuning format |
| Parquet | Apache Parquet (columnar) |

Click **Export** to download the file.

## Analytics

The **Analytics** page provides:

- **Validation Status** — Pie chart of approved/rejected/pending pairs
- **Source Distribution** — Where your Q&A pairs come from
- **Quality Distribution** — Histogram of quality scores
- **Summary metrics** — Total pairs, approval rate, average quality

## Settings

The **Settings** page shows:

- **Ollama connection status** — Whether the LLM server is reachable
- **Available models** — List of installed Ollama models with sizes
- **Application info** — Version and tech stack details

## Tips

- **Quality scores** above 70% are generally good; below 40% usually need rejection
- **Use mixed difficulty** for the most diverse dataset
- **PubMed integration** provides abstracts only — best for factual Q&A
- **XML sources** (MedQuAD) provide pre-existing Q&A pairs that skip the AI generation step
- **Chunk size of 300-500 words** typically produces the best results
- **Temperature 0.5-0.8** balances accuracy with variety
