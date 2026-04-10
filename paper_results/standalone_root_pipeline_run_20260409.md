# Standalone Root Pipeline Run

## Goal

Run the full standalone crawler pipeline in the repository root, not in a temporary directory, so that the final artifacts are written into the real project folders:

- `./paper_results`
- `./md`
- `./faiss`

## Actual Runtime Parameters

### missing_aspects

- `the definition of RNN`
- `the definition of CNN`

### Remote service config

- SSH host: `172.26.19.131`
- SSH port: `8888`
- SSH username: `root`
- SSH password: `123456.a`
- LLM remote/local port: `8001 -> 18001`
- embedding remote/local port: `8000 -> 18000`
- OCR remote/local port: `8002 -> 18002`

### Command Used

```powershell
& 'C:\Users\jack\miniconda3\shell\condabin\conda-hook.ps1'
conda activate D:\conda_envs\pyth310_new
python -m rag.crawlers.standalone `
  --aspect "the definition of RNN" `
  --aspect "the definition of CNN" `
  --use-ssh `
  --ssh-host 172.26.19.131 `
  --ssh-port 8888 `
  --ssh-username root `
  --ssh-password "123456.a" `
  --llm-remote-port 8001 `
  --llm-local-port 18001 `
  --embedding-remote-port 8000 `
  --embedding-local-port 18000 `
  --ocr-remote-port 8002 `
  --ocr-local-port 18002 `
  --output-dir "./paper_results" `
  --md-output-dir "./md" `
  --queue-path "./paper_results/pending_aspects.json" `
  --max-pages 1 `
  --max-new-papers 2 `
  --print-json
```

## First Root Run

### 1. LLM rewrite

Observed arXiv queries:

- `"recurrent neural networks are"`
- `"convolutional neural networks are"`

This confirms the standalone crawler really used the remote LLM rewrite stage before searching arXiv.

### 2. arXiv crawl

Observed logs:

```text
Searching arXiv for relevant papers...
Generated arXiv query: "recurrent neural networks are"
Finished arXiv crawl with 50 papers.
Searching arXiv for relevant papers...
Generated arXiv query: "convolutional neural networks are"
Finished arXiv crawl with 50 papers.
```

### 3. Download + OCR

Observed logs:

```text
[1/2] Downloading: arXiv:2604.06469 [ pdf , ps , other ]
[2/2] Downloading: arXiv:2603.23814 [ pdf , ps , other ]
```

Generated root artifacts:

- PDFs:
  - `paper_results/arXiv2603.23814 [ pdf , ps , other ].pdf`
  - `paper_results/arXiv2604.06469 [ pdf , ps , other ].pdf`
- markdown:
  - `md/arXiv2603.23814 [ pdf , ps , other ].md`
  - `md/arXiv2604.06469 [ pdf , ps , other ].md`
- markdown sidecars:
  - `md/arXiv2603.23814 [ pdf , ps , other ].metadata.json`
  - `md/arXiv2604.06469 [ pdf , ps , other ].metadata.json`
- paper manifests:
  - `paper_results/paper_result.csv`
  - `paper_results/formatted_papers.txt`

### 4. Embedding + FAISS

Observed logs:

```text
Loading embedding model...
Embedding service connected: http://127.0.0.1:18000/v1 model=bge-m3.
Loading markdown documents...
Loaded arXiv2603.23814 [ pdf , ps , other ].md
Loaded arXiv2604.06469 [ pdf , ps , other ].md
Loaded test.md
Semantic chunking failed, using fallback chunking: 400 Client Error: Bad Request for url: http://127.0.0.1:18000/v1/embeddings
Fallback chunking finished with 150 chunks.
FAISS index rebuilt.
Read 150 chunks from FAISS.
```

Important note for this first run:

- the final FAISS rebuild succeeded in the real root `faiss/` directory
- semantic chunking hit a remote embedding `400 Bad Request`
- the code then automatically fell back to the fallback chunking strategy
- FAISS was still rebuilt successfully

## Root Output Verification For The First Run

### `faiss/`

Observed root files:

- `faiss/index.faiss`
- `faiss/index.pkl`

Observed timestamps and sizes after the run:

- `index.faiss`: updated at `2026-04-09 23:10`, size `614445`
- `index.pkl`: updated at `2026-04-09 23:10`, size `150565`

### `md/`

Observed root markdown files:

- `md/arXiv2603.23814 [ pdf , ps , other ].md`
- `md/arXiv2603.23814 [ pdf , ps , other ].metadata.json`
- `md/arXiv2604.06469 [ pdf , ps , other ].md`
- `md/arXiv2604.06469 [ pdf , ps , other ].metadata.json`

### `paper_results/`

Observed root paper files:

- `paper_results/arXiv2603.23814 [ pdf , ps , other ].pdf`
- `paper_results/arXiv2603.23814 [ pdf , ps , other ].metadata.json`
- `paper_results/arXiv2604.06469 [ pdf , ps , other ].pdf`
- `paper_results/arXiv2604.06469 [ pdf , ps , other ].metadata.json`
- `paper_results/paper_result.csv`
- `paper_results/formatted_papers.txt`

## Final Status For The First Run

The repository-root pipeline did complete the requested end-to-end flow and produced a real FAISS index in the root `faiss/` directory.

### Confirmed

- remote LLM rewrite: yes
- arXiv crawl: yes
- PDF download: yes
- remote OCR to markdown: yes
- remote embedding service used: yes
- FAISS rebuilt in root `faiss/`: yes

### Important caveat

The rebuild succeeded through the fallback chunking path rather than the semantic chunking path, because semantic chunking received a `400` from the embedding endpoint during chunk-boundary computation.

The final FAISS index is still real and present in the root project directory.

## Why the process returned exit code 1

The command used `--print-json`, and after the pipeline had already succeeded, Python tried to print the final JSON to the Windows console using `gbk` encoding.

That caused:

- `UnicodeEncodeError: 'gbk' codec can't encode character ...`

This happened after the actual pipeline artifacts had already been written.

So:

- the pipeline work succeeded
- the final console print failed

## Recommended rerun command

To avoid the same Windows console encoding issue next time, use:

```powershell
& 'C:\Users\jack\miniconda3\shell\condabin\conda-hook.ps1'
conda activate D:\conda_envs\pyth310_new
$env:PYTHONIOENCODING = "utf-8"
python -m rag.crawlers.standalone `
  --aspect "the definition of RNN" `
  --aspect "the definition of CNN" `
  --use-ssh `
  --ssh-host 172.26.19.131 `
  --ssh-port 8888 `
  --ssh-username root `
  --ssh-password "123456.a" `
  --llm-remote-port 8001 `
  --llm-local-port 18001 `
  --embedding-remote-port 8000 `
  --embedding-local-port 18000 `
  --ocr-remote-port 8002 `
  --ocr-local-port 18002 `
  --output-dir "./paper_results" `
  --md-output-dir "./md" `
  --queue-path "./paper_results/pending_aspects.json" `
  --max-pages 1 `
  --max-new-papers 2 `
  --print-json
```

## Follow-up Root Rerun After The Chunking Fix

After the first run above, the chunking path was updated so that markdown is pre-split to about `6000` characters before semantic chunking.

Before rerunning:

- the repository-root `md/` directory was cleared
- the repository-root `faiss/` directory was cleared
- the crawler-produced root PDFs and sidecars that would trigger dedupe skips were removed from `paper_results/`

The rerun used the same missing aspects and the same remote SSH parameters as above.

### Observed Healthy Logs

```text
Generated arXiv query: "recurrent neural networks are"
Generated arXiv query: "convolutional neural networks are"
Loading embedding model...
Embedding service connected: http://127.0.0.1:18000/v1 model=bge-m3.
Loaded arXiv2603.23814 [ pdf , ps , other ].md
Loaded arXiv2604.06469 [ pdf , ps , other ].md
Semantic pre-chunking prepared 22 docs.
Semantic chunking finished with 211 chunks.
FAISS index rebuilt.
```

### Observed Output Summary

```json
{
  "requested_aspects": [
    "the definition of RNN",
    "the definition of CNN"
  ],
  "rewrites": [
    {
      "original_aspect": "the definition of RNN",
      "optimized_query_en": "recurrent neural networks are"
    },
    {
      "original_aspect": "the definition of CNN",
      "optimized_query_en": "convolutional neural networks are"
    }
  ],
  "ingestion_result": {
    "status": "success",
    "download_success_count": 2,
    "ocr_success_count": 2,
    "md_written_count": 2,
    "rebuild_success": true
  }
}
```

### Result

The latest repository-root rerun is the one that should now be treated as the current healthy baseline.

Confirmed in this rerun:

- remote LLM rewrite: yes
- rewritten queries used for arXiv search: yes
- PDF download: yes
- remote OCR to markdown: yes
- remote embedding service used: yes
- semantic chunking path succeeded: yes
- FAISS rebuilt in root `faiss/`: yes

### Root Artifacts After The Latest Rerun

- `md/arXiv2603.23814 [ pdf , ps , other ].md`
- `md/arXiv2603.23814 [ pdf , ps , other ].metadata.json`
- `md/arXiv2604.06469 [ pdf , ps , other ].md`
- `md/arXiv2604.06469 [ pdf , ps , other ].metadata.json`
- `faiss/index.faiss`
- `faiss/index.pkl`

### Chunking Behavior Now

The current indexing flow is:

1. pre-split cleaned markdown to about `6000` characters with overlap
2. run `SemanticChunker` on those pre-split documents
3. if any semantic chunk is still too large, split it again with the existing `800 / 150` character settings

This keeps the later FAISS rebuild logic unchanged while preventing the earlier semantic-chunking request from exceeding the remote embedding service limits.
