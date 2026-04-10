# Standalone Remote Pipeline Runbook

## Purpose

This document records:

1. whether the full standalone pipeline from crawler to FAISS is implemented
2. one real end-to-end verification result
3. the exact remote runtime configuration and commands for future reruns

The verified pipeline is:

`standalone crawler -> remote LLM rewrite -> arXiv search -> PDF download -> remote OCR -> md -> remote BGE embeddings -> FAISS rebuild`

## Conclusion

Yes, this full pipeline is implemented and was verified by a real run on `2026-04-09`.

It was then re-verified again after updating the chunking path to:

- pre-split markdown to about `6000` characters first
- then run semantic chunking
- then do the existing secondary split only when a semantic chunk is still too large

### Static code path

The implementation path is:

- `rag/crawlers/standalone.py`
  - `run_aspects()`
  - builds rewrites
  - calls `crawl_and_collect(...)`
- `rag/crawlers/arxiv.py`
  - `crawl_and_collect(...)`
  - prepares the ingest shortlist
  - `execute_ingestion_job(...)`
- `rag/pdf_processor.py`
  - `process_pdf(...)`
  - calls remote OCR through `create_default_ocr_client()`
- `rag/rag_system.py`
  - `setup_embeddings()`
  - connects to remote embedding service
  - `update_rag_system(...)`
  - reads markdown and rebuilds `./faiss`

### Real end-to-end verification

One isolated end-to-end run was executed successfully with remote SSH-backed services.

Run id:

- `standalone_e2e_remote_af2a83f7`

Temporary working directory:

- `C:\Users\jack\Desktop\demo\.tmp\standalone_e2e_remote_af2a83f7`

Observed result summary:

```json
{
  "requested_aspects": [
    "the definition of RNN"
  ],
  "rewrites": [
    {
      "original_aspect": "the definition of RNN",
      "optimized_query_en": "recurrent neural networks are"
    }
  ],
  "search_queries": [
    {
      "aspect": "the definition of RNN",
      "query": "\"recurrent neural networks are\""
    }
  ],
  "ingestion_result": {
    "status": "success",
    "download_success_count": 1,
    "ocr_success_count": 1,
    "md_written_count": 1,
    "rebuild_success": true
  },
  "ocr_base_url": "http://127.0.0.1:18002/v1",
  "ocr_model": "paddle-ocr-vl",
  "embedding_class": "VLLMOpenAIEmbeddings",
  "vectorstore_initialized": true,
  "faiss_doc_count": 59,
  "faiss_files": [
    "faiss/index.faiss",
    "faiss/index.pkl"
  ]
}
```

This verifies that:

- standalone crawler really calls the remote LLM rewrite stage
- rewritten query is actually used for arXiv search
- one paper was downloaded
- remote OCR generated markdown successfully
- remote `bge-m3` embeddings were used
- FAISS was rebuilt successfully

## Remote Runtime Configuration

### Python

Use:

```powershell
conda activate D:\conda_envs\pyth310_new
```

### SSH / service ports

Use this shared remote configuration:

- SSH host: `172.26.19.131`
- SSH port: `8888`
- SSH username: `root`
- SSH password: `123456.a`
- remote LLM port: `8001`
- local LLM port: `18001`
- remote embedding port: `8000`
- local embedding port: `18000`
- remote OCR port: `8002`
- local OCR port: `18002`

## Recommended Command

### Safe isolated run

This is the recommended way to rerun the full pipeline without overwriting the repository root `faiss/`, `md/`, or `paper_results/`.

Run from PowerShell:

```powershell
$repo = "C:\Users\jack\Desktop\demo"
$runId = "standalone_remote_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$runDir = Join-Path $repo ".tmp\$runId"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

Push-Location $runDir
$env:PYTHONPATH = $repo

& "D:\conda_envs\pyth310_new\python.exe" -m rag.crawlers.standalone `
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
  --queue-path "./pending_aspects.json" `
  --max-pages 1 `
  --max-new-papers 2 `
  --print-json

Pop-Location
```

### What this command does

- imports code from `C:\Users\jack\Desktop\demo`
- writes runtime outputs into the temporary run directory
- rewrites aspects with the remote LLM
- searches arXiv
- downloads selected PDFs
- calls remote OCR
- writes markdown into `./md`
- rebuilds `./faiss` inside the temporary run directory

## Direct Run In Repo Root

Use this only if you intentionally want to update the repository root outputs:

- `./paper_results`
- `./md`
- `./faiss`

```powershell
Set-Location "C:\Users\jack\Desktop\demo"

& "D:\conda_envs\pyth310_new\python.exe" -m rag.crawlers.standalone `
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
  --max-pages 1 `
  --max-new-papers 2 `
  --print-json
```

## Expected Healthy Signals

When the remote pipeline is working, logs should look roughly like this:

```text
Generated arXiv query: "recurrent neural networks are"
Loading embedding model...
Embedding service connected: http://127.0.0.1:18000/v1 model=bge-m3.
Semantic pre-chunking prepared <N> docs.
Loaded <paper>.md
Semantic chunking finished with <N> chunks.
FAISS index rebuilt.
```

And the final JSON should contain:

- `rewrites[].optimized_query_en` with rewritten academic search phrases
- `ingestion_result.status = "success"` or `partial_success`
- `ingestion_result.rebuild_success = true`

## Notes

- `--use-ssh` is now supported directly by `rag.crawlers.standalone`
- this does not change downstream business logic; it only ensures standalone passes the same remote runtime configuration that the full RAG smoke test already used
- FAISS is always written to `./faiss` relative to the current working directory, so the isolated run pattern above is the safest default

## Chunking Update

The current semantic indexing path now uses a pre-split stage before `SemanticChunker`.

Implementation summary:

- `rag/rag_system.py`
- `SEMANTIC_PRECHUNK_SIZE = 6000`
- `SEMANTIC_PRECHUNK_OVERLAP = 600`
- pre-split with `RecursiveCharacterTextSplitter`
- then semantic chunking
- then the existing secondary split at `800` chars with `150` overlap if any semantic chunk is still too long

This change keeps the downstream indexing flow the same, but prevents the previous failure mode where the embedding endpoint could receive an overly large semantic-chunking input and return `400 Bad Request`.

## Latest Root Verification After The Chunking Update

After clearing the repository-root `md/` and `faiss/` directories and rerunning with the same remote SSH parameters, the root pipeline completed successfully through the semantic path.

Observed healthy signals from the latest rerun:

```text
Generated arXiv query: "recurrent neural networks are"
Generated arXiv query: "convolutional neural networks are"
Loading embedding model...
Embedding service connected: http://127.0.0.1:18000/v1 model=bge-m3.
Semantic pre-chunking prepared 22 docs.
Semantic chunking finished with 211 chunks.
FAISS index rebuilt.
```

Observed end state:

- `download_success_count = 2`
- `ocr_success_count = 2`
- `md_written_count = 2`
- `ingestion_result.status = success`
- `ingestion_result.rebuild_success = true`

This latest rerun confirms:

- remote LLM rewrite really took effect before search
- remote OCR succeeded
- remote `bge-m3` embeddings were used
- semantic chunking now succeeds without falling back
- the root `faiss/` directory was rebuilt successfully
