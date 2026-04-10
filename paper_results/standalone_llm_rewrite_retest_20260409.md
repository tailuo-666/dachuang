# Standalone Crawler LLM Rewrite Retest

## Test Goal

Answer two concrete questions:

1. Before the prompt fix, did the standalone crawler really reach the LLM call?
2. In the current environment, when we rerun the standalone crawler, what happens from input to output?

This document focuses on the pre-crawl path:

`missing_aspect -> prompt formatting -> LLM rewrite -> fallback or rewritten query -> arXiv search -> crawl output`

It does not rerun OCR / markdown / embedding / FAISS, because the question here is specifically whether the crawler stage really called the LLM before searching papers.

## Environment Observed During Retest

### Python

- interpreter: `D:\conda_envs\pyth310_new\python.exe`

### Resolved LLM Config

`get_default_llm_config()` returned:

```json
{
  "scheme": "http",
  "host": "127.0.0.1",
  "port": 8001,
  "model": "Qwen/Qwen3.5-9B",
  "api_key": "EMPTY",
  "ssh": {
    "ssh_host": "",
    "ssh_port": 8888,
    "ssh_username": "",
    "ssh_password": "",
    "remote_host": "127.0.0.1",
    "remote_port": 8001,
    "local_host": "127.0.0.1",
    "local_port": 18001
  },
  "base_url": "http://127.0.0.1:8001/v1"
}
```

### Important Configuration Finding

During this retest:

- `ssh_host` was empty
- the resolved `base_url` was `http://127.0.0.1:8001/v1`
- no `RAG_LLM_*` environment variables were present in the current shell

So this process was **not configured to use the previous SSH-tunneled remote endpoint** described in the earlier pipeline test.

That means:

- if you were checking logs on a remote LLM server, this retest would not be expected to show up there
- this run was aimed at a local OpenAI-compatible endpoint on `127.0.0.1:8001`

## Test A: Did the Old Bug Reach `llm.invoke()`?

### Method

I recreated the old prompt pattern with unescaped JSON braces and used a spy LLM object that only increments a counter when `invoke()` is actually called.

### Result

```json
{
  "status": "failed_before_llm_invoke",
  "spy_calls": 0,
  "exception_type": "KeyError",
  "exception": "'Input to ChatPromptTemplate is missing variables {...}'"
}
```

### Interpretation

Before the prompt fix:

- `ChatPromptTemplate` tried to treat JSON keys like `"original_aspect"` as template variables
- `prompt.invoke({"aspect": ...})` raised `KeyError`
- `llm.invoke(...)` was never reached
- so the standalone crawler did **not** actually send a request to the LLM service in that failure mode

This explains why your server had no LLM call record for the old buggy path.

## Test B: Current `rewrite()` Retest With Real Runtime Config

### Input

- aspect: `the definition of RNN`

### Observed Runtime Result

The current fixed code did enter the rewrite path, but the runtime LLM call failed:

```text
Warning: failed to rewrite missing aspect 'the definition of RNN' via LLM (InternalServerError: Error code: 502); falling back to original aspect.
```

Returned payload:

```json
{
  "original_aspect": "the definition of RNN",
  "optimized_query_en": "the definition of RNN",
  "keywords_en": [
    "the definition of RNN"
  ]
}
```

### Additional Low-Level Check

I also sent a direct HTTP request to:

- `http://127.0.0.1:8001/v1/chat/completions`

Observed result:

```json
{
  "url": "http://127.0.0.1:8001/v1/chat/completions",
  "exception_type": "ConnectionError",
  "exception": "Failed to establish a new connection: [WinError 10061] ..."
}
```

### Interpretation

After the prompt fix:

- the code path now really tries to call the LLM
- but in the current environment, the resolved LLM endpoint is not healthy / not reachable as a plain HTTP endpoint
- because the rewrite call fails at runtime, the crawler falls back to the original aspect text

So the current retest result is:

- old bug is fixed
- but the current LLM service configuration is still not in a working state for successful rewrite responses

## Test C: Full Standalone Crawler Retest From Input To Output

### Test Input

- `missing_aspects`:
  - `the definition of RNN`
  - `the definition of CNN`
- `max_pages=1`
- `max_new_papers=2`
- `auto_ingest=False`

### Temporary Output Location

- run id: `20260409_retest_ebc2a6cb`
- paper output dir:
  - `C:\Users\jack\Desktop\demo\.tmp\20260409_retest_ebc2a6cb\paper_results`

### Runtime Trace

Observed console messages:

```text
Warning: failed to rewrite missing aspect 'the definition of RNN' via LLM (InternalServerError: Error code: 502); falling back to original aspect.
Warning: failed to rewrite missing aspect 'the definition of CNN' via LLM (InternalServerError: Error code: 502); falling back to original aspect.
Searching arXiv for relevant papers...
Generated arXiv query: "the definition of RNN"
Finished arXiv crawl with 1 papers.
Searching arXiv for relevant papers...
Generated arXiv query: "the definition of CNN"
Finished arXiv crawl with 1 papers.
```

### Output Summary

```json
{
  "run_id": "20260409_retest_ebc2a6cb",
  "requested_aspects": [
    "the definition of RNN",
    "the definition of CNN"
  ],
  "rewrites": [
    {
      "original_aspect": "the definition of RNN",
      "optimized_query_en": "the definition of RNN",
      "keywords_en": [
        "the definition of RNN"
      ]
    },
    {
      "original_aspect": "the definition of CNN",
      "optimized_query_en": "the definition of CNN",
      "keywords_en": [
        "the definition of CNN"
      ]
    }
  ],
  "payload_status": "success",
  "payload_message": "arXiv found 2 papers, returned 4 summary chunks, covered 2/2 missing_aspects.",
  "search_queries": [
    {
      "aspect": "the definition of RNN",
      "query": "\"the definition of RNN\""
    },
    {
      "aspect": "the definition of CNN",
      "query": "\"the definition of CNN\""
    }
  ]
}
```

### End-to-End Interpretation

For this retest, the actual flow was:

1. input aspects entered `run_aspects()`
2. `rewrite()` was called for each aspect
3. the fixed prompt no longer failed during template formatting
4. the code attempted to call the LLM client
5. the LLM request failed at runtime with `InternalServerError: 502`
6. `rewrite()` fell back to the original aspect strings
7. those original strings were used to build arXiv search queries
8. arXiv crawl still completed successfully

So in this specific rerun, the crawler **did reach the LLM stage**, but the rewrite result did not take effect because the runtime LLM call failed and triggered fallback.

## Final Conclusion

Two different situations need to be separated clearly.

### 1. Before the prompt fix

No, the buggy standalone crawler path did **not** really reach the LLM server.

Reason:

- prompt formatting failed first
- `llm.invoke()` was never called
- therefore no server-side LLM call log would appear

### 2. In the current retest after the prompt fix

Yes, the code now reaches the LLM call stage.

But the current runtime environment is not successfully serving the LLM request:

- the resolved endpoint is `http://127.0.0.1:8001/v1`
- this retest is not configured to use a remote SSH-tunneled server
- the runtime call fails and falls back to the original aspect

So if you are checking a remote server log, there are two likely reasons you see nothing:

1. the old buggy path never sent the request at all
2. the current retest is not pointing at that remote server in the first place

## Suggested Next Check

If you want the rewrite to really take effect in live runs, the next thing to verify is not the prompt anymore, but the LLM endpoint wiring:

- whether `RAG_LLM_BASE_URL` / SSH config should point to your actual server
- whether the intended local tunnel or local vLLM service is actually up
- whether `http://127.0.0.1:8001/v1` is the correct target for this machine

## Follow-up Fix: Standalone SSH Runtime Args

After the retest above, the standalone crawler entrypoint was updated to mirror the runtime wiring already used by `test_rag_full_flow.py`.

### What Was Missing

The standalone CLI previously did not inject SSH runtime settings into the current process before constructing:

- the LLM client
- the OCR client
- the embedding client

So even though the downstream factories already supported SSH tunnel mode, standalone did not expose the same runtime switch.

### What Was Changed

The standalone entrypoint now supports SSH runtime arguments and applies them before creating downstream clients:

- `--use-ssh`
- `--ssh-host`
- `--ssh-port`
- `--ssh-username`
- `--ssh-password`
- `--llm-remote-port`
- `--llm-local-port`
- `--embedding-remote-port`
- `--embedding-local-port`
- `--ocr-remote-port`
- `--ocr-local-port`

When `--use-ssh` is enabled, standalone now:

- sets shared SSH environment variables such as `RAG_SSH_HOST`
- sets service-specific remote/local ports for LLM, embedding, and OCR
- clears direct base URL overrides so the factories resolve through SSH tunnels

This change does not alter the downstream workflow logic.
It only fixes how runtime configuration is supplied to the existing factories.

## Remote Service Retest After the Runtime Fix

### Shared Runtime Config Used

- SSH host: `172.26.19.131:8888`
- SSH username: `root`
- LLM remote/local port: `8001 -> 18001`
- embedding remote/local port: `8000 -> 18000`
- OCR remote/local port: `8002 -> 18002`

### Remote LLM Rewrite Check

Input:

- `the definition of RNN`

Observed output:

```json
{
  "original_aspect": "the definition of RNN",
  "optimized_query_en": "recurrent neural networks are",
  "keywords_en": [
    "recurrent neural networks",
    "RNN",
    "recurrent architectures",
    "sequence models"
  ]
}
```

This confirms the fixed standalone rewrite stage can now really call the remote LLM and receive a non-fallback result.

### Remote OCR Check

Observed resolved OCR target:

- `http://127.0.0.1:18002/v1`

Observed models response included:

- `paddle-ocr-vl`

This confirms OCR is also resolving through the SSH tunnel path.

### Remote Embedding Check

Observed log:

```text
Loading embedding model...
Embedding service connected: http://127.0.0.1:18000/v1 model=bge-m3.
```

This confirms the embedding service is reached through the SSH tunnel path and the retriever runtime uses the remote embedding backend.

## Full Standalone Crawler Retest After the Runtime Fix

### Input

- `the definition of RNN`
- `the definition of CNN`

### Observed arXiv Queries

```json
[
  {
    "aspect": "the definition of RNN",
    "query": "\"recurrent neural networks are\""
  },
  {
    "aspect": "the definition of CNN",
    "query": "\"convolutional neural networks are\""
  }
]
```

### Observed Rewrite Payloads

```json
[
  {
    "original_aspect": "the definition of RNN",
    "optimized_query_en": "recurrent neural networks are",
    "keywords_en": [
      "recurrent neural networks",
      "RNN",
      "recurrent architectures",
      "sequence models"
    ]
  },
  {
    "original_aspect": "the definition of CNN",
    "optimized_query_en": "convolutional neural networks are",
    "keywords_en": [
      "convolutional neural networks",
      "CNN",
      "convolutional architectures",
      "deep convolutional neural networks"
    ]
  }
]
```

### Result

Observed crawler summary:

- `payload_status`: `success`
- `payload_message`: `arXiv found 98 papers, returned 4 summary chunks, covered 2/2 missing_aspects.`

This confirms that after fixing the standalone runtime configuration:

1. the crawler enters the LLM rewrite stage
2. the rewrite result is returned from the remote LLM
3. the rewritten query is actually passed into arXiv search
4. the rest of the standalone crawler logic remains unchanged
