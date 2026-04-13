# RAG Project Context

## Purpose

This file is a compact context snapshot for future AI agents. Read this first before modifying or testing the `rag` pipeline.

## Project Goal

The project implements an academic RAG workflow around a local paper knowledge base plus optional web search. The core target flow is:

`user input -> query optimization -> local retrieval -> retrieval evaluation -> web search if needed -> final answer`

## Current Execution Model

The active implementation is agent-based.

- The agent is created in `rag/agent/builder.py`
- Runtime state is shared through `rag/agent/runtime.py`
- Stage control and evidence assembly happen in `rag/agent/middleware.py`
- Local retrieval and Tavily web search tools live in `rag/agent/tools_impl.py`
- Retrieval sufficiency scoring lives in `rag/retrieval/evaluator.py`
- Query planning lives in `rag/query/optimizer.py`

## Current Control Flow

### 1. Query planning

`AcademicResearchMiddleware.before_agent(...)`:

- extracts the latest user question
- builds `AcademicQueryPlan`
- stores it in runtime context and agent state

The query plan contains:

- `normalized_query_zh`
- `retrieval_query_zh`
- `retrieval_query_en`
- `crawler_query_en`
- `keywords_zh`
- `keywords_en`
- `required_aspects`

### 2. Local retrieval

The first tool the agent may call is always:

- `retrieve_local_kb(search_query: str)`

This tool uses the configured `RAGSystem` to retrieve local documents and returns a `RetrievalPayload`.

### 3. Retrieval evaluation

After local retrieval returns, middleware immediately evaluates the retrieved docs by calling:

- `evaluate_retrieval(query_plan, payload.docs)`

The evaluator computes:

- `aspect_coverage`
- `support_strength`
- `noise_ratio`
- `covered_aspects`
- `weak_aspects`
- `missing_aspects`
- `next_action`

Middleware then:

- annotates local docs with matched aspects
- selects local evidence
- builds a local-only `FinalEvidenceBundle`

### 4. Agent-driven web search

This is the current intended behavior after the refactor:

- Middleware does **not** auto-run Tavily anymore.
- If local evidence is sufficient:
  - `retrieval_next_action = "answer"`
  - `web_search_required = False`
  - `web_search_used = False`
- If local evidence is insufficient:
  - `retrieval_next_action = "search_web"`
  - `web_search_required = True`
  - `web_search_used = False`
  - `relevance_missing_aspects` becomes the authoritative pending search list

The agent is then forced by middleware and prompt constraints to call:

- `search_web_with_tavily(missing_aspects: list[str])`

exactly once before answering.

### 5. Web search result merge

When the web-search tool returns:

- middleware parses `WebSearchPayload`
- converts `evidence_docs` into normalized final evidence items
- merges local evidence + web evidence into a new `FinalEvidenceBundle`
- updates state:
  - `web_search_used = True`
  - `web_search_required = False`
  - `retrieval_next_action = "answer"`

### 6. Final answer generation

The final answer must:

- be in Chinese
- be based on `Final Evidence Bundle`
- cite only bundle indexes like `[1]`
- return a single JSON object:

```json
{"answer":"...", "evidence_list":[1,2,3]}
```

Answer post-processing is handled in `parse_agent_answer(...)` inside `rag/agent/builder.py`.

## Tool Gating Rules

Middleware currently enforces these stages:

1. Before any retrieval result exists:
   - only `retrieve_local_kb`
2. After insufficient retrieval and before web search:
   - only `search_web_with_tavily`
3. After sufficient retrieval or after one web search:
   - no tools

This hard gating is implemented in:

- `AcademicResearchMiddleware._filter_tools(...)`

## Important Files

- `rag/agent/builder.py`
  - agent construction
  - global system prompt
  - final answer parsing
- `rag/agent/middleware.py`
  - stage control
  - evaluation hook
  - evidence merging
  - prompt injection
- `rag/agent/tools_impl.py`
  - local retrieval tool
  - Tavily search tool
- `rag/retrieval/evaluator.py`
  - sufficiency scoring logic
- `rag/schemas.py`
  - `AcademicQueryPlan`
  - `RetrievalPayload`
  - `WebSearchPayload`
  - `FinalEvidenceBundle`
  - `ResearchState`
- `test_rag_full_flow.py`
  - smoke test for the whole runtime flow
- `rag/testing/test_single_query_flow.py`
  - main unit-test file for query planning, evaluation, middleware, crawler, OCR, FAISS rebuild, and agent-stage behavior

## Runtime Dependencies and Services

No new dependencies should be installed without user approval.

Current runtime depends on:

- local or SSH-forwarded LLM service
- local or SSH-forwarded embedding service
- Tavily API key for web search
- existing `faiss/` and `md/` data

The observed working environment on this machine:

- Python env: `D:\conda_envs\pyth310_new`
- LLM base URL via SSH tunnel: `http://127.0.0.1:18001/v1`
- Embedding base URL via SSH tunnel: `http://127.0.0.1:18000/v1`

## Recommended Validation Commands

Use the environment Python directly:

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\conda_envs\pyth310_new\python.exe' -m unittest rag.testing.test_single_query_flow
```

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\conda_envs\pyth310_new\python.exe' .\test_rag_full_flow.py --query "fading memory（衰减记忆）’这一性质在动态系统中的作用，与HA-GNN模型中利用历史访问信息进行预测的机制之间，有何相似性与本质区别？" --tavily-api-key "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"
```

## What Was Recently Changed

The web-search stage was refactored from:

- middleware auto-running Tavily immediately after insufficient retrieval

to:

- middleware only deciding that web search is required
- agent performing the actual Tavily tool call
- middleware merging evidence only after the tool result comes back

This is the current expected behavior. Future edits should preserve it unless the user explicitly asks to redesign the workflow.
