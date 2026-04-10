# Full RAG Flow Test: Fading Memory vs HA-GNN

## Test Goal

Run the existing full-flow test entry:

- `test_rag_full_flow.py`

with this query:

- `fading memory（衰减记忆）’这一性质在动态系统中的作用，与HA-GNN模型中利用历史访问信息进行预测的机制之间，有何相似性与本质区别？`

This test covers:

- input question
- query planning
- local retrieval from `faiss/`
- web search through Tavily
- final answer generation

This test does **not** cover:

- standalone crawler
- OCR
- PDF download

## Runtime Command

```powershell
& 'C:\Users\jack\miniconda3\shell\condabin\conda-hook.ps1'
conda activate D:\conda_envs\pyth310_new
$env:PYTHONIOENCODING = "utf-8"
python test_rag_full_flow.py `
  --query "fading memory（衰减记忆）’这一性质在动态系统中的作用，与HA-GNN模型中利用历史访问信息进行预测的机制之间，有何相似性与本质区别？" `
  --use-ssh `
  --ssh-host 172.26.19.131 `
  --ssh-port 8888 `
  --ssh-username root `
  --ssh-password "123456.a" `
  --llm-model "Qwen/Qwen3.5-9B" `
  --embedding-model "bge-m3" `
  --tavily-api-key "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" `
  --output-json "C:\Users\jack\Desktop\demo\paper_results\rag_full_flow_fading_memory_20260409.json"
```

## What Happened Before The Fix

The first run reached:

1. local FAISS retrieval
2. insufficiency detection
3. Tavily web search
4. evidence merge

but then failed at final answer generation with:

```text
BadRequestError: Error code: 400
This model's maximum context length is 32000 tokens.
However, your prompt contains at least 32001 input tokens.
```

The root cause was not the retrieval itself.
It was the final prompt assembly:

- full retrieval/web tool payloads were being kept in `ToolMessage`
- the middleware also injected the full `Final Evidence Bundle` into the system prompt
- the same evidence was therefore duplicated in the final model call

## Fix Applied

The fix was implemented in:

- `rag/agent/middleware.py`

What changed:

- tool messages are now converted into compact summaries instead of full payload JSON
- the `Final Evidence Bundle` injected into the prompt is now a compact, prompt-friendly version
- evidence items are limited in count
- evidence content is clipped to excerpts before entering the final prompt

This does **not** change the retrieval/search workflow itself.
It only reduces prompt size at the final synthesis stage.

## Latest Successful Rerun

After the prompt-budget fix, the same full-flow test was rerun successfully.

### Runtime checks

The runtime checks passed:

- LLM endpoint resolved to `http://127.0.0.1:18001/v1`
- embedding endpoint resolved to `http://127.0.0.1:18000/v1`
- LLM invoke check returned `OK`
- embedding service check returned model `bge-m3`
- local artifacts were present:
  - `faiss/` exists
  - `md/` exists
  - `md_file_count = 2`

### Input to output trace

Observed runtime logs:

```text
查询预处理完成，已生成学术检索计划。
正在检索本地知识库: 衰减记忆 动态系统 HA-GNN历史访问预测相似性 本质区别
Read 211 chunks from FAISS.
BM25 retriever built.
本地检索相关性评估完成：sufficient=False, next_action=search_web
正在执行 Tavily 搜索，missing_aspects=['definition of fading memory in dynamic systems', 'mechanism of historical access in HA-GNN', 'similarities between fading memory and HA-GNN', 'differences between fading memory and HA-GNN', 'predictive comparison dynamic systems versus graph networks']
Tavily 搜索阶段完成，开始基于统一证据生成最终答案。
```

This means the actual path was:

1. input question entered the agent
2. query planning normalized the search intent
3. local FAISS retrieval ran successfully
4. local evidence was judged insufficient
5. Tavily search executed on five missing aspects
6. final answer generation completed successfully after prompt compaction

### Output summary

Observed full-flow result summary:

```json
{
  "ok": true,
  "retrieval_next_action": "answer",
  "web_search_status": "success",
  "web_search_message": "Tavily returned 15 results, normalized into 15 evidence docs, covered 5/5 missing_aspects.",
  "final_evidence_summary": "local_evidence=0; web_evidence=15; uncovered_aspects=0; local evidence and Tavily web evidence have been merged for final answering",
  "final_evidence_item_count": 15
}
```

### Answer preview

The system returned a real final answer in Chinese.
The answer compared:

- the formal role of fading memory in dynamical systems
- the historical access / historical embedding mechanism in HA-GNN-related work
- their similarities
- their essential differences

The answer also cited concrete web evidence items such as:

- `Fading Memory Property (FMP) in Dynamical Systems`
- `FreshGNN: Reducing Memory Access via Stable Historical ...`
- `HAGNN: Hybrid Aggregation for Heterogeneous Graph Neural Networks`

## Saved Outputs

Full JSON report:

- [rag_full_flow_fading_memory_20260409.json](C:/Users/jack/Desktop/demo/paper_results/rag_full_flow_fading_memory_20260409.json)

This markdown summary:

- [rag_full_flow_fading_memory_20260409.md](C:/Users/jack/Desktop/demo/paper_results/rag_full_flow_fading_memory_20260409.md)

## Final Conclusion

For this specific question, the full RAG test now completes successfully end to end:

- local retrieval: success
- insufficiency detection: success
- Tavily search: success
- final answer generation: success

The earlier failure was caused by duplicated full evidence in the final prompt, not by a broken retriever or broken web search path.
