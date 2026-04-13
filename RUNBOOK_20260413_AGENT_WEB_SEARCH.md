# Agent Web Search Refactor Runbook

## Purpose

This file records the actual execution path used on 2026-04-13 to validate the "agent-driven web search after retrieval evaluation" refactor.

## Environment

- Project root: `C:\Users\jack\Desktop\demo`
- Required Python environment: `pyth310_new`
- Conda env path discovered during execution: `D:\conda_envs\pyth310_new`
- Windows shell: PowerShell

## Conda Activation

If running from a normal terminal with Conda available:

```powershell
& 'C:\Users\jack\miniconda3\shell\condabin\conda-hook.ps1'
conda activate D:\conda_envs\pyth310_new
$env:PYTHONIOENCODING = 'utf-8'
```

In this session, `conda run` hit a Windows encoding problem, so the reliable fallback was to invoke the environment's Python directly:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\conda_envs\pyth310_new\python.exe' --version
```

## Validation Commands

### 1. Syntax check

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\conda_envs\pyth310_new\python.exe' -m py_compile .\rag\agent\builder.py .\rag\agent\middleware.py .\rag\testing\test_single_query_flow.py
```

### 2. Targeted middleware tests

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\conda_envs\pyth310_new\python.exe' -m unittest rag.testing.test_single_query_flow.MiddlewareAgentDrivenWebSearchTests rag.testing.test_single_query_flow.MiddlewareFilteringTests
```

### 3. Full single-query test suite

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\conda_envs\pyth310_new\python.exe' -m unittest rag.testing.test_single_query_flow
```

### 4. Full RAG flow test for the requested question

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\conda_envs\pyth310_new\python.exe' .\test_rag_full_flow.py --query "fading memory（衰减记忆）’这一性质在动态系统中的作用，与HA-GNN模型中利用历史访问信息进行预测的机制之间，有何相似性与本质区别？" --tavily-api-key "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"
```

## Test Results

### `rag.testing.test_single_query_flow`

- Command status: success
- Result summary: `Ran 44 tests in 6.194s`
- Final status: `OK`

### `test_rag_full_flow.py`

- Command status: success
- Import check: passed
- LLM runtime check: passed
- Embedding runtime check: passed
- Local artifact check: passed
- Full flow: passed

## Full Flow Behavior Observed

For the requested question, the runtime followed this sequence:

1. Query planning finished.
2. Local KB retrieval executed.
3. Retrieval evaluation judged local evidence insufficient.
4. The agent was forced into `search_web` stage by middleware.
5. The agent invoked `search_web_with_tavily`.
6. Tavily returned web evidence.
7. Middleware merged local and web evidence into the final evidence bundle.
8. The agent produced the final answer from the merged evidence.

## Key Runtime Signals from the Successful Run

- `retrieval_next_action`: `answer`
- `web_search_status`: `success`
- `web_search_message`: `Tavily returned 12 results, normalized into 11 evidence docs, covered 4/4 missing_aspects.`
- `relevance_missing_aspects`: `[]`
- `missing_aspects_for_crawler`: `[]`
- `final_evidence_summary`: `local_evidence=1; web_evidence=11; uncovered_aspects=0; local evidence and Tavily web evidence have been merged for final answering`

## Full Flow JSON Snapshot

The final `full_flow` JSON from the successful run:

```json
{
  "ok": true,
  "answer": "衰减记忆（fading memory）是动态系统中描述过去输入对当前输出影响随时间推移而衰减的特性。在动态系统中，衰减记忆确保系统的稳定性，使得遥远的历史输入对当前状态的影响微乎其微，通常表现为影响随时间的指数衰减，且在稳态下系统渐近独立于初始条件 [5][6]。衰减记忆也是线性时不变系统存在卷积表示（即核为ℓ¹可和）的充要条件，并构成了极限分析、嵌储计算和递归神经网络理论的基础 [2][4][5]。\n\nHA-GNN中利用历史访问信息进行预测的机制，本质上是通过维护节点的历史嵌入（historical embeddings）来近似当前嵌入，以解决大规模图计算中的内存瓶颈问题。这种机制允许模型使用离线存储的、批量外节点的最近历史嵌入，从而在保持恒定内存消耗的同时，利用历史依赖信息填充节点间的交叉依赖 [8][9]。然而，这种使用历史嵌入的机制也引入了近似误差，该误差主要源于历史嵌入的“陈旧性”（staleness），即历史状态与真实状态之间的差异，并受学习函数的 Lipschitz 连续性约束 [8]。\n\n两者的相似性在于：两者都体现了系统状态演化对历史信息的依赖性，且这种依赖性在一定程度上的“衰减”以保证系统的可预测性或计算的可管理性。衰减记忆在生物学和认知模型中也表现出类似的机制，即感官输入中的规律性被保留，但随时间推移，特定刺激的痕迹会逐渐淡化，这与动态系统中过去输入影响的衰减相似 [3]。\n\n两者的本质区别在于：衰减记忆是一个严格的数学稳定性性质，描述的是动态系统的固有行为（即状态演化的自然过程），其衰减通常由系统的稳定性（如平衡点的指数稳定性）决定，且在理论上与唯一稳态和渐近独立性紧密相关 [6]。而HA-GNN中的历史访问机制是一种工程化的数据存储和读取策略，用于优化计算效率和缓解内存限制，它并不直接描述系统演化的内在动力学，而是通过人工缓存和近似（可能存在误差）来模拟或补充信息传递。HA-GNN的历史机制更侧重于计算层面的近似和效率权衡，而非系统动力学意义上的稳定性保证，且其“记忆”的使用带有明显的近似误差（由历史陈旧和采样限制引起）。因此，前者是系统稳定性分析的基准，后者是深度学习模型在处理稀疏计算数据时的工程优化手段。",
  "answer_preview": "衰减记忆（fading memory）是动态系统中描述过去输入对当前输出影响随时间推移而衰减的特性。在动态系统中，衰减记忆确保系统的稳定性，使得遥远的历史输入对当前状态的影响微乎其微，通常表现为影响随时间的指数衰减，且在稳态下系统渐近独立于初始条件 [5][6]。衰减记忆也是线性时不变系统存在卷积表示（即核为ℓ¹可和）的充要条件，并构成了极限分析、嵌储计算和递归神经网络理论的基础 [2][4][5]。\n\nHA-GNN中利用历史访问信息进行预测的机制，本质上是通过维护节点的历史嵌入（historical embeddings）来近似当前嵌入，以解决大规模图计算中的内存瓶颈问题。这种机制允许模型使用离线存储的、批量外节点的最近历史嵌入，从而在保持恒定内存消耗的同时，利用历史依赖信息填充节点间的交叉依赖 [8][9]。然而，这种使用历史嵌入的机制也引入了近似误差，该误差主要源于历史嵌入的“陈旧性...(truncated)",
  "evidence_list": [5, 6, 2, 4, 8, 9, 3],
  "retrieval_next_action": "answer",
  "relevance_reason": "aspect_coverage=1.00, support_strength=0.52, noise_ratio=0.40, missing_aspects_count=0, missing_aspects=none, next_action=retrieve_more; next_action=search_web",
  "relevance_missing_aspects": [],
  "missing_aspects_for_crawler": [],
  "web_search_status": "success",
  "web_search_message": "Tavily returned 12 results, normalized into 11 evidence docs, covered 4/4 missing_aspects.",
  "final_evidence_summary": "local_evidence=1; web_evidence=11; uncovered_aspects=0; local evidence and Tavily web evidence have been merged for final answering",
  "final_evidence_item_count": 12
}
```

## Notes

- `conda run -n pyth310_new ...` was not used in the final successful path because Conda itself failed on Windows console encoding output.
- Direct invocation of `D:\conda_envs\pyth310_new\python.exe` is the recommended stable method for this project on the current machine.
- The successful run confirms that the refactor works as intended: evaluation decides the next stage, and the agent performs the actual web search.
