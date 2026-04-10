# demo

## SSH 测试命令

先激活运行环境：


```powershell
conda activate D:\conda_envs\pyth310_new
```

只测试 LLM 的 SSH 隧道与 vLLM：

```powershell
python test_ssh_vllm.py --ssh-host 172.26.19.131 --ssh-port 8888 --ssh-username root --ssh-password "123456.a" --remote-port 8001 --local-port 18001 --model "Qwen/Qwen3.5-9B"
```

测试完整 RAG 流程（LLM + bge + Tavily）：

```powershell
python test_rag_full_flow.py --query "RNN和CNN的区别是什么？" --use-ssh --ssh-host 172.26.19.131 --ssh-port 8888 --ssh-username root --ssh-password "123456.a" --llm-model "Qwen/Qwen3.5-9B" --embedding-model "bge-m3" --tavily-api-key "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"
```

如果要保存完整 JSON 测试报告：

```powershell
python test_rag_full_flow.py --query "RAG是什么？" --use-ssh --ssh-host 172.26.19.131 --ssh-port 8888 --ssh-username root --ssh-password "123456.a" --llm-model "Qwen/Qwen3.5-9B" --embedding-model "bge-m3" --tavily-api-key "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" --output-json ".\paper_results\rag_smoke_report.json"
```

## Standalone 爬虫远程运行

`rag.crawlers.standalone` 现在也支持和上面一致的 SSH 远程运行方式，可以把：

- LLM 改写
- OCR
- embedding / FAISS

都接到远程服务上。

### 当前根目录健康跑通配置

- missing aspects:
  - `the definition of RNN`
  - `the definition of CNN`
- SSH host: `172.26.19.131`
- SSH port: `8888`
- SSH username: `root`
- SSH password: `123456.a`
- LLM remote/local port: `8001 -> 18001`
- embedding remote/local port: `8000 -> 18000`
- OCR remote/local port: `8002 -> 18002`

### 推荐运行命令

建议先设置 UTF-8，避免 Windows 控制台在 `--print-json` 时出现编码报错。

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

### 当前完整流程

这条链路现在已经按下面的顺序跑通：

`standalone crawler -> remote LLM rewrite -> arXiv search -> PDF download -> remote OCR -> md -> remote BGE embeddings -> FAISS rebuild`

### 当前分块机制

为了避免 semantic chunking 阶段把过长文本直接送进 embedding 接口，当前分块逻辑是：

1. 先把 markdown 预切到大约 `6000` 字符上限
2. 再做 semantic chunking
3. 如果某个 semantic chunk 仍然太长，再走原来的二次字符切分

当前参数在 `rag/rag_system.py` 中是：

- `SEMANTIC_PRECHUNK_SIZE = 6000`
- `SEMANTIC_PRECHUNK_OVERLAP = 600`

### 最近一次根目录重跑的健康信号

最近一次清空 `md/` 和 `faiss/` 后的根目录重跑，关键日志是：

```text
Generated arXiv query: "recurrent neural networks are"
Generated arXiv query: "convolutional neural networks are"
Loading embedding model...
Embedding service connected: http://127.0.0.1:18000/v1 model=bge-m3.
Semantic pre-chunking prepared 22 docs.
Semantic chunking finished with 211 chunks.
FAISS index rebuilt.
```

说明：

- standalone 爬虫确实先走了远程 LLM 改写
- arXiv 搜索用了改写后的 query
- OCR 和 embedding 都走了远程服务
- semantic chunking 成功完成
- 根目录 `faiss/` 已经成功重建
