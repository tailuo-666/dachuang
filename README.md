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
python test_rag_full_flow.py --query "RAG是什么？" --use-ssh --ssh-host 172.26.19.131 --ssh-port 8888 --ssh-username root --ssh-password "123456.a" --llm-model "Qwen/Qwen3.5-9B" --embedding-model "bge-m3" --tavily-api-key "你的密钥"
```

如果要保存完整 JSON 测试报告：

```powershell
python test_rag_full_flow.py --query "RAG是什么？" --use-ssh --ssh-host 172.26.19.131 --ssh-port 8888 --ssh-username root --ssh-password "123456.a" --llm-model "Qwen/Qwen3.5-9B" --embedding-model "bge-m3" --tavily-api-key "你的密钥" --output-json ".\paper_results\rag_smoke_report.json"
```
