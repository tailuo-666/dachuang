from test_ssh_vllm import create_ssh_vllm_chat_openai


def create_default_llm():
    """Create the default LLM used across the academic RAG workflow."""
    # 原先直连 DashScope 的配置已停用，保留注释仅作参考。
    # return ChatOpenAI(
    #     model="qwen-plus",
    #     api_key="sk-e4b7b6386950428bb71c658d47da47ef",
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    return create_ssh_vllm_chat_openai()
