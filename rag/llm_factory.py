import os

from langchain_openai import ChatOpenAI


DEFAULT_LOCAL_VLLM_CONFIG = {
    # vLLM 服务用 `--host 0.0.0.0 --port 8001` 启动时，
    # 同机客户端应通过 127.0.0.1 访问，而不是把 0.0.0.0 当成请求地址。
    "scheme": os.getenv("RAG_LLM_SCHEME", "http"),
    "host": os.getenv("RAG_LLM_HOST", "127.0.0.1"),
    "port": int(os.getenv("RAG_LLM_PORT", "8001")),
    "model": os.getenv("RAG_LLM_MODEL", "Qwen/Qwen3.5-9B"),
    "api_key": os.getenv("RAG_LLM_API_KEY", "EMPTY"),
}


def get_default_llm_config():
    """Return the default local vLLM config used by the RAG workflow."""
    config = dict(DEFAULT_LOCAL_VLLM_CONFIG)
    base_url = os.getenv("RAG_LLM_BASE_URL")
    if base_url:
        config["base_url"] = base_url.rstrip("/")
    else:
        config["base_url"] = (
            f"{config['scheme']}://{config['host']}:{config['port']}/v1"
        )
    return config


def create_default_llm(config=None):
    """Create the default LLM used across the academic RAG workflow."""
    llm_config = get_default_llm_config()
    if config:
        llm_config.update(config)

    if "base_url" not in llm_config or not llm_config["base_url"]:
        llm_config["base_url"] = (
            f"{llm_config['scheme']}://{llm_config['host']}:{llm_config['port']}/v1"
        )

    return ChatOpenAI(
        model=llm_config["model"],
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
    )
