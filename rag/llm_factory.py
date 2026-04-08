from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

try:
    from .ssh_service import (
        build_ssh_service_config,
        discover_openai_model,
        ensure_ssh_openai_base_url,
        is_ssh_tunnel_enabled,
    )
except ImportError:
    from ssh_service import (
        build_ssh_service_config,
        discover_openai_model,
        ensure_ssh_openai_base_url,
        is_ssh_tunnel_enabled,
    )


DEFAULT_LOCAL_VLLM_CONFIG = {
    "scheme": os.getenv("RAG_LLM_SCHEME", "http"),
    "host": os.getenv("RAG_LLM_HOST", "127.0.0.1"),
    "port": int(os.getenv("RAG_LLM_PORT", "8001")),
    "model": os.getenv("RAG_LLM_MODEL", "Qwen/Qwen3.5-9B"),
    "api_key": os.getenv("RAG_LLM_API_KEY", "EMPTY"),
}


def get_default_llm_ssh_config():
    """Return the SSH tunnel config used to reach the remote LLM service."""
    return build_ssh_service_config(
        "llm",
        default_remote_port=int(os.getenv("RAG_LLM_REMOTE_PORT", os.getenv("RAG_LLM_PORT", "8001"))),
        default_local_port=int(os.getenv("RAG_LLM_LOCAL_PORT", "18001")),
    )


def get_default_llm_config():
    """Return the default LLM config used by the RAG workflow."""
    config = dict(DEFAULT_LOCAL_VLLM_CONFIG)
    config["ssh"] = get_default_llm_ssh_config()

    explicit_base_url = str(os.getenv("RAG_LLM_BASE_URL") or "").strip().rstrip("/")
    if explicit_base_url:
        config["base_url"] = explicit_base_url
    elif is_ssh_tunnel_enabled(config["ssh"]):
        config["base_url"] = ensure_ssh_openai_base_url("llm", config["ssh"])
    else:
        config["base_url"] = (
            f"{config['scheme']}://{config['host']}:{config['port']}/v1"
        )

    if not str(config.get("model") or "").strip():
        discovered_model = discover_openai_model(
            config["base_url"],
            api_key=str(config.get("api_key") or "EMPTY"),
        )
        if discovered_model:
            config["model"] = discovered_model
    return config


def create_default_llm(config=None):
    """Create the default LLM used across the academic RAG workflow."""
    llm_config = get_default_llm_config()
    ssh_config = dict(llm_config.get("ssh") or {})
    if config:
        llm_config.update({key: value for key, value in config.items() if key != "ssh"})
        if isinstance(config.get("ssh"), dict):
            ssh_config.update(config["ssh"])
    llm_config["ssh"] = ssh_config

    if "base_url" not in llm_config or not llm_config["base_url"]:
        if is_ssh_tunnel_enabled(ssh_config):
            llm_config["base_url"] = ensure_ssh_openai_base_url("llm", ssh_config)
        else:
            llm_config["base_url"] = (
                f"{llm_config['scheme']}://{llm_config['host']}:{llm_config['port']}/v1"
            )

    if not str(llm_config.get("model") or "").strip():
        discovered_model = discover_openai_model(
            llm_config["base_url"],
            api_key=str(llm_config.get("api_key") or "EMPTY"),
        )
        if discovered_model:
            llm_config["model"] = discovered_model

    return ChatOpenAI(
        model=llm_config["model"],
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
    )
