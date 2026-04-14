from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

import requests
from langchain_openai import ChatOpenAI

try:
    from .ssh_service import (
        build_ssh_service_config,
        discover_openai_model,
        ensure_ssh_openai_base_url,
        is_ssh_tunnel_enabled,
        stop_ssh_tunnel,
    )
except ImportError:
    from ssh_service import (
        build_ssh_service_config,
        discover_openai_model,
        ensure_ssh_openai_base_url,
        is_ssh_tunnel_enabled,
        stop_ssh_tunnel,
    )


DEFAULT_LLM_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_LLM_API_KEY = "EMPTY"
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_SSH_VLLM_CONFIG = {
    "ssh_host": "172.26.19.131",
    "ssh_port": 8888,
    "ssh_username": "root",
    "ssh_password": "123456.a",
    "remote_host": "127.0.0.1",
    "remote_port": 8001,
    "local_host": "127.0.0.1",
    "local_port": 18001,
    "model": DEFAULT_LLM_MODEL,
    "api_key": DEFAULT_LLM_API_KEY,
}


def _coerce_temperature(value: Any, default: float = DEFAULT_LLM_TEMPERATURE) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_base_url(base_url: Any) -> str:
    cleaned = str(base_url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _base_llm_config_from_env() -> dict[str, Any]:
    return {
        "scheme": os.getenv("RAG_LLM_SCHEME", "http"),
        "host": os.getenv("RAG_LLM_HOST", "127.0.0.1"),
        "port": int(os.getenv("RAG_LLM_PORT", "8001")),
        "model": str(os.getenv("RAG_LLM_MODEL", DEFAULT_LLM_MODEL)).strip(),
        "api_key": str(os.getenv("RAG_LLM_API_KEY", DEFAULT_LLM_API_KEY)).strip() or DEFAULT_LLM_API_KEY,
        "temperature": _coerce_temperature(os.getenv("RAG_LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE)),
    }


def get_default_llm_ssh_config() -> dict[str, Any]:
    """Return the SSH tunnel config used to reach the remote LLM service."""
    ssh_config = build_ssh_service_config(
        "llm",
        default_remote_port=int(os.getenv("RAG_LLM_REMOTE_PORT", str(DEFAULT_SSH_VLLM_CONFIG["remote_port"]))),
        default_local_port=int(os.getenv("RAG_LLM_LOCAL_PORT", str(DEFAULT_SSH_VLLM_CONFIG["local_port"]))),
        default_remote_host=str(DEFAULT_SSH_VLLM_CONFIG["remote_host"]),
        default_local_host=str(DEFAULT_SSH_VLLM_CONFIG["local_host"]),
    )
    ssh_config["ssh_host"] = str(ssh_config.get("ssh_host") or DEFAULT_SSH_VLLM_CONFIG["ssh_host"]).strip()
    ssh_config["ssh_port"] = int(ssh_config.get("ssh_port") or DEFAULT_SSH_VLLM_CONFIG["ssh_port"])
    ssh_config["ssh_username"] = str(
        ssh_config.get("ssh_username") or DEFAULT_SSH_VLLM_CONFIG["ssh_username"]
    ).strip()
    ssh_config["ssh_password"] = str(
        ssh_config.get("ssh_password") or DEFAULT_SSH_VLLM_CONFIG["ssh_password"]
    ).strip()
    ssh_config["remote_host"] = str(
        ssh_config.get("remote_host") or DEFAULT_SSH_VLLM_CONFIG["remote_host"]
    ).strip()
    ssh_config["remote_port"] = int(
        ssh_config.get("remote_port") or DEFAULT_SSH_VLLM_CONFIG["remote_port"]
    )
    ssh_config["local_host"] = str(
        ssh_config.get("local_host") or DEFAULT_SSH_VLLM_CONFIG["local_host"]
    ).strip()
    ssh_config["local_port"] = int(
        ssh_config.get("local_port") or DEFAULT_SSH_VLLM_CONFIG["local_port"]
    )
    return ssh_config


def get_default_remote_llm_config() -> dict[str, Any]:
    """Return the SSH-style remote LLM config aligned with test_ssh_vllm.py defaults."""
    return _merge_llm_config(
        {
            "scheme": "http",
            "host": str(DEFAULT_SSH_VLLM_CONFIG["local_host"]),
            "port": int(DEFAULT_SSH_VLLM_CONFIG["local_port"]),
            "model": str(DEFAULT_SSH_VLLM_CONFIG["model"]).strip(),
            "api_key": str(DEFAULT_SSH_VLLM_CONFIG["api_key"]).strip() or DEFAULT_LLM_API_KEY,
            "temperature": _coerce_temperature(os.getenv("RAG_LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE)),
            "ssh": get_default_llm_ssh_config(),
        }
    )


def _merge_llm_config(base_config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(base_config)
    ssh_config = dict(base_config.get("ssh") or {})

    if overrides:
        merged.update(
            {
                key: value
                for key, value in overrides.items()
                if key != "ssh" and value is not None
            }
        )
        if isinstance(overrides.get("ssh"), dict):
            ssh_config.update(overrides["ssh"])

    merged["ssh"] = ssh_config
    merged["api_key"] = str(merged.get("api_key") or DEFAULT_LLM_API_KEY).strip() or DEFAULT_LLM_API_KEY
    merged["temperature"] = _coerce_temperature(
        merged.get("temperature"),
        default=DEFAULT_LLM_TEMPERATURE,
    )

    explicit_base_url = _normalize_base_url(merged.get("base_url"))
    if explicit_base_url:
        merged["base_url"] = explicit_base_url
    elif is_ssh_tunnel_enabled(ssh_config):
        merged["base_url"] = ensure_ssh_openai_base_url("llm", ssh_config)
    else:
        merged["base_url"] = f"{merged['scheme']}://{merged['host']}:{merged['port']}/v1"

    if not str(merged.get("model") or "").strip():
        discovered_model = discover_openai_model(
            merged["base_url"],
            api_key=str(merged.get("api_key") or DEFAULT_LLM_API_KEY),
        )
        if discovered_model:
            merged["model"] = discovered_model

    return merged


def get_default_llm_config() -> dict[str, Any]:
    """Return the default LLM config used by the RAG workflow."""
    config = _base_llm_config_from_env()
    config["ssh"] = get_default_llm_ssh_config()

    explicit_base_url = _normalize_base_url(os.getenv("RAG_LLM_BASE_URL"))
    if explicit_base_url:
        config["base_url"] = explicit_base_url

    return _merge_llm_config(config)


def _fetch_available_models(base_url: str, api_key: str, timeout: float = 20.0) -> list[str]:
    normalized_base_url = _normalize_base_url(base_url)
    if not normalized_base_url:
        return []

    headers = {}
    cleaned_api_key = str(api_key or DEFAULT_LLM_API_KEY).strip() or DEFAULT_LLM_API_KEY
    if cleaned_api_key:
        headers["Authorization"] = f"Bearer {cleaned_api_key}"

    try:
        response = requests.get(
            f"{normalized_base_url}/models",
            timeout=timeout,
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    models: list[str] = []
    for item in payload.get("data") or []:
        model_id = str((item or {}).get("id") or "").strip()
        if model_id:
            models.append(model_id)
    return models


@dataclass
class RuntimeLLMState:
    mode: str = "remote"
    temperature: float | None = None
    api_key: str = ""
    base_url: str = ""
    model: str = ""


class LLMConfigService:
    """Own the process-local LLM runtime configuration."""

    def __init__(
        self,
        *,
        default_config_loader: Callable[[], dict[str, Any]] | None = None,
        remote_config_loader: Callable[[], dict[str, Any]] | None = None,
        llm_builder: Callable[..., Any] | None = None,
        model_fetcher: Callable[[str, str, float], list[str]] | None = None,
    ) -> None:
        self._default_config_loader = default_config_loader or get_default_llm_config
        self._remote_config_loader = remote_config_loader or get_default_remote_llm_config
        self._llm_builder = llm_builder or ChatOpenAI
        self._model_fetcher = model_fetcher or _fetch_available_models
        self._lock = threading.RLock()
        self._state = RuntimeLLMState()

    def reset(self) -> None:
        with self._lock:
            self._state = RuntimeLLMState()
        try:
            stop_ssh_tunnel("llm")
        except Exception:
            pass

    def get_runtime_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": self._state.mode,
                "temperature": self._state.temperature,
                "api_key": self._state.api_key,
                "base_url": self._state.base_url,
                "model": self._state.model,
            }

    def get_effective_llm_config(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            runtime_state = RuntimeLLMState(
                mode=self._state.mode,
                temperature=self._state.temperature,
                api_key=self._state.api_key,
                base_url=self._state.base_url,
                model=self._state.model,
            )

        if runtime_state.mode == "api":
            effective = _merge_llm_config(self._default_config_loader())
        else:
            effective = _merge_llm_config(self._remote_config_loader())
        runtime_overrides: dict[str, Any] = {}
        if runtime_state.temperature is not None:
            runtime_overrides["temperature"] = runtime_state.temperature
        if runtime_state.api_key:
            runtime_overrides["api_key"] = runtime_state.api_key
        if runtime_state.base_url:
            runtime_overrides["base_url"] = runtime_state.base_url
        if runtime_state.model:
            runtime_overrides["model"] = runtime_state.model
        effective = _merge_llm_config(effective, runtime_overrides or None)
        return _merge_llm_config(effective, config)

    def create_llm(self, config: dict[str, Any] | None = None) -> Any:
        llm_config = self.get_effective_llm_config(config=config)
        return self._llm_builder(
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            temperature=llm_config["temperature"],
        )

    def validate_connection(self, *, mode: str, config: dict[str, Any] | None = None) -> bool:
        try:
            candidate = self._build_candidate_config(mode=mode, config=config)
        except Exception:
            return False

        available_models = self._model_fetcher(
            str(candidate.get("base_url") or ""),
            str(candidate.get("api_key") or DEFAULT_LLM_API_KEY),
            20.0,
        )
        if not available_models:
            return False

        candidate_model = str(candidate.get("model") or "").strip()
        if candidate_model and candidate_model not in available_models:
            return False
        return True

    def switch_to_remote(
        self,
        temperature: float | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
    ) -> bool:
        candidate_overrides: dict[str, Any] = {
            "api_key": DEFAULT_LLM_API_KEY,
        }
        if temperature is not None:
            candidate_overrides["temperature"] = temperature
        if str(base_url or "").strip():
            candidate_overrides["base_url"] = base_url
        if str(model or "").strip():
            candidate_overrides["model"] = model
        if not self.validate_connection(mode="remote", config=candidate_overrides):
            return False

        with self._lock:
            self._state.mode = "remote"
            if temperature is not None:
                self._state.temperature = _coerce_temperature(temperature)
            self._state.api_key = DEFAULT_LLM_API_KEY
            self._state.base_url = _normalize_base_url(base_url)
            self._state.model = str(model or "").strip()
        return True

    def update_temperature(self, temperature: float) -> bool:
        return self.switch_to_remote(temperature)

    def switch_to_api(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float | None = None,
    ) -> bool:
        candidate_overrides = {
            "api_key": str(api_key or "").strip(),
            "base_url": _normalize_base_url(base_url),
            "model": str(model or "").strip(),
        }
        if temperature is not None:
            candidate_overrides["temperature"] = temperature

        if not self.validate_connection(mode="api", config=candidate_overrides):
            return False

        with self._lock:
            self._state.mode = "api"
            self._state.api_key = candidate_overrides["api_key"]
            self._state.base_url = candidate_overrides["base_url"]
            self._state.model = candidate_overrides["model"]
            if temperature is not None:
                self._state.temperature = _coerce_temperature(temperature)
        try:
            stop_ssh_tunnel("llm")
        except Exception:
            pass
        return True

    def _build_candidate_config(self, *, mode: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        if mode == "remote":
            base_config = _merge_llm_config(self._remote_config_loader())
            return _merge_llm_config(base_config, config)
        if mode == "api":
            base_config = _merge_llm_config(self._default_config_loader())
            return _merge_llm_config(base_config, config)
        raise ValueError(f"Unsupported LLM mode: {mode}")


_DEFAULT_LLM_SERVICE: LLMConfigService | None = None
_DEFAULT_LLM_SERVICE_LOCK = threading.Lock()


def get_default_llm_service() -> LLMConfigService:
    global _DEFAULT_LLM_SERVICE
    with _DEFAULT_LLM_SERVICE_LOCK:
        if _DEFAULT_LLM_SERVICE is None:
            _DEFAULT_LLM_SERVICE = LLMConfigService()
        return _DEFAULT_LLM_SERVICE


def set_default_llm_service(service: LLMConfigService) -> LLMConfigService:
    global _DEFAULT_LLM_SERVICE
    with _DEFAULT_LLM_SERVICE_LOCK:
        _DEFAULT_LLM_SERVICE = service
        return _DEFAULT_LLM_SERVICE


def reset_default_llm_service() -> LLMConfigService:
    service = LLMConfigService()
    service.reset()
    return set_default_llm_service(service)


def create_default_llm(config: dict[str, Any] | None = None) -> Any:
    """Create the default LLM used across the academic RAG workflow."""
    return get_default_llm_service().create_llm(config=config)
