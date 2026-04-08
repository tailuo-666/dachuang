from __future__ import annotations

import atexit
import os
import threading
import time
from typing import Any

import requests


_TUNNEL_LOCK = threading.Lock()
_TUNNEL_SERVERS: dict[str, Any] = {}
_TUNNEL_BASE_URLS: dict[str, str] = {}
_TUNNEL_CONFIGS: dict[str, tuple[Any, ...]] = {}


def _read_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return str(default).strip()


def _read_int_env(*names: str, default: int) -> int:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            continue
    return int(default)


def _load_ssh_tunnel_forwarder():
    try:
        import paramiko
        from sshtunnel import SSHTunnelForwarder
    except ImportError as exc:
        raise RuntimeError(
            "SSH tunnel dependencies are unavailable. Install paramiko and sshtunnel before using remote model mode."
        ) from exc

    if not hasattr(paramiko, "DSSKey"):
        fallback_key = getattr(paramiko, "RSAKey", None) or getattr(paramiko, "Ed25519Key", None)
        if fallback_key is not None:
            paramiko.DSSKey = fallback_key

    return SSHTunnelForwarder


def build_ssh_service_config(
    service_name: str,
    *,
    default_remote_port: int,
    default_local_port: int,
    default_remote_host: str = "127.0.0.1",
    default_local_host: str = "127.0.0.1",
) -> dict[str, Any]:
    service_key = service_name.upper()
    ssh_prefix = f"RAG_{service_key}_SSH"
    return {
        "ssh_host": _read_env(f"{ssh_prefix}_HOST", "RAG_SSH_HOST"),
        "ssh_port": _read_int_env(f"{ssh_prefix}_PORT", "RAG_SSH_PORT", default=8888),
        "ssh_username": _read_env(f"{ssh_prefix}_USERNAME", "RAG_SSH_USERNAME"),
        "ssh_password": _read_env(f"{ssh_prefix}_PASSWORD", "RAG_SSH_PASSWORD"),
        "remote_host": _read_env(
            f"{ssh_prefix}_REMOTE_HOST",
            f"RAG_{service_key}_REMOTE_HOST",
            "RAG_SSH_REMOTE_HOST",
            default=default_remote_host,
        ),
        "remote_port": _read_int_env(
            f"{ssh_prefix}_REMOTE_PORT",
            f"RAG_{service_key}_REMOTE_PORT",
            default=default_remote_port,
        ),
        "local_host": _read_env(
            f"{ssh_prefix}_LOCAL_HOST",
            f"RAG_{service_key}_LOCAL_HOST",
            "RAG_SSH_LOCAL_HOST",
            default=default_local_host,
        ),
        "local_port": _read_int_env(
            f"{ssh_prefix}_LOCAL_PORT",
            f"RAG_{service_key}_LOCAL_PORT",
            default=default_local_port,
        ),
    }


def is_ssh_tunnel_enabled(config: dict[str, Any] | None) -> bool:
    cfg = config or {}
    required_values = [
        str(cfg.get("ssh_host") or "").strip(),
        str(cfg.get("ssh_username") or "").strip(),
        str(cfg.get("ssh_password") or "").strip(),
    ]
    return all(required_values)


def _config_fingerprint(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        config.get("ssh_host"),
        int(config.get("ssh_port") or 0),
        config.get("ssh_username"),
        config.get("ssh_password"),
        config.get("remote_host"),
        int(config.get("remote_port") or 0),
        config.get("local_host"),
        int(config.get("local_port") or 0),
    )


def _stop_tunnel_locked(service_name: str) -> None:
    server = _TUNNEL_SERVERS.pop(service_name, None)
    _TUNNEL_BASE_URLS.pop(service_name, None)
    _TUNNEL_CONFIGS.pop(service_name, None)
    if server is not None:
        try:
            server.stop()
        except Exception:
            pass


def stop_ssh_tunnel(service_name: str) -> None:
    with _TUNNEL_LOCK:
        _stop_tunnel_locked(service_name)


def stop_all_ssh_tunnels() -> None:
    with _TUNNEL_LOCK:
        for service_name in list(_TUNNEL_SERVERS.keys()):
            _stop_tunnel_locked(service_name)


def ensure_ssh_openai_base_url(service_name: str, config: dict[str, Any]) -> str:
    cfg = dict(config or {})
    if not is_ssh_tunnel_enabled(cfg):
        raise ValueError(
            f"SSH config for {service_name} is incomplete. Set host, username, and password first."
        )

    service_key = str(service_name or "").strip().lower() or "default"
    fingerprint = _config_fingerprint(cfg)

    with _TUNNEL_LOCK:
        cached_server = _TUNNEL_SERVERS.get(service_key)
        cached_base_url = _TUNNEL_BASE_URLS.get(service_key)
        cached_fingerprint = _TUNNEL_CONFIGS.get(service_key)

        if (
            cached_server is not None
            and getattr(cached_server, "is_active", False)
            and cached_base_url
            and cached_fingerprint == fingerprint
        ):
            return cached_base_url

        if cached_server is not None:
            _stop_tunnel_locked(service_key)

        SSHTunnelForwarder = _load_ssh_tunnel_forwarder()
        server = SSHTunnelForwarder(
            (cfg["ssh_host"], int(cfg["ssh_port"])),
            ssh_username=cfg["ssh_username"],
            ssh_password=cfg["ssh_password"],
            allow_agent=False,
            host_pkey_directories=[],
            remote_bind_address=(cfg["remote_host"], int(cfg["remote_port"])),
            local_bind_address=(cfg["local_host"], int(cfg["local_port"])),
        )
        server.start()
        time.sleep(1.5)
        if not server.is_active:
            try:
                server.stop()
            except Exception:
                pass
            raise RuntimeError(f"SSH tunnel for {service_key} was not activated.")

        base_url = f"http://{cfg['local_host']}:{server.local_bind_port}/v1"
        _TUNNEL_SERVERS[service_key] = server
        _TUNNEL_BASE_URLS[service_key] = base_url
        _TUNNEL_CONFIGS[service_key] = fingerprint
        return base_url


def discover_openai_model(base_url: str, *, api_key: str = "EMPTY", timeout: float = 20.0) -> str:
    normalized_base_url = str(base_url or "").strip().rstrip("/")
    if not normalized_base_url:
        return ""

    headers = {}
    cleaned_api_key = str(api_key or "EMPTY").strip() or "EMPTY"
    if cleaned_api_key:
        headers["Authorization"] = f"Bearer {cleaned_api_key}"

    try:
        resp = requests.get(
            f"{normalized_base_url}/models",
            timeout=timeout,
            headers=headers,
        )
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("data") or []:
            model_id = str((item or {}).get("id") or "").strip()
            if model_id:
                return model_id
    except Exception:
        return ""
    return ""


atexit.register(stop_all_ssh_tunnels)
