"""
独立测试：SSH 隧道 + vLLM 可用性 + 生成接口

用法示例：
python test_ssh_vllm.py ^
  --ssh-host 172.26.19.131 --ssh-port 8888 --ssh-username root --ssh-password "123456.a" ^
  --remote-port 8001 --local-port 18001 --model "DeepSeek-R1-0528-Qwen3-8B"
"""

from __future__ import annotations

import argparse
import atexit
import sys
import time
from typing import Any, Dict

import requests
import paramiko
from sshtunnel import SSHTunnelForwarder

# 兼容新版本 paramiko 移除 DSSKey 的情况（sshtunnel 旧版本仍可能引用）
if not hasattr(paramiko, "DSSKey"):
    fallback_key = getattr(paramiko, "RSAKey", None) or getattr(paramiko, "Ed25519Key", None)
    if fallback_key is not None:
        paramiko.DSSKey = fallback_key


DEFAULT_SSH_VLLM_CONFIG = {
    "ssh_host": "172.26.19.131",
    "ssh_port": 8888,
    "ssh_username": "root",
    "ssh_password": "123456.a",
    "remote_host": "127.0.0.1",
    "remote_port": 8001,
    "local_host": "127.0.0.1",
    "local_port": 18001,
    "model": "Qwen/Qwen3.5-9B",
    "api_key": "EMPTY",
}

_SSH_TUNNEL_SERVER: SSHTunnelForwarder | None = None
_SSH_TUNNEL_BASE_URL: str | None = None


def get_default_ssh_vllm_config() -> Dict[str, Any]:
    """Return the default SSH + vLLM config used by the shared LLM factory."""
    return dict(DEFAULT_SSH_VLLM_CONFIG)


def stop_ssh_vllm_tunnel() -> None:
    """Stop the shared SSH tunnel if it is active."""
    global _SSH_TUNNEL_SERVER, _SSH_TUNNEL_BASE_URL
    if _SSH_TUNNEL_SERVER is not None:
        try:
            _SSH_TUNNEL_SERVER.stop()
            print("[INFO] 共享 SSH 隧道已关闭")
        finally:
            _SSH_TUNNEL_SERVER = None
            _SSH_TUNNEL_BASE_URL = None


def ensure_ssh_vllm_tunnel(config: Dict[str, Any] | None = None) -> str:
    """Start or reuse a shared SSH tunnel and return the local OpenAI-compatible base URL."""
    global _SSH_TUNNEL_SERVER, _SSH_TUNNEL_BASE_URL

    if _SSH_TUNNEL_SERVER is not None and _SSH_TUNNEL_SERVER.is_active and _SSH_TUNNEL_BASE_URL:
        return _SSH_TUNNEL_BASE_URL

    cfg = get_default_ssh_vllm_config()
    if config:
        cfg.update(config)

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
        raise RuntimeError("SSH 隧道未激活")

    local_port = server.local_bind_port
    base_url = f"http://{cfg['local_host']}:{local_port}/v1"
    _SSH_TUNNEL_SERVER = server
    _SSH_TUNNEL_BASE_URL = base_url
    return base_url


def create_ssh_vllm_chat_openai(config: Dict[str, Any] | None = None):
    """Create a ChatOpenAI instance backed by the SSH-tunneled vLLM endpoint."""
    from langchain_openai import ChatOpenAI

    cfg = get_default_ssh_vllm_config()
    if config:
        cfg.update(config)

    base_url = ensure_ssh_vllm_tunnel(cfg)
    return ChatOpenAI(
        model=cfg["model"],
        api_key=cfg["api_key"],
        base_url=base_url,
    )


atexit.register(stop_ssh_vllm_tunnel)


def print_step(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def request_json(method: str, url: str, timeout: float = 20.0, **kwargs: Any) -> Dict[str, Any]:
    resp = requests.request(method=method, url=url, timeout=timeout, **kwargs)
    print(f"[HTTP] {method} {url} -> {resp.status_code}")
    text_preview = resp.text[:500] if resp.text else ""
    if text_preview:
        print(f"[BODY] {text_preview}")
    try:
        return resp.json()
    except Exception:
        return {"_raw_text": resp.text, "_status_code": resp.status_code}


def test_health(base_url: str) -> None:
    print_step("1) 测试 /health")
    data = request_json("GET", f"{base_url}/health", timeout=10.0)
    print("[OK] /health 返回：", data)


def test_models(base_url: str) -> Dict[str, Any]:
    print_step("2) 测试 /v1/models")
    data = request_json("GET", f"{base_url}/v1/models", timeout=20.0)
    print("[OK] /v1/models 返回成功")
    return data


def test_chat(base_url: str, model: str, prompt: str, api_key: str) -> bool:
    print_step("3) 测试 /v1/chat/completions")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个简洁的助手，请用中文回答。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 128,
    }
    data = request_json(
        "POST",
        f"{base_url}/v1/chat/completions",
        timeout=120.0,
        headers=headers,
        json=payload,
    )
    if "choices" in data:
        print("[OK] chat/completions 调用成功")
        return True
    print("[WARN] chat/completions 未返回 choices")
    return False


def test_completions(base_url: str, model: str, prompt: str, api_key: str) -> bool:
    print_step("4) 回退测试 /v1/completions")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.2,
        "max_tokens": 128,
    }
    data = request_json(
        "POST",
        f"{base_url}/v1/completions",
        timeout=120.0,
        headers=headers,
        json=payload,
    )
    if "choices" in data:
        print("[OK] completions 调用成功")
        return True
    print("[FAIL] completions 也未成功")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="测试 SSH 隧道与 vLLM 生成能力")
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--ssh-username", required=True)
    parser.add_argument("--ssh-password", required=True)
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument("--remote-port", type=int, default=8001)
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--local-port", type=int, default=18001)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    args = parser.parse_args()

    print_step("建立 SSH 隧道")
    server = SSHTunnelForwarder(
        (args.ssh_host, args.ssh_port),
        ssh_username=args.ssh_username,
        ssh_password=args.ssh_password,
        allow_agent=False,
        host_pkey_directories=[],
        remote_bind_address=(args.remote_host, args.remote_port),
        local_bind_address=(args.local_host, args.local_port),
    )

    try:
        server.start()
        time.sleep(1.5)
        if not server.is_active:
            print("[FAIL] SSH 隧道未激活")
            return 2
        local_port = server.local_bind_port
        base_url = f"http://{args.local_host}:{local_port}"
        print(f"[OK] SSH 隧道已激活: {base_url} -> {args.remote_host}:{args.remote_port}")

        test_health(base_url)
        models_data = test_models(base_url)

        # 打印可用模型 id
        if isinstance(models_data, dict) and isinstance(models_data.get("data"), list):
            model_ids = [m.get("id") for m in models_data["data"] if isinstance(m, dict)]
            print("[INFO] 可用模型：", model_ids)

        ok = test_chat(base_url, args.model, args.prompt, args.api_key)
        if not ok:
            ok = test_completions(base_url, args.model, args.prompt, args.api_key)

        if ok:
            print("\n[PASS] SSH + vLLM 端到端测试通过")
            return 0

        print("\n[FAIL] 生成接口测试失败")
        return 1

    except Exception as e:
        print(f"[EXCEPTION] {repr(e)}")
        return 3
    finally:
        try:
            server.stop()
            print("[INFO] SSH 隧道已关闭")
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
