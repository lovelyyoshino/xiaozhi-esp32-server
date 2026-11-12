"""智控台模式下本地运行参数透传的回归验证。

@author pony
@date 2026-10-22
@version v1.4.0
@last_modified 2026-11-12
@changelog
    - v1.4.0 (2026-11-12): 验证唤醒词门控与初始化排队时长配置透传
    - v1.3.0 (2026-10-22): 验证唤醒模式和拥塞反馈参数透传
    - v1.2.0 (2026-10-22): 覆盖本地 Opus 编码复杂度透传及缺省兼容
    - v1.1.0 (2026-10-22): 覆盖共享后台任务上限的智控台配置透传
    - v1.0.0 (2026-10-22): 验证音频队列、上报预算和压缩配置不被云端配置覆盖
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock


SERVER_ROOT = Path(__file__).resolve().parents[1]


def load_config_loader(monkeypatch, remote_config):
    manage_api = ModuleType("config.manage_api_client")
    manage_api.init_service = lambda config: None
    manage_api.get_server_config = AsyncMock(return_value=remote_config)
    manage_api.get_agent_models = AsyncMock()
    monkeypatch.setitem(sys.modules, "config.manage_api_client", manage_api)

    spec = importlib.util.spec_from_file_location(
        "_config_loader_runtime_test", SERVER_ROOT / "config/config_loader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_api_config_preserves_local_runtime_budget_and_remote_auth(monkeypatch):
    remote_config = {
        "server": {"auth": {"enabled": True}, "api_only_value": "remote"},
        "prompt_template": "remote-prompt",
    }
    module = load_config_loader(monkeypatch, remote_config)
    local_config = {
        "server": {
            "ip": "127.0.0.1",
            "port": 9000,
            "http_port": 9003,
            "vision_explain": "http://local/vision",
            "auth_key": "local-auth-key",
            "audio_queue_maxsize": 24,
            "wakeup_response_mode": "short",
            "reply_admission_wait_ms": 200,
            "reply_busy_audio": "config/assets/tts_notify.mp3",
            "reply_busy_text": "请稍后重试。",
            "report_max_pending": 48,
            "connection_worker_max": 12,
            "connection_max_pending": 36,
            "websocket_compression": None,
            "tts_opus_complexity": 5,
            "wake_word_gate": {"enabled": True, "words": ["小蒜"]},
            "init_admission_wait_ms": 15000,
        },
        "manager-api": {"url": "http://manager", "secret": "local-secret"},
        "prompt_template": "local-prompt",
    }

    result = asyncio.run(module.get_config_from_api_async(local_config))

    assert result["server"] == {
        "ip": "127.0.0.1",
        "port": 9000,
        "http_port": 9003,
        "vision_explain": "http://local/vision",
        "auth_key": "local-auth-key",
        "audio_queue_maxsize": 24,
        "wakeup_response_mode": "short",
        "reply_admission_wait_ms": 200,
        "reply_busy_audio": "config/assets/tts_notify.mp3",
        "reply_busy_text": "请稍后重试。",
        "report_max_pending": 48,
        "connection_worker_max": 12,
        "connection_max_pending": 36,
        "websocket_compression": None,
        "tts_opus_complexity": 5,
        "wake_word_gate": {"enabled": True, "words": ["小蒜"]},
        "init_admission_wait_ms": 15000,
        "auth": {"enabled": True},
    }
    assert "api_only_value" not in result["server"]
    assert result["manager-api"] == {
        "url": "http://manager",
        "secret": "local-secret",
    }
    assert result["prompt_template"] == "remote-prompt"


def test_api_config_omits_optional_runtime_keys_when_local_does_not_define_them(monkeypatch):
    module = load_config_loader(monkeypatch, {"server": {"auth": {"enabled": False}}})
    local_config = {
        "server": {"ip": "0.0.0.0", "port": 8000},
        "manager-api": {"url": "http://manager", "secret": "secret"},
    }

    result = asyncio.run(module.get_config_from_api_async(local_config))

    assert "audio_queue_maxsize" not in result["server"]
    assert "wakeup_response_mode" not in result["server"]
    assert "reply_admission_wait_ms" not in result["server"]
    assert "reply_busy_audio" not in result["server"]
    assert "reply_busy_text" not in result["server"]
    assert "report_max_pending" not in result["server"]
    assert "connection_worker_max" not in result["server"]
    assert "connection_max_pending" not in result["server"]
    assert "websocket_compression" not in result["server"]
    assert "tts_opus_complexity" not in result["server"]
    assert "wake_word_gate" not in result["server"]
    assert "init_admission_wait_ms" not in result["server"]
    assert result["server"]["auth"] == {"enabled": False}
