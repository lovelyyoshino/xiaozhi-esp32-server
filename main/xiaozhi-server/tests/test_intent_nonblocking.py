"""意图识别调用大模型时不阻塞事件循环的回归验证。

@author pony
@date 2026-11-12
@version v1.0.0
@last_modified 2026-11-12
@changelog
    - v1.0.0 (2026-11-12): 大模型慢响应期间事件循环仍能处理其他任务
"""

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock


SERVER_ROOT = Path(__file__).resolve().parents[1]


def load_intent_llm(monkeypatch):
    logger = Mock()
    logger.bind.return_value = logger

    class IntentProviderBase:
        def __init__(self, config):
            self.config = config

    stubs = {
        "core.providers.intent.base": {"IntentProviderBase": IntentProviderBase},
        "plugins_func.functions.play_music": {
            "initialize_music_handler": lambda conn: {"music_file_names": ""}
        },
        "config.logger": {"setup_logging": lambda: logger},
    }
    for name, attributes in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    name = "core.providers.intent.intent_llm.intent_llm"
    spec = importlib.util.spec_from_file_location(
        name, SERVER_ROOT / "core/providers/intent/intent_llm/intent_llm.py"
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "core.providers.intent.intent_llm"
    spec.loader.exec_module(module)
    return module


def test_slow_intent_llm_does_not_block_event_loop(monkeypatch):
    module = load_intent_llm(monkeypatch)

    def slow_llm(system_prompt, user_prompt):
        time.sleep(0.3)  # 同步客户端在等网络
        return '{"function_call": {"name": "get_weather", "arguments": {}}}'

    provider = object.__new__(module.IntentProvider)
    provider.llm = SimpleNamespace(response_no_stream=slow_llm, model_name="test")
    provider.promot = "intent prompt"
    provider.history_count = 4
    provider.cache_manager = SimpleNamespace(get=lambda *args: None, set=lambda *args: None)
    provider.CacheType = SimpleNamespace(INTENT="intent")
    conn = SimpleNamespace(
        func_handler=object(), device_id="device", config={"plugins": {}},
        dialogue=SimpleNamespace(dialogue=[]),
    )

    async def scenario():
        ticks = 0

        async def other_connection_work():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(other_connection_work())
        result = await provider.detect_intent(conn, [], "今天天气怎么样")
        task.cancel()
        return ticks, result

    ticks, result = asyncio.run(scenario())
    # 0.3 秒的大模型调用期间，其他协程应持续运行（阻塞时只会跑 1 次）
    assert ticks >= 10
    assert "get_weather" in result
