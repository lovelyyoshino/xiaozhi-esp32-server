"""验证天气与地点查询的线程隔离、拥塞及取消边界。

@author pony
@date 2026-10-22
@version v1.0.0
@last_modified 2026-10-22
@changelog
    - v1.0.0 (2026-10-22): 覆盖慢插件不阻塞事件循环、全局限额与取消后名额回收
"""

import asyncio
import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


SERVER_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plugin_executor(monkeypatch):
    logger_module = ModuleType("config.logger")
    logger = Mock()
    logger.bind.return_value = logger
    logger_module.setup_logging = lambda: logger
    monkeypatch.setitem(sys.modules, "config.logger", logger_module)
    spec = importlib.util.spec_from_file_location(
        "core.providers.tools.server_plugins._runtime_test",
        SERVER_ROOT / "core/providers/tools/server_plugins/plugin_executor.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "all_function_registry", {})
    return module


@pytest.mark.parametrize("tool_name", ["get_weather", "get_place_info"])
def test_slow_network_plugin_keeps_event_loop_responsive(plugin_executor, tool_name):
    module = plugin_executor
    conn = SimpleNamespace(config={})
    release = threading.Event()
    invoked = []

    def slow_query(connection, *, location):
        invoked.append((threading.get_ident(), connection, location))
        release.wait(timeout=0.5)
        return module.ActionResponse(module.Action.REQLLM, result="查询完成")

    module.all_function_registry[tool_name] = SimpleNamespace(
        type=SimpleNamespace(code=4), func=slow_query
    )

    async def scenario():
        main_thread = threading.get_ident()
        executor = module.ServerPluginExecutor(conn)
        task = asyncio.create_task(executor.execute(conn, tool_name, {"location": "杭州"}))
        try:
            # 真实执行器调用慢同步插件期间，事件循环应仍可调度其他设备任务。
            for _ in range(200):
                if invoked:
                    break
                await asyncio.sleep(0.001)
            assert invoked
            assert invoked[0] == (invoked[0][0], conn, "杭州")
            assert invoked[0][0] != main_thread
            assert not task.done()
        finally:
            release.set()
            result = await asyncio.wait_for(task, 2)
        assert result.result == "查询完成"

    asyncio.run(scenario())


def test_network_plugin_budget_is_shared_and_survives_cancellation(plugin_executor, monkeypatch):
    module = plugin_executor
    slots = threading.BoundedSemaphore(1)
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(module, "_NETWORK_TOOL_SLOTS", slots, raising=False)
    monkeypatch.setattr(module, "_NETWORK_TOOL_EXECUTOR", pool, raising=False)
    release, entered = threading.Event(), threading.Event()
    finished = threading.Event()

    def slow_query(conn):
        entered.set()
        assert release.wait(timeout=5)
        return module.ActionResponse(module.Action.REQLLM, result="天气")

    module.all_function_registry["get_weather"] = SimpleNamespace(
        type=SimpleNamespace(code=4), func=slow_query
    )

    async def scenario():
        first_conn, second_conn = SimpleNamespace(config={}), SimpleNamespace(config={})
        first = module.ServerPluginExecutor(first_conn)
        second = module.ServerPluginExecutor(second_conn)
        task = asyncio.create_task(first.execute(first_conn, "get_weather", {}))
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(0.001)
            assert entered.is_set()
            overloaded = await second.execute(second_conn, "get_weather", {})
            assert overloaded.action == module.Action.ERROR
            assert "繁忙" in overloaded.response
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # asyncio 取消不能停止已运行的 HTTP 线程，线程退出前不得释放名额。
            assert not slots.acquire(blocking=False)
            overloaded = await second.execute(second_conn, "get_weather", {})
            assert overloaded.action == module.Action.ERROR
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        pool.submit(finished.set)
        for _ in range(200):
            if finished.is_set():
                break
            await asyncio.sleep(0.001)
        assert finished.is_set()
        assert slots.acquire(blocking=False)
        slots.release()

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        pool.shutdown(wait=True, cancel_futures=True)


def test_non_network_control_plugin_stays_on_event_loop(plugin_executor):
    module = plugin_executor
    conn = SimpleNamespace(config={})
    threads = []

    def control(connection):
        assert connection is conn
        threads.append(threading.get_ident())
        return module.ActionResponse(module.Action.NONE)

    module.all_function_registry["handle_exit_intent"] = SimpleNamespace(
        type=SimpleNamespace(code=4), func=control
    )
    asyncio.run(module.ServerPluginExecutor(conn).execute(conn, "handle_exit_intent", {}))
    assert threads == [threading.get_ident()]
