"""天气、地点和提示词在固定场所部署时的配置回归验证。

@author pony
@date 2026-10-22
@version v1.0.1
@last_modified 2026-10-22
@changelog
    - v1.0.1 (2026-10-22): 显式验证天气上下文接收同一连接及配置城市
    - v1.0.0 (2026-10-22): 覆盖固定城市、显式位置、私网 IPv6 与外部请求超时
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


SERVER_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def location_runtime(monkeypatch):
    logger = Mock()
    logger.bind.return_value = logger
    logger_module = ModuleType("config.logger")
    logger_module.setup_logging = lambda: logger
    monkeypatch.setitem(sys.modules, "config.logger", logger_module)
    util_module = ModuleType("core.utils.util")
    util_module.get_ip_info = Mock(return_value={"city": "北京"})
    monkeypatch.setitem(sys.modules, "core.utils.util", util_module)
    cache_module = ModuleType("core.utils.cache.manager")
    cache_module.CacheType = SimpleNamespace(IP_INFO="ip", WEATHER="weather", LOCATION="location", DEVICE_PROMPT="prompt")
    cache_module.cache_manager = SimpleNamespace(get=Mock(return_value=None), set=Mock())
    monkeypatch.setitem(sys.modules, "core.utils.cache.manager", cache_module)

    def load(relative_path):
        spec = importlib.util.spec_from_file_location(
            "_location_runtime_" + Path(relative_path).stem, SERVER_ROOT / relative_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return SimpleNamespace(load=load, cache=cache_module, util=util_module, logger=logger)


@pytest.mark.parametrize("location, expected", [(None, "杭州"), ("南京", "南京")])
def test_weather_fixed_city_skips_ip_and_preserves_explicit_location(location_runtime, monkeypatch, location, expected):
    runtime = location_runtime
    module = runtime.load("plugins_func/functions/get_weather.py")
    fetch_city = Mock(return_value={"fxLink": "https://weather.invalid/city"})
    monkeypatch.setattr(module, "fetch_city_info", fetch_city)
    monkeypatch.setattr(module, "fetch_weather_page", lambda _: object())
    monkeypatch.setattr(module, "parse_weather_info", lambda _: (expected, "晴", {}, []))
    conn = SimpleNamespace(client_ip="8.8.8.8", config={"plugins": {"get_weather": {
        "use_ip_location": False, "default_location": "杭州", "api_key": "test-key"
    }}})
    result = module.get_weather(conn, location=location)
    assert expected in result.result
    assert fetch_city.call_args.args[0] == expected
    runtime.util.get_ip_info.assert_not_called()


def test_weather_keeps_ip_location_compatibility(location_runtime, monkeypatch):
    runtime = location_runtime
    module = runtime.load("plugins_func/functions/get_weather.py")
    fetch_city = Mock(return_value=None)
    monkeypatch.setattr(module, "fetch_city_info", fetch_city)
    conn = SimpleNamespace(client_ip="8.8.8.8", config={"plugins": {"get_weather": {"default_location": "杭州"}}})
    module.get_weather(conn)
    assert fetch_city.call_args.args[0] == "北京"
    runtime.util.get_ip_info.assert_called_once()


def test_place_fixed_city_skips_ip_lookup(location_runtime, monkeypatch):
    runtime = location_runtime
    module = runtime.load("plugins_func/functions/get_place_info.py")
    service = Mock()
    service.search_places_by_city.return_value = []
    service.format_search_results.return_value = "城市内结果"
    monkeypatch.setattr(module, "PlaceSearchService", lambda _: service)
    conn = SimpleNamespace(client_ip="8.8.8.8", config={"plugins": {"get_place_info": {
        "use_ip_location": False, "default_city": "杭州", "api_key": "test-key"
    }}})
    module.get_place_info(conn, keyword="电梯")
    service.get_location_by_ip.assert_not_called()
    service.search_places_by_city.assert_called_once_with("杭州", "电梯")


@pytest.mark.parametrize("client_ip", ["8.8.8.8", None])
def test_prompt_uses_fixed_city_without_stale_ip_cache(location_runtime, client_ip):
    runtime = location_runtime
    module = runtime.load("core/utils/prompt_manager.py")
    manager = module.PromptManager.__new__(module.PromptManager)
    manager.config = {"plugins": {"get_weather": {"use_ip_location": False, "default_location": "杭州"}}}
    manager.cache_manager = runtime.cache.cache_manager
    manager.CacheType = runtime.cache.CacheType
    manager.logger = runtime.logger
    manager.base_prompt_template = "{{ local_address }} / {{ weather_info }}"
    manager.context_data = ""
    manager._get_current_time_info = lambda: ("2026-10-22", "星期四", "")
    runtime.cache.cache_manager.get.side_effect = lambda kind, key: "上海" if kind == "location" else None
    manager._get_weather_info = Mock()
    conn = SimpleNamespace()
    manager.update_context_info(conn, client_ip)
    manager._get_weather_info.assert_called_once_with(conn, "杭州")
    result = manager.build_enhanced_prompt("提示词", "device", client_ip)
    assert result.startswith("杭州")
    assert "上海" not in result
    runtime.util.get_ip_info.assert_not_called()


@pytest.mark.parametrize("ip", ["192.168.1.2", "10.0.0.2", "::1", "fd12::1", "fe80::1", "invalid", None])
def test_non_public_ip_never_queries_server_location(location_runtime, monkeypatch, ip):
    runtime = location_runtime
    monkeypatch.setitem(sys.modules, "opuslib_next", ModuleType("opuslib_next"))
    module = runtime.load("core/utils/util.py")
    get = Mock(side_effect=AssertionError("不可为私网或无效地址查询服务器出口位置"))
    monkeypatch.setattr(module.requests, "get", get)
    assert module.get_ip_info(ip, runtime.logger) == {"city": None}
    get.assert_not_called()


def test_place_requests_use_https_and_timeouts(location_runtime, monkeypatch):
    module = location_runtime.load("plugins_func/functions/get_place_info.py")
    response = Mock()
    response.json.return_value = {"status": "1", "geocodes": [], "pois": []}
    get = Mock(return_value=response)
    monkeypatch.setattr(module.requests, "get", get)
    service = module.PlaceSearchService("test-key")
    service.get_location_by_name("杭州")
    service.search_places_by_keyword("120", "30", "餐厅")
    service.search_places_by_city("杭州", "餐厅")
    assert get.call_count == 3
    assert response.raise_for_status.call_count == 3
    for call in get.call_args_list:
        assert call.args[0].startswith("https://")
        assert call.kwargs["timeout"] == (3, 8)
