"""服务端插件工具执行器，为天气和地点查询隔离阻塞网络请求。

@author pony
@date 2026-10-22
@version v1.0.0
@last_modified 2026-10-22
@changelog
    - v1.0.0 (2026-10-22): 网络插件使用全进程有界专用线程池，取消时按实际任务回收名额
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from ..base import ToolType, ToolDefinition, ToolExecutor
from plugins_func.register import all_function_registry, Action, ActionResponse


# 只迁移已确认的同步网络插件；控制类插件可能需要在事件循环线程操作连接。
# 专用线程池不与 VAD 共用默认 executor，总预算包含正在运行和等待的请求。
_NETWORK_TOOLS = frozenset({"get_weather", "get_place_info"})
_NETWORK_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="location-api")
_NETWORK_TOOL_SLOTS = threading.BoundedSemaphore(8)


class ServerPluginExecutor(ToolExecutor):
    """服务端插件工具执行器"""

    def __init__(self, conn):
        self.conn = conn
        self.config = conn.config

    async def execute(
        self, conn, tool_name: str, arguments: Dict[str, Any]
    ) -> ActionResponse:
        """执行服务端插件工具"""
        func_item = all_function_registry.get(tool_name)
        if not func_item:
            return ActionResponse(
                action=Action.NOTFOUND, response=f"插件函数 {tool_name} 不存在"
            )

        try:
            if tool_name in _NETWORK_TOOLS:
                slots = _NETWORK_TOOL_SLOTS
                if not slots.acquire(blocking=False):
                    return ActionResponse(
                        action=Action.ERROR, response="天气或地点查询繁忙，请稍后重试。"
                    )
                try:
                    future = _NETWORK_TOOL_EXECUTOR.submit(func_item.func, conn, **arguments)
                except BaseException:
                    slots.release()
                    raise
                # await 被取消时，正在执行的 requests 仍可能未退出，不能提前释放名额。
                future.add_done_callback(lambda _: slots.release())
                return await asyncio.wrap_future(future)

            # 根据工具类型决定如何调用
            if hasattr(func_item, "type"):
                func_type = func_item.type
                if func_type.code in [4, 5]:  # SYSTEM_CTL, IOT_CTL (需要conn参数)
                    result = func_item.func(conn, **arguments)
                elif func_type.code == 2:  # WAIT
                    result = func_item.func(**arguments)
                elif func_type.code == 3:  # CHANGE_SYS_PROMPT
                    result = func_item.func(conn, **arguments)
                else:
                    result = func_item.func(**arguments)
            else:
                # 默认不传conn参数
                result = func_item.func(**arguments)

            return result

        except Exception as e:
            return ActionResponse(
                action=Action.ERROR,
                response=str(e),
            )

    def get_tools(self) -> Dict[str, ToolDefinition]:
        """获取所有注册的服务端插件工具"""
        tools = {}

        # 获取必要的函数
        necessary_functions = ["handle_exit_intent", "get_lunar"]

        # 获取配置中的函数
        config_functions = self.config["Intent"][
            self.config["selected_module"]["Intent"]
        ].get("functions", [])

        # 转换为列表
        if not isinstance(config_functions, list):
            try:
                config_functions = list(config_functions)
            except TypeError:
                config_functions = []

        # 合并所有需要的函数
        all_required_functions = list(set(necessary_functions + config_functions))

        for func_name in all_required_functions:
            func_item = all_function_registry.get(func_name)
            if func_item:
                # 从函数注册中获取描述
                fun_description = (
                    self.config.get("plugins", {})
                    .get(func_name, {})
                    .get("description", "")
                )
                if fun_description is not None and len(fun_description) > 0:
                    if "function" in func_item.description and isinstance(
                        func_item.description["function"], dict
                    ):
                        func_item.description["function"][
                            "description"
                        ] = fun_description
                tools[func_name] = ToolDefinition(
                    name=func_name,
                    description=func_item.description,
                    tool_type=ToolType.SERVER_PLUGIN,
                )

        return tools

    def has_tool(self, tool_name: str) -> bool:
        """检查是否有指定的服务端插件工具"""
        return tool_name in all_function_registry
