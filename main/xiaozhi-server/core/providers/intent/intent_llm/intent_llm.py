from typing import List, Dict
from ..base import IntentProviderBase
from plugins_func.functions.play_music import initialize_music_handler
from config.logger import setup_logging
import re
import json
import hashlib
import time

TAG = __name__
logger = setup_logging()


class IntentProvider(IntentProviderBase):
    def __init__(self, config):
        super().__init__(config)
        self.llm = None
        self.promot = ""
        # 导入全局缓存管理器
        from core.utils.cache.manager import cache_manager, CacheType

        self.cache_manager = cache_manager
        self.CacheType = CacheType
        self.history_count = 4  # 默认使用最近4条对话记录

    def get_intent_system_prompt(self, functions_list: str) -> str:
        """
        根据配置的意图选项和可用函数动态生成系统提示词
        Args:
            functions: 可用的函数列表，JSON格式字符串
        Returns:
            格式化后的系统提示词
        """

        # 构建函数说明部分
        functions_desc = "可用的函数列表：\n"
        for func in functions_list:
            func_info = func.get("function", {})
            name = func_info.get("name", "")
            desc = func_info.get("description", "")
            params = func_info.get("parameters", {})

            functions_desc += f"\n函数名: {name}\n"
            functions_desc += f"描述: {desc}\n"

            if params:
                functions_desc += "参数:\n"
                for param_name, param_info in params.get("properties", {}).items():
                    param_desc = param_info.get("description", "")
                    param_type = param_info.get("type", "")
                    functions_desc += f"- {param_name} ({param_type}): {param_desc}\n"

            functions_desc += "---\n"

        prompt = (
            "你是意图识别助手，只返回JSON格式，禁止返回任何自然语言！\n\n"
            "【核心规则】\n"
            "1. 基础信息查询（时间/日期/农历/城市）→ result_for_context\n"
            "2. 系统状态反馈（即将到达/已到达/正在执行/抱歉XX未录入/好的正在XX）→ continue_chat\n"
            "3. 退出疑问（怎么退出？为什么退出？）→ continue_chat\n"
            "4. 明确退出指令（退出系统/结束对话/不想说话了）→ handle_exit_intent\n"
            "5. 智能家居设备：用中文名匹配，支持同义词（卫生间=厕所=洗手间，客厅=大厅，卧室=房间，书房=工作室，厨房=灶间）\n\n"
            f"{functions_desc}\n"
            "示例：\n"
            "```\n"
            "用户: 现在几点了？\n"
            '返回: {"function_call": {"name": "result_for_context"}}\n'
            "```\n"
            "```\n"
            "用户: 我想结束对话\n"
            '返回: {"function_call": {"name": "handle_exit_intent", "arguments": {"say_goodbye": "goodbye"}}}\n'
            "```\n"
            "```\n"
            "用户: 你好啊\n"
            '返回: {"function_call": {"name": "continue_chat"}}\n'
            "```\n"
            "```\n"
            "用户: 带我去车间\n"
            '返回: {"function_call": {"name": "guidebot_navigateto", "arguments": {"destination": "车间", "is_user_input": true}}}\n'
            "```\n"
            "```\n"
            "状态通知: 即将到达车间\n"
            '返回: {"function_call": {"name": "continue_chat"}}\n'
            "```\n"
            "```\n"
            "状态通知: 已到达车间\n"
            '返回: {"function_call": {"name": "continue_chat"}}\n'
            "```\n"
            "```\n"
            "状态通知: 好的，正在为您导航至车间，请稍等。\n"
            '返回: {"function_call": {"name": "continue_chat"}}\n'
            "```\n"
            "```\n"
            "错误反馈: 直接回复：主人，未找到该地点，你觉得xx可以吗？\n"
            '返回: {"function_call": {"name": "continue_chat"}}\n'
            "```\n"
            "```\n"
            "错误反馈: 直接回复：主人抱歉，该地点暂未录入系统。\n"
            '返回: {"function_call": {"name": "continue_chat"}}\n'
            "```\n"
            "```\n"
            "```\n\n"
            "注意：\n"
            "1. 只返回JSON格式，不要包含任何其他文字\n"
            '2. 优先检查用户查询是否为基础信息（时间、日期等），如是则返回{"function_call": {"name": "result_for_context"}}，不需要arguments参数\n'
            '3. 如果是机器人的状态通知（如"即将到达"、"已到达"、"正在执行"等），返回{"function_call": {"name": "continue_chat"}}\n'
            '4. 如果是机器人的错误反馈（如"抱歉，XX暂未录入"、"XX没有录入"、"建议您确认"等），返回{"function_call": {"name": "continue_chat"}}\n'
            '5. 如果没有找到匹配的函数，返回{"function_call": {"name": "continue_chat"}}\n'
            "6. 确保返回的JSON格式正确，包含所有必要的字段\n"
            "7. result_for_context不需要任何参数，系统会自动从上下文获取信息\n"
            "8. 区分用户主动请求（如'带我去XX'）和系统反馈（如'即将到达XX'、'抱歉XX未录入'），前者需要调用函数，后者返回continue_chat\n"
            "9. 特别注意：以'抱歉'、'稍后'、'建议'、'我得先'、'我马上'等开头的句子通常是系统反馈，不是用户指令\n"
            "【最终警告】绝对禁止输出任何自然语言、表情符号或解释文字！只能输出有效JSON格式！违反此规则将导致系统错误！"
        )
        return prompt

    def replyResult(self, text: str, original_text: str):
        llm_result = self.llm.response_no_stream(
            system_prompt=text,
            user_prompt="请根据以上内容，像人类一样说话的口吻回复用户，要求简洁，请直接返回结果。用户现在说："
            + original_text,
        )
        return llm_result

    async def detect_intent(self, conn, dialogue_history: List[Dict], text: str) -> str:
        if not self.llm:
            raise ValueError("LLM provider not set")
        if conn.func_handler is None:
            return '{"function_call": {"name": "continue_chat"}}'

        # 记录整体开始时间
        total_start_time = time.time()

        # 打印使用的模型信息
        model_info = getattr(self.llm, "model_name", str(self.llm.__class__.__name__))
        logger.bind(tag=TAG).debug(f"使用意图识别模型: {model_info}")

        # 计算缓存键
        cache_key = hashlib.md5((conn.device_id + text).encode()).hexdigest()

        # 检查缓存
        cached_intent = self.cache_manager.get(self.CacheType.INTENT, cache_key)
        if cached_intent is not None:
            cache_time = time.time() - total_start_time
            logger.bind(tag=TAG).debug(
                f"使用缓存的意图: {cache_key} -> {cached_intent}, 耗时: {cache_time:.4f}秒"
            )
            return cached_intent

        if self.promot == "":
            functions = conn.func_handler.get_functions()
            if hasattr(conn, "mcp_client"):
                mcp_tools = conn.mcp_client.get_available_tools()
                if mcp_tools is not None and len(mcp_tools) > 0:
                    if functions is None:
                        functions = []
                    functions.extend(mcp_tools)

            self.promot = self.get_intent_system_prompt(functions)

        music_config = initialize_music_handler(conn)
        music_file_names = music_config["music_file_names"]
        prompt_music = f"{self.promot}\n<musicNames>{music_file_names}\n</musicNames>"

        home_assistant_cfg = conn.config["plugins"].get("home_assistant")
        if home_assistant_cfg:
            devices = home_assistant_cfg.get("devices", [])
        else:
            devices = []
        if len(devices) > 0:
            hass_prompt = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【智能家居设备列表】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

格式说明：位置,设备名,entity_id

【重要】设备名称匹配注意事项：
1. 用户输入的位置或设备名称必须与下方列表中的中文名称进行匹配
2. 严禁将用户输入的中文翻译成英文后再匹配
   例如：用户说"卫生间"，不要翻译成"toilet"、"bathroom"等英文词汇
3. 支持同义词匹配：
   - 卫生间 ≈ 厕所 ≈ 洗手间 ≈ WC
   - 客厅 ≈ 大厅 ≈ 起居室
   - 卧室 ≈ 房间 ≈ 睡房
4. 支持模糊匹配：
   - "卫生间的灯" → 匹配 "卫生间,灯"
   - "客厅台灯" → 匹配 "客厅,台灯"
   - "打开厕所灯" → 匹配 "卫生间,灯"

设备列表：
"""
            for device in devices:
                hass_prompt += device + "\n"
            hass_prompt += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            prompt_music += hass_prompt

        logger.bind(tag=TAG).debug(f"User prompt: {prompt_music}")

        # 构建用户对话历史的提示
        msgStr = ""

        # 获取最近的对话历史
        start_idx = max(0, len(dialogue_history) - self.history_count)
        for i in range(start_idx, len(dialogue_history)):
            msgStr += f"{dialogue_history[i].role}: {dialogue_history[i].content}\n"

        msgStr += f"User: {text}\n"
        user_prompt = f"current dialogue:\n{msgStr}"

        # 记录预处理完成时间
        preprocess_time = time.time() - total_start_time
        logger.bind(tag=TAG).debug(f"意图识别预处理耗时: {preprocess_time:.4f}秒")

        # 使用LLM进行意图识别
        llm_start_time = time.time()
        logger.bind(tag=TAG).debug(f"开始LLM意图识别调用, 模型: {model_info}")

        intent = self.llm.response_no_stream(
            system_prompt=prompt_music, user_prompt=user_prompt
        )

        # 记录LLM调用完成时间
        llm_time = time.time() - llm_start_time
        logger.bind(tag=TAG).debug(
            f"外挂的大模型意图识别完成, 模型: {model_info}, 调用耗时: {llm_time:.4f}秒"
        )

        # 记录后处理开始时间
        postprocess_start_time = time.time()

        # 清理和解析响应
        intent = intent.strip()
        # 尝试提取JSON部分
        match = re.search(r"\{.*\}", intent, re.DOTALL)
        if match:
            intent = match.group(0)

        # 记录总处理时间
        total_time = time.time() - total_start_time
        logger.bind(tag=TAG).debug(
            f"【意图识别性能】模型: {model_info}, 总耗时: {total_time:.4f}秒, LLM调用: {llm_time:.4f}秒, 查询: '{text[:20]}...'"
        )

        # 尝试解析为JSON
        try:
            intent_data = json.loads(intent)
            # 如果包含function_call，则格式化为适合处理的格式
            if "function_call" in intent_data:
                function_data = intent_data["function_call"]
                function_name = function_data.get("name")
                function_args = function_data.get("arguments", {})

                # 记录识别到的function call
                logger.bind(tag=TAG).info(
                    f"llm 识别到意图: {function_name}, 参数: {function_args}"
                )

                # 处理不同类型的意图
                if function_name == "result_for_context":
                    # 处理基础信息查询，直接从context构建结果
                    logger.bind(tag=TAG).info(
                        "检测到result_for_context意图，将使用上下文信息直接回答"
                    )

                elif function_name == "continue_chat":
                    # 处理普通对话
                    # 保留非工具相关的消息
                    clean_history = [
                        msg
                        for msg in conn.dialogue.dialogue
                        if msg.role not in ["tool", "function"]
                    ]
                    conn.dialogue.dialogue = clean_history

                else:
                    # 处理函数调用
                    logger.bind(tag=TAG).info(f"检测到函数调用意图: {function_name}")

            # 统一缓存处理和返回
            self.cache_manager.set(self.CacheType.INTENT, cache_key, intent)
            postprocess_time = time.time() - postprocess_start_time
            logger.bind(tag=TAG).debug(f"意图后处理耗时: {postprocess_time:.4f}秒")
            return intent
        except json.JSONDecodeError:
            # 后处理时间
            postprocess_time = time.time() - postprocess_start_time
            logger.bind(tag=TAG).error(
                f"无法解析意图JSON: {intent}, 后处理耗时: {postprocess_time:.4f}秒"
            )
            # 如果解析失败，默认返回继续聊天意图
            return '{"function_call": {"name": "continue_chat"}}'
