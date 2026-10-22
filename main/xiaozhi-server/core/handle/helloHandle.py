"""处理唤醒词确认音频及动态唤醒回复缓存。

@author pony
@date 2026-10-22
@version v1.3.0
@last_modified 2026-10-22
@changelog
    - v1.3.0 (2026-10-22): 接入本地短确认音，准备完成后再停录，失败恢复状态
    - v1.2.0 (2026-10-22): 缺省兼容唤醒缓存配置并确保异常播报结束状态
    - v1.1.0 (2026-10-22): 仅在提示音发送期间屏蔽回采，复用静态提示音缓存
"""

import time
import json
import uuid
import random
import asyncio
from core.utils.dialogue import Message
from core.utils.util import audio_to_data
from core.providers.tts.dto.dto import SentenceType
from core.utils.wakeup_word import WakeupWordsConfig
from core.handle.sendAudioHandle import sendAudioMessage, send_tts_message
from core.utils.util import remove_punctuation_and_length, opus_datas_to_wav_bytes
from core.providers.tools.device_mcp import (
    MCPClient,
    send_mcp_initialize_message
)

TAG = __name__

WAKEUP_CONFIG = {
    "refresh_time": 10,
    "responses": [
        "我一直都在呢，您请说。",
        "在的呢，请随时吩咐我。",
        "来啦来啦，请告诉我吧。",
        "您请说，我正听着。",
        "请您讲话，我准备好了。",
        "请您说出指令吧。",
        "我认真听着呢，请讲。",
        "请问您需要什么帮助？",
        "我在这里，等候您的指令。",
    ],
}

# 创建全局的唤醒词配置管理器
wakeup_words_config = WakeupWordsConfig()

# 用于防止并发调用wakeupWordsResponse的锁
_wakeup_response_lock = asyncio.Lock()

DEFAULT_SHORT_WAKEUP_AUDIO = "config/assets/wakeup_confirm.wav"
DEFAULT_CACHED_WAKEUP_AUDIO = "config/assets/wakeup_words_short.wav"


def _wakeup_response_mode(conn):
    """旧配置不指定模式时沿用动态问候。"""
    mode = conn.config.get("server", {}).get("wakeup_response_mode", "cached")
    if mode not in {"short", "cached"}:
        conn.logger.bind(tag=TAG).warning(
            f"未知唤醒确认模式 {mode!r}，回退到 cached"
        )
        return "cached"
    return mode


async def _get_wakeup_response(voice, mode):
    """按模式获取唤醒音频，短模式不触发云 TTS 或文件刷新。"""
    if mode == "short":
        return {
            "voice": voice,
            "file_path": DEFAULT_SHORT_WAKEUP_AUDIO,
            "time": time.time(),
            # 提示音没有说出文字，不向对话历史写入虚构的助手发言。
            "text": None,
        }

    response = await asyncio.to_thread(
        wakeup_words_config.get_wakeup_response, voice
    )
    if response and response.get("file_path"):
        return response
    return {
        "voice": "default",
        "file_path": DEFAULT_CACHED_WAKEUP_AUDIO,
        "time": 0,
        "text": "我在这里哦！",
    }


async def handleHelloMessage(conn, msg_json):
    """处理hello消息"""
    audio_params = msg_json.get("audio_params")
    if audio_params:
        format = audio_params.get("format")
        conn.logger.bind(tag=TAG).debug(f"客户端音频格式: {format}")
        conn.audio_format = format
        conn.welcome_msg["audio_params"] = audio_params
    features = msg_json.get("features")
    if features:
        conn.logger.bind(tag=TAG).debug(f"客户端特性: {features}")
        conn.features = features
        if features.get("mcp"):
            conn.logger.bind(tag=TAG).debug("客户端支持MCP")
            conn.mcp_client = MCPClient()
            # 发送初始化
            asyncio.create_task(send_mcp_initialize_message(conn))

    await conn.websocket.send(json.dumps(conn.welcome_msg))


async def checkWakeupWords(conn, text):
    enable_wakeup_words_response_cache = conn.config.get(
        "enable_wakeup_words_response_cache", True
    )

    # 普通对话不需要等待唤醒提示音组件就绪。
    _, filtered_text = remove_punctuation_and_length(text)
    if not enable_wakeup_words_response_cache or filtered_text not in conn.config.get("wakeup_words", []):
        return False

    # 等待tts初始化，最多等待3秒
    start_time = time.monotonic()
    while time.monotonic() - start_time < 3:
        if conn.tts:
            break
        await asyncio.sleep(0.1)
    else:
        return False

    mode = _wakeup_response_mode(conn)
    voice = getattr(conn.tts, "voice", None) or "default"
    try:
        response = await _get_wakeup_response(voice, mode)
        response_path = response["file_path"]
        use_cached_audio = (
            mode == "short" or response_path == DEFAULT_CACHED_WAKEUP_AUDIO
        )
        opus_packets = await audio_to_data(response_path, use_cache=use_cached_audio)
    except Exception as error:
        if mode == "short":
            raise
        # 动态音色文件刷新/失效不能让整次唤醒失败，回退到静态确认音。
        conn.logger.bind(tag=TAG).warning(
            f"唤醒问候不可用，使用短确认音: {type(error).__name__}"
        )
        mode = "short"
        response = await _get_wakeup_response(voice, mode)
        opus_packets = await audio_to_data(response["file_path"], use_cache=True)

    # 准备音频时不提前发送 start，避免文件锁和首次编码延长设备的停录窗口。
    # 仅在本地唤醒提示音发送期间屏蔽上行音频，避免设备把自己的提示音
    # 重新送回 ASR；提示音发送完成后立即恢复，不能再固定丢弃 2 秒用户语音。
    conn.just_woken_up = True
    tts_started = False
    tts_stopped = False
    try:
        tts_started = True
        await send_tts_message(conn, "start")

        # 播放唤醒词回复
        conn.client_abort = False

        # 将唤醒词回复视为新会话，生成新的 sentence_id，确保流控器重置
        conn.sentence_id = str(uuid.uuid4().hex)

        conn.logger.bind(tag=TAG).info(f"播放唤醒确认: mode={mode}")
        await sendAudioMessage(conn, SentenceType.FIRST, opus_packets, response.get("text"))
        await sendAudioMessage(conn, SentenceType.LAST, [], None)
        tts_stopped = True
    finally:
        try:
            # FIRST 发送失败时 LAST 不会运行，仍要结束设备的播报状态。
            if tts_started and not tts_stopped and not conn.stop_event.is_set():
                try:
                    await send_tts_message(conn, "stop")
                except Exception as error:
                    conn.logger.bind(tag=TAG).warning(
                        f"唤醒提示异常后未能发送TTS结束状态: {type(error).__name__}"
                    )
        finally:
            # stop 等待发送队列和预缓冲预算，设备实际播放完成仍需真机验证。
            # 异常和取消也必须解除保护，避免后续音频被永久丢弃。
            conn.just_woken_up = False

    # 补充对话
    if response.get("text"):
        conn.dialogue.put(Message(role="assistant", content=response["text"]))

    # 检查是否需要更新唤醒词回复
    if mode == "cached" and time.time() - response.get("time", 0) > WAKEUP_CONFIG["refresh_time"]:
        if not _wakeup_response_lock.locked():
            asyncio.create_task(wakeupWordsResponse(conn))
    return True


async def wakeupWordsResponse(conn):
    if not conn.tts:
        return

    try:
        # 尝试获取锁，如果获取不到就返回
        if not await _wakeup_response_lock.acquire():
            return

        # 从预定义回复列表中随机选择一个回复
        result = random.choice(WAKEUP_CONFIG["responses"])
        if not result or len(result) == 0:
            return

        # 生成TTS音频
        tts_result = await asyncio.to_thread(conn.tts.to_tts, result)
        if not tts_result:
            return

        # 获取当前音色
        voice = getattr(conn.tts, "voice", "default")

        # 使用链接的sample_rate
        wav_bytes = opus_datas_to_wav_bytes(tts_result, sample_rate=conn.sample_rate)
        file_path = wakeup_words_config.generate_file_path(voice)
        with open(file_path, "wb") as f:
            f.write(wav_bytes)
        # 更新配置
        wakeup_words_config.update_wakeup_response(voice, file_path, result)
    finally:
        # 确保在任何情况下都释放锁
        if _wakeup_response_lock.locked():
            _wakeup_response_lock.release()
