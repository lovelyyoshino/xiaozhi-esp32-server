"""流式 ASR 音频预录缓存与声纹音频生命周期管理。

@author pony
@date 2026-10-22
@version v1.0.0
@last_modified 2026-10-22
@changelog
    - v1.0.0 (2026-10-22): 空闲只保留预录帧，语音触发后保留完整语句
"""


VOICEPRINT_PREROLL_FRAMES = 10


def cache_voiceprint_audio(conn, audio, audio_have_voice):
    """语音触发前只保留预录帧，触发后保留完整语句供声纹及上报使用。"""
    audio_buffer = getattr(conn, "asr_audio_for_voiceprint", None)
    if audio_buffer is None:
        audio_buffer = []
        conn.asr_audio_for_voiceprint = audio_buffer

    if not audio:
        return audio_buffer

    capture_active = getattr(conn, "_asr_voiceprint_capture_active", False)
    if conn.client_listen_mode == "manual" or audio_have_voice or capture_active:
        audio_buffer.append(audio)
        conn._asr_voiceprint_capture_active = True
    else:
        audio_buffer.append(audio)
        if len(audio_buffer) > VOICEPRINT_PREROLL_FRAMES:
            del audio_buffer[:-VOICEPRINT_PREROLL_FRAMES]

    return audio_buffer


def clear_voiceprint_audio(conn):
    """结束当前录音段并释放缓存，下一段重新从预录状态开始。"""
    conn.asr_audio_for_voiceprint = []
    conn._asr_voiceprint_capture_active = False
