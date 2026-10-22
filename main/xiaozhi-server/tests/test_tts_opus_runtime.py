"""验证云 TTS 的本地编码预算与设备音频兼容性。

@author pony
@date 2026-10-22
@version v1.1.0
@last_modified 2026-10-22
@changelog
    - v1.1.0 (2026-10-22): 核对确认音时长与真实 Opus 播放格式
    - v1.0.0 (2026-10-22): 覆盖真实 Opus 解码、分块收尾及预建编码器的预算应用
"""

import asyncio
import importlib.util
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


SERVER_ROOT = Path(__file__).resolve().parents[1]


def test_wakeup_confirmation_is_short_mono_pcm():
    with wave.open(str(SERVER_ROOT / "config/assets/wakeup_confirm.wav"), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        assert 0 < audio.getnframes() / audio.getframerate() <= 0.18
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.int32)
        assert 0 < np.max(np.abs(pcm)) < 32767
        assert pcm[0] == pcm[-1] == 0


def test_wakeup_confirmation_is_three_decodable_opus_packets(opus_runtime):
    opus, encoder_type = opus_runtime
    with wave.open(str(SERVER_ROOT / "config/assets/wakeup_confirm.wav"), "rb") as audio:
        pcm = audio.readframes(audio.getnframes())
    encoder = encoder_type(16000, 1, 60)
    packets = []
    try:
        encoder.encode_pcm_to_opus_stream(pcm, True, packets.append)
        assert len(packets) == 3
        decoder = opus.Decoder(16000, 1)
        decoded = b"".join(decoder.decode(packet, 960) for packet in packets)
        assert len(decoded) == 5760
    finally:
        encoder.close()


@pytest.fixture
def opus_runtime():
    opus = pytest.importorskip("opuslib_next")
    from core.utils.opus_encoder_utils import OpusEncoderUtils

    return opus, OpusEncoderUtils


@pytest.mark.parametrize("complexity", [0, 5, 10])
def test_tts_pcm_chunks_remain_decodable_at_supported_complexities(opus_runtime, complexity):
    opus, encoder_type = opus_runtime
    encoder = encoder_type(24000, 1, 60)
    decoder = opus.Decoder(24000, 1)
    # 输入两帧多 17 个采样点，模拟网络 TTS 按任意字节块返回，尾包应补齐。
    count = 1440 * 2 + 17
    pcm = (np.sin(np.arange(count) * 2 * np.pi * 440 / 24000) * 12000).astype("<i2").tobytes()
    packets = []
    try:
        encoder.set_complexity(complexity)
        assert encoder.encoder.complexity == complexity
        encoder.encode_pcm_to_opus_stream(pcm[:600], False, packets.append)
        assert not packets
        encoder.encode_pcm_to_opus_stream(pcm[600:4000], False, packets.append)
        encoder.encode_pcm_to_opus_stream(pcm[4000:], True, packets.append)
        assert len(packets) == 3
        decoded = b"".join(decoder.decode(packet, 1440) for packet in packets)
        assert len(decoded) == 3 * 1440 * 2
        assert np.max(np.abs(np.frombuffer(decoded, dtype=np.int16).astype(np.int32))) > 0
        assert len(encoder.buffer) == 0
    finally:
        encoder.close()


@pytest.mark.parametrize("invalid", [-1, 11, "invalid"])
def test_invalid_complexity_preserves_working_encoder(opus_runtime, invalid):
    _, encoder_type = opus_runtime
    encoder = encoder_type(24000, 1, 60)
    try:
        encoder.set_complexity(5)
        with pytest.raises(ValueError):
            encoder.set_complexity(invalid)
        assert encoder.encoder.complexity == 5
    finally:
        encoder.close()


@pytest.mark.parametrize("precreated", [False, True])
@pytest.mark.parametrize("server_config, expected", [({"tts_opus_complexity": 5}, 5), ({}, 10)])
def test_tts_open_applies_budget_without_overwriting_precreated_sample_rate(monkeypatch, precreated, server_config, expected):
    # 只隔离云 SDK、日志和播报线程；真实开通音频通道代码负责选择编码器与预算。
    dependencies = {
        "config.logger": {"setup_logging": lambda: Mock()},
        "core.utils.tts": {"MarkdownCleaner": Mock()},
        "core.handle.reportHandle": {"enqueue_tts_report": Mock()},
        "core.handle.sendAudioHandle": {"sendAudioMessage": Mock()},
        "core.utils.util": {"audio_bytes_to_data_stream": Mock(), "audio_to_data_stream": Mock()},
        "core.utils.opus_encoder_utils": {"OpusEncoderUtils": Mock()},
    }
    for name, attributes in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(__import__("sys").modules, name, module)
    spec = importlib.util.spec_from_file_location("_tts_opus_runtime_base", SERVER_ROOT / "core/providers/tts/base.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # core.utils 可能已缓存子模块属性，显式替换工厂，避免依赖测试运行顺序。
    monkeypatch.setattr(module.opus_encoder_utils, "OpusEncoderUtils", Mock())
    thread = Mock()
    monkeypatch.setattr(module.threading, "Thread", thread)

    class Provider(module.TTSProviderBase):
        async def text_to_speak(self, text, output_file):
            return b""

    provider = Provider({}, True)
    existing = Mock(sample_rate=24000)
    if precreated:
        provider.opus_encoder = existing
    conn = SimpleNamespace(sample_rate=16000, config={"server": server_config})
    asyncio.run(provider.open_audio_channels(conn))
    if precreated:
        assert provider.opus_encoder is existing
        assert provider.opus_encoder.sample_rate == 24000
        module.opus_encoder_utils.OpusEncoderUtils.assert_not_called()
    else:
        module.opus_encoder_utils.OpusEncoderUtils.assert_called_once_with(
            sample_rate=16000, channels=1, frame_size_ms=60
        )
    provider.opus_encoder.set_complexity.assert_called_once_with(expected)
