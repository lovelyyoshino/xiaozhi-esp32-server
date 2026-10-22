"""验证流式 ASR 的预录缓存和手动录音保留边界。

@author pony
@date 2026-10-22
@version v1.0.0
@last_modified 2026-10-22
@changelog
    - v1.0.0 (2026-10-22): 覆盖空闲预录、语音触发、手动模式和缓存重置
"""

import unittest
from types import SimpleNamespace

from core.providers.asr.audio_buffer import (
    VOICEPRINT_PREROLL_FRAMES,
    cache_voiceprint_audio,
    clear_voiceprint_audio,
)


class VoiceprintAudioBufferTest(unittest.TestCase):
    def make_connection(self, mode="auto"):
        return SimpleNamespace(client_listen_mode=mode)

    def test_silence_keeps_only_preroll_and_voice_keeps_full_utterance(self):
        conn = self.make_connection()
        frames = [bytes([index]) for index in range(40)]

        for frame in frames[:30]:
            cache_voiceprint_audio(conn, frame, audio_have_voice=False)

        self.assertEqual(
            conn.asr_audio_for_voiceprint,
            frames[30 - VOICEPRINT_PREROLL_FRAMES : 30],
        )

        cache_voiceprint_audio(conn, frames[30], audio_have_voice=True)
        for frame in frames[31:]:
            cache_voiceprint_audio(conn, frame, audio_have_voice=False)

        self.assertEqual(conn.asr_audio_for_voiceprint, frames[30 - VOICEPRINT_PREROLL_FRAMES :])

    def test_manual_mode_keeps_all_frames_without_vad_trigger(self):
        conn = self.make_connection(mode="manual")
        frames = [bytes([index]) for index in range(25)]

        for frame in frames:
            cache_voiceprint_audio(conn, frame, audio_have_voice=False)

        self.assertEqual(conn.asr_audio_for_voiceprint, frames)
        self.assertTrue(conn._asr_voiceprint_capture_active)

    def test_clear_restarts_preroll_window_and_ignores_empty_audio(self):
        conn = self.make_connection()
        cache_voiceprint_audio(conn, b"frame", audio_have_voice=True)
        cache_voiceprint_audio(conn, b"", audio_have_voice=False)

        clear_voiceprint_audio(conn)
        cache_voiceprint_audio(conn, b"next", audio_have_voice=False)

        self.assertEqual(conn.asr_audio_for_voiceprint, [b"next"])
        self.assertFalse(conn._asr_voiceprint_capture_active)


if __name__ == "__main__":
    unittest.main()
