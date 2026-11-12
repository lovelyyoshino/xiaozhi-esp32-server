"""Silero VAD 的 onnxruntime 实现：与 silero.py 接口和判定逻辑相同，不依赖 torch。

@author pony
@date 2026-11-12
@version v1.1.0
@last_modified 2026-11-12
@changelog
    - v1.1.0 (2026-11-12): 增加 is_vad_async：专用小线程池（默认 2 线程）+ 有界并发，避免挤占默认线程池、
      避免大量可运行线程加剧 GIL 争用拖慢事件循环（压测中无界 to_thread 导致 ping 最大 4.5 s）。
    - v1.0.0 (2026-11-12): 新增。每连接状态为 numpy 数组，onnxruntime 会话线程安全，无需全局模型锁；
      进程常驻内存约减少 350 MB，单路推理 CPU 约减半，多核并发吞吐约 3.6 倍（本地基准）。

配置示例（VAD 段，type 设为 silero_onnx）：
  SileroVAD:
    type: silero_onnx
    model_dir: models/snakers4_silero-vad
    threshold: 0.5
    threshold_low: 0.3
    min_silence_duration_ms: 700
    num_threads: 1
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import onnxruntime as ort
import opuslib_next

from config.logger import setup_logging
from core.providers.vad.base import VADProviderBase

TAG = __name__
logger = setup_logging()

_WINDOW = 512          # 16 kHz 下 32 ms，与 silero_vad 流式窗口一致
_CONTEXT = 64          # 官方 OnnxWrapper：每个窗口前拼接上一窗口末尾 64 个采样
_SR = np.array(16000, dtype=np.int64)


class VADProvider(VADProviderBase):
    def __init__(self, config):
        logger.bind(tag=TAG).info("SileroVAD(onnx)", config)
        model_path = config.get("model_path") or os.path.join(
            config["model_dir"], "src", "silero_vad", "data", "silero_vad.onnx"
        )
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(config.get("num_threads") or 1))
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self._warm_up()

        threshold = config.get("threshold", "0.5")
        threshold_low = config.get("threshold_low", "0.2")
        min_silence_duration_ms = config.get("min_silence_duration_ms", "1000")
        self.vad_threshold = float(threshold) if threshold else 0.5
        self.vad_threshold_low = float(threshold_low) if threshold_low else 0.2
        self.silence_threshold_ms = int(min_silence_duration_ms) if min_silence_duration_ms else 1000
        self.frame_window_threshold = 3
        # 单包推理约 0.2 ms，2 个线程即可承载数百路；线程少 = GIL 竞争者少 = 事件循环延迟低
        workers = max(1, int(config.get("executor_workers") or 2))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vad-onnx")
        self._slots = asyncio.Semaphore(max(1, int(config.get("max_pending_inferences") or 4)))

    def _warm_up(self):
        state = self._new_state()
        for _ in range(3):
            self._prob(state, np.zeros(_WINDOW, dtype=np.float32))

    @staticmethod
    def _new_state():
        return {"rnn": np.zeros((2, 1, 128), dtype=np.float32), "ctx": np.zeros(_CONTEXT, dtype=np.float32)}

    def _prob(self, state, chunk):
        x = np.concatenate((state["ctx"], chunk))[np.newaxis, :]
        out, state["rnn"] = self.session.run(None, {"input": x, "state": state["rnn"], "sr": _SR})
        state["ctx"] = chunk[-_CONTEXT:]
        return float(out[0][0])

    @staticmethod
    def _connection_decoder(conn):
        decoder = getattr(conn, "_vad_opus_decoder", None)
        if decoder is None:
            decoder = opuslib_next.Decoder(16000, 1)
            conn._vad_opus_decoder = decoder
        return decoder

    def reset_connection(self, conn):
        """新一句话或手动停止时清除该连接的模型状态。"""
        with getattr(conn, "_vad_state_lock", nullcontext()):
            if hasattr(conn, "_vad_model_state"):
                del conn._vad_model_state

    def release_connection(self, conn):
        with getattr(conn, "_vad_state_lock", nullcontext()):
            self.reset_connection(conn)
            if hasattr(conn, "_vad_opus_decoder"):
                del conn._vad_opus_decoder

    async def is_vad_async(self, conn, opus_packet):
        """在专用线程池执行；同一连接的包由 asr_audio_priority_task 顺序 await，因此每连接状态不会并发访问。"""
        if conn.client_listen_mode == "manual":
            return True
        async with self._slots:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self.is_vad, conn, opus_packet)

    def is_vad(self, conn, opus_packet):
        if conn.client_listen_mode == "manual":
            return True
        try:
            with getattr(conn, "_vad_state_lock", nullcontext()):
                pcm_frame = self._connection_decoder(conn).decode(opus_packet, 960)
                conn.client_audio_buffer.extend(pcm_frame)
                chunks = []
                while len(conn.client_audio_buffer) >= _WINDOW * 2:
                    chunk = conn.client_audio_buffer[: _WINDOW * 2]
                    del conn.client_audio_buffer[: _WINDOW * 2]
                    chunks.append(np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0)
                if not chunks:
                    return False

                state = getattr(conn, "_vad_model_state", None)
                if state is None:
                    state = self._new_state()
                    conn._vad_model_state = state

                client_have_voice = False
                for chunk in chunks:
                    speech_prob = self._prob(state, chunk)
                    if speech_prob >= self.vad_threshold:
                        is_voice = True
                    elif speech_prob <= self.vad_threshold_low:
                        is_voice = False
                    else:
                        is_voice = conn.last_is_voice
                    conn.last_is_voice = is_voice
                    conn.client_voice_window.append(is_voice)
                    client_have_voice = conn.client_voice_window.count(True) >= self.frame_window_threshold
                    if conn.client_have_voice and not client_have_voice:
                        stop_duration = time.time() * 1000 - conn.last_activity_time
                        if stop_duration >= self.silence_threshold_ms:
                            conn.client_voice_stop = True
                    if client_have_voice:
                        conn.client_have_voice = True
                        conn.last_activity_time = time.time() * 1000
                return client_have_voice
        except opuslib_next.OpusError as e:
            logger.bind(tag=TAG).info(f"解码错误: {e}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error processing audio packet: {e}")
        return False
