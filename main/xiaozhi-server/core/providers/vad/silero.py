"""在共享 Silero 模型上隔离设备流状态并限制 CPU 推理并发。

@author pony
@date 2026-10-22
@version v1.0.0
@last_modified 2026-10-22
@changelog
    - v1.0.0 (2026-10-22): 单线程预热、连接专属解码与状态、带取消保护的异步推理
"""

import asyncio
import copy
import threading
import time
from contextlib import nullcontext
import numpy as np
import torch
import opuslib_next
from config.logger import setup_logging
from core.providers.vad.base import VADProviderBase

TAG = __name__
logger = setup_logging()


class VADProvider(VADProviderBase):
    def __init__(self, config):
        logger.bind(tag=TAG).info("SileroVAD", config)
        # 小型流式模型默认单线程，避免每个音频任务再启动一组计算线程抢占 CPU。
        # PyTorch 的线程数是进程级设置；同进程还有本地大模型时可显式调整。
        torch.set_num_threads(max(1, int(config.get("num_threads") or 1)))
        self.model, _ = torch.hub.load(
            repo_or_dir=config["model_dir"],
            source="local",
            model="silero_vad",
            force_reload=False,
        )
        # Silero 的流式模型会在 model 对象内部保存 state/context。模型可以共享，
        # 但这些状态必须按连接保存和恢复，否则多设备交错发包时会互相污染。
        self._model_lock = threading.RLock()
        # 控制音频线程池中等待共享模型锁的任务数，给同一线程池的其他工作留余量。
        self._inference_slots = asyncio.Semaphore(
            max(1, int(config.get("max_pending_inferences") or 4))
        )
        self._initial_model_state = self._snapshot_model_state()
        self._warm_up_model()

        # 处理空字符串的情况
        threshold = config.get("threshold", "0.5")
        threshold_low = config.get("threshold_low", "0.2")
        min_silence_duration_ms = config.get("min_silence_duration_ms", "1000")

        self.vad_threshold = float(threshold) if threshold else 0.5
        self.vad_threshold_low = float(threshold_low) if threshold_low else 0.2

        self.silence_threshold_ms = (
            int(min_silence_duration_ms) if min_silence_duration_ms else 1000
        )

        # 至少要多少帧才算有语音
        self.frame_window_threshold = 3

    @staticmethod
    def _connection_decoder(conn):
        """返回连接专属的 Opus decoder，避免不同设备共享解码历史。"""
        decoder = getattr(conn, "_vad_opus_decoder", None)
        if decoder is None:
            decoder = opuslib_next.Decoder(16000, 1)
            conn._vad_opus_decoder = decoder
        return decoder

    def _snapshot_model_state(self):
        """复制 Silero 流式状态；兼容 JIT 和 ONNX wrapper 两种实现。"""
        snapshot = {}
        for name in ("_state", "_context"):
            value = getattr(self.model, name, None)
            if torch.is_tensor(value):
                snapshot[name] = value.detach().clone()
            elif value is not None:
                snapshot[name] = copy.deepcopy(value)
        for name in ("_last_sr", "_last_batch_size"):
            if hasattr(self.model, name):
                snapshot[name] = getattr(self.model, name)
        return snapshot

    def _restore_model_state(self, snapshot):
        # 完整快照可直接恢复，逐包 reset 会反复分配最终又被覆盖的状态张量。
        if len(snapshot) < 4 and hasattr(self.model, "reset_states"):
            self.model.reset_states()
        for name, value in snapshot.items():
            if torch.is_tensor(value):
                value = value.clone()
            setattr(self.model, name, value)

    def _warm_up_model(self):
        """启动时预跑数帧模型，避免首个设备承担TorchScript初始化延迟。"""
        try:
            with self._model_lock, torch.no_grad():
                try:
                    self._restore_model_state(self._initial_model_state)
                    warmup_frame = torch.zeros(512)
                    for _ in range(3):
                        self.model(warmup_frame, 16000)
                finally:
                    self._restore_model_state(self._initial_model_state)
        except Exception as e:
            # 预热失败不影响服务启动，真正推理时仍会走原有异常处理。
            logger.bind(tag=TAG).warning(f"Silero VAD预热失败，将在首次音频到达时重试: {e}")

    def reset_connection(self, conn):
        """清除连接的 VAD 模型状态，供新一句话或手动停止后调用。"""
        connection_lock = getattr(conn, "_vad_state_lock", nullcontext())
        with connection_lock:
            with self._model_lock:
                if hasattr(conn, "_vad_model_state"):
                    del conn._vad_model_state
                self._restore_model_state(self._initial_model_state)

    def release_connection(self, conn):
        """释放连接专属的 VAD 资源。"""
        connection_lock = getattr(conn, "_vad_state_lock", nullcontext())
        with connection_lock:
            self.reset_connection(conn)
            decoder = getattr(conn, "_vad_opus_decoder", None)
            if decoder is not None:
                try:
                    del conn._vad_opus_decoder
                except Exception:
                    pass

    def is_vad(self, conn, opus_packet):
        # 手动模式：直接返回True，不进行实时VAD检测，所有音频都缓存
        if conn.client_listen_mode == "manual":
            return True

        try:
            connection_lock = getattr(conn, "_vad_state_lock", nullcontext())
            with connection_lock:
                # 解码器状态按连接隔离，因此在共享模型锁外解码，允许不同连接并行完成解码。
                decoder = self._connection_decoder(conn)
                pcm_frame = decoder.decode(opus_packet, 960)
                conn.client_audio_buffer.extend(pcm_frame)

                audio_tensors = []
                while len(conn.client_audio_buffer) >= 512 * 2:
                    chunk = conn.client_audio_buffer[: 512 * 2]
                    del conn.client_audio_buffer[: 512 * 2]
                    audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    audio_tensors.append(torch.from_numpy(audio_float32))

                # 不足一帧时保留连接缓冲区即可，无需占用共享模型锁切换状态。
                if not audio_tensors:
                    return False

                with self._model_lock:
                    # 在共享模型上执行推理前恢复当前连接的历史，执行后保存新历史。
                    connection_state = getattr(
                        conn, "_vad_model_state", self._initial_model_state
                    )
                    self._restore_model_state(connection_state)

                    client_have_voice = False
                    with torch.no_grad():
                        for audio_tensor in audio_tensors:
                            # 检测语音活动
                            speech_prob = self.model(audio_tensor, 16000).item()

                            # 双阈值判断
                            if speech_prob >= self.vad_threshold:
                                is_voice = True
                            elif speech_prob <= self.vad_threshold_low:
                                is_voice = False
                            else:
                                is_voice = conn.last_is_voice

                            # 声音没低于最低值则延续前一个状态，判断为有声音
                            conn.last_is_voice = is_voice

                            # 更新滑动窗口
                            conn.client_voice_window.append(is_voice)
                            client_have_voice = (
                                conn.client_voice_window.count(True) >= self.frame_window_threshold
                            )

                            # 如果之前有声音，但本次没有声音，且与上次有声音的时间差已经超过了静默阈值，则认为已经说完一句话
                            if conn.client_have_voice and not client_have_voice:
                                stop_duration = time.time() * 1000 - conn.last_activity_time
                                if stop_duration >= self.silence_threshold_ms:
                                    conn.client_voice_stop = True
                            if client_have_voice:
                                conn.client_have_voice = True
                                conn.last_activity_time = time.time() * 1000

                    conn._vad_model_state = self._snapshot_model_state()
                    return client_have_voice
        except opuslib_next.OpusError as e:
            logger.bind(tag=TAG).info(f"解码错误: {e}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error processing audio packet: {e}")
        return False

    async def is_vad_async(self, conn, opus_packet):
        """在共享线程池计算 VAD，并限制同时等待模型锁的音频任务数。"""
        if conn.client_listen_mode == "manual":
            return True

        async with self._inference_slots:
            inference_task = asyncio.create_task(
                asyncio.to_thread(self.is_vad, conn, opus_packet)
            )
            try:
                return await asyncio.shield(inference_task)
            except asyncio.CancelledError:
                # 取消协程不能停止已运行的线程；等待结束后再归还并发名额。
                try:
                    await asyncio.shield(inference_task)
                except Exception:
                    pass
                raise
