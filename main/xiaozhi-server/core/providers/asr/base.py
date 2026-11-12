"""ASR 有序音频消费与识别完成后的可选声纹处理。

@author pony
@date 2026-10-22
@version v1.2.0
@last_modified 2026-11-12
@changelog
    - v1.2.0 (2026-11-12): 识别结果标记为语音来源，交由唤醒词门控检查
    - v1.1.0 (2026-10-22): 唤醒保护期间跳过回采包，避免无效 VAD 推理
    - v1.0.0 (2026-10-22): 使用异步消费任务与队列背压，跳过未启用声纹时的重复解码
"""

import os
import io
import wave
import uuid
import json
import time
import asyncio
import traceback
import opuslib_next
from abc import ABC, abstractmethod
from config.logger import setup_logging
from typing import Optional, Tuple, List
from core.handle.receiveAudioHandle import startToChat
from core.handle.reportHandle import enqueue_asr_report
from core.utils.util import remove_punctuation_and_length
from core.handle.receiveAudioHandle import handleAudioMessage

TAG = __name__
logger = setup_logging()


class ASRProviderBase(ABC):
    def __init__(self):
        pass

    # 打开音频通道
    async def open_audio_channels(self, conn):
        if conn.stop_event.is_set():
            return
        if conn.asr_audio_task is not None and not conn.asr_audio_task.done():
            return
        conn.asr_audio_task = asyncio.create_task(
            self.asr_audio_priority_task(conn),
            name=f"asr-audio-{getattr(conn, 'device_id', None) or id(conn)}",
        )

    # 有序处理ASR音频
    async def asr_audio_priority_task(self, conn):
        while not conn.stop_event.is_set():
            message = await conn.asr_audio_queue.get()
            try:
                if getattr(conn, "just_woken_up", False):
                    continue
                # 每连接顺序 await，共享有限的 asyncio 默认线程池处理本地 VAD。
                if hasattr(conn.vad, "is_vad_async"):
                    have_voice = await conn.vad.is_vad_async(conn, message)
                else:
                    have_voice = await asyncio.to_thread(
                        conn.vad.is_vad, conn, message
                    )
                await handleAudioMessage(conn, message, have_voice)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"处理ASR文本失败: {str(e)}, 类型: {type(e).__name__}, 堆栈: {traceback.format_exc()}"
                )
                continue
            finally:
                # stop 消息等待此前音频消费完毕，取消或单包失败也必须解除等待。
                conn.asr_audio_queue.task_done()

    # 接收音频
    async def receive_audio(self, conn, audio, audio_have_voice):
        if conn.client_listen_mode == "manual":
            # 手动模式：缓存音频用于ASR识别
            conn.asr_audio.append(audio)
        else:
            # 自动/实时模式：使用VAD检测
            have_voice = audio_have_voice

            conn.asr_audio.append(audio)
            if not have_voice and not conn.client_have_voice:
                conn.asr_audio = conn.asr_audio[-10:]
                return

            # 自动模式下通过VAD检测到语音停止时触发识别
            if conn.client_voice_stop:
                asr_audio_task = conn.asr_audio.copy()
                conn.asr_audio.clear()
                await conn.reset_vad_states()

                if len(asr_audio_task) > 15:
                    await self.handle_voice_stop(conn, asr_audio_task)

    # 处理语音停止
    async def handle_voice_stop(self, conn, asr_audio_task: List[bytes]):
        """并行处理ASR和声纹识别"""
        try:
            total_start_time = time.monotonic()

            wav_data = None
            # 云 ASR 已经拿到识别结果；只有声纹识别才需要再次解码完整语句。
            if conn.voiceprint_provider:
                def prepare_voiceprint_audio():
                    pcm_data = (
                        asr_audio_task
                        if conn.audio_format == "pcm"
                        else self.decode_opus(asr_audio_task)
                    )
                    return self._pcm_to_wav(b"".join(pcm_data))

                wav_data = await asyncio.to_thread(prepare_voiceprint_audio)

            # 定义ASR任务
            asr_task = self.speech_to_text(asr_audio_task, conn.session_id, conn.audio_format)

            if conn.voiceprint_provider and wav_data:
                voiceprint_task = conn.voiceprint_provider.identify_speaker(wav_data, conn.session_id)
                # 并发等待两个结果
                asr_result, voiceprint_result = await asyncio.gather(
                    asr_task, voiceprint_task, return_exceptions=True
                )
            else:
                asr_result = await asr_task
                voiceprint_result = None

            # 记录识别结果 - 检查是否为异常
            if isinstance(asr_result, Exception):
                logger.bind(tag=TAG).error(f"ASR识别失败: {asr_result}")
                raw_text = ""
            else:
                raw_text, _ = asr_result

            if isinstance(voiceprint_result, Exception):
                logger.bind(tag=TAG).error(f"声纹识别失败: {voiceprint_result}")
                speaker_name = ""
            else:
                speaker_name = voiceprint_result

            # 判断 ASR 结果类型
            if isinstance(raw_text, dict):
                # FunASR 返回的 dict 格式
                if speaker_name:
                    raw_text["speaker"] = speaker_name

                # 记录识别结果
                if raw_text.get("language"):
                    logger.bind(tag=TAG).info(f"识别语言: {raw_text['language']}")
                if raw_text.get("emotion"):
                    logger.bind(tag=TAG).info(f"识别情绪: {raw_text['emotion']}")
                if raw_text.get("content"):
                    logger.bind(tag=TAG).info(f"识别文本: {raw_text['content']}")
                if speaker_name:
                    logger.bind(tag=TAG).info(f"识别说话人: {speaker_name}")

                # 转换为 JSON 字符串用于下游
                enhanced_text = json.dumps(raw_text, ensure_ascii=False)
                content_for_length_check = raw_text.get("content", "")
            else:
                # 其他 ASR 返回的纯文本
                if raw_text:
                    logger.bind(tag=TAG).info(f"识别文本: {raw_text}")
                if speaker_name:
                    logger.bind(tag=TAG).info(f"识别说话人: {speaker_name}")

                # 构建包含说话人信息的JSON字符串
                enhanced_text = self._build_enhanced_text(raw_text, speaker_name)
                content_for_length_check = raw_text

            # 性能监控
            total_time = time.monotonic() - total_start_time
            logger.bind(tag=TAG).debug(f"总处理耗时: {total_time:.3f}s")

            # 检查文本长度
            text_len, _ = remove_punctuation_and_length(content_for_length_check)
            self.stop_ws_connection()

            if text_len > 0:
                # 使用自定义模块进行上报
                await startToChat(conn, enhanced_text, from_asr=True)
                enqueue_asr_report(conn, enhanced_text, asr_audio_task)
                
        except Exception as e:
            logger.bind(tag=TAG).error(f"处理语音停止失败: {e}")
            import traceback
            logger.bind(tag=TAG).debug(f"异常详情: {traceback.format_exc()}")

    def _build_enhanced_text(self, text: str, speaker_name: Optional[str]) -> str:
        """构建包含说话人信息的文本（仅用于纯文本ASR）"""
        if speaker_name and speaker_name.strip():
            return json.dumps({
                "speaker": speaker_name,
                "content": text
            }, ensure_ascii=False)
        else:
            return text

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """将PCM数据转换为WAV格式"""
        if len(pcm_data) == 0:
            logger.bind(tag=TAG).warning("PCM数据为空，无法转换WAV")
            return b""
        
        # 确保数据长度是偶数（16位音频）
        if len(pcm_data) % 2 != 0:
            pcm_data = pcm_data[:-1]
        
        # 创建WAV文件头
        wav_buffer = io.BytesIO()
        try:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)      # 单声道
                wav_file.setsampwidth(2)      # 16位
                wav_file.setframerate(16000)  # 16kHz采样率
                wav_file.writeframes(pcm_data)
            
            wav_buffer.seek(0)
            wav_data = wav_buffer.read()
            
            return wav_data
        except Exception as e:
            logger.bind(tag=TAG).error(f"WAV转换失败: {e}")
            return b""

    def stop_ws_connection(self):
        pass

    def save_audio_to_file(self, pcm_data: List[bytes], session_id: str) -> str:
        """PCM数据保存为WAV文件"""
        module_name = __name__.split(".")[-1]
        file_name = f"asr_{module_name}_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(pcm_data))

        return file_path

    @abstractmethod
    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        """将语音数据转换为文本"""
        pass

    @staticmethod
    def decode_opus(opus_data: List[bytes]) -> List[bytes]:
        """将Opus音频数据解码为PCM数据"""
        decoder = None
        try:
            decoder = opuslib_next.Decoder(16000, 1)
            pcm_data = []
            buffer_size = 960  # 每次处理960个采样点 (60ms at 16kHz)
            
            for i, opus_packet in enumerate(opus_data):
                try:
                    if not opus_packet or len(opus_packet) == 0:
                        continue
                    
                    pcm_frame = decoder.decode(opus_packet, buffer_size)
                    if pcm_frame and len(pcm_frame) > 0:
                        pcm_data.append(pcm_frame)
                        
                except opuslib_next.OpusError as e:
                    logger.bind(tag=TAG).warning(f"Opus解码错误，跳过数据包 {i}: {e}")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"音频处理错误，数据包 {i}: {e}")
            
            return pcm_data
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"音频解码过程发生错误: {e}")
            return []
        finally:
            if decoder is not None:
                try:
                    del decoder
                except Exception as e:
                    logger.bind(tag=TAG).debug(f"释放decoder资源时出错: {e}")
