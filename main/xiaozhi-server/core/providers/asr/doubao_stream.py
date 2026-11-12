"""豆包流式 ASR 的音频协议压缩与预录发送。

@author pony
@date 2026-10-22
@version v1.2.0
@last_modified 2026-11-12
@changelog
    - v1.2.0 (2026-11-12): 结果交付后立即释放连接引用，修复上一句后约 10 秒内语音被丢弃；旧接收任务按会话序号收尾
    - v1.1.1 (2026-10-22): 清理无格式变量的日志字符串
    - v1.1.0 (2026-10-22): 修复短指令收尾、音频快照竞争和重复停止请求
    - v1.0.0 (2026-10-22): 快速 gzip、握手超时、预录去重与有界空闲音频缓存
"""

import json
import gzip
import uuid
import asyncio
import websockets
import opuslib_next
from core.providers.asr.base import ASRProviderBase
from config.logger import setup_logging
from core.providers.asr.dto.dto import InterfaceType
from core.providers.asr.audio_buffer import cache_voiceprint_audio, clear_voiceprint_audio

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__()
        self.interface_type = InterfaceType.STREAM
        self.config = config
        self.text = ""
        self.decoder = opuslib_next.Decoder(16000, 1)
        self.asr_ws = None
        self.forward_task = None
        self.is_processing = False  # 添加处理状态标志
        self._stop_sent = False
        # 会话序号：旧接收任务据此判断是否已有新会话，避免收尾时复位新会话的状态
        self._session_seq = 0
        self.connect_timeout = float(config.get("connect_timeout", 5))
        self.ready_timeout = float(config.get("ready_timeout", 5))
        # 60ms 小包采用快速 gzip；协议仍为 gzip，可按带宽预算调整为 1–9。
        self.gzip_compresslevel = int(config.get("gzip_compresslevel", 1))
        if not 1 <= self.gzip_compresslevel <= 9:
            raise ValueError("gzip_compresslevel 必须在 1–9 之间")

        # 配置参数
        self.appid = str(config.get("appid"))
        self.cluster = config.get("cluster")
        self.access_token = config.get("access_token")
        self.boosting_table_name = config.get("boosting_table_name", "")
        self.correct_table_name = config.get("correct_table_name", "")
        self.output_dir = config.get("output_dir", "tmp/")
        self.delete_audio_file = delete_audio_file

        # 火山引擎ASR配置
        enable_multilingual = config.get("enable_multilingual", False)
        self.enable_multilingual = False if str(enable_multilingual).lower() == 'false' else True
        if self.enable_multilingual:
            self.ws_url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"
        else:
            self.ws_url = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
        self.uid = config.get("uid", "streaming_asr_service")
        self.workflow = config.get(
            "workflow", "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate"
        )
        self.result_type = config.get("result_type", "single")
        self.format = config.get("format", "pcm")
        self.codec = config.get("codec", "pcm")
        self.rate = config.get("sample_rate", 16000)
        # language参数仅在多语种模式(bigmodel_nostream)下有效
        self.language = config.get("language") if self.enable_multilingual else None
        self.bits = config.get("bits", 16)
        self.channel = config.get("channel", 1)
        self.auth_method = config.get("auth_method", "token")
        self.secret = config.get("secret", "access_secret")
        end_window_size = config.get("end_window_size")
        self.end_window_size = int(end_window_size) if end_window_size else 200

    async def open_audio_channels(self, conn):
        await super().open_audio_channels(conn)

    async def receive_audio(self, conn, audio, audio_have_voice):
        conn.asr_audio.append(audio)
        conn.asr_audio = conn.asr_audio[-10:]
        cache_voiceprint_audio(conn, audio, audio_have_voice)

        # 当没有音频数据时处理完整语音片段
        if (
            conn.client_listen_mode != "manual"
            and not audio
            and getattr(conn, "_asr_voiceprint_capture_active", False)
        ):
            await self.handle_voice_stop(conn, conn.asr_audio_for_voiceprint)
            clear_voiceprint_audio(conn)

        # 如果本次有声音，且之前没有建立连接
        started_this_packet = False
        current_audio_replayed = False
        if audio_have_voice and self.asr_ws is None and not self.is_processing:
            try:
                self.is_processing = True
                self._stop_sent = False
                self._session_seq += 1
                started_this_packet = True
                # 建立新的WebSocket连接
                headers = self.token_auth() if self.auth_method == "token" else None
                logger.bind(tag=TAG).debug("正在连接豆包 ASR 服务")

                self.asr_ws = await websockets.connect(
                    self.ws_url,
                    additional_headers=headers,
                    # 音频已有协议层 gzip，再做 WebSocket deflate 会重复压缩。
                    compression=self.config.get("websocket_compression"),
                    open_timeout=self.connect_timeout,
                    max_size=1000000000,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=10,
                )

                # 发送初始化请求
                request_params = self.construct_request(str(uuid.uuid4()))
                try:
                    payload_bytes = str.encode(json.dumps(request_params))
                    payload_bytes = gzip.compress(payload_bytes, compresslevel=self.gzip_compresslevel)
                    full_client_request = self.generate_header()
                    full_client_request.extend((len(payload_bytes)).to_bytes(4, "big"))
                    full_client_request.extend(payload_bytes)

                    logger.bind(tag=TAG).debug("发送豆包 ASR 初始化请求")
                    await self.asr_ws.send(full_client_request)

                    # 等待初始化响应
                    init_res = await asyncio.wait_for(self.asr_ws.recv(), timeout=self.ready_timeout)
                    result = self.parse_response(init_res)
                    logger.bind(tag=TAG).info(f"收到初始化响应: {result}")

                    # 检查初始化响应
                    if "code" in result and result["code"] != 1000:
                        error_msg = f"ASR服务初始化失败: {result.get('payload_msg', {}).get('error', '未知错误')}"
                        logger.bind(tag=TAG).error(error_msg)
                        raise Exception(error_msg)

                except Exception as e:
                    logger.bind(tag=TAG).error(f"发送初始化请求失败: {str(e)}")
                    if hasattr(e, "__cause__") and e.__cause__:
                        logger.bind(tag=TAG).error(f"错误原因: {str(e.__cause__)}")
                    raise e

                # 启动接收ASR结果的异步任务
                self.forward_task = asyncio.create_task(self._forward_asr_results(conn))

                # 发送缓存的音频数据
                if conn.asr_audio and len(conn.asr_audio) > 0:
                    cached_audio_frames = conn.asr_audio[-10:]
                    for frame_index, cached_audio in enumerate(cached_audio_frames):
                        try:
                            pcm_frame = self.decoder.decode(cached_audio, 960)
                            payload = gzip.compress(pcm_frame, compresslevel=self.gzip_compresslevel)
                            audio_request = bytearray(
                                self.generate_audio_default_header()
                            )
                            audio_request.extend(len(payload).to_bytes(4, "big"))
                            audio_request.extend(payload)
                            await self.asr_ws.send(audio_request)
                            if frame_index == len(cached_audio_frames) - 1:
                                current_audio_replayed = True
                        except Exception as e:
                            logger.bind(tag=TAG).info(
                                f"发送缓存音频数据时发生错误: {e}"
                            )

            except Exception as e:
                logger.bind(tag=TAG).error(f"建立ASR连接失败: {str(e)}")
                if hasattr(e, "__cause__") and e.__cause__:
                    logger.bind(tag=TAG).error(f"错误原因: {str(e.__cause__)}")
                if self.asr_ws:
                    await self.asr_ws.close()
                    self.asr_ws = None
                self.is_processing = False
                return

        # 发送当前音频数据
        if self.asr_ws and self.is_processing and not self._stop_sent and not (
            started_this_packet and current_audio_replayed
        ):
            try:
                pcm_frame = self.decoder.decode(audio, 960)
                payload = gzip.compress(pcm_frame, compresslevel=self.gzip_compresslevel)
                audio_request = bytearray(self.generate_audio_default_header())
                audio_request.extend(len(payload).to_bytes(4, "big"))
                audio_request.extend(payload)
                await self.asr_ws.send(audio_request)
            except Exception as e:
                logger.bind(tag=TAG).info(f"发送音频数据时发生错误: {e}")

    async def _forward_asr_results(self, conn):
        # 只操作本会话的连接：结果交付后 self.asr_ws 会被立即释放，新会话可能已经开始。
        ws = self.asr_ws
        session = self._session_seq
        try:
            while ws is not None and self.asr_ws is ws and not conn.stop_event.is_set():
                try:
                    response = await ws.recv()
                    # recv 期间音频消费者仍可能追加数据，必须在结果到达后复制快照。
                    audio_data = list(getattr(conn, "asr_audio_for_voiceprint", []))
                    result = self.parse_response(response)
                    logger.bind(tag=TAG).debug(f"收到ASR结果: {result}")

                    if "payload_msg" in result:
                        payload = result["payload_msg"]
                        # 检查是否是错误码1013（无有效语音）
                        if "code" in payload and payload["code"] == 1013:
                            # 静默处理，不记录错误日志
                            continue

                        if "result" in payload:
                            utterances = payload["result"].get("utterances", [])
                            utterance_finished = False
                            # 检查duration和空文本的情况
                            if (
                                not self.enable_multilingual # 注意：多语种模式不返回中间结果，需要等待最终结果
                                and payload.get("audio_info", {}).get("duration", 0) > 2000
                                and not utterances
                                and not payload["result"].get("text")
                                and conn.client_listen_mode != "manual"
                            ):
                                logger.bind(tag=TAG).error("识别文本：空")
                                self.text = ""
                                await conn.reset_vad_states()
                                if audio_data:
                                    await self.handle_voice_stop(conn, audio_data)
                                break

                            # 专门处理没有文本的识别结果（手动模式下可能已经识别完成但是没松按键）
                            elif not payload["result"].get("text") and not utterances:
                                # 多语种模式会持续返回空文本，直到最后返回完整结果，所以需要排除
                                if self.enable_multilingual:
                                    continue

                                if conn.client_listen_mode == "manual" and conn.client_voice_stop and len(audio_data) > 0:
                                    logger.bind(tag=TAG).debug("消息结束收到停止信号，触发处理")
                                    await self.handle_voice_stop(conn, audio_data)
                                    # 清理音频缓存
                                    conn.asr_audio.clear()
                                    await conn.reset_vad_states()
                                    break

                            for utterance in utterances:
                                if utterance.get("definite", False):
                                    current_text = utterance["text"]
                                    logger.bind(tag=TAG).info(
                                        f"识别到文本: {current_text}"
                                    )

                                    # 手动模式下累积识别结果
                                    if conn.client_listen_mode == "manual":
                                        if self.text:
                                            self.text += current_text
                                        else:
                                            self.text = current_text

                                        # 在接收消息中途时收到停止信号
                                        if conn.client_voice_stop and len(audio_data) > 0:
                                            logger.bind(tag=TAG).debug("消息中途收到停止信号，触发处理")
                                            await self.handle_voice_stop(conn, audio_data)
                                            utterance_finished = True
                                            # 清理音频缓存
                                            conn.asr_audio.clear()
                                            await conn.reset_vad_states()
                                        break
                                    else:
                                        # 自动模式下直接覆盖
                                        self.text = current_text
                                        await conn.reset_vad_states()
                                        if audio_data:
                                            await self.handle_voice_stop(conn, audio_data)
                                            utterance_finished = True
                                    break
                            if utterance_finished:
                                break
                        elif "error" in payload:
                            error_msg = payload.get("error", "未知错误")
                            logger.bind(tag=TAG).error(f"ASR服务返回错误: {error_msg}")
                            break

                except websockets.ConnectionClosed:
                    logger.bind(tag=TAG).info("ASR服务连接已关闭")
                    if self._session_seq == session:
                        self.is_processing = False
                    break
                except Exception as e:
                    logger.bind(tag=TAG).error(f"处理ASR结果时发生错误: {str(e)}")
                    if hasattr(e, "__cause__") and e.__cause__:
                        logger.bind(tag=TAG).error(f"错误原因: {str(e.__cause__)}")
                    if self._session_seq == session:
                        self.is_processing = False
                    break

        except Exception as e:
            logger.bind(tag=TAG).error(f"ASR结果转发任务发生错误: {str(e)}")
            if hasattr(e, "__cause__") and e.__cause__:
                logger.bind(tag=TAG).error(f"错误原因: {str(e.__cause__)}")
        finally:
            # 先在任何 await 之前同步收尾；已有新会话时不复位其状态、不清其缓存。
            if self.asr_ws is ws:
                self.asr_ws = None
            if self._session_seq == session:
                self.is_processing = False
                if conn:
                    if hasattr(conn, 'asr_audio_for_voiceprint'):
                        clear_voiceprint_audio(conn)
                    if hasattr(conn, 'asr_audio'):
                        conn.asr_audio = []
            if ws is not None:
                await ws.close()

    def stop_ws_connection(self):
        ws = self.asr_ws
        if ws:
            # 结果已交付：立即释放引用，下一句可马上建立新会话；关闭握手（最长
            # close_timeout）在后台完成。旧接收任务只操作自己的连接，不会误关新连接。
            self.asr_ws = None
            asyncio.create_task(ws.close())
        self.is_processing = False

    async def _send_stop_request(self):
        """发送最后一个音频帧以通知服务器结束"""
        if self.asr_ws and not self._stop_sent:
            try:
                # 先置位，避免 stop 与 stop 并发时重复发送，且阻止后续音频继续写入。
                self._stop_sent = True
                self.is_processing = False
                # 发送结束标记的音频帧（gzip压缩的空数据）
                empty_payload = gzip.compress(b"", compresslevel=self.gzip_compresslevel)
                last_audio_request = bytearray(self.generate_last_audio_default_header())
                last_audio_request.extend(len(empty_payload).to_bytes(4, "big"))
                last_audio_request.extend(empty_payload)
                await self.asr_ws.send(last_audio_request)
                logger.bind(tag=TAG).debug("已发送结束音频帧")
            except Exception as e:
                self._stop_sent = False
                logger.bind(tag=TAG).debug(f"发送结束音频帧时出错: {e}")

    def construct_request(self, reqid):
        req = {
            "app": {
                "appid": self.appid,
                "cluster": self.cluster,
                "token": self.access_token,
            },
            "user": {"uid": self.uid},
            "request": {
                "reqid": reqid,
                "workflow": self.workflow,
                "show_utterances": True,
                "result_type": self.result_type,
                "sequence": 1,
                "boosting_table_name": self.boosting_table_name,
                "correct_table_name": self.correct_table_name,
                "end_window_size": self.end_window_size,
            },
            "audio": {
                "format": self.format,
                "codec": self.codec,
                "rate": self.rate,
                "bits": self.bits,
                "channel": self.channel,
                "sample_rate": self.rate,
            },
        }

        # language参数仅在多语种模式下添加
        if self.enable_multilingual and self.language:
            req["audio"]["language"] = self.language

        logger.bind(tag=TAG).debug(
            f"构造请求参数: {json.dumps(req, ensure_ascii=False)}"
        )
        return req

    def token_auth(self):
        return {
            "X-Api-App-Key": self.appid,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": "volc.bigasr.sauc.duration",
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

    def generate_header(
        self,
        version=0x01,
        message_type=0x01,
        message_type_specific_flags=0x00,
        serial_method=0x01,
        compression_type=0x01,
        reserved_data=0x00,
        extension_header: bytes = b"",
    ):
        header = bytearray()
        header_size = int(len(extension_header) / 4) + 1
        header.append((version << 4) | header_size)
        header.append((message_type << 4) | message_type_specific_flags)
        header.append((serial_method << 4) | compression_type)
        header.append(reserved_data)
        header.extend(extension_header)
        return header

    def generate_audio_default_header(self):
        return self.generate_header(
            version=0x01,
            message_type=0x02,
            message_type_specific_flags=0x00,
            serial_method=0x01,
            compression_type=0x01,
        )

    def generate_last_audio_default_header(self):
        return self.generate_header(
            version=0x01,
            message_type=0x02,
            message_type_specific_flags=0x02,
            serial_method=0x01,
            compression_type=0x01,
        )

    def parse_response(self, res: bytes) -> dict:
        try:
            # 检查响应长度
            if len(res) < 4:
                logger.bind(tag=TAG).error(f"响应数据长度不足: {len(res)}")
                return {"error": "响应数据长度不足"}

            # 获取消息头
            header = res[:4]
            message_type = header[1] >> 4

            # 如果是错误响应
            if message_type == 0x0F:  # SERVER_ERROR_RESPONSE
                code = int.from_bytes(res[4:8], "big", signed=False)
                msg_length = int.from_bytes(res[8:12], "big", signed=False)
                error_msg = json.loads(res[12:].decode("utf-8"))
                return {
                    "code": code,
                    "msg_length": msg_length,
                    "payload_msg": error_msg,
                }

            # 获取JSON数据（跳过12字节头部）
            try:
                json_data = res[12:].decode("utf-8")
                result = json.loads(json_data)
                logger.bind(tag=TAG).debug(f"成功解析JSON响应: {result}")
                return {"payload_msg": result}
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.bind(tag=TAG).error(f"JSON解析失败: {str(e)}")
                logger.bind(tag=TAG).error(f"原始数据: {res}")
                raise

        except Exception as e:
            logger.bind(tag=TAG).error(f"解析响应失败: {str(e)}")
            logger.bind(tag=TAG).error(f"原始响应数据: {res.hex()}")
            raise

    async def speech_to_text(self, opus_data, session_id, audio_format):
        result = self.text
        self.text = ""  # 清空text
        return result, None

    async def close(self):
        """资源清理方法"""
        if self.asr_ws:
            await self.asr_ws.close()
            self.asr_ws = None
        if self.forward_task:
            self.forward_task.cancel()
            try:
                await self.forward_task
            except asyncio.CancelledError:
                pass
            self.forward_task = None
        self.is_processing = False
        self._stop_sent = False
        
        # 显式释放decoder资源
        if hasattr(self, 'decoder') and self.decoder is not None:
            try:
                del self.decoder
                self.decoder = None
                logger.bind(tag=TAG).debug("Doubao decoder resources released")
            except Exception as e:
                logger.bind(tag=TAG).debug(f"释放Doubao decoder资源时出错: {e}")

        # 清理所有连接的音频缓冲区
        if hasattr(self, '_connections'):
            for conn in self._connections.values():
                if hasattr(conn, 'asr_audio_for_voiceprint'):
                    conn.asr_audio_for_voiceprint = []
                if hasattr(conn, 'asr_audio'):
                    conn.asr_audio = []
