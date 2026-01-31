import uuid
from typing import Dict, Any

from core.handle.sendAudioHandle import send_stt_message
from core.handle.textMessageHandler import TextMessageHandler
from core.handle.textMessageType import TextMessageType
from core.providers.tts.dto.dto import ContentType, TTSMessageDTO, SentenceType
from core.utils.dialogue import Message

TAG = __name__


class DirectTtsMessageHandler(TextMessageHandler):
    """直接TTS消息处理器 - 不经过LLM，直接播报文本"""

    @property
    def message_type(self) -> TextMessageType:
        return TextMessageType.DIRECT_TTS

    async def handle(self, conn, msg_json: Dict[str, Any]) -> None:
        """处理直接TTS请求"""
        text = msg_json.get("text", "")
        if not text:
            conn.logger.bind(tag=TAG).warning("收到空的直接TTS请求")
            return

        conn.logger.bind(tag=TAG).info(f"直接TTS播报: {text}")

        # 生成新的sentence_id
        conn.sentence_id = str(uuid.uuid4().hex)

        # 发送STT消息（显示文本）
        await send_stt_message(conn, text)
        conn.client_abort = False

        # 直接发送到TTS队列，不经过LLM
        def process_direct_tts():
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=SentenceType.FIRST,
                    content_type=ContentType.ACTION,
                )
            )
            conn.tts.tts_one_sentence(conn, ContentType.TEXT, content_detail=text)
            conn.tts.tts_text_queue.put(
                TTSMessageDTO(
                    sentence_id=conn.sentence_id,
                    sentence_type=SentenceType.LAST,
                    content_type=ContentType.ACTION,
                )
            )
            # 记录到对话历史（可选，标记为系统消息）
            conn.dialogue.put(Message(role="assistant", content=f"[直接播报] {text}"))

        conn.executor.submit(process_direct_tts)
