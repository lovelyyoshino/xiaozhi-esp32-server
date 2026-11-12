import json

from core.utils.wake_word_gate import get_wake_word_gate

TAG = __name__


async def handleAbortMessage(conn):
    conn.logger.bind(tag=TAG).info("Abort message received")
    # 设置成打断状态，会自动打断llm、tts任务
    conn.client_abort = True
    conn.clear_queues()
    # 打断客户端说话状态
    await conn.websocket.send(
        json.dumps({"type": "tts", "state": "stop", "session_id": conn.session_id})
    )
    conn.clearSpeakStatus()
    # 被打断的回复没说完，即使最后一句是问句也不放行下一句
    gate = get_wake_word_gate(conn)
    if gate is not None:
        gate.on_speech_end(completed=False)
    conn.logger.bind(tag=TAG).info("Abort message received-end")
