"""音频并发、停止顺序和 CPU 工作边界的回归验证。

@author pony
@date 2026-10-22
@version v1.6.0
@last_modified 2026-10-22
@changelog
    - v1.6.0 (2026-10-22): 覆盖短唤醒确认、异常恢复及音频满队列断连清理
    - v1.5.0 (2026-10-22): 覆盖本地繁忙提示的音频发送依赖与协议顺序
    - v1.4.0 (2026-10-22): 验证共享后台预算、断连任务归属和拥塞结束回复
    - v1.3.0 (2026-10-22): 覆盖豆包短指令、接收期间缓存替换及停止幂等
    - v1.2.0 (2026-10-22): 覆盖上报转换并发、取消与拥塞日志边界
    - v1.1.0 (2026-10-22): 覆盖上报名额、跨线程提交、断连取消和音频转换隔离
    - v1.0.0 (2026-10-22): 覆盖真实 VAD 模型状态隔离、取消和异步音频消费
"""

import asyncio
import gzip
import importlib.util
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest


SERVER_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime_loader(monkeypatch):
    """隔离云 SDK 和日志初始化；被测音频消费与模型代码使用真实实现。"""
    def load(relative_path, **dependencies):
        logger = Mock()
        logger.bind.return_value = logger
        defaults = {
            "config.logger": {"setup_logging": lambda: logger},
            "opuslib_next": {"Decoder": Mock(), "OpusError": RuntimeError},
        }
        for name, attributes in (defaults | dependencies).items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            monkeypatch.setitem(__import__("sys").modules, name, module)
        spec = importlib.util.spec_from_file_location(
            "_audio_runtime_test_" + Path(relative_path).stem,
            SERVER_ROOT / relative_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def asr_base(runtime_loader):
    return runtime_loader(
        "core/providers/asr/base.py",
        **{
            "core.handle.receiveAudioHandle": {
                "startToChat": AsyncMock(), "handleAudioMessage": AsyncMock()
            },
            "core.handle.reportHandle": {"enqueue_asr_report": Mock()},
            "core.utils.util": {
                "remove_punctuation_and_length": lambda text: (len(text), text)
            },
        },
    )


def make_provider(asr_base):
    class Provider(asr_base.ASRProviderBase):
        async def speech_to_text(self, *args):
            return "去电梯", None

    return Provider()


def make_connection(**overrides):
    fields = dict(
        client_listen_mode="auto", client_have_voice=False,
        client_voice_stop=False, last_is_voice=False, last_activity_time=0,
        client_voice_window=deque(maxlen=5), client_audio_buffer=bytearray(),
        _vad_state_lock=threading.RLock(), stop_event=threading.Event(),
        asr_audio_queue=asyncio.Queue(maxsize=16), asr_audio_task=None,
        asr_audio=[], voiceprint_provider=None, audio_format="opus",
        session_id="test-session",
    )
    return SimpleNamespace(**(fields | overrides))


def test_no_voiceprint_does_not_decode_entire_utterance(asr_base):
    provider = make_provider(asr_base)
    provider.decode_opus = Mock(side_effect=AssertionError("不应重复解码"))
    asyncio.run(provider.handle_voice_stop(make_connection(), [b"opus"] * 20))
    provider.decode_opus.assert_not_called()
    asr_base.startToChat.assert_awaited_once()
    asr_base.enqueue_asr_report.assert_called_once()


def test_voiceprint_decode_runs_outside_event_loop(asr_base):
    provider = make_provider(asr_base)
    decode_threads = []

    def decode(_):
        decode_threads.append(threading.get_ident())
        return [b"\x00\x00" * 960]

    provider.decode_opus = decode
    voiceprint = SimpleNamespace(identify_speaker=AsyncMock(return_value="主人"))
    conn = make_connection(voiceprint_provider=voiceprint)
    main_thread = threading.get_ident()
    asyncio.run(provider.handle_voice_stop(conn, [b"opus"]))
    assert decode_threads and decode_threads[0] != main_thread
    voiceprint.identify_speaker.assert_awaited_once()
    assert voiceprint.identify_speaker.call_args.args[0].startswith(b"RIFF")
    asr_base.startToChat.assert_awaited_once()


def test_ordered_consumer_completes_queue_on_failure_and_cancel(asr_base):
    async def scenario():
        conn = make_connection(vad=SimpleNamespace(is_vad_async=AsyncMock(return_value=True)))
        provider = make_provider(asr_base)
        received = []

        async def receive(_, frame, have_voice):
            received.append(frame)
            if frame == b"bad":
                raise ValueError("单包失败")

        asr_base.handleAudioMessage.side_effect = receive
        for frame in (b"first", b"bad", b"last"):
            await conn.asr_audio_queue.put(frame)
        await provider.open_audio_channels(conn)
        original_task = conn.asr_audio_task
        await provider.open_audio_channels(conn)
        assert conn.asr_audio_task is original_task
        await asyncio.wait_for(conn.asr_audio_queue.join(), 2)
        assert received == [b"first", b"bad", b"last"]
        conn.asr_audio_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await conn.asr_audio_task
        conn.stop_event.set()
        await provider.open_audio_channels(conn)
        assert conn.asr_audio_task is original_task

    asyncio.run(scenario())


def test_listen_stop_waits_for_pending_audio(runtime_loader, asr_base):
    from core.providers.asr.dto.dto import InterfaceType

    module = runtime_loader(
        "core/handle/textHandler/listenMessageHandler.py",
        **{
            "core.handle.receiveAudioHandle": {"startToChat": AsyncMock()},
            "core.handle.reportHandle": {"enqueue_asr_report": Mock()},
            "core.handle.sendAudioHandle": {
                "send_stt_message": AsyncMock(), "send_tts_message": AsyncMock()
            },
            "core.utils.util": {"remove_punctuation_and_length": Mock()},
        },
    )

    async def scenario():
        provider = make_provider(asr_base)
        conn = make_connection(
            client_listen_mode="manual",
            vad=SimpleNamespace(is_vad_async=AsyncMock(return_value=True)),
            asr=SimpleNamespace(interface_type=InterfaceType.STREAM, _send_stop_request=AsyncMock()),
        )
        entered, release = asyncio.Event(), asyncio.Event()

        async def receive(_, frame, have_voice):
            entered.set()
            await release.wait()

        asr_base.handleAudioMessage.side_effect = receive
        await conn.asr_audio_queue.put(b"last-frame")
        await provider.open_audio_channels(conn)
        await entered.wait()
        stop_task = asyncio.create_task(module.ListenTextMessageHandler().handle(conn, {"state": "stop"}))
        await asyncio.sleep(0)
        conn.asr._send_stop_request.assert_not_called()
        assert not conn.client_voice_stop
        release.set()
        await asyncio.wait_for(stop_task, 2)
        await asyncio.sleep(0)
        conn.asr._send_stop_request.assert_awaited_once()
        conn.asr_audio_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await conn.asr_audio_task

    asyncio.run(scenario())


@pytest.fixture
def silero(runtime_loader, monkeypatch):
    torch = pytest.importorskip("torch")
    module = runtime_loader("core/providers/vad/silero.py")
    model_path = SERVER_ROOT / "models/snakers4_silero-vad/src/silero_vad/data/silero_vad.jit"
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    monkeypatch.setattr(torch.hub, "load", lambda **kwargs: (model, None))
    provider = module.VADProvider({"model_dir": "unused", "max_pending_inferences": 2})
    return module, provider


def test_real_silero_interleaved_states_match_independent_models(silero):
    module, provider = silero
    torch = module.torch
    model_path = SERVER_ROOT / "models/snakers4_silero-vad/src/silero_vad/data/silero_vad.jit"
    independent = [torch.jit.load(str(model_path)).eval() for _ in range(2)]
    connections = [make_connection() for _ in range(2)]
    # 只替代 Opus 解码，模型使用仓库真实 JIT；与独立模型逐帧比较内部流状态。
    module.opuslib_next.Decoder.side_effect = lambda *args: SimpleNamespace(decode=lambda packet, _: packet)
    rng = np.random.default_rng(42)
    for index in range(40):
        for connection, model in zip(connections, independent):
            pcm = rng.integers(-20000, 20000, size=512, dtype=np.int16)
            with torch.no_grad():
                model(torch.from_numpy(pcm.astype(np.float32) / 32768), 16000)
            provider.is_vad(connection, pcm.tobytes())
            for name in ("_state", "_context"):
                torch.testing.assert_close(connection._vad_model_state[name], getattr(model, name))
    assert connections[0]._vad_opus_decoder is not connections[1]._vad_opus_decoder
    provider.release_connection(connections[0])
    assert not hasattr(connections[0], "_vad_opus_decoder")
    assert not hasattr(connections[0], "_vad_model_state")
    assert torch.get_num_threads() == 1


def test_vad_cancel_waits_for_thread_before_releasing_slot(silero):
    _, provider = silero
    entered, release = threading.Event(), threading.Event()

    def blocked_inference(*args):
        entered.set()
        assert release.wait(timeout=5)
        return True

    provider.is_vad = blocked_inference
    provider._inference_slots = asyncio.Semaphore(1)

    async def scenario():
        task = asyncio.create_task(provider.is_vad_async(make_connection(), b"audio"))
        try:
            for _ in range(500):
                if entered.is_set():
                    break
                await asyncio.sleep(0.001)
            assert entered.is_set()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert provider._inference_slots.locked()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        assert not provider._inference_slots.locked()
        assert await provider.is_vad_async(make_connection(client_listen_mode="manual"), b"")

    asyncio.run(scenario())


def test_bailian_waits_for_ready_and_sends_preroll_once(runtime_loader, asr_base):
    async def scenario():
        ready = asyncio.Event()
        waiting = asyncio.Event()
        hold_results = asyncio.Event()
        decoded = []

        async def recv():
            if not waiting.is_set():
                waiting.set()
                await ready.wait()
                return '{"header":{"event":"task-started"}}'
            await hold_results.wait()

        ws = SimpleNamespace(send=AsyncMock(), recv=recv, close=AsyncMock())
        connect = AsyncMock(return_value=ws)
        module = runtime_loader(
            "core/providers/asr/aliyunbl_stream.py",
            **{
                "core.providers.asr.base": {"ASRProviderBase": asr_base.ASRProviderBase},
                "websockets": {"connect": connect, "ConnectionClosed": ConnectionError},
            },
        )
        provider = module.ASRProvider({}, False)

        def decode(frame, _):
            decoded.append(frame)
            return frame

        provider.decoder = SimpleNamespace(decode=decode)
        conn = make_connection(asr_audio=[b"pre-1", b"pre-2"])
        first = asyncio.create_task(provider.receive_audio(conn, b"trigger", True))
        try:
            await asyncio.wait_for(waiting.wait(), 2)
            await asyncio.sleep(0)
            # 云端尚未 ready 时必须让消费者背压，不能继续滑窗丢掉首音节。
            assert not first.done()
            ready.set()
            await asyncio.wait_for(first, 2)
            await provider.receive_audio(conn, b"next", True)
            assert decoded == [b"pre-1", b"pre-2", b"trigger", b"next"]
            assert [call.args[0] for call in ws.send.await_args_list if isinstance(call.args[0], bytes)] == decoded
            await provider._send_stop_request()
            await provider._send_stop_request()
            actions = [json.loads(call.args[0])["header"]["action"]
                       for call in ws.send.await_args_list if isinstance(call.args[0], str)]
            assert actions == ["run-task", "finish-task"]
            assert connect.call_args.kwargs["compression"] is None
        finally:
            ready.set()
            await asyncio.gather(first, return_exceptions=True)
            await provider.close()
        assert provider.forward_task is None

    asyncio.run(scenario())


def test_bailian_ready_timeout_releases_receiver_and_preserves_decoder(runtime_loader, asr_base):
    async def scenario():
        blocked = asyncio.Event()
        ws = SimpleNamespace(send=AsyncMock(), recv=blocked.wait, close=AsyncMock())
        module = runtime_loader(
            "core/providers/asr/aliyunbl_stream.py",
            **{
                "core.providers.asr.base": {"ASRProviderBase": asr_base.ASRProviderBase},
                "websockets": {"connect": AsyncMock(return_value=ws), "ConnectionClosed": ConnectionError},
            },
        )
        provider = module.ASRProvider({"ready_timeout": 0.01}, False)
        decoder = provider.decoder
        await asyncio.wait_for(provider.receive_audio(make_connection(), b"audio", True), 2)
        assert not provider.is_processing
        assert provider.asr_ws is None
        assert provider.forward_task is None
        assert provider.decoder is decoder
        ws.close.assert_awaited()

    asyncio.run(scenario())


def test_doubao_fast_gzip_replays_each_frame_once(runtime_loader, asr_base):
    async def scenario():
        blocked = asyncio.Event()
        ws = SimpleNamespace(send=AsyncMock(), recv=AsyncMock(return_value=b"ready"), close=AsyncMock())
        connect = AsyncMock(return_value=ws)
        module = runtime_loader(
            "core/providers/asr/doubao_stream.py",
            **{
                "core.providers.asr.base": {"ASRProviderBase": asr_base.ASRProviderBase},
                "websockets": {"connect": connect, "ConnectionClosed": ConnectionError},
            },
        )
        provider = module.ASRProvider({}, False)
        provider.decoder = SimpleNamespace(decode=lambda frame, _: frame)
        provider.parse_response = lambda _: {"code": 1000}
        provider._forward_asr_results = lambda _: blocked.wait()
        conn = make_connection(asr_audio=[b"pre-1", b"pre-2"])
        try:
            await provider.receive_audio(conn, b"trigger", True)
            await provider.receive_audio(conn, b"next", True)
            await provider._send_stop_request()
            packets = [call.args[0] for call in ws.send.await_args_list]
            # 第一包为初始化；后续包验证协议头、长度和 gzip 解压结果。
            decoded = []
            for packet in packets[1:]:
                assert len(packet) - 8 == int.from_bytes(packet[4:8], "big")
                assert packet[2] & 0x0F == 1
                decoded.append(gzip.decompress(packet[8:]))
            assert decoded == [b"pre-1", b"pre-2", b"trigger", b"next", b""]
            assert provider.gzip_compresslevel == 1
            assert connect.call_args.kwargs["compression"] is None
            assert connect.call_args.kwargs["open_timeout"] == 5
        finally:
            await provider.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("frame_count", [1, 16])
@pytest.mark.parametrize("replace_buffer", [False, True])
def test_doubao_final_result_uses_current_audio_even_for_short_commands(
    runtime_loader, asr_base, frame_count, replace_buffer
):
    async def scenario():
        module = runtime_loader(
            "core/providers/asr/doubao_stream.py",
            **{
                "core.providers.asr.base": {"ASRProviderBase": asr_base.ASRProviderBase},
                "websockets": {"ConnectionClosed": ConnectionError},
            },
        )
        provider = module.ASRProvider({}, False)
        frames = [b"command"] * frame_count
        conn = make_connection(
            asr_audio_for_voiceprint=[] if replace_buffer else frames,
            reset_vad_states=AsyncMock(),
        )
        calls = 0

        async def receive():
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ConnectionError("服务端结束")
            # recv 等待期间，音频消费者可能已经开始新的一份录音缓存。
            await asyncio.sleep(0)
            if replace_buffer:
                conn.asr_audio_for_voiceprint = frames
            return b"final"

        ws = SimpleNamespace(recv=receive, close=AsyncMock())
        provider.asr_ws = ws
        provider.is_processing = True
        provider.parse_response = lambda _: {"payload_msg": {"result": {
            "text": "停下", "utterances": [{"text": "停下", "definite": True}]
        }}}
        provider.handle_voice_stop = AsyncMock(wraps=provider.handle_voice_stop)
        await asyncio.wait_for(provider._forward_asr_results(conn), 2)
        provider.handle_voice_stop.assert_awaited_once()
        captured = provider.handle_voice_stop.call_args.args[1]
        assert captured == frames
        assert captured is not frames
        asr_base.startToChat.assert_awaited_once_with(conn, "停下")
        assert not conn.asr_audio_for_voiceprint
        assert not conn._asr_voiceprint_capture_active
        assert not provider.is_processing

    asyncio.run(scenario())


def test_doubao_stop_is_sent_once_and_rejects_late_audio(runtime_loader, asr_base):
    async def scenario():
        module = runtime_loader(
            "core/providers/asr/doubao_stream.py",
            **{"core.providers.asr.base": {"ASRProviderBase": asr_base.ASRProviderBase}},
        )
        provider = module.ASRProvider({}, False)
        provider.decoder = SimpleNamespace(decode=lambda audio, _: audio)
        provider.asr_ws = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
        provider.is_processing = True
        ws = provider.asr_ws
        await asyncio.gather(provider._send_stop_request(), provider._send_stop_request())
        await provider.receive_audio(make_connection(), b"late", True)
        assert ws.send.await_count == 1
        assert gzip.decompress(ws.send.call_args.args[0][8:]) == b""
        await provider.close()

    asyncio.run(scenario())


@pytest.fixture
def connection_runtime(runtime_loader):
    dependencies = {
        "core.utils.util": {
            name: Mock() for name in (
                "extract_json_from_string", "check_vad_update", "check_asr_update",
                "filter_sensitive_info",
            )
        },
        "core.utils.modules_initialize": {
            name: Mock() for name in ("initialize_modules", "initialize_tts", "initialize_asr")
        },
        "core.handle.reportHandle": {"report": AsyncMock()},
        "core.handle.sendAudioHandle": {
            "sendAudio": AsyncMock(),
            "send_tts_message": AsyncMock(),
        },
        "core.providers.tts.default": {"DefaultTTS": Mock()},
        "core.handle.textHandle": {"handleTextMessage": AsyncMock()},
        "core.providers.tools.unified_tool_handler": {"UnifiedToolHandler": Mock()},
        "plugins_func.loadplugins": {"auto_import_modules": Mock()},
        "core.auth": {"AuthenticationError": RuntimeError},
        "config.config_loader": {"get_private_config_from_api": AsyncMock()},
        "config.logger": {
            "setup_logging": Mock(), "build_module_string": Mock(),
            "create_connection_logger": Mock(),
        },
        "core.utils.prompt_manager": {"PromptManager": Mock()},
        "core.utils.voiceprint_provider": {"VoiceprintProvider": Mock()},
        "config.manage_api_client": {
            "DeviceNotFoundException": RuntimeError, "DeviceBindException": RuntimeError,
            "report": AsyncMock(),
        },
        "core.utils.dialogue": {"Message": Mock(), "Dialogue": Mock()},
    }
    dependencies["core.utils.util"]["audio_to_data"] = AsyncMock(return_value=[])
    module = runtime_loader("core/connection.py", **dependencies)

    def make(slots=None, audio_slots=None, executor=None, connection_slots=None):
        server = SimpleNamespace(
            report_slots=slots,
            report_audio_slots=audio_slots,
            connection_executor=executor,
            connection_slots=connection_slots,
        )
        conn = module.ConnectionHandler({"exit_commands": []}, None, None, None, None, None, server)
        conn.loop = asyncio.get_running_loop()
        return conn

    return module, make


def test_connections_share_bounded_executor_and_cancel_only_queued_work(connection_runtime):
    _, make = connection_runtime

    async def scenario():
        executor = ThreadPoolExecutor(max_workers=1)
        slots = threading.BoundedSemaphore(3)
        first, second = make(executor=executor, connection_slots=slots), make(
            executor=executor, connection_slots=slots
        )
        entered, release = threading.Event(), threading.Event()

        def blocked():
            entered.set()
            assert release.wait(timeout=2)

        try:
            running = first.submit_background(blocked)
            assert running is not None
            assert await asyncio.to_thread(entered.wait, 2)
            queued = second.submit_background(lambda: "must be cancelled")
            survivor = first.submit_background(lambda: "still connected")
            assert queued is not None and survivor is not None
            assert second.submit_background(lambda: None) is None
            await second.close()
            assert queued.cancelled()
            assert not survivor.cancelled()
            assert second.submit_background(lambda: None) is None
            release.set()
            await asyncio.wait_for(asyncio.wrap_future(running), 2)
            assert await asyncio.wait_for(asyncio.wrap_future(survivor), 2) == "still connected"
            await first.close()
            assert executor.submit(lambda: "still open").result(timeout=2) == "still open"
            for _ in range(3):
                assert slots.acquire(blocking=False)
        finally:
            release.set()
            await first.close()
            await second.close()
            executor.shutdown(wait=True)


def test_failed_background_task_returns_budget(connection_runtime):
    _, make = connection_runtime

    async def scenario():
        slots = threading.BoundedSemaphore(1)
        conn = make(connection_slots=slots)

        def fail():
            raise ValueError("任务失败")

        try:
            failed = conn.submit_background(fail)
            with pytest.raises(ValueError, match="任务失败"):
                await asyncio.wrap_future(failed)
            succeeding = conn.submit_background(lambda: "recovered")
            assert succeeding is not None
            assert await asyncio.wrap_future(succeeding) == "recovered"
        finally:
            await conn.close()

    asyncio.run(scenario())


def test_rejected_reply_sends_tts_stop(connection_runtime):
    module, make = connection_runtime

    async def scenario():
        slots = threading.BoundedSemaphore(1)
        assert slots.acquire(blocking=False)
        conn = make(connection_slots=slots)
        future = await conn.submit_reply(lambda: None)
        assert future is None
        assert module.send_tts_message.await_args_list[0].args == (conn, "sentence_start", "现在请求较多，请稍后再试。")
        assert module.send_tts_message.await_args_list[-1].args == (conn, "stop")
        assert module.send_tts_message.await_count == 2
        await conn.close()
        slots.release()

    asyncio.run(scenario())


def test_reply_survives_brief_capacity_spike_without_busy_response(connection_runtime):
    module, make = connection_runtime

    async def scenario():
        slots = threading.BoundedSemaphore(1)
        assert slots.acquire(blocking=False)
        conn = make(connection_slots=slots)
        asyncio.get_running_loop().call_soon(slots.release)
        work = Mock(return_value="reply")
        try:
            future = await conn.submit_reply(work)
            assert future is not None
            assert await asyncio.wait_for(asyncio.wrap_future(future), 1) == "reply"
            work.assert_called_once()
            module.send_tts_message.assert_not_awaited()
        finally:
            await conn.close()

    asyncio.run(scenario())


def test_full_audio_queue_preserves_first_and_following_packets(connection_runtime):
    _, make = connection_runtime

    async def scenario():
        conn = make()
        conn.asr_audio_queue = asyncio.Queue(maxsize=1)
        assert await conn._put_audio_packet(b"first-syllable")
        pending = asyncio.create_task(conn._put_audio_packet(b"next-syllable"))
        await asyncio.sleep(0)
        assert not pending.done()
        assert conn.audio_queue_full_count == 1
        assert await conn.asr_audio_queue.get() == b"first-syllable"
        conn.asr_audio_queue.task_done()
        assert await asyncio.wait_for(pending, 1)
        assert await conn.asr_audio_queue.get() == b"next-syllable"
        conn.asr_audio_queue.task_done()
        await asyncio.wait_for(conn.asr_audio_queue.join(), 1)
        assert not conn._audio_queue_put_tasks
        await conn.close()

    asyncio.run(scenario())


def test_close_cancels_full_queue_put_before_drain(connection_runtime):
    _, make = connection_runtime

    async def scenario():
        conn = make()
        conn.asr_audio_queue = asyncio.Queue(maxsize=1)
        assert await conn._put_audio_packet(b"first")
        pending = asyncio.create_task(conn._put_audio_packet(b"waiting"))
        await asyncio.sleep(0)
        assert not pending.done()
        await asyncio.wait_for(conn.close(), 1)
        assert not await asyncio.wait_for(pending, 1)
        assert not conn._audio_queue_put_tasks
        assert conn.asr_audio_queue.empty()
        await asyncio.wait_for(conn.asr_audio_queue.join(), 1)
        assert not await conn._put_audio_packet(b"after-close")
        assert conn.asr_audio_queue.empty()

    asyncio.run(scenario())


def test_cancelled_audio_queue_producer_does_not_enqueue_later(connection_runtime):
    _, make = connection_runtime

    async def scenario():
        conn = make()
        conn.asr_audio_queue = asyncio.Queue(maxsize=1)
        await conn._put_audio_packet(b"first")
        pending = asyncio.create_task(conn._put_audio_packet(b"cancelled"))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not conn._audio_queue_put_tasks
        assert await conn.asr_audio_queue.get() == b"first"
        conn.asr_audio_queue.task_done()
        await asyncio.sleep(0)
        assert conn.asr_audio_queue.empty()
        await asyncio.wait_for(conn.asr_audio_queue.join(), 1)
        await conn.close()

    asyncio.run(scenario())


@pytest.fixture
def wakeup_runtime(runtime_loader):
    return runtime_loader(
        "core/handle/helloHandle.py",
        **{
            "core.utils.dialogue": {"Message": Mock()},
            "core.utils.util": {
                "audio_to_data": AsyncMock(return_value=[b"confirm"]),
                "remove_punctuation_and_length": lambda text: (len(text), text),
                "opus_datas_to_wav_bytes": Mock(),
            },
            "core.utils.wakeup_word": {"WakeupWordsConfig": Mock()},
            "core.handle.sendAudioHandle": {
                "sendAudioMessage": AsyncMock(), "send_tts_message": AsyncMock(),
            },
            "core.providers.tools.device_mcp": {
                "MCPClient": Mock(), "send_mcp_initialize_message": AsyncMock(),
            },
        },
    )


def make_wakeup_connection(mode="short", **overrides):
    logger = Mock()
    logger.bind.return_value = logger
    config = {"wakeup_words": ["你好小智"], "server": {}}
    if mode is not None:
        config["server"]["wakeup_response_mode"] = mode
    return SimpleNamespace(**(dict(
        config=config, tts=SimpleNamespace(voice="test-voice"),
        stop_event=threading.Event(), logger=logger, dialogue=Mock(),
        just_woken_up=False,
    ) | overrides))


def test_short_wakeup_prepares_audio_before_start_without_dynamic_tts(wakeup_runtime):
    module = wakeup_runtime
    conn = make_wakeup_connection()

    async def prepare(*args, **kwargs):
        assert not conn.just_woken_up
        module.send_tts_message.assert_not_awaited()
        return [b"confirm"]

    async def send(*args):
        assert conn.just_woken_up

    module.audio_to_data.side_effect = prepare
    module.sendAudioMessage.side_effect = send
    assert asyncio.run(module.checkWakeupWords(conn, "你好小智"))
    module.audio_to_data.assert_awaited_once_with(
        module.DEFAULT_SHORT_WAKEUP_AUDIO, use_cache=True
    )
    module.wakeup_words_config.get_wakeup_response.assert_not_called()
    assert module.sendAudioMessage.await_count == 2
    assert module.sendAudioMessage.await_args_list[0].args[-1] is None
    conn.dialogue.put.assert_not_called()
    assert not conn.just_woken_up


def test_first_audio_after_wakeup_reaches_ordered_consumer(
    connection_runtime, asr_base, wakeup_runtime
):
    _, make = connection_runtime
    module = wakeup_runtime

    async def scenario():
        conn = make()
        conn.config.update({"server": {"wakeup_response_mode": "short"}, "wakeup_words": ["你好小智"]})
        conn.bind_completed_event.set()
        conn.need_bind = False
        conn.tts = SimpleNamespace(voice="test-voice")
        conn.vad = SimpleNamespace(is_vad_async=AsyncMock(return_value=True))
        conn.asr = make_provider(asr_base)

        async def play(*args):
            assert conn.just_woken_up
            await conn._route_message(b"speaker-echo")

        module.sendAudioMessage.side_effect = play
        await conn.asr.open_audio_channels(conn)
        try:
            assert await module.checkWakeupWords(conn, "你好小智")
            await conn._route_message(b"user-first-syllable")
            await asyncio.wait_for(conn.asr_audio_queue.join(), 1)
            asr_base.handleAudioMessage.assert_awaited_once_with(
                conn, b"user-first-syllable", True
            )
        finally:
            conn.tts = None
            await conn.close()

    asyncio.run(scenario())


def test_short_wakeup_uses_real_audio_send_start_packets_stop(wakeup_runtime, runtime_loader):
    module = wakeup_runtime
    sender = runtime_loader(
        "core/handle/sendAudioHandle.py",
        **{"core.utils.util": {"audio_to_data": AsyncMock()}},
    )
    module.sendAudioMessage = sender.sendAudioMessage
    module.send_tts_message = sender.send_tts_message
    conn = make_wakeup_connection(
        tts=SimpleNamespace(voice="test-voice", tts_audio_first_sentence=True),
        session_id="test-session", websocket=SimpleNamespace(send=AsyncMock()),
        client_abort=False, client_is_speaking=False, close_after_chat=False,
        conn_from_mqtt_gateway=False, clearSpeakStatus=Mock(),
    )
    module.audio_to_data.return_value = [b"frame-1", b"frame-2", b"frame-3"]

    async def scenario():
        try:
            assert await asyncio.wait_for(module.checkWakeupWords(conn, "你好小智"), 2)
            sent = [call.args[0] for call in conn.websocket.send.await_args_list]
            assert json.loads(sent[0])["state"] == "start"
            assert [item for item in sent if isinstance(item, bytes)] == [
                b"frame-1", b"frame-2", b"frame-3"
            ]
            assert json.loads(sent[-1])["state"] == "stop"
            assert not conn.just_woken_up
            assert not conn.client_is_speaking
        finally:
            if hasattr(conn, "audio_rate_controller"):
                task = conn.audio_rate_controller.pending_send_task
                conn.audio_rate_controller.reset()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", [None, "cached", "invalid"])
def test_cached_wakeup_preserves_existing_greeting(wakeup_runtime, mode):
    module = wakeup_runtime
    conn = make_wakeup_connection(mode)
    module.wakeup_words_config.get_wakeup_response.return_value = {
        "file_path": "dynamic-greeting.wav", "text": "我在，请说。",
        "time": module.time.time(),
    }
    assert asyncio.run(module.checkWakeupWords(conn, "你好小智"))
    module.wakeup_words_config.get_wakeup_response.assert_called_once_with("test-voice")
    module.audio_to_data.assert_awaited_once_with("dynamic-greeting.wav", use_cache=False)
    conn.dialogue.put.assert_called_once()
    assert not conn.just_woken_up


def test_missing_dynamic_greeting_falls_back_to_short_confirmation(wakeup_runtime):
    module = wakeup_runtime
    conn = make_wakeup_connection("cached")
    module.wakeup_words_config.get_wakeup_response.return_value = {
        "file_path": "missing.wav", "text": "旧问候", "time": 0,
    }
    module.audio_to_data.side_effect = [FileNotFoundError(), [b"confirm"]]
    assert asyncio.run(module.checkWakeupWords(conn, "你好小智"))
    assert module.audio_to_data.await_args_list[-1].args == (module.DEFAULT_SHORT_WAKEUP_AUDIO,)
    conn.dialogue.put.assert_not_called()
    assert not conn.just_woken_up


@pytest.mark.parametrize("failure_at", ["start", "first", "last"])
def test_wakeup_send_failure_stops_tts_and_restores_listening(wakeup_runtime, failure_at):
    module = wakeup_runtime
    conn = make_wakeup_connection()
    if failure_at == "start":
        module.send_tts_message.side_effect = [RuntimeError("send failed"), None]
    else:
        module.sendAudioMessage.side_effect = (
            [RuntimeError("send failed")] if failure_at == "first"
            else [None, RuntimeError("send failed")]
        )
    with pytest.raises(RuntimeError, match="send failed"):
        asyncio.run(module.checkWakeupWords(conn, "你好小智"))
    assert not conn.just_woken_up
    assert module.send_tts_message.await_args_list[-1].args == (conn, "stop")
    conn.dialogue.put.assert_not_called()


def test_wakeup_cancel_restores_listening(wakeup_runtime):
    module = wakeup_runtime
    conn = make_wakeup_connection()

    async def scenario():
        entered = asyncio.Event()

        async def block(*args):
            entered.set()
            await asyncio.Event().wait()

        module.sendAudioMessage.side_effect = block
        task = asyncio.create_task(module.checkWakeupWords(conn, "你好小智"))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not conn.just_woken_up
        assert module.send_tts_message.await_args_list[-1].args == (conn, "stop")

    asyncio.run(scenario())


@pytest.mark.parametrize("text, enabled", [("今天天气", True), ("你好小智", False)])
def test_non_wakeup_or_disabled_cache_does_not_pause_recording(wakeup_runtime, text, enabled):
    module = wakeup_runtime
    conn = make_wakeup_connection(tts=None)
    conn.config["enable_wakeup_words_response_cache"] = enabled
    assert not asyncio.run(module.checkWakeupWords(conn, text))
    module.audio_to_data.assert_not_awaited()
    module.send_tts_message.assert_not_awaited()
    assert not conn.just_woken_up


def test_reports_share_global_budget_and_run_on_main_loop(connection_runtime):
    module, make = connection_runtime

    async def scenario():
        slots = threading.BoundedSemaphore(1)
        first, second = make(slots), make(slots)
        entered, release = asyncio.Event(), asyncio.Event()
        main_loop = asyncio.get_running_loop()

        async def report(*args):
            assert asyncio.get_running_loop() is main_loop
            entered.set()
            await release.wait()

        module.report.side_effect = report
        try:
            assert await asyncio.to_thread(first.enqueue_report, 1, "指令", None, 0)
            await asyncio.wait_for(entered.wait(), 2)
            assert not second.enqueue_report(1, "拥塞", None, 0)
            release.set()
            await asyncio.gather(*first._report_tasks)
            assert not first._report_tasks
            assert second.enqueue_report(1, "恢复", None, 0)
            await asyncio.sleep(0)
            await asyncio.gather(*second._report_tasks)
        finally:
            release.set()
            await first.close()
            await second.close()

    asyncio.run(scenario())


def test_close_cancels_reports_and_releases_global_slot(connection_runtime):
    module, make = connection_runtime

    async def scenario():
        slots = threading.BoundedSemaphore(1)
        conn = make(slots)
        entered = asyncio.Event()

        async def report(*args):
            entered.set()
            await asyncio.Event().wait()

        module.report.side_effect = report
        assert conn.enqueue_report(1, "未完成", None, 0)
        await asyncio.wait_for(entered.wait(), 2)
        tasks = list(conn._report_tasks)
        await asyncio.wait_for(conn.close(), 2)
        await asyncio.sleep(0)
        assert conn.stop_event.is_set()
        assert not conn._report_tasks
        assert all(task.cancelled() for task in tasks)
        assert slots.acquire(blocking=False)
        slots.release()
        assert not conn.enqueue_report(1, "已断连", None, 0)
        await conn.close()

    asyncio.run(scenario())


def test_report_cancel_before_scheduling_returns_slot(connection_runtime):
    _, make = connection_runtime

    async def scenario():
        slots = threading.BoundedSemaphore(1)
        conn = make(slots)
        assert conn.enqueue_report(1, "取消", None, 0)
        await conn.close()
        await asyncio.sleep(0)
        assert not conn._report_tasks
        assert slots.acquire(blocking=False)
        slots.release()

    asyncio.run(scenario())


def test_report_audio_conversion_runs_off_loop(runtime_loader):
    manage_report = AsyncMock()
    module = runtime_loader(
        "core/handle/reportHandle.py",
        **{"config.manage_api_client": {"report": manage_report}},
    )
    worker_threads = []
    main_thread = threading.get_ident()

    def convert(conn, opus):
        worker_threads.append(threading.get_ident())
        return b"RIFF-test"

    module.opus_to_wav = convert
    conn = make_connection(
        device_id="test", logger=Mock(), _report_audio_slots=asyncio.Semaphore(1)
    )
    asyncio.run(module.report(conn, 1, "指令", [b"opus"], 0))
    assert worker_threads and worker_threads[0] != main_thread
    assert manage_report.call_args.kwargs["audio"] == b"RIFF-test"


@pytest.mark.parametrize("conversion_fails", [False, True])
def test_report_conversion_cancel_holds_global_budget_until_thread_finishes(
    connection_runtime, runtime_loader, conversion_fails
):
    connection_module, make = connection_runtime
    manage_report = AsyncMock()
    report_module = runtime_loader(
        "core/handle/reportHandle.py",
        **{"config.manage_api_client": {"report": manage_report}},
    )
    connection_module.report = report_module.report
    entered, release = threading.Event(), threading.Event()

    def blocked_conversion(*args):
        entered.set()
        assert release.wait(timeout=5)
        if conversion_fails:
            raise ValueError("模拟解码失败")
        return b"RIFF-test"

    report_module.opus_to_wav = blocked_conversion

    async def scenario():
        slots = threading.BoundedSemaphore(1)
        audio_slots = asyncio.Semaphore(1)
        conn = make(slots, audio_slots)
        close_task = None
        try:
            assert conn.enqueue_report(1, "取消中的录音", [b"opus"], 0)
            assert await asyncio.to_thread(entered.wait, 2)
            tasks = list(conn._report_tasks)
            close_task = asyncio.create_task(conn.close())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not close_task.done()
            assert audio_slots.locked()
            assert not slots.acquire(blocking=False)
            release.set()
            await asyncio.wait_for(close_task, 2)
            assert all(task.cancelled() for task in tasks)
            assert slots.acquire(blocking=False)
            slots.release()
            assert not audio_slots.locked()
            manage_report.assert_not_awaited()
        finally:
            release.set()
            if close_task is not None:
                await close_task
            await conn.close()

    asyncio.run(scenario())


def test_report_audio_conversion_is_serial_across_connections(runtime_loader):
    report_module = runtime_loader(
        "core/handle/reportHandle.py",
        **{"config.manage_api_client": {"report": AsyncMock()}},
    )
    entered, release = threading.Event(), threading.Event()
    converted = []

    def conversion(conn, opus):
        converted.append(conn.device_id)
        if conn.device_id == "first":
            entered.set()
            assert release.wait(timeout=5)
        return b"RIFF-test"

    report_module.opus_to_wav = conversion

    async def scenario():
        audio_slots = asyncio.Semaphore(1)
        first = make_connection(device_id="first", logger=Mock(), _report_audio_slots=audio_slots)
        second = make_connection(device_id="second", logger=Mock(), _report_audio_slots=audio_slots)
        tasks = [asyncio.create_task(report_module.report(first, 1, "一", [b"opus"], 0))]
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            tasks.append(asyncio.create_task(report_module.report(second, 1, "二", [b"opus"], 0)))
            await asyncio.sleep(0)
            assert converted == ["first"]
            assert not tasks[1].done()
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 2)
            assert converted == ["first", "second"]
            assert report_module.manage_report.await_count == 2
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("enqueue_name", ["enqueue_asr_report", "enqueue_tts_report"])
@pytest.mark.parametrize("accepted", [False, True])
def test_report_success_log_requires_accepted_task(runtime_loader, enqueue_name, accepted):
    report_module = runtime_loader(
        "core/handle/reportHandle.py",
        **{"config.manage_api_client": {"report": AsyncMock()}},
    )
    logger = Mock()
    logger.bind.return_value = logger
    conn = make_connection(
        logger=logger, device_id="test", read_config_from_api=True, need_bind=False,
        report_asr_enable=True, report_tts_enable=True, chat_history_conf=2,
        enqueue_report=Mock(return_value=accepted),
    )

    getattr(report_module, enqueue_name)(conn, "指令", [b"opus"])

    conn.enqueue_report.assert_called_once()
    assert logger.debug.call_count == int(accepted)
