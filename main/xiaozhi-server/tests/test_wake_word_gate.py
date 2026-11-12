"""唤醒词门控的匹配规则、免检时效和对话入口接入验证。

@author pony
@date 2026-11-12
@version v1.0.0
@last_modified 2026-11-12
@changelog
    - v1.0.0 (2026-11-12): 覆盖首尾匹配、拼音模糊、安全指令、下一句免检时效及入口拦截
"""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


SERVER_ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, SERVER_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate_module = _load("core/utils/wake_word_gate.py", "_wake_word_gate_under_test")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def matcher():
    return gate_module.WakeWordMatcher()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def gate(clock):
    return gate_module.WakeWordGate({"enabled": True}, clock=clock)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("小蒜今天上海天气怎样", "今天上海天气怎样"),
        ("今天上海天气如何小蒜", "今天上海天气如何"),
        ("今天上海天气如何，小算？", "今天上海天气如何？"),
        ("今天天气怎么样小蒜啊", "今天天气怎么样"),
        ("你好小酸，导航到卫生间", "导航到卫生间"),
        ("嘿，你好小蒜，讲个笑话", "嘿，讲个笑话"),
        ("好的小蒜，帮我查下新闻", "好的，帮我查下新闻"),
        ("晓蒜给我讲一个笑话", "给我讲一个笑话"),
        ("再见，小帅。", "再见"),
        ("停下小蒜", "停下"),
        ("诺娃，现在几点", "现在几点"),
        ("hi Nova, what's the weather like today?", "what's the weather like today?"),
        (
            "What's the weather like in Shanghai today, Nova?",
            "What's the weather like in Shanghai today?",
        ),
    ],
)
def test_wake_word_at_head_or_tail_is_stripped(matcher, text, expected):
    found = matcher.match(text)
    assert found is not None
    assert found.text == expected
    assert not found.wake_only


@pytest.mark.parametrize(
    "text",
    [
        "今天上海天气怎样",
        "他们，他们几把枪啊？",
        "你，你没有，你问他呀？",
        "不是啊？对啊，那也，他也是，我算我小岛。",
        # 句中提及：旁人谈论机器人，不算对它说话
        "你看那个小蒜会导航",
        "我问你小蒜今天天气",
        "这个小三真是的",
        "",
        "。？",
    ],
)
def test_no_wake_word_or_mid_sentence_mention_does_not_match(matcher, text):
    assert matcher.match(text) is None


@pytest.mark.parametrize(
    "text", ["你好小蒜", "小算。", "嘿，小蒜", "小蒜呀", "Hi, Nova.", "hi Nova", "Nova"]
)
def test_name_only_is_wake_only(matcher, text):
    assert matcher.is_wake_only(text)


@pytest.mark.parametrize("text", ["停", "停下！", "停下吧", "等一下啊", "站住", "别走。"])
def test_short_safety_commands_pass_without_wake_word(gate, text):
    decision = gate.evaluate(text)
    assert decision.passed and decision.reason == "safety"


@pytest.mark.parametrize("text", ["停一下我想想", "别停下", "你先停下来听我说"])
def test_longer_sentences_with_stop_words_are_not_safety_commands(gate, text):
    assert not gate.evaluate(text).passed


def test_plain_sentence_is_blocked_and_wake_sentence_is_stripped(gate):
    blocked = gate.evaluate("今天上海天气怎样")
    assert not blocked.passed and blocked.reason == "no_wake_word"

    passed = gate.evaluate("今天上海天气怎样小蒜")
    assert passed.passed and passed.reason == "wake_tail"
    assert passed.text == "今天上海天气怎样"


def test_manual_mode_skips_check(gate):
    decision = gate.evaluate("今天上海天气怎样", manual=True)
    assert decision.passed and decision.reason == "manual"


def test_device_wake_grants_only_the_next_sentence(gate, clock):
    gate.grant_follow_up("wake")
    gate.on_speech_start()
    gate.on_sentence("主人你好。")
    clock.now = 3.0
    gate.on_speech_end()

    clock.now = 12.0  # 说完后 9 秒
    first = gate.evaluate("今天上海天气怎样")
    assert first.passed and first.reason == "follow_up"
    assert not gate.evaluate("再讲一个").passed


def test_follow_up_expires_after_window(gate, clock):
    gate.grant_follow_up("wake")
    clock.now = 10.5
    assert not gate.evaluate("今天上海天气怎样").passed


def test_window_starts_when_robot_finishes_speaking(gate, clock):
    gate.grant_follow_up("wake")
    clock.now = 0.5
    gate.on_speech_start()
    clock.now = 30.0  # 回复很长，超过有效期才说完
    gate.on_speech_end()
    clock.now = 38.0
    assert gate.evaluate("那明天呢").passed


def test_expired_grant_is_not_revived_by_later_speech(gate, clock):
    gate.grant_follow_up("wake")
    clock.now = 20.0
    gate.on_speech_start()
    gate.on_sentence("已停止导航。")
    clock.now = 22.0
    gate.on_speech_end()
    assert not gate.evaluate("今天上海天气怎样").passed


def test_robot_question_grants_next_sentence(gate, clock):
    gate.on_speech_start()
    gate.on_sentence("国内新闻有这些。")
    gate.on_sentence("要了解详情吗？")
    gate.on_speech_end()
    clock.now = 5.0
    decision = gate.evaluate("要")
    assert decision.passed and decision.reason == "follow_up"


@pytest.mark.parametrize("last_sentence", ["主人觉得好笑吗~", "How can I help?"])
def test_question_detection_ignores_trailing_symbols(gate, last_sentence):
    gate.on_speech_start()
    gate.on_sentence(last_sentence)
    gate.on_speech_end()
    assert gate.evaluate("还行").passed


def test_statement_reply_does_not_grant(gate):
    gate.on_speech_start()
    gate.on_sentence("主人，我在这里呢。")
    gate.on_speech_end()
    assert not gate.evaluate("今天上海天气怎样").passed


def test_interrupted_question_does_not_grant(gate):
    gate.on_speech_start()
    gate.on_sentence("要了解详情吗？")
    gate.on_speech_end(completed=False)
    assert not gate.evaluate("今天上海天气怎样").passed


def test_saying_only_the_name_counts_as_wake(gate, clock):
    decision = gate.evaluate("小蒜")
    assert decision.passed and decision.reason == "wake_only"
    assert decision.text == "小蒜"  # 原样交给大模型，让它回应“我在”
    gate.on_speech_start()
    gate.on_sentence("我在呢。")
    clock.now = 2.0
    gate.on_speech_end()
    assert gate.evaluate("今天上海天气怎样").passed


def test_wake_sentence_also_consumes_grant(gate):
    gate.grant_follow_up("wake")
    assert gate.evaluate("小蒜今天天气怎样").passed
    assert not gate.evaluate("明天呢").passed


def test_from_config_and_per_connection_cache():
    assert gate_module.WakeWordGate.from_config(None) is None
    assert gate_module.WakeWordGate.from_config({"enabled": False}) is None

    conn = SimpleNamespace(config={"server": {"wake_word_gate": {"enabled": True}}})
    first = gate_module.get_wake_word_gate(conn)
    assert isinstance(first, gate_module.WakeWordGate)
    assert gate_module.get_wake_word_gate(conn) is first

    disabled = SimpleNamespace(config={"server": {}})
    assert gate_module.get_wake_word_gate(disabled) is None
    assert gate_module.get_wake_word_gate(disabled) is None
    assert gate_module.get_wake_word_gate(Mock()) is None


def test_custom_words_from_config(clock):
    gate = gate_module.WakeWordGate(
        {"enabled": True, "words": ["旺财"], "head_max_offset": 0}, clock=clock
    )
    assert gate.evaluate("旺财过来").passed
    assert not gate.evaluate("小蒜过来").passed
    assert not gate.evaluate("我说旺财过来").passed
    # “嘿”与问候“嗨”同音（hei），紧挨唤醒词时算作问候，不占句首字数
    assert gate.evaluate("嘿旺财过来").passed


@pytest.fixture
def receive_handle(monkeypatch):
    """入口用真实门控，云端和播报依赖替换为桩。"""
    stubs = {
        "core.utils.util": {"audio_to_data": AsyncMock()},
        "core.handle.abortHandle": {"handleAbortMessage": AsyncMock()},
        "core.handle.intentHandler": {"handle_user_intent": AsyncMock(return_value=False)},
        "core.utils.output_counter": {"check_device_output_limit": Mock(return_value=False)},
        "core.handle.sendAudioHandle": {
            "send_stt_message": AsyncMock(),
            "send_ignored_stt_message": AsyncMock(),
            "SentenceType": SimpleNamespace(FIRST=1, MIDDLE=2, LAST=3),
        },
    }
    for name, attributes in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(__import__("sys").modules, name, module)
    monkeypatch.setitem(__import__("sys").modules, "core.utils.wake_word_gate", gate_module)
    return _load("core/handle/receiveAudioHandle.py", "_receive_audio_under_test")


def _connection():
    logger = Mock()
    logger.bind.return_value = logger
    return SimpleNamespace(
        config={"server": {"wake_word_gate": {"enabled": True}}},
        logger=logger,
        client_listen_mode="auto",
        client_is_speaking=True,
        need_bind=False,
        max_output_size=0,
        headers={},
        chat=Mock(),
        submit_reply=AsyncMock(),
    )


def test_blocked_sentence_is_marked_ignored_and_does_not_interrupt(receive_handle):
    conn = _connection()
    asyncio.run(receive_handle.startToChat(conn, "他们几把枪啊？", from_asr=True))

    receive_handle.send_ignored_stt_message.assert_awaited_once_with(conn, "他们几把枪啊？")
    receive_handle.handleAbortMessage.assert_not_awaited()
    receive_handle.handle_user_intent.assert_not_awaited()
    receive_handle.send_stt_message.assert_not_awaited()
    conn.submit_reply.assert_not_awaited()


def test_wake_sentence_reaches_chat_without_wake_word(receive_handle):
    conn = _connection()
    asyncio.run(receive_handle.startToChat(conn, "小蒜今天上海天气怎样", from_asr=True))

    receive_handle.handle_user_intent.assert_awaited_once_with(conn, "今天上海天气怎样")
    receive_handle.send_stt_message.assert_awaited_once_with(conn, "今天上海天气怎样")
    conn.submit_reply.assert_awaited_once_with(conn.chat, "今天上海天气怎样")
    receive_handle.send_ignored_stt_message.assert_not_awaited()


def test_non_asr_text_bypasses_gate(receive_handle):
    conn = _connection()
    asyncio.run(receive_handle.startToChat(conn, "今天上海天气怎样"))

    receive_handle.handle_user_intent.assert_awaited_once_with(conn, "今天上海天气怎样")
    receive_handle.send_ignored_stt_message.assert_not_awaited()


def test_speaker_json_keeps_structure_and_strips_content(receive_handle):
    conn = _connection()
    text = json.dumps({"speaker": "pony", "content": "今天天气怎样小蒜"}, ensure_ascii=False)
    asyncio.run(receive_handle.startToChat(conn, text, from_asr=True))

    forwarded = receive_handle.handle_user_intent.await_args.args[1]
    assert json.loads(forwarded) == {"speaker": "pony", "content": "今天天气怎样"}


def test_speaker_json_without_wake_word_reports_plain_content(receive_handle):
    conn = _connection()
    text = json.dumps({"speaker": "pony", "content": "今天天气怎样"}, ensure_ascii=False)
    asyncio.run(receive_handle.startToChat(conn, text, from_asr=True))

    receive_handle.send_ignored_stt_message.assert_awaited_once_with(conn, "今天天气怎样")
    receive_handle.handle_user_intent.assert_not_awaited()


def test_gate_disabled_keeps_original_behavior(receive_handle):
    conn = _connection()
    conn.config = {"server": {}}
    asyncio.run(receive_handle.startToChat(conn, "今天上海天气怎样", from_asr=True))

    receive_handle.handle_user_intent.assert_awaited_once_with(conn, "今天上海天气怎样")
    receive_handle.send_ignored_stt_message.assert_not_awaited()
