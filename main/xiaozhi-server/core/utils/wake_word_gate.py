"""服务端唤醒词门控：语音识别出的句子须在句首或句尾带唤醒词才回应。

唤醒词按拼音模糊匹配（平翘舌、n/l、前后鼻音不分），同音误写如“小算”“小酸”
“晓蒜”自动命中。以下情况免检：唤醒或机器人提问后的下一句、按键说话、
“停下”等安全短指令。只处理语音识别结果，文字输入、唤醒问候不经过本门控。

@author pony
@date 2026-11-12
@version v1.0.2
@last_modified 2026-11-14
@changelog
    - v1.0.2 (2026-11-14): 默认唤醒词加入“Hinova”“Heynova”（加热词后豆包把“Hi/Hey Nova”连写成一个词）
    - v1.0.1 (2026-11-13): 默认唤醒词加入“小三”（豆包实测把“小蒜”识别成“小三”）
    - v1.0.0 (2026-11-12): 首尾唤醒词拼音模糊匹配，唤醒/提问后放行下一句，安全指令与按键说话免检
"""

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

from pypinyin import Style, pinyin

DEFAULT_WORDS = ("小蒜", "小帅", "小三", "Nova", "诺瓦", "Hinova", "Heynova")
DEFAULT_GREETINGS = ("你好", "嗨", "hi", "hey", "hello")
DEFAULT_HEAD_MAX_OFFSET = 2
DEFAULT_TAIL_PARTICLES = "啊呀吧呢哈嘛哦啦呗"
DEFAULT_FOLLOW_UP_SECONDS = 10.0
DEFAULT_SAFETY_COMMANDS = ("停", "停下", "停止", "站住", "别走", "等一下")
# 只说唤醒词时，唤醒词前后允许出现的语气词，仍算作“只叫了名字”
_INTERJECTIONS = frozenset(
    ("嘿", "嗨", "喂", "哎", "诶", "嗯", "哦", "hey", "hi", "hello", "ok", "okay")
)
_TERMINAL_PUNCTUATION = "？?！!。."
_QUESTION_ENDINGS = ("？", "?", "吗")


def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"


def _is_word_char(ch: str) -> bool:
    return _is_cjk(ch) or (ch.isascii() and ch.isalnum())


def _normalize_syllable(syllable: str) -> str:
    """把易混读音归成一类：zh/ch/sh→z/c/s，n→l，-ng→-n。"""
    s = syllable.lower()
    for retroflex, flat in (("zh", "z"), ("ch", "c"), ("sh", "s")):
        if s.startswith(retroflex):
            s = flat + s[2:]
            break
    if s.startswith("n"):
        s = "l" + s[1:]
    if s.endswith("ng"):
        s = s[:-1]
    return s


@lru_cache(maxsize=4096)
def _char_keys(ch: str) -> frozenset:
    """单个汉字所有读音的归一化集合，多音字任一读音命中即可。"""
    readings = pinyin(ch, style=Style.NORMAL, heteronym=True, errors="ignore")
    if not readings or not readings[0]:
        return frozenset((ch,))
    return frozenset(_normalize_syllable(r) for r in readings[0])


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    raw: str
    keys: frozenset


def _tokenize(text: str) -> list:
    """汉字逐字成词，连续英文字母/数字成一个词，标点空白只作分隔。"""
    tokens = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if _is_cjk(ch):
            tokens.append(_Token(i, i + 1, ch, _char_keys(ch)))
            i += 1
        elif ch.isascii() and ch.isalnum():
            j = i
            while j < n and text[j].isascii() and text[j].isalnum():
                j += 1
            word = text[i:j].lower()
            tokens.append(_Token(i, j, word, frozenset((word,))))
            i = j
        else:
            i += 1
    return tokens


def _pattern(word: str) -> tuple:
    return tuple(token.keys for token in _tokenize(word))


def _match_at(tokens: list, index: int, pattern: tuple) -> bool:
    if index < 0 or index + len(pattern) > len(tokens):
        return False
    return all(tokens[index + k].keys & keys for k, keys in enumerate(pattern))


def _rstrip_separators(text: str) -> str:
    end = len(text)
    while end > 0 and not _is_word_char(text[end - 1]):
        end -= 1
    return text[:end]


def _lstrip_separators(text: str) -> str:
    start = 0
    while start < len(text) and not _is_word_char(text[start]):
        start += 1
    return text[start:]


def _is_question(text: str) -> bool:
    """跳过句末的标点、表情，看最后一个有效字符是不是问号或“吗”。"""
    for ch in reversed(text or ""):
        if ch in _QUESTION_ENDINGS:
            return True
        if _is_word_char(ch):
            return False
    return False


@dataclass(frozen=True)
class WakeMatch:
    position: str  # head / tail
    text: str  # 去掉唤醒词后的文本
    wake_only: bool  # 整句只有唤醒词（可带语气词）


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    text: str
    reason: str  # wake_head / wake_tail / wake_only / manual / safety / follow_up / no_wake_word


class WakeWordMatcher:
    def __init__(
        self,
        words=DEFAULT_WORDS,
        greetings=DEFAULT_GREETINGS,
        head_max_offset=DEFAULT_HEAD_MAX_OFFSET,
        tail_particles=DEFAULT_TAIL_PARTICLES,
    ):
        patterns = {_pattern(w) for w in words if _pattern(w)}
        # 长的优先，避免短词抢先命中
        self._words = sorted(patterns, key=len, reverse=True)
        self._greetings = sorted(
            {_pattern(g) for g in greetings if _pattern(g)}, key=len, reverse=True
        )
        self._head_max_offset = max(0, int(head_max_offset))
        self._particles = frozenset(tail_particles or "")

    def _greeting_start(self, tokens: list, word_start: int) -> int:
        """唤醒词前紧挨着“你好/hi”时把问候算进唤醒词，不占句首字数。"""
        for greeting in self._greetings:
            start = word_start - len(greeting)
            if _match_at(tokens, start, greeting):
                return start
        return word_start

    def _find_head(self, tokens: list):
        limit = self._head_max_offset + max((len(g) for g in self._greetings), default=0)
        for i in range(min(limit, len(tokens) - 1) + 1):
            for word in self._words:
                if _match_at(tokens, i, word):
                    start = self._greeting_start(tokens, i)
                    if start <= self._head_max_offset:
                        return start, i + len(word)
        return None

    def _find_tail(self, tokens: list):
        end = len(tokens)
        while end > 0 and tokens[end - 1].raw in self._particles:
            end -= 1
        for word in self._words:
            if _match_at(tokens, end - len(word), word):
                return self._greeting_start(tokens, end - len(word)), end
        return None

    def match(self, text: str) -> Optional[WakeMatch]:
        tokens = _tokenize(text or "")
        if not tokens:
            return None
        position, span = "head", self._find_head(tokens)
        if span is None:
            position, span = "tail", self._find_tail(tokens)
        if span is None:
            return None

        first, last = span
        outside = tokens[:first] + tokens[last:]
        wake_only = all(
            t.raw in _INTERJECTIONS or t.raw in self._particles for t in outside
        )
        char_start, char_end = tokens[first].start, tokens[last - 1].end
        if position == "head":
            prefix = _rstrip_separators(text[:char_start])
            rest = _lstrip_separators(text[char_end:])
            stripped = prefix + ("，" if prefix and rest else "") + rest
        else:
            after = text[char_end:]
            terminal = next((c for c in reversed(after) if c in _TERMINAL_PUNCTUATION), "")
            stripped = _rstrip_separators(text[:char_start]) + terminal
        return WakeMatch(position, stripped, wake_only)

    def is_wake_only(self, text: str) -> bool:
        found = self.match(text)
        return bool(found and found.wake_only)


class WakeWordGate:
    """每个连接一份：记录“下一句免检”的授权和机器人是否在播报。"""

    def __init__(self, config: Optional[dict] = None, clock: Callable[[], float] = time.monotonic):
        config = config or {}
        self.matcher = WakeWordMatcher(
            words=config.get("words") or DEFAULT_WORDS,
            greetings=config.get("greetings") or DEFAULT_GREETINGS,
            head_max_offset=config.get("head_max_offset", DEFAULT_HEAD_MAX_OFFSET),
            tail_particles=config.get("tail_particles", DEFAULT_TAIL_PARTICLES),
        )
        self._particles = self.matcher._particles
        self._safety = frozenset(
            self._command_core(c)
            for c in (config.get("safety_commands") or DEFAULT_SAFETY_COMMANDS)
        )
        self._follow_up_seconds = float(
            config.get("follow_up_seconds", DEFAULT_FOLLOW_UP_SECONDS)
        )
        self._clock = clock
        self._grant_reason: Optional[str] = None
        self._grant_since = 0.0
        self._speaking = False
        self._last_sentence = ""

    @classmethod
    def from_config(cls, config) -> Optional["WakeWordGate"]:
        if not isinstance(config, dict) or not config.get("enabled", False):
            return None
        return cls(config)

    def _command_core(self, text: str) -> str:
        core = "".join(t.raw for t in _tokenize(text or ""))
        return core.rstrip("".join(self._particles)) if self._particles else core

    def is_safety_command(self, text: str) -> bool:
        return self._command_core(text) in self._safety

    def _grant_active(self, now: float) -> bool:
        if self._grant_reason is None:
            return False
        return self._speaking or now - self._grant_since <= self._follow_up_seconds

    def grant_follow_up(self, reason: str) -> None:
        """放行下一句；有效期从机器人说完话开始算。"""
        self._grant_reason = reason
        self._grant_since = self._clock()

    def on_speech_start(self) -> None:
        if not self._grant_active(self._clock()):
            self._grant_reason = None
        self._speaking = True
        self._last_sentence = ""

    def on_sentence(self, text: Optional[str]) -> None:
        if text:
            self._last_sentence = text

    def on_speech_end(self, completed: bool = True) -> None:
        """播报结束。正常说完且最后一句是问句时，放行用户的下一句。"""
        now = self._clock()
        if self._grant_reason is not None and not self._grant_active(now):
            self._grant_reason = None
        self._speaking = False
        if completed and _is_question(self._last_sentence):
            self._grant_reason = "question"
        if self._grant_reason is not None:
            self._grant_since = now
        self._last_sentence = ""

    def evaluate(self, text: str, manual: bool = False) -> GateDecision:
        now = self._clock()
        follow_up = self._grant_active(now)
        # 无论放行与否，这一句都用掉“下一句”的授权
        self._grant_reason = None

        found = self.matcher.match(text)
        if found:
            if found.wake_only:
                self.grant_follow_up("wake")
                return GateDecision(True, text, "wake_only")
            return GateDecision(True, found.text, f"wake_{found.position}")
        if manual:
            return GateDecision(True, text, "manual")
        if self.is_safety_command(text):
            return GateDecision(True, text, "safety")
        if follow_up:
            return GateDecision(True, text, "follow_up")
        return GateDecision(False, text, "no_wake_word")


_DISABLED = object()


def get_wake_word_gate(conn) -> Optional[WakeWordGate]:
    """按连接懒加载门控；未启用时返回 None，并缓存该结果。"""
    gate = getattr(conn, "_wake_word_gate", None)
    if isinstance(gate, WakeWordGate):
        return gate
    if gate is _DISABLED:
        return None
    config = getattr(conn, "config", None)
    server = config.get("server") if isinstance(config, dict) else None
    gate = WakeWordGate.from_config(
        server.get("wake_word_gate") if isinstance(server, dict) else None
    )
    try:
        conn._wake_word_gate = gate if gate is not None else _DISABLED
    except AttributeError:
        pass
    return gate
