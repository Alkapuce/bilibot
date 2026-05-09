"""Test all LLM-related code paths in bilibot."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bilibot.config import load_settings
from bilibot.llm import LLMClient
from bilibot.models import Transcript, TranscriptSegment
from bilibot.extractor import VideoInfo
from bilibot.postprocessor import postprocess_transcript
from bilibot.summarizer import summarize_video, render_basic_notes, split_text

# ── Test helpers ──────────────────────────────────────────────────────────

_passed = 0
_failed = 0

def test(name: str):
    """Decorator-like wrapper for test functions."""
    def wrapper(fn):
        global _passed, _failed
        print(f"\n{'='*70}")
        print(f"  TEST: {name}")
        print(f"{'='*70}")
        try:
            start = time.time()
            fn()
            elapsed = time.time() - start
            _passed += 1
            print(f"  RESULT: PASS ({elapsed:.1f}s)")
        except Exception as e:
            _failed += 1
            print(f"  RESULT: FAIL - {type(e).__name__}: {e}")
    return wrapper

def summary():
    global _passed, _failed
    total = _passed + _failed
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {_passed}/{total} passed, {_failed}/{total} failed")
    print(f"{'='*70}")
    return _failed == 0


# ── Setup ─────────────────────────────────────────────────────────────────

settings = load_settings()
print(f"LLM Base URL: {settings.llm_base_url}")
print(f"LLM Model:    {settings.llm_model}")
print(f"LLM Timeout:  {settings.llm_timeout}s")
print(f"Chunk chars:  {settings.chunk_chars}")
print(f"Subtitle postprocess: {settings.subtitle_postprocess}")
print(f"Postprocess model:    {settings.subtitle_postprocess_model}")

# Mock video info (no B站 API call needed)
info = VideoInfo(
    bvid="BV1xxTest123",
    title="Python 异步编程入门教程",
    author="教学UP主",
    desc="本视频介绍 Python asyncio 的核心概念",
    duration=1234,
    cover="",
    url="https://www.bilibili.com/video/BV1xxTest123",
)

# Short mock transcript (Chinese, ~300 chars - fits single chunk)
short_segments = [
    TranscriptSegment(start=0.0, text="大家好，今天我们来学习 Python 的异步编程。", end=5.0),
    TranscriptSegment(start=5.0, text="异步编程可以让程序同时处理多个任务，提高效率。", end=10.0),
    TranscriptSegment(start=10.0, text="Python 中核心的是 asyncio 库。", end=15.0),
    TranscriptSegment(start=15.0, text="我们来看一个例子，使用 async 和 await 关键字。", end=20.0),
    TranscriptSegment(start=20.0, text="总结一下，异步编程适合 IO 密集型任务。", end=25.0),
]
short_transcript = Transcript(
    source="bilibili_subtitle",
    language="zh",
    segments=short_segments,
)

# Long mock transcript (enough chars for multi-chunk test)
_long_lines = [
    "在这一段视频中，我们将深入探讨机器学习的核心概念和应用场景。",
    "机器学习是人工智能的一个重要分支，它通过数据训练模型来做出预测和决策。",
    "首先我们来了解监督学习。监督学习使用带标签的数据进行训练，",
    "常见的算法包括线性回归、逻辑回归、决策树和随机森林等。",
    "其中线性回归主要用于预测连续值，比如预测房价、股票价格等。",
    "逻辑回归虽然名字里带回归，但实际是一个分类算法，常用于二分类问题。",
    "决策树通过树形结构来做决策，每一个节点代表一个特征的判断条件。",
    "随机森林则是多个决策树的集成，通过投票或平均来提高准确率和防止过拟合。",
]
_long_segments = []
_t = 0.0
for line in _long_lines:
    _long_segments.append(TranscriptSegment(start=_t, text=line, end=_t + 15.0))
    _t += 15.0
long_transcript = Transcript(
    source="whisper_asr",
    language="zh",
    segments=_long_segments,
)


# ── Test 1: LLMClient basic connectivity ──────────────────────────────────

@test("LLMClient 基本连接测试")
def _test_basic_llm():
    client = LLMClient(settings)
    # Non-streaming simple call
    response = client.complete([
        {"role": "user", "content": "请用中文回复：你好，请说 1+1=2。"}
    ])
    print(f"  非流式响应: {response[:120]}")
    assert len(response) > 0, "LLM returned empty response"

    # Streaming call
    response2 = client.complete_stream([
        {"role": "user", "content": "用中文回复一个字：好"}
    ])
    print(f"  流式响应: {response2[:120]}")
    assert len(response2) > 0, "LLM streaming returned empty response"


# ── Test 2: Postprocessor ─────────────────────────────────────────────────

@test("字幕后处理 (postprocess_transcript)")
def _test_postprocess():
    result = postprocess_transcript(info, short_transcript, settings)
    print(f"  后处理完成: postprocessed={result.postprocessed}, model={result.postprocess_model}")
    print(f"  处理前段数: {len(short_transcript.segments)}")
    print(f"  处理后段数: {len(result.segments)}")
    if result.segments:
        print(f"  第一段: {result.segments[0].text[:80]}")
    assert result.postprocessed is True
    assert len(result.segments) == len(short_transcript.segments)
    assert len(result.text) > 0


# ── Test 3: Summarizer single chunk ───────────────────────────────────────

@test("笔记生成 - 单块 (summarize_video, single chunk)")
def _test_summarize_single():
    result = summarize_video(info, short_transcript, settings)
    print(f"  笔记长度: {len(result)} 字符")
    print(f"  笔记前200字符: {result[:200]}...")
    assert len(result) > 0
    assert info.title in result or info.bvid in result, "笔记应包含视频标题或BV号"


# ── Test 4: Summarizer multi-chunk ────────────────────────────────────────

@test("笔记生成 - 多块 (summarize_video, multi-chunk + merge)")
def _test_summarize_multi():
    # Use a very small chunk_chars to force multi-chunk behavior
    small_settings = load_settings(**{"chunk_chars": 200})
    chunks = split_text(long_transcript.text, small_settings.chunk_chars)
    print(f"  字幕字符数: {len(long_transcript.text)}, 分块数: {len(chunks)}")
    assert len(chunks) >= 2, f"Expected >=2 chunks but got {len(chunks)}"

    result = summarize_video(info, long_transcript, small_settings)
    print(f"  笔记长度: {len(result)} 字符")
    print(f"  笔记前200字符: {result[:200]}...")
    assert len(result) > 0
    assert info.title in result or info.bvid in result, "笔记应包含视频标题或BV号"


# ── Test 5: render_basic_notes (no LLM path) ──────────────────────────────

@test("基础笔记生成 (render_basic_notes, 无LLM路径)")
def _test_render_basic():
    result = render_basic_notes(info, short_transcript, "测试说明")
    print(f"  基础笔记长度: {len(result)} 字符")
    assert "未调用 LLM" in result or "测试说明" in result
    assert info.bvid in result


# ── Test 6: Postprocessor with empty transcript ───────────────────────────

@test("字幕后处理 - 空字幕边界情况")
def _test_postprocess_empty():
    empty = Transcript(source="bilibili_subtitle", language="zh", segments=[])
    result = postprocess_transcript(info, empty, settings)
    assert result is empty or result.segments == []
    print("  空字幕正确处理（跳过后处理）")


# ── Test 7: Split text edge cases ─────────────────────────────────────────

@test("split_text 边界情况")
def _test_split_text():
    # Empty
    assert split_text("", 100) == [""]
    # Short
    assert split_text("hello", 100) == ["hello"]
    # Long single line
    chunks = split_text("a" * 500, 100)
    assert len(chunks) == 5, f"Expected 5 chunks, got {len(chunks)}"
    # Normal multiline
    chunks = split_text("a\nb\nc\nd\ne", 4)
    assert len(chunks) > 1, f"Expected multi chunks, got {len(chunks)}"
    print(f"  split_text 各边界情况正常")


# ── Run ───────────────────────────────────────────────────────────────────

print("=" * 70)
print("  BILIBOT LLM FLOWS TEST")
print("=" * 70)

ok = summary()
sys.exit(0 if ok else 1)
