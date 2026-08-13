from __future__ import annotations

import inspect
import unittest

from bilibot.extractor import VideoInfo
from bilibot.models import Transcript, TranscriptSegment
from bilibot.postprocessor import _parse_json_array, postprocess_transcript
from bilibot.summarizer import (
    SYSTEM_PROMPT,
    _merge_notes,
    _summarize_chunk,
    _summarize_single,
    effective_summary_chunk_chars,
    render_basic_notes,
    split_text,
)
from bilibot.config import Settings


class SummarizerTests(unittest.TestCase):
    def test_system_prompt_is_general_purpose(self) -> None:
        self.assertNotIn("CS231n", SYSTEM_PROMPT)
        self.assertIn("视频", SYSTEM_PROMPT)
        self.assertIn("易读摘要", SYSTEM_PROMPT)
        self.assertIn("不是套固定模板", SYSTEM_PROMPT)
        self.assertIn("中等详细摘要", SYSTEM_PROMPT)
        self.assertIn("控制标题层级", SYSTEM_PROMPT)
        self.assertIn("避免大段文字墙", SYSTEM_PROMPT)

    def test_summary_prompts_are_adaptive_not_rigid_templates(self) -> None:
        prompt_source = "\n".join(
            inspect.getsource(fn)
            for fn in (_summarize_single, _summarize_chunk, _merge_notes)
        )

        self.assertIn("自行设计输出结构", prompt_source)
        self.assertIn("不要套固定模板", prompt_source)
        self.assertIn("不要强制输出术语表、行动建议、核实清单、完整时间线或固定观点列表", prompt_source)
        self.assertIn("约 2500-5000 个中文字符", prompt_source)
        self.assertIn("4-8 个二级标题", prompt_source)
        self.assertIn("3-6 个要点或短段落", prompt_source)
        self.assertNotIn("输出结构必须严格包含", prompt_source)
        self.assertNotIn("## 术语和概念", prompt_source)
        self.assertNotIn("## 可行动建议", prompt_source)
        self.assertNotIn("## 仍需核实", prompt_source)

    def test_render_basic_notes_uses_video_metadata(self) -> None:
        info = VideoInfo(
            bvid="BV1xxTest123",
            title="Python 异步编程入门教程",
            author="教学UP主",
            desc="本视频介绍 Python asyncio 的核心概念",
            duration=1234,
            cover="",
            url="https://www.bilibili.com/video/BV1xxTest123",
        )
        transcript = Transcript(
            source="bilibili_subtitle",
            language="zh",
            segments=[TranscriptSegment(start=0.0, end=5.0, text="大家好。")],
        )

        notes = render_basic_notes(info, transcript, "测试说明")

        self.assertIn("# Python 异步编程入门教程", notes)
        self.assertIn("BV1xxTest123", notes)
        self.assertIn("测试说明", notes)

    def test_split_text_handles_empty_short_and_long_inputs(self) -> None:
        self.assertEqual(split_text("", 100), [""])
        self.assertEqual(split_text("hello", 100), ["hello"])
        self.assertEqual(split_text("a" * 500, 100), ["a" * 100] * 5)
        self.assertGreater(len(split_text("a\nb\nc\nd\ne", 4)), 1)

    def test_effective_summary_chunk_chars_caps_long_requests(self) -> None:
        self.assertEqual(
            effective_summary_chunk_chars(Settings(chunk_chars=200000, summary_max_single_chunk_chars=60000)),
            60000,
        )
        self.assertEqual(
            effective_summary_chunk_chars(Settings(chunk_chars=8000, summary_max_single_chunk_chars=60000)),
            8000,
        )
        self.assertEqual(
            effective_summary_chunk_chars(Settings(chunk_chars=200000, summary_max_single_chunk_chars=0)),
            200000,
        )


class PostprocessorTests(unittest.TestCase):
    def test_parse_json_array_accepts_markdown_fences(self) -> None:
        parsed = _parse_json_array(
            '```json\n[{"index": 0, "text": "你好。"}, {"index": 1, "text": "世界。"}]\n```'
        )

        self.assertEqual(
            parsed,
            [{"index": 0, "text": "你好。"}, {"index": 1, "text": "世界。"}],
        )

    def test_postprocess_empty_transcript_skips_llm(self) -> None:
        info = VideoInfo(
            bvid="BV1xxTest123",
            title="空字幕测试",
            author="",
            desc="",
            duration=0,
            cover="",
            url="https://www.bilibili.com/video/BV1xxTest123",
        )
        transcript = Transcript(source="bilibili_subtitle", language="zh", segments=[])

        result = postprocess_transcript(info, transcript, settings=object())  # type: ignore[arg-type]

        self.assertIs(result, transcript)


if __name__ == "__main__":
    unittest.main()
