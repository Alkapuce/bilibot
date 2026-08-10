from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bilibot.cli import _normalize_argv, build_parser, main
from bilibot.models import Transcript, TranscriptSegment
from bilibot.qwen_asr_backend import _resolve_language
from bilibot.storage import save_local_transcript_artifacts


class LocalAsrCliTests(unittest.TestCase):
    def test_asr_command_accepts_local_audio_and_asr_options(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "asr",
                "meeting.m4a",
                "--asr-backend",
                "whisper",
                "--asr-preset",
                "best",
                "--output-dir",
                "out",
                "--json",
            ]
        )

        self.assertEqual(args.command, "asr")
        self.assertEqual(args.audio, ["meeting.m4a"])
        self.assertEqual(args.asr_backend, "whisper")
        self.assertEqual(args.asr_preset, "best")
        self.assertEqual(args.output_dir, "out")
        self.assertTrue(args.json)

    def test_asr_command_is_not_normalized_to_summarize(self) -> None:
        self.assertEqual(_normalize_argv(["asr", "meeting.m4a"]), ["asr", "meeting.m4a"])

    def test_asr_command_routes_local_audio_without_real_model(self) -> None:
        transcript = Transcript(
            source="mock",
            language="zh",
            segments=[TranscriptSegment(start=0.0, end=1.0, text="测试。")],
        )

        with TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "meeting.m4a"
            audio.write_bytes(b"fake audio")

            with patch("bilibot.cli_cmds._transcribe_local_audio", return_value=transcript) as transcribe:
                rc = main(["asr", str(audio), "--output-dir", tmpdir, "--quiet", "--json"])

            self.assertEqual(rc, 0)
            transcribe.assert_called_once()
            self.assertTrue((Path(tmpdir) / "local_asr" / "meeting" / "transcript.json").is_file())


class LocalAsrStorageTests(unittest.TestCase):
    def test_save_local_transcript_artifacts_uses_audio_stem_directory(self) -> None:
        transcript = Transcript(
            source="whisper/large-v3",
            language="zh",
            segments=[TranscriptSegment(start=0.0, end=1.5, text="你好。")],
        )

        with TemporaryDirectory() as tmpdir:
            paths = save_local_transcript_artifacts(
                Path(tmpdir),
                Path("2026年05月27日 20点48分.m4a"),
                transcript,
            )

            self.assertEqual(paths["transcript_json"].parent.name, "2026年05月27日 20点48分")
            self.assertTrue(paths["transcript_json"].is_file())
            self.assertTrue(paths["transcript_md"].is_file())
            self.assertTrue(paths["captions_txt"].is_file())
            self.assertIn("你好。", paths["captions_txt"].read_text(encoding="utf-8"))


class QwenAsrBackendTests(unittest.TestCase):
    def test_resolve_language_maps_common_language_codes(self) -> None:
        self.assertEqual(_resolve_language("zh"), "Chinese")
        self.assertEqual(_resolve_language("en"), "English")
        self.assertIsNone(_resolve_language("auto"))


if __name__ == "__main__":
    unittest.main()
