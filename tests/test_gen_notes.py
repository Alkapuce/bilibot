from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bilibot.config import Settings
from bilibot.gen_notes import run


class GenNotesTests(unittest.TestCase):
    def test_run_reads_existing_artifacts_with_pathlib_paths(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            bvid = "BV1xxTest123"
            data_dir = output_dir / bvid
            data_dir.mkdir()
            (data_dir / "测试视频_信息.json").write_text(
                json.dumps(
                    {
                        "bvid": bvid,
                        "title": "测试视频",
                        "author": "UP主",
                        "duration": 12,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (data_dir / "测试视频_字幕.json").write_text(
                json.dumps(
                    {
                        "source": "bilibili_subtitle",
                        "language": "zh",
                        "segments": [{"start": 0.0, "end": 1.0, "text": "你好"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch("bilibot.gen_notes.summarize_video", return_value="# 测试笔记"):
                path = run(bvid, Settings(output_dir=output_dir))

            self.assertEqual(path, data_dir / "测试视频_笔记.md")
            self.assertEqual(path.read_text(encoding="utf-8"), "# 测试笔记\n")


if __name__ == "__main__":
    unittest.main()
