"""One-shot: generate notes from existing transcript."""
import json
import sys
from glob import glob
from openai import OpenAI

bvid = sys.argv[1] if len(sys.argv) > 1 else "BV1YkR1BXEow"
data_dir = f"data/{bvid}"

# Find files by suffix pattern (filenames now include video title prefix)
transcript_files = sorted(glob(f"{data_dir}/*_字幕.json"))
meta_files = sorted(glob(f"{data_dir}/*_信息.json"))
title_files = sorted(glob(f"{data_dir}/*_字幕.txt"))  # title prefix same across files

if not transcript_files or not meta_files:
    print(f"错误: 在 {data_dir}/ 中未找到 *_字幕.json 或 *_信息.json")
    sys.exit(1)

with open(transcript_files[-1]) as f:   # use latest if multiple
    transcript = json.load(f)
with open(meta_files[-1]) as f:
    meta = json.load(f)

# Derive title prefix from filename for notes output
prefix = "untitled"
if title_files:
    prefix = title_files[-1].rsplit("_字幕.txt", 1)[0].split("/")[-1]

segments = transcript.get("segments", [])
text = "\n".join(s.get("text", "") for s in segments)
print(f"字幕: {len(text)} 字符, {len(segments)} 段")

client = OpenAI(base_url="http://localhost:5001/v1", api_key="wyyxhxbc", timeout=300)

system_prompt = (
    "你是一个严谨的视频学习笔记助手。你的任务是根据视频信息和字幕生成中文笔记。"
    "要求：只基于给定内容总结，不编造未出现的信息。保留重要时间点，方便用户回看。"
    '输出结构清晰的 Markdown。如果信息不足，明确写"字幕中未提供"。'
)

prompt = f"""请根据下面的视频信息和完整字幕生成一份完整中文 Markdown 笔记。

输出结构：
# {meta['title']}
## 视频信息
## 一句话总结
## 核心内容
## 时间线
## 关键观点
## 术语和概念
## 可行动建议
## 仍需核实

视频信息：
标题：{meta['title']}
BV号：{meta['bvid']}
作者：{meta['author']}
时长：{meta['duration']}秒
链接：{meta['url']}

完整字幕：
{text}
"""

print("调用 LLM 中...")
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ],
    temperature=0.3,
    max_tokens=4096,
)
notes = response.choices[0].message.content

notes_path = f"{data_dir}/{prefix}_笔记.md"
with open(notes_path, "w") as f:
    f.write(notes)
print(f"笔记已保存到 {notes_path} ({len(notes)} 字符)")
