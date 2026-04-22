#!/usr/bin/env python3
"""
meeting_summary.py — whisper.cpp のJSON文字起こしから会議サマリーを生成する
Usage: python3 meeting_summary.py <transcription.json> [--lang ja|en]
"""

import sys
import json
import argparse
from pathlib import Path
import anthropic

SYSTEM_PROMPT = """あなたは会議の文字起こしからプロフェッショナルな会議サマリーを作成するアシスタントです。

以下の形式でMarkdownを出力してください：

# 会議サマリー

## 概要
（会議の目的・全体像を2〜3文で）

## 主な議題と議論内容
（箇条書きで主要なトピックとその内容）

## 決定事項
（会議で決まったことのリスト）

## アクションアイテム
（誰が・何を・いつまでに、のフォーマットで。不明な場合は「担当者未定」）

## 次回に持ち越した課題
（未解決のまま持ち越された項目）

---
情報が不足している項目は「記録なし」と記載してください。
文字起こしに誤りや不明瞭な部分があっても、文脈から最善の解釈をしてください。"""

def load_transcription(json_path: Path) -> tuple[str, bool]:
    """whisper.cpp のJSONから発言テキストを抽出して結合する。
    話者ラベル付き（diarize.py 出力）かどうかも判定して返す。
    Returns: (transcript_text, has_speakers)
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("transcription", [])
    if not segments:
        raise ValueError("transcription キーが見つからないか空です")

    has_speakers = "speaker" in segments[0]
    lines = []

    if has_speakers:
        prev_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "SPEAKER_??")
            text = seg.get("text", "").strip()
            start = seg.get("start", "")
            if not text:
                continue
            if speaker != prev_speaker:
                lines.append(f"\n[{speaker}]")
                prev_speaker = speaker
            lines.append(f"  [{start}] {text}")
    else:
        for seg in segments:
            text = seg.get("text", "").strip()
            if text:
                t = seg.get("timestamps", {})
                start = t.get("from", "")
                lines.append(f"[{start}] {text}" if start else text)

    return "\n".join(lines).strip(), has_speakers


def generate_summary(transcription_text: str, lang: str = "ja", has_speakers: bool = False) -> str:
    """Claude API を使って会議サマリーを生成する（ストリーミング）"""
    client = anthropic.Anthropic()

    speaker_note = (
        "文字起こしには [SPEAKER_XX] 形式の話者ラベルが含まれています。"
        "アクションアイテムや発言の帰属を話者ごとに記載してください。"
        if has_speakers else ""
    )

    user_message = f"""以下は会議の文字起こしです。サマリーを作成してください。{f" {speaker_note}" if speaker_note else ""}

--- 文字起こし開始 ---
{transcription_text}
--- 文字起こし終了 ---"""

    print("Claudeがサマリーを生成中...\n", flush=True)

    result_text = ""
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            result_text += text

    print()  # 最後に改行
    return result_text


def main():
    parser = argparse.ArgumentParser(description="会議文字起こしJSONからサマリーを生成")
    parser.add_argument("json_file", help="whisper.cpp の出力JSONファイル")
    parser.add_argument("--lang", default="ja", choices=["ja", "en"], help="出力言語（デフォルト: ja）")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: ファイルが見つかりません: {json_path}", file=sys.stderr)
        sys.exit(1)
    if json_path.suffix != ".json":
        print(f"Error: JSONファイルを指定してください（.json）", file=sys.stderr)
        sys.exit(1)

    print(f"文字起こしを読み込み中: {json_path}")
    transcription, has_speakers = load_transcription(json_path)
    if has_speakers:
        print("話者ラベル付きデータを検出しました。話者情報をサマリーに反映します。")
    token_estimate = len(transcription) // 3  # 日本語は1文字≒1トークン目安
    print(f"文字数: {len(transcription)}文字（トークン推定: ~{token_estimate}）\n")

    summary = generate_summary(transcription, args.lang, has_speakers)

    # サマリーをファイルに保存
    output_path = json_path.with_suffix(".summary.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n保存完了: {output_path}")


if __name__ == "__main__":
    main()
