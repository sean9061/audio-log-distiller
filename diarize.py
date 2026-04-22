#!/usr/bin/env python3
"""
diarize.py — whisper.cpp JSON + 音声ファイルから話者識別付き文字起こしを生成

Usage:
  python3 diarize.py <audio_file> <transcription.json> [options]

Options:
  --speakers N          話者数を指定（省略時は自動検出）
  --hf-token TOKEN      HuggingFace トークン（環境変数 HF_TOKEN でも可）
  --output txt|srt|vtt|json 出力形式（デフォルト: txt）

事前準備:
  pip install pyannote.audio torch
  以下のモデル利用規約に同意（HuggingFace アカウント必要）:
    https://hf.co/pyannote/speaker-diarization-3.1
    https://hf.co/pyannote/segmentation-3.0
"""

import sys
import json
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

# .env ファイルがあれば読み込む（python-dotenv 不要）
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


def convert_to_wav(audio_path: Path, out_wav: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(out_wav), "-loglevel", "error",
        ],
        check=True,
    )


def ts_to_sec(ts: str) -> float:
    """'HH:MM:SS.mmm' or 'HH:MM:SS,mmm' → float seconds"""
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def sec_to_srt_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = round((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def sec_to_vtt_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = round((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def assign_speaker(start: float, end: float, diarization) -> str:
    """最も重複時間が長い話者を返す"""
    best, best_overlap = "SPEAKER_??", 0.0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > best_overlap:
            best_overlap, best = overlap, speaker
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="話者識別付き文字起こしを生成")
    parser.add_argument("audio_file", help="元の音声ファイル")
    parser.add_argument("json_file", help="whisper.cpp の出力 JSON ファイル")
    parser.add_argument("--speakers", type=int, default=None, help="話者数（省略時は自動検出）")
    parser.add_argument("--hf-token", default=None, help="HuggingFace トークン")
    parser.add_argument("--output", choices=["txt", "srt", "vtt", "json"], default="txt", help="出力形式")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    json_path = Path(args.json_file)

    for p in (audio_path, json_path):
        if not p.exists():
            sys.exit(f"Error: {p} が見つかりません")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        sys.exit(
            "Error: HuggingFace トークンが必要です。\n"
            "  --hf-token TOKEN  または  export HF_TOKEN=xxx"
        )

    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        sys.exit(
            "Error: pyannote.audio が未インストールです。\n"
            "  pip install pyannote.audio torch"
        )

    # whisper JSON 読み込み
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("transcription", [])
    if not segments:
        sys.exit("Error: transcription が空です")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "input.wav"

        print(f"音声変換中: {audio_path.name} → WAV (16kHz mono)", flush=True)
        convert_to_wav(audio_path, wav_path)

        print("pyannote モデルをロード中...", flush=True)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        pipeline.to(torch.device(device))
        print(f"デバイス: {device}", flush=True)

        print("話者識別を実行中...", flush=True)
        diarize_kwargs = {}
        if args.speakers:
            diarize_kwargs["num_speakers"] = args.speakers
        output = pipeline(str(wav_path), **diarize_kwargs)
        # pyannote 3.x は DiarizeOutput (dataclass) を返す
        diarization = output.speaker_diarization if hasattr(output, "speaker_diarization") else output

    # whisper セグメントに話者を割り当て
    print("文字起こしと話者情報をマージ中...", flush=True)
    results = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        t = seg.get("timestamps", {})
        start_str = t.get("from", "00:00:00.000")
        end_str = t.get("to", start_str)
        speaker = assign_speaker(ts_to_sec(start_str), ts_to_sec(end_str), diarization)
        results.append({"speaker": speaker, "start": start_str, "end": end_str, "text": text})

    # 出力
    out_base = json_path.with_suffix("")
    fmt = args.output

    if fmt == "txt":
        out_path = Path(str(out_base) + ".diarized.txt")
        lines = []
        prev = None
        for r in results:
            if r["speaker"] != prev:
                lines.append(f"\n[{r['speaker']}]")
                prev = r["speaker"]
            lines.append(f"  [{r['start']}] {r['text']}")
        out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    elif fmt == "srt":
        out_path = Path(str(out_base) + ".diarized.srt")
        blocks = []
        for i, r in enumerate(results, 1):
            start_ts = sec_to_srt_ts(ts_to_sec(r["start"]))
            end_ts = sec_to_srt_ts(ts_to_sec(r["end"]))
            blocks.append(f"{i}\n{start_ts} --> {end_ts}\n[{r['speaker']}] {r['text']}")
        out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

    elif fmt == "vtt":
        out_path = Path(str(out_base) + ".diarized.vtt")
        lines = ["WEBVTT", ""]
        for r in results:
            start_ts = sec_to_vtt_ts(ts_to_sec(r["start"]))
            end_ts = sec_to_vtt_ts(ts_to_sec(r["end"]))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(f"[{r['speaker']}] {r['text']}")
            lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")

    else:  # json
        out_path = Path(str(out_base) + ".diarized.json")
        out_path.write_text(
            json.dumps({"transcription": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"\n完了: {out_path}")


if __name__ == "__main__":
    main()
