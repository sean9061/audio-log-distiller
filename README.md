# audio-log-distiller

音声ファイルを文字起こし → 話者識別 → LLM サマリーに変換するパイプラインツール。

## 機能

- 音声文字起こし（[whisper.cpp](https://github.com/ggml-org/whisper.cpp) 使用、GPU 対応）
- 話者識別（pyannote.audio、CUDA / Apple Silicon MPS 対応）
- 会議サマリー生成（Claude API）
- 長時間音声のチャンク分割処理

## 対応フォーマット

`.m4a`, `.mp3`, `.mp4`, `.flac`, `.ogg`, `.wav` など ffmpeg が対応するすべての形式

## セットアップ

### 1. whisper.cpp をビルド

```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp

# Linux (CUDA)
cmake -B build -DWHISPER_CUDA=ON
cmake --build build --config Release -j$(nproc)

# macOS (Metal)
cmake -B build -DWHISPER_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)

# モデルをダウンロード
bash models/download-ggml-model.sh large-v3   # 2.95GB（推奨）
bash models/download-ggml-model.sh small       # 487MB（軽量）
```

### 2. このリポジトリをクローン

```bash
git clone https://github.com/sean9061/audio-log-distiller
cd audio-log-distiller
pip install -r requirements.txt
```

### 3. 環境設定

```bash
cp .env.example .env
# .env を編集して各パスとトークンを設定
```

### 4. 話者識別モデルの利用規約に同意（初回のみ）

HuggingFace アカウントで以下のページにアクセスし「Agree」を押す：
- https://hf.co/pyannote/speaker-diarization-3.1
- https://hf.co/pyannote/segmentation-3.0
- https://hf.co/pyannote/speaker-diarization-community-1

## 使い方

```bash
# 基本
./transcribe.sh audio.m4a

# 日本語 + 高精度モデル
./transcribe.sh -m large-v3 -l ja audio.m4a

# 話者識別付き（SRT 出力）
./transcribe.sh -m large-v3 -l ja -o srt --diarize --speakers 2 audio.m4a

# 会議サマリー生成
python3 meeting_summary.py audio.json
python3 meeting_summary.py audio.diarized.json  # 話者ラベル付き
```

### オプション一覧

| オプション | 説明 | デフォルト |
|---|---|---|
| `-m`, `--model` | モデル: `tiny` / `small` / `large-v3` | `small` |
| `-l`, `--lang` | 言語: `ja`, `en`, `auto` など | `auto` |
| `-o`, `--output` | 出力形式: `txt` / `srt` / `vtt` / `json` | `txt` |
| `-t`, `--threads` | スレッド数 | `4` |
| `-d`, `--diarize` | 話者識別を有効化 | オフ |
| `-s`, `--speakers` | 話者数を指定（`--diarize` と併用） | 自動検出 |

## 依存関係

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) — 別途ビルドが必要
- ffmpeg (`apt install ffmpeg` / `brew install ffmpeg`)
- Python 3.10 以上
- `pip install -r requirements.txt`
