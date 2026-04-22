#!/usr/bin/env bash
# transcribe.sh — whisper.cpp wrapper that supports .m4a and other formats
# Usage: ./transcribe.sh [options] <audio_file>
#
# Options:
#   -m, --model      Model to use: tiny/small/large-v3 (default: small)
#   -l, --lang       Language code, e.g. ja, en, auto (default: auto)
#   -o, --output     Output format: txt/srt/vtt/json (default: txt)
#   -t, --threads    Number of threads (default: 4)
#   -d, --diarize    Enable speaker diarization (requires pyannote.audio + HF_TOKEN)
#   -s, --speakers N Number of speakers (optional, used with --diarize)
#   -h, --help       Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# .env ファイルがあれば読み込む
[[ -f "$SCRIPT_DIR/.env" ]] && set -a && source "$SCRIPT_DIR/.env" && set +a

WHISPER_BIN="${WHISPER_BIN:-$SCRIPT_DIR/build/bin/whisper-cli}"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/models}"
PYTHON="${PYTHON:-python3}"

# Defaults
MODEL="small"
LANG="auto"
OUTPUT_FORMAT="txt"
THREADS=4
CHUNK_SEC=270  # 4.5分チャンク（5分で破綻するので余裕を持たせる）
DIARIZE=false
SPEAKERS=""

usage() {
  sed -n '2,10p' "$0" | sed 's/^# //'
  exit 0
}

die() { echo "Error: $*" >&2; exit 1; }

# Parse arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)    MODEL="$2";         shift 2 ;;
    -l|--lang)     LANG="$2";          shift 2 ;;
    -o|--output)   OUTPUT_FORMAT="$2"; shift 2 ;;
    -t|--threads)  THREADS="$2";       shift 2 ;;
    -d|--diarize)  DIARIZE=true;       shift   ;;
    -s|--speakers) SPEAKERS="$2";      shift 2 ;;
    -h|--help)     usage ;;
    -*) die "Unknown option: $1" ;;
    *)  POSITIONAL+=("$1"); shift ;;
  esac
done

[[ ${#POSITIONAL[@]} -eq 0 ]] && { echo "Usage: $0 [options] <audio_file>"; exit 1; }
INPUT_FILE="${POSITIONAL[0]}"
[[ -f "$INPUT_FILE" ]] || die "File not found: $INPUT_FILE"

# --diarize 時は whisper を JSON モードで実行し、後で指定フォーマットに変換する
FINAL_FORMAT="$OUTPUT_FORMAT"
if [[ "$DIARIZE" == true ]]; then
  OUTPUT_FORMAT="json"
fi

# Resolve model path
MODEL_FILE="$MODELS_DIR/ggml-${MODEL}.bin"
[[ -f "$MODEL_FILE" ]] || die "Model not found: $MODEL_FILE\nRun: bash models/download-ggml-model.sh $MODEL"

# Check whisper binary
[[ -x "$WHISPER_BIN" ]] || die "whisper-cli not found. Run: cmake --build build --config Release"

# File paths
EXT="${INPUT_FILE##*.}"
EXT_LOWER="$(echo "$EXT" | tr '[:upper:]' '[:lower:]')"
INPUT_DIR="$(dirname "$INPUT_FILE")"
BASENAME="$(basename "$INPUT_FILE" ".$EXT")"
OUTPUT_PATH="$INPUT_DIR/$BASENAME"

# Temp dir (cleaned up on exit)
WORK_DIR="${INPUT_DIR}/.whisper_$$"
mkdir -p "$WORK_DIR"
trap "rm -rf '$WORK_DIR'" EXIT

# Convert to 16kHz mono WAV
WORK_WAV="${WORK_DIR}/input.wav"
if [[ "$EXT_LOWER" != "wav" ]]; then
  echo "Converting $EXT_LOWER → WAV (16kHz mono)..."
fi
ffmpeg -y -i "$INPUT_FILE" -ar 16000 -ac 1 -c:a pcm_s16le "$WORK_WAV" -loglevel error

# Whisper output format flags
case "$OUTPUT_FORMAT" in
  txt)  FMT_FLAG="--output-txt" ;;
  srt)  FMT_FLAG="--output-srt" ;;
  vtt)  FMT_FLAG="--output-vtt" ;;
  json) FMT_FLAG="--output-json" ;;
  *)    die "Unknown output format: $OUTPUT_FORMAT (use txt/srt/vtt/json)" ;;
esac

# Language flag
LANG_FLAG=""
[[ "$LANG" != "auto" ]] && LANG_FLAG="-l $LANG"

# Whisper実行関数（チャンクごとに呼ぶ）
run_whisper() {
  local in_wav="$1"
  local out_prefix="$2"
  "$WHISPER_BIN" \
    -m "$MODEL_FILE" \
    -f "$in_wav" \
    -t "$THREADS" \
    $LANG_FLAG \
    $FMT_FLAG \
    --output-file "$out_prefix" \
    --max-context 0 \
    --entropy-thold 2.8 \
    --logprob-thold -1.0 \
    --no-fallback \
    --print-progress
}

# 音声の長さを取得
TOTAL_SEC=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$WORK_WAV")
TOTAL_INT=$(echo "$TOTAL_SEC" | cut -d. -f1)

echo "Model   : $MODEL"
echo "Language: ${LANG}"
echo "Duration: $((TOTAL_INT / 60))m $((TOTAL_INT % 60))s"
if [[ "$DIARIZE" == true ]]; then
  echo "Output  : $OUTPUT_PATH.diarized.$FINAL_FORMAT (with speaker labels)"
else
  echo "Output  : $OUTPUT_PATH.$OUTPUT_FORMAT"
fi
echo "---"

if [[ "$TOTAL_INT" -le "$CHUNK_SEC" ]]; then
  # 短いファイルはそのまま処理
  run_whisper "$WORK_WAV" "$OUTPUT_PATH"
else
  # 長いファイルはチャンク分割して処理
  CHUNK_COUNT=$(( (TOTAL_INT + CHUNK_SEC - 1) / CHUNK_SEC ))
  echo "Long audio detected: splitting into $CHUNK_COUNT chunks of ${CHUNK_SEC}s..."

  i=0
  START=0
  while [[ "$START" -lt "$TOTAL_INT" ]]; do
    PAD=$(printf '%03d' $i)
    CHUNK_WAV="${WORK_DIR}/chunk_${PAD}.wav"
    CHUNK_OUT="${WORK_DIR}/chunk_${PAD}"
    END_MIN=$(( (START + CHUNK_SEC) / 60 ))
    START_MIN=$(( START / 60 ))

    echo "  Chunk $((i+1))/$CHUNK_COUNT (${START_MIN}min → ${END_MIN}min)..."
    ffmpeg -y -ss "$START" -t "$CHUNK_SEC" -i "$WORK_WAV" \
      -ar 16000 -ac 1 -c:a pcm_s16le "$CHUNK_WAV" -loglevel error

    run_whisper "$CHUNK_WAV" "$CHUNK_OUT"

    START=$(( START + CHUNK_SEC ))
    i=$(( i + 1 ))
  done

  echo "Merging $CHUNK_COUNT chunks..."

  case "$OUTPUT_FORMAT" in
    txt)
      cat "${WORK_DIR}"/chunk_*.txt > "${OUTPUT_PATH}.txt"
      ;;
    srt)
      "$PYTHON" - "${WORK_DIR}" "$CHUNK_SEC" "${OUTPUT_PATH}.srt" <<'PYEOF'
import sys, re, os, glob

work_dir, chunk_sec, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def ts_to_ms(ts):
    h, m, s_ms = ts.split(':')
    s, ms = s_ms.split(',')
    return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)

def ms_to_ts(ms):
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

chunks = sorted(glob.glob(os.path.join(work_dir, 'chunk_*.srt')))
all_blocks = []
seq = 1
for idx, path in enumerate(chunks):
    offset_ms = idx * chunk_sec * 1000
    content = open(path).read().strip()
    if not content:
        continue
    for block in re.split(r'\n{2,}', content):
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        ts_line = lines[1]
        m = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', ts_line)
        if not m:
            continue
        new_ts = f"{ms_to_ts(ts_to_ms(m.group(1))+offset_ms)} --> {ms_to_ts(ts_to_ms(m.group(2))+offset_ms)}"
        text = '\n'.join(lines[2:])
        all_blocks.append(f"{seq}\n{new_ts}\n{text}")
        seq += 1

with open(out_path, 'w') as f:
    f.write('\n\n'.join(all_blocks) + '\n')
PYEOF
      ;;
    vtt)
      "$PYTHON" - "${WORK_DIR}" "$CHUNK_SEC" "${OUTPUT_PATH}.vtt" <<'PYEOF'
import sys, re, os, glob

work_dir, chunk_sec, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def ts_to_ms(ts):
    h, m, s_ms = ts.split(':')
    s, ms = s_ms.split('.')
    return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)

def ms_to_ts(ms):
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

chunks = sorted(glob.glob(os.path.join(work_dir, 'chunk_*.vtt')))
lines_out = ['WEBVTT', '']
for idx, path in enumerate(chunks):
    offset_ms = idx * chunk_sec * 1000
    content = open(path).read()
    for line in content.splitlines():
        m = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', line)
        if m:
            lines_out.append(f"{ms_to_ts(ts_to_ms(m.group(1))+offset_ms)} --> {ms_to_ts(ts_to_ms(m.group(2))+offset_ms)}")
        elif line.strip() == 'WEBVTT':
            continue
        else:
            lines_out.append(line)

with open(out_path, 'w') as f:
    f.write('\n'.join(lines_out) + '\n')
PYEOF
      ;;
    json)
      "$PYTHON" - "${WORK_DIR}" "$CHUNK_SEC" "${OUTPUT_PATH}.json" <<'PYEOF'
import sys, json, glob, os

work_dir, chunk_sec, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def ts_to_ms(ts):
    ts = ts.replace(',', '.')
    h, m, s_ms = ts.split(':')
    s, ms = s_ms.split('.')
    return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)

def ms_to_ts(ms):
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

chunks = sorted(glob.glob(os.path.join(work_dir, 'chunk_*.json')))
result = []
for idx, p in enumerate(chunks):
    offset_ms = idx * chunk_sec * 1000
    d = json.load(open(p))
    for seg in d.get('transcription', []):
        t = seg.get('timestamps', {})
        if 'from' in t:
            t['from'] = ms_to_ts(ts_to_ms(t['from']) + offset_ms)
        if 'to' in t:
            t['to'] = ms_to_ts(ts_to_ms(t['to']) + offset_ms)
        result.append(seg)

json.dump({'transcription': result}, open(out_path, 'w'), ensure_ascii=False, indent=2)
PYEOF
      ;;
  esac
fi

echo "---"
echo "Done: $OUTPUT_PATH.$OUTPUT_FORMAT"

# 話者識別（--diarize が指定された場合に実行）
if [[ "$DIARIZE" == true ]]; then
  DIARIZE_SCRIPT="$SCRIPT_DIR/diarize.py"
  [[ -f "$DIARIZE_SCRIPT" ]] || die "diarize.py が見つかりません: $DIARIZE_SCRIPT"
  echo ""
  echo "話者識別を開始..."
  DIARIZE_ARGS=("$INPUT_FILE" "$OUTPUT_PATH.json" --output "$FINAL_FORMAT")
  [[ -n "$SPEAKERS" ]] && DIARIZE_ARGS+=(--speakers "$SPEAKERS")
  "$PYTHON" "$DIARIZE_SCRIPT" "${DIARIZE_ARGS[@]}"
fi
