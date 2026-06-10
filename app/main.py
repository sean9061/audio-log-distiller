import asyncio
import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import uuid
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

UPLOAD_DIR = Path("/tmp/audio_jobs")
UPLOAD_DIR.mkdir(exist_ok=True)

VALID_MODELS = {"tiny", "small", "medium", "large-v3"}

SUMMARY_PROMPTS: dict[str, str] = {
    "lecture": (
        "以下は大学の講義の音声文字起こしです。日本語で以下の形式でまとめてください。\n"
        "【講義トピック】\n【主要な概念・理論】\n【重要なポイント・結論】\n【キーワード】\n\n"
    ),
    "meeting": (
        "以下は仕事のミーティングの音声文字起こしです。日本語で以下の形式でまとめてください。\n"
        "【議題】\n【決定事項】\n【アクションアイテム（担当者・期限があれば明記）】\n【共有事項・懸念点】\n\n"
    ),
    "casual": (
        "以下は雑談の音声文字起こしです。日本語で話題ごとに簡潔にまとめてください。\n"
        "堅苦しくなく、会話の雰囲気を保ちながら要点を整理してください。\n\n"
    ),
    "briefing": (
        "以下は説明会の音声文字起こしです。日本語で以下の形式でまとめてください。\n"
        "【説明会の目的・概要】\n【主要な説明内容】\n【重要な日程・締め切り・手続き】\n【Q&Aハイライト（あれば）】\n\n"
    ),
    "general": (
        "以下の音声文字起こしを日本語で要約してください。"
        "主なトピック・結論・重要なポイントを簡潔にまとめてください。\n\n"
    ),
}

app = FastAPI(title="Audio Log Distiller")
_executor = ThreadPoolExecutor(max_workers=1)  # one job at a time — GPU is shared
jobs: dict[str, dict] = {}

_model_cache: dict = {}  # {"obj": WhisperModel, "name": str}


def _load_whisper_model(model_size: str):
    import torch
    from faster_whisper import WhisperModel

    if _model_cache.get("name") == model_size:
        return _model_cache["obj"]

    if "obj" in _model_cache:
        del _model_cache["obj"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    _model_cache["obj"] = WhisperModel(model_size, device=device, compute_type=compute)
    _model_cache["name"] = model_size
    return _model_cache["obj"]


def _sec_to_ts(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")


def _summarize_with_ollama(transcript: str, job_id: str, summary_type: str = "general") -> str:
    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

    prefix = SUMMARY_PROMPTS.get(summary_type, SUMMARY_PROMPTS["general"])
    prompt = prefix + transcript

    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": True}
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    accumulated = ""
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw_line in resp:
            line = raw_line.strip()
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            accumulated += chunk.get("response", "")
            jobs[job_id]["partial_summary"] = accumulated
            if chunk.get("done"):
                break

    return accumulated.strip()


def _process_job(
    job_id: str,
    audio_path: str,
    model_size: str,
    language: str,
    do_diarize: bool,
    num_speakers: Optional[int],
    do_summarize: bool,
    summary_type: str,
) -> None:
    whisper_json_path = Path(audio_path + ".json")
    diarized_json_path = Path(audio_path + ".diarized.json")

    try:
        # ── 1. Transcription (stream segments live) ───────────────────────
        jobs[job_id]["status"] = "transcribing"
        jobs[job_id]["partial_transcript"] = ""

        model = _load_whisper_model(model_size)
        lang = None if language == "auto" else language
        seg_gen, _ = model.transcribe(audio_path, language=lang, beam_size=5)

        segments = []
        partial_lines: list[str] = []
        for seg in seg_gen:
            segments.append(seg)
            partial_lines.append(seg.text.strip())
            jobs[job_id]["partial_transcript"] = "\n".join(partial_lines)

        plain_transcript = "\n".join(seg.text.strip() for seg in segments)

        whisper_json = {
            "transcription": [
                {
                    "timestamps": {
                        "from": _sec_to_ts(seg.start),
                        "to": _sec_to_ts(seg.end),
                    },
                    "text": " " + seg.text.strip(),
                }
                for seg in segments
            ]
        }
        whisper_json_path.write_text(
            json.dumps(whisper_json, ensure_ascii=False), encoding="utf-8"
        )

        result: dict = {"transcript": plain_transcript}

        # ── 2. Start Ollama summarization in background (parallel with diarization) ──
        summary_slot: dict = {"value": None, "error": None}
        ollama_thread: threading.Thread | None = None
        if do_summarize and plain_transcript.strip():
            jobs[job_id]["partial_summary"] = ""

            def _run_summary() -> None:
                try:
                    summary_slot["value"] = _summarize_with_ollama(plain_transcript, job_id, summary_type)
                except Exception as e:
                    summary_slot["error"] = str(e)[:300]

            ollama_thread = threading.Thread(target=_run_summary, daemon=True)
            ollama_thread.start()

        # ── 3. Speaker diarization ────────────────────────────────────────
        if do_diarize:
            jobs[job_id]["status"] = "diarizing"

            diarize_script = Path(__file__).parent.parent / "diarize.py"
            cmd = [
                sys.executable,
                str(diarize_script),
                audio_path,
                str(whisper_json_path),
                "--output",
                "json",
            ]
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                cmd += ["--hf-token", hf_token]
            if num_speakers:
                cmd += ["--speakers", str(num_speakers)]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

            if proc.returncode == 0 and diarized_json_path.exists():
                diarized = json.loads(diarized_json_path.read_text(encoding="utf-8"))
                lines: list[str] = []
                prev_speaker = None
                for seg in diarized.get("transcription", []):
                    spk = seg.get("speaker", "SPEAKER_??")
                    text = seg.get("text", "").strip()
                    if not text:
                        continue
                    if spk != prev_speaker:
                        lines.append(f"\n[{spk}]")
                        prev_speaker = spk
                    start = seg.get("start", "")
                    lines.append(f"  [{start}] {text}" if start else f"  {text}")
                result["diarized_transcript"] = "\n".join(lines).strip()
            else:
                result["diarize_error"] = (proc.stderr or "diarize.py failed")[:500]

        # ── 4. Wait for Ollama summary ────────────────────────────────────
        if ollama_thread is not None:
            if jobs[job_id]["status"] != "diarizing":
                jobs[job_id]["status"] = "summarizing"
            ollama_thread.join()
            if summary_slot["value"] is not None:
                result["summary"] = summary_slot["value"]
            elif summary_slot["error"]:
                result["summary_error"] = summary_slot["error"]

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        for p in [Path(audio_path), whisper_json_path, diarized_json_path]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    model: str = Form("small"),
    language: str = Form("ja"),
    diarize: str = Form("false"),
    speakers: str = Form(""),
    summarize: str = Form("true"),
    summary_type: str = Form("general"),
):
    if model not in VALID_MODELS:
        raise HTTPException(400, f"model must be one of {VALID_MODELS}")

    job_id = str(uuid.uuid4())
    suffix = Path(file.filename or "audio").suffix or ".mp3"
    audio_path = UPLOAD_DIR / f"{job_id}{suffix}"

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "result": None,
        "error": None,
    }

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _process_job,
        job_id,
        str(audio_path),
        model,
        language,
        diarize.lower() == "true",
        int(speakers) if speakers.strip().isdigit() else None,
        summarize.lower() == "true",
        summary_type if summary_type in SUMMARY_PROMPTS else "general",
    )

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")
