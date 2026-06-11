import asyncio
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

UPLOAD_DIR = Path("/tmp/audio_jobs")
UPLOAD_DIR.mkdir(exist_ok=True)

# 履歴の永続化先（compose で /data をボリュームにマウント）。ジョブ完了時に
# 結果をここへ保存し、ページを閉じても/再起動しても過去データを参照できる。
HISTORY_DIR = Path(os.environ.get("HISTORY_DIR", "/data/history"))
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

_ID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")  # uuid4 形式のみ許可（パストラバーサル防止）

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


def _history_path(job_id: str) -> Path:
    return HISTORY_DIR / f"{job_id}.json"


def _persist_history(job_id: str) -> None:
    """完了/失敗したジョブをディスクへ保存する（履歴として永続化）。"""
    job = jobs.get(job_id)
    if not job:
        return
    record = {
        "id": job_id,
        "filename": job.get("filename"),
        "created_at": job.get("created_at"),
        "finished_at": time.time(),
        "status": job.get("status"),
        "model": job.get("model"),
        "language": job.get("language"),
        "diarize": job.get("diarize"),
        "summary_type": job.get("summary_type"),
        "ollama_model": job.get("ollama_model"),
        "result": job.get("result"),
        "error": job.get("error"),
    }
    try:
        tmp = _history_path(job_id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_history_path(job_id))  # 原子的に差し替え
    except Exception:
        pass


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


def _summarize_with_ollama(
    transcript: str, job_id: str, summary_type: str = "general", ollama_model: str = ""
) -> str:
    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    model = ollama_model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

    prefix = SUMMARY_PROMPTS.get(summary_type, SUMMARY_PROMPTS["general"])
    prompt = prefix + transcript

    # コンテキスト長を入力サイズに合わせて動的に確保する。
    # Ollama のデフォルト num_ctx=4096 では長い文字起こしがコンテキスト枠を
    # 使い切り、(1) 先頭の指示文が切り捨てられ (2) 生成余地が残らず1トークンで
    # 打ち切られる（要約が1文字しか出ない）。入力＋出力が収まる長さを確保する。
    # 日本語は概ね 0.6 token/文字。num_predict 分の生成余地も上乗せする。
    NUM_PREDICT = 2048
    est_prompt_tokens = int(len(prompt) * 0.6) + 256
    need = est_prompt_tokens + NUM_PREDICT
    num_ctx = min(32768, max(8192, ((need + 4095) // 4096) * 4096))

    # think を無効化: qwen3 系の thinking モデルは要約タスクでも長大な思考を
    # 出力し、本文(response)が出る前に出力上限へ達して response が空のまま
    # 完了することがある（→ 要約が表示されない）。思考を切り本文を直接出させる。
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "options": {"num_ctx": num_ctx, "num_predict": NUM_PREDICT},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    accumulated = ""
    thinking = ""
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw_line in resp:
            line = raw_line.strip()
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            accumulated += chunk.get("response", "")
            thinking += chunk.get("thinking") or ""
            jobs[job_id]["partial_summary"] = accumulated
            if chunk.get("done"):
                break

    # 万一 response が空（モデルが think:false を無視した等）でも思考内容を返す
    return accumulated.strip() or thinking.strip()


def _process_job(
    job_id: str,
    audio_path: str,
    model_size: str,
    language: str,
    do_diarize: bool,
    num_speakers: Optional[int],
    do_summarize: bool,
    summary_type: str,
    ollama_model: str,
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
                    summary_slot["value"] = _summarize_with_ollama(plain_transcript, job_id, summary_type, ollama_model)
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
        # 終了状態（成功/失敗いずれも）を履歴に保存。クライアントが切断していても
        # 処理は executor 上で継続しており、ここで確実に永続化される。
        if jobs.get(job_id, {}).get("status") in ("done", "error"):
            _persist_history(job_id)


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    model: str = Form("small"),
    language: str = Form("ja"),
    diarize: str = Form("false"),
    speakers: str = Form(""),
    summarize: str = Form("true"),
    summary_type: str = Form("general"),
    ollama_model: str = Form(""),
):
    if model not in VALID_MODELS:
        raise HTTPException(400, f"model must be one of {VALID_MODELS}")

    job_id = str(uuid.uuid4())
    suffix = Path(file.filename or "audio").suffix or ".mp3"
    audio_path = UPLOAD_DIR / f"{job_id}{suffix}"

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    do_diarize = diarize.lower() == "true"
    summary_type = summary_type if summary_type in SUMMARY_PROMPTS else "general"
    ollama_model = ollama_model.strip()

    jobs[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "result": None,
        "error": None,
        "created_at": time.time(),
        "model": model,
        "language": language,
        "diarize": do_diarize,
        "summary_type": summary_type,
        "ollama_model": ollama_model,
    }

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor,
        _process_job,
        job_id,
        str(audio_path),
        model,
        language,
        do_diarize,
        int(speakers) if speakers.strip().isdigit() else None,
        summarize.lower() == "true",
        summary_type,
        ollama_model,
    )

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/history")
async def list_history():
    """保存済み履歴の一覧（メタ情報のみ・新しい順）。"""
    items = []
    for p in HISTORY_DIR.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        result = rec.get("result") or {}
        items.append(
            {
                "id": rec.get("id"),
                "filename": rec.get("filename"),
                "created_at": rec.get("created_at"),
                "finished_at": rec.get("finished_at"),
                "status": rec.get("status"),
                "model": rec.get("model"),
                "language": rec.get("language"),
                "summary_type": rec.get("summary_type"),
                "has_summary": bool(result.get("summary")),
                "has_diarized": bool(result.get("diarized_transcript")),
                "preview": (
                    result.get("summary")
                    or result.get("transcript")
                    or rec.get("error")
                    or ""
                )[:120],
            }
        )
    items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return {"history": items}


@app.get("/api/history/{job_id}")
async def get_history(job_id: str):
    """履歴の詳細。永続化済みファイルを優先し、なければ実行中ジョブを返す。"""
    if not _ID_RE.match(job_id):
        raise HTTPException(400, "Invalid id")
    p = _history_path(job_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            raise HTTPException(500, "Failed to read history")
    job = jobs.get(job_id)
    if job:
        return {"id": job_id, **job}
    raise HTTPException(404, "Not found")


@app.delete("/api/history/{job_id}")
async def delete_history(job_id: str):
    if not _ID_RE.match(job_id):
        raise HTTPException(400, "Invalid id")
    _history_path(job_id).unlink(missing_ok=True)
    jobs.pop(job_id, None)
    return {"ok": True}


@app.get("/api/models")
async def list_models():
    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", [])]
    except Exception:
        models = []
    return {"models": models}


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")
