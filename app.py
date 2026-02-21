import os
import base64
import uuid
import json
import cv2
import numpy as np
import speech_recognition as sr
import time
import asyncio
from datetime import datetime
from threading import Lock
import shutil
import tempfile

from dotenv import load_dotenv
load_dotenv()

# FastAPI imports
from fastapi import FastAPI, Request, File, UploadFile, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import socketio

# === custom modules (keep these in your project) ===
from src import llm        # your LLM helper (llm.py)
from src import model      # your ATS model (model.py)
from src.emotion_analyzer import EmotionAnalyzer
from src.eye_tracking import EyeTracker

# === JOBS SCRAPER imports ===
from sqlalchemy.orm import Session as _Session
from src.jobs_scraper.models import SessionLocal, init_db, Job
from src.jobs_scraper.scrapers import run_all_scrapers

# === audio helper libs ===
try:
    from imageio_ffmpeg import get_ffmpeg_exe
except Exception:
    get_ffmpeg_exe = None
from pydub import AudioSegment

# -----------------------
# Attempt to import Vosk (optional)
# -----------------------
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    import wave as _wave
    import json as _json
    VOSK_IMPORT_OK = True
except Exception:
    VOSK_IMPORT_OK = False

# --- ffmpeg setup (best-effort) ---
ffmpeg_path = None
try:
    if get_ffmpeg_exe:
        ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        ffprobe_path = os.path.join(ffmpeg_dir, "ffprobe.exe")
        try:
            if not os.path.exists(ffprobe_path):
                shutil.copy(ffmpeg_path, ffprobe_path)
        except Exception:
            pass
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffmpeg = ffmpeg_path
        AudioSegment.ffprobe = ffprobe_path if os.path.exists(ffprobe_path) else ffmpeg_path
        os.environ["PATH"] += os.pathsep + ffmpeg_dir
    else:
        ffmpeg_path = None
except Exception as e:
    print("[FFMPEG] not configured or not found:", e)
    ffmpeg_path = None

# === directories & config ===
BASE_DIR = os.getcwd()
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
AUDIO_DIR = os.path.join(UPLOAD_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

ALLOWED_EXT = {'.pdf', '.docx', '.doc', '.txt'}

# === FastAPI + SocketIO app ===
app = FastAPI(title="AI Interviewer")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Templates and Static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Socket.IO AsyncServer
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, app)

# === init models (your custom modules) ===
try:
    emotion_analyzer = EmotionAnalyzer(model_path="emotion_recognition_model.h5")
except Exception as e:
    print("Warning: EmotionAnalyzer init failed:", e)
    emotion_analyzer = None

try:
    eye_tracker = EyeTracker(log_file=os.path.join(BASE_DIR, "logs", "eye_tracking_log.json"))
except Exception as e:
    print("Warning: EyeTracker init failed:", e)
    eye_tracker = None

# === Vosk model initialization (if available) ===
VOSK_MODEL_PATH = os.path.join(BASE_DIR, "models", "vosk-small-en-us-0.15")
_vosk_model = None
if VOSK_IMPORT_OK:
    if os.path.exists(VOSK_MODEL_PATH):
        try:
            _vosk_model = VoskModel(VOSK_MODEL_PATH)
            print("[VOSK] model loaded from", VOSK_MODEL_PATH)
        except Exception as e:
            print("[VOSK] failed to load model:", e)
            _vosk_model = None
    else:
        print("[VOSK] import ok but model folder not found at:", VOSK_MODEL_PATH)
else:
    print("[VOSK] not installed; will fallback to speech_recognition (Google)")

# === in-memory session store ===
sessions = {}
lock = Lock()

# === JOBS SCRAPER init ===
try:
    init_db()
    print("[JOBS] Database initialized")
except Exception as e:
    print("[JOBS] Initialization failed:", e)

# Dependency to get Job DB session
def get_jobs_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------
# Utilities
# -----------------------
def json_safe(obj):
    import datetime as _dt
    import numpy as _np
    def conv(o):
        if isinstance(o, (_np.integer,)): return int(o)
        if isinstance(o, (_np.floating,)): return float(o)
        if isinstance(o, (_np.bool_, bool)): return bool(o)
        if isinstance(o, _np.ndarray): return o.tolist()
        if isinstance(o, _dt.datetime): return o.isoformat()
        return str(o)
    try:
        return json.loads(json.dumps(obj, default=conv))
    except Exception:
        return str(obj)

# -----------------------
# Resume extraction helpers
# -----------------------
def extract_text_from_pdf(path):
    try:
        import fitz
        doc = fitz.open(path)
        pages = [page.get_text("text") for page in doc]
        return "\n".join(pages)
    except Exception as e:
        print("PDF extraction failed:", e)
        return ""

def extract_text_from_docx(path):
    try:
        from docx import Document
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)
    except Exception as e:
        print("DOCX extraction failed:", e)
        return ""

def simple_section_parser(full_text):
    text = (full_text or "").strip()
    if not text:
        return {"education_details": "", "experience_details": "", "skill": ""}
    lower = text.lower()
    sections = {"education_details": "", "experience_details": "", "skill": ""}

    def find_any(names):
        for n in names:
            idx = lower.find(n)
            if idx != -1:
                return idx, n
        return -1, None

    edu_idx, _ = find_any(['education', 'educational qualifications', 'academic qualifications'])
    exp_idx, _ = find_any(['experience', 'work experience', 'professional experience'])
    skill_idx, _ = find_any(['skills', 'technical skills', 'skills & technologies', 'technical summary', 'technical skills:'])
    end = len(text)

    def slice_between(start, stop):
        return text[start:stop].strip()

    if skill_idx != -1:
        nexts = sorted([i for i in [edu_idx, exp_idx] if i > skill_idx] + [end])
        sections['skill'] = slice_between(skill_idx, nexts[0])
    if exp_idx != -1:
        nexts = sorted([i for i in [edu_idx, skill_idx] if i > exp_idx] + [end])
        sections['experience_details'] = slice_between(exp_idx, nexts[0])
    if edu_idx != -1:
        nexts = sorted([i for i in [exp_idx, skill_idx] if i > edu_idx] + [end])
        sections['education_details'] = slice_between(edu_idx, nexts[0])
    if not any(sections.values()):
        sections['skill'] = text[:20000]
    for k in sections:
        if sections[k] and len(sections[k]) > 20000:
            sections[k] = sections[k][:20000]
    return sections

# -----------------------
# Templates & simple pages
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/resume", response_class=HTMLResponse)
async def resume_page(request: Request):
    return templates.TemplateResponse("resume.html", {"request": request})

@app.get("/interview", response_class=HTMLResponse)
async def interview(request: Request):
    return templates.TemplateResponse("interview.html", {"request": request})

# -----------------------
# Evaluate resume (ATS)
# -----------------------
@app.post("/evaluate_resume")
async def evaluate_resume(
    resume: UploadFile = File(None),
    job_description: str = Form(""),
    jobDesc: str = Form("")
):
    try:
        if not resume:
            return JSONResponse({"error": "no file uploaded"}, status_code=400)
            
        filename = resume.filename or "uploaded"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXT:
            return JSONResponse({"error": f"unsupported file type: {ext}"}, status_code=400)

        uid = uuid.uuid4().hex
        debug_filename = f"debug_{uid}_{filename}"
        debug_path = os.path.join(UPLOAD_DIR, debug_filename)
        
        # Save the uploaded file
        with open(debug_path, "wb") as buffer:
            shutil.copyfileobj(resume.file, buffer)
            
        print(f"[evaluate_resume] saved uploaded file to: {debug_path}")

        raw_text = ""
        try:
            if ext == ".pdf":
                raw_text = extract_text_from_pdf(debug_path)
            elif ext in (".docx", ".doc"):
                raw_text = extract_text_from_docx(debug_path)
            elif ext == ".txt":
                with open(debug_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_text = f.read()
        except Exception as e:
            print("EXTRACTION ERROR:", e)
            raw_text = ""

        resume_data = simple_section_parser(raw_text)

        try:
            # Running synchronous model prediction in an executor is better for FastAPI
            # but for simplicity we keep it inline, it's fast enough in most cases
            results = model.get_ats_score(resume_data)
        except Exception as e:
            log_path = os.path.join(UPLOAD_DIR, f"eval_error_{uid}.log")
            with open(log_path, "w", encoding="utf-8") as logf:
                import traceback
                traceback.print_exc(file=logf)
            print(f"[evaluate_resume] model.get_ats_score failed; log written to {log_path}")
            return JSONResponse({"error":"model scoring failed","debug_file":debug_path,"error_log":log_path}, status_code=500)

        matched_job = results.get("matched_job", "No match found")
        ats_score = results.get("ats_score", 0.0)
        top_matches = results.get("top_matches", [])

        response = {"matched_job": matched_job, "ats_score": ats_score, "top_matches": top_matches}

        jd = job_description or jobDesc
        if jd and jd.strip():
            try:
                custom = model.calculate_custom_ats_score(resume_data, jd)
                if isinstance(custom, dict) and "ats_score" in custom:
                    response["custom_ats_score"] = custom["ats_score"]
            except Exception as e:
                log_path = os.path.join(UPLOAD_DIR, f"eval_custom_error_{uid}.log")
                with open(log_path, "w", encoding="utf-8") as logf:
                    import traceback
                    traceback.print_exc(file=logf)
                response["custom_ats_error"] = True
                response["custom_ats_log"] = log_path

        return JSONResponse(response)

    except Exception as e:
        uid = uuid.uuid4().hex
        log_path = os.path.join(UPLOAD_DIR, f"evaluate_unhandled_{uid}.log")
        with open(log_path, "w", encoding="utf-8") as logf:
            import traceback
            traceback.print_exc(file=logf)
        print(f"[evaluate_resume] unhandled exception; see {log_path}")
        return JSONResponse({"error":"internal server error","error_log":log_path}, status_code=500)

# -----------------------
# Socket.IO events
# -----------------------
@sio.on("connect")
async def on_connect(sid, environ):
    sessions[sid] = {"questions": [], "answers": [], "emotion": [], "eye": [], "current_idx": 0}
    print(f"Client connected: {sid}")
    await sio.emit("connected", {"sid": sid}, to=sid)

@sio.on("disconnect")
async def on_disconnect(sid):
    sessions.pop(sid, None)
    print(f"Client disconnected: {sid}")

@sio.on("video_frame")
async def handle_video_frame(sid, data):
    # This runs asynchronously
    frame_b64 = data.get("frame")
    if not frame_b64:
        return
    try:
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",")[1]
        
        # Offload decoding and processing to thread executor since it's blocking
        def process_frame_sync():
            arr = np.frombuffer(base64.b64decode(frame_b64), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            em_data = None
            ey_data = None
            try:
                if emotion_analyzer:
                    em_data = emotion_analyzer.process_frame(img)
            except Exception as e:
                print("Emotion processing error:", e)
            try:
                if eye_tracker:
                    ey_data = eye_tracker.analyze_frame(img)
            except Exception as e:
                print("Eye processing error:", e)
            return em_data, ey_data

        loop = asyncio.get_event_loop()
        emotion_data, eye_data = await loop.run_in_executor(None, process_frame_sync)

        with lock:
            if emotion_data is not None:
                sessions[sid]["emotion"].append({"ts": datetime.now().isoformat(), "data": emotion_data})
            if eye_data is not None:
                sessions[sid]["eye"].append({"ts": datetime.now().isoformat(), "data": eye_data})
                
        await sio.emit("analysis", {"emotion": json_safe(emotion_data), "eye": json_safe(eye_data)}, to=sid)
    except Exception as e:
        print("Frame processing error:", e)

# -----------------------
# Minimal helper: aggregate recent metrics for a session for per-question feedback
# -----------------------
def aggregate_recent_metrics_for_sid(sid, window_secs=30):
    sess = sessions.get(sid, {})
    now_ts = datetime.utcnow().timestamp()
    emotion_entries = []
    eye_entries = []
    
    with lock:
        emo_list = list(sess.get("emotion", []))
        eye_list = list(sess.get("eye", []))
        
    for e in emo_list:
        try:
            ts = e.get("ts")
            t = datetime.fromisoformat(ts).timestamp() if ts else now_ts
            if now_ts - t <= window_secs:
                emotion_entries.append(e)
        except Exception:
            emotion_entries.append(e)
            
    for e in eye_list:
        try:
            ts = e.get("ts")
            t = datetime.fromisoformat(ts).timestamp() if ts else now_ts
            if now_ts - t <= window_secs:
                eye_entries.append(e)
        except Exception:
            eye_entries.append(e)

    emo_summary = analyze_emotions(emotion_entries) if emotion_entries else {"counts": {}, "dominant": "unknown", "percent_nervous": 0, "percent_calm": 0, "total": 0}
    eye_summary = analyze_eye(eye_entries) if eye_entries else {"total":0, "avg_fixations":0, "looked_away_pct":0, "frequent_direction":"center"}
    return {"emotion_summary": emo_summary, "eye_summary": eye_summary, "raw_emotions": emotion_entries, "raw_eye": eye_entries}

@sio.on("start_questions")
async def handle_start_questions(sid, data):
    role = (data.get("role") or "").strip()
    num_q = int(data.get("num_questions", 2))
    
    await sio.emit("questions", {"questions": ["⏳ Generating questions... please wait"]}, to=sid)

    async def gen_task():
        try:
            loop = asyncio.get_event_loop()
            if role:
                print(f"[Groq] Generating technical questions for '{role}'")
                qlist = await loop.run_in_executor(None, llm.get_technical_questions, role)
            else:
                print("[Groq] Generating HR questions")
                qlist = await loop.run_in_executor(None, llm.get_hr_questions)

            if isinstance(qlist, str):
                qlist = [q.strip() for q in qlist.split("\n") if q.strip()]
            qlist = qlist[:num_q]

            with lock:
                if sid in sessions:
                    sessions[sid]["questions"] = qlist
                    sessions[sid]["current_idx"] = 0

            await sio.emit("questions", {"questions": qlist}, to=sid)
            if qlist:
                await sio.emit("chat_message", {"role": "interviewer", "text": qlist[0], "idx": 0}, to=sid)
        except Exception as e:
            print("[Groq Error]", e)
            fallback = [
                f"Tell me about your experience as a {role or 'professional'}.",
                "What motivated you to apply for this role?",
                "Describe a challenge you solved recently.",
                "What are your strengths and weaknesses?",
                "Where do you see yourself in 5 years?",
            ][:num_q]
            with lock:
                if sid in sessions:
                    sessions[sid]["questions"] = fallback
                    sessions[sid]["current_idx"] = 0
            await sio.emit("questions", {"questions": fallback}, to=sid)
            await sio.emit("chat_message", {"role": "interviewer", "text": fallback[0], "idx": 0}, to=sid)

    # Launch task asynchronously
    asyncio.create_task(gen_task())

@sio.on("request_next_question")
async def handle_request_next(sid, data):
    idx = int(data.get("idx", 0))
    qs = sessions.get(sid, {}).get("questions", [])
    if 0 <= idx < len(qs):
        await sio.emit("chat_message", {"role": "interviewer", "text": qs[idx], "idx": idx}, to=sid)
        with lock:
            sessions[sid]["current_idx"] = idx

# -----------------------
# Audio upload + transcription + LLM feedback
# -----------------------
@app.post("/upload_audio")
async def upload_audio(
    sid: str = Form(None),
    question_idx: int = Form(-1),
    expected_answer: str = Form(""),
    file: UploadFile = File(None)
):
    if not file:
        return JSONResponse({"error": "no file"}, status_code=400)

    fname = f"{uuid.uuid4().hex}_{file.filename}"
    path = os.path.join(AUDIO_DIR, fname)
    
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    transcript = "[no speech]"
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise ValueError("Empty or missing audio file")

        print(f"[Audio Upload] Saved file: {path} ({os.path.getsize(path)} bytes)")

        # Synchronous audio processing
        def process_audio():
            nonlocal transcript
            lower = path.lower()
            wav_path = None

            if lower.endswith(".wav"):
                wav_path = path
            else:
                if ffmpeg_path:
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
                            AudioSegment.from_file(path).export(tmpwav.name, format="wav")
                            wav_path = tmpwav.name
                            print(f"[Audio Conversion] converted {path} -> {wav_path}")
                    except Exception as e:
                        print("[Audio Conversion Error]", e)
                else:
                    print("[Audio Conversion] ffmpeg not available; expecting client to upload WAV")

            if wav_path and _vosk_model is not None:
                try:
                    wf = _wave.open(wav_path, "rb")
                    rec = KaldiRecognizer(_vosk_model, wf.getframerate())
                    rec.SetWords(True)
                    results = []
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            results.append(_json.loads(rec.Result()))
                    results.append(_json.loads(rec.FinalResult()))
                    texts = [r.get("text","") for r in results if isinstance(r, dict)]
                    transcript = " ".join([t for t in texts if t]).strip() or "[no speech detected]"
                    wf.close()
                    print(f"[VOSK Transcript] {transcript[:120]}...")
                except Exception as e:
                    print("[VOSK] transcription failed:", e)
                    transcript = f"[vosk error: {e}]"
            else:
                try:
                    target_path = wav_path or path
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(target_path) as src:
                        audio_data = recognizer.record(src)
                        try:
                            fdlen = len(audio_data.frame_data)
                        except Exception:
                            fdlen = None
                        if fdlen == 0:
                            raise ValueError("No audio frames captured")
                        transcript = recognizer.recognize_google(audio_data)
                    print(f"[SR Transcript] {transcript[:120]}...")
                except Exception as e:
                    transcript = f"[transcription error: {e}]"
                    print("[Audio Conversion/Transcription Error]", e)

            try:
                if wav_path and wav_path != path:
                    os.remove(wav_path)
            except Exception:
                pass

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, process_audio)

    except Exception as e:
        transcript = f"[transcription error: {e}]"
        print("[Audio Upload Error]", e)

    # --- aggregate recent emotion & eye metrics for this question ---
    metrics = aggregate_recent_metrics_for_sid(sid, window_secs=30)
    emotion_summary = metrics.get("emotion_summary")
    eye_summary = metrics.get("eye_summary")

    try:
        loop = asyncio.get_event_loop()
        feedback = await loop.run_in_executor(None, llm.evaluate_candidate_answers, [expected_answer], [transcript], "technical")
    except Exception as e:
        feedback = f"[LLM feedback error: {e}]"
        print("[LLM Feedback Error]", e)

    q_idx = int(question_idx)

    # store answer + summaries into session
    if sid in sessions:
        with lock:
            sessions[sid]["answers"].append({
                "q_idx": q_idx,
                "question": expected_answer,
                "transcript": transcript,
                "feedback": feedback,
                "emotion": emotion_summary,
                "eye": eye_summary,
                "ts": datetime.now().isoformat()
            })

    # emit structured per-question feedback to the client
    try:
        await sio.emit("question_feedback", {
            "q_idx": q_idx,
            "transcript": transcript,
            "feedback": feedback,
            "emotion_summary": emotion_summary,
            "eye_summary": eye_summary
        }, to=sid)
    except Exception as e:
        print("[emit question_feedback] failed:", e)

    # also send the human-readable chat_message (keeps backward compatibility)
    try:
        await sio.emit("chat_message", {
            "role": "interviewer",
            "text": feedback,
            "idx": q_idx,
            "eval": True
        }, to=sid)
    except Exception as e:
        print("[emit chat_message] failed:", e)

    # advance to next question if exists (or finish)
    qs = sessions.get(sid, {}).get("questions", [])
    next_idx = q_idx + 1
    if next_idx < len(qs):
        try:
            await sio.emit("chat_message", {"role": "interviewer", "text": qs[next_idx], "idx": next_idx}, to=sid)
            with lock:
                sessions[sid]["current_idx"] = next_idx
        except Exception as e:
            print("[emit next question] failed:", e)
    else:
        # last answer - finalize: persist and emit interview_complete with redirect
        try:
            saved = persist_session_report(sid)
            if saved:
                print(f"[persist] session {sid} saved to {saved}")
            await sio.emit("interview_complete", {"redirect": f"/feedback?sid={sid}"}, to=sid)
        except Exception as e:
            print("[emit interview_complete] failed:", e)

    return JSONResponse({
        "status":"ok",
        "transcript":transcript,
        "feedback":feedback,
        "emotion_summary":emotion_summary,
        "eye_summary":eye_summary
    })

# -----------------------
# helpers for robust parsing & safe persistence
# -----------------------
def find_answers_recursive(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and any(k in obj[0] for k in ("transcript", "question", "q_idx", "feedback")):
            return obj
        for item in obj:
            res = find_answers_recursive(item)
            if res:
                return res
        return []
    if isinstance(obj, dict):
        for k in ("answers", "entries", "responses", "data"):
            v = obj.get(k)
            if isinstance(v, list):
                if v and isinstance(v[0], dict) and any(x in v[0] for x in ("transcript", "question", "q_idx", "feedback")):
                    return v
                res = find_answers_recursive(v)
                if res:
                    return res
        for v in obj.values():
            if isinstance(v, (list, dict)):
                res = find_answers_recursive(v)
                if res:
                    return res
        return []
    return []

def append_ndjson(path, entries):
    try:
        with open(path, "a", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, default=str) + "\n")
        return True
    except Exception as e:
        print("[append_ndjson] failed:", e)
        return False

def persist_raw_emotion_ndjson(sid, emotion_entries):
    try:
        if not emotion_entries:
            return
        ndpath = os.path.join(UPLOAD_DIR, "emotion_ndjson.log")
        append_ndjson(ndpath, emotion_entries)
        perpath = os.path.join(UPLOAD_DIR, f"emotion_{sid}.ndjson")
        with open(perpath, "w", encoding="utf-8") as f:
            for e in emotion_entries:
                f.write(json.dumps(e, default=str) + "\n")
    except Exception as e:
        print("[persist_raw_emotion_ndjson] failed:", e)

def persist_session_report(sid):
    try:
        s = sessions.get(sid)
        if not s:
            return None
        out = {
            "sid": sid,
            "ts": datetime.now().isoformat(),
            "questions": s.get("questions", []),
            "answers": s.get("answers", []),
            "emotion": s.get("emotion", []),
            "eye": s.get("eye", [])
        }
        # 1) append to main interview_log.json (read/append/write)
        log_path = os.path.join(UPLOAD_DIR, "interview_log.json")
        existing = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    existing = json.load(f) or []
            except Exception:
                existing = []
        existing.append(out)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        # 2) write a per-session JSON file (guaranteed structure)
        per_path = os.path.join(UPLOAD_DIR, f"session_{sid}.json")
        with open(per_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)

        # 3) persist raw emotion frames as ndjson (append) and per-session ndjson
        try:
            persist_raw_emotion_ndjson(sid, s.get("emotion", []))
        except Exception as e:
            print("[persist] persist_raw_emotion_ndjson failed:", e)

        # 4) persist per-session eye data (overwrite)
        try:
            eye_path = os.path.join(UPLOAD_DIR, f"eye_{sid}.json")
            with open(eye_path, "w", encoding="utf-8") as f:
                json.dump(s.get("eye", []), f, indent=2, default=str)
        except Exception as e:
            print("[persist] eye per-session persist failed:", e)

        return log_path
    except Exception as e:
        print("[persist_session_report] failed:", e)
        return None

def try_load_json_candidates(names):
    extra_names = [
        "emotion_analysis_log.json",
        "emotion_analysis.json",
        "emotion_log.json",
        "emotion_data.json",
        "eye_tracking_log.json",
        "eye_log.json",
        "interview_log.json",
        "interview_log_full.json",
        "interview_log"
    ]
    all_names = list(dict.fromkeys((names or []) + extra_names))
    candidates = []
    for d in [BASE_DIR, UPLOAD_DIR, os.getcwd()]:
        for n in all_names:
            candidates.append(os.path.join(d, n))
    candidates = list(dict.fromkeys(candidates))
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[feedback] loaded JSON from: {p}")
            return data, p
        except Exception as e:
            print(f"[feedback] failed to parse {p}: {e}")
            continue
    print("[feedback] no JSON candidate loaded from:", candidates)
    return None, None

def analyze_emotions(emotion_entries):
    counts = {}
    total = 0
    for e in emotion_entries:
        total += 1
        dat = e.get("data") if isinstance(e, dict) else e
        label = None
        if isinstance(dat, dict):
            for k in ("label","emotion","pred","predicted","dominant_emotion"):
                if k in dat:
                    label = dat[k]; break
            if label is None:
                probs = dat.get("emotions") or dat.get("probs") or dat.get("scores") or dat.get("scores_dict") or dat.get("probabilities")
                if isinstance(probs, dict):
                    label = max(probs.items(), key=lambda x: x[1])[0]
            if label is None:
                label = dat.get("dominant_emotion") if isinstance(dat, dict) else None
            if label is None:
                label = str(dat)
        else:
            label = str(dat)
        label = str(label).lower().strip()
        counts[label] = counts.get(label,0) + 1

    nervous_labels = {'fear','anxious','nervous','sad','surprise','surprised','angry'}
    calm_labels = {'neutral','happy','content','calm','relaxed','smile','confident'}
    nervous_count = sum(v for k,v in counts.items() if k in nervous_labels)
    calm_count = sum(v for k,v in counts.items() if k in calm_labels)
    dominant = max(counts.items(), key=lambda x: x[1])[0] if counts else "unknown"
    percent_nervous = (nervous_count / total * 100) if total else 0
    percent_calm = (calm_count / total * 100) if total else 0
    return {"counts":counts,"dominant":dominant,"total":total,"percent_nervous":round(percent_nervous,1),"percent_calm":round(percent_calm,1)}

def analyze_eye(eye_entries):
    total = 0
    looked_away = 0
    dir_counts = {}
    
    for e in eye_entries:
        total += 1
        dat = e.get("data") if isinstance(e, dict) else e
        
        if isinstance(dat, dict) and "is_looking_at_camera" in dat:
            direction = dat.get("direction", "unknown")
            dir_counts[direction] = dir_counts.get(direction, 0) + 1
            
            if not dat.get("is_looking_at_camera", True):
                looked_away += 1
        else:
            gaze = None
            if isinstance(dat, dict):
                gaze = dat.get("gaze") or dat.get("direction") or dat.get("look") or dat.get("position")
            
            if isinstance(gaze, str):
                g = gaze.lower()
                dir_counts[g] = dir_counts.get(g,0) + 1
                if g not in ('center','straight','forward'):
                    looked_away += 1
            elif isinstance(gaze, (list,tuple)) and len(gaze)>=2:
                looked_away += 1
                
    looked_away_pct = (looked_away/total*100) if total else 0
    
    frequent_direction = "center"
    max_count = 0
    for d, c in dir_counts.items():
        if d != "center" and c > max_count:
            max_count = c
            frequent_direction = d
            
    return {
        "total": total,
        "looked_away_pct": round(looked_away_pct, 1),
        "frequent_direction": frequent_direction,
        "time_not_looking_front": f"{round(looked_away_pct, 1)}%"
    }

def synthesize_answer_feedback(interview_entries):
    total = 0
    short_answers = []
    long_answers = []
    all_feedback_texts = []
    for a in interview_entries:
        total += 1
        txt = (a.get("transcript") or "") if isinstance(a, dict) else str(a)
        fb = (a.get("feedback") or "") if isinstance(a, dict) else ""
        all_feedback_texts.append(str(fb))
        if len(txt.split()) < 6:
            short_answers.append({"q": a.get("question", ""), "transcript": txt, "idx": a.get("q_idx", -1)})
        else:
            long_answers.append({"q": a.get("question",""), "transcript": txt, "idx": a.get("q_idx",-1)})
    weak = short_answers[:3] if short_answers else (long_answers[:3] if long_answers else [])
    return {"count": total, "weak_examples": weak, "feedback_texts": all_feedback_texts}

def generate_improvement_paragraphs(emotion_summary, eye_summary):
    p = {}
    pn = emotion_summary.get("percent_nervous", 0)
    dominant = emotion_summary.get("dominant","")
    if pn >= 45:
        p['confidence'] = ("You appeared noticeably nervous during the session. Work on building calm, confident delivery: practice breathing exercises before answers, pause briefly to collect your thoughts, use short structured responses (Situation → Task → Action → Result) and rehearse common questions out loud. Mock interviews with a friend or recording yourself and reviewing the playback can desensitize interview anxiety. Also try to slow your speech slightly and add small, intentional pauses — it signals control and thoughtfulness.")
    elif pn >= 18:
        p['confidence'] = ("You showed some signs of nervousness in parts of the interview. Focus on short grounding techniques (deep breaths before answering) and practice concise STAR-format answers for behavioral questions. Confidence grows with repetition — 5–10 short practice answers daily will make a big difference.")
    else:
        p['confidence'] = ("Your emotional profile looks calm and composed overall — good job. Keep practicing to maintain steady pacing and clarity; try to vary tone to emphasize achievements and keep the interviewer engaged.")

    dom = dominant.lower() if isinstance(dominant,str) else ""
    if dom in ('happy','confident','smile') and emotion_summary.get("percent_calm",0) < 20 and pn <= 20:
        p['cocky'] = ("At times your delivery felt overly assured. While confidence is important, balance it with humility: acknowledge contributions by the team, avoid absolute language (e.g., 'always', 'never'), and show curiosity by asking the interviewer follow-up questions. This builds rapport and avoids coming across as dismissive.")
    else:
        p['cocky'] = ("No signs of overconfidence were detected. Keep combining confidence with curiosity — ask clarifying questions and show collaborative language (e.g., 'we', 'team').")

    la = eye_summary.get("looked_away_pct", 0)
    dirc = eye_summary.get("frequent_direction", "center")
    p['eye'] = (f"Eye tracking shows you looked away about {la}% of the time and tended to glance towards '{dirc}'. Frequent looking-away can indicate searching for answers or nervousness. To improve: practice steady eye contact (soft gaze) with the camera for 50–70% of your speaking time, briefly look away to collect thoughts but return to center quickly, and when referencing examples, imagine speaking to one person—this reduces scattered glances.")
    return p

@app.get("/_debug_sessions")
async def debug_sessions():
    try:
        out = {}
        for sid, s in sessions.items():
            out[sid] = {
                "questions": s.get("questions", []),
                "current_idx": s.get("current_idx", 0),
                "answers_count": len(s.get("answers", [])),
                "last_answers": s.get("answers", [])[-10:],
                "emotion_count": len(s.get("emotion", [])),
                "eye_count": len(s.get("eye", []))
            }
        return JSONResponse({"ok": True, "sessions": out})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})

@app.get("/feedback", response_class=HTMLResponse)
async def feedback_page(request: Request, sid: str = None):
    # If sid provided and there's a per-session file, prefer that (guaranteed structure)
    if sid:
        per_path = os.path.join(UPLOAD_DIR, f"session_{sid}.json")
        if os.path.exists(per_path):
            try:
                with open(per_path, "r", encoding="utf-8") as f:
                    session_obj = json.load(f)
                interview_entries = session_obj.get("answers", []) or []
                emotion_entries = [{"ts": e.get("ts") if isinstance(e, dict) else None, "data": e.get("data") if isinstance(e, dict) else e} for e in (session_obj.get("emotion") or [])]
                eye_entries = [{"ts": e.get("ts") if isinstance(e, dict) else None, "data": e.get("data") if isinstance(e, dict) else e} for e in (session_obj.get("eye") or [])]

                emotion_summary = analyze_emotions(emotion_entries) if emotion_entries else {"counts": {}, "dominant": "unknown", "percent_nervous": 0, "percent_calm": 0, "total": 0}
                eye_summary = analyze_eye(eye_entries) if eye_entries else {"total":0,"avg_fixations":0,"looked_away_pct":0,"frequent_direction":"center"}
                answer_summary = synthesize_answer_feedback(interview_entries or [])
                answer_summary["answers"] = interview_entries or []
                paragraphs = generate_improvement_paragraphs(emotion_summary, eye_summary)

                try:
                    if hasattr(llm, "summarize_session"):
                        llm_summary = llm.summarize_session(interview_entries[:10])
                    elif hasattr(llm, "get_summary"):
                        llm_summary = llm.get_summary(interview_entries[:10])
                    else:
                        llm_summary = (f"Interviewed on {datetime.now().date().isoformat()}: answered {answer_summary.get('count',0)} questions. Primary emotional signal: {emotion_summary.get('dominant','unknown')}.")
                except Exception:
                    llm_summary = (f"Interview summary: answered {answer_summary.get('count',0)} questions. Primary emotion: {emotion_summary.get('dominant','unknown')}.")

                report = {
                    "interview_path": per_path,
                    "eye_path": os.path.join(UPLOAD_DIR, f"eye_{sid}.json") if os.path.exists(os.path.join(UPLOAD_DIR, f"eye_{sid}.json")) else None,
                    "emotion_summary": emotion_summary,
                    "eye_summary": eye_summary,
                    "answer_summary": answer_summary,
                    "paragraphs": paragraphs,
                    "llm_summary": llm_summary
                }
                return templates.TemplateResponse("feedback.html", {"request": request, "report": report})
            except Exception as e:
                print("[feedback] failed to read per-session file:", e)

    emo_file, emo_path = try_load_json_candidates(["emotion_analysis_log.json", "emotion_data.json", "emotion_log.json"])
    eye_file, eye_path = try_load_json_candidates(["eye_tracking_log.json", "eye_log.json", "eye_tracking.json"])
    interview_file, interview_path = try_load_json_candidates(["interview_log.json", "interview_log_full.json", "interview_log"])

    interview_entries = []
    if interview_file:
        interview_entries = find_answers_recursive(interview_file)

    if not interview_entries:
        for sid_, s in sessions.items():
            if s.get("answers"):
                interview_entries.extend(s.get("answers"))

    emotion_entries = []
    if emo_file:
        if isinstance(emo_file, list):
            for e in emo_file:
                ts = e.get("timestamp") if isinstance(e, dict) else None
                emotion_entries.append({"ts": ts, "data": e})
        elif isinstance(emo_file, dict):
            if "entries" in emo_file and isinstance(emo_file["entries"], list):
                for e in emo_file["entries"]:
                    emotion_entries.append({"ts": e.get("timestamp") if isinstance(e, dict) else None, "data": e})
            else:
                for v in emo_file.values():
                    if isinstance(v, list):
                        for e in v:
                            if isinstance(e, dict):
                                emotion_entries.append({"ts": e.get("timestamp") or e.get("ts"), "data": e})
                        break

    if not emotion_entries and interview_entries:
        for a in interview_entries:
            if isinstance(a, dict) and a.get("emotion"):
                emotion_entries.append({"ts": a.get("ts") or a.get("time"), "data": a.get("emotion")})

    eye_entries = []
    if eye_file:
        if isinstance(eye_file, list):
            for e in eye_file:
                ts = e.get("timestamp") if isinstance(e, dict) else None
                eye_entries.append({"ts": ts, "data": e})
        elif isinstance(eye_file, dict) and "entries" in eye_file and isinstance(eye_file["entries"], list):
            for e in eye_file["entries"]:
                eye_entries.append({"ts": e.get("timestamp") or e.get("ts"), "data": e})

    if not eye_entries and interview_entries:
        for a in interview_entries:
            if isinstance(a, dict) and a.get("eye"):
                eye_entries.append({"ts": a.get("ts") or a.get("time"), "data": a.get("eye")})

    emotion_summary = analyze_emotions(emotion_entries) if emotion_entries else {"counts": {}, "dominant": "unknown", "percent_nervous": 0, "percent_calm": 0, "total": 0}
    eye_summary = analyze_eye(eye_entries) if eye_entries else {"total":0,"avg_fixations":0,"looked_away_pct":0,"frequent_direction":"center"}
    answer_summary = synthesize_answer_feedback(interview_entries or [])
    answer_summary["answers"] = interview_entries or []
    paragraphs = generate_improvement_paragraphs(emotion_summary, eye_summary)

    llm_summary = None
    try:
        if hasattr(llm, "summarize_session"):
            llm_summary = llm.summarize_session(interview_entries[:10])
        elif hasattr(llm, "get_summary"):
            llm_summary = llm.get_summary(interview_entries[:10])
        else:
            llm_summary = (f"Interviewed on {datetime.now().date().isoformat()}: answered {answer_summary.get('count',0)} questions. Primary emotional signal: {emotion_summary.get('dominant','unknown')}. See per-question feedback and suggested improvements below.")
    except Exception as e:
        print("[feedback] LLM summary failed:", e)
        llm_summary = (f"Interview summary: answered {answer_summary.get('count',0)} questions. Primary emotion: {emotion_summary.get('dominant','unknown')}.")

    report = {
        "interview_path": interview_path,
        "eye_path": eye_path,
        "emotion_summary": emotion_summary,
        "eye_summary": eye_summary,
        "answer_summary": answer_summary,
        "paragraphs": paragraphs,
        "llm_summary": llm_summary
    }

    return templates.TemplateResponse("feedback.html", {"request": request, "report": report})

# -----------------------
# JOBS SCRAPER ROUTES
# -----------------------
@app.get("/jobs", response_class=HTMLResponse)
async def jobs_dashboard(request: Request, db: _Session = Depends(get_jobs_db)):
    """Serves the job listing dashboard."""
    jobs = db.query(Job).order_by(Job.date_added.desc()).all()
    return templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs})

@app.get("/api/jobs")
async def get_jobs_json(
    db: _Session = Depends(get_jobs_db), 
    company: str = None, 
    title: str = None
):
    """Returns job listings in JSON format with optional filters."""
    query = db.query(Job)
    if company:
        query = query.filter(Job.company_name.contains(company))
    if title:
        query = query.filter(Job.job_title.contains(title))
    
    jobs = query.order_by(Job.date_added.desc()).all()
    return [
        {
            "id": j.id,
            "company_name": j.company_name,
            "job_title": j.job_title,
            "job_url": j.job_url,
            "location": j.location,
            "date_added": j.date_added.strftime("%Y-%m-%d")
        } 
        for j in jobs
    ]

@app.post("/api/scrape")
async def trigger_scrape(db: _Session = Depends(get_jobs_db)):
    """Manually trigger the scraping engine."""
    try:
        new_jobs_added = run_all_scrapers(db)
        return {"message": "Scraping completed", "new_jobs_added": new_jobs_added}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    import uvicorn
    print("✅ FastAPI SocketIO AI Interviewer running at http://127.0.0.1:5000")
    uvicorn.run("app:socket_app", host="127.0.0.1", port=5000, reload=True)
