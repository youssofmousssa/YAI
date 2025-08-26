#!/usr/bin/env python3
"""
ai_stream.py

Single-file FastAPI backend with streaming support for Groq-based models and
external media APIs (Flux Pro / DarkAI). Designed to behave like a ChatGPT-style
backend with real-time streaming, session memory, and media endpoints.

Run:
  uvicorn ai_stream:app --reload --host 0.0.0.0 --port 8001
"""

import os
import uuid
import json
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

# Groq SDK (must be installed and configured)
from groq import Groq

# optional fallback TTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# ---------- Configuration ----------
LOG = logging.getLogger("ai_stream")
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
FLUX_PRO_BASE = os.getenv("FLUX_PRO_BASE", "https://sii3.moayman.top/api")
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "./media"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Groq client (SDK)
try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else Groq()
except TypeError:
    # Some SDKs may not accept api_key param
    groq_client = Groq()

# Async HTTP client for external endpoints
httpx_client: Optional[httpx.AsyncClient] = None

app = FastAPI(title="AI Streaming Backend", version="1.2")

# In-memory sessions: {session_id: [ {"role":"user"/"assistant","content":str}, ... ]}
SESSIONS: Dict[str, List[Dict[str, str]]] = {}


# ---------- Utilities ----------
def unique_filename(prefix: str, ext: str) -> Path:
    return MEDIA_DIR / f"{prefix}-{uuid.uuid4().hex}.{ext.lstrip('.')}"

async def startup_event():
    global httpx_client
    httpx_client = httpx.AsyncClient(timeout=60.0)

async def shutdown_event():
    global httpx_client
    if httpx_client:
        await httpx_client.aclose()

app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)


def _extract_text_from_completion(completion: Any) -> str:
    """
    Defensive extraction of text from various possible SDK response shapes.
    """
    try:
        # common: completion.choices[0].message.content
        if hasattr(completion, "choices"):
            ch0 = completion.choices[0]
            if hasattr(ch0, "message") and hasattr(ch0.message, "content"):
                return ch0.message.content
            # fallback: maybe .choices[0].text or .choices[0].delta.content
            if hasattr(ch0, "text"):
                return ch0.text
            if hasattr(ch0, "delta") and hasattr(ch0.delta, "content"):
                return ch0.delta.content
        # if it's a dict-like
        if isinstance(completion, dict):
            try:
                return completion["choices"][0]["message"]["content"]
            except Exception:
                pass
            try:
                return completion["text"]
            except Exception:
                pass
        return str(completion)
    except Exception:
        return str(completion)


# ---------- Session management ----------
@app.post("/session/create")
async def session_create():
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = []
    return {"session_id": session_id}


@app.post("/session/append")
async def session_append(session_id: str = Form(...), role: str = Form(...), content: str = Form(...)):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")
    if role not in ("user", "assistant", "system"):
        raise HTTPException(status_code=400, detail="role must be 'user'|'assistant'|'system'")
    SESSIONS[session_id].append({"role": role, "content": content})
    return {"ok": True}


@app.get("/session/history")
async def session_history(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "history": SESSIONS[session_id]}


# ---------- Streaming helper (SSE-like) ----------
def start_stream_thread_and_queue(make_iterable_callable, queue: asyncio.Queue):
    """
    Run make_iterable_callable in a background thread, iterate it synchronously,
    and put text chunks into asyncio.Queue. Put None sentinel at the end.
    """

    def _worker():
        try:
            iterable = make_iterable_callable()
            for chunk in iterable:
                # attempt to extract text
                text = ""
                try:
                    # If chunk is object with choices -> delta -> content
                    if hasattr(chunk, "choices"):
                        c0 = chunk.choices[0]
                        delta = getattr(c0, "delta", None)
                        if delta is not None:
                            content = getattr(delta, "content", None)
                            if content is not None:
                                text = content
                        # sometimes chunk.choices[0].text
                        if not text and hasattr(c0, "text"):
                            text = c0.text
                    elif isinstance(chunk, dict):
                        try:
                            text = chunk["choices"][0]["delta"].get("content", "")
                        except Exception:
                            text = chunk.get("text", "") or str(chunk)
                    else:
                        text = str(chunk)
                except Exception:
                    text = str(chunk)
                # Put text (can be empty)
                asyncio.run_coroutine_threadsafe(queue.put(text), asyncio.get_event_loop())
        except Exception as e:
            # put error string to queue
            asyncio.run_coroutine_threadsafe(queue.put(f"[STREAM ERROR] {e}"), asyncio.get_event_loop())
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), asyncio.get_event_loop())

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


async def sse_event_generator(make_iterable_callable):
    """
    Async generator that yields SSE 'data: <json>\n\n' strings for each chunk.
    """
    q: asyncio.Queue = asyncio.Queue()
    start_stream_thread_and_queue(make_iterable_callable, q)

    while True:
        item = await q.get()
        if item is None:
            # close
            yield "event: done\ndata: [DONE]\n\n"
            break
        # safe JSON payload
        payload = {"delta": item}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------- Endpoints (streaming + non-streaming) ----------

@app.post("/stream_chat")
async def stream_chat(session_id: Optional[str] = Form(None),
                      user_input: str = Form(...),
                      model: str = Form("llama-3.3-70b-versatile"),
                      temperature: float = Form(0.7)):
    """
    Streamed chat endpoint (SSE). Usage example (curl):
      curl -N -X POST http://127.0.0.1:8001/stream_chat -F "user_input=Hello" -F "session_id=<id>"
    If session_id is provided, conversation history will be included in the prompt.
    """
    # Build messages from session if exists
    messages = []
    if session_id and session_id in SESSIONS:
        messages.extend(SESSIONS[session_id])
    # Append new user message (and optionally store it)
    messages.append({"role": "user", "content": user_input})
    if session_id:
        SESSIONS.setdefault(session_id, []).append({"role": "user", "content": user_input})

    # Groq SDK call must be created inside callable for the streaming helper
    def make_iterable():
        # Use the SDK to create a streaming completion iterable
        payload_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        return groq_client.chat.completions.create(
            model=model,
            messages=payload_messages,
            temperature=float(temperature),
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
        )

    return StreamingResponse(sse_event_generator(make_iterable), media_type="text/event-stream")


@app.post("/chat")
async def chat_nonstream(session_id: Optional[str] = Form(None),
                         user_input: str = Form(...),
                         model: str = Form("llama-3.3-70b-versatile"),
                         temperature: float = Form(0.7)):
    """
    Non-streaming chat (returns final text).
    """
    messages = []
    if session_id and session_id in SESSIONS:
        messages.extend(SESSIONS[session_id])
    messages.append({"role": "user", "content": user_input})
    if session_id:
        SESSIONS.setdefault(session_id, []).append({"role": "user", "content": user_input})
    try:
        completion = await asyncio.to_thread(lambda: groq_client.chat.completions.create(
            model=model,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=float(temperature),
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
        ))
        text = _extract_text_from_completion(completion)
        # store assistant reply in session
        if session_id:
            SESSIONS[session_id].append({"role": "assistant", "content": text})
        return {"response": text}
    except Exception as e:
        LOG.exception("chat_nonstream error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deep_reasoning")
async def deep_reasoning(user_input: str = Form(...),
                         session_id: Optional[str] = Form(None),
                         stream: Optional[bool] = Form(False)):
    """
    Deep reasoning endpoint. Set stream=true to stream responses (SSE).
    """
    model = "openai/gpt-oss-120b"
    messages = []
    if session_id and session_id in SESSIONS:
        messages.extend(SESSIONS[session_id])
    messages.append({"role": "user", "content": user_input})
    if session_id:
        SESSIONS.setdefault(session_id, []).append({"role": "user", "content": user_input})

    if stream:
        def make_iterable():
            return groq_client.chat.completions.create(
                model=model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=1,
                max_completion_tokens=8192,
                reasoning_effort="medium",
                stream=True,
            )
        return StreamingResponse(sse_event_generator(make_iterable), media_type="text/event-stream")
    else:
        try:
            completion = await asyncio.to_thread(lambda: groq_client.chat.completions.create(
                model=model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=1,
                max_completion_tokens=8192,
                reasoning_effort="medium",
                stream=False,
            ))
            text = _extract_text_from_completion(completion)
            if session_id:
                SESSIONS[session_id].append({"role": "assistant", "content": text})
            return {"response": text}
        except Exception as e:
            LOG.exception("deep_reasoning error")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/function_calling")
async def function_calling(user_input: str = Form(...), image_url: Optional[str] = Form(None),
                           session_id: Optional[str] = Form(None), stream: Optional[bool] = Form(False)):
    model = "meta-llama/llama-4-scout-17b-16e-instruct"
    content = [{"type": "text", "text": user_input}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages = [{"role":"user","content": content}]
    if session_id:
        SESSIONS.setdefault(session_id, []).append({"role": "user", "content": user_input})

    if stream:
        def make_iterable():
            return groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_completion_tokens=1024,
                stream=True,
            )
        return StreamingResponse(sse_event_generator(make_iterable), media_type="text/event-stream")
    else:
        try:
            completion = await asyncio.to_thread(lambda: groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_completion_tokens=1024,
                stream=False,
            ))
            text = _extract_text_from_completion(completion)
            if session_id:
                SESSIONS[session_id].append({"role":"assistant","content":text})
            return {"response": text}
        except Exception as e:
            LOG.exception("function_calling error")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/image_understanding")
async def image_understanding(user_input: str = Form(...), image_url: str = Form(...),
                              session_id: Optional[str] = Form(None)):
    model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    content = [
        {"type":"text","text": user_input},
        {"type":"image_url","image_url":{"url": image_url}}
    ]
    messages = [{"role":"user","content": content}]
    if session_id:
        SESSIONS.setdefault(session_id, []).append({"role": "user", "content": user_input})

    try:
        completion = await asyncio.to_thread(lambda: groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            stream=False,
        ))
        text = _extract_text_from_completion(completion)
        if session_id:
            SESSIONS[session_id].append({"role":"assistant","content":text})
        return {"response": text}
    except Exception as e:
        LOG.exception("image_understanding error")
        raise HTTPException(status_code=500, detail=str(e))


# External media endpoints (async)
@app.post("/image_generation")
async def image_generation(prompt: str = Form(...)):
    url = f"{FLUX_PRO_BASE.rstrip('/')}/flux-pro.php"
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(url, data={"text": prompt}, timeout=60)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        LOG.exception("image_generation error")
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/img_to_video")
@app.post("/image_to_video")
async def image_to_video(text: str = Form(...), link: str = Form(...)):
    url = f"{FLUX_PRO_BASE.rstrip('/')}/img-to-video.php"
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(url, data={"text": text, "link": link}, timeout=120)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        LOG.exception("image_to_video error")
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/text_to_video")
async def text_to_video(prompt: str = Form(...)):
    url = f"{FLUX_PRO_BASE.rstrip('/')}/text-to-video.php"
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(url, data={"text": prompt}, timeout=120)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        LOG.exception("text_to_video error")
        raise HTTPException(status_code=502, detail=str(e))


# TTS endpoint
@app.post("/tts")
async def tts(text: str = Form(...), voice: str = Form("Aaliyah-PlayAI"), lang: str = Form("en")):
    """
    Try Groq PlayAI (groq_client.audio.speech.create) with write_to_file or other flows.
    Fallback to gTTS if Groq TTS isn't available or fails.
    Returns FileResponse with audio file.
    """
    # first try Groq
    out_wav = unique_filename("tts", "wav")
    try:
        response = await asyncio.to_thread(lambda: groq_client.audio.speech.create(
            model="playai-tts",
            voice=voice,
            response_format="wav",
            input=text,
        ))
        # Many SDKs provide write_to_file on BinaryAPIResponse
        if hasattr(response, "write_to_file"):
            await asyncio.to_thread(lambda: response.write_to_file(str(out_wav)))
            return FileResponse(out_wav, media_type="audio/wav", filename=out_wav.name)
        elif isinstance(response, (bytes, bytearray)):
            out_wav.write_bytes(response)
            return FileResponse(out_wav, media_type="audio/wav", filename=out_wav.name)
        elif isinstance(response, dict) and response.get("url"):
            # download URL
            async with httpx.AsyncClient() as c:
                r = await c.get(response["url"], timeout=120)
                r.raise_for_status()
                out_wav.write_bytes(r.content)
                return FileResponse(out_wav, media_type="audio/wav", filename=out_wav.name)
        else:
            # write fallback textual representation
            out_wav.write_text(str(response))
            return FileResponse(out_wav, media_type="text/plain", filename=out_wav.name)
    except Exception as e:
        LOG.warning("Groq TTS failed: %s", e)

    # fallback: gTTS
    if GTTS_AVAILABLE:
        out_mp3 = unique_filename("tts", "mp3")
        try:
            await asyncio.to_thread(lambda: gTTS(text=text, lang="en" if lang.startswith("en") else "ar").save(str(out_mp3)))
            return FileResponse(out_mp3, media_type="audio/mpeg", filename=out_mp3.name)
        except Exception as e:
            LOG.exception("gTTS fallback failed: %s", e)
            raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    raise HTTPException(status_code=500, detail="TTS not available (Groq failed and gTTS not installed).")


# STT endpoint (Whisper)
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="file required")
    contents = await file.read()
    filename = unique_filename("upload", file.filename.split("/")[-1] or "audio")
    filename.write_bytes(contents)
    try:
        transcription = await asyncio.to_thread(lambda: groq_client.audio.transcriptions.create(
            file=(str(filename), contents),
            model="whisper-large-v3",
            response_format="verbose_json",
        ))
        # try common attributes
        if hasattr(transcription, "text"):
            return {"text": transcription.text}
        if isinstance(transcription, dict) and "text" in transcription:
            return {"text": transcription["text"]}
        return {"result": str(transcription)}
    except Exception as e:
        LOG.exception("stt error")
        raise HTTPException(status_code=500, detail=str(e))


# Multilingual non-stream chat
@app.post("/multilingual")
async def multilingual(user_input: str = Form(...), session_id: Optional[str] = Form(None)):
    try:
        return await chat_nonstream(session_id=session_id, user_input=user_input, model="llama-3.3-70b-versatile")
    except Exception as e:
        LOG.exception("multilingual error")
        raise HTTPException(status_code=500, detail=str(e))


# Moderation
@app.post("/moderation")
async def moderation(user_input: str = Form(...)):
    try:
        completion = await asyncio.to_thread(lambda: groq_client.chat.completions.create(
            model="meta-llama/llama-guard-4-12b",
            messages=[{"role":"user","content":user_input}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
        ))
        text = _extract_text_from_completion(completion)
        return {"response": text}
    except Exception as e:
        LOG.exception("moderation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
async def healthz():
    return {"status":"ok", "version":"1.2"}
