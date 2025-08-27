#!/usr/bin/env python3
"""
ai_stream.py

Improved single-file FastAPI backend with:
 - SSE-style streaming endpoints that reliably emit chunks
 - CORS enabled for browser usage (image generation from web clients)
 - All original endpoints preserved (chat, stream_chat, deep_reasoning,
   function_calling, image_understanding, image_generation, image_to_video,
   text_to_video, tts, stt, multilingual, moderation, session management, health)
 - Robust handling for different Groq SDK return shapes and TTS fallbacks

Run (termux / local):
  uvicorn ai_stream:app --reload --host 0.0.0.0 --port 8001

When deploying (Render / other), start with:
  uvicorn ai_stream:app --host 0.0.0.0 --port $PORT
"""

import os
import uuid
import json
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Groq SDK (ensure correct version installed)
from groq import Groq

# optional fallback TTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# ---------- Configuration ----------
LOG = logging.getLogger("ai_stream")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
FLUX_PRO_BASE = os.getenv("FLUX_PRO_BASE", "https://sii3.moayman.top/api")
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "./media"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# CORS config: allow browser clients. Use env var ALLOW_ORIGINS to restrict in production.
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",") if os.getenv("ALLOW_ORIGINS") else ["*"]

# Initialize Groq client (SDK) defensively
try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else Groq()
except TypeError:
    groq_client = Groq()

# Global HTTPX client reused on startup
httpx_client: Optional[httpx.AsyncClient] = None

app = FastAPI(title="AI Streaming Backend (improved)", version="1.3")

# Add CORS middleware so browser fetches won't fail
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory sessions: {session_id: [ {"role":"user"/"assistant"/"system","content":str}, ... ]}
SESSIONS: Dict[str, List[Dict[str, str]]] = {}

# ---------- Utilities ----------
def unique_filename(prefix: str, ext: str) -> Path:
    return MEDIA_DIR / f"{prefix}-{uuid.uuid4().hex}.{ext.lstrip('.')}"

async def _startup_event():
    global httpx_client
    httpx_client = httpx.AsyncClient(timeout=60.0)
    LOG.info("HTTPX client created")

async def _shutdown_event():
    global httpx_client
    if httpx_client:
        await httpx_client.aclose()
        LOG.info("HTTPX client closed")

app.add_event_handler("startup", _startup_event)
app.add_event_handler("shutdown", _shutdown_event)


def _extract_text_from_completion(completion: Any) -> str:
    """
    Defensive extraction of text from various possible SDK response shapes.
    """
    try:
        # object-like with choices
        if completion is None:
            return ""
        if hasattr(completion, "choices"):
            ch = completion.choices
            if isinstance(ch, (list, tuple)) and len(ch) > 0:
                c0 = ch[0]
                # common shapes
                if hasattr(c0, "message") and getattr(c0.message, "content", None) is not None:
                    return c0.message.content
                if getattr(c0, "text", None) is not None:
                    return c0.text
                # streaming delta
                if getattr(c0, "delta", None) and getattr(c0.delta, "content", None) is not None:
                    return c0.delta.content
        # dict-like shapes
        if isinstance(completion, dict):
            # try message
            try:
                return completion["choices"][0]["message"]["content"]
            except Exception:
                pass
            if "text" in completion:
                return completion["text"]
            # try nested forms
            if "choices" in completion and completion["choices"]:
                ch0 = completion["choices"][0]
                if isinstance(ch0, dict) and "text" in ch0:
                    return ch0["text"]
        # fallback string
        return str(completion)
    except Exception:
        return str(completion)


def _text_from_chunk(chunk: Any) -> str:
    """
    Extract content text from streaming chunk objects or dicts.
    """
    try:
        if chunk is None:
            return ""
        # if object with choices and delta
        if hasattr(chunk, "choices"):
            choices = getattr(chunk, "choices")
            if choices and len(choices) > 0:
                c0 = choices[0]
                # delta content
                delta = getattr(c0, "delta", None)
                if delta is not None:
                    cont = getattr(delta, "content", None)
                    if cont is not None:
                        return cont
                # direct text
                if getattr(c0, "text", None) is not None:
                    return c0.text
                # message
                msg = getattr(c0, "message", None)
                if msg is not None and getattr(msg, "content", None) is not None:
                    return msg.content
        # dict shape
        if isinstance(chunk, dict):
            # try several keys
            if "choices" in chunk and chunk["choices"]:
                try:
                    # streaming delta dict
                    delta = chunk["choices"][0].get("delta", {})
                    if isinstance(delta, dict) and delta.get("content"):
                        return delta.get("content")
                except Exception:
                    pass
                # try text fields
                try:
                    return chunk["choices"][0].get("text") or chunk["choices"][0].get("message", {}).get("content", "") or str(chunk)
                except Exception:
                    pass
            if "text" in chunk:
                return chunk["text"]
            # fallback to str
            return str(chunk)
        # bytes
        if isinstance(chunk, (bytes, bytearray)):
            try:
                return chunk.decode("utf-8", errors="ignore")
            except Exception:
                return str(chunk)
        # fallback
        return str(chunk)
    except Exception as e:
        LOG.debug("Error extracting chunk text: %s", e)
        return str(chunk)


# ---------- Session management ----------
@app.post("/session/create")
async def session_create():
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = []
    LOG.info("Created session %s", session_id)
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
def start_stream_thread_and_queue(make_iterable_callable: Callable[[], Any],
                                  q: asyncio.Queue,
                                  main_loop: asyncio.AbstractEventLoop):
    """
    Start a background thread to iterate the (possibly sync) iterable returned by
    make_iterable_callable and push text chunks into the provided asyncio.Queue.
    Use main_loop to schedule queue.put coroutines so they are executed in the
    main event loop (the one serving the client).
    """

    def _worker():
        try:
            iterable = make_iterable_callable()
            # If the iterable is an async generator/coro-like, run it in a new loop here
            if hasattr(iterable, "__aiter__") or asyncio.iscoroutine(iterable):
                # run async iteration in a separate loop inside this thread
                try:
                    async def _run_async_iter():
                        # if it's a coroutine that returns an async generator, await it
                        gen = await iterable if asyncio.iscoroutine(iterable) else iterable
                        async for chunk in gen:
                            text = _text_from_chunk(chunk)
                            asyncio.run_coroutine_threadsafe(q.put(text), main_loop)
                    asyncio.run(_run_async_iter())
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(q.put(f"[STREAM ERROR] {exc}"), main_loop)
            else:
                # synchronous iterable/generator
                for chunk in iterable:
                    text = _text_from_chunk(chunk)
                    asyncio.run_coroutine_threadsafe(q.put(text), main_loop)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(q.put(f"[STREAM ERROR] {exc}"), main_loop)
        finally:
            # sentinel None to indicate close
            asyncio.run_coroutine_threadsafe(q.put(None), main_loop)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


async def sse_event_generator(make_iterable_callable: Callable[[], Any]):
    """
    Async generator that yields SSE 'data: <json>\n\n' strings for each chunk.
    We capture the running loop and pass it to the background thread so run_coroutine_threadsafe
    has the right target loop.
    """
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    start_stream_thread_and_queue(make_iterable_callable, q, loop)

    while True:
        item = await q.get()
        if item is None:
            yield "event: done\ndata: [DONE]\n\n"
            break
        # build a JSON payload per SSE best practice
        payload = {"delta": item}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------- Endpoints (streaming + non-streaming) ----------

@app.post("/stream_chat")
async def stream_chat(session_id: Optional[str] = Form(None),
                      user_input: str = Form(...),
                      model: str = Form("llama-3.3-70b-versatile"),
                      temperature: float = Form(0.7)):
    """
    Streamed chat endpoint (SSE).
    Example:
      curl -N -X POST http://127.0.0.1:8001/stream_chat -F "user_input=Hello"
    """
    messages = []
    if session_id and session_id in SESSIONS:
        messages.extend(SESSIONS[session_id])
    messages.append({"role": "user", "content": user_input})
    if session_id:
        SESSIONS.setdefault(session_id, []).append({"role": "user", "content": user_input})

    def make_iterable():
        payload_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        return groq_client.chat.completions.create(
            model=model,
            messages=payload_messages,
            temperature=float(temperature),
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
        )

    return StreamingResponse(sse_event_generator(make_iterable), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/chat")
async def chat_nonstream(session_id: Optional[str] = Form(None),
                         user_input: str = Form(...),
                         model: str = Form("llama-3.3-70b-versatile"),
                         temperature: float = Form(0.7)):
    messages = []
    if session_id and session_id in SESSIONS:
        messages.extend(SESSIONS[session_id])
    messages.append({"role": "user", "content": user_input})
    if session_id:
        SESSIONS.setdefault(session_id, []).append({"role": "user", "content": user_input})
    try:
        # run blocking SDK call in thread to avoid blocking event loop
        completion = await asyncio.to_thread(lambda: groq_client.chat.completions.create(
            model=model,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=float(temperature),
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
        ))
        text = _extract_text_from_completion(completion)
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
        return StreamingResponse(sse_event_generator(make_iterable), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
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
    messages = [{"role": "user", "content": content}]
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
        return StreamingResponse(sse_event_generator(make_iterable), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
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
                SESSIONS[session_id].append({"role": "assistant", "content": text})
            return {"response": text}
        except Exception as e:
            LOG.exception("function_calling error")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/image_understanding")
async def image_understanding(user_input: str = Form(...), image_url: str = Form(...),
                              session_id: Optional[str] = Form(None)):
    model = "meta-llama/llama-4-maverick-17b-128e-instruct"
    content = [
        {"type": "text", "text": user_input},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    messages = [{"role": "user", "content": content}]
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
            SESSIONS[session_id].append({"role": "assistant", "content": text})
        return {"response": text}
    except Exception as e:
        LOG.exception("image_understanding error")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------
# External media endpoints (async)
# ------------------------
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


# ------------------------
# TTS endpoint (Groq PlayAI first, fallback to gTTS)
# ------------------------
@app.post("/tts")
async def tts(text: str = Form(...), voice: str = Form("Aaliyah-PlayAI"), lang: str = Form("en")):
    out_wav = unique_filename("tts", "wav")
    # Try Groq-based TTS
    try:
        response = await asyncio.to_thread(lambda: groq_client.audio.speech.create(
            model="playai-tts",
            voice=voice,
            response_format="wav",
            input=text,
        ))
        # Some SDK responses expose write_to_file or write_file methods
        if hasattr(response, "write_to_file"):
            await asyncio.to_thread(lambda: response.write_to_file(str(out_wav)))
            return FileResponse(out_wav, media_type="audio/wav", filename=out_wav.name)
        if hasattr(response, "write_file"):
            await asyncio.to_thread(lambda: response.write_file(str(out_wav)))
            return FileResponse(out_wav, media_type="audio/wav", filename=out_wav.name)
        # If bytes-like
        if isinstance(response, (bytes, bytearray)):
            out_wav.write_bytes(response)
            return FileResponse(out_wav, media_type="audio/wav", filename=out_wav.name)
        # If dict with URL -> download
        if isinstance(response, dict) and response.get("url"):
            async with httpx.AsyncClient() as c:
                r = await c.get(response["url"], timeout=120)
                r.raise_for_status()
                out_wav.write_bytes(r.content)
                return FileResponse(out_wav, media_type="audio/wav", filename=out_wav.name)
        # Otherwise write textual representation so user sees something
        out_wav.write_text(str(response))
        return FileResponse(out_wav, media_type="text/plain", filename=out_wav.name)
    except Exception as e:
        LOG.warning("Groq TTS failed: %s", e)

    # fallback to gTTS (if available)
    if GTTS_AVAILABLE:
        out_mp3 = unique_filename("tts", "mp3")
        try:
            await asyncio.to_thread(lambda: gTTS(text=text, lang="en" if lang.startswith("en") else "ar").save(str(out_mp3)))
            return FileResponse(out_mp3, media_type="audio/mpeg", filename=out_mp3.name)
        except Exception as e:
            LOG.exception("gTTS fallback failed: %s", e)
            raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    raise HTTPException(status_code=500, detail="TTS not available (Groq failed and gTTS not installed).")


# ------------------------
# STT endpoint (Whisper)
# ------------------------
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
        if hasattr(transcription, "text"):
            return {"text": transcription.text}
        if isinstance(transcription, dict) and "text" in transcription:
            return {"text": transcription["text"]}
        return {"result": str(transcription)}
    except Exception as e:
        LOG.exception("stt error")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------
# Multilingual wrapper
# ------------------------
@app.post("/multilingual")
async def multilingual(user_input: str = Form(...), session_id: Optional[str] = Form(None)):
    try:
        return await chat_nonstream(session_id=session_id, user_input=user_input, model="llama-3.3-70b-versatile")
    except Exception as e:
        LOG.exception("multilingual error")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------
# Moderation
# ------------------------
@app.post("/moderation")
async def moderation(user_input: str = Form(...)):
    try:
        completion = await asyncio.to_thread(lambda: groq_client.chat.completions.create(
            model="meta-llama/llama-guard-4-12b",
            messages=[{"role": "user", "content": user_input}],
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
    return {"status": "ok", "version": "1.3"}

# End of file
