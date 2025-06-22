from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import base64
import io
import soundfile as sf
from uuid import uuid4
import numpy as np
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import json
import logging
import requests
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

app = FastAPI()

TTS_URL = "http://localhost:49999/cosyvoice"
LLM_URL = "http://localhost:20056/v1"
SLM_URL = "http://localhost:20066/v1"


@app.get("/")
async def index():
    html = open("static/frontend.html").read()
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory="static", html=True), name="static")

llm_client = OpenAI(
    base_url=LLM_URL,
    api_key="EMPTY",
)

slm_client = OpenAI(
    base_url=SLM_URL,
    api_key="EMPTY",
)


# event handlers for legochat
async def general_ping_handler(websocket: WebSocket, data: dict):
    await websocket.send_text(json.dumps({"type": data["type"], "data": "pong", "id": data["id"]}))


# event handlers for llm
async def llm_ping_handler(websocket: WebSocket, data: dict):
    try:
        models = llm_client.models.list()
        await websocket.send_text(json.dumps({"type": data["type"], "data": "pong", "id": data["id"]}))
    except:
        await websocket.send_text(json.dumps({"type": data["type"], "data": "error", "id": data["id"]}))


async def llm_invoke_handler(websocket: WebSocket, data: dict):
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": data["prompt"]}], stream=True
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            await websocket.send_text(
                json.dumps({"type": data["type"], "delta": delta, "done": False, "id": data["id"]})
            )
        await websocket.send_text(json.dumps({"type": data["type"], "delta": "", "done": True, "id": data["id"]}))
    except Exception as e:
        await websocket.send_text(
            json.dumps({"type": data["type"], "delta": "[ERROR]", "done": True, "id": data["id"]})
        )


async def slm_ping_handler(websocket: WebSocket, data: dict):
    try:
        models = slm_client.models.list()
        await websocket.send_text(json.dumps({"type": data["type"], "data": "pong", "id": data["id"]}))
    except:
        await websocket.send_text(json.dumps({"type": data["type"], "data": "error", "id": data["id"]}))


wav, sr = sf.read("./static/guess_age_gender.wav")
wav = (wav * 32767).astype(np.int16)
audio_bytes = wav.tobytes()


async def slm_invoke_handler(websocket: WebSocket, data: dict):
    try:
        messages = []
        if data["prompt"]:
            messages.append({"role": "user", "content": [{"type": "text", "text": data["prompt"]}]})
        else:
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            buffer = io.BytesIO()
            sf.write(buffer, audio, samplerate=sr, format="wav")
            audio_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}}],
                }
            )
        response = slm_client.chat.completions.create(model="gpt-4o", messages=messages, stream=True)
        for chunk in response:
            delta = chunk.choices[0].delta.content
            await websocket.send_text(
                json.dumps({"type": data["type"], "delta": delta, "done": False, "id": data["id"]})
            )
        await websocket.send_text(json.dumps({"type": data["type"], "delta": "", "done": True, "id": data["id"]}))
    except Exception as e:
        await websocket.send_text(
            json.dumps({"type": data["type"], "delta": "[ERROR]", "done": True, "id": data["id"]})
        )


async def tts_ping_handler(websocket: WebSocket, data: dict):
    try:
        requests.get(TTS_URL)
        await websocket.send_text(json.dumps({"type": data["type"], "data": "pong", "id": data["id"]}))
    except:
        await websocket.send_text(json.dumps({"type": data["type"], "data": "error", "id": data["id"]}))


# event handlers for tts
async def tts_handler(websocket: WebSocket, data: dict):
    if data["type"] == "tts":
        test_id = data["id"]
        await websocket.send_text(json.dumps({"event": "first-byte", "id": test_id}))

        async for chunk in stream_wav_pcm("tts_sample.wav", chunk_size=1024):
            await websocket.send_bytes(chunk)
            await asyncio.sleep(0.02)  # simulate real-time

        await websocket.send_text(json.dumps({"event": "done", "id": test_id}))


async def tts_invoke_handler(websocket: WebSocket, data: dict):
    text = data["text"]
    control_params = data.get("control_params", {})
    control_params = control_params.copy()
    text = text.strip()
    if not text:
        raise StopIteration("No text to synthesize")

    control_params = {"speed": "default", "emotion": "default", "voice": "default"}
    control_params["stream"] = True
    control_params["text_frontend"] = True
    control_params["gen_text"] = text
    control_params["response_id"] = uuid4().hex
    control_params["dtype"] = "np.int16"
    control_params["ref_text"] = control_params.pop("transcript", "")

    control_params["speech"] = b"0x00" * 16000  # Placeholder for PCM audio data

    try:
        files = {"ref_audio": pcm_to_wav(control_params.pop("speech"))}
    except Exception as e:
        logging.error(f"Error converting PCM to WAV: {e}")
        files = {}

    data = {"params": json.dumps(control_params)}
    with requests.post(TTS_URL, files=files, data=data, stream=True) as response:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                await websocket.send_bytes(chunk)


event_handles = {
    "tts": tts_handler,
    "general_ping": general_ping_handler,
    "llm_ping": llm_ping_handler,
    "llm_invoke": llm_invoke_handler,
    "slm_ping": slm_ping_handler,
    "slm_invoke": slm_invoke_handler,
    "tts_ping": tts_ping_handler,
    "tts_invoke": tts_invoke_handler,
}


@app.websocket("/stream/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive()

            if message["type"] != "websocket.receive":
                continue

            if "text" in message:
                data = json.loads(message["text"])
                asyncio.create_task(event_handles.get(data["type"], lambda ws, d: None)(websocket, data))

            elif "bytes" in message:
                audio_bytes = message["bytes"]

            if data["type"] == "tts":
                test_id = data["id"]
                await websocket.send_text(json.dumps({"event": "first-byte", "id": test_id}))

                async for chunk in stream_wav_pcm("tts_sample.wav", chunk_size=1024):
                    await websocket.send_bytes(chunk)
                    await asyncio.sleep(0.02)  # simulate real-time

                await websocket.send_text(json.dumps({"event": "done", "id": test_id}))
    except WebSocketDisconnect:
        print("Client disconnected")


async def stream_wav_pcm(wav_path: str, chunk_size=1024):
    import torchaudio

    waveform, sr = torchaudio.load(wav_path)
    waveform = torchaudio.functional.resample(waveform, sr, 16000) if sr != 16000 else waveform
    mono = waveform.mean(dim=0)  # downmix to mono if stereo
    audio_int16 = (mono.numpy() * 32768).clip(-32768, 32767).astype(np.int16)

    cursor = 0
    while cursor < len(audio_int16):
        chunk = audio_int16[cursor : cursor + chunk_size]
        yield chunk.tobytes()
        cursor += chunk_size
        await asyncio.sleep(0)


def pcm_to_wav(pcm_bytes, sample_rate=16000, num_channels=1, bits_per_sample=16):
    import io
    import wave

    byte_io = io.BytesIO()
    with wave.open(byte_io, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return byte_io.getvalue()


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
