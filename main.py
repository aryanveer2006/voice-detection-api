from fastapi import FastAPI, Header, HTTPException
import base64, io, os
import numpy as np

app = FastAPI()

API_KEY = os.getenv("API_KEY")

SUPPORTED_LANGUAGES = ["en", "hi", "ta", "ml", "te"]

@app.post("/detect-voice")
async def detect_voice(data: dict, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if data.get("language") not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    try:
        import librosa  # 👈 moved inside
        audio_bytes = base64.b64decode(data["audio_base64"])
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=None)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid audio")

    score = np.mean(librosa.feature.mfcc(y=y, sr=sr))

    if score < -150:
        return {"classification": "AI_GENERATED", "confidence": 0.90}
    else:
        return {"classification": "HUMAN", "confidence": 0.85}
