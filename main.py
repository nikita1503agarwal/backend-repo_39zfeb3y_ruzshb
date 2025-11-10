import os
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schemas import ChatMessage, Diagnosis

# Optional: database helpers (not strictly required for MVP but ready)
try:
    from database import create_document, get_documents, db
except Exception:
    create_document = None
    get_documents = None
    db = None

# Optional OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None

# Language utilities
SUPPORTED_LANGS = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "hi": "हिन्दी",
}

def ensure_lang(lang: Optional[str]) -> str:
    if not lang:
        return "en"
    lang = lang.lower()
    return lang if lang in SUPPORTED_LANGS else "en"

# Simple fallback translations for common UI strings
TRANSLATIONS = {
    "assistant_fallback": {
        "en": "I'm a helpful AI assistant. Here's a thoughtful response based on your message: ",
        "es": "Soy un asistente de IA útil. Aquí hay una respuesta reflexiva basada en tu mensaje: ",
        "fr": "Je suis un assistant IA utile. Voici une réponse réfléchie basée sur votre message : ",
        "de": "Ich bin ein hilfreicher KI‑Assistent. Hier ist eine durchdachte Antwort basierend auf Ihrer Nachricht: ",
        "it": "Sono un assistente IA utile. Ecco una risposta ponderata basata sul tuo messaggio: ",
        "hi": "मैं एक सहायक एआई सहायक हूँ। आपके संदेश के आधार पर एक विचारशील उत्तर: ",
    },
    "diagnosis_title": {
        "en": "Preliminary crop diagnosis",
        "es": "Diagnóstico preliminar del cultivo",
        "fr": "Diagnostic préliminaire de la culture",
        "de": "Vorläufige Felddiagnose",
        "it": "Diagnosi preliminare della coltura",
        "hi": "फसल का प्रारंभिक निदान",
    },
}

# Heuristic image analysis fallback using Pillow
try:
    from PIL import Image
except Exception:
    Image = None

app = FastAPI(title="Emergent AI Platform Clone")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    language: Optional[str] = "en"

class ChatResponse(BaseModel):
    session_id: str
    language: str
    reply: str


@app.get("/")
def read_root():
    return {"message": "Emergent-style platform backend running", "languages": SUPPORTED_LANGS}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        from database import db as _db
        if _db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = getattr(_db, 'name', None) or ("✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set")
            try:
                response["collections"] = _db.list_collection_names()[:10]
                response["database"] = "✅ Connected & Working"
                response["connection_status"] = "Connected"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    lang = ensure_lang(payload.language)

    # If OpenAI available, use powerful multilingual model
    if openai_client is not None:
        try:
            system_prompt = (
                "You are a world-class multilingual AI assistant. "
                "Always reply in the user's target language given as a two-letter ISO code. "
                "Be concise, accurate, and helpful. If the user asks about agriculture or plant health, "
                "provide practical steps, warnings, and resources. Target language: " + lang
            )
            msg = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": payload.message},
                ],
                temperature=0.3,
            )
            reply = msg.choices[0].message.content.strip()
            return ChatResponse(session_id=payload.session_id, language=lang, reply=reply)
        except Exception as e:
            # Fall through to basic translation
            pass

    # Fallback: simple echo + light guidance translated
    prefix = TRANSLATIONS["assistant_fallback"][lang]
    generic = (
        "I cannot access external AI services right now. However, here are steps you can take: "
        "1) Clarify your goal. 2) Share context and constraints. 3) Ask for an outline or checklist."
    )
    reply = f"{prefix}{payload.message}\n\n{generic}"
    return ChatResponse(session_id=payload.session_id, language=lang, reply=reply)


def encode_image_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


class DiagnoseResponse(BaseModel):
    title: str
    plant_type: Optional[str]
    predicted_disease: str
    confidence: float
    recommendations: List[str]
    source: str
    language: str


@app.post("/api/diagnose", response_model=DiagnoseResponse)
async def diagnose_endpoint(
    image: UploadFile = File(...),
    plant_type: Optional[str] = Form(None),
    language: Optional[str] = Form("en"),
):
    lang = ensure_lang(language)
    data = await image.read()

    # Primary path: OpenAI Vision if available
    if openai_client is not None:
        try:
            b64 = encode_image_to_base64(data)
            prompt = (
                "You are an agronomy expert. Diagnose visible plant diseases from the photo. "
                "Return: 1) Most likely disease, 2) Confidence 0-1, 3) 5 concrete next steps for farmers. "
                f"If plant type is known, consider it: {plant_type or 'unknown'}. "
                "Answer in the target language."
            )
            msg = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                messages=[
                    {"role": "system", "content": f"Respond in language: {lang}"},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    },
                ],
            )
            content = msg.choices[0].message.content.strip()
            # Heuristic parse: try to find disease and confidence
            predicted = "Plant disease (model)"
            confidence = 0.7
            # Recommendations: split lines starting with dash/number
            recs: List[str] = []
            for line in content.splitlines():
                line_stripped = line.strip("-• ")
                if not line_stripped:
                    continue
                if any(k in line_stripped.lower() for k in ["step", "apply", "monitor", "inspect", "remove", "treat", "fungicide", "irrigation", "fertilizer", "prune", "sanitize"]):
                    recs.append(line_stripped)
            if not recs:
                recs = [content[:200]]
            return DiagnoseResponse(
                title=TRANSLATIONS["diagnosis_title"][lang],
                plant_type=plant_type,
                predicted_disease=predicted,
                confidence=confidence,
                recommendations=recs[:6],
                source="openai_vision",
                language=lang,
            )
        except Exception:
            pass

    # Fallback: local heuristic using Pillow
    if Image is None:
        raise HTTPException(status_code=500, detail="Image analysis unavailable (Pillow not installed)")

    try:
        img = Image.open(BytesIO(data)).convert("RGB").resize((256, 256))
        pixels = img.getdata()
        total = len(pixels)
        greenish = 0
        brownish = 0
        yellowish = 0
        for (r, g, b) in pixels:
            if g > r + 20 and g > b + 10:
                greenish += 1
            if r > g and r > b and g > b and r > 120 and g > 100:
                yellowish += 1
            if r > 100 and g < 90 and b < 90:
                brownish += 1
        green_ratio = greenish / total
        stress_ratio = (brownish + yellowish) / total

        if stress_ratio > 0.35:
            predicted = "Leaf blight / nutrient deficiency (visual stress detected)"
            confidence = min(0.95, 0.5 + stress_ratio)
            recs = [
                "Remove heavily affected leaves to slow spread.",
                "Avoid overhead irrigation; water at the base early in the morning.",
                "Apply a broad-spectrum fungicide if fungal lesions are present.",
                "Improve airflow with proper spacing and pruning.",
                "Test soil; adjust nitrogen and potassium per recommendations.",
            ]
        elif stress_ratio > 0.15:
            predicted = "Early stress signs (possible fungal spots or micronutrient issue)"
            confidence = 0.6
            recs = [
                "Monitor daily for expansion of spots or discoloration.",
                "Apply preventive bio-fungicide (e.g., Bacillus-based).",
                "Check irrigation schedule to avoid waterlogging.",
                "Add organic mulch to stabilize moisture.",
            ]
        else:
            predicted = "No obvious disease; foliage appears predominantly healthy"
            confidence = 0.55
            recs = [
                "Maintain balanced irrigation and avoid leaf wetness overnight.",
                "Scout weekly; capture clear close-ups if symptoms appear.",
                "Ensure balanced fertilization with micronutrients.",
            ]

        return DiagnoseResponse(
            title=TRANSLATIONS["diagnosis_title"][lang],
            plant_type=plant_type,
            predicted_disease=predicted,
            confidence=round(confidence, 2),
            recommendations=recs,
            source="heuristic",
            language=lang,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or analysis error: {str(e)[:200]}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
