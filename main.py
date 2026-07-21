import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google import genai
from google.genai.types import GenerateContentConfig, SafetySetting

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
PUBLIC_DIR = BASE_DIR / "public"

API_KEY = os.getenv("GEMINI_API_KEY")
PORT = int(os.getenv("PORT", "3000"))

if not API_KEY:
    logger.error("❌ GEMINI_API_KEY is not set in environment variables")
    if os.getenv("NODE_ENV") == "production":
        logger.warning("⚠️ Running without Gemini API key - some features will not work")

client: Optional[genai.Client] = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
        logger.info("✅ Gemini API initialized successfully")
    except Exception as error:
        logger.error("❌ Failed to initialize Gemini: %s", str(error))

app = FastAPI(title="Resume ATS Tailor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if PUBLIC_DIR.exists():
    app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")


class TailorResumeRequest(BaseModel):
    resumeText: str
    jobDescriptions: Optional[str] = ""


class GenerateCVRequest(BaseModel):
    jobTitle: str
    keywords: List[str]
    currentResume: Optional[str] = ""


def _require_client() -> genai.Client:
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="Gemini API not configured. Please set GEMINI_API_KEY in environment variables.",
        )
    return client


def generate_with_fallback(prompt: str, temperature: float, max_output_tokens: int, top_p: float, top_k: int) -> str:
    active_client = _require_client()

    models_to_try = [
        "gemini-3.1-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro",
    ]

    safety_settings = [
        SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_MEDIUM_AND_ABOVE"),
        SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_MEDIUM_AND_ABOVE"),
    ]

    last_error: Optional[Exception] = None

    for model_name in models_to_try:
        try:
            logger.info("🔄 Trying model: %s", model_name)
            response = active_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    safety_settings=safety_settings,
                ),
            )

            text = (response.text or "").strip()
            if text:
                logger.info("✅ Success with model: %s", model_name)
                return text
            raise RuntimeError("Empty response from model")

        except Exception as error:
            message = str(error)
            logger.info("❌ Failed with %s: %s", model_name, message)
            last_error = error

            lower_message = message.lower()
            if "404" in lower_message or "not found" in lower_message:
                continue
            if "api key" in lower_message or "auth" in lower_message:
                break

    raise HTTPException(status_code=500, detail=str(last_error or "All models failed to generate content"))


@app.post("/api/tailor-resume")
async def tailor_resume(payload: TailorResumeRequest):
    if not payload.resumeText or not payload.resumeText.strip():
        raise HTTPException(status_code=400, detail="Resume text is required")

    logger.info("📝 Tailoring resume for job descriptions: %s", payload.jobDescriptions)

    prompt = f"""
You are an expert ATS (Applicant Tracking System) resume optimizer.

Original Resume:
{payload.resumeText}

Job Context (Available positions):
{payload.jobDescriptions or 'General tech positions'}

Task: Optimize this resume for ATS systems.

Instructions:
1. Identify and incorporate relevant keywords from the job descriptions
2. Use strong action verbs and quantifiable achievements
3. Format with clear sections: Summary, Skills, Experience, Education
4. Remove any formatting that might confuse ATS
5. Highlight relevant skills and experience
6. Add a professional summary at the top

Return ONLY the optimized resume text with clear section headers.
Do not add any extra commentary or explanation.
""".strip()

    tailored_text = generate_with_fallback(
        prompt=prompt,
        temperature=0.7,
        max_output_tokens=1500,
        top_p=0.8,
        top_k=40,
    )

    return {"success": True, "tailored": tailored_text}


@app.post("/api/generate-cv")
async def generate_cv(payload: GenerateCVRequest):
    if not payload.jobTitle or not payload.keywords:
        raise HTTPException(status_code=400, detail="Job title and keywords are required")

    logger.info("📄 Generating CV for: %s", payload.jobTitle)

    current_resume_section = (
        f"Current Resume (for reference, use as base): {payload.currentResume}" if payload.currentResume else ""
    )

    prompt = f"""
Create a professional, ATS-friendly CV.

Job Title: {payload.jobTitle}
Required Keywords: {', '.join(payload.keywords)}
{current_resume_section}

Instructions:
1. Incorporate ALL the provided keywords naturally
2. Use a clean, professional format
3. Include these sections:
   - Professional Summary (3-4 sentences)
   - Core Competencies (bullet points with keywords)
   - Professional Experience (2-3 roles with achievements)
   - Education
   - Certifications (if relevant)
4. Use strong action verbs and quantifiable achievements
5. Tailor specifically for the {payload.jobTitle} role
6. Keep it between 400-600 words

Return ONLY the CV text with clear section headers.
Do not add any extra commentary.
""".strip()

    cv_text = generate_with_fallback(
        prompt=prompt,
        temperature=0.8,
        max_output_tokens=1500,
        top_p=0.9,
        top_k=40,
    )

    return {"success": True, "cv": cv_text}


@app.get("/api/test")
async def api_test():
    try:
        result = generate_with_fallback(
            prompt="Say 'API is working' in one sentence.",
            temperature=0.1,
            max_output_tokens=50,
            top_p=0.8,
            top_k=40,
        )
        return {
            "success": True,
            "message": "API is working",
            "response": result,
            "apiKeyConfigured": bool(API_KEY),
            "models": ["gemini-3.1-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        }
    except HTTPException as error:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": error.detail,
                "apiKeyConfigured": bool(API_KEY),
            },
        )


@app.get("/api/health")
async def health_check():
    return {
        "status": "OK",
        "message": "Gemini API proxy is running",
        "apiKeyConfigured": bool(API_KEY),
        "environment": os.getenv("NODE_ENV", "development"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
async def serve_index():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(INDEX_FILE))


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("❌ Unhandled error: %s", str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("\n🚀 Server running on http://localhost:%s", PORT)
    logger.info("📡 API endpoint: http://localhost:%s/api/tailor-resume", PORT)
    logger.info("🔍 Test endpoint: http://localhost:%s/api/test", PORT)
    logger.info("✅ Gemini API %s", "configured ✅" if API_KEY else "MISSING ❌")
    logger.info("🌐 Models available: gemini-3.1-flash-lite, gemini-1.5-flash, gemini-1.5-pro, gemini-pro")
    logger.info("\n💡 To test the API, visit: http://localhost:%s/api/test\n", PORT)

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
