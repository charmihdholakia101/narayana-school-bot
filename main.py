"""
Narayana Educational Institutions Voice Bot — FastAPI Backend
- Uses boto3 with long-term AWS credentials (never expires)
- Scrapes narayanagroup.com on startup
- Answers in English + Telugu
- Handles 25 concurrent users
"""

import os
import io
import json
import base64
import asyncio
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import boto3
import httpx
from gtts import gTTS
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG — long-term credentials via boto3
# ─────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID              = "us.amazon.nova-lite-v1:0"

LLM_SEMAPHORE = asyncio.Semaphore(15)
SCHOOL_CONTEXT = ""
executor = ThreadPoolExecutor(max_workers=10)

SCHOOL_URLS = [
    "https://narayanagroup.com/",
    "https://narayanagroup.com/academics/",
    "https://narayanagroup.com/centers/",
    "https://narayanagroup.com/experience-narayana/",
    "https://narayanagroup.com/digital/",
    "https://narayanagroup.com/nsports/",
    "https://narayanagroup.com/uniforms/",
    "https://narayanagroup.com/careers/",
    "https://narayanagroup.com/contact-us/",
    "https://narayanagroup.com/blog/",
]

http_client: Optional[httpx.AsyncClient] = None

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(title="Narayana School Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


@app.on_event("startup")
async def startup():
    global http_client, SCHOOL_CONTEXT
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_connections=30, max_keepalive_connections=20),
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SchoolBot/1.0)"},
    )
    log.info("Scraping Narayana website...")
    SCHOOL_CONTEXT = await scrape_all_pages()
    log.info(f"School context loaded: {len(SCHOOL_CONTEXT):,} chars")


@app.on_event("shutdown")
async def shutdown():
    await http_client.aclose()
    executor.shutdown(wait=False)


# ─────────────────────────────────────────────
# SCRAPER
# ─────────────────────────────────────────────

def extract_text_from_html(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "img", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return f"\n\n=== PAGE: {url} ===\n" + "\n".join(lines)


async def scrape_page(url: str) -> str:
    try:
        resp = await http_client.get(url)
        resp.raise_for_status()
        return extract_text_from_html(resp.text, url)
    except Exception as e:
        log.warning(f"Failed to scrape {url}: {e}")
        return ""


async def scrape_all_pages() -> str:
    results = await asyncio.gather(*[scrape_page(u) for u in SCHOOL_URLS])
    combined = "\n".join(r for r in results if r)
    return combined[:80000]


@app.post("/rescrape")
async def rescrape():
    global SCHOOL_CONTEXT
    SCHOOL_CONTEXT = await scrape_all_pages()
    return {"status": "ok", "chars": len(SCHOOL_CONTEXT)}


# ─────────────────────────────────────────────
# BEDROCK LLM (boto3 — never expires)
# ─────────────────────────────────────────────

def _call_bedrock_sync(system: str, user: str, max_tokens: int) -> str:
    client = get_bedrock_client()
    payload = {
        "messages": [
            {"role": "user", "content": [{"text": f"{system}\n\n{user}"}]}
        ],
        "inferenceConfig": {"max_new_tokens": max_tokens},
    }
    resp = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    return result["output"]["message"]["content"][0]["text"].strip()


async def call_claude(system: str, user: str, max_tokens: int = 600) -> str:
    async with LLM_SEMAPHORE:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                executor, _call_bedrock_sync, system, user, max_tokens
            )
        except Exception as e:
            log.error(f"Bedrock error: {e}")
            raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")


async def detect_language(text: str) -> str:
    result = await call_claude(
        "Detect language. Reply ONLY one word: 'en' for English, 'te' for Telugu.",
        text, max_tokens=5,
    )
    r = result.strip().lower()
    return r if r in ("en", "te") else "en"


async def translate_to_english(text: str) -> str:
    return await call_claude(
        "Translate to English. Output ONLY the English translation.",
        text, max_tokens=256,
    )


async def get_answer(question_en: str) -> str:
    return await call_claude(
        (
            "You are the official voice assistant for Narayana Educational Institutions. "
            "Narayana has 950+ institutes across 23 states and 250+ cities in India, "
            "with 600K+ learners annually and 50K+ skilled employees, established in 1979. "
            "Answer questions ONLY from the school information provided. "
            "Be warm, helpful, and concise (2-4 sentences). "
            "If not available, say so politely and suggest visiting narayanagroup.com."
        ),
        f"SCHOOL INFORMATION:\n{SCHOOL_CONTEXT}\n\nQUESTION: {question_en}",
        max_tokens=400,
    )


async def translate_to_telugu(text: str) -> str:
    return await call_claude(
        "Translate to Telugu. Output ONLY the Telugu translation.",
        text, max_tokens=400,
    )


def make_tts(text: str, lang: str) -> Optional[str]:
    try:
        buf = io.BytesIO()
        gTTS(text=text[:2000], lang=lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        log.warning(f"TTS failed lang={lang}: {e}")
        return None


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    tts: bool = True


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "school": "Narayana Educational Institutions",
        "context_chars": len(SCHOOL_CONTEXT),
        "context_loaded": len(SCHOOL_CONTEXT) > 100,
    }


@app.post("/ask")
async def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")
    if not SCHOOL_CONTEXT:
        raise HTTPException(status_code=503, detail="School data not loaded yet")

    lang = await detect_language(question)
    log.info(f"lang={lang} q={question[:80]}")

    question_en = await translate_to_english(question) if lang == "te" else question
    en_answer = await get_answer(question_en)
    te_answer = await translate_to_telugu(en_answer)

    audio = {}
    if req.tts:
        loop = asyncio.get_event_loop()
        en_tts, te_tts = await asyncio.gather(
            loop.run_in_executor(executor, make_tts, en_answer, "en"),
            loop.run_in_executor(executor, make_tts, te_answer, "te"),
        )
        if en_tts: audio["en"] = en_tts
        if te_tts: audio["te"] = te_tts

    return {
        "lang_detected": lang,
        "question_en": question_en,
        "answers": {"en": en_answer, "te": te_answer},
        "audio": audio,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
