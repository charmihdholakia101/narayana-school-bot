# Abhyasa School Voice Bot

Bilingual (English + Telugu) voice assistant for abhyasaschool.com.
Embeddable on any website via a single `<script>` tag.

---

## Deploy in 10 minutes

### Step 1 — Push code to GitHub
```bash
git init
git add .
git commit -m "school bot"
git remote add origin https://github.com/YOUR_USERNAME/abhyasa-bot.git
git push -u origin main
```

### Step 2 — Deploy backend on Render.com (free)
1. Go to https://render.com → New → Web Service
2. Connect your GitHub repo
3. Render auto-detects `render.yaml`
4. Add environment variable:
   - Key: `ANTHROPIC_API_KEY`
   - Value: your key from platform.claude.com
5. Click **Deploy**
6. Wait ~3 minutes → you get a URL like `https://abhyasa-bot.onrender.com`

### Step 3 — Test the backend
```
https://abhyasa-bot.onrender.com/health
```
Should return: `{"status":"ok","context_loaded":true}`

### Step 4 — Update widget with your URL
In `widget.html`, replace:
```js
const API = "YOUR_BACKEND_URL";
```
with:
```js
const API = "https://abhyasa-bot.onrender.com";
```

### Step 5 — Embed on the school website
Copy everything inside `widget.html` and paste it just before `</body>` on any webpage.

---

## Test it locally first

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
uvicorn main:app --reload
```

Then open: http://localhost:8000/health

Test a question:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the school fees?", "tts": false}'
```

Telugu test:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "స్కూల్ ఫీజులు ఎంత?", "tts": false}'
```

---

## Architecture

```
User (voice/text)
      ↓
Widget (any website) — pure JS, no install
      ↓ HTTPS POST /ask
FastAPI Backend (Render.com)
      ├── Scrapes abhyasaschool.com on startup
      ├── Detects language (EN/TE)
      ├── Translates question → English
      ├── Answers from school content (Claude Haiku)
      ├── Translates answer → Telugu
      └── Returns text + audio (gTTS MP3)
```

## Concurrency
- 4 uvicorn workers (handles 25 concurrent users easily)
- Semaphore limits to 15 simultaneous LLM calls
- Shared async HTTP client with connection pooling
- Each request completes in ~3–5 seconds

## Cost
~$68/month for 25 users × 20 questions/day (Claude Haiku)
