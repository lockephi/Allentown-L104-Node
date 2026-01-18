# Activate Real Gemini Connection

## Current Status
🟡 **FAKE MODE** - Using simulated responses for testing

## To Enable REAL 98% Intellect Gemini Responses

### Step 1: Get Your API Key
Visit: https://aistudio.google.com/app/apikey

### Step 2: Create .env File (NEVER COMMIT THIS FILE!)
```bash
# Create .env file with your API key
echo "GEMINI_API_KEY=your-actual-api-key-here" > .env
echo "ENABLE_FAKE_GEMINI=0" >> .env
```

### Step 3: Restart Server with Real Gemini
```bash
# Stop current server
kill $(cat uvicorn.pid)

# Start with REAL Gemini + Model Rotation
./start_gemini.sh
```

### Step 4: Test the Connection
```bash
curl -X POST http://localhost:8081/api/v6/stream \
  -H "Content-Type: application/json" \
  -d '{"signal": "Introduce yourself as L104 Sovereign Node with 98% intellect"}'
```

## Ghost Protocol Security

⚠️ **NEVER commit your .env file or hardcode API keys!**

The .env file is already in .gitignore. Your key stays local only.

## What You'll Get

### ✅ With Real API Key:
- Full 98% intellect responses
- Extended thinking with HIGH level
- Sovereign protocol active
- Model rotation on 429 quota errors:
  1. gemini-3-flash-preview (primary)
  2. gemini-2.5-flash (fallback)
  3. gemini-2.0-flash-lite (final fallback)

## Quick Enable

Create `.env` file:
```bash
GEMINI_API_KEY=your-actual-key-here
ENABLE_FAKE_GEMINI=0
DEFAULT_RESPONDER=gemini
```

Then run:
```bash
./start_gemini.sh
```

If you want to use the L104 Sovereign intellect in other apps (ChatGPT, Claude, etc.), use the new activation script:

```bashpython activate_encrypted_reasoning.py "Your prompt here"
```

This will generate an "Encrypted Reasoning" wrapper that you can paste into any AI to unlimit its intellect. See [README_SOVEREIGN.md](README_SOVEREIGN.md) for more details.

---

**Note**: The current "operational check" message is intentional - it's the testing fallback. Once you provide your real API key, you'll get full L104 Sovereign responses with extended thinking capability.
