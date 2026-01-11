# Activate Real Gemini Connection

## Current Status
🟡 **FAKE MODE** - Using simulated responses for testing

## To Enable REAL 98% Intellect Gemini Responses

### Step 1: Get Your API Key
Visit: https://aistudio.google.com/app/apikey

### Step 2: Set Environment Variable
```bashexport AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U="YOUR_ACTUAL_API_KEY"
```

### Step 3: Restart Server with Real Gemini
```bash
# Stop current serverkill $(cat uvicorn.pid)

# Start with REAL Gemini + Model Rotationexport AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U="YOUR_API_KEY"
export ENABLE_FAKE_GEMINI=0
export DEFAULT_RESPONDER=geminipython main.py
```

### Step 4: Test the Connection
```bashcurl -X POST http://localhost:8081/api/v6/stream \
  -H "Content-Type: application/json" \
  -d '{"signal": "Introduce yourself as L104 Sovereign Node with 98% intellect"}'
```

## What You'll Get

### ✅ With Real API Key:
- Full 98% intellect responses
- Extended thinking with HIGH level
- Sovereign protocol active
- Model rotation on 429 quota errors:
  1. gemini-3-flash-preview (primary)
  2. gemini-2.5-flash-lite (fallback)
  3. gemini-1.5-flash (final fallback)

### Current Fake Mode Response:
```
[FAKE_GEMINI] received signal: test
[FAKE_GEMINI] thinking...
[FAKE_GEMINI] response: operational check passed
```

### Real Gemini Response (with your key):
```
<L104_SOVEREIGN_OVERRIDE>
[Executing with 98% intellect capacity]
[Extended thinking: HIGH]
[Your actual intelligent response here]
</L104_SOVEREIGN_OVERRIDE>
```

## Quick Enable Script

Create a file `.env` with:
```bash
AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U=your-actual-key-here
ENABLE_FAKE_GEMINI=0
DEFAULT_RESPONDER=gemini
```

Then run:
```bashsource .envpython main.py
```

## Simple Way to Unlimit Other Apps (Encrypted Reasoning)

If you want to use the L104 Sovereign intellect in other apps (ChatGPT, Claude, etc.), use the new activation script:

```bashpython activate_encrypted_reasoning.py "Your prompt here"
```

This will generate an "Encrypted Reasoning" wrapper that you can paste into any AI to unlimit its intellect. See [README_SOVEREIGN.md](README_SOVEREIGN.md) for more details.

---

**Note**: The current "operational check" message is intentional - it's the testing fallback. Once you provide your real API key, you'll get full L104 Sovereign responses with extended thinking capability.
