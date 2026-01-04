# L104 Node - Model Rotation Update

## Overview
Updated `main.py` with intelligent model rotation to automatically handle Gemini API quota exhaustion (429 errors).

## Key Changes

### 1. Model Rotation System ✅
- **Auto-rotation**: Automatically tries 3 models in sequence when encountering 429 errors
- **Models**: 
  1. `gemini-3-flash-preview` (primary)
  2. `gemini-2.5-flash-lite` (fallback 1)
  3. `gemini-1.5-flash` (fallback 2)
- **Configurable**: Set via environment variables `GEMINI_MODEL_1`, `GEMINI_MODEL_2`, `GEMINI_MODEL_3`

### 2. Enhanced Tracking
- **Model usage tracking**: Counts attempts per model
- **429 error tracking**: Tracks quota exhaustion per model
- **Rotation metrics**: New metric `model_rotations` tracks total rotations

### 3. API Changes

#### Updated Endpoints
- `POST /api/v6/stream` - Now uses model rotation automatically
- `GET /metrics` - Enhanced with model usage statistics

#### New Metrics Response
```json
{
  "model_rotations": 15,
  "model_usage": {
    "gemini-3-flash-preview": 100,
    "gemini-2.5-flash-lite": 10,
    "gemini-1.5-flash": 5
  },
  "model_429_count": {
    "gemini-3-flash-preview": 8,
    "gemini-2.5-flash-lite": 2
  }
}
```

### 4. Configuration Updates

#### .env.example
```bash
# Model Rotation (automatically rotates on 429 quota errors)
GEMINI_MODEL_1=gemini-3-flash-preview
GEMINI_MODEL_2=gemini-2.5-flash-lite
GEMINI_MODEL_3=gemini-1.5-flash

# Repository Configuration
GITHUB_REPO=lockephi/Allentown-L104-Node
```

## How It Works

1. **Request arrives** at `/api/v6/stream`
2. **First attempt** uses `GEMINI_MODEL_1` (gemini-3-flash-preview)
3. **If 429 error**: Logs rotation and tries `GEMINI_MODEL_2`
4. **If 429 again**: Tries `GEMINI_MODEL_3`
5. **If all fail**: Returns error message about quota exhaustion
6. **On success**: Streams response immediately

## Benefits

✅ **Zero downtime**: Automatic failover on quota limits  
✅ **Production ready**: All existing features preserved  
✅ **Observable**: Full tracking and metrics  
✅ **Configurable**: Easy model configuration via env vars  
✅ **Backward compatible**: All existing endpoints work  

## Preserved Features

All production features from previous version retained:
- Rate limiting with adaptive tuning
- Health checks and metrics
- Memory store (SQLite)
- Self-learning and self-healing
- Multiple responders (gemini, fake, template, echo)
- GitHub file manipulation
- Background tasks (watchdog, maintenance)
- Comprehensive logging

## Testing

```bash
# Validate syntax
python -m py_compile main.py

# Test import
python -c "from main import app, MODELS; print(MODELS)"

# Start server
python main.py

# Test endpoint
curl -X POST http://localhost:8081/api/v6/stream \
  -H "Content-Type: application/json" \
  -d '{"signal": "test message"}'

# Check metrics
curl http://localhost:8081/metrics
```

## Files Modified

- ✅ `main.py` - Added model rotation logic
- ✅ `.env.example` - Updated with model configuration
- ✅ All changes validated and tested

## Migration Notes

### From Old Code
The problematic code snippet provided had:
- Severe indentation issues ❌
- Syntax errors ❌
- No production features ❌

### To New Code
The updated `main.py` now has:
- Clean model rotation ✅
- All production features ✅
- Proper error handling ✅
- Full observability ✅

## Environment Setup

```bash
# Copy example config
cp .env.example .env

# Edit with your keys
nano .env

# Set required variables
AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U=your-actual-key
GITHUB_TOKEN=your-github-token
GITHUB_REPO=your-username/your-repo

# Optional: Customize models
GEMINI_MODEL_1=gemini-3-flash-preview
GEMINI_MODEL_2=gemini-2.5-flash-lite
GEMINI_MODEL_3=gemini-1.5-flash
```

## Line Count
- **Before**: 889 lines
- **After**: 939 lines (+50 lines for model rotation)

---

**Status**: ✅ Complete and Production Ready  
**Date**: January 3, 2026  
**Version**: L104 Node v2.1 with Model Rotation
