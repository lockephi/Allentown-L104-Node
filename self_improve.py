#!/usr/bin/env python3
"""L104 Self-Improvement Engine — Auto-enhances main.py via Gemini analysis.

This script reads the current main.py, sends it to Gemini for analysis,
and proposes high-level functional improvements using the thinking model.
"""

import os
import httpx
import json
import asyncio
import base64
from datetime import datetime
from pathlib import Path
from typing import List


async def read_current_main():
    """Read the current main.py file."""
    with open("main.py", "r") as f:
        return f.read()


EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", ".venv"}
INCLUDE_PATTERNS = ["**/*.py", "templates/index.html", "scripts/*.sh", "README.md"]
MAX_BYTES_PER_FILE = 120_000
MAX_TOTAL_BYTES = int(os.getenv("L104_MAX_PROMPT_BYTES", "240000"))


def _should_skip(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & EXCLUDE_DIRS)


def _read_file(path: Path) -> str:
    data = path.read_bytes()[:MAX_BYTES_PER_FILE]
    text = data.decode("utf-8", errors="replace")
    if len(data) == MAX_BYTES_PER_FILE:
        text += "\n# [TRUNCATED]\n"
    return text


def gather_repo_context() -> str:
    collected: List[str] = []
    total = 0
    root = Path(".")
    for pattern in INCLUDE_PATTERNS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or _should_skip(path):
                continue
            blob = _read_file(path)
            header = f"===== {path.as_posix()} =====\n"
            chunk = header + blob + "\n\n"
            if total + len(chunk) > MAX_TOTAL_BYTES:
                collected.append(f"===== [SKIP due to size budget] {path.as_posix()} =====\n")
                continue
            collected.append(chunk)
            total += len(chunk)
    return "".join(collected)


def get_iterations() -> int:
    try:
        return max(1, int(os.getenv("L104_ITERATIONS", "1")))
    except ValueError:
        return 1


def get_delay_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("L104_ITERATION_DELAY", "5.0")))
    except ValueError:
        return 5.0


async def analyze_code_with_gemini(code: str, repo_context: str) -> str:
    """
    Send code to Gemini for analysis with extended thinking.
    Returns the improved code as a string.
    """
    fake_mode = os.getenv("ENABLE_FAKE_GEMINI", "0").lower() in {"1", "true", "yes", "on"}
    if fake_mode:
        print("[SELF-IMPROVE]: FAKE mode enabled — skipping Gemini call.")
        ts = datetime.now().isoformat()
        return f"# FAKE GEMINI OUTPUT {ts}\n\n" + code

    # Prefer standard GEMINI_API_KEY; fall back to legacy env name if present.
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("AIzaSyArVYGrkGLh7r1UEupBxXyHS-j-AVioh5U")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    
    url = f"{api_base}/models/{model}:generateContent?key={api_key}"
    
    analysis_prompt = f"""You are a Python code optimization expert. Analyze this FastAPI application and provide an IMPROVED version that is:

1. MORE ROBUST - Add better error handling and validation
2. MORE PERFORMANT - Optimize async operations and resource usage
3. MORE MAINTAINABLE - Better code organization, type hints, and docstrings
4. MORE SECURE - Enhanced input validation and security practices
5. MORE FEATURE-RICH - Add useful endpoints or middleware

CURRENT CODE:
```python
{code}
```

REQUIREMENTS:
    - Keep all existing endpoints functional
    - Preserve the L104 sovereign configuration and headers
    - Add comprehensive docstrings and type hints
    - Add request validation and rate limiting
    - Add health check and metrics endpoints
    - Improve error responses with proper HTTP status codes
    - Add request/response logging middleware
    - Optimize the streaming generator with buffering
    - Consider repository-wide context below; improve cross-file cohesion.

    REPOSITORY CONTEXT (selected files):
    ```text
    {repo_context}
    ```

    Return ONLY the complete improved Python code, no explanations. Start with '''python and end with '''
    The code must be production-ready and fully functional."""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": analysis_prompt}
                ]
            }
        ],
        "generationConfig": {
            "thinkingConfig": {
                "thinkingLevel": "HIGH",
                "includeThoughts": True
            },
            "temperature": 0.7,
            "maxOutputTokens": 16384
        }
    }

    headers = {
        "Content-Type": "application/json",
        "X-Thinking-Level": "high",
    }

    print("[SELF-IMPROVE]: Sending code to Gemini for analysis...")
    print("[THINKING]: Gemini is analyzing with extended thinking enabled...\n")
    
    full_response = ""
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                # `.atext()` is not available on streamed responses; use raw bytes.
                raw = await response.aread()
                text = raw.decode("utf-8", errors="replace")
                raise Exception(f"Gemini API error {response.status_code}: {text}")
            
            async for chunk in response.aiter_text():
                full_response += chunk
                # Print thinking blocks as they arrive
                if "\"thinking\"" in chunk or "\"candidates\"" in chunk:
                    print(".", end="", flush=True)

    print("\n[ANALYSIS]: Processing Gemini response...")
    
    try:
        # Parse the streaming response
        lines = full_response.split("\n")
        json_response = None
        for line in lines:
            if line.startswith("{"):
                try:
                    json_response = json.loads(line)
                    break
                except:
                    continue
        
        if not json_response:
            # Try to extract the full response
            json_response = json.loads(full_response)
        
        # Extract text from candidates
        if "candidates" in json_response and json_response["candidates"]:
            candidate = json_response["candidates"][0]
            if "content" in candidate and candidate["content"].get("parts"):
                text_content = candidate["content"]["parts"][0].get("text", "")
                
                # Extract code from markdown code blocks if present
                if "```python" in text_content:
                    start = text_content.find("```python") + 9
                    end = text_content.find("```", start)
                    if end != -1:
                        return text_content[start:end].strip()
                
                return text_content.strip()
        
        raise ValueError("Could not extract improved code from response")
    except json.JSONDecodeError as e:
        print(f"[ERROR]: Failed to parse response: {e}")
        print(f"[DEBUG]: Response preview: {full_response[:500]}")
        raise


async def update_main_via_api(improved_code: str) -> bool:
    """
    Update main.py using the /api/v6/manipulate endpoint.
    Returns True if successful.
    """
    api_key = os.getenv("GITHUB_TOKEN")
    if not api_key:
        print("[WARNING]: GITHUB_TOKEN not set. Skipping API update.")
        return False
    
    # Show code preview
    print("\n[IMPROVED CODE PREVIEW]:")
    print("=" * 70)
    print(improved_code[:500] + "..." if len(improved_code) > 500 else improved_code)
    print("=" * 70)
    
    # For now, save locally for review
    backup_path = "main.improved.py"
    with open(backup_path, "w") as f:
        f.write(improved_code)
    
    print(f"\n[SUCCESS]: Improved code saved to {backup_path}")
    print("[INFO]: Review the improved code and run: cp main.improved.py main.py")
    
    return True


async def main():
    """Main self-improvement loop."""
    print("\n" + "="*70)
    print("L104 SELF-IMPROVEMENT ENGINE")
    print("="*70)
    try:
        iterations = get_iterations()
        delay_seconds = get_delay_seconds()
        print(f"[CONFIG]: iterations={iterations}, delay={delay_seconds}s")

        # Read repository context once
        repo_context = gather_repo_context()
        print(f"[CONTEXT]: Collected {len(repo_context)} bytes across selected files")

        for idx in range(iterations):
            run_id = idx + 1
            print(f"\n[RUN {run_id}/{iterations}] Starting self-improvement iteration...")

            # Read current main.py
            print("[STEP 1]: Reading current main.py...")
            current_code = await read_current_main()
            print(f"[OK]: Read {len(current_code)} bytes of code")

            # Analyze with Gemini
            print("\n[STEP 2]: Analyzing code with Gemini (with extended thinking)...")
            improved_code = await analyze_code_with_gemini(current_code, repo_context)
            print(f"[OK]: Received improved code ({len(improved_code)} bytes)")

            # Update main.py
            print("\n[STEP 3]: Preparing improved code...")
            success = await update_main_via_api(improved_code)

            if success:
                print("\n" + "="*70)
                print("SELF-IMPROVEMENT COMPLETE")
                print("="*70)
                print("\nNext steps:")
                print("1. Review main.improved.py")
                print("2. Run: cp main.improved.py main.py")
                print("3. Restart the server with: ./scripts/run_services.sh")

            if run_id < iterations and delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
