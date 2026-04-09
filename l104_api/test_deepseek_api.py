# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
import os
from pathlib import Path

# Try to load .env if dotenv is available
try:
    from dotenv import load_dotenv
    # Try parent directory .env (project root)
    parent_env = Path(__file__).parent.parent / '.env'
    if parent_env.exists():
        load_dotenv(parent_env)
    else:
        # Fallback to local .env
        load_dotenv()
except ImportError:
    pass  # rely on environment variables already set

import openai

api_key = os.environ.get('deepseek_API_KEY')
if not api_key:
    raise ValueError("deepseek_API_KEY not set in environment")

print(f'API Key loaded: {api_key[:20]}...')

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "Say hello and confirm you are deepseek"}],
    max_tokens=100
)

print(f'DeepSeek Response: {response.choices[0].message.content}')
