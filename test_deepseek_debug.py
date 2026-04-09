#!/usr/bin/env python3
# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
"""Debug DeepSeek API - show detailed errors"""
import os
import warnings

API_KEY = os.getenv('deepseek_API_KEY')
if not API_KEY:
    raise ValueError('deepseek_API_KEY not set - load from .env')

print(f'API Key: {API_KEY[:10]}...{API_KEY[-4:]}')
print(f'Key length: {len(API_KEY)}')
print()

# Check if packages are installed
print('=== Checking installed packages ===')
try:
    import openai
    print('openai: INSTALLED')
except ImportError as e:
    print(f'openai: NOT INSTALLED ({e})')

print()
print('=== Testing DeepSeek via OpenAI-compatible API ===')
try:
    import openai
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url="https://api.deepseek.com/v1"
    )
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=100
    )
    print(f'SUCCESS: {response.choices[0].message.content}')
except Exception as e:
    import traceback
    print(f'FAILED: {type(e).__name__}')
    print(f'Error: {e}')
    traceback.print_exc()

print()
print('=== Testing DeepSeek via raw HTTP request ===')
try:
    import requests
    import json
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': 'deepseek-reasoner',
        'messages': [{'role': 'user', 'content': 'Say hello'}],
        'max_tokens': 100
    }
    resp = requests.post('https://api.deepseek.com/v1/chat/completions', headers=headers, json=payload)
    if resp.status_code == 200:
        data = resp.json()
        print(f'SUCCESS: {data["choices"][0]["message"]["content"]}')
    else:
        print(f'HTTP {resp.status_code}: {resp.text}')
except Exception as e:
    import traceback
    print(f'FAILED: {type(e).__name__}')
    print(f'Error: {e}')
    traceback.print_exc()
