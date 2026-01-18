#!/usr/bin/env python3
"""Debug Gemini API - show detailed errors"""
import os
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError('GEMINI_API_KEY not set - load from .env')

print(f'API Key: {API_KEY[:10]}...{API_KEY[-4:]}')
print(f'Key length: {len(API_KEY)}')
print()

# Check if packages are installed
print('=== Checking installed packages ===')
try:
    from google import genai
    print('google-genai: INSTALLED')
except ImportError as e:
    print(f'google-genai: NOT INSTALLED ({e})')

try:
    import google.generativeai
    print('google-generativeai: INSTALLED')
except ImportError as e:
    print(f'google-generativeai: NOT INSTALLED ({e})')

print()
print('=== Testing NEW google-genai package ===')
try:
    from google import genai
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents='Say hello'
    )
    print(f'SUCCESS: {response.text}')
except Exception as e:
    import traceback
    print(f'FAILED: {type(e).__name__}')
    print(f'Error: {e}')
    traceback.print_exc()

print()
print('=== Testing OLD google-generativeai package ===')
try:
    import google.generativeai as genai_old
    genai_old.configure(api_key=API_KEY)
    model = genai_old.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content('Say hello')
    print(f'SUCCESS: {response.text}')
except Exception as e:
    import traceback
    print(f'FAILED: {type(e).__name__}')
    print(f'Error: {e}')
    traceback.print_exc()
