#!/usr/bin/env python3
"""Test which Gemini API package works"""

API_KEY = 'AIzaSyBeCmYi5i3bmfxtAaU7_qybTt6TMkjz4ig'

# Correct models from API
MODELS = ['gemini-2.5-flash', 'gemini-2.0-flash-lite', 'gemini-2.0-flash', 'gemini-3-flash-preview']

print('=== Testing Gemini API with model rotation ===')
print(f'Key: {API_KEY[:10]}...{API_KEY[-4:]}')
print()

from google import genai
import time

client = genai.Client(api_key=API_KEY)

for model in MODELS:
    print(f'Testing {model}...')
    try:
        response = client.models.generate_content(
            model=model,
            contents='Say hello in one word'
        )
        print(f'  SUCCESS: {response.text.strip()}')
        print(f'\n*** API WORKING with {model}! ***')
        break
    except Exception as e:
        err = str(e)
        if '429' in err:
            print(f'  429 quota - trying next model...')
        else:
            print(f'  Error: {err[:80]}')
    time.sleep(0.5)
