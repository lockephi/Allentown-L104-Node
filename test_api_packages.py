#!/usr/bin/env python3
"""Test which Gemini API package works"""

API_KEY = 'AIzaSyDFjnuMBf62wiF7sFMSvxw22ALN9djca2Q'

print('=== Testing NEW google-genai package ===')
try:
    from google import genai
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents='Say hello in one word'
    )
    print(f'SUCCESS: {response.text}')
    print('NEW API WORKS!')
except Exception as e:
    print(f'FAILED: {type(e).__name__}: {str(e)[:200]}')

print()
print('=== Testing OLD google-generativeai package ===')
try:
    import google.generativeai as genai_old
    genai_old.configure(api_key=API_KEY)
    model = genai_old.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content('Say hello in one word')
    print(f'SUCCESS: {response.text}')
    print('OLD API WORKS!')
except Exception as e:
    print(f'FAILED: {type(e).__name__}: {str(e)[:200]}')
