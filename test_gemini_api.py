# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
import os
os.chdir('/workspaces/Allentown-L104-Node')

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

api_key = os.environ.get('GEMINI_API_KEY')
print(f'API Key loaded: {api_key[:20]}...')

genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content('Say hello and confirm you are Gemini')

print(f'Gemini Response: {response.text}')
